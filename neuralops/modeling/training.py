"""Reproducible GRU training, artifact loading, and evaluation."""

from __future__ import annotations

import json
import platform
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neuralops import __version__
from neuralops.data.io import read_json, read_records, sha256_file, write_json
from neuralops.data.records import SequenceRecord
from neuralops.device import resolve_device
from neuralops.metrics import binary_metrics
from neuralops.modeling.dataset import Batch, SequenceDataset, collate_sequences
from neuralops.modeling.gru import GRUClassifier, GRUConfig
from neuralops.policy import select_policy


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    max_sequence_length: int = 128
    device: str = "auto"
    minimum_coverage: float = 0.80


@dataclass(slots=True)
class LoadedGRU:
    model: GRUClassifier
    vocabulary: dict[str, int]
    policy: dict[str, Any]
    metadata: dict[str, Any]
    max_sequence_length: int
    device: torch.device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_loader(
    records: list[SequenceRecord],
    vocabulary: dict[str, int],
    settings: TrainingSettings,
    *,
    shuffle: bool,
) -> DataLoader[Batch]:
    generator = torch.Generator().manual_seed(settings.seed)
    dataset = SequenceDataset(records, vocabulary, settings.max_sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_sequences,
        generator=generator,
    )
    return cast(DataLoader[Batch], loader)


def _run_epoch(
    model: GRUClassifier,
    batches: Iterable[Batch],
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    *,
    optimizer: AdamW | None,
    gradient_clip_norm: float,
) -> tuple[float, list[int], list[float]]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    sample_total = 0
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in batches:
            moved = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(moved.tokens, moved.lengths)["binary"]
            loss = criterion(logits, moved.anomaly)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            batch_size = moved.anomaly.numel()
            loss_total += float(loss.detach().cpu()) * batch_size
            sample_total += batch_size
            all_labels.extend(moved.anomaly.detach().cpu().to(torch.int64).tolist())
            all_probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return loss_total / max(sample_total, 1), all_labels, all_probabilities


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _validated_vocabulary(path: Path) -> dict[str, int]:
    raw = read_json(path)
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, int) for key, value in raw.items()
    ):
        raise ValueError(f"Invalid vocabulary: {path}")
    return dict(raw)


def train_gru(
    processed_dir: Path,
    artifact_dir: Path,
    *,
    model_config: GRUConfig,
    settings: TrainingSettings,
) -> dict[str, Any]:
    seed_everything(settings.seed)
    device = resolve_device(settings.device)
    vocabulary = _validated_vocabulary(processed_dir / "vocabulary.json")
    if model_config.vocabulary_size != len(vocabulary):
        raise ValueError("Model vocabulary_size does not match prepared vocabulary")
    train_records = read_records(processed_dir / "splits" / "train.jsonl")
    validation_records = read_records(processed_dir / "splits" / "validation.jsonl")
    test_records = read_records(processed_dir / "splits" / "test.jsonl")
    positives = sum(record.anomaly for record in train_records)
    negatives = len(train_records) - positives
    if not positives or not negatives:
        raise ValueError("GRU training requires both classes in train")

    model = GRUClassifier(model_config).to(device)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    train_loader = _make_loader(train_records, vocabulary, settings, shuffle=True)
    validation_loader = _make_loader(validation_records, vocabulary, settings, shuffle=False)

    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.epochs + 1):
        train_loss, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=settings.gradient_clip_norm,
        )
        validation_loss, labels, probabilities = _run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            gradient_clip_norm=settings.gradient_clip_norm,
        )
        validation_metrics = binary_metrics(labels, probabilities, 0.5)
        score_value = validation_metrics["pr_auc"]
        if score_value is None:
            raise ValueError("Validation PR-AUC requires both classes")
        score = float(score_value)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_pr_auc": score,
            "validation_f1_at_0_5": validation_metrics["f1"],
        }
        history.append(epoch_result)
        print(json.dumps({"training": epoch_result}, sort_keys=True), flush=True)
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = _cpu_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    validation_loader = _make_loader(validation_records, vocabulary, settings, shuffle=False)
    _, validation_labels, validation_probabilities = _run_epoch(
        model,
        validation_loader,
        criterion,
        device,
        optimizer=None,
        gradient_clip_norm=settings.gradient_clip_norm,
    )
    policy = select_policy(
        validation_labels,
        validation_probabilities,
        minimum_coverage=settings.minimum_coverage,
    )
    threshold = float(policy["threshold"])
    test_loader = _make_loader(test_records, vocabulary, settings, shuffle=False)
    _, test_labels, test_probabilities = _run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        gradient_clip_norm=settings.gradient_clip_norm,
    )
    metrics = {
        "validation": binary_metrics(validation_labels, validation_probabilities, threshold),
        "test": binary_metrics(test_labels, test_probabilities, threshold),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    torch.save(best_state, model_path)
    write_json(artifact_dir / "model_config.json", model_config.to_dict())
    write_json(artifact_dir / "vocabulary.json", vocabulary)
    write_json(artifact_dir / "policy.json", policy)
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "training_history.json",
        {"best_epoch": best_epoch, "monitor": "validation_pr_auc", "epochs": history},
    )
    data_manifest = read_json(processed_dir / "manifest.json")
    metadata = {
        "schema_version": 1,
        "artifact_type": "pytorch-packed-bidirectional-gru",
        "neuralops_version": __version__,
        "source": str(data_manifest.get("source", train_records[0].source)),
        "profile": str(data_manifest.get("profile", train_records[0].source)),
        "label_provenance": "public"
        if train_records[0].source == "loghub-hdfs-v1"
        else "fixture-or-synthetic",
        "seed": settings.seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "artifact_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "model_config_sha256": sha256_file(artifact_dir / "model_config.json"),
        "vocabulary_sha256": sha256_file(artifact_dir / "vocabulary.json"),
        "policy_sha256": sha256_file(artifact_dir / "policy.json"),
        "data_manifest_sha256": sha256_file(processed_dir / "manifest.json"),
        "best_epoch": best_epoch,
        "max_sequence_length": settings.max_sequence_length,
        "truncated_records": {
            split_name: sum(len(record.events) > settings.max_sequence_length for record in records)
            for split_name, records in (
                ("train", train_records),
                ("validation", validation_records),
                ("test", test_records),
            )
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "training_settings": asdict(settings),
    }
    write_json(artifact_dir / "metadata.json", metadata)
    return {"policy": policy, "metrics": metrics, "metadata": metadata}


def load_gru_artifact(artifact_dir: Path, requested_device: str = "auto") -> LoadedGRU:
    device = resolve_device(requested_device)
    model_raw = read_json(artifact_dir / "model_config.json")
    policy_raw = read_json(artifact_dir / "policy.json")
    metadata_raw = read_json(artifact_dir / "metadata.json")
    if not all(isinstance(item, dict) for item in (model_raw, policy_raw, metadata_raw)):
        raise ValueError("Artifact JSON metadata must contain objects")
    integrity_files = {
        "model_sha256": "model.pt",
        "model_config_sha256": "model_config.json",
        "vocabulary_sha256": "vocabulary.json",
        "policy_sha256": "policy.json",
    }
    for metadata_key, filename in integrity_files.items():
        expected = metadata_raw.get(metadata_key)
        if expected is not None and sha256_file(artifact_dir / filename) != expected:
            raise ValueError(f"Artifact integrity check failed for {filename}")
    vocabulary = _validated_vocabulary(artifact_dir / "vocabulary.json")
    model = GRUClassifier(GRUConfig.from_dict(model_raw))
    state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return LoadedGRU(
        model=model,
        vocabulary=vocabulary,
        policy=dict(policy_raw),
        metadata=dict(metadata_raw),
        max_sequence_length=int(metadata_raw["max_sequence_length"]),
        device=device,
    )


def predict_probabilities(loaded: LoadedGRU, records: list[SequenceRecord]) -> list[float]:
    settings = TrainingSettings(
        batch_size=256,
        max_sequence_length=loaded.max_sequence_length,
        device=str(loaded.device),
    )
    loader = _make_loader(records, loaded.vocabulary, settings, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    _, _, probabilities = _run_epoch(
        loaded.model,
        loader,
        criterion,
        loaded.device,
        optimizer=None,
        gradient_clip_norm=1.0,
    )
    return probabilities


def evaluate_artifact(
    artifact_dir: Path,
    processed_dir: Path,
    *,
    split: str,
    requested_device: str = "auto",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    loaded = load_gru_artifact(artifact_dir, requested_device)
    records = read_records(processed_dir / "splits" / f"{split}.jsonl")
    probabilities = predict_probabilities(loaded, records)
    return binary_metrics(
        [record.anomaly for record in records],
        probabilities,
        float(loaded.policy["threshold"]),
    )
