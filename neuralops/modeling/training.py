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
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neuralops import __version__
from neuralops.data.io import read_json, read_records, sha256_file, write_json
from neuralops.data.records import SequenceRecord
from neuralops.device import resolve_device
from neuralops.metrics import binary_metrics, multiclass_metrics
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
    category_loss_weight: float = 0.5
    severity_loss_weight: float = 0.3


@dataclass(slots=True)
class LoadedGRU:
    model: GRUClassifier
    vocabulary: dict[str, int]
    policy: dict[str, Any]
    metadata: dict[str, Any]
    label_mappings: dict[str, list[str]]
    max_sequence_length: int
    device: torch.device


@dataclass(slots=True)
class EpochOutput:
    loss: float
    binary_labels: list[int]
    binary_probabilities: list[float]
    category_labels: list[int]
    category_predictions: list[int]
    severity_labels: list[int]
    severity_predictions: list[int]


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
    category_to_index: dict[str, int] | None = None,
    severity_to_index: dict[str, int] | None = None,
) -> DataLoader[Batch]:
    generator = torch.Generator().manual_seed(settings.seed)
    dataset = SequenceDataset(
        records,
        vocabulary,
        settings.max_sequence_length,
        category_to_index,
        severity_to_index,
    )
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
    category_loss_weight: float = 0.0,
    severity_loss_weight: float = 0.0,
) -> EpochOutput:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    sample_total = 0
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    category_labels: list[int] = []
    category_predictions: list[int] = []
    severity_labels: list[int] = []
    severity_predictions: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in batches:
            moved = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(moved.tokens, moved.lengths)
            logits = outputs["binary"]
            loss = criterion(logits, moved.anomaly)
            category_mask = moved.category >= 0
            if "category" in outputs and bool(category_mask.any()):
                category_logits = outputs["category"][category_mask]
                category_targets = moved.category[category_mask]
                loss = loss + category_loss_weight * F.cross_entropy(
                    category_logits, category_targets
                )
                category_labels.extend(category_targets.detach().cpu().tolist())
                category_predictions.extend(category_logits.argmax(dim=1).detach().cpu().tolist())
            severity_mask = moved.severity >= 0
            if "severity" in outputs and bool(severity_mask.any()):
                severity_logits = outputs["severity"][severity_mask]
                severity_targets = moved.severity[severity_mask]
                loss = loss + severity_loss_weight * F.cross_entropy(
                    severity_logits, severity_targets
                )
                severity_labels.extend(severity_targets.detach().cpu().tolist())
                severity_predictions.extend(severity_logits.argmax(dim=1).detach().cpu().tolist())
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            batch_size = moved.anomaly.numel()
            loss_total += float(loss.detach().cpu()) * batch_size
            sample_total += batch_size
            all_labels.extend(moved.anomaly.detach().cpu().to(torch.int64).tolist())
            all_probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return EpochOutput(
        loss=loss_total / max(sample_total, 1),
        binary_labels=all_labels,
        binary_probabilities=all_probabilities,
        category_labels=category_labels,
        category_predictions=category_predictions,
        severity_labels=severity_labels,
        severity_predictions=severity_predictions,
    )


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _validated_vocabulary(path: Path) -> dict[str, int]:
    raw = read_json(path)
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, int) for key, value in raw.items()
    ):
        raise ValueError(f"Invalid vocabulary: {path}")
    return dict(raw)


def _evaluation_metrics(
    output: EpochOutput,
    threshold: float,
    label_mappings: dict[str, list[str]],
) -> dict[str, Any]:
    binary = binary_metrics(output.binary_labels, output.binary_probabilities, threshold)
    if not label_mappings:
        return binary
    metrics: dict[str, Any] = {"binary": binary}
    if label_mappings.get("category") and output.category_labels:
        metrics["category"] = multiclass_metrics(
            output.category_labels,
            output.category_predictions,
            label_mappings["category"],
        )
    if label_mappings.get("severity") and output.severity_labels:
        metrics["severity"] = multiclass_metrics(
            output.severity_labels,
            output.severity_predictions,
            label_mappings["severity"],
        )
    return metrics


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
    category_names = sorted(
        {record.category for record in train_records if record.category is not None}
    )
    severity_names = sorted(
        {record.severity for record in train_records if record.severity is not None}
    )
    if model_config.category_classes != len(category_names):
        raise ValueError("category_classes does not match train-only category labels")
    if model_config.severity_classes != len(severity_names):
        raise ValueError("severity_classes does not match train-only severity labels")
    label_mappings = {
        key: values
        for key, values in (("category", category_names), ("severity", severity_names))
        if values
    }
    category_to_index = {name: index for index, name in enumerate(category_names)}
    severity_to_index = {name: index for index, name in enumerate(severity_names)}
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
    train_loader = _make_loader(
        train_records,
        vocabulary,
        settings,
        shuffle=True,
        category_to_index=category_to_index,
        severity_to_index=severity_to_index,
    )
    validation_loader = _make_loader(
        validation_records,
        vocabulary,
        settings,
        shuffle=False,
        category_to_index=category_to_index,
        severity_to_index=severity_to_index,
    )

    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    monitor_name = "validation_multitask_score" if label_mappings else "validation_pr_auc"
    for epoch in range(1, settings.epochs + 1):
        train_output = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=settings.gradient_clip_norm,
            category_loss_weight=settings.category_loss_weight,
            severity_loss_weight=settings.severity_loss_weight,
        )
        validation_output = _run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            gradient_clip_norm=settings.gradient_clip_norm,
            category_loss_weight=settings.category_loss_weight,
            severity_loss_weight=settings.severity_loss_weight,
        )
        validation_metrics = binary_metrics(
            validation_output.binary_labels,
            validation_output.binary_probabilities,
            0.5,
        )
        score_value = validation_metrics["pr_auc"]
        if score_value is None:
            raise ValueError("Validation PR-AUC requires both classes")
        binary_pr_auc = float(score_value)
        category_macro_f1: float | None = None
        severity_macro_f1: float | None = None
        if label_mappings:
            task_metrics = _evaluation_metrics(validation_output, 0.5, label_mappings)
            category_macro_f1 = float(task_metrics["category"]["macro_f1"])
            severity_macro_f1 = float(task_metrics["severity"]["macro_f1"])
            score = float(np.mean([binary_pr_auc, category_macro_f1, severity_macro_f1]))
        else:
            score = binary_pr_auc
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_output.loss,
            "validation_loss": validation_output.loss,
            "validation_pr_auc": binary_pr_auc,
            "validation_f1_at_0_5": validation_metrics["f1"],
            "validation_category_macro_f1": category_macro_f1,
            "validation_severity_macro_f1": severity_macro_f1,
            "monitor": monitor_name,
            "monitor_score": score,
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

    validation_loader = _make_loader(
        validation_records,
        vocabulary,
        settings,
        shuffle=False,
        category_to_index=category_to_index,
        severity_to_index=severity_to_index,
    )
    validation_output = _run_epoch(
        model,
        validation_loader,
        criterion,
        device,
        optimizer=None,
        gradient_clip_norm=settings.gradient_clip_norm,
        category_loss_weight=settings.category_loss_weight,
        severity_loss_weight=settings.severity_loss_weight,
    )
    policy = select_policy(
        validation_output.binary_labels,
        validation_output.binary_probabilities,
        minimum_coverage=settings.minimum_coverage,
    )
    threshold = float(policy["threshold"])
    test_loader = _make_loader(
        test_records,
        vocabulary,
        settings,
        shuffle=False,
        category_to_index=category_to_index,
        severity_to_index=severity_to_index,
    )
    test_output = _run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        gradient_clip_norm=settings.gradient_clip_norm,
        category_loss_weight=settings.category_loss_weight,
        severity_loss_weight=settings.severity_loss_weight,
    )
    metrics = {
        "validation": _evaluation_metrics(validation_output, threshold, label_mappings),
        "test": _evaluation_metrics(test_output, threshold, label_mappings),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    torch.save(best_state, model_path)
    write_json(artifact_dir / "model_config.json", model_config.to_dict())
    write_json(artifact_dir / "vocabulary.json", vocabulary)
    write_json(artifact_dir / "label_mappings.json", label_mappings)
    write_json(artifact_dir / "policy.json", policy)
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "training_history.json",
        {"best_epoch": best_epoch, "monitor": monitor_name, "epochs": history},
    )
    data_manifest = read_json(processed_dir / "manifest.json")
    artifact_source = str(data_manifest.get("source", train_records[0].source))
    label_provenance = (
        "public"
        if artifact_source == "loghub-hdfs-v1"
        else "synthetic"
        if artifact_source == "opsforge-sim-v1"
        else "fixture"
    )
    metadata = {
        "schema_version": 1,
        "artifact_type": "pytorch-packed-bidirectional-gru",
        "neuralops_version": __version__,
        "source": artifact_source,
        "profile": str(data_manifest.get("profile", train_records[0].source)),
        "label_provenance": label_provenance,
        "seed": settings.seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "artifact_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "model_config_sha256": sha256_file(artifact_dir / "model_config.json"),
        "vocabulary_sha256": sha256_file(artifact_dir / "vocabulary.json"),
        "policy_sha256": sha256_file(artifact_dir / "policy.json"),
        "label_mappings_sha256": sha256_file(artifact_dir / "label_mappings.json"),
        "data_manifest_sha256": sha256_file(processed_dir / "manifest.json"),
        "best_epoch": best_epoch,
        "monitor": monitor_name,
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
        "label_mappings_sha256": "label_mappings.json",
    }
    for metadata_key, filename in integrity_files.items():
        expected = metadata_raw.get(metadata_key)
        if expected is not None and sha256_file(artifact_dir / filename) != expected:
            raise ValueError(f"Artifact integrity check failed for {filename}")
    vocabulary = _validated_vocabulary(artifact_dir / "vocabulary.json")
    mappings_path = artifact_dir / "label_mappings.json"
    mappings_raw = read_json(mappings_path) if mappings_path.exists() else {}
    if not isinstance(mappings_raw, dict) or not all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        for key, value in mappings_raw.items()
    ):
        raise ValueError("Artifact label mappings must contain string lists")
    label_mappings = {str(key): list(value) for key, value in mappings_raw.items()}
    model = GRUClassifier(GRUConfig.from_dict(model_raw))
    state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return LoadedGRU(
        model=model,
        vocabulary=vocabulary,
        policy=dict(policy_raw),
        metadata=dict(metadata_raw),
        label_mappings=label_mappings,
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
    output = _run_epoch(
        loaded.model,
        loader,
        criterion,
        loaded.device,
        optimizer=None,
        gradient_clip_norm=1.0,
    )
    return output.binary_probabilities


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
    settings = TrainingSettings(
        batch_size=256,
        max_sequence_length=loaded.max_sequence_length,
        device=str(loaded.device),
    )
    category_to_index = {
        name: index for index, name in enumerate(loaded.label_mappings.get("category", []))
    }
    severity_to_index = {
        name: index for index, name in enumerate(loaded.label_mappings.get("severity", []))
    }
    loader = _make_loader(
        records,
        loaded.vocabulary,
        settings,
        shuffle=False,
        category_to_index=category_to_index,
        severity_to_index=severity_to_index,
    )
    output = _run_epoch(
        loaded.model,
        loader,
        nn.BCEWithLogitsLoss(),
        loaded.device,
        optimizer=None,
        gradient_clip_norm=1.0,
    )
    return _evaluation_metrics(
        output,
        float(loaded.policy["threshold"]),
        loaded.label_mappings,
    )
