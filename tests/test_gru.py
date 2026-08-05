import json
from pathlib import Path

import pytest
import torch

from neuralops.data.prepare import prepare_dataset
from neuralops.data.records import SequenceRecord
from neuralops.modeling.benchmark import benchmark_artifact
from neuralops.modeling.gru import GRUClassifier, GRUConfig
from neuralops.modeling.training import (
    TrainingSettings,
    evaluate_artifact,
    load_gru_artifact,
    train_gru,
)
from neuralops.predictor import GRUPredictor


def _records() -> list[SequenceRecord]:
    result: list[SequenceRecord] = []
    for index in range(160):
        anomaly = index % 4 == 0
        result.append(
            SequenceRecord(
                session_id=f"gru-{index}",
                group_id=f"gru-group-{index}",
                events=(
                    "E_START",
                    "E_TIMEOUT" if anomaly else "E_OK",
                    f"E_VARIANT_{index}",
                    "E_END",
                ),
                anomaly=int(anomaly),
                source="test-fixture",
            )
        )
    return result


def test_packed_encoder_uses_lengths_not_padding() -> None:
    torch.manual_seed(42)
    model = GRUClassifier(
        GRUConfig(vocabulary_size=10, embedding_dim=4, hidden_dim=6, dropout=0.0)
    ).eval()
    short = torch.tensor([[2, 3, 4]], dtype=torch.long)
    padded = torch.tensor([[2, 3, 4, 0, 0]], dtype=torch.long)
    length = torch.tensor([3], dtype=torch.long)
    with torch.inference_mode():
        short_output = model(short, length)["binary"]
        padded_output = model(padded, length)["binary"]
    assert torch.allclose(short_output, padded_output, atol=1e-6)


def test_training_restores_loadable_checkpoint_and_benchmarks(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    artifact = tmp_path / "artifact"
    prepare_dataset(
        _records(),
        processed,
        source="test-fixture",
        seed=42,
        ratios=(0.70, 0.15, 0.15),
    )
    # Read the JSON object to get its actual number of event IDs.
    vocabulary_size = len(json.loads((processed / "vocabulary.json").read_text()))
    result = train_gru(
        processed,
        artifact,
        model_config=GRUConfig(
            vocabulary_size=vocabulary_size,
            embedding_dim=8,
            hidden_dim=12,
            dropout=0.1,
        ),
        settings=TrainingSettings(
            seed=42,
            batch_size=32,
            epochs=2,
            patience=2,
            learning_rate=0.01,
            max_sequence_length=16,
            device="cpu",
        ),
    )
    assert result["metadata"]["best_epoch"] >= 1
    loaded = load_gru_artifact(artifact, "cpu")
    assert loaded.metadata["artifact_type"] == "pytorch-packed-bidirectional-gru"
    prediction = GRUPredictor(artifact, "cpu").predict(["E_START", "E_TIMEOUT", "E_END"])
    assert prediction.category is None and prediction.severity is None
    assert prediction.label_provenance == "fixture"
    evaluation = evaluate_artifact(artifact, processed, split="test", requested_device="cpu")
    assert evaluation["samples"] > 0
    benchmark = benchmark_artifact(
        artifact,
        processed,
        iterations=2,
        warmup=1,
        batch_size=2,
        requested_device="cpu",
    )
    assert benchmark["throughput_sequences_per_second"] > 0

    policy_path = artifact / "policy.json"
    policy_path.write_text(policy_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        load_gru_artifact(artifact, "cpu")
