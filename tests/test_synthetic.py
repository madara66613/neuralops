import json
import re
from pathlib import Path

from neuralops.data.io import read_records
from neuralops.data.prepare import prepare_dataset
from neuralops.data.synthetic import INCIDENT_PATTERNS, SEVERITIES, generate_synthetic_records
from neuralops.modeling.gru import GRUConfig
from neuralops.modeling.training import TrainingSettings, load_gru_artifact, train_gru


def test_synthetic_generator_keeps_labels_out_of_event_tokens() -> None:
    records = generate_synthetic_records(300, seed=42)
    assert {record.category for record in records if record.category} == set(INCIDENT_PATTERNS)
    assert {record.severity for record in records if record.severity} == set(SEVERITIES)
    for record in records:
        assert all(re.fullmatch(r"[EP]\d{2,3}", event) for event in record.events)
        if record.category is not None:
            assert record.category not in " ".join(record.events)
    labels_by_group: dict[str, set[tuple[int, str | None, str | None]]] = {}
    for record in records:
        labels_by_group.setdefault(record.group_id, set()).add(
            (record.anomaly, record.category, record.severity)
        )
    assert all(len(labels) == 1 for labels in labels_by_group.values())


def test_multitask_training_reports_category_and_severity_separately(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    artifact = tmp_path / "artifact"
    prepare_dataset(
        generate_synthetic_records(600, seed=7),
        processed,
        source="opsforge-sim-v1",
        profile="test-synthetic-multitask",
        seed=7,
        ratios=(0.70, 0.15, 0.15),
    )
    train = read_records(processed / "splits" / "train.jsonl")
    categories = {record.category for record in train if record.category is not None}
    severities = {record.severity for record in train if record.severity is not None}
    vocabulary_size = len(json.loads((processed / "vocabulary.json").read_text()))
    result = train_gru(
        processed,
        artifact,
        model_config=GRUConfig(
            vocabulary_size=vocabulary_size,
            embedding_dim=8,
            hidden_dim=12,
            dropout=0.1,
            category_classes=len(categories),
            severity_classes=len(severities),
        ),
        settings=TrainingSettings(
            seed=7,
            batch_size=64,
            epochs=2,
            patience=2,
            learning_rate=0.01,
            max_sequence_length=32,
            device="cpu",
        ),
    )
    assert set(result["metrics"]["test"]) == {"binary", "category", "severity"}
    assert result["metrics"]["test"]["category"]["samples"] > 0
    assert result["metrics"]["test"]["severity"]["macro_f1"] >= 0
    assert result["metadata"]["label_provenance"] == "synthetic"
    loaded = load_gru_artifact(artifact, "cpu")
    assert set(loaded.label_mappings) == {"category", "severity"}
