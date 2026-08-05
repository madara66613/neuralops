from pathlib import Path

from neuralops.baseline.model import train_baseline
from neuralops.data.prepare import prepare_dataset
from neuralops.data.records import SequenceRecord


def _records() -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for index in range(300):
        anomaly = index % 4 == 0
        signal = "E_FAILURE" if anomaly else "E_HEALTHY"
        records.append(
            SequenceRecord(
                session_id=f"s-{index}",
                group_id=f"g-{index}",
                events=("E_START", signal, f"E_UNIQUE_{index}", "E_END"),
                anomaly=int(anomaly),
                source="test-fixture",
            )
        )
    return records


def test_baseline_artifact_contains_separate_validation_and_test_metrics(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    artifact = tmp_path / "artifact"
    prepare_dataset(
        _records(),
        processed,
        source="test-fixture",
        seed=42,
        ratios=(0.70, 0.15, 0.15),
    )
    result = train_baseline(processed, artifact, seed=42)
    assert set(result["metrics"]) == {"validation", "test"}
    assert result["policy"]["selection_split"] == "validation"
    assert result["metadata"]["label_provenance"] == "fixture-or-synthetic"
    assert result["metadata"]["profile"] == "test-fixture"
    assert (artifact / "model.joblib").stat().st_size > 0
