import zipfile
from pathlib import Path

import pytest

from neuralops.data.hdfs import _safe_extract, load_hdfs_records, md5_file
from neuralops.data.prepare import deduplicate_records, prepare_dataset
from neuralops.data.records import SequenceRecord
from neuralops.data.split import assert_disjoint, split_records
from neuralops.data.vocab import PAD_TOKEN, UNK_TOKEN, build_vocabulary, unknown_rate

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hdfs"


def record(index: int, anomaly: int, events: tuple[str, ...] | None = None) -> SequenceRecord:
    return SequenceRecord(
        session_id=f"session-{index}",
        group_id=f"group-{index}",
        events=events or (("E_BAD", f"E_{index}") if anomaly else ("E_OK", f"E_{index}")),
        anomaly=anomaly,
        source="test-fixture",
    )


def test_hdfs_trace_loader_preserves_block_labels() -> None:
    records = load_hdfs_records(FIXTURE_DIR)
    assert len(records) == 4
    assert [item.anomaly for item in records] == [0, 1, 0, 1]
    assert records[0].events == ("E1", "E2", "E3")
    assert all(item.category is None and item.severity is None for item in records)


def test_download_helpers_hash_and_reject_zip_traversal(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("neuralops", encoding="utf-8")
    assert md5_file(payload) == "2fb0c4192a02a13d55545979ba7c6a94"

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "blocked")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        _safe_extract(archive, tmp_path / "target")


def test_duplicate_fingerprints_cannot_cross_splits() -> None:
    records = [record(index, index % 2) for index in range(80)]
    records.extend(
        [
            record(100, 1, ("E_BAD", "E_DUP")),
            record(101, 1, ("E_BAD", "E_DUP")),
        ]
    )
    splits = split_records(records, seed=42)
    assert_disjoint(splits)
    duplicate_locations = {
        split_name
        for split_name, items in splits.items()
        if any(item.events == ("E_BAD", "E_DUP") for item in items)
    }
    assert len(duplicate_locations) == 1


def test_conflicting_duplicate_labels_still_remain_in_one_split() -> None:
    splits = split_records(
        [record(1, 0, ("E_DUP",)), record(2, 1, ("E_DUP",))],
        seed=42,
    )
    locations = {
        split_name
        for split_name, items in splits.items()
        if any(item.events == ("E_DUP",) for item in items)
    }
    assert len(locations) == 1


def test_deduplication_keeps_one_representative_per_fingerprint_and_label() -> None:
    selected, report = deduplicate_records(
        [
            record(1, 0, ("E_SAME",)),
            record(2, 0, ("E_SAME",)),
            record(3, 1, ("E_SAME",)),
        ]
    )
    assert len(selected) == 2
    assert report["removed_records"] == 1
    assert report["conflicting_label_fingerprints"] == 1
    assert selected[0].group_id == selected[1].group_id


def test_vocabulary_is_explicitly_train_only() -> None:
    vocabulary = build_vocabulary([("E1", "E2"), ("E1",)])
    assert vocabulary[PAD_TOKEN] == 0
    assert vocabulary[UNK_TOKEN] == 1
    assert "E_TEST_ONLY" not in vocabulary
    assert unknown_rate([("E1", "E_TEST_ONLY")], vocabulary) == 0.5


def test_prepare_emits_disjoint_manifest_and_quality_report(tmp_path: Path) -> None:
    records = [record(index, index % 2) for index in range(200)]
    result = prepare_dataset(
        records,
        tmp_path,
        source="test-fixture",
        seed=42,
        ratios=(0.7, 0.15, 0.15),
    )
    assert result["data_quality"]["vocabulary_fitted_on"] == "train"
    assert result["data_quality"]["leakage_checks"]["fingerprint_disjoint"] is True
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "splits" / "test.jsonl").is_file()
