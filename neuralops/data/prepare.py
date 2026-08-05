"""Dataset preparation and auditable quality reporting."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

from neuralops import __version__
from neuralops.data.io import sha256_file, write_json, write_records
from neuralops.data.records import SequenceRecord
from neuralops.data.split import SPLIT_NAMES, assert_disjoint, split_records
from neuralops.data.vocab import build_vocabulary, unknown_rate


def deduplicate_records(
    records: list[SequenceRecord],
) -> tuple[list[SequenceRecord], dict[str, int]]:
    """Keep one representative per fingerprint/label before split assignment."""
    buckets: dict[tuple[str, int], list[SequenceRecord]] = {}
    labels_by_fingerprint: dict[str, set[int]] = {}
    for record in records:
        buckets.setdefault((record.fingerprint, record.anomaly), []).append(record)
        labels_by_fingerprint.setdefault(record.fingerprint, set()).add(record.anomaly)

    selected: list[SequenceRecord] = []
    for (fingerprint, _label), bucket in sorted(buckets.items()):
        representative = min(bucket, key=lambda item: item.session_id)
        if len(labels_by_fingerprint[fingerprint]) > 1:
            representative = replace(representative, group_id=f"duplicate:{fingerprint}")
        selected.append(representative)
    selected.sort(key=lambda item: item.session_id)
    return selected, {
        "input_records": len(records),
        "output_records": len(selected),
        "removed_records": len(records) - len(selected),
        "unique_fingerprints": len(labels_by_fingerprint),
        "conflicting_label_fingerprints": sum(
            len(labels) > 1 for labels in labels_by_fingerprint.values()
        ),
    }


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(records: list[SequenceRecord], vocabulary: dict[str, int]) -> dict[str, Any]:
    lengths = [len(record.events) for record in records]
    labels = Counter(str(record.anomaly) for record in records)
    labels_by_fingerprint: dict[str, set[int]] = {}
    for record in records:
        labels_by_fingerprint.setdefault(record.fingerprint, set()).add(record.anomaly)
    return {
        "records": len(records),
        "class_counts": dict(sorted(labels.items())),
        "anomaly_rate": labels.get("1", 0) / len(records) if records else 0.0,
        "groups": len({record.group_id for record in records}),
        "unique_fingerprints": len({record.fingerprint for record in records}),
        "duplicate_records": len(records) - len({record.fingerprint for record in records}),
        "label_conflict_fingerprints": sum(
            len(fingerprint_labels) > 1 for fingerprint_labels in labels_by_fingerprint.values()
        ),
        "sequence_length": {
            "min": min(lengths, default=0),
            "median": median(lengths) if lengths else 0,
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths, default=0),
        },
        "unknown_event_rate": unknown_rate((record.events for record in records), vocabulary),
    }


def prepare_dataset(
    records: list[SequenceRecord],
    output_dir: Path,
    *,
    source: str,
    profile: str | None = None,
    seed: int,
    ratios: tuple[float, float, float],
    deduplicate: bool = True,
) -> dict[str, Any]:
    """Prepare canonical splits, train-only vocabulary, and provenance manifests."""
    input_records = records
    if deduplicate:
        records, deduplication = deduplicate_records(input_records)
    else:
        deduplication = {
            "input_records": len(input_records),
            "output_records": len(input_records),
            "removed_records": 0,
            "unique_fingerprints": len({record.fingerprint for record in input_records}),
            "conflicting_label_fingerprints": 0,
        }
    splits = split_records(records, seed=seed, ratios=ratios)
    assert_disjoint(splits)
    if not splits["train"]:
        raise ValueError("Stable split produced an empty training set")
    vocabulary = build_vocabulary(record.events for record in splits["train"])

    split_hashes: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        split_path = output_dir / "splits" / f"{split_name}.jsonl"
        write_records(split_path, splits[split_name])
        split_hashes[split_name] = sha256_file(split_path)

    vocabulary_path = output_dir / "vocabulary.json"
    write_json(vocabulary_path, vocabulary)
    report = {
        "schema_version": 1,
        "source": source,
        "profile": profile or source,
        "label_provenance": "public" if source == "loghub-hdfs-v1" else "synthetic",
        "seed": seed,
        "requested_ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "records_total": len(records),
        "deduplication": {"enabled": deduplicate, **deduplication},
        "vocabulary_size": len(vocabulary),
        "vocabulary_fitted_on": "train",
        "splits": {
            split_name: _summary(splits[split_name], vocabulary) for split_name in SPLIT_NAMES
        },
        "leakage_checks": {
            "session_disjoint": True,
            "group_disjoint": True,
            "fingerprint_disjoint": True,
        },
    }
    write_json(output_dir / "data_quality.json", report)

    assignment_payload = {
        split_name: [record.session_id for record in splits[split_name]]
        for split_name in SPLIT_NAMES
    }
    assignment_hash = hashlib.sha256(
        json.dumps(assignment_payload, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "neuralops_version": __version__,
        "source": source,
        "profile": profile or source,
        "seed": seed,
        "split_strategy": "duplicate-clustered-stable-hash",
        "split_files_sha256": split_hashes,
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "assignment_sha256": assignment_hash,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return {"manifest": manifest, "data_quality": report}
