"""Reproducible error analysis for a locked GRU artifact."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from neuralops.data.io import read_records
from neuralops.data.records import SequenceRecord
from neuralops.predictor import GRUPredictor, Prediction


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    event_occurrences: Counter[str] = Counter()
    event_documents: Counter[str] = Counter()
    for case in cases:
        events = case["events"]
        if not isinstance(events, list):
            raise TypeError("Case events must be a list")
        event_occurrences.update(str(event) for event in events)
        event_documents.update({str(event) for event in events})
    case_count = len(cases)
    common_events = (
        [
            {
                "event": event,
                "occurrences": count,
                "case_count": event_documents[event],
                "case_share": event_documents[event] / case_count,
            }
            for event, count in event_occurrences.most_common(10)
        ]
        if case_count
        else []
    )
    return {
        "count": case_count,
        "anomaly_probability": _distribution(
            [float(case["anomaly_probability"]) for case in cases]
        ),
        "sequence_length": _distribution([float(case["sequence_length"]) for case in cases]),
        "unique_event_count": _distribution([float(case["unique_event_count"]) for case in cases]),
        "manual_review_count": sum(bool(case["manual_review"]) for case in cases),
        "truncated_count": sum(bool(case["truncated"]) for case in cases),
        "unknown_event_count": sum(int(case["unknown_event_count"]) for case in cases),
        "common_events": common_events,
    }


def _case(record: SequenceRecord, prediction: Prediction) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "fingerprint": record.fingerprint,
        "true_label": record.anomaly,
        "predicted_label": int(prediction.predicted_anomaly),
        "anomaly_probability": prediction.anomaly_probability,
        "decision": prediction.decision,
        "manual_review": prediction.manual_review,
        "sequence_length": len(record.events),
        "unique_event_count": len(set(record.events)),
        "unknown_event_count": prediction.unknown_event_count,
        "truncated": prediction.truncated,
        "events": list(record.events),
    }


def analyze_errors(
    artifact_dir: Path,
    processed_dir: Path,
    *,
    split: str = "test",
    requested_device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Score one prepared split and summarize every false positive/negative."""
    if split not in {"validation", "test"}:
        raise ValueError("Error analysis split must be validation or test")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = read_records(processed_dir / "splits" / f"{split}.jsonl")
    predictor = GRUPredictor(artifact_dir, requested_device)
    predictions: list[Prediction] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        predictions.extend(predictor.predict_batch([list(record.events) for record in batch]))

    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    true_positives = 0
    true_negatives = 0
    manual_review_count = 0
    for record, prediction in zip(records, predictions, strict=True):
        manual_review_count += int(prediction.manual_review)
        if record.anomaly == 1 and prediction.predicted_anomaly:
            true_positives += 1
        elif record.anomaly == 0 and not prediction.predicted_anomaly:
            true_negatives += 1
        elif record.anomaly == 0:
            false_positives.append(_case(record, prediction))
        else:
            false_negatives.append(_case(record, prediction))

    metadata = predictor.loaded.metadata
    return {
        "schema_version": 1,
        "analysis": "locked-artifact-sequence-error-analysis",
        "split": split,
        "profile": metadata["profile"],
        "source": metadata["source"],
        "label_provenance": metadata["label_provenance"],
        "model_sha256": metadata["model_sha256"],
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "decision_threshold": predictor.loaded.policy["threshold"],
        "review_band": {
            "low": predictor.loaded.policy["review_low"],
            "high": predictor.loaded.policy["review_high"],
        },
        "runtime_device": str(predictor.loaded.device),
        "samples": len(records),
        "class_balance": {
            "normal": sum(record.anomaly == 0 for record in records),
            "anomaly": sum(record.anomaly == 1 for record in records),
        },
        "confusion_matrix": {
            "tn": true_negatives,
            "fp": len(false_positives),
            "fn": len(false_negatives),
            "tp": true_positives,
        },
        "manual_review_count": manual_review_count,
        "false_positive_summary": _summarize_cases(false_positives),
        "false_negative_summary": _summarize_cases(false_negatives),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "limitations": [
            "This analysis covers the deduplicated HDFS v1 test profile only.",
            "Event IDs are parser templates, not raw log messages or causal explanations.",
            "Common events are descriptive and do not establish root cause.",
        ],
    }
