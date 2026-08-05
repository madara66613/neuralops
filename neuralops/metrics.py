"""Metrics with explicit edge-case handling."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays")
    predictions = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    both_classes = np.unique(y_true).size == 2
    return {
        "samples": int(y_true.size),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
