"""Validation-only decision and uncertainty policy selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def select_policy(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    minimum_coverage: float = 0.80,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0 or y_true.shape != y_score.shape:
        raise ValueError("Validation labels and probabilities must be non-empty and aligned")
    if np.unique(y_true).size != 2:
        raise ValueError("Validation policy selection requires both binary classes")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")

    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score)))
    scored = [
        (
            float(f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    validation_f1, _, threshold = max(scored)

    review_candidates: list[tuple[float, float, float, float, float]] = []
    for margin in np.linspace(0.0, 0.25, 26):
        low = max(0.0, threshold - float(margin))
        high = min(1.0, threshold + float(margin))
        automatic = (y_score < low) | (y_score >= high)
        coverage = float(np.mean(automatic))
        if coverage + 1e-12 < minimum_coverage or not automatic.any():
            continue
        selective_f1 = float(
            f1_score(
                y_true[automatic],
                (y_score[automatic] >= threshold).astype(int),
                zero_division=0,
            )
        )
        review_candidates.append((selective_f1, float(margin), coverage, low, high))
    selective_f1, _, coverage, review_low, review_high = max(review_candidates)
    return {
        "threshold": threshold,
        "review_low": review_low,
        "review_high": review_high,
        "selection_split": "validation",
        "threshold_objective": "f1",
        "validation_f1": validation_f1,
        "review_objective": "maximize_selective_f1_at_minimum_coverage",
        "minimum_coverage": minimum_coverage,
        "selected_coverage": coverage,
        "selected_selective_f1": selective_f1,
    }


def apply_policy(probability: float, policy: dict[str, Any]) -> dict[str, bool]:
    threshold = float(policy["threshold"])
    review_low = float(policy["review_low"])
    review_high = float(policy["review_high"])
    return {
        "anomaly": probability >= threshold,
        "manual_review": review_low <= probability < review_high,
    }
