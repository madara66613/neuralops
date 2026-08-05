"""TF-IDF event n-grams with class-weighted logistic regression."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from neuralops import __version__
from neuralops.data.io import read_json, read_records, sha256_file, write_json
from neuralops.data.records import SequenceRecord
from neuralops.metrics import binary_metrics
from neuralops.policy import select_policy


def _documents(records: list[SequenceRecord]) -> list[str]:
    return [" ".join(record.events) for record in records]


def build_baseline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    token_pattern=r"(?u)\b\S+\b",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _probabilities(model: Pipeline, records: list[SequenceRecord]) -> np.ndarray:
    values = model.predict_proba(_documents(records))
    return np.asarray(values[:, 1], dtype=np.float64)


def train_baseline(
    processed_dir: Path,
    artifact_dir: Path,
    *,
    seed: int,
    minimum_coverage: float = 0.80,
) -> dict[str, Any]:
    train = read_records(processed_dir / "splits" / "train.jsonl")
    validation = read_records(processed_dir / "splits" / "validation.jsonl")
    test = read_records(processed_dir / "splits" / "test.jsonl")
    if set(record.anomaly for record in train) != {0, 1}:
        raise ValueError("Baseline training requires both classes in the training split")

    model = build_baseline(seed)
    model.fit(_documents(train), [record.anomaly for record in train])
    validation_probabilities = _probabilities(model, validation)
    policy = select_policy(
        [record.anomaly for record in validation],
        validation_probabilities,
        minimum_coverage=minimum_coverage,
    )
    threshold = float(policy["threshold"])
    test_probabilities = _probabilities(model, test)
    metrics = {
        "validation": binary_metrics(
            [record.anomaly for record in validation], validation_probabilities, threshold
        ),
        "test": binary_metrics([record.anomaly for record in test], test_probabilities, threshold),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path)
    classifier = model.named_steps["classifier"]
    data_manifest = read_json(processed_dir / "manifest.json")
    parameter_count = int(classifier.coef_.size + classifier.intercept_.size)
    metadata = {
        "schema_version": 1,
        "artifact_type": "tfidf-logistic-regression",
        "neuralops_version": __version__,
        "label_provenance": "public"
        if all(record.source == "loghub-hdfs-v1" for record in train)
        else "fixture-or-synthetic",
        "source": train[0].source,
        "profile": str(data_manifest.get("profile", train[0].source)),
        "seed": seed,
        "parameter_count": parameter_count,
        "artifact_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "data_manifest_sha256": sha256_file(processed_dir / "manifest.json"),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(artifact_dir / "policy.json", policy)
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "metadata.json", metadata)
    return {"policy": policy, "metrics": metrics, "metadata": metadata}
