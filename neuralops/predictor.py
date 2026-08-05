"""Thread-safe inference contract shared by CLI and API."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from neuralops.data.records import SequenceRecord
from neuralops.data.vocab import UNK_TOKEN
from neuralops.modeling.dataset import SequenceDataset, collate_sequences
from neuralops.modeling.training import load_gru_artifact
from neuralops.policy import apply_policy


@dataclass(frozen=True, slots=True)
class Prediction:
    predicted_anomaly: bool
    anomaly_probability: float
    confidence: float
    confidence_kind: str
    decision: str
    manual_review: bool
    category: str | None
    category_confidence: float | None
    severity: str | None
    severity_confidence: float | None
    input_event_count: int
    unknown_event_count: int
    unknown_event_rate: float
    truncated: bool
    profile: str
    label_provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GRUPredictor:
    """Load one verified artifact and serve synchronized batch inference."""

    def __init__(self, artifact_dir: Path, requested_device: str = "auto") -> None:
        self.artifact_dir = artifact_dir
        self.loaded = load_gru_artifact(artifact_dir, requested_device)
        self._lock = threading.Lock()

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "metadata": self.loaded.metadata,
            "policy": self.loaded.policy,
            "label_mappings": self.loaded.label_mappings,
            "device": str(self.loaded.device),
        }

    def predict(self, events: list[str]) -> Prediction:
        return self.predict_batch([events])[0]

    def predict_batch(self, sequences: list[list[str]]) -> list[Prediction]:
        if not sequences or any(not events for events in sequences):
            raise ValueError("Every prediction sequence must contain at least one event")
        records = [
            SequenceRecord(
                session_id=f"request-{index}",
                group_id=f"request-{index}",
                events=tuple(events),
                anomaly=0,
                source="inference-request",
            )
            for index, events in enumerate(sequences)
        ]
        dataset = SequenceDataset(
            records,
            self.loaded.vocabulary,
            self.loaded.max_sequence_length,
        )
        batch = collate_sequences([dataset[index] for index in range(len(dataset))]).to(
            self.loaded.device
        )
        with self._lock, torch.inference_mode():
            outputs = self.loaded.model(batch.tokens, batch.lengths)
            binary_probabilities = torch.sigmoid(outputs["binary"]).detach().cpu().tolist()
            category_probabilities = (
                torch.softmax(outputs["category"], dim=1).detach().cpu()
                if "category" in outputs
                else None
            )
            severity_probabilities = (
                torch.softmax(outputs["severity"], dim=1).detach().cpu()
                if "severity" in outputs
                else None
            )

        unknown_index = self.loaded.vocabulary[UNK_TOKEN]
        predictions: list[Prediction] = []
        for index, (events, probability) in enumerate(
            zip(sequences, binary_probabilities, strict=True)
        ):
            policy_result = apply_policy(float(probability), self.loaded.policy)
            predicted_anomaly = policy_result["anomaly"]
            manual_review = policy_result["manual_review"]
            category, category_confidence = self._auxiliary_prediction(
                category_probabilities,
                self.loaded.label_mappings.get("category", []),
                index,
                predicted_anomaly,
            )
            severity, severity_confidence = self._auxiliary_prediction(
                severity_probabilities,
                self.loaded.label_mappings.get("severity", []),
                index,
                predicted_anomaly,
            )
            considered = events[: self.loaded.max_sequence_length]
            unknown_count = sum(
                self.loaded.vocabulary.get(event, unknown_index) == unknown_index
                for event in considered
            )
            predictions.append(
                Prediction(
                    predicted_anomaly=predicted_anomaly,
                    anomaly_probability=float(probability),
                    confidence=float(probability if predicted_anomaly else 1 - probability),
                    confidence_kind="raw_model_outcome_probability_not_calibrated",
                    decision=(
                        "manual_review"
                        if manual_review
                        else "anomaly"
                        if predicted_anomaly
                        else "normal"
                    ),
                    manual_review=manual_review,
                    category=category,
                    category_confidence=category_confidence,
                    severity=severity,
                    severity_confidence=severity_confidence,
                    input_event_count=len(events),
                    unknown_event_count=unknown_count,
                    unknown_event_rate=unknown_count / len(considered),
                    truncated=len(events) > self.loaded.max_sequence_length,
                    profile=str(self.loaded.metadata["profile"]),
                    label_provenance=str(self.loaded.metadata["label_provenance"]),
                )
            )
        return predictions

    @staticmethod
    def _auxiliary_prediction(
        probabilities: torch.Tensor | None,
        class_names: list[str],
        index: int,
        predicted_anomaly: bool,
    ) -> tuple[str | None, float | None]:
        if probabilities is None or not class_names or not predicted_anomaly:
            return None, None
        row = probabilities[index]
        class_index = int(row.argmax())
        return class_names[class_index], float(row[class_index])
