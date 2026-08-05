import pytest

from neuralops.predictor import GRUPredictor


class _StubPrediction:
    def __init__(self, probability: float) -> None:
        self.anomaly_probability = probability

    def to_dict(self) -> dict[str, float]:
        return {"anomaly_probability": self.anomaly_probability}


def test_leave_one_out_ranks_largest_absolute_change(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = object.__new__(GRUPredictor)
    probabilities = iter([0.2, 0.8, 0.45])
    monkeypatch.setattr(predictor, "predict", lambda events: _StubPrediction(0.5))
    monkeypatch.setattr(
        predictor,
        "predict_batch",
        lambda sequences: [_StubPrediction(next(probabilities)) for _ in sequences],
    )

    result = predictor.leave_one_out_sensitivity(["E1", "E2", "E3"])
    assert result["method"] == "leave_one_event_out_probability_sensitivity"
    assert result["evidence"][0]["event"] == "E2"
    assert result["evidence"][0]["effect"] == "suppresses_anomaly"


def test_leave_one_out_single_event_has_no_empty_ablation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = object.__new__(GRUPredictor)
    monkeypatch.setattr(predictor, "predict", lambda events: _StubPrediction(0.4))
    result = predictor.leave_one_out_sensitivity(["E1"])
    assert result["evidence"] == []
    assert result["evaluated_event_count"] == 1
