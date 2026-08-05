from typing import Any

from fastapi.testclient import TestClient

from neuralops.api.app import create_app
from neuralops.api.schemas import MAX_BODY_BYTES, MAX_EVENTS
from neuralops.predictor import Prediction


class FakePredictor:
    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "metadata": {
                "profile": "fake-profile",
                "label_provenance": "fixture",
                "model_sha256": "abc123",
            },
            "policy": {"threshold": 0.5, "review_low": 0.4, "review_high": 0.6},
            "label_mappings": {},
            "device": "cpu",
        }

    def predict(self, events: list[str]) -> Prediction:
        anomalous = "E_BAD" in events
        probability = 0.9 if anomalous else 0.1
        return Prediction(
            predicted_anomaly=anomalous,
            anomaly_probability=probability,
            confidence=0.9,
            confidence_kind="raw_model_outcome_probability_not_calibrated",
            decision="anomaly" if anomalous else "normal",
            manual_review=False,
            category=None,
            category_confidence=None,
            severity=None,
            severity_confidence=None,
            input_event_count=len(events),
            unknown_event_count=0,
            unknown_event_rate=0.0,
            truncated=False,
            profile="fake-profile",
            label_provenance="fixture",
        )

    def predict_batch(self, sequences: list[list[str]]) -> list[Prediction]:
        return [self.predict(events) for events in sequences]

    def leave_one_out_sensitivity(self, events: list[str]) -> dict[str, Any]:
        prediction = self.predict(events)
        return {
            "prediction": prediction.to_dict(),
            "method": "leave_one_event_out_probability_sensitivity",
            "interpretation": (
                "Descriptive model sensitivity; not a causal or root-cause explanation."
            ),
            "input_event_count": len(events),
            "evaluated_event_count": len(events),
            "evaluation_limited": False,
            "evidence": [
                {
                    "event_index": 1,
                    "event": events[1],
                    "anomaly_probability_without_event": 0.1,
                    "anomaly_probability_delta": 0.8,
                    "absolute_delta": 0.8,
                    "effect": "supports_anomaly",
                }
            ],
        }


def test_health_request_id_readiness_and_model_metadata() -> None:
    client = TestClient(create_app(predictor=FakePredictor()))
    health = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "request_id": "trace-123"}
    assert health.headers["X-Request-ID"] == "trace-123"
    assert health.headers["Cache-Control"] == "no-store"
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/ready").json()["status"] == "ready"
    model = client.get("/model").json()["model"]
    assert model["metadata"]["profile"] == "fake-profile"
    assert client.get("/version").json()["api_schema_version"] == 1


def test_unready_service_is_live_but_returns_structured_503() -> None:
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"
    assert response.headers["X-Request-ID"]


def test_single_and_batch_prediction_contracts() -> None:
    client = TestClient(create_app(predictor=FakePredictor()))
    single = client.post("/predict", json={"events": ["E_START", "E_BAD"]})
    assert single.status_code == 200
    assert single.json()["model_version"] == "1.0.0"
    assert single.json()["inference_ms"] >= 0
    assert single.json()["prediction"]["predicted_anomaly"] is True
    assert single.json()["prediction"]["confidence_kind"].endswith("not_calibrated")

    batch = client.post(
        "/predict/batch",
        json={"sequences": [{"events": ["E_OK"]}, {"events": ["E_BAD"]}]},
    )
    assert batch.status_code == 200
    assert batch.json()["count"] == 2
    assert [item["decision"] for item in batch.json()["predictions"]] == ["normal", "anomaly"]

    sensitivity = client.post("/predict/sensitivity", json={"events": ["E_START", "E_BAD"]})
    assert sensitivity.status_code == 200
    assert sensitivity.json()["method"] == "leave_one_event_out_probability_sensitivity"
    assert sensitivity.json()["evidence"][0]["event"] == "E_BAD"


def test_validation_and_payload_limits_return_safe_errors() -> None:
    client = TestClient(create_app(predictor=FakePredictor()))
    empty = client.post("/predict", json={"events": []})
    assert empty.status_code == 422
    assert empty.json()["error"]["code"] == "VALIDATION_ERROR"
    assert empty.json()["error"]["message"] == "Request validation failed"

    too_many = client.post("/predict", json={"events": ["E1"] * (MAX_EVENTS + 1)})
    assert too_many.status_code == 422
    controls = client.post("/predict", json={"events": ["E1\nE2"]})
    assert controls.status_code == 422
    extra = client.post("/predict", json={"events": ["E1"], "admin": True})
    assert extra.status_code == 422

    oversized = client.post(
        "/predict",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
