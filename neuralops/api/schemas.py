"""Strict request and response schemas."""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_EVENTS = 512
MAX_BATCH = 64
MAX_EVENT_LENGTH = 128
MAX_BODY_BYTES = 1024 * 1024
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")

EventToken = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_EVENT_LENGTH),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SequenceInput(StrictModel):
    events: list[EventToken] = Field(min_length=1, max_length=MAX_EVENTS)

    @field_validator("events")
    @classmethod
    def reject_control_characters(cls, events: list[str]) -> list[str]:
        if any(CONTROL_CHARACTERS.search(event) for event in events):
            raise ValueError("event tokens cannot contain control characters")
        return events


class BatchInput(StrictModel):
    sequences: list[SequenceInput] = Field(min_length=1, max_length=MAX_BATCH)


class PredictionPayload(StrictModel):
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


class PredictionResponse(StrictModel):
    request_id: str
    prediction: PredictionPayload


class BatchPredictionResponse(StrictModel):
    request_id: str
    count: int
    predictions: list[PredictionPayload]


class StatusResponse(StrictModel):
    status: str
    request_id: str


class VersionResponse(StrictModel):
    version: str
    api_schema_version: int
    request_id: str


class ModelResponse(StrictModel):
    request_id: str
    model: dict[str, Any]


class ErrorDetail(StrictModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(StrictModel):
    request_id: str
    error: ErrorDetail
