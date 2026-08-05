"""Production-shaped FastAPI application factory."""

from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response

from neuralops import __version__
from neuralops.api.logging import configure_logging
from neuralops.api.schemas import (
    MAX_BODY_BYTES,
    BatchInput,
    BatchPredictionResponse,
    ErrorResponse,
    ModelResponse,
    PredictionResponse,
    SequenceInput,
    StatusResponse,
    VersionResponse,
)
from neuralops.predictor import GRUPredictor, Prediction

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
logger = configure_logging()


class Predictor(Protocol):
    @property
    def model_info(self) -> dict[str, Any]: ...

    def predict(self, events: list[str]) -> Prediction: ...

    def predict_batch(self, sequences: list[list[str]]) -> list[Prediction]: ...


def _request_id(request: Request) -> str:
    value = request.headers.get("X-Request-ID", "")
    return value if REQUEST_ID_PATTERN.fullmatch(value) else str(uuid.uuid4())


def _error(
    request_id: str,
    code: str,
    message: str,
    status_code: int,
    *,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "error": {"code": code, "message": message, "details": details},
        },
        headers={"X-Request-ID": request_id},
    )


def create_app(
    *,
    predictor: Predictor | None = None,
    artifact_dir: Path | None = None,
    requested_device: str = "auto",
) -> FastAPI:
    application = FastAPI(
        title="NeuralOps Inference API",
        version=__version__,
        description="Verified log-sequence anomaly inference with explicit provenance.",
    )
    origins = [
        origin.strip()
        for origin in os.getenv(
            "NEURALOPS_CORS_ORIGINS",
            (
                "http://localhost:4173,http://localhost:5173,"
                "http://127.0.0.1:4173,http://127.0.0.1:5173"
            ),
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    application.state.predictor = predictor
    application.state.load_error = None
    selected_artifact = artifact_dir or (
        Path(os.environ["NEURALOPS_ARTIFACT_DIR"]) if os.getenv("NEURALOPS_ARTIFACT_DIR") else None
    )
    if application.state.predictor is None and selected_artifact is not None:
        try:
            application.state.predictor = GRUPredictor(selected_artifact, requested_device)
        except Exception as exc:  # Startup remains live but explicitly unready.
            application.state.load_error = type(exc).__name__
            logger.exception("model_load_failed", extra={"error": type(exc).__name__})

    @application.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = _request_id(request)
        started = time.perf_counter_ns()
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > MAX_BODY_BYTES:
                return _error(
                    request.state.request_id,
                    "PAYLOAD_TOO_LARGE",
                    f"Request body exceeds {MAX_BODY_BYTES} bytes",
                    413,
                )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000
        logger.info(
            "request_completed",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return _error(
            request.state.request_id,
            "VALIDATION_ERROR",
            "Request validation failed",
            422,
            details=details,
        )

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        code = "MODEL_NOT_READY" if exc.status_code == 503 else "HTTP_ERROR"
        return _error(request.state.request_id, code, str(exc.detail), exc.status_code)

    @application.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_request_error",
            extra={"request_id": request.state.request_id, "error": type(exc).__name__},
        )
        return _error(
            request.state.request_id,
            "INTERNAL_ERROR",
            "An internal error occurred",
            500,
        )

    def require_predictor() -> Predictor:
        active: Predictor | None = application.state.predictor
        if active is None:
            raise HTTPException(status_code=503, detail="Model artifact is not ready")
        return active

    @application.get("/health", response_model=StatusResponse)
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "request_id": request.state.request_id}

    @application.get("/ready", response_model=StatusResponse)
    def ready(request: Request) -> dict[str, str]:
        require_predictor()
        return {"status": "ready", "request_id": request.state.request_id}

    @application.get("/version", response_model=VersionResponse)
    def version(request: Request) -> dict[str, str | int]:
        return {
            "version": __version__,
            "api_schema_version": 1,
            "request_id": request.state.request_id,
        }

    @application.get("/model", response_model=ModelResponse)
    def model(request: Request) -> dict[str, Any]:
        active = require_predictor()
        return {"request_id": request.state.request_id, "model": active.model_info}

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def predict(request: Request, payload: SequenceInput) -> dict[str, Any]:
        result = require_predictor().predict(payload.events)
        return {"request_id": request.state.request_id, "prediction": result.to_dict()}

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def predict_batch(request: Request, payload: BatchInput) -> dict[str, Any]:
        results = require_predictor().predict_batch(
            [sequence.events for sequence in payload.sequences]
        )
        return {
            "request_id": request.state.request_id,
            "count": len(results),
            "predictions": [result.to_dict() for result in results],
        }

    return application


app = create_app()
