# Inference API

NeuralOps serves a single SHA-256-verified GRU artifact through FastAPI. The same `GRUPredictor` powers CLI and HTTP inference, so policy thresholds and nullable auxiliary outputs cannot drift between interfaces.

## Start locally

```bash
neuralops serve \
  --artifact artifacts/opsforge-sim-v1/multitask-gru \
  --host 127.0.0.1 \
  --port 8000 \
  --device cpu
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. CORS defaults to the `localhost` and `127.0.0.1` loopback origins on ports 4173/5173; override it with a comma-separated `NEURALOPS_CORS_ORIGINS` value.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process liveness; does not imply a loaded model |
| GET | `/ready` | Returns 200 only when an integrity-checked artifact is loaded |
| GET | `/version` | Package and API schema versions |
| GET | `/model` | Profile, label provenance, policy, labels, hashes, and runtime metadata |
| POST | `/predict` | One event sequence |
| POST | `/predict/sensitivity` | One prediction plus bounded leave-one-event-out score changes |
| POST | `/predict/batch` | Up to 64 event sequences |

## Request example

```bash
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-001' \
  --data '{"events":["E001","E012","E104","E207","E104","E431","P07","E087","E099"]}'
```

Responses contain software version, measured model inference time, raw anomaly probability, the validation-selected decision, manual-review status, input diagnostics, profile, and label provenance. `confidence` is the raw probability of the selected binary outcome and is explicitly marked **not calibrated**. Category and severity are nullable: they appear only for predicted anomalies from an artifact with auxiliary heads.

`/predict/sensitivity` removes each of at most the first 64 event positions, scores those ablations as one synchronized batch, and returns the eight largest absolute changes from the full-sequence anomaly probability. Repeated tokens retain separate positions. A positive delta means removing that event lowered the anomaly score; a negative delta means removal raised it. This is local model sensitivity, not causal attribution or root-cause evidence.

## Limits

- request body: 1 MiB;
- sequences per batch: 64;
- events per sequence: 512;
- event token length: 128 characters;
- control characters and unknown fields: rejected;
- sequences longer than the artifact limit: accepted but reported as `truncated=true`;
- unknown tokens: mapped to `<UNK>` and counted in the response.

## Errors and observability

Errors use one response shape:

```json
{
  "request_id": "4aab...",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": []
  }
}
```

The server accepts a safe `X-Request-ID` or creates a UUID, echoes it in every response header/body, and writes JSON logs with method, path, status, and duration. It never logs request payloads. Responses set `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.
