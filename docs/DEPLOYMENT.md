# Local deployment

## One-command demo

```bash
docker compose up --build
```

Open `http://127.0.0.1:4173`. The API is also exposed on `http://127.0.0.1:8001`; override host ports with `NEURALOPS_UI_PORT` and `NEURALOPS_API_PORT`.

The default stack mounts the checked-in 660 KiB synthetic demo artifact read-only. No API key, paid service, GPU, or model download is required. Stop the stack with:

```bash
docker compose down
```

## Container boundary

- `neuralops-api:1.0.0` uses Python 3.12.11 and the PyTorch 2.7.1 CPU wheel.
- `neuralops-console:1.0.0` builds with Node 22 and serves static assets from unprivileged nginx.
- Both services run without root, drop Linux capabilities, enable `no-new-privileges`, and use read-only root filesystems with small temporary mounts.
- The browser calls `/api`; nginx proxies to the private Compose API service.
- Health checks gate console startup on a SHA-256-verified model loaded by `/ready`.

## Smoke test

```bash
curl --fail http://127.0.0.1:8001/ready

curl --fail http://127.0.0.1:4173/api/predict \
  -H 'Content-Type: application/json' \
  --data '{"events":["E001","E012","E104","E207","E104","E431","P07","E087","E099"]}'
```

## Use another trusted artifact

Replace the `api` volume source in `docker-compose.yml` or use a Compose override that mounts a complete artifact directory at `/app/artifact:ro`. Never mount an artifact received from an untrusted request. NeuralOps uses `torch.load(..., weights_only=True)` and verifies recorded hashes, but operators must still control the artifact path and provenance.

The default container is inference-only. Training remains a local reproducible workflow because it requires data preparation and is not part of the serving image.
