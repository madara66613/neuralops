# Security policy

## Supported version

Security fixes target the latest tagged NeuralOps release. This is a portfolio research project, not a hosted service or supported commercial product.

## Report a vulnerability

Open a GitHub security advisory for the repository rather than a public issue when a report contains an exploitable detail. Do not include real log payloads, credentials, private model artifacts, or personal data.

## Trust boundary

- Treat every API payload as untrusted; body, batch, sequence, and token limits are enforced before inference.
- Input is parsed as data and is never passed to a shell, SQL engine, or remediation workflow.
- Request logs exclude event payloads and return structured errors without stack traces.
- The service has no authentication or public rate limiter. Place it behind a trusted gateway before any shared deployment.
- Model paths are operator configuration, not request input. Load only reviewed artifacts. PyTorch serialization can be unsafe when artifact trust is unknown even though NeuralOps uses `weights_only=True` and verifies recorded hashes.
- The bundled synthetic artifact contains no credentials or production logs.

See the [deployment guide](docs/DEPLOYMENT.md) and [model card](reports/model-card.md) for operational controls and limitations.
