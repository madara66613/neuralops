# Acceptance criteria

NeuralOps v1.0 is complete only when all criteria below are evidenced by automated output or a linked artifact. A checked box must not rely on an unsupported prose claim.

## Data and research

- [ ] Official HDFS v1 download is checksum-verified and attribution is preserved.
- [ ] Preparation emits schema, class balance, length distribution, unknown-token rate, and manifest hashes.
- [ ] Automated checks prove group and duplicate fingerprints are split-disjoint.
- [ ] Vocabulary and transforms are fitted from train only.
- [ ] CI fixture and public benchmark profiles are unmistakably separated.

## Modeling

- [ ] TF-IDF + logistic regression baseline is trained and evaluated.
- [ ] PyTorch GRU uses packed/masked variable-length input and checkpoint restoration.
- [ ] Seeds, device choice, early stopping, gradient clipping, and training history are recorded.
- [ ] Threshold and review band are selected exclusively on validation.
- [ ] Binary precision, recall, F1, PR-AUC, ROC-AUC, FPR, and FNR are reported on test.
- [ ] Synthetic category metrics include macro/weighted F1 and a confusion matrix.
- [ ] Parameter count, artifact size, latency, and throughput include measurement context.

## Product and quality

- [ ] CLI supports prepare, baseline/train, evaluate, predict, and benchmark.
- [ ] API exposes health, readiness, version, model, single, and batch prediction endpoints.
- [ ] API enforces payload limits, structured errors, request IDs, and structured logs.
- [ ] React/TypeScript console works for samples, pasted sequences, batches, and error cases.
- [ ] Pytest, Ruff, mypy, Vitest, Playwright, and Docker image builds pass in CI.
- [ ] Docker Compose starts the inference and frontend services from a clean clone plus artifact.
- [ ] Model card, error analysis, architecture, screenshots, limitations, and reproducibility commands are current.
- [ ] Final changes are merged through milestone pull requests and tagged as a GitHub release.

## Claim policy

- No rounded or placeholder metric appears in README, UI, release notes, or model card.
- Every numerical performance claim can be regenerated from a committed command and machine-readable report.
- Synthetic results are labeled synthetic at the point of display.
- The repository does not claim production deployment, real-world superiority, or incident prevention without evidence.

