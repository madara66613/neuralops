# NeuralOps

[![CI](https://github.com/madara66613/neuralops/actions/workflows/ci.yml/badge.svg)](https://github.com/madara66613/neuralops/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GRU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)

NeuralOps is a production-minded log-sequence anomaly detection system: reproducible data preparation, a transparent statistical baseline, a PyTorch GRU, calibrated decision policy, FastAPI inference, and a React operator console.

The repository deliberately separates two evidence tracks:

- **Public benchmark — HDFS v1:** binary anomaly detection at HDFS block/session level. HDFS does not provide incident category or severity labels, so NeuralOps does not invent them.
- **Product demonstration — OpsForge Sim v1:** clearly labeled synthetic operations sequences with anomaly category and severity targets. Its metrics never stand in for real-world performance.

## Current status

Development is organized as small reviewable milestone pull requests. M0 established the research contract and CI foundation; M1 adds verified HDFS preparation and the statistical baseline. Numerical claims come only from generated, versioned evaluation artifacts.

### Verified baseline

| Profile | Model | Test samples | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HDFS v1 deduplicated sequence | TF-IDF + logistic regression | 2,740 | 0.9527 | 0.9788 | 0.9656 | 0.9934 | 0.9981 |

These are not conventional all-block HDFS numbers: 575,061 block traces were reduced to 18,383 deterministic `(fingerprint, label)` representatives before splitting. Exact values, confusion matrix, hashes, environment, and reproduction commands are in [the machine-readable baseline report](reports/hdfs-v1-deduplicated-baseline.json).

## Research contract

- Split by block/session group, never by individual log line.
- Assign duplicate normalized sequences to one split before measuring performance.
- Fit event vocabulary and transforms on training data only.
- Select thresholds and uncertainty policy on validation data only.
- Touch the test set once for the final report associated with an artifact.
- Report dataset identity, split manifest, seed, model hash, and measurement hardware beside metrics.

See [dataset evidence](docs/DATASETS.md), [architecture](docs/ARCHITECTURE.md), and [acceptance criteria](docs/ACCEPTANCE.md).

## Command surface

```bash
neuralops download-hdfs
neuralops prepare --config configs/hdfs-binary.yaml
neuralops train-baseline --config configs/hdfs-binary.yaml
neuralops train-baseline --config configs/hdfs-binary.yaml \
  --artifact artifacts/hdfs-v1-deduplicated/baseline
```

Neural training, evaluation, prediction, benchmarking, and serving commands land in subsequent milestones and are validated before the first release.

## License and data

NeuralOps source code is MIT licensed. Loghub data is a separate CC BY 4.0 dataset and is not bundled here. The official record, required attribution, checksum, and citations are documented in [docs/DATASETS.md](docs/DATASETS.md).
