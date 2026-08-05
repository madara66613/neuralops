# Local data workspace

Raw and processed datasets are reproducible but intentionally excluded from Git.

```text
data/raw/hdfs-v1/          checksum-verified, extracted Loghub HDFS v1
data/processed/hdfs-v1/    grouped splits, quality report, manifest, vocabulary
data/processed/opsforge-sim-v1/  generated synthetic splits and metadata
```

Prepare the public profile with:

```bash
neuralops download-hdfs
neuralops prepare --config configs/hdfs-binary.yaml
```

Prepare the synthetic profile with:

```bash
neuralops prepare --config configs/opsforge-sim.yaml
```

Do not commit raw Loghub data. It is licensed separately under CC BY 4.0; attribution, checksum, and citations are in [`docs/DATASETS.md`](../docs/DATASETS.md).
