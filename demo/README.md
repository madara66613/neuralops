# Bundled demo artifact

`demo/artifact/` is the 660 KiB OpsForge Sim v1 multi-task GRU used by Docker Compose. It is bundled so a clean clone can start without credentials, object storage, or an unrecorded download.

The artifact is trained only on authored synthetic sequences. Its category and severity output demonstrates the product contract; it is not evidence of real-world incident classification. Public HDFS evidence remains separate in `reports/hdfs-v1-deduplicated-gru.json`.

The loader verifies the model, vocabulary, policy, and model-configuration SHA-256 values stored in `metadata.json` before serving. The complete package hashes are listed in `SHA256SUMS`.

Regenerate the synthetic artifact with:

```bash
neuralops prepare --config configs/opsforge-sim.yaml
neuralops train --config configs/opsforge-sim.yaml \
  --artifact artifacts/opsforge-sim-v1/multitask-gru --device cpu
```
