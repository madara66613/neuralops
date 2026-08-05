# OpsForge Sim v1 methodology

OpsForge Sim v1 is a NeuralOps-owned **synthetic behavior profile**, not a public benchmark and not a sample of production incidents. It exists to exercise category/severity product behavior that HDFS v1 cannot support.

## Label space

An anomalous incident belongs to one of nine generated categories:

1. authentication failure;
2. cache degradation;
3. database contention;
4. deployment regression;
5. inventory sync failure;
6. message queue backlog;
7. payment gateway failure;
8. resource exhaustion;
9. service dependency timeout.

Severity is generated as low, medium, or high. Normal sequences have `category=null` and `severity=null`; auxiliary losses and metrics therefore apply only to anomalous records.

## Generation controls

- Three record variations share one synthetic `incident_id` and cannot cross splits.
- Event tokens are opaque codes such as `E104` and `P07`; category/severity words are never embedded in model input.
- Patterns include randomized noise events, parameter tokens, and severity-dependent repetitions.
- Exact duplicates are removed before splitting.
- A seed controls generation and duplicate-aware stable-hash assignment.
- Vocabulary and label mappings are learned from the training split only.

## Interpretation

Binary anomaly status and incident category are deliberately separable in this generator. Perfect scores only demonstrate that the pipeline learns its authored rules; they do not estimate real-world performance. Severity shares noisier repetition cues and is intentionally less separable.

The artifact is selected by a validation composite: the mean of binary PR-AUC, category macro-F1, and severity macro-F1. This prevents a trivial binary head from stopping training before auxiliary heads learn. The public HDFS track continues to use validation PR-AUC alone.

## Known limitations

- Authored scenarios cover a small and closed event vocabulary.
- No concept drift, parser failure, operator annotation noise, or unknown incident class is simulated.
- Category patterns are easier to separate than real incident causes.
- Severity is an ordinal concept but v1 treats it as three-class classification.
- Results must never be merged into or compared numerically with the public HDFS report as though both represented the same population.
