# NeuralOps v1.0 model card

## Summary

NeuralOps is an experimental sequence classifier with two deliberately separate evidence tracks. The public HDFS v1 artifact predicts only binary block-level anomaly status. The bundled OpsForge Sim v1 artifact demonstrates binary, category, and severity outputs on authored synthetic operations scenarios.

It is not an autonomous incident-response system, root-cause engine, or production-proven detector. It must not execute remediation actions. Uncertain or high-impact decisions require human review.

## Model details

| Field | HDFS public artifact | Bundled demo artifact |
| --- | --- | --- |
| Architecture | embedding + packed bidirectional GRU + binary head | shared embedding + packed bidirectional GRU + binary/category/severity heads |
| Parameters | 151,233 | 157,581 |
| Maximum modeled length | 128 events | 128 events |
| Label provenance | public HDFS normal/anomaly label | synthetic OpsForge Sim v1 labels |
| Model SHA-256 | `fbc444909e91a1c8a33d885ee0be3b73367870bc8f7ab0817791070fd41ca1bc` | `7dd9b374de227bd161e6335d3698b1553c450bd21e7803a9450452d49db35972` |
| Serialized size | 608,544 bytes | 634,828 bytes |

Both models use train-only vocabularies, packed variable-length sequences, AdamW, class-weighted binary loss, gradient clipping, early stopping, and best-checkpoint restoration. The synthetic model masks category and severity loss for normal records.

## Intended use

- portfolio demonstration of an end-to-end PyTorch ML system;
- reproducible research on HDFS block-sequence anomaly detection;
- local API/UI evaluation of thresholding, unknown events, batching, provenance, and uncertainty behavior;
- educational comparison of a statistical baseline and a recurrent neural model.

## Out-of-scope use

- automatic remediation, access control, paging, or service shutdown;
- safety-critical, legal, employment, financial, or medical decisions;
- claims about causal root cause or event importance;
- raw unparsed logs from arbitrary applications;
- category or severity inference presented as public HDFS evidence;
- production monitoring without domain-specific validation, calibration, drift monitoring, and an operator runbook.

## Training and evaluation data

The public track uses LogPAI Loghub HDFS v1 under CC BY 4.0. Upstream binary labels apply to HDFS block IDs. NeuralOps reduces 575,061 block traces to 18,383 deterministic `(sequence fingerprint, label)` representatives, keeps conflicting identical fingerprints together, and assigns duplicate components to train/validation/test before fitting a 31-token train-only vocabulary. Final split sizes are 12,838 / 2,805 / 2,740.

The product track uses 5,615 deduplicated records from 6,000 generated OpsForge Sim v1 examples. Three variations share an incident group and cannot cross splits. Its nine category labels and three severity labels are synthetic.

See [dataset evidence](../docs/DATASETS.md), [synthetic methodology](../docs/SYNTHETIC_DATA.md), and machine-readable reports in this directory.

## Metrics

On the untouched 2,740-sequence HDFS deduplicated test split, the packed GRU produced precision `0.9820089955022488`, recall `0.9939301972685888`, F1 `0.9879336349924586`, PR-AUC `0.9989935365790357`, and ROC-AUC `0.9996809780520192`, with TN/FP/FN/TP = 2,069/12/4/655. The TF-IDF logistic baseline reached F1 `0.9655688622754491` and PR-AUC `0.993398403251326` on the same profile.

OpsForge Sim v1 test results are binary F1 `1.0`, category macro-F1 `1.0`, and severity macro-F1 `0.6856184110306733`. The perfect binary/category values indicate separable authored patterns, not real-world generalization.

## Decision policy and confidence

The HDFS anomaly threshold `0.8597967624664307` and review band `[0.7297967624664307, 0.9897967624664307]` were selected on validation only. The demo artifact uses its separately stored synthetic policy. `confidence` is the raw model probability of the selected binary outcome and is explicitly not calibrated.

Leave-one-event-out evidence reports how the anomaly score changes when one event is removed. It is a local sensitivity check, not a causal explanation, attention map, or root-cause claim.

## Limitations and representativeness

- HDFS is an older distributed-storage workload and does not represent modern web, ERP, cloud, or security telemetry.
- Deduplicated-sequence evaluation is intentionally not the conventional all-block HDFS benchmark.
- Event-template parsing quality, unseen templates, label noise, concept drift, and changed sequence grouping can dominate field performance.
- HDFS provides no credible incident category or severity labels.
- Synthetic categories are closed-set and easier to separate than real incidents; synthetic severity is non-ordinal three-class classification.
- Probabilities are uncalibrated. The review band reduces automatic coverage but cannot guarantee safe decisions.
- Sequences beyond 128 events are truncated for model input and disclosed in responses.
- The test set was used once for the locked artifact report; further model selection requires a new evaluation protocol.

## Bias, security, and operational considerations

Dataset frequency and parser-template bias may make common operational patterns appear more or less anomalous. Unknown systems, rare normal maintenance, and new incident types can fail unpredictably.

API input is untrusted and size-limited. The service never executes submitted content and does not log payloads. PyTorch files can be unsafe if loaded from untrusted sources; NeuralOps accepts operator-selected artifact paths and loads a state dictionary with `weights_only=True`, then verifies recorded SHA-256 values. Only reviewed artifacts should be mounted.

Before any operational trial, define costs for false positives/negatives, validate on time-separated domain data, calibrate probabilities, test parser drift, add authentication/rate limiting at the deployment boundary, and establish human escalation and rollback procedures.

## Retraining triggers

Retraining and revalidation are required after parser or label changes, material unknown-token growth, sequence-length drift, new application versions, changed incident taxonomy, or monitored precision/recall falling below an operator-approved target. Do not tune against the existing locked test split.
