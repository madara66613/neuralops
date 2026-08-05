# Experiment comparison

All values below are copied from committed machine-readable evaluation artifacts. Model selection used validation data; each locked test report was generated once.

| Track | Model | Test samples | Binary F1 | PR-AUC | ROC-AUC | Category macro-F1 | Severity macro-F1 | Parameters | Artifact bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HDFS v1 deduplicated | TF-IDF + logistic regression | 2,740 | 0.9655688622754491 | 0.993398403251326 | 0.9980636279248842 | n/a | n/a | 286 | 9,296 |
| HDFS v1 deduplicated | packed bidirectional GRU | 2,740 | 0.9879336349924586 | 0.9989935365790357 | 0.9996809780520192 | n/a | n/a | 151,233 | 608,544 |
| OpsForge Sim v1 synthetic | multi-task packed bidirectional GRU | 806 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6856184110306733 | 157,581 | 634,828 |

The HDFS neural model improves F1 by `0.0223647727170095` over the baseline on this exact deduplicated profile. This does not imply universal neural-model superiority. The synthetic row evaluates a different population and label space and must not be compared as if it were an HDFS result.

The HDFS GRU benchmark measured CPU batch-1 median latency of `0.265417` ms and batch-32 throughput of `6582.435206503571` sequences/s on the recorded Apple M3 environment. Those timings are hardware- and software-specific, exclude network/UI overhead, and are documented in `hdfs-v1-deduplicated-gru.json`.
