# HDFS v1 error analysis

This report analyzes the locked packed-GRU artifact on the untouched HDFS v1 deduplicated-sequence test split. The machine-readable source is [`hdfs-v1-error-analysis.json`](hdfs-v1-error-analysis.json); regenerate it with:

```bash
neuralops analyze-errors \
  --artifact artifacts/hdfs-v1-deduplicated/gru \
  --processed-dir data/processed/hdfs-v1 \
  --split test --device cpu \
  --output reports/hdfs-v1-error-analysis.json
```

## Outcome

Among 2,740 sequences (2,081 normal, 659 anomalous), the locked threshold produced 2,069 true negatives, 12 false positives, 4 false negatives, and 655 true positives. There were no unknown event IDs in any error. Thirty-five test cases fell inside the validation-selected manual-review band.

| Error type | Count | Probability median | Probability range | Length median | Length range | In review band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| False positive | 12 | 0.9833710193634033 | 0.8685352802276611–0.9974996447563171 | 32 | 28–222 | 7 |
| False negative | 4 | 0.3750489130616188 | 0.010469530709087849–0.8312728404998779 | 32.5 | 15–39 | 1 |

## Representative cases

- False positive `blk_1241292020494577039` scored `0.8685352802276611`, just above the `0.8597967624664307` anomaly threshold, and was correctly marked for manual review by the wider review band.
- False positive `blk_8378035312070777131` contains 222 events; only the first 128 were modeled, and the response disclosed truncation. This single long case raises false-positive mean length to `49.416666666666664` while the median remains 32.
- False negative `blk_-5006275964087510687` scored `0.8312728404998779` and fell in manual review rather than automatic normal handling.
- False negative `blk_2429961036831384938` scored `0.010469530709087849` on a 15-event sequence, showing that some upstream anomalies look strongly normal to the learned representation.

All 16 case IDs, fingerprints, ordered event templates, probabilities, and diagnostics are retained in the JSON report. No raw log message content is exposed.

## Pattern observations

Event `E3` occurs in all 12 false positives and in three of four false negatives; `E26`, `E5`, `E11`, `E9`, and `E22` occur in every error case. These are descriptive template frequencies, not causal evidence. Since several are also common HDFS workflow events, their presence alone cannot explain a prediction.

Seven of 12 false positives and one of four false negatives are intercepted by the review policy. The remaining three low-scoring false negatives are the more serious blind spot because threshold adjustment alone would substantially increase false positives.

## Leakage and shortcut audit

- normalized sequence fingerprints and block groups are disjoint across splits;
- vocabulary is fitted only on training records;
- identical fingerprints with conflicting upstream labels stay in the same split;
- threshold and review band come only from validation;
- zero unknown tokens in these errors means they are not explained by unseen event IDs;
- one false positive is truncated, but truncation is not a general explanation for the other 15 errors.

The current analysis does not claim category confusion for HDFS because HDFS has no category labels. Synthetic category confusion is reported separately and must not be treated as public evidence.

## Recommended next experiments

1. Evaluate on a time- or host-separated public split to stress distribution shift.
2. Compare learned pooling or a small Transformer only after defining the same locked protocol.
3. Calibrate probability on an additional calibration partition, not the test set.
4. Analyze prediction stability under parser-template changes and controlled unknown events.
5. Investigate the three automatically normal false negatives with sequence alignment and leave-one-event-out sensitivity, while avoiding causal claims.
6. Replace synthetic category/severity evidence with a legally usable public incident dataset if one with defensible labels becomes available.
