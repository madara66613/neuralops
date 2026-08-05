# Architecture and scientific design

## System boundary

```mermaid
flowchart LR
    raw["Official HDFS archive or OpsForge simulator"] --> prepare["Validate, normalize, group, deduplicate"]
    prepare --> split["Group-disjoint train / validation / test"]
    split --> vocab["Train-only vocabulary and transforms"]
    vocab --> baseline["TF-IDF + logistic regression"]
    vocab --> gru["Embedding + packed bidirectional GRU"]
    baseline --> eval["Validation policy + untouched test evaluation"]
    gru --> eval
    eval --> artifact["Versioned model, policy, metrics, provenance"]
    artifact --> api["FastAPI predictor"]
    api --> ui["React operator console"]
```

## Prediction tasks

### HDFS public track

- Input: ordered event IDs for one HDFS block.
- Output: anomaly probability, binary decision, confidence band, and manual-review flag.
- Target: upstream binary anomaly label only.
- Explicitly absent: category and severity.

### OpsForge synthetic track

- Shared encoder with binary, category, and severity heads.
- Category and severity losses apply only to anomalous sequences.
- API fields are nullable and model metadata declares `label_provenance=synthetic`.
- Public and synthetic metrics live in separate report sections and artifact directories.

## Data preparation and experiment flow

```mermaid
flowchart TD
    hdfs["HDFS v1 archive · MD5 verified"] --> parse["Parse block IDs and ordered event templates"]
    sim["Seeded OpsForge Sim v1 generator"] --> parseSim["Opaque event IDs · grouped incident variants"]
    parse --> normalize["Validate · normalize · fingerprint"]
    parseSim --> normalize
    normalize --> components["Connect duplicate and conflicting fingerprints"]
    components --> splits["Stable-hash component assignment"]
    splits --> checks["Assert group and fingerprint disjointness"]
    checks --> train["Train split"]
    checks --> validation["Validation split"]
    checks --> test["Locked test split"]
    train --> fit["Fit vocabulary, transforms, class weights"]
    fit --> baseline["TF-IDF logistic baseline"]
    fit --> gru["Packed bidirectional GRU"]
    validation --> selection["Early stopping · threshold · review band"]
    baseline --> selection
    gru --> selection
    selection --> locked["Restore and hash locked artifact"]
    locked --> test
    test --> report["One final report + error analysis"]
```

## Inference and sensitivity flow

```mermaid
sequenceDiagram
    participant Operator
    participant Console as React console
    participant Proxy as nginx /api proxy
    participant API as FastAPI
    participant Predictor as Verified GRU predictor
    Operator->>Console: ordered event IDs
    Console->>Proxy: POST /api/predict/sensitivity
    Proxy->>API: validated JSON + request ID
    API->>Predictor: full sequence
    Predictor->>Predictor: encode + packed GRU + locked policy
    Predictor->>Predictor: batch leave-one-event-out ablations
    Predictor-->>API: prediction + score deltas + provenance
    API-->>Console: strict response + inference time
    Console-->>Operator: decision, uncertainty, diagnostics, non-causal sensitivity
```

The Compose stack mounts a reviewed artifact read-only. The API verifies artifact hashes before `/ready` succeeds. Neither input nor model output triggers shell commands, SQL, or remediation.

## Leakage controls

1. Normalize and fingerprint complete sequences before splitting.
2. Deduplicate identical `(fingerprint, label)` records, retaining a deterministic representative.
3. Assign clusters deterministically using a seeded stable hash of the cluster ID.
4. Assert block/session groups and fingerprints are disjoint across all splits.
5. Fit vocabulary, TF-IDF, class weights, and sequence-length limits on train only.
6. Select early stopping, decision threshold, and uncertainty band using validation only.
7. Evaluate the locked artifact once against test; record the manifest hash.

If identical normalized sequences carry conflicting upstream labels, NeuralOps keeps
one representative for each label, places both in the same component/split, and
reports the conflict count. It does not hide ambiguity or let identical features
leak across boundaries. This deduplicated-sequence profile is reported explicitly;
it is not presented as the conventional all-block HDFS benchmark.

The default target ratios are 70/15/15. Realized ratios and class balance are reported rather than assumed because group constraints take priority over exact percentages.

## Model design

The primary neural model uses a train-only event vocabulary with reserved padding and unknown tokens, a learned embedding, packed variable-length sequences, a bidirectional GRU, dropout, and a small classification head. Training includes seeded loaders, class-weighted loss, AdamW, gradient clipping, early stopping, and best-checkpoint restoration.

Runtime device resolution is CUDA, then Apple MPS, then CPU. An explicit unavailable device fails loudly.

## Decision policy

The validation set selects a binary threshold by the configured objective. A second validation-only search chooses a low/high probability band for manual review while preserving minimum automatic-decision coverage. The artifact stores both values and their selection criterion. API consumers never silently substitute `0.5`.

## Artifact contract

Every publishable artifact contains:

- model state or serialized baseline;
- architecture config and package version;
- vocabulary and label mappings;
- decision and uncertainty thresholds;
- dataset DOI/source, split manifest hash, seed, and label provenance;
- validation and test metrics in separate objects;
- parameter count, serialized size, and benchmark context;
- SHA-256 hashes for integrity checks.

## Serving constraints

The API validates event count, event length, batch size, and body size. It returns request IDs, structured errors, software version, measured inference duration, model provenance, and nullable multi-task fields. `/health` proves process liveness; `/ready` only succeeds when a verified model is loaded. Leave-one-event-out sensitivity evaluates at most 64 positions and returns the eight largest absolute score changes; it is descriptive rather than causal.
