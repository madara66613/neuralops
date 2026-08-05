# Operator console

The NeuralOps console is a React and TypeScript client for the shared FastAPI predictor. It presents model output as decision support, keeps raw confidence distinct from calibrated probability, and shows synthetic label provenance beside every relevant result.

## Run locally

Start the API from the repository root:

```bash
neuralops serve \
  --artifact artifacts/opsforge-sim-v1/multitask-gru \
  --host 127.0.0.1 --port 8000 --device cpu
```

Start the console separately:

```bash
cd frontend
npm ci
npm run dev
```

The client derives the API host from the browser host and uses port 8000. Set `VITE_API_URL` to override that origin.

## Behaviors

- Single mode accepts ordered event IDs separated by whitespace or commas.
- Batch mode accepts one sequence per line and calls the batch endpoint once.
- Four samples expose normal, authentication, resource-pressure, and unknown-token behavior.
- Empty, loading, API error, and offline states are explicit and retain actionable copy.
- Prediction cards disclose decision, anomaly probability, raw outcome confidence, nullable category and severity, unknown events, truncation, profile, and label provenance.
- Single-sequence results rank leave-one-event-out probability changes and explicitly label them descriptive rather than causal.
- Artifact identity, SHA-256 prefix, parameter count, runtime, and locked threshold remain visible above inference controls.

## Accessibility and responsive QA

Tabs, samples, the editor, and submission controls use semantic elements and visible focus states. Dynamic results use polite live regions; the probability graphic has a textual accessible label. The layout was manually inspected at 1440×1000 and 390×844 with no clipping or horizontal overflow.

Automated coverage includes Biome lint, TypeScript checking, Vitest component tests, and Playwright Chromium tests for the primary sample flow, sensitivity evidence, provenance disclosure, batch inference, and keyboard submission:

```bash
cd frontend
npm test
npm run lint
npm run typecheck
npm run build
npm run test:e2e
```

The checked-in screenshots under `output/playwright/` were captured against the real synthetic model artifact and FastAPI server, not mocked predictions.
