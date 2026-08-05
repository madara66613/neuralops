import { useEffect, useMemo, useState } from "react";

import { ApiError, fetchSystem, predictBatch, predictWithSensitivity } from "./api";
import type { ModelInfo, Prediction, SensitivityEvidence } from "./types";

type Mode = "single" | "batch";

const samples = [
  {
    label: "Healthy flow",
    tone: "normal",
    value: "E001 E012 E024 E035 P04 E099",
  },
  {
    label: "Auth failure",
    tone: "anomaly",
    value: "E001 E012 E104 E207 E104 E431 P07 E087 E099",
  },
  {
    label: "Resource pressure",
    tone: "anomaly",
    value: "E001 E018 E201 E389 E389 E541 P12 E087 E099",
  },
  {
    label: "Unknown events",
    tone: "review",
    value: "E001 E012 X_EXTERNAL_42 X_EXTERNAL_87 E099",
  },
] as const;

function tokens(value: string): string[] {
  return value
    .trim()
    .split(/[\s,]+/)
    .map((event) => event.trim())
    .filter(Boolean);
}

function batchSequences(value: string): string[][] {
  return value
    .split("\n")
    .map(tokens)
    .filter((sequence) => sequence.length > 0);
}

function formatLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function Mark({ kind }: { kind: "spark" | "pulse" | "shield" | "stack" }) {
  const paths = {
    spark: "M12 2 8.8 9H3l5 3.6L6.5 22 15 11h6l-5-3.5L17.5 2 12 7V2Z",
    pulse: "M3 12h4l2-6 4 12 2-6h6",
    shield: "M12 3 5 6v5c0 4.6 2.9 8.1 7 10 4.1-1.9 7-5.4 7-10V6l-7-3Z",
    stack: "m4 7 8-4 8 4-8 4-8-4Zm0 5 8 4 8-4M4 17l8 4 8-4",
  };
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d={paths[kind]} />
    </svg>
  );
}

function ProbabilityGauge({ value, anomaly }: { value: number; anomaly: boolean }) {
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  return (
    <div className={`probability-gauge ${anomaly ? "is-anomaly" : "is-normal"}`}>
      <svg viewBox="0 0 120 120" role="img" aria-label={`Anomaly probability ${percent(value)}`}>
        <circle className="gauge-track" cx="60" cy="60" r={radius} />
        <circle
          className="gauge-value"
          cx="60"
          cy="60"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value)}
        />
      </svg>
      <div className="gauge-copy">
        <strong>{percent(value)}</strong>
        <span>anomaly</span>
      </div>
    </div>
  );
}

function DecisionBadge({ prediction }: { prediction: Prediction }) {
  const label = prediction.manual_review
    ? "Manual review"
    : prediction.predicted_anomaly
      ? "Anomaly detected"
      : "Normal sequence";
  return <span className={`decision-badge ${prediction.decision}`}>{label}</span>;
}

function ResultDetails({ prediction }: { prediction: Prediction }) {
  return (
    <div className="result-details">
      <div className="detail-cell">
        <span>Outcome confidence</span>
        <strong>{percent(prediction.confidence)}</strong>
        <small>Raw, not calibrated</small>
      </div>
      <div className="detail-cell">
        <span>Incident category</span>
        <strong>{prediction.category ? formatLabel(prediction.category) : "Not assigned"}</strong>
        <small>
          {prediction.category_confidence === null
            ? "Nullable by artifact contract"
            : `${percent(prediction.category_confidence)} model probability`}
        </small>
      </div>
      <div className="detail-cell">
        <span>Severity</span>
        <strong>{prediction.severity ? formatLabel(prediction.severity) : "Not assigned"}</strong>
        <small>
          {prediction.severity_confidence === null
            ? "Only emitted for synthetic anomalies"
            : `${percent(prediction.severity_confidence)} model probability`}
        </small>
      </div>
    </div>
  );
}

function SensitivityPanel({ evidence }: { evidence: SensitivityEvidence[] }) {
  if (evidence.length === 0) return null;
  const maximum = Math.max(...evidence.map((item) => item.absolute_delta), Number.EPSILON);
  return (
    <section className="sensitivity-panel" aria-label="Leave-one-event-out sensitivity">
      <div className="sensitivity-heading">
        <div>
          <span>Event sensitivity</span>
          <strong>Leave-one-out score change</strong>
        </div>
        <small>Descriptive, not causal</small>
      </div>
      <div className="sensitivity-list">
        {evidence.slice(0, 5).map((item) => (
          <div className="sensitivity-row" key={`${item.event_index}-${item.event}`}>
            <code>#{item.event_index + 1} {item.event}</code>
            <div className="sensitivity-track" aria-hidden="true">
              <i
                className={item.effect}
                style={{ width: `${Math.max(4, (item.absolute_delta / maximum) * 100)}%` }}
              />
            </div>
            <span className={item.effect}>
              {item.anomaly_probability_delta >= 0 ? "+" : ""}
              {(item.anomaly_probability_delta * 100).toFixed(2)} pp
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function ResultPanel({
  results,
  loading,
  evidence,
}: {
  results: Prediction[];
  loading: boolean;
  evidence: SensitivityEvidence[];
}) {
  if (loading) {
    return (
      <div className="panel-state" aria-live="polite">
        <div className="analysis-orbit" />
        <strong>Analyzing event topology</strong>
        <span>Encoding sequence and applying the locked decision policy…</span>
      </div>
    );
  }
  if (results.length === 0) {
    return (
      <div className="panel-state empty-state">
        <div className="empty-icon">
          <Mark kind="pulse" />
        </div>
        <strong>No inference yet</strong>
        <span>Choose a sample or paste an ordered event sequence to inspect its behavior.</span>
      </div>
    );
  }
  if (results.length > 1) {
    return (
      <div className="batch-results" aria-live="polite">
        <div className="result-heading">
          <div>
            <span className="eyebrow">Batch complete</span>
            <h2>{results.length} sequences scored</h2>
          </div>
          <span className="batch-summary">
            {results.filter((item) => item.predicted_anomaly).length} anomalies
          </span>
        </div>
        <div className="batch-list">
          {results.map((prediction, index) => (
            <article
              className="batch-row"
              key={`${prediction.profile}-${prediction.anomaly_probability}-${index}`}
            >
              <span className="sequence-index">{String(index + 1).padStart(2, "0")}</span>
              <div>
                <DecisionBadge prediction={prediction} />
                <strong>{prediction.category ? formatLabel(prediction.category) : "Binary only"}</strong>
                <small>{prediction.input_event_count} events · {prediction.unknown_event_count} unknown</small>
              </div>
              <span className="batch-probability">{percent(prediction.anomaly_probability)}</span>
            </article>
          ))}
        </div>
      </div>
    );
  }

  const prediction = results[0];
  return (
    <div className="single-result" aria-live="polite">
      <div className="result-heading">
        <div>
          <span className="eyebrow">Decision</span>
          <h2>Sequence assessment</h2>
        </div>
        <DecisionBadge prediction={prediction} />
      </div>
      <div className="result-hero">
        <ProbabilityGauge value={prediction.anomaly_probability} anomaly={prediction.predicted_anomaly} />
        <div className="result-narrative">
          <span>Policy outcome</span>
          <strong>{formatLabel(prediction.decision)}</strong>
          <p>
            {prediction.manual_review
              ? "The probability falls inside the validation-selected uncertainty band. Route this sequence to a human reviewer."
              : prediction.predicted_anomaly
                ? "The sequence exceeds the validation-selected anomaly threshold."
                : "The sequence remains below the validation-selected anomaly threshold."}
          </p>
        </div>
      </div>
      <ResultDetails prediction={prediction} />
      <div className="diagnostic-strip">
        <span><b>{prediction.input_event_count}</b> input events</span>
        <span><b>{prediction.unknown_event_count}</b> unknown</span>
        <span><b>{prediction.truncated ? "Yes" : "No"}</b> truncated</span>
        <span><b>{formatLabel(prediction.label_provenance)}</b> labels</span>
      </div>
      <SensitivityPanel evidence={evidence} />
    </div>
  );
}

function Provenance({ model }: { model: ModelInfo | null }) {
  if (!model) {
    return (
      <div className="provenance skeleton-lines" role="status" aria-label="Model metadata unavailable">
        <span /> <span /> <span />
      </div>
    );
  }
  return (
    <div className="provenance">
      <div className="provenance-title">
        <Mark kind="shield" />
        <div>
          <span>Verified artifact</span>
          <strong>{model.metadata.profile}</strong>
        </div>
      </div>
      <dl>
        <div><dt>Provenance</dt><dd>{model.metadata.label_provenance}</dd></div>
        <div><dt>Runtime</dt><dd>{model.device}</dd></div>
        <div><dt>Threshold</dt><dd>{model.policy.threshold.toFixed(3)}</dd></div>
        <div><dt>Parameters</dt><dd>{model.metadata.parameter_count.toLocaleString()}</dd></div>
      </dl>
      <code title={model.metadata.model_sha256}>{model.metadata.model_sha256.slice(0, 16)}…</code>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("single");
  const [input, setInput] = useState<string>(samples[0].value);
  const [results, setResults] = useState<Prediction[]>([]);
  const [evidence, setEvidence] = useState<SensitivityEvidence[]>([]);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ message: string; meta?: string } | null>(null);

  useEffect(() => {
    let active = true;
    fetchSystem()
      .then(({ model: modelInfo }) => {
        if (active) {
          setModel(modelInfo);
          setStatus("online");
        }
      })
      .catch(() => active && setStatus("offline"));
    return () => {
      active = false;
    };
  }, []);

  const parsed = useMemo(
    () => (mode === "single" ? [tokens(input)].filter((item) => item.length) : batchSequences(input)),
    [input, mode],
  );
  const eventCount = parsed.reduce((total, sequence) => total + sequence.length, 0);

  function changeMode(nextMode: Mode) {
    setMode(nextMode);
    setResults([]);
    setEvidence([]);
    setError(null);
    setInput(
      nextMode === "single"
        ? samples[0].value
        : `${samples[0].value}\n${samples[1].value}\n${samples[2].value}`,
    );
  }

  async function analyze() {
    if (parsed.length === 0 || loading) return;
    setLoading(true);
    setError(null);
    setEvidence([]);
    try {
      if (mode === "single") {
        const result = await predictWithSensitivity(parsed[0]);
        setResults([result.prediction]);
        setEvidence(result.evidence);
      } else {
        setResults(await predictBatch(parsed));
      }
    } catch (caught) {
      const apiError = caught instanceof ApiError ? caught : null;
      setResults([]);
      setError({
        message: apiError?.message ?? "Could not reach the NeuralOps inference service.",
        meta: apiError
          ? `${apiError.code}${apiError.requestId ? ` · ${apiError.requestId}` : ""}`
          : "Check that the API is running on port 8000.",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="NeuralOps home">
          <span className="brand-mark"><Mark kind="spark" /></span>
          <span><strong>Neural</strong>Ops</span>
        </a>
        <div className="header-context">
          <span className={`service-status ${status}`}>
            <i /> {status === "online" ? "Inference online" : status === "offline" ? "API offline" : "Checking API"}
          </span>
          <span className="profile-chip">{model?.metadata.profile ?? "No artifact metadata"}</span>
        </div>
      </header>

      <main id="top">
        <section className="intro">
          <div>
            <span className="eyebrow">Sequence intelligence console</span>
            <h1>Turn ordered log events into an auditable incident decision.</h1>
            <p>
              Inspect anomaly probability, uncertainty, incident context, and model provenance — without hiding synthetic labels or unknown input.
            </p>
          </div>
          <Provenance model={model} />
        </section>

        <section className="workspace">
          <article className="input-panel surface">
            <div className="panel-header">
              <div>
                <span className="eyebrow">Input</span>
                <h2>Event sequence</h2>
              </div>
              <div className="mode-switch" role="tablist" aria-label="Prediction mode">
                <button type="button" role="tab" aria-selected={mode === "single"} onClick={() => changeMode("single")}>Single</button>
                <button type="button" role="tab" aria-selected={mode === "batch"} onClick={() => changeMode("batch")}>Batch</button>
              </div>
            </div>

            {mode === "single" && (
              <fieldset className="samples">
                <legend className="sr-only">Example sequences</legend>
                {samples.map((sample) => (
                  <button
                    type="button"
                    className={`sample-chip ${sample.tone}`}
                    key={sample.label}
                    onClick={() => {
                      setInput(sample.value);
                      setResults([]);
                      setEvidence([]);
                      setError(null);
                    }}
                  >
                    <i /> {sample.label}
                  </button>
                ))}
              </fieldset>
            )}

            <label className="editor-label" htmlFor="sequence-input">
              {mode === "single" ? "Ordered event IDs" : "One sequence per line"}
              <span>{mode === "single" ? `${eventCount} events` : `${parsed.length} sequences · ${eventCount} events`}</span>
            </label>
            <div className="editor-wrap">
              <div className="line-numbers" aria-hidden="true">
                {Array.from(
                  { length: Math.max(5, input.split("\n").length) },
                  (_, index) => index + 1,
                ).map((lineNumber) => <span key={`line-${lineNumber}`}>{lineNumber}</span>)}
              </div>
              <textarea
                id="sequence-input"
                value={input}
                onChange={(event) => {
                  setInput(event.target.value);
                  setResults([]);
                  setEvidence([]);
                  setError(null);
                }}
                spellCheck={false}
                aria-describedby="editor-help"
              />
            </div>
            <div className="editor-footer" id="editor-help">
              <span>Whitespace or comma separated · Order is preserved</span>
              <span>≤ 512 events</span>
            </div>

            {error && (
              <div className="error-banner" role="alert">
                <strong>{error.message}</strong>
                <span>{error.meta}</span>
              </div>
            )}

            <button type="button" className="analyze-button" disabled={parsed.length === 0 || loading} onClick={analyze}>
              {loading ? <><i className="button-spinner" /> Analyzing…</> : <><Mark kind="pulse" /> Analyze sequence</>}
            </button>
          </article>

          <article className="result-panel surface">
            <ResultPanel results={results} loading={loading} evidence={evidence} />
          </article>
        </section>

        <section className="evidence-strip">
          <div><Mark kind="shield" /><span><strong>Integrity checked</strong>SHA-256 verified before load</span></div>
          <div><Mark kind="pulse" /><span><strong>Validation policy</strong>Threshold and review band are locked</span></div>
          <div><Mark kind="stack" /><span><strong>Evidence separated</strong>Public binary ≠ synthetic multi-task</span></div>
        </section>
      </main>

      <footer>
        <span>NeuralOps · portfolio research system</span>
        <span>Decisions support operators; they do not replace incident review.</span>
      </footer>
    </div>
  );
}
