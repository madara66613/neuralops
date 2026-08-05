import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { Prediction } from "./types";

const model = {
  metadata: {
    profile: "opsforge-sim-v1-multitask",
    label_provenance: "synthetic",
    model_sha256: "abc123def4567890abc123def4567890",
    parameter_count: 157581,
    artifact_size_bytes: 634828,
  },
  policy: { threshold: 0.5, review_low: 0.25, review_high: 0.75 },
  label_mappings: { category: ["authentication_failure"], severity: ["high", "low"] },
  device: "cpu",
};

const anomalyPrediction: Prediction = {
  predicted_anomaly: true,
  anomaly_probability: 0.991,
  confidence: 0.991,
  confidence_kind: "raw_model_outcome_probability_not_calibrated",
  decision: "anomaly",
  manual_review: false,
  category: "authentication_failure",
  category_confidence: 0.98,
  severity: "high",
  severity_confidence: 0.78,
  input_event_count: 9,
  unknown_event_count: 0,
  unknown_event_rate: 0,
  truncated: false,
  profile: "opsforge-sim-v1-multitask",
  label_provenance: "synthetic",
};

function response(body: unknown, status = 200): Promise<Response> {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response);
}

describe("NeuralOps console", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/health")) return response({ status: "ok" });
        if (url.endsWith("/model")) return response({ model });
        if (url.endsWith("/predict/sensitivity")) {
          return response({
            prediction: anomalyPrediction,
            method: "leave_one_event_out_probability_sensitivity",
            interpretation: "Descriptive model sensitivity; not causal.",
            input_event_count: 9,
            evaluated_event_count: 9,
            evaluation_limited: false,
            evidence: [
              {
                event_index: 2,
                event: "E104",
                anomaly_probability_without_event: 0.72,
                anomaly_probability_delta: 0.271,
                absolute_delta: 0.271,
                effect: "supports_anomaly",
              },
            ],
          });
        }
        return response({}, 404);
      }),
    );
  });

  it("shows loaded artifact provenance and an honest empty state", async () => {
    render(<App />);
    expect(screen.getByText("No inference yet")).toBeInTheDocument();
    expect(await screen.findByText("Inference online")).toBeInTheDocument();
    expect(screen.getAllByText("opsforge-sim-v1-multitask").length).toBeGreaterThan(0);
    expect(screen.getByText("synthetic", { exact: true })).toBeInTheDocument();
  });

  it("loads a sample and renders the API decision and diagnostics", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Auth failure" }));
    await user.click(screen.getByRole("button", { name: "Analyze sequence" }));
    expect(await screen.findByText("Anomaly detected")).toBeInTheDocument();
    expect(screen.getByText("Authentication Failure")).toBeInTheDocument();
    expect(screen.getAllByText("99.1%")).toHaveLength(2);
    expect(screen.getByText("Synthetic", { exact: true })).toBeInTheDocument();
    expect(screen.getByText("Leave-one-out score change")).toBeInTheDocument();
    expect(screen.getByText("#3 E104")).toBeInTheDocument();
  });

  it("renders a structured validation error with request context", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/health")) return response({ status: "ok" });
      if (url.endsWith("/model")) return response({ model });
      return response(
        {
          request_id: "req-failed",
          error: { code: "VALIDATION_ERROR", message: "Request validation failed" },
        },
        422,
      );
    });
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Analyze sequence" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Request validation failed");
    expect(screen.getByRole("alert")).toHaveTextContent("VALIDATION_ERROR · req-failed");
  });

  it("renders the validation-selected manual-review state", async () => {
    const reviewPrediction: Prediction = {
      ...anomalyPrediction,
      predicted_anomaly: false,
      anomaly_probability: 0.55,
      confidence: 0.45,
      decision: "manual_review",
      manual_review: true,
      category: null,
      category_confidence: null,
      severity: null,
      severity_confidence: null,
    };
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/health")) return response({ status: "ok" });
      if (url.endsWith("/model")) return response({ model });
      return response({
        prediction: reviewPrediction,
        method: "leave_one_event_out_probability_sensitivity",
        interpretation: "Descriptive model sensitivity; not causal.",
        input_event_count: 6,
        evaluated_event_count: 6,
        evaluation_limited: false,
        evidence: [],
      });
    });
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Analyze sequence" }));
    expect(await screen.findByText("Manual review")).toBeInTheDocument();
    expect(screen.getByText(/uncertainty band/i)).toBeInTheDocument();
  });
});
