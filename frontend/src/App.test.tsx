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
        if (url.endsWith("/predict")) return response({ prediction: anomalyPrediction });
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
  });

  it("renders a safe API error with request context", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/health")) return response({ status: "ok" });
      if (url.endsWith("/model")) return response({ model });
      return response(
        {
          request_id: "req-failed",
          error: { code: "MODEL_NOT_READY", message: "Model artifact is not ready" },
        },
        503,
      );
    });
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Analyze sequence" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Model artifact is not ready");
    expect(screen.getByRole("alert")).toHaveTextContent("MODEL_NOT_READY · req-failed");
  });
});
