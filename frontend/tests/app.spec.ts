import { expect, test } from "@playwright/test";

const model = {
  metadata: {
    profile: "opsforge-sim-v1-multitask",
    label_provenance: "synthetic",
    model_sha256: "abc123def4567890abc123def4567890",
    parameter_count: 157581,
    artifact_size_bytes: 634828,
  },
  policy: { threshold: 0.5, review_low: 0.25, review_high: 0.75 },
  label_mappings: {},
  device: "cpu",
};

test.beforeEach(async ({ page }) => {
  page.on("console", (message) => {
    if (message.type() === "error") throw new Error(`Browser console error: ${message.text()}`);
  });
  await page.route(/http:\/\/(localhost|127\.0\.0\.1):8000\/.*/, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/health") return route.fulfill({ json: { status: "ok" } });
    if (path === "/model") return route.fulfill({ json: { model } });
    if (path === "/predict/sensitivity") {
      return route.fulfill({
        json: {
          prediction: {
            predicted_anomaly: true,
            anomaly_probability: 0.992,
            confidence: 0.992,
            confidence_kind: "raw_model_outcome_probability_not_calibrated",
            decision: "anomaly",
            manual_review: false,
            category: "authentication_failure",
            category_confidence: 0.981,
            severity: "high",
            severity_confidence: 0.774,
            input_event_count: 9,
            unknown_event_count: 0,
            unknown_event_rate: 0,
            truncated: false,
            profile: "opsforge-sim-v1-multitask",
            label_provenance: "synthetic",
          },
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
              anomaly_probability_delta: 0.272,
              absolute_delta: 0.272,
              effect: "supports_anomaly",
            },
          ],
        },
      });
    }
    if (path === "/predict/batch") {
      const body = route.request().postDataJSON() as { sequences: Array<{ events: string[] }> };
      return route.fulfill({
        json: {
          predictions: body.sequences.map((sequence, index) => ({
            predicted_anomaly: index > 0,
            anomaly_probability: index > 0 ? 0.96 : 0.04,
            confidence: 0.96,
            confidence_kind: "raw_model_outcome_probability_not_calibrated",
            decision: index > 0 ? "anomaly" : "normal",
            manual_review: false,
            category: index > 0 ? "resource_exhaustion" : null,
            category_confidence: index > 0 ? 0.93 : null,
            severity: index > 0 ? "medium" : null,
            severity_confidence: index > 0 ? 0.71 : null,
            input_event_count: sequence.events.length,
            unknown_event_count: 0,
            unknown_event_rate: 0,
            truncated: false,
            profile: "opsforge-sim-v1-multitask",
            label_provenance: "synthetic",
          })),
        },
      });
    }
    return route.fulfill({ status: 404, json: {} });
  });
});

test("analyzes a sample and exposes provenance", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Inference online")).toBeVisible();
  await page.getByRole("button", { name: "Auth failure" }).click();
  await page.getByRole("button", { name: "Analyze sequence" }).click();
  await expect(page.getByText("Anomaly detected")).toBeVisible();
  await expect(page.getByText("Authentication Failure")).toBeVisible();
  await expect(page.getByText("Leave-one-out score change")).toBeVisible();
  await expect(page.getByText("Public binary ≠ synthetic multi-task")).toBeVisible();
});

test("scores a batch and remains keyboard operable", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "Batch" }).click();
  await expect(page.getByText("3 sequences · 24 events")).toBeVisible();
  await page.getByRole("button", { name: "Analyze sequence" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText("3 sequences scored")).toBeVisible();
  await expect(page.getByText("2 anomalies")).toBeVisible();
});
