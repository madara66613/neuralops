export type Prediction = {
  predicted_anomaly: boolean;
  anomaly_probability: number;
  confidence: number;
  confidence_kind: string;
  decision: "normal" | "anomaly" | "manual_review";
  manual_review: boolean;
  category: string | null;
  category_confidence: number | null;
  severity: string | null;
  severity_confidence: number | null;
  input_event_count: number;
  unknown_event_count: number;
  unknown_event_rate: number;
  truncated: boolean;
  profile: string;
  label_provenance: string;
};

export type ModelInfo = {
  metadata: {
    profile: string;
    label_provenance: string;
    model_sha256: string;
    parameter_count: number;
    artifact_size_bytes: number;
    best_epoch?: number;
  };
  policy: {
    threshold: number;
    review_low: number;
    review_high: number;
  };
  label_mappings: Record<string, string[]>;
  device: string;
};

export type SensitivityEvidence = {
  event_index: number;
  event: string;
  anomaly_probability_without_event: number;
  anomaly_probability_delta: number;
  absolute_delta: number;
  effect: "supports_anomaly" | "suppresses_anomaly" | "neutral";
};

export type SensitivityResult = {
  prediction: Prediction;
  method: string;
  interpretation: string;
  input_event_count: number;
  evaluated_event_count: number;
  evaluation_limited: boolean;
  evidence: SensitivityEvidence[];
};

export type ApiErrorBody = {
  request_id?: string;
  error?: {
    code?: string;
    message?: string;
  };
};
