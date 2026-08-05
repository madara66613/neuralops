import type { ApiErrorBody, ModelInfo, Prediction, SensitivityResult } from "./types";

const browserDefault = `${window.location.protocol}//${window.location.hostname}:8000`;
const API_URL = (import.meta.env.VITE_API_URL ?? browserDefault).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly requestId?: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const body = (await response.json()) as T & ApiErrorBody;
  if (!response.ok) {
    throw new ApiError(
      body.error?.message ?? "The NeuralOps service rejected the request.",
      response.status,
      body.error?.code ?? "API_ERROR",
      body.request_id,
    );
  }
  return body;
}

export async function fetchSystem(): Promise<{ health: string; model: ModelInfo }> {
  const [health, model] = await Promise.all([
    request<{ status: string }>("/health"),
    request<{ model: ModelInfo }>("/model"),
  ]);
  return { health: health.status, model: model.model };
}

export async function predict(events: string[]): Promise<Prediction> {
  const result = await request<{ prediction: Prediction }>("/predict", {
    method: "POST",
    body: JSON.stringify({ events }),
  });
  return result.prediction;
}

export async function predictWithSensitivity(events: string[]): Promise<SensitivityResult> {
  return request<SensitivityResult>("/predict/sensitivity", {
    method: "POST",
    body: JSON.stringify({ events }),
  });
}

export async function predictBatch(sequences: string[][]): Promise<Prediction[]> {
  const result = await request<{ predictions: Prediction[] }>("/predict/batch", {
    method: "POST",
    body: JSON.stringify({ sequences: sequences.map((events) => ({ events })) }),
  });
  return result.predictions;
}
