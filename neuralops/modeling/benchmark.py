"""Local inference latency and throughput measurements."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuralops.data.io import read_records
from neuralops.modeling.dataset import SequenceDataset, collate_sequences
from neuralops.modeling.training import load_gru_artifact


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_artifact(
    artifact_dir: Path,
    processed_dir: Path,
    *,
    iterations: int = 100,
    warmup: int = 10,
    batch_size: int = 32,
    requested_device: str = "auto",
) -> dict[str, Any]:
    if iterations < 1 or warmup < 0 or batch_size < 1:
        raise ValueError("iterations and batch_size must be positive; warmup cannot be negative")
    loaded = load_gru_artifact(artifact_dir, requested_device)
    records = read_records(processed_dir / "splits" / "test.jsonl")
    if not records:
        raise ValueError("Test split is empty")
    selected = [records[index % len(records)] for index in range(batch_size)]
    dataset = SequenceDataset(selected, loaded.vocabulary, loaded.max_sequence_length)
    batch = collate_sequences([dataset[index] for index in range(len(dataset))]).to(loaded.device)

    def infer() -> None:
        with torch.inference_mode():
            loaded.model(batch.tokens, batch.lengths)
        _synchronize(loaded.device)

    for _ in range(warmup):
        infer()
    latencies_ms: list[float] = []
    started_total = time.perf_counter_ns()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        infer()
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    elapsed_seconds = (time.perf_counter_ns() - started_total) / 1_000_000_000
    return {
        "schema_version": 1,
        "device": str(loaded.device),
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup_iterations": warmup,
        "latency_ms_per_batch": {
            "mean": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
        },
        "throughput_sequences_per_second": iterations * batch_size / elapsed_seconds,
        "measurement_context": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "synchronization": "backend synchronized after every batch",
        },
    }
