"""Deterministic runtime device selection."""

from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["cuda", "mps", "cpu"]


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve CUDA -> Apple MPS -> CPU, or validate an explicit device."""
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if normalized == "mps" and not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        raise RuntimeError("Apple MPS was requested but is not available")
    if normalized not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(normalized)

