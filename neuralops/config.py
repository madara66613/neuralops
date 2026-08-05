"""Configuration loading with explicit path normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject malformed root values."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def nested(config: dict[str, Any], *keys: str) -> Any:
    """Read a required nested configuration value."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("Missing configuration key: " + ".".join(keys))
        value = value[key]
    return value
