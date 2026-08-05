"""Canonical sequence record schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """One ordered log-event sequence and its labels."""

    session_id: str
    group_id: str
    events: tuple[str, ...]
    anomaly: int
    source: str
    category: str | None = None
    severity: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.events, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = list(self.events)
        data["fingerprint"] = self.fingerprint
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SequenceRecord:
        events = raw.get("events")
        if not isinstance(events, list) or not all(isinstance(item, str) for item in events):
            raise ValueError("events must be a list of strings")
        return cls(
            session_id=str(raw["session_id"]),
            group_id=str(raw["group_id"]),
            events=tuple(events),
            anomaly=int(raw["anomaly"]),
            source=str(raw["source"]),
            category=str(raw["category"]) if raw.get("category") is not None else None,
            severity=str(raw["severity"]) if raw.get("severity") is not None else None,
        )


def validate_record(record: SequenceRecord) -> None:
    if not record.session_id or not record.group_id:
        raise ValueError("session_id and group_id must be non-empty")
    if not record.events or any(not event for event in record.events):
        raise ValueError(f"Sequence {record.session_id} contains no usable events")
    if record.anomaly not in {0, 1}:
        raise ValueError(f"Sequence {record.session_id} has invalid anomaly label")
    if record.anomaly == 0 and (record.category is not None or record.severity is not None):
        raise ValueError("Normal sequences cannot carry incident category or severity labels")
