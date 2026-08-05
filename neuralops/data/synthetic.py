"""OpsForge-inspired, explicitly synthetic incident sequence generator."""

from __future__ import annotations

import random

from neuralops.data.records import SequenceRecord, validate_record

INCIDENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "authentication_failure": ("E104", "E207", "E104", "E431"),
    "database_contention": ("E118", "E322", "E322", "E509"),
    "message_queue_backlog": ("E141", "E366", "E366", "E472"),
    "cache_degradation": ("E153", "E284", "E401", "E284"),
    "payment_gateway_failure": ("E165", "E298", "E498", "E165"),
    "inventory_sync_failure": ("E177", "E345", "E177", "E523"),
    "service_dependency_timeout": ("E189", "E255", "E255", "E444"),
    "resource_exhaustion": ("E201", "E389", "E389", "E541"),
    "deployment_regression": ("E213", "E310", "E477", "E310"),
}
SEVERITIES = ("low", "medium", "high")
NORMAL_CORES = (
    ("E001", "E012", "E024", "E035", "E099"),
    ("E001", "E018", "E029", "E043", "E099"),
    ("E001", "E015", "E027", "E038", "E099"),
    ("E001", "E021", "E033", "E046", "E099"),
)
NOISE_EVENTS = ("E061", "E063", "E067", "E071", "E073", "E079")


def _mutate_sequence(
    core: tuple[str, ...],
    *,
    rng: random.Random,
    severity: str | None,
) -> tuple[str, ...]:
    events = list(rng.choice(NORMAL_CORES)[:2])
    if rng.random() < 0.65:
        events.append(rng.choice(NOISE_EVENTS))
    repeats = {None: 1, "low": 1, "medium": 2, "high": 3}[severity]
    for event in core:
        events.extend([event] * repeats if rng.random() < 0.35 else [event])
        if rng.random() < 0.25:
            events.append(rng.choice(NOISE_EVENTS))
    events.append(f"P{rng.randrange(32):02d}")
    events.extend(("E087", "E099") if severity is not None else ("E099",))
    return tuple(events)


def generate_synthetic_records(count: int, *, seed: int) -> list[SequenceRecord]:
    """Generate variable, grouped records without putting label words in event tokens."""
    if count < 30:
        raise ValueError("Synthetic dataset requires at least 30 records")
    rng = random.Random(seed)
    categories = tuple(INCIDENT_PATTERNS)
    records: list[SequenceRecord] = []
    group_size = 3
    for index in range(count):
        incident_index = index // group_size
        group_rng = random.Random(seed * 1_000_003 + incident_index)
        anomalous = group_rng.random() < 0.40
        category = categories[incident_index % len(categories)] if anomalous else None
        severity = (
            SEVERITIES[(incident_index // len(categories)) % len(SEVERITIES)] if anomalous else None
        )
        core = INCIDENT_PATTERNS[category] if category is not None else rng.choice(NORMAL_CORES)[2:]
        events = _mutate_sequence(core, rng=rng, severity=severity)
        record = SequenceRecord(
            session_id=f"sim-{incident_index:05d}-{index % group_size}",
            group_id=f"incident-{incident_index:05d}",
            events=events,
            anomaly=int(anomalous),
            source="opsforge-sim-v1",
            category=category,
            severity=severity,
        )
        validate_record(record)
        records.append(record)
    return records
