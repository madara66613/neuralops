"""Duplicate-aware, group-disjoint deterministic splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from neuralops.data.records import SequenceRecord, validate_record

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(slots=True)
class _UnionFind:
    parent: dict[str, str]

    def __init__(self) -> None:
        self.parent = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_records(
    records: Iterable[SequenceRecord],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[SequenceRecord]]:
    """Keep connected group/fingerprint components in one stable-hash split."""
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be positive and sum to 1")
    materialized = list(records)
    if not materialized:
        raise ValueError("Cannot split an empty dataset")
    union_find = _UnionFind()
    for record in materialized:
        validate_record(record)
        union_find.union(f"group:{record.group_id}", f"fingerprint:{record.fingerprint}")

    components: defaultdict[str, list[SequenceRecord]] = defaultdict(list)
    for record in materialized:
        components[union_find.find(f"group:{record.group_id}")].append(record)

    boundaries = (ratios[0], ratios[0] + ratios[1])
    result: dict[str, list[SequenceRecord]] = {name: [] for name in SPLIT_NAMES}
    for component_id, component in sorted(components.items()):
        # Some real HDFS sequences are identical but carry conflicting labels. The
        # complete connected component still belongs in one split: separating or
        # dropping the inconvenient examples would bias evaluation.
        fraction = _stable_fraction(component_id, seed)
        if fraction < boundaries[0]:
            split_name = "train"
        elif fraction < boundaries[1]:
            split_name = "validation"
        else:
            split_name = "test"
        result[split_name].extend(component)

    for split_records_list in result.values():
        split_records_list.sort(key=lambda record: record.session_id)
    assert_disjoint(result)
    return result


def assert_disjoint(splits: dict[str, list[SequenceRecord]]) -> None:
    """Prove sessions, groups, and fingerprints do not cross split boundaries."""
    checks: tuple[tuple[str, Callable[[SequenceRecord], str]], ...] = (
        ("session", lambda record: record.session_id),
        ("group", lambda record: record.group_id),
        ("fingerprint", lambda record: record.fingerprint),
    )
    for field_name, getter in checks:
        seen: dict[str, str] = {}
        for split_name in SPLIT_NAMES:
            for record in splits.get(split_name, []):
                value = getter(record)
                previous = seen.setdefault(value, split_name)
                if previous != split_name:
                    raise AssertionError(
                        f"{field_name} {value} appears in both {previous} and {split_name}"
                    )
