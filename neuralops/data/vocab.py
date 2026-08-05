"""Train-only event vocabulary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocabulary(sequences: Iterable[Sequence[str]], min_frequency: int = 1) -> dict[str, int]:
    counts: Counter[str] = Counter(event for sequence in sequences for event in sequence)
    ordered = sorted(
        (event for event, count in counts.items() if count >= min_frequency),
        key=lambda event: (-counts[event], event),
    )
    return {PAD_TOKEN: 0, UNK_TOKEN: 1, **{event: index + 2 for index, event in enumerate(ordered)}}


def unknown_rate(sequences: Iterable[Sequence[str]], vocabulary: dict[str, int]) -> float:
    unknown = 0
    total = 0
    for sequence in sequences:
        for event in sequence:
            total += 1
            unknown += int(event not in vocabulary)
    return unknown / total if total else 0.0
