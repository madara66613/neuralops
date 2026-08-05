"""PyTorch datasets for canonical NeuralOps sequence records."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from neuralops.data.records import SequenceRecord
from neuralops.data.vocab import UNK_TOKEN


@dataclass(slots=True)
class Batch:
    tokens: Tensor
    lengths: Tensor
    anomaly: Tensor
    category: Tensor
    severity: Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            tokens=self.tokens.to(device),
            # pack_padded_sequence requires CPU lengths on every backend.
            lengths=self.lengths,
            anomaly=self.anomaly.to(device),
            category=self.category.to(device),
            severity=self.severity.to(device),
        )


@dataclass(slots=True)
class EncodedItem:
    tokens: Tensor
    anomaly: int
    category: int
    severity: int


class SequenceDataset(Dataset[EncodedItem]):
    def __init__(
        self,
        records: list[SequenceRecord],
        vocabulary: dict[str, int],
        max_sequence_length: int,
        category_to_index: dict[str, int] | None = None,
        severity_to_index: dict[str, int] | None = None,
    ) -> None:
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.records = records
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length
        self.unknown_index = vocabulary[UNK_TOKEN]
        self.category_to_index = category_to_index or {}
        self.severity_to_index = severity_to_index or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EncodedItem:
        record = self.records[index]
        encoded = [
            self.vocabulary.get(event, self.unknown_index)
            for event in record.events[: self.max_sequence_length]
        ]
        return EncodedItem(
            tokens=torch.tensor(encoded, dtype=torch.long),
            anomaly=record.anomaly,
            category=(
                self.category_to_index.get(record.category, -100)
                if record.category is not None
                else -100
            ),
            severity=(
                self.severity_to_index.get(record.severity, -100)
                if record.severity is not None
                else -100
            ),
        )


def collate_sequences(items: list[EncodedItem]) -> Batch:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    sequences = [item.tokens for item in items]
    return Batch(
        tokens=pad_sequence(sequences, batch_first=True, padding_value=0),
        lengths=torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long),
        anomaly=torch.tensor([item.anomaly for item in items], dtype=torch.float32),
        category=torch.tensor([item.category for item in items], dtype=torch.long),
        severity=torch.tensor([item.severity for item in items], dtype=torch.long),
    )
