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

    def to(self, device: torch.device) -> Batch:
        return Batch(
            tokens=self.tokens.to(device),
            # pack_padded_sequence requires CPU lengths on every backend.
            lengths=self.lengths,
            anomaly=self.anomaly.to(device),
        )


class SequenceDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        records: list[SequenceRecord],
        vocabulary: dict[str, int],
        max_sequence_length: int,
    ) -> None:
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.records = records
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length
        self.unknown_index = vocabulary[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        encoded = [
            self.vocabulary.get(event, self.unknown_index)
            for event in record.events[: self.max_sequence_length]
        ]
        return torch.tensor(encoded, dtype=torch.long), record.anomaly


def collate_sequences(items: list[tuple[Tensor, int]]) -> Batch:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    sequences, labels = zip(*items, strict=True)
    return Batch(
        tokens=pad_sequence(list(sequences), batch_first=True, padding_value=0),
        lengths=torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long),
        anomaly=torch.tensor(labels, dtype=torch.float32),
    )
