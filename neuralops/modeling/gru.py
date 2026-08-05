"""Packed variable-length GRU encoder with optional multi-task heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True, slots=True)
class GRUConfig:
    vocabulary_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.2
    category_classes: int = 0
    severity_classes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GRUConfig:
        return cls(
            vocabulary_size=int(raw["vocabulary_size"]),
            embedding_dim=int(raw["embedding_dim"]),
            hidden_dim=int(raw["hidden_dim"]),
            layers=int(raw["layers"]),
            bidirectional=bool(raw["bidirectional"]),
            dropout=float(raw["dropout"]),
            category_classes=int(raw.get("category_classes", 0)),
            severity_classes=int(raw.get("severity_classes", 0)),
        )


class GRUClassifier(nn.Module):
    """Encode event sequences and produce binary plus optional auxiliary logits."""

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        if config.vocabulary_size < 2:
            raise ValueError("vocabulary_size must include PAD and UNK")
        if config.layers < 1:
            raise ValueError("layers must be positive")
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        directions = 2 if config.bidirectional else 1
        encoded_dim = config.hidden_dim * directions
        self.dropout = nn.Dropout(config.dropout)
        self.binary_head = nn.Linear(encoded_dim, 1)
        self.category_head = (
            nn.Linear(encoded_dim, config.category_classes) if config.category_classes > 0 else None
        )
        self.severity_head = (
            nn.Linear(encoded_dim, config.severity_classes) if config.severity_classes > 0 else None
        )

    def encode(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        if self.config.bidirectional:
            encoded = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            encoded = cast(Tensor, hidden[-1])
        return cast(Tensor, self.dropout(encoded))

    def forward(self, tokens: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        encoded = self.encode(tokens, lengths)
        outputs = {"binary": self.binary_head(encoded).squeeze(1)}
        if self.category_head is not None:
            outputs["category"] = self.category_head(encoded)
        if self.severity_head is not None:
            outputs["severity"] = self.severity_head(encoded)
        return outputs
