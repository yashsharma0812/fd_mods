"""Stateless LSTM fraud-precursor classifier."""

from __future__ import annotations

import torch
from torch import nn


class FraudPrecursorLSTM(nn.Module):
    """Classify prior transaction windows without carrying hidden state."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Omitting an incoming hidden state makes every window stateless.
        _, (hidden, _) = self.lstm(inputs)
        return self.classifier(hidden[-1]).squeeze(-1)
