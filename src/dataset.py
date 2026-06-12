"""PyTorch dataset wrappers."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FraudSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tensor dataset for fixed-length transaction windows."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(features) != len(labels):
            raise ValueError("Feature and label counts do not match")
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
