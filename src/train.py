"""Stateless LSTM training loop."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .evaluate import classification_metrics, predict_probabilities

LOGGER = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    train_labels: np.ndarray,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    patience: int,
    gradient_clip: float,
    checkpoint_path: str | Path,
    checkpoint_metadata: dict[str, Any],
    threshold: float = 0.5,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    """Train with weighted BCE, clipping, early stopping, and checkpointing."""
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    if positives == 0:
        raise ValueError("Training split has no fraud-preceding sequences")
    pos_weight = torch.tensor([negatives / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        train_loss = total_loss / max(total_samples, 1)
        validation_loss = _loss_on_loader(
            model, validation_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch,
            epochs,
            train_loss,
            validation_loss,
        )

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": best_state,
                "best_validation_loss": best_loss,
                "epoch": epoch,
                **checkpoint_metadata,
            }
            destination = Path(checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, destination)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping after epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    validation_labels, validation_probabilities = predict_probabilities(
        model, validation_loader, device
    )
    validation_metrics = classification_metrics(
        validation_labels, validation_probabilities, threshold
    )
    return model, history, validation_metrics


def _loss_on_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, labels in loader:
            labels = labels.to(device)
            loss = criterion(model(features.to(device)), labels)
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
    if total_samples == 0:
        raise ValueError("Validation split produced no sequences")
    return total_loss / total_samples
