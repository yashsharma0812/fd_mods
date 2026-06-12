"""Model evaluation and metric calculation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and sigmoid probabilities for a loader."""
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.numpy())
    if not labels:
        return np.empty(0), np.empty(0)
    return np.concatenate(labels), np.concatenate(probabilities)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate imbalance-aware binary classification metrics."""
    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty split")
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "samples": int(len(labels)),
        "fraud_preceding_sequences": int(labels.sum()),
        "normal_preceding_sequences": int(len(labels) - labels.sum()),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
        metrics["pr_auc"] = average_precision_score(labels, probabilities)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics
