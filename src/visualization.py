"""EDA and model-result visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve


def generate_eda_plots(
    frame: pd.DataFrame,
    plots_dir: str | Path,
    user_column: str,
    target_column: str,
) -> None:
    destination = Path(plots_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _count_plot(frame[target_column], "Fraud class distribution", destination / "fraud_class_distribution.png")
    _histogram(
        frame.groupby(user_column).size(),
        "Transactions per user",
        destination / "transactions_per_user.png",
    )
    _histogram(
        frame.groupby(user_column)[target_column].sum(),
        "Frauds per user",
        destination / "frauds_per_user.png",
    )
    _histogram(frame["Amount"], "Transaction amount", destination / "amount_distribution.png")
    _histogram(
        np.log1p(frame["Amount"].abs()),
        "Log transaction amount",
        destination / "log_amount_distribution.png",
    )


def plot_training_history(history: dict[str, list[float]], path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["validation_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    _save(path)


def plot_evaluation(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    plots_dir: str | Path,
    split_name: str,
    threshold: float = 0.5,
) -> None:
    destination = Path(plots_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        labels, predictions, labels=[0, 1], cmap="Blues"
    )
    display.ax_.set_title(f"{split_name.title()} confusion matrix")
    _save(destination / f"{split_name}_confusion_matrix.png")

    if len(np.unique(labels)) == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
        plt.figure(figsize=(7, 5))
        plt.plot(false_positive_rate, true_positive_rate, label=f"AUC={metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{split_name.title()} ROC curve")
        plt.legend()
        _save(destination / f"{split_name}_roc_curve.png")

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, label=f"AP={metrics['pr_auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{split_name.title()} precision-recall curve")
        plt.legend()
        _save(destination / f"{split_name}_precision_recall_curve.png")


def _count_plot(values: pd.Series, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.countplot(x=values)
    plt.title(title)
    _save(path)


def _histogram(values: pd.Series, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(values.dropna(), bins=50)
    plt.title(title)
    _save(path)


def _save(path: str | Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
