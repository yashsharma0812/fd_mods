"""Dataset summaries and final Markdown reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config


def dataset_summary(
    frame: pd.DataFrame,
    user_column: str,
    target_column: str,
    sequence_length: int,
) -> dict[str, Any]:
    per_user = frame.groupby(user_column).size()
    fraud_count = int(frame[target_column].sum())
    return {
        "total_transactions": int(len(frame)),
        "total_users": int(frame[user_column].nunique()),
        "fraud_transactions": fraud_count,
        "non_fraud_transactions": int(len(frame) - fraud_count),
        "fraud_percentage": 100.0 * fraud_count / max(len(frame), 1),
        "transactions_per_user": {
            "min": int(per_user.min()),
            "median": float(per_user.median()),
            "mean": float(per_user.mean()),
            "max": int(per_user.max()),
        },
        "users_with_at_least_n_plus_1_transactions": int(
            (per_user >= sequence_length + 1).sum()
        ),
    }


def write_final_report(
    path: str | Path,
    config: Config,
    data_summary: dict[str, Any],
    split_users: dict[str, list[str]],
    metrics: dict[str, dict[str, Any]],
) -> None:
    """Create the requested final project report."""
    lines = [
        "# Fraud Precursor LSTM Final Report",
        "",
        "## Problem Statement",
        "",
        "This project classifies a user's previous N transactions as a "
        "fraud-preceding or normal-preceding trajectory. The labeled target "
        "transaction is never included in the input window.",
        "",
        "## Dataset Summary",
        "",
        f"- Transactions: {data_summary['total_transactions']:,}",
        f"- Users: {data_summary['total_users']:,}",
        f"- Fraud transactions: {data_summary['fraud_transactions']:,}",
        f"- Fraud rate: {data_summary['fraud_percentage']:.4f}%",
        f"- Sequence length: {config.sequence_length}",
        "",
        "## Preprocessing And Sequence Logic",
        "",
        "Amounts are parsed numerically, timestamps are constructed from available "
        "date/time fields, missing categories use `UNKNOWN`, and behavioral "
        "features are calculated using prior history only. For target `Tn`, the "
        "input is exactly `Tn-N ... Tn-1` and the target is `label(Tn)`.",
        "",
        "## Stateless LSTM And Identity Leakage",
        "",
        "The PyTorch LSTM receives no hidden state from a previous batch or window. "
        "Raw user IDs are used only for grouping and splitting, never as model "
        "features. Raw card ID is disabled by default. Users are disjoint across "
        "train, validation, and test splits.",
        "",
        "## Split Details",
        "",
        f"- Train users: {len(split_users['train']):,}",
        f"- Validation users: {len(split_users['validation']):,}",
        f"- Test users: {len(split_users['test']):,}",
        "",
        "## Model And Training",
        "",
        f"- Hidden size: {config.hidden_size}",
        f"- LSTM layers: {config.num_layers}",
        f"- Dropout: {config.dropout}",
        f"- Batch size: {config.batch_size}",
        f"- Learning rate: {config.learning_rate}",
        "- Loss: weighted BCEWithLogitsLoss",
        "- Optimizer: Adam",
        "",
        "## Evaluation Metrics",
        "",
    ]
    for split_name, split_metrics in metrics.items():
        lines.extend(
            [
                f"### {split_name.title()}",
                "",
                f"- Samples: {split_metrics['samples']:,}",
                f"- Precision: {split_metrics['precision']:.4f}",
                f"- Recall: {split_metrics['recall']:.4f}",
                f"- F1: {split_metrics['f1']:.4f}",
                f"- PR-AUC: {_format_metric(split_metrics.get('pr_auc'))}",
                f"- ROC-AUC: {_format_metric(split_metrics.get('roc_auc'))}",
                f"- MCC: {split_metrics['mcc']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "The model identifies retrospective associations, not causal precursors. "
            "Results depend on label quality, transaction coverage, user split "
            "composition, and concept drift.",
            "",
            "## Future Work",
            "",
            "Evaluate longer windows, tune the decision threshold on validation "
            "data, test calibration, study temporal drift, and add explanation "
            "methods that preserve the retrospective interpretation.",
            "",
        ]
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _format_metric(value: float | None) -> str:
    return "N/A (single class)" if value is None else f"{value:.4f}"
