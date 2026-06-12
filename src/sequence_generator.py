"""Backward-looking sequence generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Sequences:
    features: np.ndarray
    labels: np.ndarray
    target_indices: np.ndarray
    target_users: np.ndarray


def generate_sequences(
    frame: pd.DataFrame,
    transformed_features: np.ndarray,
    sequence_length: int,
    user_column: str,
    target_column: str,
) -> Sequences:
    """Generate ``Tn-N ... Tn-1`` inputs with ``label(Tn)`` targets."""
    if len(frame) != len(transformed_features):
        raise ValueError("Frame and transformed feature row counts do not match")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    target_indices: list[int] = []
    target_users: list[object] = []

    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for user, group in frame.groupby(user_column, sort=False):
        group_positions = positions.loc[group.index].to_numpy()
        group_labels = group[target_column].to_numpy(dtype=np.float32)
        for target_offset in range(sequence_length, len(group_positions)):
            start = target_offset - sequence_length
            input_positions = group_positions[start:target_offset]
            windows.append(transformed_features[input_positions])
            labels.append(int(group_labels[target_offset]))
            target_indices.append(int(group.index[target_offset]))
            target_users.append(user)

    feature_count = transformed_features.shape[1]
    if not windows:
        empty_x = np.empty((0, sequence_length, feature_count), dtype=np.float32)
        return Sequences(
            empty_x,
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=object),
        )
    return Sequences(
        features=np.stack(windows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        target_indices=np.asarray(target_indices, dtype=np.int64),
        target_users=np.asarray(target_users, dtype=object),
    )


def sequence_summary(sequences: Sequences) -> dict[str, int]:
    positives = int(sequences.labels.sum())
    return {
        "total_windows": int(len(sequences.labels)),
        "fraud_preceding_windows": positives,
        "normal_preceding_windows": int(len(sequences.labels) - positives),
    }
