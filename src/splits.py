"""Leakage-aware train, validation, and test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    users: dict[str, list[str]]


def split_transactions(
    frame: pd.DataFrame,
    user_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    user_level: bool = True,
) -> DataSplits:
    """Split by user by default; row-level mode exists only for debugging."""
    if user_level:
        users = frame[user_column].drop_duplicates().to_numpy()
        if len(users) < 3:
            raise ValueError("At least three users are required for user-level splitting")
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(users)
        train_end = min(
            max(1, int(len(shuffled) * train_ratio)),
            len(shuffled) - 2,
        )
        val_end = max(train_end + 1, int(len(shuffled) * (train_ratio + val_ratio)))
        val_end = min(val_end, len(shuffled) - 1)
        split_users = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }
        frames = {
            name: frame[frame[user_column].isin(values)].copy()
            for name, values in split_users.items()
        }
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(frame))
        train_end = int(len(order) * train_ratio)
        val_end = int(len(order) * (train_ratio + val_ratio))
        indices = {
            "train": order[:train_end],
            "validation": order[train_end:val_end],
            "test": order[val_end:],
        }
        frames = {name: frame.iloc[values].copy() for name, values in indices.items()}
        split_users = {
            name: values[user_column].drop_duplicates().to_numpy()
            for name, values in frames.items()
        }

    serialized_users = {
        name: [str(value) for value in values.tolist()]
        for name, values in split_users.items()
    }
    return DataSplits(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        users=serialized_users,
    )
