"""Leakage-safe behavioral feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_NUMERICAL_FEATURES = [
    "amount",
    "log_amount",
    "hour",
    "day_of_week",
    "month",
    "time_since_prev_transaction_minutes",
    "rolling_amount_mean_user",
    "rolling_amount_std_user",
    "amount_to_user_rolling_mean_ratio",
    "amount_zscore_user",
    "transaction_velocity_last_1h",
    "transaction_velocity_last_24h",
    "merchant_seen_before_flag",
    "city_seen_before_flag",
    "state_seen_before_flag",
]

OPTIONAL_CATEGORICAL_FEATURES = [
    "Use Chip",
    "Merchant City",
    "Merchant State",
    "MCC",
    "Errors?",
]


def engineer_features(
    frame: pd.DataFrame,
    user_column: str = "User",
    use_card_as_feature: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Create per-transaction features using no future transaction information."""
    data = frame.copy()
    grouped = data.groupby(user_column, sort=False, group_keys=False)
    # Zero is a context-safe fallback here; train-fitted median imputation is
    # applied later to model features without consulting validation/test users.
    amount = data["Amount"].fillna(0.0)

    data["amount"] = amount.astype(float)
    data["log_amount"] = np.sign(amount) * np.log1p(np.abs(amount))
    data["hour"] = data["datetime"].dt.hour.astype(float)
    data["day_of_week"] = data["datetime"].dt.dayofweek.astype(float)
    data["month"] = data["datetime"].dt.month.astype(float)
    data["time_since_prev_transaction_minutes"] = (
        grouped["datetime"].diff().dt.total_seconds().div(60).clip(lower=0).fillna(0.0)
    )

    prior_amount = grouped["Amount"].shift(1)
    rolling = prior_amount.groupby(data[user_column], sort=False).rolling(
        window=20, min_periods=1
    )
    rolling_mean = rolling.mean().reset_index(level=0, drop=True)
    rolling_std = rolling.std(ddof=0).reset_index(level=0, drop=True)
    data["rolling_amount_mean_user"] = rolling_mean.fillna(0.0)
    data["rolling_amount_std_user"] = rolling_std.fillna(0.0)
    denominator = data["rolling_amount_mean_user"].abs().clip(lower=1e-6)
    data["amount_to_user_rolling_mean_ratio"] = data["amount"] / denominator
    data["amount_zscore_user"] = (
        data["amount"] - data["rolling_amount_mean_user"]
    ) / data["rolling_amount_std_user"].clip(lower=1e-6)
    data["amount_zscore_user"] = data["amount_zscore_user"].replace(
        [np.inf, -np.inf], 0.0
    )

    data["transaction_velocity_last_1h"] = _past_velocity(data, user_column, "1h")
    data["transaction_velocity_last_24h"] = _past_velocity(data, user_column, "24h")
    data["merchant_seen_before_flag"] = _seen_before(
        data, user_column, "Merchant Name"
    )
    data["city_seen_before_flag"] = _seen_before(data, user_column, "Merchant City")
    data["state_seen_before_flag"] = _seen_before(
        data, user_column, "Merchant State"
    )

    numerical = [column for column in BASE_NUMERICAL_FEATURES if column in data]
    categorical = [
        column for column in OPTIONAL_CATEGORICAL_FEATURES if column in data.columns
    ]
    frequency = ["Merchant Name"] if "Merchant Name" in data.columns else []
    if use_card_as_feature and "Card" in data.columns:
        frequency.append("Card")
    return data, numerical, categorical, frequency


def _seen_before(frame: pd.DataFrame, user_column: str, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    occurrence = frame.groupby([user_column, value_column], sort=False).cumcount()
    return occurrence.gt(0).astype(float)


def _past_velocity(
    frame: pd.DataFrame, user_column: str, window: str
) -> pd.Series:
    """Count prior transactions in a time window, excluding the current row."""
    result = pd.Series(0.0, index=frame.index, dtype=float)
    delta = pd.Timedelta(window)
    for _, group in frame.groupby(user_column, sort=False):
        timestamps = group["datetime"].astype("int64").to_numpy()
        left = np.searchsorted(timestamps, timestamps - delta.value, side="left")
        counts = np.arange(len(group)) - left
        result.loc[group.index] = counts.astype(float)
    return result
