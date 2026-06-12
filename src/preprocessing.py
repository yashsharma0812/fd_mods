"""Basic transaction cleaning and train-fitted feature preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_fraud_label(value: Any) -> int:
    """Convert common fraud label representations to 0 or 1."""
    if pd.isna(value):
        raise ValueError("Fraud target contains missing values")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "fraud", "fraudulent", "1"}:
        return 1
    if normalized in {"no", "n", "false", "not fraud", "non-fraud", "0"}:
        return 0
    raise ValueError(f"Unrecognized fraud label: {value!r}")


def clean_transactions(
    frame: pd.DataFrame,
    user_column: str = "User",
    target_column: str = "Is Fraud?",
) -> pd.DataFrame:
    """Clean amount, target, missing categories, and chronological ordering."""
    required = {user_column, target_column, "Amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = frame.copy()
    cleaned = cleaned.dropna(subset=[user_column])
    amount = (
        cleaned["Amount"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
    )
    cleaned["Amount"] = pd.to_numeric(amount, errors="coerce")
    cleaned[target_column] = cleaned[target_column].map(parse_fraud_label).astype("int8")

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].fillna("UNKNOWN").astype(str)

    cleaned["datetime"] = _build_datetime(cleaned)
    cleaned["_source_order"] = np.arange(len(cleaned), dtype=np.int64)
    cleaned = cleaned.sort_values(
        [user_column, "datetime", "_source_order"], kind="mergesort"
    ).reset_index(drop=True)
    return cleaned


def _build_datetime(frame: pd.DataFrame) -> pd.Series:
    date_parts = {"Year", "Month", "Day"}
    if date_parts.issubset(frame.columns):
        time_values = (
            frame["Time"].astype(str).str.strip()
            if "Time" in frame
            else pd.Series("00:00", index=frame.index)
        )
        dates = pd.to_datetime(
            {
                "year": pd.to_numeric(frame["Year"], errors="coerce"),
                "month": pd.to_numeric(frame["Month"], errors="coerce"),
                "day": pd.to_numeric(frame["Day"], errors="coerce"),
            },
            errors="coerce",
        )
        parsed_time = pd.to_timedelta(
            time_values.where(time_values.str.match(r"^\d{1,2}:\d{2}"), "00:00") + ":00",
            errors="coerce",
        ).fillna(pd.Timedelta(0))
        result = dates + parsed_time
        if result.notna().any():
            fallback = pd.Timestamp("1970-01-01") + pd.to_timedelta(
                np.arange(len(frame)), unit="s"
            )
            return result.fillna(pd.Series(fallback, index=frame.index))
    return pd.Series(
        pd.Timestamp("1970-01-01")
        + pd.to_timedelta(np.arange(len(frame)), unit="s"),
        index=frame.index,
    )


@dataclass
class FeaturePreprocessor:
    """Frequency encode high-cardinality fields and transform remaining columns."""

    numerical_columns: list[str]
    categorical_columns: list[str]
    frequency_columns: list[str]
    transformer: ColumnTransformer | None = None
    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        working = self._with_required_columns(frame)
        for column in self.frequency_columns:
            frequencies = working[column].fillna("UNKNOWN").astype(str).value_counts(
                normalize=True
            )
            self.frequency_maps[column] = frequencies.to_dict()
            working[f"{column}_frequency"] = (
                working[column].astype(str).map(self.frequency_maps[column]).fillna(0.0)
            )

        numeric = self.numerical_columns + [
            f"{column}_frequency" for column in self.frequency_columns
        ]
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        self.transformer = ColumnTransformer(
            [
                ("numeric", numeric_pipeline, numeric),
                ("categorical", categorical_pipeline, self.categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.transformer.fit(working)
        self.feature_names = self.transformer.get_feature_names_out().tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform")
        working = self._with_required_columns(frame)
        for column, mapping in self.frequency_maps.items():
            working[f"{column}_frequency"] = (
                working[column].fillna("UNKNOWN").astype(str).map(mapping).fillna(0.0)
            )
        return self.transformer.transform(working).astype(np.float32)

    def _with_required_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        for column in self.numerical_columns:
            if column not in working:
                working[column] = np.nan
        for column in self.categorical_columns + self.frequency_columns:
            if column not in working:
                working[column] = "UNKNOWN"
        return working

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FeaturePreprocessor":
        return joblib.load(path)
