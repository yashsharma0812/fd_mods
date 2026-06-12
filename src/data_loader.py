"""CSV loading and IBM schema normalization."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

CANONICAL_COLUMNS = {
    "user": "User",
    "userid": "User",
    "card": "Card",
    "cardid": "Card",
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "time": "Time",
    "amount": "Amount",
    "usechip": "Use Chip",
    "chip": "Use Chip",
    "merchantname": "Merchant Name",
    "merchantcity": "Merchant City",
    "merchantstate": "Merchant State",
    "zip": "Zip",
    "zipcode": "Zip",
    "mcc": "MCC",
    "errors": "Errors?",
    "error": "Errors?",
    "isfraud": "Is Fraud?",
    "fraud": "Is Fraud?",
}


def _column_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map minor naming variants to the IBM dataset's canonical names."""
    rename = {
        column: CANONICAL_COLUMNS[_column_key(column)]
        for column in frame.columns
        if _column_key(column) in CANONICAL_COLUMNS
    }
    normalized = frame.rename(columns=rename)
    duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Column normalization produced duplicates: {duplicates}")
    return normalized


def load_transactions(
    path: str | Path,
    chunksize: int | None = 500_000,
) -> pd.DataFrame:
    """Load a transaction CSV, using chunks to bound parser memory."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Update dataset_path in config.yaml."
        )
    LOGGER.info("Loading transactions from %s", csv_path)
    if chunksize:
        chunks = [normalize_columns(chunk) for chunk in pd.read_csv(csv_path, chunksize=chunksize)]
        frame = pd.concat(chunks, ignore_index=True)
    else:
        frame = normalize_columns(pd.read_csv(csv_path))
    LOGGER.info("Loaded %s rows and %s columns", f"{len(frame):,}", len(frame.columns))
    return frame
