"""Prediction for a single prior-transaction sequence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .config import Config
from .data_loader import normalize_columns
from .feature_engineering import engineer_features
from .model import FraudPrecursorLSTM
from .preprocessing import FeaturePreprocessor, clean_transactions


def predict_sequence(
    input_path: str | Path,
    config: Config,
    preprocessor: FeaturePreprocessor,
    checkpoint: dict,
    device: torch.device,
) -> float:
    """Classify N prior transactions as fraud-preceding or normal-preceding."""
    frame = normalize_columns(pd.read_csv(input_path))
    if len(frame) != config.sequence_length:
        raise ValueError(
            f"Prediction input must contain exactly {config.sequence_length} rows; "
            f"received {len(frame)}"
        )
    frame[config.user_column] = "PREDICTION_USER"
    frame[config.target_column] = 0
    cleaned = clean_transactions(frame, config.user_column, config.target_column)
    featured, _, _, _ = engineer_features(
        cleaned, config.user_column, config.use_card_as_feature
    )
    transformed = preprocessor.transform(featured)
    tensor = torch.from_numpy(transformed).unsqueeze(0).float().to(device)

    model = FraudPrecursorLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor)).item())
