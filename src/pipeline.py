"""Shared data preparation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config
from .data_loader import load_transactions
from .feature_engineering import engineer_features
from .preprocessing import FeaturePreprocessor, clean_transactions
from .sequence_generator import Sequences, generate_sequences, sequence_summary
from .splits import DataSplits, split_transactions

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedData:
    frame: object
    splits: DataSplits
    preprocessor: FeaturePreprocessor
    sequences: dict[str, Sequences]
    numerical_columns: list[str]
    categorical_columns: list[str]
    frequency_columns: list[str]


def prepare_data(
    config: Config,
    fitted_preprocessor: FeaturePreprocessor | None = None,
) -> PreparedData:
    """Load, clean, split, fit preprocessing, and create backward windows."""
    raw = load_transactions(config.dataset_path, config.chunksize)
    cleaned = clean_transactions(raw, config.user_column, config.target_column)
    featured, numerical, categorical, frequency = engineer_features(
        cleaned, config.user_column, config.use_card_as_feature
    )
    splits = split_transactions(
        featured,
        config.user_column,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed,
        config.use_user_level_split,
    )
    preprocessor = fitted_preprocessor or FeaturePreprocessor(
        numerical, categorical, frequency
    ).fit(splits.train)

    sequence_sets: dict[str, Sequences] = {}
    for name, split_frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        transformed = preprocessor.transform(split_frame)
        sequence_sets[name] = generate_sequences(
            split_frame,
            transformed,
            config.sequence_length,
            config.user_column,
            config.target_column,
        )
        LOGGER.info("%s sequences: %s", name, sequence_summary(sequence_sets[name]))

    return PreparedData(
        frame=featured,
        splits=splits,
        preprocessor=preprocessor,
        sequences=sequence_sets,
        numerical_columns=numerical,
        categorical_columns=categorical,
        frequency_columns=frequency,
    )
