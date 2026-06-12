"""Project configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Runtime settings loaded from ``config.yaml``."""

    dataset_path: str = "data/raw/credit_card_transactions-ibm_v2.csv"
    target_column: str = "Is Fraud?"
    user_column: str = "User"
    sequence_length: int = 5
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 50
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    use_user_level_split: bool = True
    use_card_as_feature: bool = False
    model_save_path: str = "outputs/models/best_model.pt"
    outputs_dir: str = "outputs"
    chunksize: int = 500_000
    patience: int = 7
    gradient_clip: float = 1.0
    num_workers: int = 0
    threshold: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config values from YAML and reject unknown options."""
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        config = cls(**values)
        config._resolve_paths(config_path.parent)
        config.validate()
        return config

    def _resolve_paths(self, base_dir: Path) -> None:
        for name in ("dataset_path", "model_save_path", "outputs_dir"):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                setattr(self, name, str((base_dir / value).resolve()))

    def validate(self) -> None:
        """Validate settings that affect data and model correctness."""
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.num_layers < 1 or self.hidden_size < 1:
            raise ValueError("num_layers and hidden_size must be positive")
        ratios = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratios - 1.0) > 1e-8:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
