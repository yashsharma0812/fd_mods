"""Shared utility functions."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def setup_logging(log_path: str | Path | None = None) -> None:
    """Configure console and optional file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    """Set deterministic random seeds where supported."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dirs(outputs_dir: str | Path) -> dict[str, Path]:
    """Create and return standard output directories."""
    root = Path(outputs_dir)
    paths = {
        "root": root,
        "models": root / "models",
        "plots": root / "plots",
        "reports": root / "reports",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON, converting common NumPy scalar types."""

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=default)
