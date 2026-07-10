"""
utils.py
--------
Shared utilities used across the credit scoring pipeline:
- Project path configuration
- Logging setup
- Generic save/load helpers for models and dataframes
- Reproducibility helpers
"""

import os
import json
import random
import logging
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

# --------------------------------------------------------------------------
# Project paths (resolved relative to this file, so it works from anywhere)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "credit_data.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_credit_data.csv"

TARGET_COLUMN = "target"
RANDOM_STATE = 9


def ensure_dirs():
    """Create all project directories if they don't already exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints to stdout with timestamps."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_model(model, filename: str):
    """Save a fitted model/pipeline to models/ using joblib."""
    ensure_dirs()
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str):
    """Load a model/pipeline previously saved with save_model()."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}")
    return joblib.load(path)


def save_json(data: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)