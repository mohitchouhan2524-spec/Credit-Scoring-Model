"""
preprocessing.py
-----------------
Data loading and cleaning for the credit scoring dataset:
- Load raw CSV
- Basic sanity checks / dedup
- Missing value imputation
- Train/test split (stratified on the target)

This module deliberately does NOT do feature engineering (ratios, binning,
etc.) -- that lives in features.py. Keeping cleaning and engineering
separate makes each step easier to test and reuse in the app.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    get_logger,
    load_csv,
)

logger = get_logger(__name__)

# Columns we never want to feed into the model even if present in the raw data
ID_COLUMNS = ["customer_id"]


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw credit dataset from disk."""
    logger.info(f"Loading raw data from {path}")
    df = load_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_duplicates_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and pure identifier columns."""
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    id_cols_present = [c for c in ID_COLUMNS if c in df.columns]
    if id_cols_present:
        df = df.drop(columns=id_cols_present)
        logger.info(f"Dropped identifier columns: {id_cols_present}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.
    Numeric columns  -> median
    Categorical cols -> mode ("most frequent")
    Rows missing the target are dropped (we cannot use them for training).
    """
    df = df.copy()

    if TARGET_COLUMN in df.columns:
        before = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} rows with missing target")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COLUMN]
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    for col in categorical_cols:
        n_missing = df[col].isna().sum()
        if n_missing:
            mode_val = df[col].mode(dropna=True)
            mode_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Imputed {n_missing} missing values in '{col}' with mode='{mode_val}'")

    return df


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN, positive_label=None) -> pd.DataFrame:
    """
    Explicitly map a string/categorical target to 0/1, so the positive class
    (1 = default / bad credit risk) is never left to sklearn's alphabetical
    class ordering, which can silently flip precision/recall/ROC-AUC.

    positive_label: the raw value in your data that means "defaulted / bad
    credit risk" (e.g. "bad", "2", 1 -- check with df[target_col].unique()
    first). If None, assumes the target is already 0/1 and does nothing.
    """
    df = df.copy()
    if positive_label is None:
        return df

    unique_vals = df[target_col].unique().tolist()
    if positive_label not in unique_vals:
        raise ValueError(
            f"positive_label={positive_label!r} not found in {target_col} values: {unique_vals}"
        )

    mapping = {v: (1 if v == positive_label else 0) for v in unique_vals}
    logger.info(f"Encoding target '{target_col}' -> {mapping}")
    df[target_col] = df[target_col].map(mapping).astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: dedup -> drop IDs -> impute missing values."""
    df = drop_duplicates_and_ids(df)
    df = handle_missing_values(df)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split so the default rate is preserved in both sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Split data -> train: {X_train.shape[0]} rows, test: {X_test.shape[0]} rows "
        f"(train default rate={y_train.mean():.2%}, test default rate={y_test.mean():.2%})"
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience entry point: load + clean in one call."""
    df = load_raw_data(path)
    df = clean_data(df)
    return df


if __name__ == "__main__":
    df = run_preprocessing()
    print(df.head())
    print(df.info())