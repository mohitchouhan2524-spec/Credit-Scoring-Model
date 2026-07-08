"""
features.py
------------
Feature engineering from raw financial history fields.
Assumed raw columns (adjust FEATURE COLUMN NAMES below to match your real
dataset if the names differ -- e.g. from a Kaggle credit dataset):
    age, income, employment_length, existing_loans, debt, loan_amount,
    credit_history_length, num_credit_lines, num_late_payments,
    credit_utilization, savings

Engineered features:
    debt_to_income_ratio     = debt / income
    loan_to_income_ratio     = loan_amount / income
    savings_to_income_ratio  = savings / income
    payment_history_score    = inverse of late payments relative to credit history
    credit_mix_score         = num_credit_lines normalized by age
    is_high_utilization      = flag for utilization > 0.7
    age_bucket                = binned age group (categorical -> one-hot later)
    income_bucket             = binned income group

This module also builds the full sklearn ColumnTransformer used by train.py,
so the exact same transformation is applied at training and inference time.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import get_logger

logger = get_logger(__name__)

EPS = 1e-6  # avoid division by zero


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived financial ratio and bucket features. Returns a new dataframe."""
    df = df.copy()

    if {"n_credits", "age"}.issubset(df.columns):
        df["credit_mix_score"] = df["n_credits"] / (df["age"] + EPS)

    if {"credit_amount", "month_duration"}.issubset(df.columns):
        df["installment_amount"] = df["credit_amount"] / (df["month_duration"] + EPS)

    if "age" in df.columns:
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
        ).astype(str)

    logger.info(f"Feature engineering complete -> {df.shape[1]} columns")
    return df


def get_feature_lists(df: pd.DataFrame, target_col: str = None):
    """Split columns into numeric vs categorical feature lists for the ColumnTransformer."""
    cols = [c for c in df.columns if c != target_col]
    numeric_features = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[cols].select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer:
    - numeric features -> standard scaled
    - categorical features -> one-hot encoded
    This gets embedded as the first step of the model Pipeline in train.py,
    so scaling/encoding is fit ONLY on training data and reused at inference.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor