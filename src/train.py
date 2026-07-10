"""
train.py
--------
Trains and compares three classifiers for credit default prediction:
    - Logistic Regression
    - Decision Tree
    - Random Forest

Each model is wrapped in a single sklearn Pipeline together with the
ColumnTransformer from features.py, so preprocessing + model are saved
and loaded as one artifact (no train/inference skew).

Run:
    python -m src.train
"""

import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.utils import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    ensure_dirs,
    get_logger,
    set_seed,
    save_model,
    save_csv,
)
from src.preprocessing import run_preprocessing, split_data, encode_target
from src.features import engineer_features, get_feature_lists, build_preprocessor
from src.evaluate import evaluate_model, compare_models, save_metrics

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        },
    },
    "decision_tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 5, 10],
            "model__class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [6, 10, None],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"],
        },
    },
}


def build_pipeline(estimator, numeric_features, categorical_features) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    return pipeline


def train_all_models(X_train, y_train, X_test, y_test, numeric_features, categorical_features, cv_folds=5):
    """Grid-search each model in MODEL_REGISTRY, evaluate on the test set, return results."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    fitted_pipelines = {}

    for name, spec in MODEL_REGISTRY.items():
        logger.info(f"Training {name} ...")
        pipeline = build_pipeline(spec["estimator"], numeric_features, categorical_features)

        search = GridSearchCV(
            pipeline,
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        logger.info(f"{name} best params: {search.best_params_} (CV ROC-AUC={search.best_score_:.4f})")

        metrics = evaluate_model(best_pipeline, X_test, y_test, model_name=name)
        results[name] = metrics
        fitted_pipelines[name] = best_pipeline

    return results, fitted_pipelines


def main(data_path=RAW_DATA_PATH, cv_folds=5, positive_label=None):
    """
    positive_label: the raw value in your target column that means
    "defaulted / bad credit risk" (e.g. "bad", "2"). Required if your target
    column is string/categorical -- sklearn's alphabetical class ordering
    cannot be trusted to pick the right "positive" class for you. Leave as
    None only if your target is already encoded as 0/1.
    """
    set_seed()
    ensure_dirs()

    # 1. Load + clean
    df = run_preprocessing(data_path)

    # 1b. Encode target to 0/1 if it isn't numeric already
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        if positive_label is None:
            raise ValueError(
                f"'{TARGET_COLUMN}' is non-numeric (values: {df[TARGET_COLUMN].unique().tolist()}). "
                f"Pass positive_label=<value that means default/bad credit risk> to main(), "
                f"e.g. main(positive_label='bad')."
            )
        df = encode_target(df, target_col=TARGET_COLUMN, positive_label=positive_label)

    # 2. Feature engineering
    df = engineer_features(df)
    save_csv(df, PROCESSED_DATA_PATH)
    logger.info(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df, target_col=TARGET_COLUMN)
    numeric_features, categorical_features = get_feature_lists(df, target_col=TARGET_COLUMN)
    logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # 4. Train + tune all models
    results, fitted_pipelines = train_all_models(
        X_train, y_train, X_test, y_test, numeric_features, categorical_features, cv_folds=cv_folds
    )

    # 5. Compare and pick the best model (by ROC-AUC on the held-out test set)
    comparison_df = compare_models(results)
    logger.info("\n" + comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    logger.info(f"Best model: {best_model_name}")

    # 6. Persist artifacts
    save_model(best_pipeline, "best_model.pkl")
    for name, pipeline in fitted_pipelines.items():
        save_model(pipeline, f"{name}.pkl")

    save_metrics(results, comparison_df, best_model_name)

    return results, best_model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit scoring models")
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    main(data_path=args.data_path, cv_folds=args.cv_folds)