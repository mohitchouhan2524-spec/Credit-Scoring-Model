"""
evaluate.py
-----------
Model evaluation utilities:
- Precision, Recall, F1, ROC-AUC, Accuracy
- Confusion matrix
- ROC curve plot
- Feature importance plot (tree-based models) / coefficient plot (logistic regression)
- Model comparison table + persisted JSON report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import chi2_contingency

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.utils import FIGURES_DIR, REPORTS_DIR, get_logger, save_json, ensure_dirs

logger = get_logger(__name__)


def evaluate_model(pipeline, X_test, y_test, model_name: str = "model") -> dict:
    """Compute standard classification metrics for a fitted pipeline on test data."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }

    logger.info(
        f"[{model_name}] acc={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1_score']:.4f} roc_auc={metrics['roc_auc']:.4f}"
    )

    plot_confusion_matrix(metrics["confusion_matrix"], model_name)
    plot_roc_curve(y_test, y_proba, model_name)

    return metrics


def compare_models(results: dict) -> pd.DataFrame:
    """Build a sorted comparison table (best ROC-AUC first) across all trained models."""
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
            }
        )
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    plot_model_comparison(df)
    return df


def plot_confusion_matrix(cm, model_name: str):
    ensure_dirs()
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot -> {path}")


def plot_roc_curve(y_test, y_proba, model_name: str):
    ensure_dirs()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"roc_curve_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curve plot -> {path}")


def plot_model_comparison(comparison_df: pd.DataFrame):
    ensure_dirs()
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    melted = comparison_df.melt(id_vars="model", value_vars=metrics_to_plot, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved model comparison plot -> {path}")


def plot_feature_importance(pipeline, model_name: str, top_n: int = 15):
    """
    Works for tree-based models (feature_importances_) and logistic regression
    (coef_). Reads feature names out of the fitted ColumnTransformer so
    one-hot-encoded columns are labeled correctly.
    """
    ensure_dirs()
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        logger.warning("Could not retrieve feature names from preprocessor")
        return

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importance - {model_name}"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"Feature Coefficients (abs) - {model_name}"
    else:
        logger.warning(f"Model {model_name} has no interpretable importances")
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title(title)
    fig.tight_layout()
    path = FIGURES_DIR / f"feature_importance_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved feature importance plot -> {path}")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Cramer's V: association strength between two categorical variables, 0-1
    (0 = no association, 1 = perfect association). Derived from the chi-square
    statistic, with a bias correction (Bergsma 2013) so small samples / high
    cardinality don't inflate the score.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape

    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((k_corr - 1), (r_corr - 1))

    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def categorical_association_matrix(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Build a symmetric matrix of Cramer's V scores for every pair of categorical
    columns. Treat this like a correlation matrix, but for categorical data
    (Pearson correlation is not valid for non-numeric/non-ordinal features).
    """
    if columns is None:
        columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    matrix = pd.DataFrame(np.ones((len(columns), len(columns))), index=columns, columns=columns)
    for col1, col2 in combinations(columns, 2):
        v = cramers_v(df[col1], df[col2])
        matrix.loc[col1, col2] = v
        matrix.loc[col2, col1] = v

    return matrix


def plot_categorical_association(df: pd.DataFrame, columns=None, filename="categorical_association.png"):
    """Heatmap of Cramer's V association strength between categorical columns."""
    ensure_dirs()
    matrix = categorical_association_matrix(df, columns=columns)

    fig, ax = plt.subplots(figsize=(max(5, len(matrix) * 0.9), max(4, len(matrix) * 0.8)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, ax=ax)
    ax.set_title("Categorical Feature Association (Cramer's V)")
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved categorical association heatmap -> {path}")
    return matrix


def save_metrics(results: dict, comparison_df: pd.DataFrame, best_model_name: str):
    """Persist all metrics (minus confusion matrices/reports for the summary) to outputs/reports/."""
    ensure_dirs()
    summary = {
        "best_model": best_model_name,
        "comparison": comparison_df.to_dict(orient="records"),
        "full_results": {
            name: {k: v for k, v in m.items() if k != "classification_report"}
            for name, m in results.items()
        },
    }
    path = REPORTS_DIR / "metrics_summary.json"
    save_json(summary, path)
    logger.info(f"Saved metrics summary -> {path}")