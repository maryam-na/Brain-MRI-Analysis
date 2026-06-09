from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

from data_agent import RANDOM_STATE
from model_agent import get_feature_names


METRIC_NAMES = ["roc_auc", "sensitivity", "specificity", "accuracy", "f1"]


def _positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_score = _positive_scores(model, X_test)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "roc_auc": roc_auc_score(y_test, y_score),
        "sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "specificity": specificity_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def cross_validate_model(model, X_train, y_train) -> dict:
    scoring = {
        "roc_auc": "roc_auc",
        "sensitivity": "recall",
        "specificity": make_scorer(specificity_score),
        "accuracy": "accuracy",
        "f1": "f1",
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
        }
        for metric in scoring
    }


def tune_decision_threshold_cv(
    model,
    X_train,
    y_train,
    min_sensitivity: float = 0.85,
) -> dict:
    """Choose a decision threshold from out-of-fold training probabilities.

    The primary rule keeps sensitivity at or above min_sensitivity, then picks
    the threshold with the best F1-score. If no threshold reaches that target,
    it falls back to Youden's J statistic.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if hasattr(model, "predict_proba"):
        y_score = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    else:
        y_score = cross_val_predict(model, X_train, y_train, cv=cv, method="decision_function", n_jobs=-1)

    candidates = []
    for threshold in np.unique(y_score):
        y_pred = (y_score >= threshold).astype(int)
        sensitivity = recall_score(y_train, y_pred, zero_division=0)
        specificity = specificity_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        candidates.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "f1": float(f1),
                "youden_j": float(sensitivity + specificity - 1),
            }
        )

    candidate_table = pd.DataFrame(candidates)
    sensitivity_safe = candidate_table[candidate_table["sensitivity"] >= min_sensitivity]
    if not sensitivity_safe.empty:
        best = sensitivity_safe.sort_values(["f1", "specificity"], ascending=False).iloc[0]
        strategy = f"max_f1_with_sensitivity_at_least_{min_sensitivity:.2f}"
    else:
        best = candidate_table.sort_values(["youden_j", "f1"], ascending=False).iloc[0]
        strategy = "max_youden_j_fallback"

    return {
        "threshold": float(best["threshold"]),
        "strategy": strategy,
        "cv_sensitivity": float(best["sensitivity"]),
        "cv_specificity": float(best["specificity"]),
        "cv_f1": float(best["f1"]),
    }


def compare_models(results_dict: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_name, results in results_dict.items():
        row = {"model": model_name}
        test_metrics = results.get("test", results)
        for metric in METRIC_NAMES:
            value = test_metrics.get(metric)
            if isinstance(value, dict):
                value = value.get("mean")
            row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def print_cv_results(results_dict: Dict[str, dict]) -> None:
    for model_name, results in results_dict.items():
        print(f"\n{model_name} CV results")
        for metric, values in results["cv"].items():
            print(f"  {metric}: {values['mean']:.3f} +/- {values['std']:.3f}")


def print_test_results(results_dict: Dict[str, dict]) -> None:
    for model_name, results in results_dict.items():
        print(f"\n{model_name} test results")
        print(f"  threshold: {results['test']['threshold']:.3f}")
        for metric in METRIC_NAMES:
            print(f"  {metric}: {results['test'][metric]:.3f}")
        print(f"  confusion_matrix:\n{results['test']['confusion_matrix']}")


def best_model_name(results_dict: Dict[str, dict], metric: str = "roc_auc") -> str:
    return max(results_dict, key=lambda name: results_dict[name]["test"][metric])


def ensure_plot_dir(output_dir: str | Path = "outputs/plots") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_roc_curves(models: Dict[str, object], X_test, y_test, output_dir: str | Path = "outputs/plots") -> Path:
    output_path = ensure_plot_dir(output_dir) / "roc_curves.png"
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_score = _positive_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_confusion_matrices(results_dict: Dict[str, dict], output_dir: str | Path = "outputs/plots") -> Dict[str, Path]:
    output_path = ensure_plot_dir(output_dir)
    paths = {}
    for model_name, results in results_dict.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=results["test"]["confusion_matrix"],
            display_labels=["No dementia", "Dementia"],
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{model_name} confusion matrix")
        fig.tight_layout()
        path = output_path / f"{model_name.lower()}_confusion_matrix.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[model_name] = path
    return paths


def plot_metric_bars(results_dict: Dict[str, dict], output_dir: str | Path = "outputs/plots") -> Path:
    output_path = ensure_plot_dir(output_dir) / "metric_comparison.png"
    table = compare_models(results_dict)
    long_table = table.melt(id_vars="model", value_vars=METRIC_NAMES, var_name="metric", value_name="score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=long_table, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Model metric comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_feature_importance(model, output_dir: str | Path = "outputs/plots", top_n: int = 12) -> Path:
    output_path = ensure_plot_dir(output_dir) / "randomforest_feature_importance.png"
    estimator = model.named_steps["model"]
    importances = pd.Series(estimator.feature_importances_, index=get_feature_names(model))
    importances = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("RandomForest feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str | Path = "outputs/plots") -> Path:
    output_path = ensure_plot_dir(output_dir) / "correlation_heatmap.png"
    numeric = df.select_dtypes(include=["number"])
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Clinical and volumetric correlation heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def age_brain_volume_correlation(df: pd.DataFrame) -> float:
    if "Age" not in df.columns or "nWBV" not in df.columns:
        raise ValueError("Age and nWBV are required for Age vs brain volume correlation.")
    return float(df["Age"].corr(df["nWBV"]))
