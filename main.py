from __future__ import annotations

import json
from pathlib import Path

from data_agent import load_oasis_data, prepare_dataset, split_data, validate_z_scores, z_score_table
from evaluation import (
    age_brain_volume_correlation,
    best_model_name,
    compare_models,
    cross_validate_model,
    evaluate_model,
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_metric_bars,
    plot_roc_curves,
    print_cv_results,
    print_test_results,
    tune_decision_threshold_cv,
)
from model_agent import tune_models


def run_pipeline(data_path: str = "data/OASIS_cross_tbl_df.csv", n_iter: int = 25) -> dict:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_oasis_data(data_path)
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    tuned = tune_models(X_train, y_train, n_iter=n_iter)
    estimators = {name: tuned_model.best_estimator for name, tuned_model in tuned.items()}

    results = {}
    for name, model in estimators.items():
        threshold_info = tune_decision_threshold_cv(model, X_train, y_train)
        results[name] = {
            "cv": cross_validate_model(model, X_train, y_train),
            "test": evaluate_model(model, X_test, y_test, threshold=threshold_info["threshold"]),
            "threshold_tuning": threshold_info,
            "best_params": tuned[name].best_params,
            "search_best_roc_auc": tuned[name].best_score,
        }

    z_scores = z_score_table(df)
    z_validation = validate_z_scores(z_scores)
    age_volume_corr = age_brain_volume_correlation(df)

    plot_paths = {
        "roc": str(plot_roc_curves(estimators, X_test, y_test)),
        "metric_bars": str(plot_metric_bars(results)),
        "feature_importance": str(plot_feature_importance(estimators["RandomForest"])),
        "correlation_heatmap": str(plot_correlation_heatmap(df)),
        "confusion_matrices": {k: str(v) for k, v in plot_confusion_matrices(results).items()},
    }

    comparison = compare_models(results)
    winner = best_model_name(results)

    print_cv_results(results)
    print_test_results(results)
    print("\nModel comparison")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest-performing model by test ROC-AUC: {winner}")
    print("\nZ-score distribution checks")
    print(z_validation.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nAge vs nWBV correlation: {age_volume_corr:.3f}")

    serializable = {
        "comparison": comparison.to_dict(orient="records"),
        "best_model": winner,
        "z_score_validation": z_validation.to_dict(orient="records"),
        "age_brain_volume_correlation": age_volume_corr,
        "plot_paths": plot_paths,
        "best_params": {name: results[name]["best_params"] for name in results},
        "threshold_tuning": {name: results[name]["threshold_tuning"] for name in results},
    }
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return results


if __name__ == "__main__":
    run_pipeline()
