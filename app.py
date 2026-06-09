from __future__ import annotations

import streamlit as st

from data_agent import load_oasis_data, prepare_dataset, split_data, validate_z_scores, z_score_table
from evaluation import (
    age_brain_volume_correlation,
    compare_models,
    cross_validate_model,
    evaluate_model,
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_metric_bars,
    plot_roc_curves,
)
from model_agent import tune_models


st.set_page_config(page_title="OASIS MRI Model Evaluation", layout="wide")


@st.cache_data
def load_data(data_path: str):
    return load_oasis_data(data_path)


@st.cache_resource
def train_and_evaluate(data_path: str, n_iter: int):
    df = load_oasis_data(data_path)
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    tuned = tune_models(X_train, y_train, n_iter=n_iter)
    estimators = {name: item.best_estimator for name, item in tuned.items()}

    results = {}
    for name, model in estimators.items():
        results[name] = {
            "cv": cross_validate_model(model, X_train, y_train),
            "test": evaluate_model(model, X_test, y_test),
            "best_params": tuned[name].best_params,
        }

    plot_paths = {
        "roc": plot_roc_curves(estimators, X_test, y_test),
        "bars": plot_metric_bars(results),
        "feature_importance": plot_feature_importance(estimators["RandomForest"]),
        "correlation": plot_correlation_heatmap(df),
        "confusion": plot_confusion_matrices(results),
    }
    z_validation = validate_z_scores(z_score_table(df))
    age_corr = age_brain_volume_correlation(df)
    comparison = compare_models(results)
    return df, results, comparison, plot_paths, z_validation, age_corr


st.title("OASIS MRI Volumetric ML Evaluation")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("CSV path", "data/OASIS_cross_tbl_df.csv")
    n_iter = st.slider("Randomized search iterations", min_value=5, max_value=50, value=25, step=5)
    selected_model = st.selectbox("Model", ["RandomForest", "GradientBoosting"])

df, results, comparison, plot_paths, z_validation, age_corr = train_and_evaluate(data_path, n_iter)

st.subheader("Selected Model Metrics")
metrics = results[selected_model]["test"]
cols = st.columns(5)
for col, metric in zip(cols, ["roc_auc", "sensitivity", "specificity", "accuracy", "f1"]):
    col.metric(metric.replace("_", " ").title(), f"{metrics[metric]:.3f}")

st.caption("Best hyperparameters")
st.json(results[selected_model]["best_params"])

left, right = st.columns([1.1, 0.9])
with left:
    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True)
    st.image(str(plot_paths["bars"]), caption="Metric comparison")

with right:
    st.subheader("ROC Curve")
    st.image(str(plot_paths["roc"]), caption="RandomForest vs GradientBoosting")

st.subheader("Confusion Matrix")
st.image(str(plot_paths["confusion"][selected_model]), caption=f"{selected_model} confusion matrix")

st.subheader("Clinical-Style Validation")
val_left, val_right = st.columns(2)
with val_left:
    st.write("Z-score distribution")
    st.dataframe(z_validation, use_container_width=True)
with val_right:
    st.metric("Age vs nWBV correlation", f"{age_corr:.3f}")
    st.image(str(plot_paths["correlation"]), caption="Correlation heatmap")

st.subheader("RandomForest Feature Importance")
st.image(str(plot_paths["feature_importance"]), caption="Top RandomForest features")

with st.expander("Dataset preview"):
    st.dataframe(df.head(50), use_container_width=True)
