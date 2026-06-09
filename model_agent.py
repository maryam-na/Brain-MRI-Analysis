from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from data_agent import RANDOM_STATE, build_preprocessor


@dataclass
class TunedModel:
    name: str
    search: RandomizedSearchCV
    best_estimator: Pipeline
    best_params: dict
    best_score: float


def build_model_pipelines(X_train: pd.DataFrame, random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor(X_train)
    return {
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def parameter_distributions() -> Dict[str, dict]:
    """Regularized search spaces to reduce overfitting on a small dataset."""
    return {
        "RandomForest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 8, 10],
            "model__min_samples_split": [5, 10, 15],
            "model__min_samples_leaf": [2, 4, 6],
            "model__max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "model__n_estimators": [50, 100, 150],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_split": [5, 10, 15],
            "model__min_samples_leaf": [2, 4, 6],
            "model__subsample": [0.6, 0.7, 0.8],
        },
    }


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
    random_state: int = RANDOM_STATE,
    scoring: str = "roc_auc",
) -> Dict[str, TunedModel]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    pipelines = build_model_pipelines(X_train, random_state=random_state)
    grids = parameter_distributions()
    tuned_models: Dict[str, TunedModel] = {}

    for name, pipeline in pipelines.items():
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=grids[name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        tuned_models[name] = TunedModel(
            name=name,
            search=search,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_score=float(search.best_score_),
        )

    return tuned_models


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        return [f"feature_{i}" for i in range(model.named_steps["model"].n_features_in_)]
