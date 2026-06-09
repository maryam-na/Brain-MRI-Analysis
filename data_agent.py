from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Dementia"
DEFAULT_DATA_PATH = Path("data/OASIS_cross_tbl_df.csv")


def load_oasis_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load an OASIS cross-sectional CSV or create a documented demo fallback."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)

    return create_demo_oasis_like_data()


def create_demo_oasis_like_data(n_samples: int = 436) -> pd.DataFrame:
    """Create a deterministic OASIS-shaped demo table when the CSV is absent.

    This keeps the project runnable for structure and UI checks. Replace it with
    data/OASIS_cross_tbl_df.csv for real analysis.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(18, 97, n_samples)
    sex = rng.choice(["M", "F"], n_samples)
    educ = rng.normal(14, 3, n_samples).clip(6, 23).round()
    ses = rng.integers(1, 6, n_samples)
    etiv = rng.normal(1500, 150, n_samples) + np.where(sex == "M", 80, -40)
    nwbv = 0.86 - (age - 18) * 0.0017 + rng.normal(0, 0.025, n_samples)
    nwbv = nwbv.clip(0.62, 0.92)
    mmse = 30 - np.maximum(age - 60, 0) * 0.045 + rng.normal(0, 1.8, n_samples)
    cdr_risk = -8 + 0.08 * age - 8.0 * (nwbv - 0.72) - 0.25 * (mmse - 27)
    dementia = rng.binomial(1, 1 / (1 + np.exp(-cdr_risk)))
    cdr = np.where(dementia == 1, rng.choice([0.5, 1.0, 2.0], n_samples, p=[0.65, 0.3, 0.05]), 0.0)

    return pd.DataFrame(
        {
            "ID": [f"OAS1_{i:04d}" for i in range(n_samples)],
            "M/F": sex,
            "Hand": "R",
            "Age": age,
            "Educ": educ,
            "SES": ses,
            "MMSE": mmse.clip(12, 30).round(1),
            "CDR": cdr,
            "eTIV": etiv.round(1),
            "nWBV": nwbv.round(3),
            "ASF": (1.2 - (etiv - 1500) / 5000).round(3),
            "Delay": "",
        }
    )


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if TARGET_COLUMN not in df.columns:
        if "CDR" not in df.columns:
            raise ValueError("Expected either a Dementia target or CDR column.")
        df[TARGET_COLUMN] = (pd.to_numeric(df["CDR"], errors="coerce").fillna(0) > 0).astype(int)

    drop_columns = [col for col in ["ID", "Delay", "CDR", TARGET_COLUMN] if col in df.columns]
    X = df.drop(columns=drop_columns)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def z_score_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    selected = columns or [col for col in ["Age", "Educ", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"] if col in numeric]
    scores = numeric[selected].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    return scores.add_suffix("_z")


def validate_z_scores(z_scores: pd.DataFrame, tolerance: float = 0.15) -> pd.DataFrame:
    rows = []
    for column in z_scores.columns:
        mean = z_scores[column].mean()
        std = z_scores[column].std(ddof=0)
        rows.append(
            {
                "feature": column,
                "mean": mean,
                "std": std,
                "mean_ok": abs(mean) <= tolerance,
                "std_ok": abs(std - 1) <= tolerance,
            }
        )
    return pd.DataFrame(rows)
