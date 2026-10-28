from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FEATURE_COLUMNS

NUMERIC_FEATURES = [
    "age",
    "year_of_study",
    "attendance_pct",
    "internal_marks",
    "assignment_completion_pct",
    "previous_gpa",
    "current_gpa",
    "failed_subjects",
    "study_hours_weekly",
    "commute_minutes",
    "fee_payment_delay_days",
    "engagement_score",
    "counselling_sessions",
]
CATEGORICAL_FEATURES = [
    "department",
    "family_income_band",
]


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"Missing required feature column: {column}")
    return frame[FEATURE_COLUMNS]
