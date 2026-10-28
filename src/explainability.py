from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS


def build_risk_explanation(row: pd.Series, model_name: str) -> str:
    # This is a lightweight, model-compatible explanation generator that relies on feature comparisons
    # with the cohort distribution rather than claiming causality.
    factor_messages = []
    row = row.copy()
    if row.get("attendance_pct", 100) < 75:
        factor_messages.append("low attendance")
    if row.get("current_gpa", 10.0) < 6.875:
        factor_messages.append("a lower GPA than the expected academic threshold")
    if row.get("failed_subjects", 0) > 1:
        factor_messages.append("multiple failed subjects")
    if row.get("engagement_score", 100) < 55:
        factor_messages.append("limited engagement")
    if row.get("fee_payment_delay_days", 0) > 15:
        factor_messages.append("payment delays and financial pressure")
    if row.get("commute_minutes", 0) > 60:
        factor_messages.append("long commute burden")
    if not factor_messages:
        factor_messages.append("stable academic engagement and consistent attendance")

    protective_messages = []
    if row.get("attendance_pct", 0) >= 85:
        protective_messages.append("strong attendance")
    if row.get("current_gpa", 0) >= 7.5:
        protective_messages.append("healthy GPA progression")
    if row.get("assignment_completion_pct", 0) >= 80:
        protective_messages.append("high assignment completion")
    if row.get("engagement_score", 0) >= 70:
        protective_messages.append("high engagement")
    if not protective_messages:
        protective_messages.append("supportive monitoring and early intervention")

    base = ", ".join(factor_messages[:3])
    protective = ", ".join(protective_messages[:3])
    return (
        f"{base.title()} are associated with elevated predicted dropout risk in this model. "
        f"Protective indicators such as {protective} suggest the student may respond well to structured support. "
        f"Predictions are estimates and can be wrong; they should be reviewed by staff."
    )


def compute_global_permutation_importance(model, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=42)
    importances = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return importances


def summarize_top_risk_factors(row: pd.Series) -> list[str]:
    factors = []
    if row.get("attendance_pct", 100) < 75:
        factors.append("Attendance below 75%")
    if row.get("current_gpa", 10.0) < 6.25:
        factors.append("Current GPA below 6.25/10")
    if row.get("failed_subjects", 0) > 1:
        factors.append("Multiple failed subjects")
    if row.get("engagement_score", 100) < 50:
        factors.append("Low engagement")
    if row.get("fee_payment_delay_days", 0) > 15:
        factors.append("Fee payment delay")
    if row.get("commute_minutes", 0) > 60:
        factors.append("Long commute burden")
    return factors[:4]


def summarize_protective_factors(row: pd.Series) -> list[str]:
    factors = []
    if row.get("attendance_pct", 0) >= 80:
        factors.append("Strong attendance")
    if row.get("current_gpa", 0) >= 7.5:
        factors.append("Healthy GPA")
    if row.get("assignment_completion_pct", 0) >= 80:
        factors.append("High assignment completion")
    if row.get("engagement_score", 0) >= 70:
        factors.append("Strong engagement")
    return factors[:4]
