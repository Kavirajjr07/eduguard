from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_risk_categories(df: pd.DataFrame) -> dict[str, int]:
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for value in df["risk_level"] if "risk_level" in df.columns else []:
        if value == "Low":
            counts["Low"] += 1
        elif value == "Medium":
            counts["Medium"] += 1
        elif value == "High":
            counts["High"] += 1
    return counts


def aggregate_by_department(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["department", "avg_risk", "student_count"])
    return (
        df.groupby("department", dropna=False)
        .agg(avg_risk=("probability", "mean"), student_count=("student_id", "count"))
        .reset_index()
    )


def risk_heatmap_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["department", "year_of_study"], dropna=False).agg(avg_risk=("probability", "mean")).reset_index()
    return grouped


def fairness_monitor(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"groups": {}, "summary": "No data available."}
    groups = []
    for dept, group in df.groupby("department"):
        group_avg = float(group["probability"].mean())
        groups.append({"department": dept, "average_risk": round(group_avg, 3), "n": int(len(group))})
    return {"groups": groups, "summary": "Model disparities should be interpreted cautiously and validated with institutional data."}
