from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
from .counselling import build_case_summary
from .explainability import build_risk_explanation, summarize_protective_factors, summarize_top_risk_factors
from .preprocessing import prepare_features


def load_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file is missing at {path}. Run python -m src.train_model first.")
    return joblib.load(path)


def load_metadata(metadata_path: str | Path = METADATA_PATH):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_student(student: dict[str, Any], model=None, metadata=None):
    if model is None:
        model = load_model()
    if metadata is None:
        metadata = load_metadata()
    frame = pd.DataFrame([student])
    feature_frame = prepare_features(frame)
    probability = float(model.predict_proba(feature_frame)[0, 1])
    level = "Low" if probability < 0.30 else "Medium" if probability < 0.60 else "High"
    risk_factors = summarize_top_risk_factors(frame.iloc[0])
    protective_factors = summarize_protective_factors(frame.iloc[0])
    explanation = build_risk_explanation(frame.iloc[0], metadata.get("selected_model", "unknown"))
    support_summary = build_case_summary(frame.iloc[0], probability)
    return {
        "student_id": frame.iloc[0].get("student_id", "UNKNOWN"),
        "probability": probability,
        "risk_level": level,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "intervention": support_summary["recommended_actions"][0],
        "explanation": explanation,
        "model_version": metadata.get("model_version", "unknown"),
        "assessment_date": __import__("datetime").datetime.utcnow().date().isoformat(),
    }


def bulk_predict(df: pd.DataFrame, model=None, metadata=None):
    if model is None:
        model = load_model()
    if metadata is None:
        metadata = load_metadata()
    records = []
    for _, row in df.iterrows():
        result = predict_student(row.to_dict(), model=model, metadata=metadata)
        records.append({**row.to_dict(), **result})
    result_df = pd.DataFrame(records)
    result_df["probability_pct"] = result_df["probability"] * 100
    return result_df
