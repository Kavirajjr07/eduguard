from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN

EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


class DataValidationError(ValueError):
    pass


def validate_dataset_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    required_columns = EXPECTED_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")

    if "student_id" in df.columns and df.duplicated(subset=["student_id"]).any():
        duplicates = df.loc[df.duplicated(subset=["student_id"], keep=False), "student_id"].nunique()
        raise DataValidationError(f"Duplicate student IDs detected ({duplicates} unique duplicates).")

    if df.isnull().any().any():
        null_columns = df.columns[df.isnull().any()].tolist()
        raise DataValidationError(f"Missing values found in: {null_columns}")

    numeric_checks = {
        "age": df["age"].between(15, 40),
        "year_of_study": df["year_of_study"].between(1, 6),
        "attendance_pct": df["attendance_pct"].between(0, 100),
        "internal_marks": df["internal_marks"].between(0, 100),
        "assignment_completion_pct": df["assignment_completion_pct"].between(0, 100),
        "previous_gpa": df["previous_gpa"].between(0.0, 10.0),
        "current_gpa": df["current_gpa"].between(0.0, 10.0),
        "failed_subjects": df["failed_subjects"].between(0, 8),
        "study_hours_weekly": df["study_hours_weekly"].between(0, 50),
        "commute_minutes": df["commute_minutes"].between(0, 180),
        "fee_payment_delay_days": df["fee_payment_delay_days"].between(0, 180),
        "engagement_score": df["engagement_score"].between(0, 100),
        "counselling_sessions": df["counselling_sessions"].between(0, 20),
    }
    if require_target:
        numeric_checks["dropout_next_semester"] = df["dropout_next_semester"].isin([0, 1])

    invalid = {name: int((~mask).sum()) for name, mask in numeric_checks.items() if not mask.all()}
    if invalid:
        details = ", ".join(f"{key}={count}" for key, count in invalid.items())
        raise DataValidationError(f"Invalid numeric ranges found: {details}")

    allowed_departments = {"CSE", "ECE", "ME", "EEE", "Civil", "IT", "Biotech", "MBA", "Law", "Arts"}
    invalid_departments = sorted(set(df["department"]) - allowed_departments)
    if invalid_departments:
        raise DataValidationError(f"Unexpected department values: {invalid_departments[:10]}")

    if "student_id" in df.columns and not df["student_id"].astype(str).str.fullmatch(r"STU\d{4,}").all():
        raise DataValidationError("student_id values must match the pattern STU#######.")

    if not df["first_generation_student"].isin([0, 1]).all():
        raise DataValidationError("first_generation_student must be 0/1")
    if not df["scholarship"].isin([0, 1]).all():
        raise DataValidationError("scholarship must be 0/1")

    return df.copy()


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    return validate_dataset_dataframe(df)


def validate_student_record(student: dict[str, Any]) -> dict[str, Any]:
    required = {
        "student_id",
        "age",
        "year_of_study",
        "department",
        "attendance_pct",
        "internal_marks",
        "assignment_completion_pct",
        "previous_gpa",
        "current_gpa",
        "failed_subjects",
        "study_hours_weekly",
        "commute_minutes",
        "family_income_band",
        "first_generation_student",
        "scholarship",
        "fee_payment_delay_days",
        "engagement_score",
        "counselling_sessions",
    }
    missing = sorted(required - set(student.keys()))
    if missing:
        raise DataValidationError(f"Missing required fields: {missing}")
    return validate_dataset_dataframe(pd.DataFrame([student]), require_target=False).to_dict(orient="records")[0]


def validate_csv_upload(file_like: Any) -> pd.DataFrame:
    if file_like is None:
        raise DataValidationError("No file was uploaded.")
    try:
        df = pd.read_csv(file_like)
    except Exception as exc:  # pragma: no cover
        raise DataValidationError(f"Could not parse the uploaded CSV: {exc}") from exc
    if df.empty:
        raise DataValidationError("Uploaded CSV is empty.")
    return validate_dataset_dataframe(df, require_target=False)
