from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
PAGES_DIR = BASE_DIR / "pages"

DATASET_PATH = DATA_DIR / "sih2026_student_dropout_10000.csv"
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "eduguard.db")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "dropout_model.joblib")))
METADATA_PATH = MODELS_DIR / "model_metadata.json"
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "dev-secret")

FEATURE_COLUMNS = [
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
]
TEMPLATE_COLUMNS = FEATURE_COLUMNS + ["student_id"]
TARGET_COLUMN = "dropout_next_semester"

ROLE_PERMISSIONS = {
    "ADMIN": {"manage_users", "view_analytics", "manage_model", "export_reports"},
    "FACULTY": {"view_students", "review_predictions", "refer_support", "track_interventions"},
    "COUNSELLOR": {"view_cases", "create_interventions", "update_status", "record_followups"},
    "STUDENT": {"view_own_profile", "view_own_recommendations", "request_support"},
}
