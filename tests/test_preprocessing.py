import pandas as pd

from src.preprocessing import build_preprocessor, prepare_features


def test_preprocessor_shape_is_valid():
    df = pd.DataFrame([
        {
            "student_id": "STU000001",
            "age": 20,
            "year_of_study": 2,
            "department": "CSE",
            "attendance_pct": 80,
            "internal_marks": 72,
            "assignment_completion_pct": 90,
            "previous_gpa": 8.5,
            "current_gpa": 8.0,
            "failed_subjects": 1,
            "study_hours_weekly": 18,
            "commute_minutes": 35,
            "family_income_band": "Middle",
            "first_generation_student": 0,
            "scholarship": 1,
            "fee_payment_delay_days": 7,
            "engagement_score": 74,
            "counselling_sessions": 2,
        }
    ])
    feature_df = prepare_features(df)
    transformed = build_preprocessor().fit_transform(feature_df)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 1
