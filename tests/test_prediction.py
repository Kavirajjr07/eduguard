from src.config import MODEL_PATH
from src.predict import load_model, predict_student


def test_prediction_probability_range_and_keys():
    model = load_model(MODEL_PATH)
    student = {
        "student_id": "STU000999",
        "age": 20,
        "year_of_study": 2,
        "department": "CSE",
        "attendance_pct": 88,
        "internal_marks": 76,
        "assignment_completion_pct": 83,
        "previous_gpa": 8.75,
        "current_gpa": 8.5,
        "failed_subjects": 0,
        "study_hours_weekly": 15,
        "commute_minutes": 30,
        "family_income_band": "Middle",
        "first_generation_student": 0,
        "scholarship": 1,
        "fee_payment_delay_days": 3,
        "engagement_score": 81,
        "counselling_sessions": 1,
    }
    result = predict_student(student, model=model, metadata={"model_version": "test-v1", "selected_model": "random_forest"})
    assert 0.0 <= result["probability"] <= 1.0
    assert "risk_level" in result
    assert "risk_factors" in result
    assert "protective_factors" in result
