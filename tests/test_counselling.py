from src.counselling import build_recommendation, risk_level_from_probability


def test_risk_level_mapping():
    assert risk_level_from_probability(0.20) == "Low"
    assert risk_level_from_probability(0.45) == "Medium"
    assert risk_level_from_probability(0.85) == "High"


def test_recommendation_engine_for_risk_signals():
    student = {
        "attendance_pct": 60,
        "current_gpa": 5.25,
        "failed_subjects": 2,
        "engagement_score": 40,
        "commute_minutes": 80,
        "fee_payment_delay_days": 25,
    }
    recs = build_recommendation(student)
    assert len(recs) >= 3
    assert any("Attendance" in item["title"] for item in recs)
    assert any("Academic" in item["title"] for item in recs)
