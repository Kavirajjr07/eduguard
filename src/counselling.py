from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def risk_level_from_probability(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def get_thresholds() -> dict[str, float]:
    return {"low": 0.30, "medium": 0.60}


def build_recommendation(student: dict[str, Any]) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if student.get("attendance_pct", 100) < 75:
        recommendations.append({
            "title": "Attendance improvement plan",
            "action": "Create a 30-day attendance recovery plan with faculty mentoring and check-ins.",
            "priority": "High",
        })
    if student.get("current_gpa", 10.0) < 6.875:
        recommendations.append({
            "title": "Academic tutoring",
            "action": "Assign remedial support and a weekly study plan for core subjects.",
            "priority": "High",
        })
    if student.get("failed_subjects", 0) > 1:
        recommendations.append({
            "title": "Subject-specific tutoring",
            "action": "Schedule revision sessions for failed courses and monitor weekly progress.",
            "priority": "High",
        })
    if student.get("engagement_score", 100) < 55:
        recommendations.append({
            "title": "Mentor and engagement support",
            "action": "Assign a peer mentor and encourage participation in clubs or learning groups.",
            "priority": "Medium",
        })
    if student.get("commute_minutes", 0) > 60:
        recommendations.append({
            "title": "Timetable flexibility consultation",
            "action": "Review transport constraints and explore timetable adjustments with the department.",
            "priority": "Medium",
        })
    if student.get("fee_payment_delay_days", 0) > 15:
        recommendations.append({
            "title": "Financial aid guidance",
            "action": "Refer the student to scholarship and fee support consultation with confidential handling.",
            "priority": "High",
        })
    if not recommendations:
        recommendations.append({
            "title": "Positive support plan",
            "action": "Continue regular monitoring and recognition of academic strengths.",
            "priority": "Low",
        })
    return recommendations


def build_case_summary(student: dict[str, Any], probability: float) -> dict[str, Any]:
    level = risk_level_from_probability(probability)
    recommendations = build_recommendation(student)
    return {
        "risk_level": level,
        "recommended_actions": recommendations,
        "support_note": "Support plans are non-punitive and must be reviewed by authorized staff before action.",
    }


def intervention_status_options() -> list[str]:
    return ["Pending", "Scheduled", "In Progress", "Completed", "Follow-up Required"]


def next_followup_date(days: int = 14) -> str:
    return (date.today() + timedelta(days=days)).isoformat()
