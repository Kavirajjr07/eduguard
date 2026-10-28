from __future__ import annotations

from datetime import datetime
from typing import Any, Generator

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import APP_SECRET_KEY, DB_PATH

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default="STUDENT")
    full_name = Column(String(120), nullable=False)
    department = Column(String(120), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(120), nullable=False)
    age = Column(Integer, nullable=False)
    year_of_study = Column(Integer, nullable=False)
    department = Column(String(80), nullable=False)
    attendance_pct = Column(Integer, nullable=False)
    internal_marks = Column(Integer, nullable=False)
    assignment_completion_pct = Column(Integer, nullable=False)
    previous_gpa = Column(Numeric(5, 2), nullable=False)
    current_gpa = Column(Numeric(5, 2), nullable=False)
    failed_subjects = Column(Integer, nullable=False)
    study_hours_weekly = Column(Integer, nullable=False)
    commute_minutes = Column(Integer, nullable=False)
    family_income_band = Column(String(40), nullable=False)
    first_generation_student = Column(Boolean, nullable=False)
    scholarship = Column(Boolean, nullable=False)
    fee_payment_delay_days = Column(Integer, nullable=False)
    engagement_score = Column(Integer, nullable=False)
    counselling_sessions = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), ForeignKey("students.student_id"), nullable=False, index=True)
    probability = Column(Numeric(5, 4), nullable=False)
    risk_level = Column(String(20), nullable=False)
    model_version = Column(String(80), nullable=False)
    assessment_date = Column(Date, nullable=False, default=datetime.utcnow().date())
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    student = relationship("Student")


class CounsellingCase(Base):
    __tablename__ = "counselling_cases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), ForeignKey("students.student_id"), nullable=False, index=True)
    counsellor_username = Column(String(80), nullable=True)
    status = Column(String(30), default="Pending")
    priority = Column(String(20), default="Medium")
    created_date = Column(Date, default=datetime.utcnow().date())
    follow_up_date = Column(Date, nullable=True)
    notes = Column(Text, nullable=True)
    outcome = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    student = relationship("Student")


class Intervention(Base):
    __tablename__ = "interventions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), ForeignKey("students.student_id"), nullable=False, index=True)
    intervention_id = Column(String(80), unique=True, nullable=False)
    action = Column(Text, nullable=False)
    assigned_counsellor = Column(String(80), nullable=True)
    priority = Column(String(20), default="Medium")
    created_date = Column(Date, default=datetime.utcnow().date())
    follow_up_date = Column(Date, nullable=True)
    status = Column(String(30), default="Pending")
    counsellor_notes = Column(Text, nullable=True)
    outcome = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    student = relationship("Student")


class FollowUp(Base):
    __tablename__ = "follow_ups"
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("counselling_cases.id"), nullable=False)
    scheduled_date = Column(Date, nullable=False)
    notes = Column(Text, nullable=True)
    status = Column(String(30), default="Pending")
    outcome = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    actor = Column(String(80), nullable=False)
    action = Column(String(160), nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(80), nullable=False)
    version = Column(String(80), nullable=False)
    training_date = Column(DateTime, nullable=False)
    dataset_version = Column(String(80), nullable=False)
    metrics = Column(JSON, nullable=True)
    selected_threshold = Column(Numeric(5, 4), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


DB_PATH.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def create_db_and_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db_session() -> Generator[Any, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


__all__ = [
    "Base",
    "User",
    "Student",
    "Prediction",
    "CounsellingCase",
    "Intervention",
    "FollowUp",
    "AuditLog",
    "ModelVersion",
    "SessionLocal",
    "engine",
    "create_db_and_tables",
    "get_db_session",
]
