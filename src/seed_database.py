from __future__ import annotations

import os

from .auth import hash_password
from .database import SessionLocal, User, create_db_and_tables

DEMO_USERS = [
    {
        "username": "admin",
        "email": "admin@eduguard.ai",
        "password": os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123"),
        "role": "ADMIN",
        "full_name": "Dr. Ananya Sharma",
        "department": "Administration",
    },
    {
        "username": "faculty",
        "email": "faculty@eduguard.ai",
        "password": os.getenv("DEFAULT_FACULTY_PASSWORD", "faculty123"),
        "role": "FACULTY",
        "full_name": "Prof. Nikhil Rao",
        "department": "Computer Science",
    },
    {
        "username": "counsellor",
        "email": "counsellor@eduguard.ai",
        "password": os.getenv("DEFAULT_COUNSELLOR_PASSWORD", "counsellor123"),
        "role": "COUNSELLOR",
        "full_name": "Meera Iyer",
        "department": "Student Support",
    },
    {
        "username": "student",
        "email": "student@eduguard.ai",
        "password": os.getenv("DEFAULT_STUDENT_PASSWORD", "student123"),
        "role": "STUDENT",
        "full_name": "Riya Sethi",
        "department": "Computer Science",
    },
]


def seed_demo_users() -> None:
    create_db_and_tables()
    with SessionLocal() as session:
        for user_info in DEMO_USERS:
            existing = session.query(User).filter(User.username == user_info["username"]).first()
            if existing:
                continue
            user = User(
                username=user_info["username"],
                email=user_info["email"],
                password_hash=hash_password(user_info["password"]),
                role=user_info["role"],
                full_name=user_info["full_name"],
                department=user_info["department"],
                is_active=True,
            )
            session.add(user)
        session.commit()


if __name__ == "__main__":
    seed_demo_users()
    print("Demo users seeded successfully.")
