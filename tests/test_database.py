from src.database import SessionLocal, User, create_db_and_tables
from src.seed_database import seed_demo_users


def test_database_initializes_and_seeds_demo_users():
    create_db_and_tables()
    seed_demo_users()
    with SessionLocal() as session:
        assert session.query(User).count() >= 4
        names = [user.username for user in session.query(User).all()]
        assert "admin" in names
        assert "student" in names
