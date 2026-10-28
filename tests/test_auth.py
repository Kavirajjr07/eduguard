from src.auth import hash_password, login_user, verify_password
from src.seed_database import DEMO_USERS


def test_password_hash_round_trip():
    password = "S3cure!Pass"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("wrong", hashed)


def test_login_with_demo_user():
    assert login_user("admin", "admin123")


def test_all_seeded_demo_accounts_are_accessible():
    for demo_user in DEMO_USERS:
        assert login_user(f"  {demo_user['username'].upper()}  ", demo_user["password"])
        assert not login_user(demo_user["username"], "incorrect-password")


def test_invalid_or_oversized_password_is_rejected_without_exception():
    hashed = hash_password("valid-password")
    assert not verify_password("", hashed)
    assert not verify_password("x" * 73, hashed)
    assert not verify_password("valid-password", "not-a-valid-password-hash")
