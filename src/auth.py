from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import streamlit as st
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

from .config import ROLE_PERMISSIONS

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MAX_BCRYPT_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    if not password or len(password.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
        raise ValueError("Password must contain 1 to 72 UTF-8 bytes.")
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password or len(password.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
        return False
    try:
        return pwd_context.verify(password, password_hash)
    except (UnknownHashError, ValueError):
        return False


def login_user(username: str, password: str) -> bool:
    from .database import SessionLocal, User

    normalized_username = username.strip().casefold()
    if not normalized_username or not password:
        return False

    with SessionLocal() as session:
        user = session.query(User).filter(User.username == normalized_username).first()
        if not user or not user.is_active or not verify_password(password, user.password_hash):
            return False
        st.session_state["authenticated"] = True
        st.session_state["user"] = {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "full_name": user.full_name,
            "department": user.department,
        }
        return True


def logout_user() -> None:
    for key in ["authenticated", "user"]:
        st.session_state.pop(key, None)


def require_role(*roles: str) -> Callable:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not st.session_state.get("authenticated"):
                st.warning("Please log in to continue.")
                st.stop()
            user_role = st.session_state.get("user", {}).get("role")
            if user_role not in roles:
                st.error("You do not have permission to access this section.")
                st.stop()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def user_has_permission(permission: str) -> bool:
    user = st.session_state.get("user") or {}
    role = user.get("role")
    permitted = ROLE_PERMISSIONS.get(role, set())
    return permission in permitted


def require_permission(permission: str) -> Callable:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not user_has_permission(permission):
                st.error("You do not have the required authorization for this action.")
                st.stop()
            return func(*args, **kwargs)

        return wrapper

    return decorator
