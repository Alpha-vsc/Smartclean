# ==============================================================================
# SmartClean — User Model
# ==============================================================================

import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash
from .database import get_db

security_logger = logging.getLogger("security")


# ── Read ───────────────────────────────────────────────────────────────────────

def get_user_by_id(user_id: int) -> Optional[dict]:
    row = get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[dict]:
    row = get_db().execute(
        "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
    ).fetchone()
    return dict(row) if row else None


def get_user_by_username_or_email(username: str, email: str) -> Optional[dict]:
    row = get_db().execute(
        "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
    ).fetchone()
    return dict(row) if row else None


# ── Write ──────────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str, full_name: str) -> int:
    db = get_db()
    db.execute(
        "INSERT INTO users (username, email, password_hash, full_name, role, is_active, email_verified)"
        " VALUES (?,?,?,?,?,?,?)",
        (username, email, generate_password_hash(password), full_name, "user", 0, 0),
    )
    db.commit()
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def activate_user(user_id: int) -> None:
    db = get_db()
    db.execute("UPDATE users SET is_active=1, email_verified=1 WHERE id=?", (user_id,))
    db.commit()


def update_last_login(user_id: int) -> None:
    db = get_db()
    db.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_id))
    db.commit()


def update_password(user_id: int, new_password: str) -> None:
    db = get_db()
    db.execute(
        "UPDATE users SET password_hash=? WHERE id=?",
        (generate_password_hash(new_password), user_id),
    )
    db.commit()


def toggle_user_status(user_id: int) -> int:
    db  = get_db()
    row = db.execute("SELECT is_active FROM users WHERE id=?", (user_id,)).fetchone()
    new_status = 0 if row["is_active"] else 1
    db.execute("UPDATE users SET is_active=? WHERE id=?", (new_status, user_id))
    db.commit()
    return new_status


def delete_user(user_id: int) -> None:
    db = get_db()
    for table in ["processing_history", "activity_logs", "user_settings", "email_verifications"]:
        db.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
    db.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.commit()


# ── Settings ───────────────────────────────────────────────────────────────────

def get_user_settings(user_id: int) -> dict:
    db  = get_db()
    row = db.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        db.execute("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))
        db.commit()
        row = db.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row)


def save_user_settings(user_id: int, data: dict) -> None:
    db = get_db()
    db.execute(
        """
        INSERT INTO user_settings
            (user_id, always_preview, save_history, confirm_delete,
             default_missing, default_duplicates, default_outliers, default_normalize,
             default_format, preview_rows)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            always_preview=excluded.always_preview,
            save_history=excluded.save_history,
            confirm_delete=excluded.confirm_delete,
            default_missing=excluded.default_missing,
            default_duplicates=excluded.default_duplicates,
            default_outliers=excluded.default_outliers,
            default_normalize=excluded.default_normalize,
            default_format=excluded.default_format,
            preview_rows=excluded.preview_rows
        """,
        (
            user_id,
            data.get("always_preview", True),
            data.get("save_history", True),
            data.get("confirm_delete", True),
            data.get("default_missing", False),
            data.get("default_duplicates", False),
            data.get("default_outliers", False),
            data.get("default_normalize", False),
            data.get("default_format", "csv"),
            data.get("preview_rows", 10),
        ),
    )
    db.commit()


# ── Email verification ─────────────────────────────────────────────────────────

def create_verification_code(user_id: int, purpose: str = "verify") -> str:
    code    = str(secrets.randbelow(900_000) + 100_000)
    expires = datetime.now() + timedelta(minutes=15)
    db = get_db()
    db.execute(
        "UPDATE email_verifications SET used=1 WHERE user_id=? AND purpose=? AND used=0",
        (user_id, purpose),
    )
    db.execute(
        "INSERT INTO email_verifications (user_id, code, purpose, expires_at) VALUES (?,?,?,?)",
        (user_id, code, purpose, expires),
    )
    db.commit()
    return code


def verify_code(user_id: int, code: str, purpose: str = "verify") -> bool:
    db  = get_db()
    row = db.execute(
        "SELECT id FROM email_verifications"
        " WHERE user_id=? AND code=? AND purpose=? AND used=0 AND expires_at > ?",
        (user_id, code, purpose, datetime.now()),
    ).fetchone()
    if row:
        db.execute("UPDATE email_verifications SET used=1 WHERE id=?", (row["id"],))
        db.commit()
    return row is not None


# ── Activity logging ───────────────────────────────────────────────────────────

def log_activity(user_id: int, action: str, details: str = None, ip: str = None) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO activity_logs (user_id, action, details, ip_address) VALUES (?,?,?,?)",
        (user_id, action, details, ip),
    )
    db.commit()
