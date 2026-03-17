# ==============================================================================
# SmartClean — Database Layer
# ==============================================================================

import os
import stat
import sqlite3
import secrets
import logging
from datetime import datetime
from flask import current_app, g
from werkzeug.security import generate_password_hash

security_logger = logging.getLogger("security")


# ── Connection management ──────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Return the per-request DB connection, creating it if necessary."""
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(e=None) -> None:
    """Close the per-request DB connection."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _ensure_db_writable() -> None:
    db_path = current_app.config["DB_PATH"]
    if os.path.exists(db_path) and not os.access(db_path, os.W_OK):
        try:
            os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        except Exception as e:
            raise PermissionError(
                f"Database is read-only: {db_path}\n"
                f"Linux/Mac: chmod 644 smartclean.db\nError: {e}"
            )


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT    UNIQUE NOT NULL,
        email         TEXT    UNIQUE NOT NULL,
        password_hash TEXT    NOT NULL,
        full_name     TEXT,
        role          TEXT    DEFAULT 'user',
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login    TIMESTAMP,
        is_active     BOOLEAN DEFAULT 1,
        email_verified BOOLEAN DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS email_verifications (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        code       TEXT    NOT NULL,
        purpose    TEXT    DEFAULT 'verify',
        expires_at TIMESTAMP,
        used       BOOLEAN DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );

    CREATE TABLE IF NOT EXISTS processing_history (
        id                       INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id                  INTEGER NOT NULL,
        filename                 TEXT    NOT NULL,
        original_rows            INTEGER,
        processed_rows           INTEGER,
        original_columns         INTEGER,
        processed_columns        INTEGER,
        missing_values_applied   BOOLEAN,
        duplicates_applied       BOOLEAN,
        outliers_applied         BOOLEAN,
        normalize_applied        BOOLEAN,
        output_format            TEXT,
        processing_date          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_size_kb             REAL,
        processing_time_seconds  REAL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );

    CREATE TABLE IF NOT EXISTS activity_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        action     TEXT    NOT NULL,
        details    TEXT,
        ip_address TEXT,
        timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );

    CREATE TABLE IF NOT EXISTS user_settings (
        user_id            INTEGER PRIMARY KEY,
        always_preview     BOOLEAN DEFAULT 1,
        save_history       BOOLEAN DEFAULT 1,
        confirm_delete     BOOLEAN DEFAULT 1,
        default_missing    BOOLEAN DEFAULT 0,
        default_duplicates BOOLEAN DEFAULT 0,
        default_outliers   BOOLEAN DEFAULT 0,
        default_normalize  BOOLEAN DEFAULT 0,
        default_format     TEXT    DEFAULT 'csv',
        preview_rows       INTEGER DEFAULT 10,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
"""


def init_db() -> None:
    """Create all tables and bootstrap the admin account."""
    _ensure_db_writable()
    db = get_db()
    db.executescript(_SCHEMA)
    db.commit()

    # Create admin if none exists
    if db.execute('SELECT COUNT(*) FROM users WHERE role = "admin"').fetchone()[0] == 0:
        pwd = os.environ.get("ADMIN_PASSWORD")
        if not pwd:
            pwd = secrets.token_urlsafe(16)
            print("=" * 60)
            print("⚠️  ADMIN_PASSWORD not set in .env")
            print(f"   Generated password: {pwd}")
            print("   Add ADMIN_PASSWORD=<your_password> to your .env")
            print("=" * 60)
            security_logger.warning("Admin created with a random password.")
        db.execute(
            "INSERT INTO users (username, email, password_hash, full_name, role, is_active, email_verified)"
            " VALUES (?,?,?,?,?,?,?)",
            ("admin", "admin@smartclean.com", generate_password_hash(pwd), "Administrator", "admin", 1, 1),
        )
        db.commit()
        print("✓ Admin account created — Username: admin")

    current_app.teardown_appcontext(close_db)
