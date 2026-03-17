# ==============================================================================
# SmartClean — Security Utilities
# ==============================================================================

import time
import secrets
import logging
import threading
from collections import defaultdict
from functools import wraps
from flask import session, request, flash, redirect, url_for

security_logger = logging.getLogger("security")

# ── CSRF ───────────────────────────────────────────────────────────────────────

def generate_csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def validate_csrf() -> bool:
    token         = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    session_token = session.get("csrf_token")
    valid         = bool(token and session_token and token == session_token)
    if not valid:
        security_logger.warning(
            f"CSRF invalid — IP: {get_client_ip()}, endpoint: {request.endpoint}"
        )
    return valid


def csrf_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "POST" and not validate_csrf():
            flash("Session expired. Please try again.", "danger")
            return redirect(request.referrer or url_for("auth.login"))
        return f(*args, **kwargs)
    return wrapper


# ── Rate limiting ──────────────────────────────────────────────────────────────

_rl_store: dict = defaultdict(list)
_rl_lock        = threading.Lock()

RATE_LIMITS = {
    "login":    (5,  60),    # 5 attempts / 60 s
    "register": (3,  300),   # 3 registrations / 5 min
    "preview":  (20, 60),    # 20 uploads / min
}


def is_rate_limited(ip: str, endpoint: str) -> bool:
    if endpoint not in RATE_LIMITS:
        return False
    max_calls, window = RATE_LIMITS[endpoint]
    now = time.time()
    key = f"{ip}:{endpoint}"
    with _rl_lock:
        _rl_store[key] = [t for t in _rl_store[key] if now - t < window]
        if len(_rl_store[key]) >= max_calls:
            security_logger.warning(f"Rate limit — IP: {ip}, endpoint: {endpoint}")
            return True
        _rl_store[key].append(now)
    return False


# ── Client IP ──────────────────────────────────────────────────────────────────

def get_client_ip() -> str:
    return (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.remote_addr
        or "unknown"
    )


# ── File validation ────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "xml", "tsv", "parquet"}

_FILE_SIGNATURES = {
    "xlsx": b"PK\x03\x04",
    "xls":  b"\xd0\xcf\x11\xe0",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file_content(file, extension: str) -> bool:
    """Verify that the file content matches the declared extension."""
    sig = _FILE_SIGNATURES.get(extension)
    if sig:
        header = file.read(len(sig))
        file.seek(0)
        return header == sig
    # Text file: verify encoding
    header = file.read(512)
    file.seek(0)
    for enc in ("utf-8", "latin-1", "utf-16"):
        try:
            header.decode(enc)
            return True
        except (UnicodeDecodeError, Exception):
            continue
    return False


# ── Auth decorators ────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        if session.get("role") != "admin":
            flash("Administrator access required.", "danger")
            return redirect(url_for("main.index"))
        return f(*args, **kwargs)
    return wrapper
