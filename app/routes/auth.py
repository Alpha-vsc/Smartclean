# ==============================================================================
# SmartClean — Authentication Routes
# ==============================================================================

import secrets
from flask import Blueprint, render_template, request, redirect, url_for, flash, session

from ..models.user        import (get_user_by_username, get_user_by_username_or_email,
                                  create_user, activate_user, update_last_login,
                                  update_password, get_user_by_id,
                                  create_verification_code, verify_code, log_activity)
from ..services.email_service import send_verification_email, send_new_code_email
from ..utils.security     import csrf_required, is_rate_limited, get_client_ip, login_required
from werkzeug.security    import check_password_hash
import re

bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["GET", "POST"])
@csrf_required
def login():
    if request.method == "POST":
        ip = get_client_ip()

        if is_rate_limited(ip, "login"):
            flash("Too many attempts. Please wait 1 minute.", "danger")
            return render_template("login.html"), 429

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please fill in all fields.", "danger")
            return render_template("login.html")

        user = get_user_by_username(username)

        if user and check_password_hash(user["password_hash"], password):
            if not user["is_active"]:
                session["pending_verify_user_id"] = user["id"]
                session["pending_verify_email"]   = user["email"]
                flash("Please verify your email before logging in.", "warning")
                return redirect(url_for("auth.verify_email_page"))

            # Prevent session fixation
            old_csrf = session.get("csrf_token")
            session.clear()
            session["csrf_token"] = old_csrf or secrets.token_hex(32)
            session["user_id"]    = user["id"]
            session["username"]   = user["username"]
            session["role"]       = user["role"]
            session.permanent     = True

            update_last_login(user["id"])
            log_activity(user["id"], "LOGIN", f"Login from {ip}", ip)
            flash(f"Welcome, {user['full_name'] or user['username']}!", "success")
            return redirect(url_for("admin.dashboard") if user["role"] == "admin" else url_for("main.index"))

        flash("Invalid credentials.", "danger")

    return render_template("login.html")


@bp.route("/register", methods=["GET", "POST"])
@csrf_required
def register():
    if request.method == "POST":
        ip = get_client_ip()

        if is_rate_limited(ip, "register"):
            flash("Too many registrations from this address. Try again in 5 minutes.", "danger")
            return render_template("register.html"), 429

        username  = request.form.get("username", "").strip()
        email     = request.form.get("email", "").strip()
        password  = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "danger")
            return redirect(url_for("auth.register"))
        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            flash("Username may only contain letters, digits and _.", "danger")
            return redirect(url_for("auth.register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("auth.register"))

        if get_user_by_username_or_email(username, email):
            flash("Username or email already in use.", "danger")
            return redirect(url_for("auth.register"))

        user_id = create_user(username, email, password, full_name)
        log_activity(user_id, "REGISTER", "New account pending verification", ip)

        code = create_verification_code(user_id, purpose="verify")
        ok   = send_verification_email(email, code)

        if not ok:
            # SMTP not configured → activate directly (dev mode)
            activate_user(user_id)
            flash("Account created! (SMTP not configured — account activated directly)", "success")
            return redirect(url_for("auth.login"))

        session["pending_verify_user_id"] = user_id
        session["pending_verify_email"]   = email
        flash(f"A verification code has been sent to {email}.", "info")
        return redirect(url_for("auth.verify_email_page"))

    return render_template("register.html")


@bp.route("/verify-email", methods=["GET", "POST"])
@csrf_required
def verify_email_page():
    user_id = session.get("pending_verify_user_id")
    email   = session.get("pending_verify_email", "")

    if not user_id:
        flash("Session expired. Please register again.", "danger")
        return redirect(url_for("auth.register"))

    if request.method == "POST":
        code = request.form.get("code", "").strip()
        if verify_code(user_id, code, purpose="verify"):
            activate_user(user_id)
            log_activity(user_id, "EMAIL_VERIFIED", "Account activated")
            session.pop("pending_verify_user_id", None)
            session.pop("pending_verify_email", None)
            flash("Email verified! You can now log in.", "success")
            return redirect(url_for("auth.login"))
        flash("Invalid or expired code. Please try again.", "danger")

    return render_template("verify_email.html", email=email, user_id=user_id)


@bp.route("/resend-verification", methods=["POST"])
@csrf_required
def resend_verification():
    user_id = session.get("pending_verify_user_id")
    email   = session.get("pending_verify_email", "")
    if not user_id or not email:
        flash("Session expired.", "danger")
        return redirect(url_for("auth.register"))

    code = create_verification_code(user_id, purpose="verify")
    ok   = send_new_code_email(email, code)
    flash(f"New code sent to {email}." if ok else "SMTP error. Check your .env config.", "info" if ok else "danger")
    return redirect(url_for("auth.verify_email_page"))


@bp.route("/logout")
def logout():
    if "user_id" in session:
        log_activity(session["user_id"], "LOGOUT", "Logout", get_client_ip())
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/change_password", methods=["POST"])
@login_required
@csrf_required
def change_password():
    current_pw = request.form.get("current_password", "")
    new_pw     = request.form.get("new_password", "")
    confirm_pw = request.form.get("confirm_password", "")
    user       = get_user_by_id(session["user_id"])

    if not check_password_hash(user["password_hash"], current_pw):
        flash("Current password is incorrect.", "danger")
        return redirect(url_for("main.settings"))
    if len(new_pw) < 6:
        flash("New password must be at least 6 characters.", "danger")
        return redirect(url_for("main.settings"))
    if new_pw != confirm_pw:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("main.settings"))

    update_password(session["user_id"], new_pw)
    log_activity(session["user_id"], "PASSWORD_CHANGED", "Password changed", get_client_ip())
    flash("Password changed successfully.", "success")
    return redirect(url_for("main.settings"))


@bp.route("/send_verification_email", methods=["POST"])
@login_required
@csrf_required
def send_verification_email_route():
    user = get_user_by_id(session["user_id"])
    if user.get("email_verified"):
        flash("Your email is already verified.", "info")
        return redirect(url_for("main.settings"))

    code = create_verification_code(session["user_id"], purpose="verify")
    ok   = send_new_code_email(user["email"], code)
    flash(
        f"Code sent to {user['email']}. Check your inbox." if ok
        else "Email error. Check SMTP config in .env.",
        "success" if ok else "danger",
    )
    return redirect(url_for("main.settings"))


@bp.route("/confirm_verification", methods=["POST"])
@login_required
@csrf_required
def confirm_verification():
    code = request.form.get("verification_code", "").strip()
    if verify_code(session["user_id"], code, purpose="verify"):
        from ..models.database import get_db
        db = get_db()
        db.execute("UPDATE users SET email_verified=1 WHERE id=?", (session["user_id"],))
        db.commit()
        log_activity(session["user_id"], "EMAIL_VERIFIED", "Email verified")
        flash("Email verified successfully!", "success")
    else:
        flash("Invalid or expired code.", "danger")
    return redirect(url_for("main.settings"))
