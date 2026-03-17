# ==============================================================================
# SmartClean — Application Factory
# ==============================================================================

import os
import secrets
import logging
from datetime import timedelta
from flask import Flask
from .utils.security import generate_csrf_token


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    _configure_app(app)
    _configure_logging(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _configure_jinja(app)
    _configure_security_headers(app)

    # Initialize the database
    from .models.database import init_db
    with app.app_context():
        init_db()

    # Start background cleanup thread
    from .utils.cleanup import start_cleanup_thread
    start_cleanup_thread(app.config["UPLOAD_DIR"], app.config["TEMP_DIR"])

    return app


# ── Private helpers ────────────────────────────────────────────────────────────

def _configure_app(app: Flask) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    secret = os.environ.get("SECRET_KEY")
    if not secret:
        secret = secrets.token_hex(32)
        logging.getLogger("security").warning(
            "SECRET_KEY not set. Random key used — all sessions will be lost on restart. "
            "Set SECRET_KEY in your .env file."
        )

    app.secret_key = secret
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("HTTPS", "false") == "true",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50 MB

        # Paths
        BASE_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        PERSISTENT_DIR=os.environ.get("PERSISTENT_DISK_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))),
    )
    app.config["UPLOAD_DIR"] = os.path.join(app.config["PERSISTENT_DIR"], "uploads")
    app.config["TEMP_DIR"]   = os.path.join(app.config["PERSISTENT_DIR"], "temp_previews")
    app.config["DB_PATH"]    = os.path.join(app.config["PERSISTENT_DIR"], "smartclean.db")

    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["TEMP_DIR"],   exist_ok=True)


def _configure_logging(app: Flask) -> None:
    log_dir = app.config["PERSISTENT_DIR"]
    file_handler = logging.FileHandler(os.path.join(log_dir, "smartclean_security.log"))
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    security_logger = logging.getLogger("security")
    security_logger.addHandler(file_handler)
    security_logger.setLevel(logging.WARNING)

    logging.getLogger("werkzeug").setLevel(logging.INFO)


def _register_blueprints(app: Flask) -> None:
    from .routes.auth    import bp as auth_bp
    from .routes.main    import bp as main_bp
    from .routes.data    import bp as data_bp
    from .routes.admin   import bp as admin_bp
    from .routes.convert import bp as convert_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(convert_bp)


def _register_error_handlers(app: Flask) -> None:
    from flask import flash, redirect, url_for, render_template

    @app.errorhandler(413)
    def file_too_large(e):
        flash("Fichier trop volumineux. Taille maximale : 50 MB.", "danger")
        return redirect(url_for("main.index"))

    @app.errorhandler(404)
    def not_found(e):
        # Return JSON for API/fetch requests, HTML for browser navigation
        from flask import request as req, jsonify
        if req.path.startswith("/api/") or "application/json" in req.headers.get("Accept", ""):
            return jsonify({"error": "Not found", "path": req.path}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("errors/500.html"), 500

    # ── API stub routes ────────────────────────────────────────────────────────
    # Graceful stubs for endpoints called by the frontend that are not yet
    # implemented in the backend (prevents noisy 404s in the console).

    from flask import jsonify as _json

    @app.route("/api/notifications/unread-count")
    def api_notifications_unread():
        return _json({"count": 0})

    @app.route("/api/notifications")
    def api_notifications():
        return _json({"notifications": []})


def _configure_jinja(app: Flask) -> None:
    app.jinja_env.globals["csrf_token"] = generate_csrf_token


def _configure_security_headers(app: Flask) -> None:
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"]  = "nosniff"
        response.headers["X-Frame-Options"]          = "DENY"
        response.headers["X-XSS-Protection"]         = "1; mode=block"
        response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"]   = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
            "font-src 'self' cdnjs.cloudflare.com; "
            "img-src 'self' data:;"
        )
        return response
