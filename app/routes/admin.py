# ==============================================================================
# SmartClean — Admin Routes
# ==============================================================================

import logging

from flask import Blueprint, render_template, request, jsonify, session

from ..models.user    import get_user_by_id, toggle_user_status, delete_user, log_activity
from ..models.history import get_admin_statistics
from ..utils.security import admin_required

security_logger = logging.getLogger("security")
bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("/")
@admin_required
def dashboard():
    return render_template("admin_dashboard.html", user=get_user_by_id(session["user_id"]))


@bp.route("/statistics")
@admin_required
def statistics():
    return jsonify(get_admin_statistics())


@bp.route("/toggle_user/<int:user_id>", methods=["POST"])
@admin_required
def toggle_user(user_id: int):
    try:
        new_status = toggle_user_status(user_id)
        action = "ACTIVATE_USER" if new_status else "DEACTIVATE_USER"
        log_activity(session["user_id"], action, f"User {user_id}")
        return jsonify({"success": True, "new_status": new_status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/delete_user/<int:user_id>", methods=["DELETE"])
@admin_required
def remove_user(user_id: int):
    if user_id == session["user_id"]:
        return jsonify({"success": False, "error": "Cannot delete your own account"}), 400
    try:
        delete_user(user_id)
        log_activity(session["user_id"], "DELETE_USER", f"User {user_id} deleted")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
