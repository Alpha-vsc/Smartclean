# ==============================================================================
# SmartClean — Data Analysis Routes
# ==============================================================================

import uuid
import logging

from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename

from ..services.analyzer import analyze_dataframe
from ..services.file_io  import read_file
from ..utils.security    import csrf_required, login_required, allowed_file
from ..models.user       import log_activity

security_logger = logging.getLogger("security")
bp = Blueprint("data", __name__)


@bp.route("/analyze", methods=["POST"])
@login_required
@csrf_required
def analyze():
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No file provided"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported format"}), 400

        safe_name = secure_filename(file.filename)
        file_ext  = safe_name.rsplit(".", 1)[1].lower()

        df      = read_file(file, file_ext)
        summary = analyze_dataframe(df, safe_name)

        log_activity(session["user_id"], "ANALYZE", f"Analysis of {safe_name}", ip)
        return jsonify(summary)

    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] analyze: {e}", exc_info=True)
        return jsonify({"error": f"Error ({error_id})"}), 500
