# ==============================================================================
# SmartClean — File Conversion Routes
# ==============================================================================

import os
import uuid
import logging

from flask import (Blueprint, current_app, render_template, request,
                   redirect, url_for, flash, session, send_file)
from werkzeug.utils import secure_filename

from ..services.file_io import read_file, write_file, EXPORT_FORMATS, SUPPORTED_FORMATS
from ..utils.security   import login_required, get_client_ip
from ..models.user      import log_activity

security_logger = logging.getLogger("security")
bp = Blueprint("convert", __name__)


@bp.route("/convert", methods=["GET", "POST"])
@login_required
def convert():
    if request.method == "GET":
        return render_template("convert.html")

    ip      = get_client_ip()
    tmp_path = None
    try:
        file       = request.files.get("file")
        target_fmt = request.form.get("target_format", "csv").lower()

        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(url_for("convert.convert"))
        if target_fmt not in EXPORT_FORMATS:
            flash("Unsupported target format.", "danger")
            return redirect(url_for("convert.convert"))

        safe_name = secure_filename(file.filename)
        src_ext   = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""

        if src_ext not in SUPPORTED_FORMATS:
            flash(f"Unsupported source format: .{src_ext}", "danger")
            return redirect(url_for("convert.convert"))

        # Save to temp file — pd.read_xml / parquet don't support Flask FileStorage directly
        tmp_path = os.path.join(current_app.config["TEMP_DIR"], f"{uuid.uuid4()}.{src_ext}")
        file.save(tmp_path)

        df = read_file(tmp_path, src_ext)

        base_name = safe_name.rsplit(".", 1)[0]
        out_name  = f"{base_name}_converted.{target_fmt}"
        out_path  = os.path.join(current_app.config["UPLOAD_DIR"], out_name)

        write_file(df, out_path, target_fmt)

        log_activity(session["user_id"], "CONVERT", f"{safe_name} -> {target_fmt}", ip)
        return send_file(out_path, as_attachment=True, download_name=out_name)

    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] convert: {e}", exc_info=True)
        flash(f"Conversion error (ref: {error_id}): {str(e)[:120]}", "danger")
        return redirect(url_for("convert.convert"))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
