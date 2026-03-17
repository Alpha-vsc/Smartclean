# ==============================================================================
# SmartClean — Main Routes
# ==============================================================================

import os
import time
import uuid
import logging

import pandas as pd
from flask import (Blueprint, current_app, render_template, request,
                   redirect, url_for, flash, session, jsonify, send_file)
from werkzeug.utils import secure_filename

from ..models.user    import get_user_by_id, get_user_settings, save_user_settings, log_activity
from ..models.history import save_to_history, get_history, get_statistics, delete_history_item, clear_history
from ..services.cleaner  import process_data
from ..services.file_io  import read_file, write_file, EXPORT_FORMATS
from ..utils.security    import (csrf_required, login_required, is_rate_limited,
                                  get_client_ip, allowed_file, validate_file_content)

security_logger = logging.getLogger("security")
bp = Blueprint("main", __name__)


# ── Index ──────────────────────────────────────────────────────────────────────

@bp.route("/")
@login_required
def index():
    user     = get_user_by_id(session["user_id"])
    settings = get_user_settings(session["user_id"])
    return render_template("Nyx_ax_v3.1.html", user=user, settings=settings)


# ── Preview & Clean ────────────────────────────────────────────────────────────

@bp.route("/preview", methods=["POST"])
@login_required
@csrf_required
def preview():
    start_time = time.time()
    ip         = get_client_ip()

    if is_rate_limited(ip, "preview"):
        flash("Too many files sent. Please wait a minute.", "danger")
        return redirect(url_for("main.index"))

    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(url_for("main.index"))

        if not allowed_file(file.filename):
            flash("Unsupported format. Accepted: CSV, Excel, JSON, XML.", "danger")
            return redirect(url_for("main.index"))

        safe_name = secure_filename(file.filename)
        file_ext  = safe_name.rsplit(".", 1)[1].lower()

        # Measure size
        file.seek(0, os.SEEK_END)
        file_size_kb = round(file.tell() / 1024, 2)
        file.seek(0)

        if not validate_file_content(file, file_ext):
            flash("File content does not match its extension.", "danger")
            security_logger.warning(f"Upload rejected (invalid content) — {safe_name}, IP: {ip}")
            return redirect(url_for("main.index"))

        config = {
            "duplicates":     "duplicates"     in request.form,
            "missing_values": "missing_values" in request.form,
            "outliers":       "outliers"        in request.form,
            "normalize":      "normalize"       in request.form,
            "format":         request.form.get("output_format", "csv"),
        }
        if config["format"] not in ("csv", "xlsx", "json"):
            config["format"] = "csv"

        df            = read_file(file, file_ext)
        original_shape = df.shape
        df_clean, changes = process_data(df.copy(), config)
        cleaned_shape = df_clean.shape

        # Persist cleaned df to disk (avoids 4 KB cookie limit)
        temp_id   = str(uuid.uuid4())
        temp_path = os.path.join(current_app.config["TEMP_DIR"], f"{temp_id}.pkl")
        df_clean.to_pickle(temp_path)
        session["preview_temp_id"]  = temp_id
        session["preview_format"]   = config["format"]
        session["preview_filename"] = safe_name

        stats = {
            "original_rows": original_shape[0],
            "cleaned_rows":  cleaned_shape[0],
            "removed_rows":  original_shape[0] - cleaned_shape[0],
            "columns":       cleaned_shape[1],
        }

        processing_time = round(time.time() - start_time, 2)
        save_to_history(
            {
                "filename":       safe_name,
                "original_rows":  original_shape[0],
                "processed_rows": cleaned_shape[0],
                "original_cols":  original_shape[1],
                "processed_cols": cleaned_shape[1],
                "missing":        config["missing_values"],
                "duplicates":     config["duplicates"],
                "outliers":       config["outliers"],
                "normalize":      config["normalize"],
                "format":         config["format"],
                "file_size_kb":   file_size_kb,
                "processing_time": processing_time,
            },
            session["user_id"],
        )
        log_activity(session["user_id"], "PREVIEW", f"Preview of {safe_name}", ip)

        user_settings      = get_user_settings(session["user_id"])
        preview_rows_count = int(user_settings.get("preview_rows", 10))

        def to_display(frame, n):
            return frame.head(n).astype(object).where(frame.head(n).notna(), other="N/A").values.tolist()

        return render_template(
            "preview.html",
            stats=stats,
            changes=changes,
            columns=df_clean.columns.tolist(),
            preview_data=to_display(df_clean, preview_rows_count),
            before_cols=df.columns.tolist(),
            before_data=to_display(df, preview_rows_count),
            format=config["format"],
        )

    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] preview user={session.get('username')}: {e}", exc_info=True)
        log_activity(session["user_id"], "PREVIEW_ERROR", f"[{error_id}] {type(e).__name__}")
        flash(f"Processing error (ref: {error_id}). Please check your file.", "danger")
        return redirect(url_for("main.index"))


# ── Download cleaned file ──────────────────────────────────────────────────────

@bp.route("/download_preview")
@login_required
def download_preview():
    try:
        if "preview_temp_id" not in session:
            flash("No data to download. Please upload a file first.", "warning")
            return redirect(url_for("main.index"))

        temp_path = os.path.join(current_app.config["TEMP_DIR"], f"{session['preview_temp_id']}.pkl")
        if not os.path.exists(temp_path):
            flash("Preview session expired. Please re-upload your file.", "warning")
            return redirect(url_for("main.index"))

        df        = pd.read_pickle(temp_path)
        fmt       = session.get("preview_format", "csv")
        orig_name = session.get("preview_filename", "data.csv")

        if fmt not in ("csv", "xlsx", "json"):
            fmt = "csv"

        base_name = secure_filename(orig_name).rsplit(".", 1)[0]
        out_name  = f"{base_name}_cleaned.{fmt}"
        out_path  = os.path.join(current_app.config["UPLOAD_DIR"], out_name)

        write_file(df, out_path, fmt)
        log_activity(session["user_id"], "DOWNLOAD", f"Download of {out_name}")

        # Cleanup
        try:
            os.remove(temp_path)
        except Exception:
            pass
        for key in ("preview_temp_id", "preview_format", "preview_filename"):
            session.pop(key, None)

        return send_file(out_path, as_attachment=True, download_name=out_name)

    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] download user={session.get('username')}: {e}", exc_info=True)
        flash(f"Download error (ref: {error_id}).", "danger")
        return redirect(url_for("main.index"))


# ── History ────────────────────────────────────────────────────────────────────

@bp.route("/history")
@login_required
def history():
    limit   = request.args.get("limit", 50, type=int)
    user_id = session["user_id"] if session.get("role") != "admin" else None
    return jsonify(get_history(user_id=user_id, limit=limit))


@bp.route("/statistics")
@login_required
def statistics():
    user_id = session["user_id"] if session.get("role") != "admin" else None
    return jsonify(get_statistics(user_id=user_id))


@bp.route("/delete_history/<int:history_id>", methods=["DELETE"])
@login_required
def delete_history_item_route(history_id: int):
    try:
        record = delete_history_item(history_id)
        if not record:
            return jsonify({"success": False, "error": "Not found"}), 404
        if record["user_id"] != session["user_id"] and session.get("role") != "admin":
            return jsonify({"success": False, "error": "Unauthorized"}), 403
        log_activity(session["user_id"], "DELETE_HISTORY", f"Item {history_id} deleted")
        return jsonify({"success": True})
    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] delete_history: {e}")
        return jsonify({"success": False, "error": f"Error ({error_id})"}), 500


@bp.route("/clear_history", methods=["POST"])
@login_required
def clear_history_route():
    try:
        user_id = None if session.get("role") == "admin" else session["user_id"]
        clear_history(user_id)
        log_activity(session["user_id"], "CLEAR_HISTORY", "History cleared")
        return jsonify({"success": True})
    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] clear_history: {e}")
        return jsonify({"success": False, "error": f"Error ({error_id})"}), 500


# ── Settings ───────────────────────────────────────────────────────────────────

@bp.route("/settings")
@login_required
def settings():
    return render_template(
        "settings.html",
        user=get_user_by_id(session["user_id"]),
        settings=get_user_settings(session["user_id"]),
    )


@bp.route("/save_settings", methods=["POST"])
@login_required
@csrf_required
def save_settings():
    try:
        preview_rows = int(request.form.get("preview_rows", 10))
        if preview_rows not in (5, 10, 20, 50):
            preview_rows = 10
        default_format = request.form.get("default_format", "csv")
        if default_format not in ("csv", "xlsx", "json"):
            default_format = "csv"

        save_user_settings(
            session["user_id"],
            {
                "always_preview":     "always_preview"    in request.form,
                "save_history":       "save_history"       in request.form,
                "confirm_delete":     "confirm_delete"     in request.form,
                "default_missing":    "default_missing"    in request.form,
                "default_duplicates": "default_duplicates" in request.form,
                "default_outliers":   "default_outliers"   in request.form,
                "default_normalize":  "default_normalize"  in request.form,
                "default_format":     default_format,
                "preview_rows":       preview_rows,
            },
        )
        log_activity(session["user_id"], "SETTINGS_UPDATED", "Settings updated")
        flash("Settings saved.", "success")
    except Exception as e:
        security_logger.error(f"save_settings user={session.get('username')}: {e}")
        flash("Error saving settings.", "danger")

    return redirect(url_for("main.settings"))
