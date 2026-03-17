# ==============================================================================
# SmartClean — History & Statistics Model
# ==============================================================================

import sqlite3
from datetime import datetime, timedelta, date
from typing import Optional
from .database import get_db

# Python 3.12 deprecated the default datetime adapters — register explicit ones
sqlite3.register_adapter(datetime, lambda d: d.isoformat())
sqlite3.register_adapter(date,     lambda d: d.isoformat())
sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))
sqlite3.register_converter("DATE",      lambda b: date.fromisoformat(b.decode()))


def save_to_history(data: dict, user_id: int) -> None:
    db = get_db()
    db.execute(
        """
        INSERT INTO processing_history
            (user_id, filename, original_rows, processed_rows,
             original_columns, processed_columns,
             missing_values_applied, duplicates_applied,
             outliers_applied, normalize_applied,
             output_format, file_size_kb, processing_time_seconds)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            data["filename"],
            data["original_rows"],
            data["processed_rows"],
            data["original_cols"],
            data["processed_cols"],
            data["missing"],
            data["duplicates"],
            data["outliers"],
            data["normalize"],
            data["format"],
            data["file_size_kb"],
            data["processing_time"],
        ),
    )
    db.commit()


def get_history(user_id: Optional[int] = None, limit: int = 50) -> list[dict]:
    db     = get_db()
    query  = """
        SELECT ph.*, u.username, u.full_name
        FROM processing_history ph
        JOIN users u ON ph.user_id = u.id
        {where}
        ORDER BY ph.processing_date DESC LIMIT ?
    """
    if user_id:
        rows = db.execute(query.format(where="WHERE ph.user_id = ?"), (user_id, limit)).fetchall()
    else:
        rows = db.execute(query.format(where=""), (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_history_item(history_id: int) -> Optional[dict]:
    db  = get_db()
    row = db.execute(
        "SELECT user_id FROM processing_history WHERE id=?", (history_id,)
    ).fetchone()
    if row:
        db.execute("DELETE FROM processing_history WHERE id=?", (history_id,))
        db.commit()
    return dict(row) if row else None


def clear_history(user_id: Optional[int] = None) -> None:
    db = get_db()
    if user_id:
        db.execute("DELETE FROM processing_history WHERE user_id=?", (user_id,))
    else:
        db.execute("DELETE FROM processing_history")
    db.commit()


def get_statistics(user_id: Optional[int] = None) -> dict:
    db     = get_db()
    params = (user_id,) if user_id else ()
    where  = "WHERE user_id = ?" if user_id else ""

    total_files = db.execute(f"SELECT COUNT(*) FROM processing_history {where}", params).fetchone()[0]
    total_rows  = db.execute(f"SELECT COALESCE(SUM(processed_rows),0) FROM processing_history {where}", params).fetchone()[0]
    avg_time    = db.execute(f"SELECT COALESCE(AVG(processing_time_seconds),0) FROM processing_history {where}", params).fetchone()[0]
    ops = db.execute(
        f"""
        SELECT
            SUM(CASE WHEN missing_values_applied=1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN duplicates_applied=1     THEN 1 ELSE 0 END),
            SUM(CASE WHEN outliers_applied=1       THEN 1 ELSE 0 END),
            SUM(CASE WHEN normalize_applied=1      THEN 1 ELSE 0 END)
        FROM processing_history {where}
        """,
        params,
    ).fetchone()

    return {
        "total_files": total_files,
        "total_rows":  int(total_rows),
        "avg_processing_time": round(avg_time, 2),
        "operations": {
            "missing_values": ops[0] or 0,
            "duplicates":     ops[1] or 0,
            "outliers":       ops[2] or 0,
            "normalize":      ops[3] or 0,
        },
    }


def get_admin_statistics() -> dict:
    db = get_db()
    total_users  = db.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
    active_users = db.execute(
        "SELECT COUNT(*) FROM users WHERE last_login >= ?",
        (datetime.now() - timedelta(days=7),),
    ).fetchone()[0]
    total_ops    = db.execute("SELECT COUNT(*) FROM processing_history").fetchone()[0]
    ops_today    = db.execute(
        "SELECT COUNT(*) FROM processing_history WHERE DATE(processing_date)=?",
        (datetime.now().date(),),
    ).fetchone()[0]
    usage_by_day = [dict(r) for r in db.execute("""
        SELECT DATE(processing_date) as date, COUNT(*) as count
        FROM processing_history
        WHERE processing_date >= date('now','-30 days')
        GROUP BY DATE(processing_date) ORDER BY date DESC
    """).fetchall()]
    top_users = [dict(r) for r in db.execute("""
        SELECT u.username, u.full_name, COUNT(ph.id) as count
        FROM users u LEFT JOIN processing_history ph ON u.id=ph.user_id
        WHERE u.role != 'admin'
        GROUP BY u.id ORDER BY count DESC LIMIT 10
    """).fetchall()]
    recent_activity = [dict(r) for r in db.execute("""
        SELECT u.username, al.action, al.details, al.timestamp
        FROM activity_logs al JOIN users u ON al.user_id=u.id
        ORDER BY al.timestamp DESC LIMIT 20
    """).fetchall()]
    users_list = [dict(r) for r in db.execute("""
        SELECT id, username, email, full_name, role, created_at, last_login, is_active
        FROM users ORDER BY created_at DESC
    """).fetchall()]

    return {
        "total_users":     total_users,
        "active_users":    active_users,
        "total_api_calls": total_ops,
        "api_calls_today": ops_today,
        "usage_by_day":    usage_by_day,
        "top_users":       top_users,
        "recent_activity": recent_activity,
        "users_list":      users_list,
    }
