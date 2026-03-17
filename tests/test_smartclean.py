# ==============================================================================
# SmartClean — Test Suite
# Run with: python -m pytest tests/ -v
# ==============================================================================

import io
import json
import pytest
import pandas as pd
import numpy as np

from app import create_app


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    application = create_app()
    application.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
        SERVER_NAME="localhost",
    )
    with application.app_context():
        yield application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_client(app, client):
    """A test client already logged in as the admin user."""
    with client.session_transaction() as sess:
        sess["user_id"]  = 1
        sess["username"] = "admin"
        sess["role"]     = "admin"
        sess["csrf_token"] = "test-csrf-token"
    return client


# ── Helpers ────────────────────────────────────────────────────────────────────

def _csv_bytes(rows=50):
    """Generate a simple in-memory CSV file."""
    cities = (["Paris", "Lyon", "Marseille", None] * ((rows // 4) + 1))[:rows]
    df = pd.DataFrame({
        "id":    range(1, rows + 1),
        "name":  [f"User {i}" for i in range(1, rows + 1)],
        "score": np.random.randint(0, 100, rows).tolist(),
        "city":  cities,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _xlsx_bytes(rows=20):
    df  = pd.DataFrame({"a": range(rows), "b": [float(i) * 1.5 for i in range(rows)]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf


# ==============================================================================
# SERVICE TESTS — no HTTP, pure business logic
# ==============================================================================

class TestCleaner:

    def test_remove_duplicates(self):
        from app.services.cleaner import process_data
        df  = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out, changes = process_data(df, {"duplicates": True})
        assert len(out) == 2
        assert any("duplicate" in c.lower() for c in changes)

    def test_fill_missing_numeric(self):
        from app.services.cleaner import process_data
        df  = pd.DataFrame({"score": [10.0, 20.0, None, 30.0]})
        out, changes = process_data(df, {"missing_values": True})
        assert out["score"].isna().sum() == 0

    def test_fill_missing_text(self):
        from app.services.cleaner import process_data
        df  = pd.DataFrame({"city": ["Paris", "Lyon", None, "Paris"]})
        out, changes = process_data(df, {"missing_values": True})
        assert out["city"].isna().sum() == 0

    def test_no_ops_message(self):
        from app.services.cleaner import process_data
        # Use floats with decimals so Int64 conversion is not triggered,
        # and text so no numeric coercion fires — truly nothing to do.
        df  = pd.DataFrame({"label": ["apple", "banana", "cherry"]})
        _, changes = process_data(df, {})
        assert any("no changes applied" in c.lower() for c in changes)

    def test_normalize_missing_replaces_na_variants(self):
        from app.services.cleaner import normalize_missing
        # Only NA variants — no numeric coercion should happen here
        df = pd.DataFrame({"v": ["n/a", "null", "--", "hello", "world"]})
        out, changes = normalize_missing(df)
        # 3 NA variants should become NaN; "hello" and "world" remain
        assert out["v"].isna().sum() == 3
        assert out["v"].dropna().tolist() == ["hello", "world"]

    def test_is_id_column_by_name(self):
        from app.services.cleaner import is_id_column
        id_series    = pd.Series(range(100))
        # Detected by name
        assert is_id_column(id_series, "user_id") is True
        # Non-ID name + low-cardinality values (only 3 unique out of 100)
        low_card = pd.Series([1, 2, 3] * 33 + [1])
        assert is_id_column(low_card, "score") is False

    def test_outlier_removal(self):
        from app.services.cleaner import process_data
        # 50 normal values + 2 extreme outliers
        vals = list(range(50)) + [10000, -10000]
        df   = pd.DataFrame({"v": vals})
        out, changes = process_data(df, {"outliers": True})
        assert len(out) < len(df)
        assert any("outlier" in c.lower() for c in changes)

    def test_normalize_minmax(self):
        from app.services.cleaner import process_data
        df  = pd.DataFrame({"v": [0.0, 50.0, 100.0]})
        out, _ = process_data(df, {"normalize": True})
        assert out["v"].min() >= 0.0
        assert out["v"].max() <= 1.0


class TestAnalyzer:

    def test_numeric_stats_present(self):
        from app.services.analyzer import analyze_dataframe
        df      = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
        summary = analyze_dataframe(df, "test.csv")
        col     = summary["columns"][0]
        assert "min" in col and "max" in col and "mean" in col and "std" in col

    def test_text_top_values_present(self):
        from app.services.analyzer import analyze_dataframe
        df      = pd.DataFrame({"city": ["Paris", "Paris", "Lyon", "Marseille"]})
        summary = analyze_dataframe(df, "test.csv")
        col     = summary["columns"][0]
        assert "top_values" in col
        assert "Paris" in col["top_values"]

    def test_summary_shape(self):
        from app.services.analyzer import analyze_dataframe
        df      = pd.DataFrame({"a": [1, 2, None], "b": ["x", "x", "y"]})
        summary = analyze_dataframe(df, "test.csv")
        assert summary["rows"] == 3
        assert summary["cols"] == 2
        assert summary["total_missing"] == 1

    def test_duplicates_count(self):
        from app.services.analyzer import analyze_dataframe
        df      = pd.DataFrame({"a": [1, 1, 2]})
        summary = analyze_dataframe(df, "test.csv")
        assert summary["duplicates"] == 1


class TestFileIO:

    def test_read_csv(self):
        from app.services.file_io import read_file
        buf = _csv_bytes(10)
        df  = read_file(buf, "csv")
        assert len(df) == 10
        assert "score" in df.columns

    def test_read_xlsx(self):
        from app.services.file_io import read_file
        buf = _xlsx_bytes(5)
        df  = read_file(buf, "xlsx")
        assert len(df) == 5

    def test_read_json(self):
        from app.services.file_io import read_file
        data = json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).encode()
        buf  = io.BytesIO(data)
        df   = read_file(buf, "json")
        assert len(df) == 2

    def test_write_csv(self, tmp_path):
        from app.services.file_io import write_file
        df   = pd.DataFrame({"x": [1, 2, 3]})
        path = str(tmp_path / "out.csv")
        write_file(df, path, "csv")
        result = pd.read_csv(path)
        assert list(result["x"]) == [1, 2, 3]

    def test_write_json(self, tmp_path):
        from app.services.file_io import write_file
        df   = pd.DataFrame({"name": ["Alice", "Bob"]})
        path = str(tmp_path / "out.json")
        write_file(df, path, "json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_write_xlsx(self, tmp_path):
        from app.services.file_io import write_file
        df   = pd.DataFrame({"v": [10, 20, 30]})
        path = str(tmp_path / "out.xlsx")
        write_file(df, path, "xlsx")
        result = pd.read_excel(path)
        assert list(result["v"]) == [10, 20, 30]

    def test_unsupported_format_raises(self):
        from app.services.file_io import read_file
        with pytest.raises(ValueError):
            read_file(io.BytesIO(b"data"), "docx")


# ==============================================================================
# SECURITY UTILS TESTS
# ==============================================================================

class TestSecurity:

    def test_allowed_file_valid(self):
        from app.utils.security import allowed_file
        assert allowed_file("data.csv")  is True
        assert allowed_file("data.xlsx") is True
        assert allowed_file("data.json") is True
        assert allowed_file("data.xml")  is True

    def test_allowed_file_invalid(self):
        from app.utils.security import allowed_file
        assert allowed_file("data.exe")  is False
        assert allowed_file("data.php")  is False
        assert allowed_file("noextension") is False

    def test_rate_limiter_blocks_after_limit(self):
        from app.utils.security import is_rate_limited, _rl_store
        # Clear any previous state for this test key
        _rl_store.clear()
        ip = "10.0.0.99"
        # login limit is 5 per 60s
        for _ in range(5):
            is_rate_limited(ip, "login")
        assert is_rate_limited(ip, "login") is True

    def test_rate_limiter_allows_under_limit(self):
        from app.utils.security import is_rate_limited, _rl_store
        _rl_store.clear()
        ip = "10.0.0.100"
        for _ in range(4):
            result = is_rate_limited(ip, "login")
        assert result is False


# ==============================================================================
# MODEL TESTS (require app context)
# ==============================================================================

class TestUserModel:

    def test_get_user_settings_creates_defaults(self, app):
        from app.models.user     import get_user_settings, create_user
        from app.models.database import get_db
        with app.app_context():
            # Create a temporary user so the FK constraint is satisfied
            db = get_db()
            db.execute(
                "INSERT OR IGNORE INTO users (username, email, password_hash, is_active)"
                " VALUES ('tmpuser', 'tmp@test.com', 'x', 1)"
            )
            db.commit()
            uid = db.execute(
                "SELECT id FROM users WHERE username='tmpuser'"
            ).fetchone()["id"]

            settings = get_user_settings(uid)
            assert "preview_rows"   in settings
            assert settings["preview_rows"]   == 10
            assert settings["default_format"] == "csv"

    def test_save_and_reload_settings(self, app):
        from app.models.user import save_user_settings, get_user_settings
        with app.app_context():
            save_user_settings(1, {"preview_rows": 20, "default_format": "xlsx"})
            s = get_user_settings(1)
            assert s["preview_rows"]    == 20
            assert s["default_format"]  == "xlsx"

    def test_get_user_by_id(self, app):
        from app.models.user import get_user_by_id
        with app.app_context():
            user = get_user_by_id(1)
            assert user is not None
            assert user["username"] == "admin"


class TestHistoryModel:

    def test_save_and_retrieve(self, app):
        from app.models.history import save_to_history, get_history
        with app.app_context():
            save_to_history(
                {
                    "filename": "test.csv",
                    "original_rows": 100, "processed_rows": 90,
                    "original_cols": 5,   "processed_cols": 5,
                    "missing": True, "duplicates": True,
                    "outliers": False, "normalize": False,
                    "format": "csv", "file_size_kb": 12.5,
                    "processing_time": 0.42,
                },
                user_id=1,
            )
            history = get_history(user_id=1, limit=10)
            assert len(history) >= 1
            assert history[0]["filename"] == "test.csv"

    def test_statistics_structure(self, app):
        from app.models.history import get_statistics
        with app.app_context():
            stats = get_statistics(user_id=1)
            assert "total_files" in stats
            assert "operations"  in stats
            assert "total_rows"  in stats


# ==============================================================================
# HTTP ROUTE TESTS
# ==============================================================================

class TestAuthRoutes:

    def test_login_page_loads(self, client):
        r = client.get("/login")
        assert r.status_code == 200
        assert b"login" in r.data.lower()

    def test_register_page_loads(self, client):
        r = client.get("/register")
        assert r.status_code == 200

    def test_login_invalid_credentials(self, client):
        r = client.post("/login", data={
            "username": "nobody", "password": "wrong",
            "csrf_token": "x"
        }, follow_redirects=True)
        # Either 200 with flash or redirect — just confirm no 500
        assert r.status_code in (200, 302, 429)

    def test_logout_redirects(self, auth_client):
        r = auth_client.get("/logout", follow_redirects=False)
        assert r.status_code == 302

    def test_protected_route_redirects_unauthenticated(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 302
        assert "/login" in r.headers["Location"]


class TestMainRoutes:

    def test_index_requires_login(self, client):
        r = client.get("/")
        assert r.status_code == 302

    def test_settings_page_loads(self, auth_client):
        r = auth_client.get("/settings")
        assert r.status_code == 200

    def test_history_returns_json(self, auth_client):
        r = auth_client.get("/history")
        assert r.status_code == 200
        assert r.is_json

    def test_statistics_returns_json(self, auth_client):
        r = auth_client.get("/statistics")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_files" in data

    def test_preview_no_file(self, auth_client):
        r = auth_client.post("/preview", data={"csrf_token": "test-csrf-token"},
                             follow_redirects=True)
        assert r.status_code in (200, 302)

    def test_preview_with_csv(self, auth_client):
        csv_data = b"name,score\nAlice,90\nBob,85\nAlice,90\n"
        r = auth_client.post(
            "/preview",
            data={
                "file":          (io.BytesIO(csv_data), "test.csv"),
                "duplicates":    "on",
                "output_format": "csv",
                "csrf_token":    "test-csrf-token",
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert r.status_code == 200

    def test_download_without_preview_redirects(self, auth_client):
        # Clear any existing session preview keys
        with auth_client.session_transaction() as sess:
            sess.pop("preview_temp_id", None)
        r = auth_client.get("/download_preview", follow_redirects=False)
        assert r.status_code == 302


class TestDataRoutes:

    def test_analyze_no_file(self, auth_client):
        r = auth_client.post("/analyze", data={"csrf_token": "test-csrf-token"})
        assert r.status_code == 400
        assert r.is_json

    def test_analyze_csv(self, auth_client):
        csv_data = b"name,score,city\nAlice,90,Paris\nBob,85,Lyon\nCharlie,,Paris\n"
        r = auth_client.post(
            "/analyze",
            data={
                "file":       (io.BytesIO(csv_data), "sample.csv"),
                "csrf_token": "test-csrf-token",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["rows"] == 3
        assert data["cols"] == 3
        assert data["total_missing"] == 1

    def test_analyze_unsupported_format(self, auth_client):
        r = auth_client.post(
            "/analyze",
            data={
                "file":       (io.BytesIO(b"data"), "bad.exe"),
                "csrf_token": "test-csrf-token",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 400


class TestConvertRoutes:

    def test_convert_page_loads(self, auth_client):
        r = auth_client.get("/convert")
        assert r.status_code == 200

    def test_convert_csv_to_json(self, auth_client):
        csv_data = b"name,score\nAlice,90\nBob,85\n"
        r = auth_client.post(
            "/convert",
            data={
                "file":          (io.BytesIO(csv_data), "input.csv"),
                "target_format": "json",
            },
            content_type="multipart/form-data",
        )
        # Either the file downloads (200) or redirects on error (302)
        assert r.status_code in (200, 302)

    def test_convert_no_file_redirects(self, auth_client):
        r = auth_client.post(
            "/convert",
            data={"target_format": "json"},
            follow_redirects=False,
        )
        assert r.status_code == 302


class TestAdminRoutes:

    def test_admin_dashboard_accessible_by_admin(self, auth_client):
        r = auth_client.get("/admin/")
        assert r.status_code == 200

    def test_admin_statistics_returns_json(self, auth_client):
        r = auth_client.get("/admin/statistics")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_users" in data

    def test_admin_routes_blocked_for_non_admin(self, client):
        # Log in as regular user
        with client.session_transaction() as sess:
            sess["user_id"]  = 999
            sess["username"] = "regular"
            sess["role"]     = "user"
        r = client.get("/admin/", follow_redirects=False)
        assert r.status_code == 302


# ==============================================================================
# INTEGRATION TESTS — full clean pipeline end-to-end
# ==============================================================================

class TestIntegration:

    def test_full_clean_pipeline(self):
        from app.services.cleaner import process_data
        df = pd.DataFrame({
            "id":    range(1, 21),
            "score": [10, 20, 30, None, 50, 60, 10, 20, "n/a", 90,
                      10, 20, 30, 40,  50, 60, 70, 80, 90, 100],
            "city":  ["Paris", "Lyon", None, "Paris", "Paris"] * 4,
        })
        out, changes = process_data(df, {
            "duplicates":     True,
            "missing_values": True,
            "outliers":       False,
            "normalize":      True,
        })
        assert out["score"].isna().sum() == 0
        assert out["city"].isna().sum()  == 0
        assert isinstance(changes, list)
        assert len(changes) > 0

    def test_analyze_then_clean_consistent(self):
        from app.services.analyzer import analyze_dataframe
        from app.services.cleaner  import process_data
        df = pd.DataFrame({
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "y": ["a", "b",  "a", None, "b"],
        })
        before = analyze_dataframe(df, "test.csv")
        assert before["total_missing"] == 2

        cleaned, _ = process_data(df, {"missing_values": True})
        after = analyze_dataframe(cleaned, "test.csv")
        assert after["total_missing"] == 0

    def test_round_trip_csv_to_json(self, tmp_path):
        from app.services.file_io import read_file, write_file
        original = pd.DataFrame({"name": ["Alice", "Bob"], "score": [90, 85]})
        csv_buf  = io.BytesIO()
        original.to_csv(csv_buf, index=False)
        csv_buf.seek(0)

        df        = read_file(csv_buf, "csv")
        json_path = str(tmp_path / "out.json")
        write_file(df, json_path, "json")

        with open(json_path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["name"] == "Alice"
