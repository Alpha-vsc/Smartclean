# SmartClean

A web application for importing, cleaning, analysing and converting data files.

**Authors:** Alpha O. Diallo & Aicha Diop

---

## Features

| Feature | Description |
|---|---|
| **Upload & Clean** | CSV, Excel, JSON, XML — remove duplicates, fill missing values, drop outliers, normalise |
| **Preview** | Before/after comparison with transformation statistics |
| **Analyse** | Per-column stats: type, missing %, min/max/mean/median/std, top values |
| **Convert** | CSV ↔ XLSX ↔ JSON ↔ XML ↔ TSV ↔ Parquet |
| **History** | Per-user processing history with timing and applied operations |
| **Settings** | Configurable preview rows, default format, default cleaning options |
| **Auth** | Registration, login, email verification, password change |
| **Admin** | User management, activity logs, global statistics |
| **Security** | CSRF, rate limiting, file content validation, security headers |

---

## Project Structure

```
smartclean/
├── run.py                  # Entry point
├── .env.example            # Environment variable template
├── requirements.txt
└── app/
    ├── __init__.py         # Application factory (create_app)
    ├── models/
    │   ├── database.py     # SQLite connection + schema initialisation
    │   ├── user.py         # User CRUD, settings, email verification
    │   └── history.py      # Processing history + statistics
    ├── services/
    │   ├── cleaner.py      # Data cleaning pipeline (pure Python, no Flask)
    │   ├── analyzer.py     # Statistical profiling
    │   ├── file_io.py      # Unified file reader / writer (all formats)
    │   └── email_service.py# SMTP email sending
    ├── routes/
    │   ├── auth.py         # /login  /register  /logout  /verify-email
    │   ├── main.py         # /  /preview  /download_preview  /settings  /history
    │   ├── data.py         # /analyze
    │   ├── convert.py      # /convert
    │   └── admin.py        # /admin/*
    ├── utils/
    │   ├── security.py     # CSRF, rate limiting, file validation, auth decorators
    │   └── cleanup.py      # Background temp-file cleanup thread
    ├── templates/          # Jinja2 HTML templates
    └── static/             # CSS, JS, images
```

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/your-org/smartclean.git
cd smartclean

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set SECRET_KEY and ADMIN_PASSWORD at minimum

# 5. Run
python run.py
# App is available at http://localhost:5000
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | **Yes** | random | Flask session secret — set a long random string in production |
| `ADMIN_PASSWORD` | **Yes** | random | Password for the admin account |
| `FLASK_DEBUG` | No | `false` | Enable debug mode (never use in production) |
| `PERSISTENT_DISK_PATH` | No | project root | Directory for uploads, temp files and the SQLite DB |
| `HTTPS` | No | `false` | Set `true` behind a reverse proxy with SSL |
| `SMTP_HOST` | No | `smtp.gmail.com` | SMTP server |
| `SMTP_PORT` | No | `587` | SMTP port |
| `SMTP_USER` | No | — | SMTP username — if not set, email verification is skipped |
| `SMTP_PASS` | No | — | SMTP password / app password |
| `FROM_EMAIL` | No | `SMTP_USER` | Sender address |

---

## Architecture Decisions

### Application Factory (`create_app`)
Enables multiple app instances for testing and separates configuration from code.

### Flask `g` for DB connections
`get_db()` stores the connection in Flask's `g` object and registers a teardown handler — one connection per request, automatically closed.

### Service layer with no Flask imports
`cleaner.py`, `analyzer.py` and `file_io.py` have zero Flask dependencies.  
They can be called from unit tests, CLI scripts or background jobs without an application context.

### Single `read_file` / `write_file`
The format-dispatch logic existed in three separate routes. Centralising it in `file_io.py` eliminates duplication and makes adding a new format a one-line change.

### Blueprint organisation
| Blueprint | Prefix | Responsibility |
|---|---|---|
| `auth` | — | Authentication flows |
| `main` | — | Core app (upload, preview, download, settings) |
| `data` | — | Analysis API |
| `convert` | — | Format conversion |
| `admin` | `/admin` | Admin dashboard and user management |

---

## Security

- **CSRF** — custom token checked on every POST
- **Rate limiting** — in-memory sliding window (no external dependency)
- **File validation** — extension allowlist + magic-byte / encoding check
- **Session hardening** — HttpOnly, SameSite=Lax, session fixation prevention
- **Security headers** — CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy
- **Password hashing** — Werkzeug PBKDF2-SHA256

---

## Deployment (Render / Railway / Fly.io)

```
Start command : gunicorn "app:create_app()" --bind 0.0.0.0:$PORT
Build command : pip install -r requirements.txt
```

Set `PERSISTENT_DISK_PATH` to your mounted disk path so uploads and the database survive redeploys.
