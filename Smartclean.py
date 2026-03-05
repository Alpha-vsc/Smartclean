# ==============================================================================
# SmartClean — Backend v3.2
# SmartClean | Auteurs : Alpha O. Diallo & Aicha Diop
# ==============================================================================

import os
import re
import time
import uuid
import stat
import secrets
import logging
import threading
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import (Flask, render_template, request, send_file,
                   jsonify, redirect, url_for, flash, session)
from sklearn.preprocessing import MinMaxScaler
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from collections import defaultdict
from datetime import datetime, timedelta
import sqlite3
import json

# ── Charger le fichier .env ──
load_dotenv()

# ==============================================================================
# CONFIGURATION DU LOGGING
# ==============================================================================

# Logs de sécurité → fichier uniquement
_log_dir = os.environ.get('PERSISTENT_DISK_PATH', os.path.abspath(os.path.dirname(__file__)))
file_handler = logging.FileHandler(os.path.join(_log_dir, 'smartclean_security.log'))
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

security_logger = logging.getLogger('security')
security_logger.addHandler(file_handler)
security_logger.setLevel(logging.WARNING)

# Werkzeug (Flask) → terminal uniquement
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)

# ==============================================================================
# INITIALISATION FLASK
# ==============================================================================

import os
app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

# ── Clé secrète ──
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    _secret = secrets.token_hex(32)
    security_logger.warning(
        "SECRET_KEY non définie. Clé aléatoire utilisée : "
        "toutes les sessions seront perdues au redémarrage. "
        "Définissez SECRET_KEY dans votre .env"
    )
app.secret_key = _secret

# ── Configuration session ──
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV', 'production') != 'development'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# ── Headers de sécurité sur toutes les réponses ──
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
        "font-src 'self' cdnjs.cloudflare.com; "
        "img-src 'self' data:;"
    )
    return response

# ==============================================================================
# CHEMINS & DOSSIERS
# ==============================================================================

BASE_DIR        = os.path.abspath(os.path.dirname(__file__))

# Disque persistant Render (variable d'env PERSISTENT_DISK_PATH=/data) sinon fallback local
PERSISTENT_DIR  = os.environ.get('PERSISTENT_DISK_PATH', BASE_DIR)
UPLOAD_DIR      = os.path.join(PERSISTENT_DIR, 'uploads')
TEMP_DIR        = os.path.join(PERSISTENT_DIR, 'temp_previews')
DB_PATH         = os.path.join(PERSISTENT_DIR, 'smartclean.db')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

# ==============================================================================
# SÉCURITÉ FICHIERS — EXTENSIONS & MAGIC BYTES
# ==============================================================================

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'xml', 'tsv', 'parquet'}

FILE_SIGNATURES = {
    'xlsx': b'PK\x03\x04',
    'xls':  b'\xd0\xcf\x11\xe0',
    # csv, json, xml : fichiers texte, pas de magic bytes
}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_content(file, extension: str) -> bool:
    """Vérifie que le contenu du fichier correspond à son extension."""
    sig = FILE_SIGNATURES.get(extension)
    if sig:
        header = file.read(len(sig))
        file.seek(0)
        return header == sig
    else:
        # Fichier texte : vérifier l'encodage
        header = file.read(512)
        file.seek(0)
        for enc in ('utf-8', 'latin-1', 'utf-16'):
            try:
                header.decode(enc)
                return True
            except (UnicodeDecodeError, Exception):
                continue
        return False

# ==============================================================================
# RATE LIMITING (sans dépendance externe)
# ==============================================================================

_rl_store = defaultdict(list)
_rl_lock  = threading.Lock()

RATE_LIMITS = {
    'login':    (5,  60),    # 5 tentatives / 60 s
    'register': (3,  300),   # 3 inscriptions / 5 min
    'preview':  (20, 60),    # 20 uploads / min
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

def get_client_ip() -> str:
    return (request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
            or request.remote_addr or 'unknown')

# ==============================================================================
# PROTECTION CSRF (sans dépendance externe)
# ==============================================================================

def generate_csrf_token() -> str:
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']

def validate_csrf() -> bool:
    token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
    session_token = session.get('csrf_token')
    valid = bool(token and session_token and token == session_token)
    if not valid:
        security_logger.warning(
            f"CSRF invalide — IP: {get_client_ip()}, endpoint: {request.endpoint}"
        )
    return valid

def csrf_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == 'POST' and not validate_csrf():
            flash('Session expirée. Veuillez réessayer.', 'danger')
            return redirect(request.referrer or url_for('login'))
        return f(*args, **kwargs)
    return wrapper

# Disponible dans tous les templates
app.jinja_env.globals['csrf_token'] = generate_csrf_token

# ==============================================================================
# BASE DE DONNÉES — INITIALISATION
# ==============================================================================

def ensure_db_writable():
    """Vérifie et corrige les permissions de la base de données."""
    if os.path.exists(DB_PATH) and not os.access(DB_PATH, os.W_OK):
        try:
            os.chmod(DB_PATH, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        except Exception as e:
            raise PermissionError(
                f"Base de données en lecture seule : {DB_PATH}\n"
                f"Windows : clic droit → Propriétés → décocher Lecture seule\n"
                f"Linux/Mac : chmod 644 smartclean.db\nErreur : {e}"
            )

def get_db():
    """Connexion SQLite avec row_factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Crée toutes les tables et le compte admin si nécessaire."""
    ensure_db_writable()
    conn = get_db()
    cur = conn.cursor()

    cur.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            full_name     TEXT,
            role          TEXT    DEFAULT 'user',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login    TIMESTAMP,
            is_active     BOOLEAN DEFAULT 1,
            email_verified BOOLEAN DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS email_verifications (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            code       TEXT    NOT NULL,
            purpose    TEXT    DEFAULT 'verify',
            used       BOOLEAN DEFAULT 0,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS processing_history (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id                  INTEGER NOT NULL,
            filename                 TEXT    NOT NULL,
            original_rows            INTEGER,
            processed_rows           INTEGER,
            original_columns         INTEGER,
            processed_columns        INTEGER,
            missing_values_applied   BOOLEAN,
            duplicates_applied       BOOLEAN,
            outliers_applied         BOOLEAN,
            normalize_applied        BOOLEAN,
            output_format            TEXT,
            processing_date          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size_kb             REAL,
            processing_time_seconds  REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS activity_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            action     TEXT    NOT NULL,
            details    TEXT,
            ip_address TEXT,
            timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS user_settings (
            user_id            INTEGER PRIMARY KEY,
            always_preview     BOOLEAN DEFAULT 1,
            save_history       BOOLEAN DEFAULT 1,
            confirm_delete     BOOLEAN DEFAULT 1,
            default_missing    BOOLEAN DEFAULT 0,
            default_duplicates BOOLEAN DEFAULT 0,
            default_outliers   BOOLEAN DEFAULT 0,
            default_normalize  BOOLEAN DEFAULT 0,
            default_format     TEXT    DEFAULT 'csv',
            preview_rows       INTEGER DEFAULT 10,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
    ''')
    conn.commit()

    # ── Migration : ajouter email_verified si absent (bases existantes) ──
    cols = [row[1] for row in conn.execute('PRAGMA table_info(users)').fetchall()]
    if 'email_verified' not in cols:
        conn.execute('ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0')
        conn.commit()

    # Créer le compte admin si absent
    cur.execute('SELECT COUNT(*) FROM users WHERE role = "admin"')
    if cur.fetchone()[0] == 0:
        pwd = os.environ.get('ADMIN_PASSWORD')
        if not pwd:
            pwd = secrets.token_urlsafe(16)
            print("=" * 60)
            print("⚠️  ADMIN_PASSWORD non défini dans .env")
            print(f"   Mot de passe généré : {pwd}")
            print("   Ajoutez ADMIN_PASSWORD=<votre_mdp> dans votre .env")
            print("=" * 60)
            security_logger.warning("Admin créé avec un mot de passe aléatoire.")
        cur.execute(
            'INSERT INTO users (username, email, password_hash, full_name, role) VALUES (?,?,?,?,?)',
            ('admin', 'admin@smartclean.com', generate_password_hash(pwd), 'Administrator', 'admin')
        )
        conn.commit()
        print("✓ Compte admin créé — Username: admin")

    conn.close()

init_db()

# ==============================================================================
# DÉCORATEURS D'AUTHENTIFICATION
# ==============================================================================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter.', 'warning')
            return redirect(url_for('login'))
        # Vérification en base : compte toujours actif ?
        conn = get_db()
        user = conn.execute(
            'SELECT is_active FROM users WHERE id = ?', (session['user_id'],)
        ).fetchone()
        conn.close()
        if not user or not user['is_active']:
            session.clear()
            flash('Compte désactivé ou introuvable.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = get_client_ip()
        # 1. Session présente ?
        if 'user_id' not in session:
            security_logger.warning(
                f"Tentative accès admin sans session — IP: {ip}, path: {request.path}"
            )
            flash('Veuillez vous connecter.', 'warning')
            return redirect(url_for('login'))
        # 2. Vérification en BASE DE DONNÉES (jamais faire confiance à la session seule)
        conn = get_db()
        user = conn.execute(
            'SELECT role, is_active FROM users WHERE id = ?', (session['user_id'],)
        ).fetchone()
        conn.close()
        if not user or not user['is_active']:
            session.clear()
            security_logger.warning(
                f"Accès admin refusé — compte inactif/introuvable "
                f"— user_id={session.get('user_id')} — IP: {ip}"
            )
            flash('Compte désactivé ou introuvable.', 'danger')
            return redirect(url_for('login'))
        if user['role'] != 'admin':
            security_logger.warning(
                f"Accès admin non autorisé — user_id={session['user_id']} "
                f"— role={user['role']} — IP: {ip} — path: {request.path}"
            )
            flash('Accès administrateur requis.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return wrapper

# ==============================================================================
# FONCTIONS UTILISATEURS
# ==============================================================================

def log_activity(user_id, action, details=None, ip=None):
    conn = get_db()
    conn.execute(
        'INSERT INTO activity_logs (user_id, action, details, ip_address) VALUES (?,?,?,?)',
        (user_id, action, details, ip)
    )
    conn.commit()
    conn.close()

def get_user_by_id(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

def update_last_login(user_id):
    conn = get_db()
    conn.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now(), user_id))
    conn.commit()
    conn.close()

def get_user_settings(user_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM user_settings WHERE user_id = ?', (user_id,)).fetchone()
    if not row:
        conn.execute('INSERT INTO user_settings (user_id) VALUES (?)', (user_id,))
        conn.commit()
        row = conn.execute('SELECT * FROM user_settings WHERE user_id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(row)

def save_user_settings(user_id, data):
    conn = get_db()
    conn.execute('''
        INSERT INTO user_settings
            (user_id, always_preview, save_history, confirm_delete,
             default_missing, default_duplicates, default_outliers, default_normalize,
             default_format, preview_rows)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            always_preview=excluded.always_preview,
            save_history=excluded.save_history,
            confirm_delete=excluded.confirm_delete,
            default_missing=excluded.default_missing,
            default_duplicates=excluded.default_duplicates,
            default_outliers=excluded.default_outliers,
            default_normalize=excluded.default_normalize,
            default_format=excluded.default_format,
            preview_rows=excluded.preview_rows
    ''', (
        user_id,
        data.get('always_preview', True),
        data.get('save_history', True),
        data.get('confirm_delete', True),
        data.get('default_missing', False),
        data.get('default_duplicates', False),
        data.get('default_outliers', False),
        data.get('default_normalize', False),
        data.get('default_format', 'csv'),
        data.get('preview_rows', 10),
    ))
    conn.commit()
    conn.close()


# ==============================================================================
# FONCTIONS EMAIL
# ==============================================================================

def send_email(to_email, subject, body_html):
    smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER', '')
    smtp_pass = os.environ.get('SMTP_PASS', '')
    from_addr = os.environ.get('FROM_EMAIL', smtp_user)
    if not smtp_user or not smtp_pass:
        security_logger.warning('SMTP non configure - email non envoye')
        return False
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = 'SmartClean <' + from_addr + '>'
        msg['To']      = to_email
        msg.attach(MIMEText(body_html, 'html'))
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, to_email, msg.as_string())
        return True
    except Exception as e:
        security_logger.error('Erreur SMTP : ' + str(e))
        return False

def create_verification_code(user_id, purpose='verify'):
    code    = str(secrets.randbelow(900000) + 100000)
    expires = datetime.now() + timedelta(minutes=15)
    conn = get_db()
    conn.execute('UPDATE email_verifications SET used=1 WHERE user_id=? AND purpose=? AND used=0', (user_id, purpose))
    conn.execute('INSERT INTO email_verifications (user_id, code, purpose, expires_at) VALUES (?,?,?,?)', (user_id, code, purpose, expires))
    conn.commit()
    conn.close()
    return code

def verify_code(user_id, code, purpose='verify'):
    conn = get_db()
    row = conn.execute(
        'SELECT id FROM email_verifications WHERE user_id=? AND code=? AND purpose=? AND used=0 AND expires_at > ?',
        (user_id, code, purpose, datetime.now())
    ).fetchone()
    if row:
        conn.execute('UPDATE email_verifications SET used=1 WHERE id=?', (row['id'],))
        conn.commit()
    conn.close()
    return row is not None

# ==============================================================================
# FONCTIONS HISTORIQUE & STATISTIQUES
# ==============================================================================

def save_to_history(data, user_id):
    conn = get_db()
    conn.execute('''
        INSERT INTO processing_history
        (user_id, filename, original_rows, processed_rows,
         original_columns, processed_columns,
         missing_values_applied, duplicates_applied,
         outliers_applied, normalize_applied,
         output_format, file_size_kb, processing_time_seconds)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        user_id,
        data['filename'],       data['original_rows'],    data['processed_rows'],
        data['original_cols'],  data['processed_cols'],
        data['missing'],        data['duplicates'],
        data['outliers'],       data['normalize'],
        data['format'],         data['file_size_kb'],     data['processing_time'],
    ))
    conn.commit()
    conn.close()

def get_history(user_id=None, limit=50):
    conn = get_db()
    if user_id:
        rows = conn.execute('''
            SELECT ph.*, u.username, u.full_name
            FROM processing_history ph
            JOIN users u ON ph.user_id = u.id
            WHERE ph.user_id = ?
            ORDER BY ph.processing_date DESC LIMIT ?
        ''', (user_id, limit)).fetchall()
    else:
        rows = conn.execute('''
            SELECT ph.*, u.username, u.full_name
            FROM processing_history ph
            JOIN users u ON ph.user_id = u.id
            ORDER BY ph.processing_date DESC LIMIT ?
        ''', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_statistics(user_id=None):
    conn = get_db()
    params = (user_id,) if user_id else ()
    where  = 'WHERE user_id = ?' if user_id else ''

    total_files = conn.execute(f'SELECT COUNT(*) FROM processing_history {where}', params).fetchone()[0]
    total_rows  = conn.execute(f'SELECT COALESCE(SUM(processed_rows),0) FROM processing_history {where}', params).fetchone()[0]
    avg_time    = conn.execute(f'SELECT COALESCE(AVG(processing_time_seconds),0) FROM processing_history {where}', params).fetchone()[0]
    ops = conn.execute(f'''
        SELECT
            SUM(CASE WHEN missing_values_applied=1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN duplicates_applied=1     THEN 1 ELSE 0 END),
            SUM(CASE WHEN outliers_applied=1       THEN 1 ELSE 0 END),
            SUM(CASE WHEN normalize_applied=1      THEN 1 ELSE 0 END)
        FROM processing_history {where}
    ''', params).fetchone()
    conn.close()
    return {
        'total_files': total_files,
        'total_rows':  int(total_rows),
        'avg_processing_time': round(avg_time, 2),
        'operations': {
            'missing_values': ops[0] or 0,
            'duplicates':     ops[1] or 0,
            'outliers':       ops[2] or 0,
            'normalize':      ops[3] or 0,
        }
    }

def get_admin_statistics():
    conn = get_db()
    total_users    = conn.execute('SELECT COUNT(*) FROM users WHERE is_active=1').fetchone()[0]
    active_users   = conn.execute(
        'SELECT COUNT(*) FROM users WHERE last_login >= ?',
        (datetime.now() - timedelta(days=7),)
    ).fetchone()[0]
    total_api      = conn.execute('SELECT COUNT(*) FROM processing_history').fetchone()[0]
    api_today      = conn.execute(
        'SELECT COUNT(*) FROM processing_history WHERE DATE(processing_date)=?',
        (datetime.now().date(),)
    ).fetchone()[0]
    usage_by_day   = [dict(r) for r in conn.execute('''
        SELECT DATE(processing_date) as date, COUNT(*) as count
        FROM processing_history
        WHERE processing_date >= date('now','-30 days')
        GROUP BY DATE(processing_date) ORDER BY date DESC
    ''').fetchall()]
    top_users      = [dict(r) for r in conn.execute('''
        SELECT u.username, u.full_name, COUNT(ph.id) as count
        FROM users u LEFT JOIN processing_history ph ON u.id=ph.user_id
        WHERE u.role != 'admin'
        GROUP BY u.id ORDER BY count DESC LIMIT 10
    ''').fetchall()]
    recent_activity = [dict(r) for r in conn.execute('''
        SELECT u.username, al.action, al.details, al.timestamp
        FROM activity_logs al JOIN users u ON al.user_id=u.id
        ORDER BY al.timestamp DESC LIMIT 20
    ''').fetchall()]
    users_list     = [dict(r) for r in conn.execute('''
        SELECT id, username, email, full_name, role, created_at, last_login, is_active
        FROM users ORDER BY created_at DESC
    ''').fetchall()]
    conn.close()
    return {
        'total_users':     total_users,
        'active_users':    active_users,
        'total_api_calls': total_api,
        'api_calls_today': api_today,
        'usage_by_day':    usage_by_day,
        'top_users':       top_users,
        'recent_activity': recent_activity,
        'users_list':      users_list,
    }

# ==============================================================================
# TRAITEMENT DES DONNÉES
# ==============================================================================

NA_VARIANTS = {'na', 'n/a', 'nan', 'null', 'none', '--', '-', 'n.a.', 'n.a', '?', 'missing', ''}
EXTRA_NA    = ['na','n/a','NA','N/A','null','NULL','none','None','NONE','--','-','?','missing','MISSING','n.a.','N.A.']

def normalize_missing(df):
    """
    Remplace toutes les variantes textuelles de NA par de vrais NaN,
    puis essaie de convertir les colonnes majoritairement numériques.
    Retourne (df_nettoyé, liste_de_changements).
    """
    changes = []
    df = df.copy()

    # Uniformiser StringDtype → object
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)

    for col in df.columns:
        # Étape 1 : remplacer les variantes NA
        str_vals = df[col].apply(lambda v: '' if pd.isna(v) else str(v).strip().lower())
        mask = str_vals.isin(NA_VARIANTS)
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            changes.append(f"'{col}' : {n} valeur(s) non-standard (na, n/a, --…) → NaN")

        # Étape 2 : coercion numérique si ≥ 50% de la colonne est numérique
        if df[col].dtype == object:
            already_nan  = df[col].isna()
            coerced      = pd.to_numeric(df[col], errors='coerce')
            new_nan      = coerced.isna()
            invalid      = (~already_nan) & new_nan
            valid_num    = (~already_nan) & (~new_nan)
            non_null     = int((~already_nan).sum())
            mostly_num   = non_null > 0 and int(valid_num.sum()) / non_null >= 0.5

            if mostly_num:
                n_invalid = int(invalid.sum())
                if n_invalid:
                    ex = df.loc[invalid, col].iloc[0]
                    changes.append(f"'{col}' : {n_invalid} valeur(s) invalide(s) (ex: '{ex}') → NaN")
                df[col] = coerced

    # Étape 3 : conversion automatique en entier pour colonnes numériques entières
    # Une colonne est considérée entière si >= 80% de ses valeurs n ont pas de décimales
    for col in df.select_dtypes(include=[np.number]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        pct_integer = (non_null % 1 == 0).sum() / len(non_null)
        if pct_integer >= 0.8:
            df[col] = df[col].round().astype('Int64')
            changes.append(f"'{col}' : convertie en entier (Int64)")

    return df, changes

def process_data(df, config):
    """
    Pipeline de nettoyage complet.
    Retourne (df_traité, liste_de_changements).
    """
    changes = []

    # 1. Normalisation des NA
    df, na_changes = normalize_missing(df)
    changes.extend(na_changes)

    # 2. Suppression des doublons
    if config.get('duplicates'):
        before = len(df)
        df = df.drop_duplicates()
        n = before - len(df)
        if n:
            changes.append(f"{n} doublon(s) supprimé(s)")

    # 1b. Détection automatique des colonnes catégorielles
    for col in df.select_dtypes(include=[np.number]).columns:
        n_unique = df[col].nunique()
        n_total  = len(df[col].dropna())
        if n_total == 0:
            continue
        unique_ratio = n_unique / n_total
        # Exclure les colonnes ID (ratio unique élevé ou nom contenant 'id')
        is_id_col = bool(re.search(r'(^id$|_id$|^pid$)', col, re.IGNORECASE))
        # Catégoriel si : peu de valeurs uniques (<= 10), faible ratio (<= 30%), pas un ID
        if not is_id_col and n_unique <= 10 and unique_ratio <= 0.3:
            df[col] = df[col].astype('category')
            changes.append(f"'{col}' : convertie en catégorielle ({n_unique} valeurs uniques)")

    # 2b. Correction colonnes Y/N (ex: OWN_OCCUPIED) — valeurs invalides → NaN
    for col in df.columns:
        if df[col].dtype == object:
            vals_upper = df[col].dropna().str.upper().unique()
            yn_present = any(v in ('Y', 'N') for v in vals_upper)
            if yn_present:
                invalid_mask = df[col].notna() & ~df[col].str.upper().isin(['Y', 'N'])
                n_inv = int(invalid_mask.sum())
                if n_inv:
                    df.loc[invalid_mask, col] = np.nan
                    changes.append(f"'{col}' : {n_inv} valeur(s) invalide(s) → NaN")

    # 3. Valeurs manquantes
    if config.get('missing_values'):
        filled = 0
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            # Exclure les colonnes ID : ne jamais remplir un identifiant unique
            # Exclure les colonnes ID par nom uniquement (ex: id, pid, user_id...)
            is_id_col = bool(re.search(r'(^id$|_id$|^pid$)', col, re.IGNORECASE))
            if is_id_col:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = pd.to_numeric(df[col].dropna(), errors="coerce").dropna()
                fill_val = float(df[col].median())
                # Si toutes les valeurs sont entières → arrondir à l entier
                if len(non_null) > 0 and (non_null % 1 == 0).all():
                    df[col] = df[col].fillna(int(round(fill_val)))
                else:
                    # Décimales : garder 1 chiffre max (ex: 1.5 → ok, 1.357 → 1.4)
                    df[col] = df[col].fillna(round(fill_val, 1))
            else:
                modes = df[col].dropna().mode()
                df[col] = df[col].fillna(modes.iloc[0] if not modes.empty else "N/A")
            filled += n_missing
        if filled:
            changes.append(f"{filled} valeur(s) manquante(s) remplissées (médiane/mode)")

    # 4. Outliers (méthode IQR)
    if config.get('outliers'):
        before = len(df)
        skipped = []
        for col in df.select_dtypes(include=[np.number]).columns:
            # Minimum 30 lignes par colonne pour éviter les faux outliers sur petits datasets
            if df[col].dropna().count() < 30:
                skipped.append(col)
                continue
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue  # Toutes les valeurs identiques, pas d outliers possibles
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        n = before - len(df)
        if n:
            changes.append(f"{n} outlier(s) supprimé(s) (méthode IQR)")
        if skipped:
            changes.append(f"Outliers ignorés pour {len(skipped)} colonne(s) (moins de 30 lignes)")

    # 5. Normalisation Min-Max (exclut les colonnes ID)
    if config.get('normalize'):
        num_cols = df.select_dtypes(include=[np.number]).columns
        # Exclure les colonnes dont le nom contient 'id' (insensible à la casse)
        id_cols  = [c for c in num_cols if re.search(r'\bid\b', c, re.IGNORECASE)]
        cols_to_normalize = [c for c in num_cols if c not in id_cols]
        if cols_to_normalize:
            df[cols_to_normalize] = MinMaxScaler().fit_transform(df[cols_to_normalize])
            changes.append(f"Normalisation Min-Max appliquée ({len(cols_to_normalize)} colonne(s), colonnes ID exclues)")

    if not changes:
        changes.append("Aucun changement appliqué")

    return df, changes

# ==============================================================================
# NETTOYAGE AUTOMATIQUE DES FICHIERS TEMPORAIRES
# ==============================================================================

def _cleanup_loop(max_age_min=30, interval_min=15):
    while True:
        time.sleep(interval_min * 60)
        cutoff  = time.time() - max_age_min * 60
        cleaned = 0
        for folder in [TEMP_DIR, UPLOAD_DIR]:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                try:
                    if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        cleaned += 1
                except Exception:
                    pass
        if cleaned:
            security_logger.info(f"Nettoyage auto : {cleaned} fichier(s) supprimé(s)")

threading.Thread(target=_cleanup_loop, daemon=True).start()

# ==============================================================================
# ROUTES — AUTHENTIFICATION
# ==============================================================================

@app.errorhandler(413)
def file_too_large(e):
    flash('Fichier trop volumineux. Taille maximale : 50 MB.', 'danger')
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
@csrf_required
def login():
    if request.method == 'POST':
        ip = get_client_ip()

        if is_rate_limited(ip, 'login'):
            flash('Trop de tentatives. Patientez 1 minute.', 'danger')
            return render_template('login.html'), 429

        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Veuillez remplir tous les champs.', 'danger')
            return render_template('login.html')

        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username=? AND is_active=1', (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            # Verifier si email est valide (is_active=0 = en attente de verification)
            if not user['is_active']:
                session['pending_verify_user_id'] = user['id']
                session['pending_verify_email']   = user['email']
                flash('Veuillez verifier votre email avant de vous connecter.', 'warning')
                return redirect(url_for('verify_email_page'))

            # Prévention session fixation
            old_csrf = session.get('csrf_token')
            session.clear()
            session['csrf_token'] = old_csrf or secrets.token_hex(32)
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            session.permanent   = True
            update_last_login(user['id'])
            log_activity(user['id'], 'LOGIN', f'Connexion depuis {ip}', ip)
            flash(f"Bienvenue, {user['full_name'] or user['username']} !", 'success')
            return redirect(url_for('admin_dashboard') if user['role'] == 'admin' else url_for('index'))
        else:
            flash('Identifiants invalides.', 'danger')
            security_logger.warning(f"Échec login — user: {username!r}, IP: {ip}")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
@csrf_required
def register():
    if request.method == 'POST':
        ip = get_client_ip()

        if is_rate_limited(ip, 'register'):
            flash("Trop d'inscriptions depuis cette adresse. Réessayez dans 5 minutes.", 'danger')
            return render_template('register.html'), 429

        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip()
        password  = request.form.get('password', '')
        full_name = request.form.get('full_name', '').strip()

        if len(username) < 3:
            flash('Le nom d\'utilisateur doit contenir au moins 3 caractères.', 'danger')
            return redirect(url_for('register'))
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            flash('Le nom d\'utilisateur ne peut contenir que des lettres, chiffres et _.', 'danger')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Le mot de passe doit contenir au moins 6 caractères.', 'danger')
            return redirect(url_for('register'))

        conn = get_db()
        existing = conn.execute(
            'SELECT id FROM users WHERE username=? OR email=?', (username, email)
        ).fetchone()

        if existing:
            flash('Nom d\'utilisateur ou email déjà utilisé.', 'danger')
            conn.close()
            return redirect(url_for('register'))

        # Compte créé inactif — activation après vérification email
        conn.execute(
            'INSERT INTO users (username, email, password_hash, full_name, role, is_active, email_verified) VALUES (?,?,?,?,?,?,?)',
            (username, email, generate_password_hash(password), full_name, 'user', 0, 0)
        )
        conn.commit()
        user_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.close()

        log_activity(user_id, 'REGISTER', 'Nouveau compte en attente de verification', ip)

        # Envoyer le code de verification
        code = create_verification_code(user_id, purpose='verify')
        html = (
            '<div style="font-family:sans-serif;max-width:500px;margin:auto;padding:30px">'
            '<h2 style="color:#1a2a6c">Bienvenue sur SmartClean !</h2>'
            '<p>Merci de vous etre inscrit(e). Pour activer votre compte, entrez ce code :</p>'
            '<div style="font-size:2.5rem;font-weight:bold;letter-spacing:10px;color:#1a2a6c;'
            'padding:25px;background:#f0f4ff;border-radius:10px;text-align:center;margin:20px 0">' + code + '</div>'
            '<p style="color:#666">Ce code expire dans <strong>15 minutes</strong>.</p>'
            '<p style="color:#999;font-size:.85rem">Si vous n avez pas cree de compte, ignorez cet email.</p>'
            '</div>'
        )
        ok = send_email(email, 'SmartClean - Activez votre compte', html)
        if not ok:
            # SMTP non configure : activer directement (mode dev)
            conn2 = get_db()
            conn2.execute('UPDATE users SET is_active=1, email_verified=1 WHERE id=?', (user_id,))
            conn2.commit()
            conn2.close()
            flash('Compte cree avec succes ! (Email SMTP non configure - compte active directement)', 'success')
            return redirect(url_for('login'))

        # Stocker user_id en session pour la page de verification
        session['pending_verify_user_id'] = user_id
        session['pending_verify_email']   = email
        flash('Un code de verification a ete envoye a ' + email + '.', 'info')
        return redirect(url_for('verify_email_page'))

    return render_template('register.html')


@app.route('/verify-email', methods=['GET', 'POST'])
@csrf_required
def verify_email_page():
    user_id = session.get('pending_verify_user_id')
    email   = session.get('pending_verify_email', '')
    if not user_id:
        flash('Session expiree. Veuillez vous inscrire a nouveau.', 'danger')
        return redirect(url_for('register'))

    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        if verify_code(user_id, code, purpose='verify'):
            conn = get_db()
            conn.execute('UPDATE users SET is_active=1, email_verified=1 WHERE id=?', (user_id,))
            conn.commit()
            conn.close()
            log_activity(user_id, 'EMAIL_VERIFIED', 'Compte active')
            session.pop('pending_verify_user_id', None)
            session.pop('pending_verify_email', None)
            flash('Email verifie ! Vous pouvez maintenant vous connecter.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Code invalide ou expire. Reessayez.', 'danger')

    return render_template('verify_email.html', email=email, user_id=user_id)


@app.route('/resend-verification', methods=['POST'])
@csrf_required
def resend_verification():
    user_id = session.get('pending_verify_user_id')
    email   = session.get('pending_verify_email', '')
    if not user_id or not email:
        flash('Session expiree.', 'danger')
        return redirect(url_for('register'))
    code = create_verification_code(user_id, purpose='verify')
    html = (
        '<div style="font-family:sans-serif;max-width:500px;margin:auto;padding:30px">'
        '<h2 style="color:#1a2a6c">SmartClean - Nouveau code</h2>'
        '<p>Votre nouveau code de verification :</p>'
        '<div style="font-size:2.5rem;font-weight:bold;letter-spacing:10px;color:#1a2a6c;'
        'padding:25px;background:#f0f4ff;border-radius:10px;text-align:center;margin:20px 0">' + code + '</div>'
        '<p style="color:#666">Expire dans <strong>15 minutes</strong>.</p>'
        '</div>'
    )
    ok = send_email(email, 'SmartClean - Nouveau code de verification', html)
    if ok:
        flash('Nouveau code envoye a ' + email, 'info')
    else:
        flash('Erreur SMTP. Verifiez votre config .env', 'danger')
    return redirect(url_for('verify_email_page'))


@app.route('/logout')
def logout():
    if 'user_id' in session:
        log_activity(session['user_id'], 'LOGOUT', 'Déconnexion', get_client_ip())
    session.clear()
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('login'))

# ==============================================================================
# ROUTES — APPLICATION PRINCIPALE
# ==============================================================================

@app.route('/')
@login_required
def index():
    user     = get_user_by_id(session['user_id'])
    settings = get_user_settings(session['user_id'])
    return render_template('Nyx_ax_v3.1.html', user=user, settings=settings)


@app.route('/history')
@login_required
def history():
    limit   = request.args.get('limit', 50, type=int)
    user_id = session['user_id'] if session.get('role') != 'admin' else None
    return jsonify(get_history(user_id=user_id, limit=limit))


@app.route('/statistics')
@login_required
def statistics():
    user_id = session['user_id'] if session.get('role') != 'admin' else None
    return jsonify(get_statistics(user_id=user_id))


@app.route('/delete_history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history_item(history_id):
    try:
        conn = get_db()
        record = conn.execute(
            'SELECT user_id FROM processing_history WHERE id=?', (history_id,)
        ).fetchone()
        if not record:
            return jsonify({'success': False, 'error': 'Introuvable'}), 404
        if record['user_id'] != session['user_id'] and session.get('role') != 'admin':
            return jsonify({'success': False, 'error': 'Non autorisé'}), 403
        conn.execute('DELETE FROM processing_history WHERE id=?', (history_id,))
        conn.commit()
        conn.close()
        log_activity(session['user_id'], 'DELETE_HISTORY', f'Item {history_id} supprimé')
        return jsonify({'success': True})
    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] delete_history: {e}")
        return jsonify({'success': False, 'error': f'Erreur ({error_id})'}), 500


@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    try:
        conn = get_db()
        if session.get('role') == 'admin':
            conn.execute('DELETE FROM processing_history')
        else:
            conn.execute('DELETE FROM processing_history WHERE user_id=?', (session['user_id'],))
        conn.commit()
        conn.close()
        log_activity(session['user_id'], 'CLEAR_HISTORY', 'Historique vidé')
        return jsonify({'success': True})
    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] clear_history: {e}")
        return jsonify({'success': False, 'error': f'Erreur ({error_id})'}), 500

# ==============================================================================
# ROUTES — ADMIN
# ==============================================================================

@app.route('/admin')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', user=get_user_by_id(session['user_id']))

@app.route('/admin/statistics')
@admin_required
def admin_statistics():
    return jsonify(get_admin_statistics())

@app.route('/admin/toggle_user/<int:user_id>', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    try:
        conn = get_db()
        row = conn.execute('SELECT is_active FROM users WHERE id=?', (user_id,)).fetchone()
        new_status = 0 if row['is_active'] else 1
        conn.execute('UPDATE users SET is_active=? WHERE id=?', (new_status, user_id))
        conn.commit()
        conn.close()
        action = 'ACTIVATE_USER' if new_status else 'DEACTIVATE_USER'
        log_activity(session['user_id'], action, f'User {user_id}')
        return jsonify({'success': True, 'new_status': new_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/delete_user/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    if user_id == session['user_id']:
        return jsonify({'success': False, 'error': 'Impossible de supprimer votre propre compte'}), 400
    try:
        conn = get_db()
        for table in ['processing_history', 'activity_logs', 'user_settings']:
            conn.execute(f'DELETE FROM {table} WHERE user_id=?', (user_id,))
        conn.execute('DELETE FROM users WHERE id=?', (user_id,))
        conn.commit()
        conn.close()
        log_activity(session['user_id'], 'DELETE_USER', f'User {user_id} supprimé')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==============================================================================
# ROUTES — PREVIEW & TÉLÉCHARGEMENT
# ==============================================================================

@app.route('/preview', methods=['POST'])
@login_required
@csrf_required
def preview():
    start_time = time.time()
    ip = get_client_ip()

    if is_rate_limited(ip, 'preview'):
        flash('Trop de fichiers envoyés. Attendez une minute.', 'danger')
        return redirect(url_for('index'))

    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Aucun fichier sélectionné.', 'danger')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            flash('Format non supporté. Formats acceptés : CSV, Excel, JSON, XML.', 'danger')
            return redirect(url_for('index'))

        safe_name = secure_filename(file.filename)
        file_ext  = safe_name.rsplit('.', 1)[1].lower()

        # Taille réelle
        file.seek(0, os.SEEK_END)
        file_size_kb = round(file.tell() / 1024, 2)
        file.seek(0)

        # Validation du contenu
        if not validate_file_content(file, file_ext):
            flash('Le contenu du fichier ne correspond pas à son extension.', 'danger')
            security_logger.warning(f"Upload refusé (contenu invalide) — {safe_name}, IP: {ip}")
            return redirect(url_for('index'))

        config = {
            'duplicates':     'duplicates'     in request.form,
            'missing_values': 'missing_values' in request.form,
            'outliers':       'outliers'        in request.form,
            'normalize':      'normalize'       in request.form,
            'format':         request.form.get('output_format', 'csv'),
        }
        if config['format'] not in ('csv', 'xlsx', 'json'):
            config['format'] = 'csv'

        # Lecture du fichier
        if file_ext == 'csv':
            df = pd.read_csv(file, na_values=EXTRA_NA, keep_default_na=True)
        elif file_ext in ('xlsx', 'xls'):
            df = pd.read_excel(file, na_values=EXTRA_NA, keep_default_na=True)
        elif file_ext == 'json':
            try:
                df = pd.read_json(file)
            except ValueError:
                file.seek(0)
                df = pd.read_json(file, orient='records')
        elif file_ext == 'xml':
            df = pd.read_xml(file)

        original_shape  = df.shape
        df_clean, changes = process_data(df.copy(), config)
        cleaned_shape   = df_clean.shape

        # Sauvegarde temporaire sur disque (évite la limite 4KB du cookie)
        temp_id   = str(uuid.uuid4())
        temp_path = os.path.join(TEMP_DIR, f"{temp_id}.pkl")
        df_clean.to_pickle(temp_path)
        session['preview_temp_id']  = temp_id
        session['preview_format']   = config['format']
        session['preview_filename'] = safe_name

        # Statistiques
        stats = {
            'original_rows': original_shape[0],
            'cleaned_rows':  cleaned_shape[0],
            'removed_rows':  original_shape[0] - cleaned_shape[0],
            'columns':       cleaned_shape[1],
        }

        processing_time = round(time.time() - start_time, 2)

        # Enregistrement dans l'historique
        save_to_history({
            'filename':       safe_name,
            'original_rows':  original_shape[0],
            'processed_rows': cleaned_shape[0],
            'original_cols':  original_shape[1],
            'processed_cols': cleaned_shape[1],
            'missing':        config['missing_values'],
            'duplicates':     config['duplicates'],
            'outliers':       config['outliers'],
            'normalize':      config['normalize'],
            'format':         config['format'],
            'file_size_kb':   file_size_kb,
            'processing_time': processing_time,
        }, session['user_id'])

        log_activity(session['user_id'], 'PREVIEW', f'Preview de {safe_name}', ip)

        user_settings     = get_user_settings(session['user_id'])
        preview_rows_count = int(user_settings.get('preview_rows', 10))

        def df_display(frame, n):
            return frame.head(n).astype(object).where(frame.head(n).notna(), other='N/A').values.tolist()
        return render_template('preview.html',
                               stats=stats,
                               changes=changes,
                               columns=df_clean.columns.tolist(),
                               preview_data=df_display(df_clean, preview_rows_count),
                               before_cols=df.columns.tolist(),
                               before_data=df_display(df, preview_rows_count),
                               format=config['format'])

    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] preview user={session.get('username')}: {e}", exc_info=True)
        log_activity(session['user_id'], 'PREVIEW_ERROR', f'[{error_id}] {type(e).__name__}')
        flash(f'Erreur lors du traitement (réf : {error_id}). Vérifiez votre fichier.', 'danger')
        return redirect(url_for('index'))


@app.route('/download_preview')
@login_required
def download_preview():
    try:
        if 'preview_temp_id' not in session:
            flash('Aucune donnée à télécharger. Veuillez d\'abord uploader un fichier.', 'warning')
            return redirect(url_for('index'))

        temp_path = os.path.join(TEMP_DIR, f"{session['preview_temp_id']}.pkl")
        if not os.path.exists(temp_path):
            flash('Session de preview expirée. Veuillez re-uploader votre fichier.', 'warning')
            return redirect(url_for('index'))

        df          = pd.read_pickle(temp_path)
        fmt         = session.get('preview_format', 'csv')
        orig_name   = session.get('preview_filename', 'data.csv')

        if fmt not in ('csv', 'xlsx', 'json'):
            fmt = 'csv'

        base_name   = secure_filename(orig_name).rsplit('.', 1)[0]
        out_name    = f"{base_name}_cleaned.{fmt}"
        out_path    = os.path.join(UPLOAD_DIR, out_name)

        if fmt == 'csv':
            df.to_csv(out_path, index=False)
        elif fmt == 'xlsx':
            df.to_excel(out_path, index=False)
        elif fmt == 'json':
            df.to_json(out_path, orient='records', indent=2)

        log_activity(session['user_id'], 'DOWNLOAD', f'Téléchargement de {out_name}')

        # Nettoyage
        try:
            os.remove(temp_path)
        except Exception:
            pass
        for key in ('preview_temp_id', 'preview_format', 'preview_filename'):
            session.pop(key, None)

        return send_file(out_path, as_attachment=True, download_name=out_name)

    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error(f"[{error_id}] download user={session.get('username')}: {e}", exc_info=True)
        flash(f'Erreur lors du téléchargement (réf : {error_id}).', 'danger')
        return redirect(url_for('index'))

# ==============================================================================
# ROUTES — PARAMÈTRES
# ==============================================================================

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html',
                           user=get_user_by_id(session['user_id']),
                           settings=get_user_settings(session['user_id']))

@app.route('/save_settings', methods=['POST'])
@login_required
@csrf_required
def save_settings():
    try:
        preview_rows = int(request.form.get('preview_rows', 10))
        if preview_rows not in (5, 10, 20, 50):
            preview_rows = 10
        default_format = request.form.get('default_format', 'csv')
        if default_format not in ('csv', 'xlsx', 'json'):
            default_format = 'csv'

        save_user_settings(session['user_id'], {
            'always_preview':     'always_preview'    in request.form,
            'save_history':       'save_history'       in request.form,
            'confirm_delete':     'confirm_delete'     in request.form,
            'default_missing':    'default_missing'    in request.form,
            'default_duplicates': 'default_duplicates' in request.form,
            'default_outliers':   'default_outliers'   in request.form,
            'default_normalize':  'default_normalize'  in request.form,
            'default_format':     default_format,
            'preview_rows':       preview_rows,
        })
        log_activity(session['user_id'], 'SETTINGS_UPDATED', 'Paramètres mis à jour')
        flash('Paramètres sauvegardés.', 'success')
    except Exception as e:
        security_logger.error(f"save_settings user={session.get('username')}: {e}")
        flash('Erreur lors de la sauvegarde des paramètres.', 'danger')

    return redirect(url_for('settings'))


# ==============================================================================
# ROUTES — MOT DE PASSE & VERIFICATION EMAIL
# ==============================================================================

@app.route('/change_password', methods=['POST'])
@login_required
@csrf_required
def change_password():
    current_pw = request.form.get('current_password', '')
    new_pw     = request.form.get('new_password', '')
    confirm_pw = request.form.get('confirm_password', '')
    user = get_user_by_id(session['user_id'])
    if not check_password_hash(user['password_hash'], current_pw):
        flash('Mot de passe actuel incorrect.', 'danger')
        return redirect(url_for('settings'))
    if len(new_pw) < 6:
        flash('Le nouveau mot de passe doit contenir au moins 6 caracteres.', 'danger')
        return redirect(url_for('settings'))
    if new_pw != confirm_pw:
        flash('Les mots de passe ne correspondent pas.', 'danger')
        return redirect(url_for('settings'))
    conn = get_db()
    conn.execute('UPDATE users SET password_hash=? WHERE id=?', (generate_password_hash(new_pw), session['user_id']))
    conn.commit()
    conn.close()
    log_activity(session['user_id'], 'PASSWORD_CHANGED', 'Mot de passe modifie', get_client_ip())
    flash('Mot de passe modifie avec succes.', 'success')
    return redirect(url_for('settings'))


@app.route('/send_verification_email', methods=['POST'])
@login_required
@csrf_required
def send_verification_email():
    user = get_user_by_id(session['user_id'])
    if user.get('email_verified'):
        flash('Votre email est deja verifie.', 'info')
        return redirect(url_for('settings'))
    code = create_verification_code(session['user_id'], purpose='verify')
    html = (
        '<div style="font-family:sans-serif;max-width:500px;margin:auto">'
        '<h2 style="color:#1a2a6c">SmartClean - Verification email</h2>'
        '<p>Votre code de verification est :</p>'
        '<div style="font-size:2rem;font-weight:bold;letter-spacing:8px;color:#1a2a6c;'
        'padding:20px;background:#f0f4ff;border-radius:8px;text-align:center">' + code + '</div>'
        '<p style="color:#666">Ce code expire dans <strong>15 minutes</strong>.</p>'
        '</div>'
    )
    ok = send_email(user['email'], 'SmartClean - Code de verification', html)
    if ok:
        flash('Code envoye a ' + user['email'] + '. Verifiez votre boite mail.', 'success')
    else:
        flash('Erreur envoi email. Verifiez la config SMTP dans .env.', 'danger')
    return redirect(url_for('settings'))


@app.route('/confirm_verification', methods=['POST'])
@login_required
@csrf_required
def confirm_verification():
    code = request.form.get('verification_code', '').strip()
    if verify_code(session['user_id'], code, purpose='verify'):
        conn = get_db()
        conn.execute('UPDATE users SET email_verified=1 WHERE id=?', (session['user_id'],))
        conn.commit()
        conn.close()
        log_activity(session['user_id'], 'EMAIL_VERIFIED', 'Email verifie')
        flash('Email verifie avec succes !', 'success')
    else:
        flash('Code invalide ou expire.', 'danger')
    return redirect(url_for('settings'))


# ==============================================================================
# ROUTES — ANALYSE DES DONNEES
# ==============================================================================

@app.route('/analyze', methods=['POST'])
@login_required
@csrf_required
def analyze():
    ip = get_client_ip()
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'error': 'Aucun fichier'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format non supporte'}), 400
        safe_name = secure_filename(file.filename)
        file_ext  = safe_name.rsplit('.', 1)[1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file, na_values=EXTRA_NA, keep_default_na=True)
        elif file_ext in ('xlsx', 'xls'):
            df = pd.read_excel(file, na_values=EXTRA_NA, keep_default_na=True)
        elif file_ext == 'json':
            try:
                df = pd.read_json(file)
            except ValueError:
                file.seek(0)
                df = pd.read_json(file, orient='records')
        elif file_ext == 'xml':
            df = pd.read_xml(file)
        else:
            return jsonify({'error': 'Format non supporte'}), 400

        col_stats = []
        for col in df.columns:
            s = df[col]
            stat = {
                'name':        col,
                'dtype':       str(s.dtype),
                'total':       len(s),
                'missing':     int(s.isna().sum()),
                'missing_pct': round(s.isna().mean() * 100, 1),
                'unique':      int(s.nunique()),
            }
            if pd.api.types.is_numeric_dtype(s):
                nn = s.dropna()
                if len(nn):
                    stat.update({
                        'min':    round(float(nn.min()), 4),
                        'max':    round(float(nn.max()), 4),
                        'mean':   round(float(nn.mean()), 4),
                        'median': round(float(nn.median()), 4),
                        'std':    round(float(nn.std()), 4),
                    })
            else:
                top = s.dropna().value_counts().head(3).to_dict()
                stat['top_values'] = {str(k): int(v) for k, v in top.items()}
            col_stats.append(stat)

        summary = {
            'filename':      safe_name,
            'rows':          len(df),
            'cols':          len(df.columns),
            'total_missing': int(df.isna().sum().sum()),
            'duplicates':    int(df.duplicated().sum()),
            'columns':       col_stats,
        }
        log_activity(session['user_id'], 'ANALYZE', 'Analyse de ' + safe_name, ip)
        return jsonify(summary)
    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error('[' + error_id + '] analyze: ' + str(e), exc_info=True)
        return jsonify({'error': 'Erreur (' + error_id + ')'}), 500


# ==============================================================================
# ROUTES — CONVERSION DE FICHIERS
# ==============================================================================

CONVERSION_FORMATS = {'csv', 'xlsx', 'json', 'xml', 'tsv', 'parquet'}

@app.route('/convert', methods=['GET', 'POST'])
@login_required
def convert():
    if request.method == 'GET':
        return render_template('convert.html')
    ip = get_client_ip()
    tmp_path = None
    try:
        file       = request.files.get('file')
        target_fmt = request.form.get('target_format', 'csv').lower()

        if not file or file.filename == '':
            flash('Aucun fichier selectionne.', 'danger')
            return redirect(url_for('convert'))
        if target_fmt not in CONVERSION_FORMATS:
            flash('Format cible non supporte.', 'danger')
            return redirect(url_for('convert'))

        safe_name = secure_filename(file.filename)
        src_ext   = safe_name.rsplit('.', 1)[-1].lower() if '.' in safe_name else ''

        if src_ext not in CONVERSION_FORMATS:
            flash('Format source non supporte : .' + src_ext, 'danger')
            return redirect(url_for('convert'))

        # Sauvegarder dans un fichier temp — pd.read_xml/parquet ne
        # supportent pas directement les objets FileStorage de Flask
        tmp_path = os.path.join(TEMP_DIR, str(uuid.uuid4()) + '.' + src_ext)
        file.save(tmp_path)

        # Lecture selon le format source
        if src_ext == 'csv':
            df = pd.read_csv(tmp_path)
        elif src_ext in ('xlsx', 'xls'):
            df = pd.read_excel(tmp_path)
        elif src_ext == 'json':
            try:
                df = pd.read_json(tmp_path)
            except ValueError:
                df = pd.read_json(tmp_path, orient='records')
        elif src_ext == 'xml':
            try:
                df = pd.read_xml(tmp_path)
            except Exception:
                import xml.etree.ElementTree as ET
                tree = ET.parse(tmp_path)
                root = tree.getroot()
                children = list(root)
                if children:
                    df = pd.read_xml(tmp_path, xpath='./' + children[0].tag)
                else:
                    raise ValueError('Structure XML non supportee')
        elif src_ext == 'tsv':
            df = pd.read_csv(tmp_path, sep='\t')
        elif src_ext == 'parquet':
            df = pd.read_parquet(tmp_path)
        else:
            flash('Format source non reconnu : ' + src_ext, 'danger')
            return redirect(url_for('convert'))

        # Normaliser les types pandas 2.x (StringDtype -> object)
        # pour assurer la compatibilité avec openpyxl, to_xml, etc.
        for col in df.columns:
            if str(df[col].dtype) in ('string', 'StringDtype') or hasattr(df[col].dtype, 'na_value'):
                df[col] = df[col].astype(object)

        # Ecriture dans le format cible
        base_name = safe_name.rsplit('.', 1)[0]
        out_name  = base_name + '_converted.' + target_fmt
        out_path  = os.path.join(UPLOAD_DIR, out_name)

        if target_fmt == 'csv':
            df.to_csv(out_path, index=False)
        elif target_fmt == 'xlsx':
            df.to_excel(out_path, index=False, engine='openpyxl')
        elif target_fmt == 'json':
            df.to_json(out_path, orient='records', indent=2, force_ascii=False)
        elif target_fmt == 'xml':
            df.to_xml(out_path, index=False)
        elif target_fmt == 'tsv':
            df.to_csv(out_path, sep='\t', index=False)
        elif target_fmt == 'parquet':
            df.to_parquet(out_path, index=False)

        log_activity(session['user_id'], 'CONVERT', safe_name + ' -> ' + target_fmt, ip)
        return send_file(out_path, as_attachment=True, download_name=out_name)

    except Exception as e:
        error_id = uuid.uuid4().hex[:8].upper()
        security_logger.error('[' + error_id + '] convert: ' + str(e), exc_info=True)
        flash('Erreur conversion (ref: ' + error_id + ') : ' + str(e)[:120], 'danger')
        return redirect(url_for('convert'))
    finally:
        # Nettoyer le fichier temp
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ==============================================================================
# LANCEMENT
# ==============================================================================

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    if debug:
        security_logger.warning("FLASK_DEBUG activé — ne pas utiliser en production !")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug)
