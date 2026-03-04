# ==============================================================================
# SmartClean — Backend v3.2 (Render-ready)
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

file_handler = logging.FileHandler('smartclean_security.log')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

security_logger = logging.getLogger('security')
security_logger.addHandler(file_handler)
security_logger.setLevel(logging.WARNING)

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)

# ==============================================================================
# INITIALISATION FLASK
# ==============================================================================
import os
app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

_secret = os.environ.get('SECRET_KEY')
if not _secret:
    _secret = secrets.token_hex(32)
    security_logger.warning(
        "SECRET_KEY non définie. Clé aléatoire utilisée. "
        "Définissez SECRET_KEY dans vos variables d'environnement Render."
    )
app.secret_key = _secret

app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('HTTPS', 'false') == 'true'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

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
# Render : les fichiers écrits hors du disque persistant sont perdus au redémarrage.
# Le disque persistant est monté sur /opt/render/project/src (défini dans render.yaml).
# ==============================================================================

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
TEMP_DIR   = os.path.join(BASE_DIR, 'temp_previews')
DB_PATH    = os.path.join(BASE_DIR, 'smartclean.db')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

# ==============================================================================
# SÉCURITÉ FICHIERS — EXTENSIONS & MAGIC BYTES
# ==============================================================================

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'xml'}

FILE_SIGNATURES = {
    'xlsx': b'PK\x03\x04',
    'xls':  b'\xd0\xcf\x11\xe0',
}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_content(file, extension: str) -> bool:
    sig = FILE_SIGNATURES.get(extension)
    if sig:
        header = file.read(len(sig))
        file.seek(0)
        return header == sig
    else:
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
# RATE LIMITING
# ==============================================================================

_rl_store = defaultdict(list)
_rl_lock  = threading.Lock()

RATE_LIMITS = {
    'login':    (5,  60),
    'register': (3,  300),
    'preview':  (20, 60),
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
# PROTECTION CSRF
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

app.jinja_env.globals['csrf_token'] = generate_csrf_token

# ==============================================================================
# BASE DE DONNÉES — INITIALISATION
# ==============================================================================

def ensure_db_writable():
    if os.path.exists(DB_PATH) and not os.access(DB_PATH, os.W_OK):
        try:
            os.chmod(DB_PATH, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        except Exception as e:
            raise PermissionError(f"Base de données en lecture seule : {DB_PATH}\nErreur : {e}")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
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
            is_active     BOOLEAN DEFAULT 1
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

    cur.execute('SELECT COUNT(*) FROM users WHERE role = "admin"')
    if cur.fetchone()[0] == 0:
        pwd = os.environ.get('ADMIN_PASSWORD')
        if not pwd:
            pwd = secrets.token_urlsafe(16)
            print("=" * 60)
            print("⚠️  ADMIN_PASSWORD non défini dans les variables d'environnement")
            print(f"   Mot de passe généré : {pwd}")
            print("   Ajoutez ADMIN_PASSWORD dans vos variables Render")
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
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter.', 'warning')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
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
    changes = []
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)

    for col in df.columns:
        str_vals = df[col].apply(lambda v: '' if pd.isna(v) else str(v).strip().lower())
        mask = str_vals.isin(NA_VARIANTS)
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            changes.append(f"'{col}' : {n} valeur(s) non-standard (na, n/a, --…) → NaN")

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

    return df, changes

def process_data(df, config):
    changes = []

    df, na_changes = normalize_missing(df)
    changes.extend(na_changes)

    if config.get('duplicates'):
        before = len(df)
        df = df.drop_duplicates()
        n = before - len(df)
        if n:
            changes.append(f"{n} doublon(s) supprimé(s)")

    if config.get('missing_values'):
        filled = 0
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                modes = df[col].dropna().mode()
                df[col] = df[col].fillna(modes.iloc[0] if not modes.empty else 'N/A')
            filled += n_missing
        if filled:
            changes.append(f"{filled} valeur(s) manquante(s) remplissées (moyenne/mode)")

    if config.get('outliers'):
        before = len(df)
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        n = before - len(df)
        if n:
            changes.append(f"{n} outlier(s) supprimé(s) (méthode IQR)")

    if config.get('normalize'):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
            changes.append(f"Normalisation Min-Max appliquée ({len(num_cols)} colonne(s))")

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

        conn.execute(
            'INSERT INTO users (username, email, password_hash, full_name, role) VALUES (?,?,?,?,?)',
            (username, email, generate_password_hash(password), full_name, 'user')
        )
        conn.commit()
        user_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.close()

        log_activity(user_id, 'REGISTER', 'Nouveau compte', ip)
        flash('Compte créé avec succès ! Vous pouvez vous connecter.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


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

        file.seek(0, os.SEEK_END)
        file_size_kb = round(file.tell() / 1024, 2)
        file.seek(0)

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

        temp_id   = str(uuid.uuid4())
        temp_path = os.path.join(TEMP_DIR, f"{temp_id}.pkl")
        df_clean.to_pickle(temp_path)
        session['preview_temp_id']  = temp_id
        session['preview_format']   = config['format']
        session['preview_filename'] = safe_name

        stats = {
            'original_rows': original_shape[0],
            'cleaned_rows':  cleaned_shape[0],
            'removed_rows':  original_shape[0] - cleaned_shape[0],
            'columns':       cleaned_shape[1],
        }

        processing_time = round(time.time() - start_time, 2)

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

        user_settings      = get_user_settings(session['user_id'])
        preview_rows_count = int(user_settings.get('preview_rows', 10))

        return render_template('preview.html',
                               stats=stats,
                               changes=changes,
                               columns=df_clean.columns.tolist(),
                               preview_data=df_clean.head(preview_rows_count).values.tolist(),
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
# LANCEMENT — Render utilise gunicorn, ce bloc ne s'exécute pas en prod
# ==============================================================================

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug)
