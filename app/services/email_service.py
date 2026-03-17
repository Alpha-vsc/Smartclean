# ==============================================================================
# SmartClean — Email Service
# ==============================================================================

import os
import logging
import smtplib
from email.mime.text    import MIMEText
from email.mime.multipart import MIMEMultipart

security_logger = logging.getLogger("security")


def send_email(to_email: str, subject: str, body_html: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    from_addr = os.environ.get("FROM_EMAIL", smtp_user)

    if not smtp_user or not smtp_pass:
        security_logger.warning("SMTP not configured — email not sent")
        return False

    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"SmartClean <{from_addr}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, to_email, msg.as_string())
        return True
    except Exception as e:
        security_logger.error(f"SMTP error: {e}")
        return False


def _verification_html(code: str, title: str = "SmartClean") -> str:
    return (
        '<div style="font-family:sans-serif;max-width:500px;margin:auto;padding:30px">'
        f'<h2 style="color:#1a2a6c">{title}</h2>'
        "<p>Your verification code:</p>"
        '<div style="font-size:2.5rem;font-weight:bold;letter-spacing:10px;color:#1a2a6c;'
        "padding:25px;background:#f0f4ff;border-radius:10px;text-align:center;margin:20px 0\">"
        f"{code}</div>"
        '<p style="color:#666">This code expires in <strong>15 minutes</strong>.</p>'
        "</div>"
    )


def send_verification_email(to_email: str, code: str) -> bool:
    html = _verification_html(code, "Welcome to SmartClean!")
    return send_email(to_email, "SmartClean — Activate your account", html)


def send_new_code_email(to_email: str, code: str) -> bool:
    html = _verification_html(code, "SmartClean — New verification code")
    return send_email(to_email, "SmartClean — New verification code", html)
