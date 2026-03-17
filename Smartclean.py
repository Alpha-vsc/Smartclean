"""WSGI entrypoint compatibility module.

Allows process managers configured with `gunicorn Smartclean:app` to
start the Flask application without changing provider-level start commands.
"""

from app import create_app

app = create_app()
