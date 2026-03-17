# ==============================================================================
# SmartClean — Background Cleanup Utility
# ==============================================================================

import os
import time
import logging
import threading

security_logger = logging.getLogger("security")


def _cleanup_loop(upload_dir: str, temp_dir: str, max_age_min: int = 30, interval_min: int = 15) -> None:
    while True:
        time.sleep(interval_min * 60)
        cutoff  = time.time() - max_age_min * 60
        cleaned = 0
        for folder in (temp_dir, upload_dir):
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                try:
                    if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        cleaned += 1
                except Exception:
                    pass
        if cleaned:
            security_logger.info(f"Auto-cleanup: {cleaned} file(s) removed")


def start_cleanup_thread(upload_dir: str, temp_dir: str) -> None:
    """Start the background cleanup daemon (call once at app startup)."""
    t = threading.Thread(
        target=_cleanup_loop,
        args=(upload_dir, temp_dir),
        daemon=True,
    )
    t.start()
