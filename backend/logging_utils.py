"""Bounded application diagnostics shared by CLI, desktop and Streamlit."""
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading

_lock = threading.Lock()


def log_path():
    return Path(os.environ.get("DATAYAB_LOG_DIR", Path(__file__).resolve().parents[1] / "logs")) / "datayab.log"


def get_logger():
    logger = logging.getLogger("datayab")
    with _lock:
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logger.propagate = False
            try:
                path = log_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            except OSError:
                handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
            logger.addHandler(handler)
    return logger


def recent_log(limit=64_000):
    """Read only the bounded tail; never load a whole log into the UI."""
    try:
        with log_path().open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return "No file log available. Check the application console."
