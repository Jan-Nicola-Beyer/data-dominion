"""Project persistence — settings, autosave, logging, and session state.

Handles:
  - Settings persistence (theme, row limit) in a JSON file
  - Project autosave/restore (tags, slices, dataset metadata)
  - Background logging to a rotating log file
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR = pathlib.Path.home() / ".datalens"
APP_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = APP_DIR / "settings.json"
AUTOSAVE_FILE = APP_DIR / "autosave.json"
LOG_FILE      = APP_DIR / "datalens.log"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "theme":     "dark",
    "row_limit": 2000,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Settings (persisted between sessions)
# ══════════════════════════════════════════════════════════════════════════════

def load_settings() -> dict:
    """Load settings from disk, falling back to defaults."""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge with defaults so new keys are always present
            merged = {**DEFAULT_SETTINGS, **saved}
            return merged
    except Exception:
        pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    """Write settings to disk."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as exc:
        get_logger().warning("Failed to save settings: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
#  Autosave (project state recovery)
# ══════════════════════════════════════════════════════════════════════════════

def save_project_state(state: dict) -> None:
    """Save project state (tags, slices, dataset metadata) to autosave file.

    The *state* dict should contain JSON-serialisable data only.
    DataFrames are NOT saved here — only metadata and lightweight state.
    """
    try:
        state["_timestamp"] = time.time()
        state["_version"]   = 1
        with open(AUTOSAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        get_logger().info("Autosaved project state (%d bytes)", AUTOSAVE_FILE.stat().st_size)
    except Exception as exc:
        get_logger().warning("Autosave failed: %s", exc)


def load_project_state() -> Optional[dict]:
    """Load the last autosaved project state, or None if unavailable."""
    try:
        if AUTOSAVE_FILE.exists():
            with open(AUTOSAVE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state
    except Exception as exc:
        get_logger().warning("Failed to load autosave: %s", exc)
    return None


def has_autosave() -> bool:
    """Check if an autosave file exists and is recent (< 24 hours)."""
    try:
        if not AUTOSAVE_FILE.exists():
            return False
        age = time.time() - AUTOSAVE_FILE.stat().st_mtime
        return age < 86400  # 24 hours
    except Exception:
        return False


def clear_autosave() -> None:
    """Remove the autosave file."""
    try:
        if AUTOSAVE_FILE.exists():
            AUTOSAVE_FILE.unlink()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════════════

_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Return the app-wide logger, creating it on first call.

    Writes to ~/.datalens/datalens.log with rotation (2 MB max, 3 backups).
    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("datalens")
        _logger.setLevel(logging.INFO)
        # Avoid duplicate handlers on reimport
        if not _logger.handlers:
            handler = RotatingFileHandler(
                str(LOG_FILE), maxBytes=2 * 1024 * 1024, backupCount=3,
                encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-7s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"))
            _logger.addHandler(handler)
        _logger.info("Logger initialised — log file: %s", LOG_FILE)
    return _logger
