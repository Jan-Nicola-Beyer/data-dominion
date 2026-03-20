"""In-app ML package installer for Topic Modelling.

Provides a one-click install button that downloads sentence-transformers
(which pulls torch automatically) plus bertopic, umap-learn, hdbscan, and
scikit-learn into ~/.datalens/ml_packages/.  The app picks them up on
next restart.

Works inside a frozen Nuitka exe by calling pip's internal API directly.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip

# ── Target directory ─────────────────────────────────────────────────────────
ML_DIR = Path.home() / ".datalens" / "ml_packages"

# All packages in one list — pip resolves torch as a dependency automatically
_PACKAGES = [
    "sentence-transformers",
    "bertopic",
    "umap-learn",
    "hdbscan",
    "scikit-learn",
]


def ml_installed() -> bool:
    """Check if the core ML packages are available."""
    for pkg in ("sentence_transformers", "bertopic"):
        if importlib.util.find_spec(pkg) is None:
            # Check bundled (next to exe) and user-installed directories
            bundled = Path(sys.executable).resolve().parent / "ml_packages" / pkg
            user = ML_DIR / pkg
            if not bundled.exists() and not user.exists():
                return False
    return True


def ml_status_text() -> str:
    """Return a short status string for display."""
    return "ML Pack installed" if ml_installed() else "ML Pack not installed"


def _find_pip():
    """Find a working way to invoke pip.

    Returns (callable, description) or (None, None).
    Tries: bundled runtime Python > sys.executable > pip on PATH > python on PATH.
    """
    import logging
    import shutil
    log = logging.getLogger("datalens.ml_install")

    def _subprocess_runner(cmd, label):
        """Create a runner that invokes pip via subprocess with logging."""
        def _run(args):
            full = cmd + args
            log.info("Running: %s", " ".join(full))
            try:
                r = subprocess.run(
                    full, capture_output=True, text=True, timeout=900)
                if r.stdout:
                    for line in r.stdout.strip().splitlines()[-10:]:
                        log.info("pip: %s", line)
                if r.returncode and r.stderr:
                    for line in r.stderr.strip().splitlines()[-10:]:
                        log.error("pip err: %s", line)
                return r.returncode
            except Exception as exc:
                log.error("subprocess failed: %s", exc)
                return 1
        return _run, label

    # 1. Bundled runtime Python (embedded distribution shipped with the app)
    import os
    _script_dir = Path(os.path.abspath(sys.argv[0])).parent
    _app_root = _script_dir.parent
    for candidate in [
        _app_root / "runtime" / "python.exe",   # embedded dist layout
        _script_dir / "runtime" / "python.exe",  # dev mode fallback
    ]:
        if candidate.exists():
            return _subprocess_runner(
                [str(candidate), "-m", "pip"],
                f"bundled runtime ({candidate})")

    # 2. sys.executable -m pip (dev mode)
    if not getattr(sys, "frozen", False):
        return _subprocess_runner(
            [sys.executable, "-m", "pip"], "python -m pip (dev)")

    # 3. pip on PATH
    pip_exe = shutil.which("pip") or shutil.which("pip3")
    if pip_exe:
        return _subprocess_runner([pip_exe], f"pip ({pip_exe})")

    # 4. python on PATH
    py_exe = shutil.which("python") or shutil.which("python3")
    if py_exe:
        return _subprocess_runner(
            [py_exe, "-m", "pip"], f"python -m pip ({py_exe})")

    return None, None


def run_auto_install(status_cb=None, done_cb=None):
    """Run ML package install in a background thread.

    Parameters
    ----------
    status_cb : callable(str) or None
        Called on the *calling* thread with progress messages.
        Caller must dispatch to the GUI thread if needed.
    done_cb : callable(bool) or None
        Called when finished — True on success, False on failure.
    """
    import logging
    log = logging.getLogger("datalens.ml_install")
    # Ensure console output for debugging (visible with --windows-console-mode=force)
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        _ch = logging.StreamHandler()
        _ch.setFormatter(logging.Formatter(
            "%(asctime)s ML-INSTALL %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"))
        log.addHandler(_ch)

    def _worker():
        try:
            ML_DIR.mkdir(parents=True, exist_ok=True)
            log.info("ML target dir: %s", ML_DIR)

            pip_fn, pip_source = _find_pip()
            log.info("pip method: %s", pip_source)
            if pip_fn is None:
                msg = "Could not find pip. Install Python 3.8+ and restart."
                log.error(msg)
                if status_cb:
                    status_cb(msg)
                if done_cb:
                    done_cb(False)
                return

            if status_cb:
                status_cb(f"Downloading ML packages via {pip_source}…")

            rc = pip_fn([
                "install"] + _PACKAGES + [
                "--target", str(ML_DIR),
                "--no-warn-script-location",
            ])
            log.info("pip returned: %s", rc)

            if rc:
                msg = "Install failed — check internet connection."
                log.error(msg)
                if status_cb:
                    status_cb(msg)
                if done_cb:
                    done_cb(False)
                return

            # Make the newly-installed packages importable immediately
            ml_str = str(ML_DIR)
            if ml_str not in sys.path:
                sys.path.insert(0, ml_str)
            log.info("ML packages installed to %s", ml_str)

            if status_cb:
                status_cb("ML packages installed successfully.")
            if done_cb:
                done_cb(True)
        except Exception as exc:
            log.exception("Auto-install failed: %s", exc)
            if status_cb:
                status_cb(f"Install error: {exc}")
            if done_cb:
                done_cb(False)

    threading.Thread(target=_worker, daemon=True).start()


class MLInstallButton(ctk.CTkFrame):
    """Drop-in install button for ML dependencies.

    Shows current status and a one-click installer.
    After install completes, shows 'Restart app to activate'.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=C.CARD, corner_radius=8, **kwargs)
        self._thread = None
        self._done = None
        self._build()

    def _build(self):
        if ml_installed():
            ctk.CTkLabel(self, text="ML Pack installed",
                         text_color=C.SUCCESS, font=("Segoe UI", 11)
                         ).pack(padx=16, pady=10)
            return

        ctk.CTkLabel(self, text="Topic Modelling requires additional packages",
                     text_color=C.TEXT, font=("Segoe UI", 12, "bold")
                     ).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(self,
                     text="Downloads sentence-transformers, bertopic, and dependencies.\n"
                          "Requires internet connection. No Python installation needed.\n"
                          "Installed to: ~/.datalens/ml_packages/",
                     text_color=C.MUTED, font=("Segoe UI", 10),
                     justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 8))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 4))

        self._install_btn = ctk.CTkButton(
            btn_row, text="Install ML Pack", fg_color=C.ACCENT,
            width=160, height=36, font=("Segoe UI", 12, "bold"),
            command=self._start_install)
        self._install_btn.pack(side="left")
        tip(self._install_btn,
            "Download and install ML packages for Topic Modelling.\n"
            "Restart the app after installation to activate.")

        self._status_var = tk.StringVar(value="")
        self._status_lbl = ctk.CTkLabel(
            btn_row, textvariable=self._status_var,
            text_color=C.MUTED, font=("Segoe UI", 10))
        self._status_lbl.pack(side="left", padx=12)

        self._prog_bar = ctk.CTkProgressBar(
            self, width=400, height=10, mode="indeterminate")

    def _start_install(self):
        if self._thread and self._thread.is_alive():
            return
        self._install_btn.configure(state="disabled", fg_color=C.DIM)
        self._status_var.set("Finding pip...")
        self._prog_bar.pack(padx=16, pady=(0, 12))
        self._prog_bar.start()
        self._done = None
        self._thread = threading.Thread(target=self._install_worker, daemon=True)
        self._thread.start()
        self._poll()

    def _install_worker(self):
        """Install ML packages using the best available pip method."""
        ML_DIR.mkdir(parents=True, exist_ok=True)

        pip_fn, pip_source = _find_pip()
        if pip_fn is None:
            self._set_status(
                "Could not find pip. Install Python 3.8+ from python.org and restart.")
            self._done = "error"
            return

        self._set_status(f"Downloading packages via {pip_source}... this may take several minutes")

        rc = pip_fn([
            "install"] + _PACKAGES + [
            "--target", str(ML_DIR),
            "--quiet",
            "--no-warn-script-location",
        ])

        if rc:
            self._set_status("Error installing packages. Check your internet connection and try again.")
            self._done = "error"
            return

        # Make the newly-installed packages importable immediately
        ml_str = str(ML_DIR)
        if ml_str not in sys.path:
            sys.path.insert(0, ml_str)

        self._done = "ok"

    def _set_status(self, msg):
        self.after(0, lambda m=msg: self._status_var.set(m))

    def _poll(self):
        if self._done is not None:
            self._prog_bar.stop()
            self._prog_bar.pack_forget()
            if self._done == "ok":
                self._status_var.set("Installed! ML features are now available.")
                self._status_lbl.configure(text_color=C.SUCCESS)
                self._install_btn.configure(
                    text="Installed", state="disabled", fg_color=C.SUCCESS)
            else:
                self._install_btn.configure(
                    text="Retry", state="normal", fg_color=C.ACCENT)
                self._done = None
        else:
            self.after(500, self._poll)
