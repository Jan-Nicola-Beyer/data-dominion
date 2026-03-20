"""Settings frame — display preferences, persistence, and predictor info."""

from __future__ import annotations
from typing import TYPE_CHECKING

import tkinter as tk

import customtkinter as ctk

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip
from datalens_v3_opt.ui.ml_installer import MLInstallButton, ml_installed
from datalens_v3_opt.data.persistence import save_settings, LOG_FILE


def _check_sbert() -> bool:
    """Check if sentence-transformers is importable without loading it."""
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App


class SettingsFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Settings", font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=20, pady=20)

        body = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=8)
        body.pack(fill="x", padx=14, pady=8)

        # ── Row limit ─────────────────────────────────────────────────────────
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=10)
        lim_lbl = ctk.CTkLabel(row, text="Display row limit:", text_color=C.MUTED,
                               font=("Segoe UI", 11), width=200)
        lim_lbl.pack(side="left")
        tip(lim_lbl,
            "Maximum rows shown in the Table view at once.\n\n"
            "This does NOT affect filtering or exports — those always\n"
            "use all matching rows. Only the Table display is capped.")
        self._limit_var = tk.IntVar(value=self.app.row_limit)
        for val in [500, 1000, 2000, 5000, 10000]:
            rb = ctk.CTkRadioButton(row, text=str(val), variable=self._limit_var,
                                    value=val, command=self._apply_limit)
            rb.pack(side="left", padx=6)
            tip(rb, f"Show up to {val:,} rows at a time.")

        # ── Persistence info ──────────────────────────────────────────────────
        persist = ctk.CTkFrame(body, fg_color=C.CARD, corner_radius=6)
        persist.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(persist, text="Settings are saved automatically",
                     text_color=C.TEXT, font=("Segoe UI", 11)
                     ).pack(anchor="w", padx=12, pady=(8, 2))
        ctk.CTkLabel(persist,
                     text="Theme, row limit, and other preferences persist between sessions.\n"
                          "Tags and slices are autosaved and restored on next launch.",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=12, pady=(0, 8))

        # ── Reset ─────────────────────────────────────────────────────────────
        reset_row = ctk.CTkFrame(body, fg_color="transparent")
        reset_row.pack(fill="x", padx=12, pady=(0, 8))
        rb = ctk.CTkButton(reset_row, text="Reset to Defaults", fg_color=C.BTN,
                           width=160, command=self._reset_defaults)
        rb.pack(side="left")
        tip(rb, "Reset row limit and theme to default values.")

        # ── Predictor info ────────────────────────────────────────────────────
        info = ctk.CTkFrame(body, fg_color=C.CARD, corner_radius=6)
        info.pack(fill="x", padx=12, pady=8)
        has_sbert = _check_sbert()
        mode = ("sentence-transformers (semantic)" if has_sbert
                else "fuzzy string matching (heuristic)")
        pred_lbl = ctk.CTkLabel(info, text=f"Column predictor: {mode}",
                                text_color=C.TEXT, font=("Segoe UI", 11))
        pred_lbl.pack(anchor="w", padx=12, pady=8)
        tip(pred_lbl,
            "Data Dominion automatically matches columns to known fields.\n\n"
            "Semantic mode (sentence-transformers) is more accurate.\n"
            "Fuzzy mode uses string similarity.")
        if not has_sbert:
            ctk.CTkLabel(info,
                         text="Install sentence-transformers for better predictions:\n"
                              "pip install sentence-transformers",
                         text_color=C.MUTED, font=("Segoe UI", 10)
                         ).pack(anchor="w", padx=12, pady=(0, 8))

        # ── ML Pack ───────────────────────────────────────────────────────────
        MLInstallButton(body).pack(fill="x", padx=12, pady=8)

        # ── Log info ──────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(body, fg_color=C.CARD, corner_radius=6)
        log_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(log_frame, text=f"Log file:  {LOG_FILE}",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=12, pady=8)

        # ── About ─────────────────────────────────────────────────────────────
        about = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=8)
        about.pack(fill="x", padx=14, pady=8)
        ctk.CTkLabel(about, text="Data Dominion",
                     font=("Segoe UI", 13, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(about,
                     text="Multi-dataset social media analysis studio.\n"
                          "Supports CSV, TSV, Excel. Predicts core columns automatically.",
                     text_color=C.MUTED, font=("Segoe UI", 11)
                     ).pack(anchor="w", padx=14, pady=(0, 12))

    def _apply_limit(self):
        self.app.row_limit = self._limit_var.get()
        self._persist()

    def _reset_defaults(self):
        self._limit_var.set(2000)
        self.app.row_limit = 2000
        self._persist()

    def _persist(self):
        """Save current settings to disk."""
        save_settings({
            "theme":     C.CURRENT_THEME,
            "row_limit": self.app.row_limit,
        })

    def rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._build()
