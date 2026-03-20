"""Main application window — wires together all frames and manages shared state."""

from __future__ import annotations
import logging
import re
import warnings
from tkinter import messagebox

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.data.core_columns import CORE_COLUMNS
from datalens_v3_opt.data.manager import DatasetEntry, DatasetManager
from datalens_v3_opt.data.persistence import (
    load_settings, save_settings, save_project_state,
    load_project_state, has_autosave, clear_autosave, get_logger,
)
from datalens_v3_opt.io.file_utils import safe_date_parse
from datalens_v3_opt.ui.sidebar import Sidebar
from datalens_v3_opt.ui.wizards import ImportWizard

# Frame classes are imported lazily in _ensure_frame() to speed up startup.
# Only HomeFrame is imported eagerly since it's shown immediately.
from datalens_v3_opt.ui.frames.home import HomeFrame

_log = get_logger()

# Map nav keys to (module_path, class_name) for lazy imports
_FRAME_REGISTRY: dict[str, tuple[str, str]] = {
    "home":      ("datalens_v3_opt.ui.frames.home",      "HomeFrame"),
    "datasets":  ("datalens_v3_opt.ui.frames.datasets",  "DatasetsFrame"),
    "table":     ("datalens_v3_opt.ui.frames.table",     "TableFrame"),
    "analytics": ("datalens_v3_opt.ui.frames.analytics", "AnalyticsFrame"),
    "coding":    ("datalens_v3_opt.ui.frames.coding",    "CodingFrame"),
    "topics":    ("datalens_v3_opt.ui.frames.topics",    "TopicModellingFrame"),
    "slicer":    ("datalens_v3_opt.ui.frames.slicer",    "SlicerFrame"),
    "export":    ("datalens_v3_opt.ui.frames.export",    "ExportFrame"),
    "settings":  ("datalens_v3_opt.ui.frames.settings",  "SettingsFrame"),
}


class App(ctk.CTk):
    """Root window.  Owns shared state and coordinates navigation."""

    def __init__(self):
        super().__init__()
        self.title("Data Dominion")
        self.geometry("1400x860")
        self.minsize(1100, 660)

        # ── Taskbar / window icon ───────────────────────────────────────────
        try:
            import os
            _icon = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), "..", "..", "app_icon.ico")
            if not os.path.exists(_icon):
                _icon = os.path.join(C._asset_root(), "..", "app_icon.ico")
            if os.path.exists(_icon):
                self.iconbitmap(_icon)
        except Exception:
            pass

        # ── Load persisted settings ───────────────────────────────────────────
        settings = load_settings()
        if settings.get("theme", "dark") != C.CURRENT_THEME:
            C.apply_theme(settings["theme"])
        self.configure(fg_color=C.BG)

        # ── Shared state ──────────────────────────────────────────────────────
        self.dm           = DatasetManager()
        self.df           = pd.DataFrame()
        self.filtered_df  = pd.DataFrame()
        self.col_map: dict = {}
        self.tags: dict    = {}
        self.slices: dict  = {}
        self.filter_state  = dict(search="", platform="All", language="All",
                                  date_from="", date_to="", tag="")
        self.row_limit        = settings.get("row_limit", 2000)
        self.active_nav       = "home"
        self.date_source_col  = None
        self._dirty           = False   # True if there is unsaved work
        self._theme_dirty: set = set()  # frames needing rebuild after theme switch
        self._date_candidates_cache = None  # cached date_column_candidates result

        # ── Show splash ──────────────────────────────────────────────────────
        self._splash = self._show_splash()
        self.update_idletasks()

        # ── Restore autosave ──────────────────────────────────────────────────
        self._restore_autosave()

        self._build_layout()
        self._show("home")

        # ── Close splash ──────────────────────────────────────────────────────
        if self._splash:
            self._splash.destroy()
            self._splash = None

        # ── Close handler ─────────────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Autosave timer (every 60 seconds) ─────────────────────────────────
        self._schedule_autosave()

        _log.info("App started — theme=%s, row_limit=%d", C.CURRENT_THEME, self.row_limit)

    # ── Splash screen ────────────────────────────────────────────────────────

    def _show_splash(self):
        """Show a lightweight splash screen while the app initialises."""
        try:
            splash = ctk.CTkToplevel(self)
            splash.overrideredirect(True)
            splash.configure(fg_color=C.PANEL)

            # Centre on screen
            sw, sh = splash.winfo_screenwidth(), splash.winfo_screenheight()
            w, h = 380, 160
            splash.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
            splash.wm_attributes("-topmost", True)

            ctk.CTkLabel(splash, text="Data Dominion",
                         font=("Segoe UI", 24, "bold"),
                         text_color=C.TEXT).pack(pady=(30, 4))
            ctk.CTkLabel(splash, text="Loading…",
                         font=("Segoe UI", 12),
                         text_color=C.MUTED).pack(pady=(0, 8))
            prog = ctk.CTkProgressBar(splash, width=280, height=8,
                                      mode="indeterminate")
            prog.pack(pady=(4, 0))
            prog.start()

            splash.update()
            return splash
        except Exception:
            return None

    # ── Autosave ──────────────────────────────────────────────────────────────

    def _schedule_autosave(self):
        self._autosave()
        self.after(60_000, self._schedule_autosave)

    def _autosave(self):
        """Save lightweight session state (tags, slices, settings) if changed."""
        if not self._dirty:
            return
        try:
            state = {
                "tags":   self.tags,
                "slices": self.slices,
                "settings": {
                    "theme":     C.CURRENT_THEME,
                    "row_limit": self.row_limit,
                },
            }
            save_project_state(state)
        except Exception as exc:
            _log.warning("Autosave error: %s", exc)

    def _restore_autosave(self):
        """Offer to restore tags/slices from the last autosave."""
        if not has_autosave():
            return
        state = load_project_state()
        if not state:
            return
        # Only restore if there's meaningful state
        has_tags   = bool(state.get("tags"))
        has_slices = bool(state.get("slices"))
        if not has_tags and not has_slices:
            return
        parts = []
        if has_tags:
            parts.append(f"{len(state['tags'])} tag(s)")
        if has_slices:
            parts.append(f"{len(state['slices'])} slice(s)")
        desc = " and ".join(parts)

        if messagebox.askyesno(
            "Restore Session",
            f"A previous session was found with {desc}.\n\n"
            "Would you like to restore this work?",
        ):
            self.tags   = state.get("tags", {})
            self.slices = state.get("slices", {})
            # Migrate old tag format: {name: color_str} → {name: {color, desc, group, exclusive}}
            for name, val in list(self.tags.items()):
                if isinstance(val, str):
                    self.tags[name] = {"color": val, "desc": "", "group": "", "exclusive": False}
            _log.info("Restored autosave: %s", desc)
        else:
            clear_autosave()

    # ── Close handler ─────────────────────────────────────────────────────────

    def _on_close(self):
        """Prompt before closing if there is work in progress."""
        has_work = (
            bool(self.tags)
            or bool(self.slices)
            or not self.df.empty
            or bool(self.dm.entries)
        )
        if has_work:
            if not messagebox.askyesno(
                "Quit Data Dominion",
                "You have datasets, tags, or slices in your session.\n\n"
                "Your tags and slices will be saved automatically\n"
                "and offered for restore next time.\n\n"
                "Quit now?",
            ):
                return

        # Force final autosave before exit
        self._dirty = True
        self._autosave()

        # Close any open Toplevel windows (dashboards, progress, etc.)
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkToplevel):
                try:
                    widget.destroy()
                except Exception:
                    pass

        # Close matplotlib figures if loaded
        import sys
        if "matplotlib.pyplot" in sys.modules:
            import matplotlib.pyplot as _plt
            _plt.close("all")

        _log.info("App closed")
        logging.shutdown()
        self.destroy()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self._sidebar = Sidebar(self, self._navigate, self)
        self._sidebar.pack(side="left", fill="y")

        self._main = ctk.CTkFrame(self, fg_color=C.BG, corner_radius=0)
        self._main.pack(side="left", fill="both", expand=True)

        # Frames are created lazily on first navigation.
        # Only HomeFrame is built immediately since it's the landing page.
        self._frames: dict = {}
        self._ensure_frame("home")

    # ── Lazy frame creation ──────────────────────────────────────────────────

    def _ensure_frame(self, key: str):
        """Create the frame for *key* if it doesn't exist yet."""
        if key in self._frames:
            return
        mod_path, cls_name = _FRAME_REGISTRY[key]
        import importlib
        module = importlib.import_module(mod_path)
        cls = getattr(module, cls_name)
        f = cls(self._main, self)
        f.place(x=0, y=0, relwidth=1, relheight=1)
        self._frames[key] = f

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate(self, key: str):
        if key == "import":
            self.open_import_wizard()
        else:
            self._show(key)

    def _show(self, key: str):
        self.active_nav = key
        # Create frame lazily on first visit
        self._ensure_frame(key)
        # Deferred theme rebuild: if this frame is dirty, rebuild it now
        if key in self._theme_dirty:
            self._frames[key].rebuild()
            self._theme_dirty.discard(key)
        self._frames[key].tkraise()
        self._sidebar.set_active(key)
        if key == "datasets":
            self._frames["datasets"].refresh()
        elif key == "table":
            self.apply_filters()
            self._frames["table"].refresh()
        elif key == "analytics":
            self.apply_filters()
        elif key == "coding":
            self._frames["coding"].refresh()
        elif key == "topics":
            self._frames["topics"].refresh()
        elif key == "slicer":
            self._frames["slicer"]._refresh_slices()

    # ── Theme switching ───────────────────────────────────────────────────────

    def toggle_theme(self):
        new_theme = "light" if C.CURRENT_THEME == "dark" else "dark"
        C.apply_theme(new_theme)

        self.configure(fg_color=C.BG)
        self._main.configure(fg_color=C.BG)

        # Only rebuild the sidebar and the active frame immediately;
        # mark all others dirty so they rebuild on next navigation.
        self._sidebar.rebuild()
        active = self.active_nav
        if active in self._frames:
            self._frames[active].rebuild()
        self._theme_dirty = {k for k in self._frames if k != active}

        if active in self._frames:
            self._frames[active].tkraise()
            if active == "table":
                self._frames["table"].refresh()
            elif active == "analytics":
                self._frames["analytics"].refresh()
            elif active == "home":
                self._frames["home"].refresh()

        # Persist theme choice
        save_settings({"theme": new_theme, "row_limit": self.row_limit})
        _log.info("Theme switched to %s", new_theme)

    # ── Data loading ──────────────────────────────────────────────────────────

    def open_import_wizard(self):
        ImportWizard(self, self._on_import_finish, self.dm)

    def _on_import_finish(self, entry: DatasetEntry):
        self.load_entry(entry)
        self._show("datasets")
        if "home" in self._frames:
            self._frames["home"].refresh()
        _log.info("Imported dataset '%s' — %d rows, %d cols",
                  entry.name, entry.row_count, entry.col_count)

    def load_entry(self, entry: DatasetEntry):
        self.df      = entry.df.copy()
        self.col_map = dict(entry.col_map)
        self._date_candidates_cache = None  # invalidate cache
        self._ensure_tags()
        self._build_date_col()
        self.apply_filters()
        self._dirty = True

    def load_merged(self, merged_df: pd.DataFrame):
        self.df      = merged_df.copy()
        self.col_map = {f: f for f in merged_df.columns if f in CORE_COLUMNS}
        self._date_candidates_cache = None  # invalidate cache
        self._ensure_tags()
        self._build_date_col()
        self.apply_filters()
        self._dirty = True
        _log.info("Merged %d datasets — %d rows", len(self.dm.entries), len(merged_df))

    def _ensure_tags(self):
        if "_tags" not in self.df.columns:
            self.df["_tags"] = ""
        if "_tagged_at" not in self.df.columns:
            self.df["_tagged_at"] = ""

    def _build_date_col(self):
        if "_date" in self.df.columns and self.df["_date"].notna().any():
            if not self.date_source_col:
                for canonical in ("created_at", "collected_at"):
                    col = self.resolve_col(canonical, self.df)
                    if col:
                        self.date_source_col = col
                        break
            return

        for canonical in ("created_at", "collected_at"):
            col = self.resolve_col(canonical, self.df)
            if col:
                parsed = safe_date_parse(self.df[col])
                if parsed.notna().any():
                    self.df["_date"] = parsed
                    self.date_source_col = col
                    return

        best_col, best_rate = None, 0.0
        for col in self.df.columns:
            if col.startswith("_"):
                continue
            sample = self.df[col].dropna()
            if sample.empty or len(sample) < 3:
                continue
            if pd.api.types.is_numeric_dtype(sample):
                continue
            try:
                parsed = pd.to_datetime(sample.head(30), utc=True, errors="coerce")
                rate = parsed.notna().mean()
                if rate > best_rate:
                    best_rate, best_col = rate, col
            except Exception:
                continue

        if best_col and best_rate >= 0.7:
            self.df["_date"] = safe_date_parse(self.df[best_col])
            self.date_source_col = best_col
        else:
            self.df["_date"] = pd.NaT
            self.date_source_col = None

    def date_column_candidates(self) -> list[str]:
        """Return column names that look like dates (parseable to datetime)."""
        if self.df.empty:
            return []
        if self._date_candidates_cache is not None:
            return self._date_candidates_cache
        candidates = []
        for col in self.df.columns:
            if col.startswith("_"):
                continue
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                candidates.append(col)
                continue
            sample = self.df[col].dropna()
            if sample.empty or len(sample) < 3:
                continue
            if pd.api.types.is_numeric_dtype(sample):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(sample.head(20), utc=True, errors="coerce")
                if parsed.notna().mean() >= 0.5:
                    candidates.append(col)
            except Exception:
                continue
        self._date_candidates_cache = candidates
        return candidates

    def switch_date_column(self, col: str):
        """Re-parse _date from a different source column."""
        if col == self.date_source_col:
            return
        if col not in self.df.columns:
            return
        parsed = safe_date_parse(self.df[col])
        if parsed.notna().any():
            self.df["_date"] = parsed
            self.date_source_col = col
            self.apply_filters()

    # ── Filtering ─────────────────────────────────────────────────────────────

    def resolve_col(self, canonical: str, df: "pd.DataFrame") -> "str | None":
        if canonical in df.columns:
            return canonical
        mapped = self.col_map.get(canonical)
        return mapped if mapped and mapped in df.columns else None

    def apply_filters(self):
        if self.df.empty:
            self.filtered_df = self.df
            return
        fs = self.filter_state

        # Skip copy if no filters are active
        has_filter = (
            fs["search"] or fs["platform"] != "All" or fs["language"] != "All"
            or fs["date_from"] or fs["date_to"] or fs["tag"]
        )
        if not has_filter:
            self.filtered_df = self.df
            return

        df = self.df.copy()

        text_col = self.resolve_col("content_text", df)
        if fs["search"] and text_col:
            mask = df[text_col].astype(str).str.contains(
                re.escape(fs["search"]), case=False, na=False)
            df = df[mask]

        plat_col = (self.resolve_col("platform", df) or
                    ("_source_dataset" if "_source_dataset" in df.columns else None))
        if fs["platform"] != "All" and plat_col:
            df = df[df[plat_col].astype(str) == fs["platform"]]

        lang_col = self.resolve_col("language", df)
        if fs["language"] != "All" and lang_col:
            df = df[df[lang_col].astype(str) == fs["language"]]

        if "_date" in df.columns:
            if fs["date_from"]:
                try:
                    df = df[df["_date"] >= pd.Timestamp(fs["date_from"], tz="UTC")]
                except Exception:
                    pass
            if fs["date_to"]:
                try:
                    df = df[df["_date"] <= pd.Timestamp(fs["date_to"], tz="UTC")]
                except Exception:
                    pass

        if fs["tag"] and "_tags" in df.columns:
            df = df[df["_tags"].str.contains(fs["tag"], na=False)]

        self.filtered_df = df
