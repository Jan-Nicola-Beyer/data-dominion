"""Table frame — filterable, sortable row browser."""

from __future__ import annotations
from typing import TYPE_CHECKING
import re

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip, _style_treeview, DateRangeSlider

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App


class TableFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._sort_col = None
        self._sort_asc = True
        self._search_var  = tk.StringVar()
        self._plat_var    = tk.StringVar(value="All")
        self._lang_var    = tk.StringVar(value="All")
        self._df_var      = tk.StringVar()
        self._dt_var      = tk.StringVar()
        self._date_col_var = tk.StringVar()
        self._build()

    def _build(self):
        # ── Filter bar ────────────────────────────────────────────────────────
        fb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=54)
        fb.pack(fill="x")
        fb.pack_propagate(False)

        self._search_var.trace_add("write", lambda *_: self._on_filter())
        search_entry = ctk.CTkEntry(fb, placeholder_text="Search text…", width=220,
                                    textvariable=self._search_var)
        search_entry.pack(side="left", padx=8, pady=8)
        tip(search_entry,
            "Type any word or phrase to filter rows (case-insensitive).")

        self._plat_om = ctk.CTkOptionMenu(fb, variable=self._plat_var,
                                          values=["All"], width=140,
                                          command=lambda _: self._on_filter())
        self._plat_om.pack(side="left", padx=4)
        tip(self._plat_om, "Filter by platform or data source.")

        self._lang_om = ctk.CTkOptionMenu(fb, variable=self._lang_var,
                                          values=["All"], width=110,
                                          command=lambda _: self._on_filter())
        self._lang_om.pack(side="left", padx=4)
        tip(self._lang_om, "Filter by language code (e.g. en, de, fr).")

        self._date_col_om = ctk.CTkOptionMenu(
            fb, variable=self._date_col_var, values=["—"],
            width=120, height=24, command=self._on_date_col_change)
        self._date_col_om.pack(side="left", padx=(8, 2))
        tip(self._date_col_om, "Choose which date column to filter on\n"
                               "(e.g. created_at, posted_at, collected_at).")

        self._df_var.trace_add("write", lambda *_: self._on_filter())
        self._dt_var.trace_add("write", lambda *_: self._on_filter())
        self._date_slider = DateRangeSlider(fb, from_var=self._df_var,
                                            to_var=self._dt_var, width=240)
        self._date_slider.pack(side="left", padx=2)

        clear_btn = ctk.CTkButton(fb, text="Clear", width=60, fg_color=C.BTN,
                                   command=self._clear_filters)
        clear_btn.pack(side="left", padx=6)
        tip(clear_btn, "Remove all active filters and show every row.")

        self._filter_badge = ctk.CTkLabel(fb, text="", text_color=C.WARN,
                                           font=("Segoe UI", 11))
        self._filter_badge.pack(side="left", padx=6)

        # ── Table ─────────────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=C.BG)
        body.pack(fill="both", expand=True)

        table_frame = ctk.CTkFrame(body, fg_color=C.CARD, corner_radius=0)
        table_frame.pack(side="left", fill="both", expand=True)

        self._tv = ttk.Treeview(table_frame, show="headings", selectmode="browse")
        _style_treeview(self._tv)
        ysb = ttk.Scrollbar(table_frame, orient="vertical", command=self._tv.yview)
        xsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self._tv.xview)
        self._tv.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self._tv.pack(fill="both", expand=True)
        self._tv.bind("<<TreeviewSelect>>", self._on_row_select)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="No data loaded.")
        ctk.CTkLabel(self, textvariable=self._status_var, text_color=C.MUTED,
                     font=("Segoe UI", 11), anchor="w"
                     ).pack(anchor="w", padx=12, pady=4)

    # ── Filter logic ───────────────────────────────────────────────────────────

    def _on_filter(self):
        fs = self.app.filter_state
        fs["search"]    = self._search_var.get()
        fs["platform"]  = self._plat_var.get()
        fs["language"]  = self._lang_var.get()
        fs["date_from"] = self._df_var.get()
        fs["date_to"]   = self._dt_var.get()
        self.app.apply_filters()
        self._populate()

    def _clear_filters(self):
        self._search_var.set("")
        self._plat_var.set("All")
        self._lang_var.set("All")
        self._df_var.set("")
        self._dt_var.set("")
        self._on_filter()

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _on_date_col_change(self, col):
        """User picked a different date column — re-parse and refresh slider."""
        if not col or col == "—":
            return
        self.app.switch_date_column(col)
        # Reset slider to full new range
        if "_date" in self.app.df.columns:
            dates = self.app.df["_date"].dropna()
            if not dates.empty:
                self._date_slider.set_date_range(dates.min(), dates.max())
        # Clear existing date filter so the slider starts fresh
        self._df_var.set("")
        self._dt_var.set("")
        self._on_filter()

    def refresh(self):
        df = self.app.df
        if df.empty:
            return

        plat_col = (self.app.resolve_col("platform", df) or
                    ("_source_dataset" if "_source_dataset" in df.columns else None))
        if plat_col:
            vals = ["All"] + sorted(df[plat_col].dropna().astype(str).unique().tolist())
            self._plat_om.configure(values=vals)

        lang_col = self.app.resolve_col("language", df)
        if lang_col:
            vals = ["All"] + sorted(df[lang_col].dropna().astype(str).unique().tolist())
            self._lang_om.configure(values=vals)

        # Date column dropdown
        candidates = self.app.date_column_candidates()
        if candidates:
            self._date_col_om.configure(values=candidates)
            cur = self.app.date_source_col
            if cur and cur in candidates:
                self._date_col_var.set(cur)
            else:
                self._date_col_var.set(candidates[0])
        else:
            self._date_col_om.configure(values=["—"])
            self._date_col_var.set("—")

        if "_date" in df.columns:
            dates = df["_date"].dropna()
            if not dates.empty:
                self._date_slider.set_date_range(dates.min(), dates.max())

        self._populate()

    def _populate(self):
        df = self.app.filtered_df
        self._tv.delete(*self._tv.get_children())

        display_cols = [c for c in df.columns if not c.startswith("_")][:20]
        self._tv.configure(columns=display_cols)
        for col in display_cols:
            self._tv.heading(col, text=col, command=lambda c=col: self._sort(c))
            self._tv.column(col, width=140, minwidth=60)

        cap = min(self.app.row_limit, len(df))
        for row in df.head(cap)[display_cols].values:
            vals = [str(v)[:120] if pd.notna(v) else "" for v in row]
            self._tv.insert("", "end", values=vals)

        total    = len(self.app.df)
        filt     = len(df)
        date_src = self.app.date_source_col
        date_hint = (f"  |  Date filter on: {date_src}" if date_src
                     else "  |  No date column detected")
        self._status_var.set(
            f"Showing {min(cap, filt)} of {filt} filtered rows  (total: {total}){date_hint}")

        active = sum(1 for k, v in self.app.filter_state.items()
                     if v and v not in ("All", ""))
        self._filter_badge.configure(
            text=f"  {active} filter(s) active" if active else "")

    def _sort(self, col: str):
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        df = self.app.filtered_df
        if col in df.columns:
            self.app.filtered_df = df.sort_values(
                col, ascending=self._sort_asc, na_position="last")
        self._populate()

    def _on_row_select(self, _event=None):
        pass

    def rebuild(self):
        # Save filter values before destroying widgets
        search   = self._search_var.get()
        plat     = self._plat_var.get()
        lang     = self._lang_var.get()
        df_val   = self._df_var.get()
        dt_val   = self._dt_var.get()
        date_col = self._date_col_var.get()

        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)

        # Create fresh StringVars so stale trace callbacks from destroyed
        # widgets don't fire on the new values.
        self._search_var  = tk.StringVar()
        self._plat_var    = tk.StringVar(value="All")
        self._lang_var    = tk.StringVar(value="All")
        self._df_var      = tk.StringVar()
        self._dt_var      = tk.StringVar()
        self._date_col_var = tk.StringVar()

        self._build()

        # Restore — traces now point only to the new widgets
        self._search_var.set(search)
        self._plat_var.set(plat)
        self._lang_var.set(lang)
        self._df_var.set(df_val)
        self._dt_var.set(dt_val)
        self._date_col_var.set(date_col)
        self.refresh()
