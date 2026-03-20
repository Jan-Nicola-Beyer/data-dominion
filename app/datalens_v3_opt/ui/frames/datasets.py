"""Datasets frame — list of loaded datasets, coverage table, and merge wizard."""

from __future__ import annotations
from typing import TYPE_CHECKING

import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.data.core_columns import CORE_COLUMNS, CATEGORY_COLORS
from datalens_v3_opt.data.manager import DatasetEntry
from datalens_v3_opt.ui.widgets import tip
from datalens_v3_opt.ui.wizards import MergeWizard

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App


class DatasetsFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._build()

    def _build(self):
        tb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=52)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="Datasets", font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(side="left", padx=16, pady=14)

        btn_add = ctk.CTkButton(tb, text="+ Add Dataset", fg_color=C.ACCENT, width=140,
                                command=self.app.open_import_wizard)
        btn_add.pack(side="left", padx=8)
        tip(btn_add,
            "Import another CSV, TSV or Excel file.\n\n"
            "Once you have at least two, use 'Merge All' to combine them.")

        btn_merge = ctk.CTkButton(tb, text="Merge All…", fg_color=C.SUCCESS, width=120,
                                  command=self._open_merge)
        btn_merge.pack(side="left", padx=4)
        tip(btn_merge,
            "Combine ALL loaded datasets into one table.\n\n"
            "Fields absent in some datasets are filled with N/A automatically.")

        btn_clear = ctk.CTkButton(tb, text="Clear All", fg_color=C.BTN, width=100,
                                  command=self._clear_all)
        btn_clear.pack(side="left", padx=4)
        tip(btn_clear, "Remove ALL loaded datasets and reset the workspace.")

        pane = ctk.CTkFrame(self, fg_color=C.BG)
        pane.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(pane, fg_color=C.PANEL, corner_radius=8, width=320)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        ctk.CTkLabel(left, text="Loaded Datasets", font=("Segoe UI", 12, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=12, pady=(12, 4))
        self._list_frame = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self._list_frame.pack(fill="both", expand=True, padx=4, pady=4)

        right = ctk.CTkFrame(pane, fg_color=C.BG, corner_radius=0)
        right.pack(side="left", fill="both", expand=True)

        cov_title = ctk.CTkLabel(right, text="Column Coverage Across Datasets",
                                 font=("Segoe UI", 13, "bold"), text_color=C.TEXT)
        cov_title.pack(anchor="w", padx=4, pady=(4, 4))
        tip(cov_title,
            "Which standard fields were detected in each loaded dataset.\n\n"
            "Green = found · Grey dash = absent (will be N/A when merged).")
        ctk.CTkLabel(right,
                     text="Green = field present  |  Grey dash = field missing (will be N/A)",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=4, pady=(0, 6))

        self._coverage_frame = ctk.CTkScrollableFrame(right, fg_color=C.BG)
        self._coverage_frame.pack(fill="both", expand=True)

        self._active_label = ctk.CTkLabel(self, text="", text_color=C.MUTED,
                                          font=("Segoe UI", 11))
        self._active_label.pack(anchor="w", padx=14, pady=4)

    # ── Refresh ────────────────────────────────────────────────────────────────

    def refresh(self):
        for w in self._list_frame.winfo_children():
            w.destroy()

        if not self.app.dm.entries:
            ctk.CTkLabel(self._list_frame,
                         text="No datasets loaded.\nUse '+ Add Dataset'.",
                         text_color=C.MUTED, font=("Segoe UI", 11)
                         ).pack(pady=20)
        else:
            for entry in self.app.dm.entries:
                self._add_entry_card(entry)

        self._refresh_coverage()
        rows = len(self.app.df)
        cols = len(self.app.df.columns)
        self._active_label.configure(
            text=f"Active workspace: {rows} rows × {cols} columns")

    def _add_entry_card(self, entry: DatasetEntry):
        card = ctk.CTkFrame(self._list_frame, fg_color=C.CARD, corner_radius=6)
        card.pack(fill="x", pady=3, padx=2)
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(top, text=entry.name, font=("Segoe UI", 12, "bold"),
                     text_color=C.TEXT, anchor="w").pack(side="left")

        btn_x = ctk.CTkButton(top, text="X", width=26, height=22, fg_color=C.DANGER,
                               font=("Segoe UI", 10),
                               command=lambda uid=entry.uid: self._remove(uid))
        btn_x.pack(side="right")
        tip(btn_x, "Remove this dataset. The original file is not affected.")

        btn_load = ctk.CTkButton(top, text="Load", width=50, height=22, fg_color=C.ACCENT,
                                  font=("Segoe UI", 10),
                                  command=lambda e=entry: self._load_single(e))
        btn_load.pack(side="right", padx=4)
        tip(btn_load, "Set as active dataset and open in the Table view.")

        ctk.CTkLabel(card,
                     text=f"{entry.row_count} rows × {entry.col_count} cols  |  "
                          f"{len(entry.col_map)} fields mapped",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=8, pady=(0, 6))

    def _refresh_coverage(self):
        for w in self._coverage_frame.winfo_children():
            w.destroy()
        if not self.app.dm.entries:
            return

        suggestion = self.app.dm.suggest_merge()
        coverage   = suggestion.get("field_coverage", {})
        ds_names   = [e.name for e in self.app.dm.entries]

        hdr = ctk.CTkFrame(self._coverage_frame, fg_color=C.PANEL,
                            corner_radius=4, height=30)
        hdr.pack(fill="x", pady=(0, 4))
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Field", text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold"), width=160
                     ).pack(side="left", padx=8)
        ctk.CTkLabel(hdr, text="Category", text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold"), width=100
                     ).pack(side="left")
        for name in ds_names:
            ctk.CTkLabel(hdr, text=name[:20], text_color=C.MUTED,
                         font=("Segoe UI", 9, "bold"), width=160
                         ).pack(side="left", padx=2)

        for field, info in CORE_COLUMNS.items():
            if field not in coverage:
                continue
            cat   = info["category"]
            color = CATEGORY_COLORS.get(cat, C.MUTED)
            row = ctk.CTkFrame(self._coverage_frame, fg_color=C.CARD,
                                corner_radius=4, height=26)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)
            ctk.CTkLabel(row, text=field, text_color=C.TEXT,
                         font=("Segoe UI", 11), width=155, anchor="w"
                         ).pack(side="left", padx=8)
            ctk.CTkLabel(row, text=cat, text_color=color,
                         font=("Segoe UI", 10), width=95, anchor="w"
                         ).pack(side="left")
            for name in ds_names:
                orig = coverage.get(field, {}).get(name, "")
                ctk.CTkLabel(row, text=orig[:20] if orig else "—",
                             text_color=C.SUCCESS if orig else C.DIM,
                             font=("Segoe UI", 10), width=155, anchor="w"
                             ).pack(side="left", padx=2)

    # ── Actions ────────────────────────────────────────────────────────────────

    def _load_single(self, entry: DatasetEntry):
        self.app.load_entry(entry)
        self.app._show("table")

    def _remove(self, uid: str):
        entry = next((e for e in self.app.dm.entries if e.uid == uid), None)
        name = entry.name if entry else "this dataset"
        if not messagebox.askyesno("Remove Dataset",
                                   f"Remove '{name}'?\n\n"
                                   "The original file is not affected.",
                                   parent=self):
            return
        self.app.dm.remove(uid)
        self.refresh()

    def _clear_all(self):
        if messagebox.askyesno("Clear all", "Remove all loaded datasets?"):
            self.app.dm.clear()
            self.app.df = pd.DataFrame()
            self.app.filtered_df = pd.DataFrame()
            self.refresh()

    def _open_merge(self):
        if len(self.app.dm.entries) < 2:
            messagebox.showinfo("Merge", "Load at least 2 datasets to merge.",
                                parent=self)
            return
        MergeWizard(self, self.app.dm, self._on_merge)

    def _on_merge(self, merged_df: pd.DataFrame):
        self.app.load_merged(merged_df)
        self.refresh()
        self.app._show("table")

    def rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._build()
        self.refresh()
