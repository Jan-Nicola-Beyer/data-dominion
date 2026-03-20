"""Export frame — save the filtered dataset in various formats."""

from __future__ import annotations
from typing import TYPE_CHECKING
import io
import zipfile

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App


class ExportFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._build()

    def _build(self):
        tb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=48)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="Export", font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(side="left", padx=16)

        body = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=8)
        body.pack(fill="both", expand=True, padx=14, pady=12)

        options = [
            ("Export current view (CSV)", self._export_csv,
             "Save the filtered rows to a CSV file. Internal columns are hidden."),
            ("Export with tags (CSV)", self._export_tags,
             "Save the current view to CSV including the '_tags' column."),
            ("Export to Excel (.xlsx)", self._export_excel,
             "Save as a Microsoft Excel workbook. Requires openpyxl."),
            ("Export all datasets (ZIP)", self._export_all,
             "Bundle every loaded dataset as its own CSV inside a ZIP archive."),
        ]
        for label, cmd, tooltip_text in options:
            row = ctk.CTkFrame(body, fg_color=C.CARD, corner_radius=6)
            row.pack(fill="x", padx=10, pady=5)
            lbl = ctk.CTkLabel(row, text=label, text_color=C.TEXT,
                               font=("Segoe UI", 12))
            lbl.pack(side="left", padx=14, pady=12)
            tip(lbl, tooltip_text)
            btn = ctk.CTkButton(row, text="Export", fg_color=C.ACCENT, width=100,
                                command=cmd)
            btn.pack(side="right", padx=12, pady=8)
            tip(btn, tooltip_text)

    # ── Export actions ─────────────────────────────────────────────────────────

    def _export_csv(self):
        df = self.app.filtered_df
        if df.empty:
            messagebox.showinfo("Export", "No data to export.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            df[[c for c in df.columns if not c.startswith("_")]].to_csv(p, index=False)
            messagebox.showinfo("Done", f"Saved {len(df)} rows to {p}")

    def _export_tags(self):
        df = self.app.filtered_df
        if df.empty:
            messagebox.showinfo("Export", "No data to export.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            cols = [c for c in df.columns if not c.startswith("_")] + ["_tags"]
            cols = [c for c in cols if c in df.columns]
            df[cols].to_csv(p, index=False)
            messagebox.showinfo("Done", f"Saved {len(df)} rows to {p}")

    def _export_excel(self):
        df = self.app.filtered_df
        if df.empty:
            messagebox.showinfo("Export", "No data.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                         filetypes=[("Excel", "*.xlsx")])
        if p:
            try:
                out = df[[c for c in df.columns if not c.startswith("_")]].copy()
                # Excel doesn't support tz-aware datetimes — strip timezone info
                for col in out.select_dtypes(include=["datetimetz"]).columns:
                    out[col] = out[col].dt.tz_localize(None)
                out.to_excel(p, index=False)
                messagebox.showinfo("Done", f"Saved to {p}")
            except ImportError:
                messagebox.showerror("Error", "pip install openpyxl")

    def _export_all(self):
        entries = self.app.dm.entries
        if not entries:
            messagebox.showinfo("Export", "No datasets loaded.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".zip",
                                         filetypes=[("ZIP", "*.zip")])
        if p:
            with zipfile.ZipFile(p, "w") as zf:
                for e in entries:
                    buf = io.StringIO()
                    e.df.to_csv(buf, index=False)
                    zf.writestr(f"{e.name}.csv", buf.getvalue())
            messagebox.showinfo("Done", f"Exported {len(entries)} datasets to {p}")

    def rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._build()
