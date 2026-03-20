"""
Modal wizard dialogs:

  ImportWizard  — 4-step dialog for loading a single CSV/TSV/Excel file.
  MergeWizard   — dialog for concatenating all loaded datasets.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.data.core_columns import CORE_COLUMNS, CATEGORY_COLORS
from datalens_v3_opt.data.manager import DatasetEntry, DatasetManager
from datalens_v3_opt.data.predictor import get_predictor
from datalens_v3_opt.io.file_utils import (
    detect_encoding, read_raw_lines, detect_delimiter,
    detect_table_start, safe_date_parse,
)
from datalens_v3_opt.ui.widgets import tip, get_samples, FIELD_TIPS


# ── Import Wizard ──────────────────────────────────────────────────────────────

class ImportWizard(ctk.CTkToplevel):
    """
    Four-step import dialog:
      1. File preview + encoding / delimiter / header detection
      2. Column prediction + interactive selector
      3. Final column mapping review
      4. Quality report
    """

    def __init__(self, master, on_finish, dataset_manager: DatasetManager):
        super().__init__(master)
        self.title("Import Dataset — Data Dominion")
        self.geometry("1100x750")
        self.resizable(True, True)
        self.configure(fg_color=C.BG)
        self.grab_set()

        self.on_finish = on_finish
        self.dm = dataset_manager

        # wizard state
        self.path        = ""
        self.encoding    = "utf-8"
        self.delimiter   = ","
        self.header_row  = 0
        self.raw_lines   = []
        self.df_raw      = None
        self.predictions = {}
        self.col_vars    = {}   # {col: BooleanVar}
        self.map_vars    = {}   # {col: StringVar}

        self._build_ui()
        self._show_step(1)

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        self._step_label = ctk.CTkLabel(hdr, text="", font=("Segoe UI", 14, "bold"),
                                        text_color=C.TEXT)
        self._step_label.pack(side="left", padx=20, pady=16)

        dot_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        dot_frame.pack(side="right", padx=20)
        self._dots = []
        for _ in range(4):
            d = ctk.CTkLabel(dot_frame, text="●", font=("Segoe UI", 14),
                             text_color=C.MUTED)
            d.pack(side="left", padx=4)
            self._dots.append(d)

        self._content = ctk.CTkFrame(self, fg_color=C.BG, corner_radius=0)
        self._content.pack(fill="both", expand=True)

        ftr = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=54)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)
        self._btn_back = ctk.CTkButton(ftr, text="Back", width=100,
                                       fg_color=C.DIM, hover_color=C.BORDER,
                                       command=self._back)
        self._btn_back.pack(side="left", padx=16, pady=10)
        tip(self._btn_back, "Go back to the previous step.")
        self._btn_next = ctk.CTkButton(ftr, text="Next", width=130,
                                       fg_color=C.ACCENT, command=self._next)
        self._btn_next.pack(side="right", padx=16, pady=10)
        tip(self._btn_next,
            "Continue to the next step.\n"
            "On the last step this button becomes 'Finish' and adds the dataset.")
        self._status_var = tk.StringVar()
        ctk.CTkLabel(ftr, textvariable=self._status_var, text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=8)

        self._step = 1
        self._frames: dict = {}

    def _show_step(self, n: int):
        self._step = n
        for f in self._frames.values():
            f.pack_forget()

        labels = [
            "Step 1 of 4 — Preview & Detection",
            "Step 2 of 4 — Column Prediction",
            "Step 3 of 4 — Field Mapping",
            "Step 4 of 4 — Quality Report",
        ]
        self._step_label.configure(text=labels[n - 1])
        for i, d in enumerate(self._dots):
            d.configure(text_color=C.ACCENT if i < n else C.MUTED)

        builder = [None, self._build_step1, self._build_step2,
                   self._build_step3, self._build_step4][n]
        if n not in self._frames:
            f = ctk.CTkFrame(self._content, fg_color=C.BG, corner_radius=0)
            f.pack(fill="both", expand=True)
            self._frames[n] = f
            builder(f)
        else:
            self._frames[n].pack(fill="both", expand=True)

        self._btn_back.configure(state="normal" if n > 1 else "disabled")
        if n == 4:
            self._btn_next.configure(text="Finish", fg_color=C.SUCCESS)
        else:
            self._btn_next.configure(text="Next", fg_color=C.ACCENT)

    # ── Step 1 — File preview ──────────────────────────────────────────────────

    def _build_step1(self, frame):
        top = ctk.CTkFrame(frame, fg_color=C.PANEL, corner_radius=8)
        top.pack(fill="x", padx=16, pady=(12, 6))

        browse_btn = ctk.CTkButton(top, text="Browse File…", command=self._browse,
                                   width=140, fg_color=C.ACCENT)
        browse_btn.pack(side="left", padx=12, pady=10)
        tip(browse_btn, "Select a CSV, TSV or Excel file (.csv .tsv .xlsx .xls).")
        self._path_label = ctk.CTkLabel(top, text="No file selected",
                                        text_color=C.MUTED, font=("Segoe UI", 11))
        self._path_label.pack(side="left", padx=8)

        settings = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        settings.pack(fill="x", padx=16, pady=6)

        _enc_tips = {
            "Encoding":
                "Character encoding of the file.\n\n"
                "• UTF-8 — works for almost all modern files.\n"
                "• latin-1 / cp1252 — try these if you see garbled characters.\n"
                "• utf-16 — rare; used by some Windows tools.",
            "Delimiter":
                "The character used to separate columns.\n\n"
                "• Comma (,) — standard CSV.\n"
                "• Semicolon (;) — common in German/French Excel exports.\n"
                "• Tab — used in TSV files.\n"
                "• Pipe (|) — occasionally used in custom exports.",
        }
        for label, attr, vals in [
            ("Encoding",  "encoding",  ["utf-8", "latin-1", "cp1252", "utf-16"]),
            ("Delimiter", "delimiter", [",", ";", "\t", "|"]),
        ]:
            lbl = ctk.CTkLabel(settings, text=label, text_color=C.MUTED,
                               font=("Segoe UI", 11))
            lbl.pack(side="left", padx=(12, 2), pady=8)
            tip(lbl, _enc_tips[label])
            var = tk.StringVar(value=getattr(self, attr))
            om = ctk.CTkOptionMenu(settings, variable=var, values=vals, width=110,
                                   command=lambda v, a=attr, sv=var: self._on_setting(a, sv))
            om.pack(side="left", padx=(0, 12), pady=8)
            tip(om, _enc_tips[label])
            setattr(self, f"_{attr}_var", var)

        hrow_lbl = ctk.CTkLabel(settings, text="Header row", text_color=C.MUTED,
                                font=("Segoe UI", 11))
        hrow_lbl.pack(side="left", padx=(12, 2))
        tip(hrow_lbl,
            "Row number where the column names appear (0 = very first row).\n\n"
            "The app tries to detect this automatically — amber lines are the header.")
        self._hrow_spin = tk.Spinbox(settings, from_=0, to=30, width=4,
                                     bg=C.CARD, fg=C.TEXT, buttonbackground=C.BORDER,
                                     command=self._refresh_preview)
        self._hrow_spin.pack(side="left", padx=(0, 12))

        self._summary_var = tk.StringVar(value="")
        ctk.CTkLabel(settings, textvariable=self._summary_var,
                     text_color=C.MUTED, font=("Segoe UI", 10)).pack(side="left", padx=8)

        prev_frame = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        prev_frame.pack(fill="both", expand=True, padx=16, pady=6)
        self._preview_text = tk.Text(prev_frame, wrap="none",
                                     bg=C.CARD, fg=C.TEXT, font=("Consolas", 10),
                                     bd=0, highlightthickness=0,
                                     insertbackground=C.TEXT)
        ysb = ttk.Scrollbar(prev_frame, orient="vertical",
                             command=self._preview_text.yview)
        xsb = ttk.Scrollbar(prev_frame, orient="horizontal",
                             command=self._preview_text.xview)
        self._preview_text.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self._preview_text.pack(fill="both", expand=True, padx=4, pady=4)

        self._preview_text.tag_configure("preamble", foreground=C.MUTED)
        self._preview_text.tag_configure("header",   foreground=C.WARN)
        self._preview_text.tag_configure("data",     foreground=C.TEXT)

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Data files", "*.csv *.tsv *.xlsx *.xls"),
                       ("All files", "*.*")])
        if not p:
            return
        self.path = p
        self._path_label.configure(text=os.path.basename(p))
        self.encoding = detect_encoding(p)
        setattr(self, "_encoding_var", tk.StringVar(value=self.encoding))
        self._refresh_preview()

    def _on_setting(self, attr, var):
        setattr(self, attr, var.get())
        self._refresh_preview()

    def _refresh_preview(self):
        if not self.path:
            return
        try:
            self.header_row = int(self._hrow_spin.get())
        except Exception:
            self.header_row = 0
        self.raw_lines = read_raw_lines(self.path, self.encoding)
        if not self.raw_lines:
            return
        self.delimiter = detect_delimiter(self.raw_lines, self.header_row)
        auto_h = detect_table_start(self.raw_lines, self.delimiter)
        if int(self._hrow_spin.get()) == 0 and auto_h > 0:
            self._hrow_spin.delete(0, "end")
            self._hrow_spin.insert(0, str(auto_h))
            self.header_row = auto_h

        self._preview_text.configure(state="normal")
        self._preview_text.delete("1.0", "end")
        for i, line in enumerate(self.raw_lines[:80]):
            tag = ("preamble" if i < self.header_row
                   else "header" if i == self.header_row
                   else "data")
            self._preview_text.insert("end", line + "\n", tag)
        self._preview_text.configure(state="disabled")

        n_data = max(0, len(self.raw_lines) - self.header_row - 1)
        self._summary_var.set(
            f"Header: line {self.header_row + 1}  |  "
            f"Data from line {self.header_row + 2}  |  "
            f"~{n_data} data rows  |  Delimiter: {repr(self.delimiter)}")

    # ── Step 2 — Column prediction ─────────────────────────────────────────────

    def _build_step2(self, frame):
        self._load_raw_df()
        if self.df_raw is None:
            ctk.CTkLabel(frame,
                         text="Could not load file. Go back and check settings.",
                         text_color=C.DANGER).pack(pady=40)
            return

        self.predictions = {}
        self.col_vars    = {}
        self.map_vars    = {}

        samples = {c: get_samples(self.df_raw, c) for c in self.df_raw.columns}
        pred = get_predictor().predict_all(list(self.df_raw.columns), samples)
        self.predictions = pred

        hdr = ctk.CTkFrame(frame, fg_color=C.PANEL, corner_radius=0, height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        pred_obj = get_predictor()
        info = (f"{len(self.df_raw.columns)} columns detected  |  "
                f"{'Semantic model' if hasattr(pred_obj, '_model') and pred_obj._model else 'Fuzzy matching'} used")
        ctk.CTkLabel(hdr, text=info, text_color=C.MUTED, font=("Segoe UI", 11)
                     ).pack(side="left", padx=14, pady=12)

        btn_fr = ctk.CTkFrame(hdr, fg_color="transparent")
        btn_fr.pack(side="right", padx=10)
        b_all = ctk.CTkButton(btn_fr, text="Select All", width=90, fg_color=C.DIM,
                              command=lambda: self._bulk_select(True))
        b_all.pack(side="left", padx=4)
        tip(b_all, "Tick every column so all of them are imported.")
        b_core = ctk.CTkButton(btn_fr, text="Core Only", width=90, fg_color=C.ACCENT,
                               command=self._select_core_only)
        b_core.pack(side="left", padx=4)
        tip(b_core, "Tick only recognised standard social-media fields.")
        b_none = ctk.CTkButton(btn_fr, text="Deselect All", width=90, fg_color=C.DIM,
                               command=lambda: self._bulk_select(False))
        b_none.pack(side="left", padx=4)
        tip(b_none, "Untick every column to hand-pick from scratch.")

        col_hdr = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=0, height=30)
        col_hdr.pack(fill="x")
        col_hdr.pack_propagate(False)
        for text, w in [("Incl.", 50), ("Column Name", 200), ("Predicted Field", 180),
                        ("Category", 110), ("Conf.", 70), ("Sample Values", 0)]:
            ctk.CTkLabel(col_hdr, text=text, text_color=C.MUTED,
                         font=("Segoe UI", 9, "bold"), width=w, anchor="w"
                         ).pack(side="left", padx=(6, 0))

        scroll = ctk.CTkScrollableFrame(frame, fg_color=C.BG, corner_radius=0)
        scroll.pack(fill="both", expand=True)
        field_opts = ["(ignore)"] + list(CORE_COLUMNS.keys())

        for col in self.df_raw.columns:
            p = pred[col]
            field = p["field"] or "(ignore)"
            conf  = p["confidence"]
            cat   = p["category"]
            samp  = ", ".join(str(v)[:25] for v in samples.get(col, [])[:3])
            conf_color = C.SUCCESS if conf >= 0.75 else C.WARN if conf >= 0.45 else C.MUTED

            row = ctk.CTkFrame(scroll, fg_color=C.CARD, corner_radius=4, height=34)
            row.pack(fill="x", padx=6, pady=2)
            row.pack_propagate(False)

            var_inc = tk.BooleanVar(value=(field != "(ignore)"))
            self.col_vars[col] = var_inc
            ctk.CTkCheckBox(row, text="", variable=var_inc, width=30,
                            checkbox_width=16, checkbox_height=16,
                            onvalue=True, offvalue=False
                            ).pack(side="left", padx=(8, 0))

            ctk.CTkLabel(row, text=col, text_color=C.TEXT,
                         font=("Segoe UI", 11), width=190, anchor="w"
                         ).pack(side="left", padx=(4, 0))

            var_map = tk.StringVar(value=field)
            self.map_vars[col] = var_map
            field_om = ctk.CTkOptionMenu(row, variable=var_map, values=field_opts,
                                         width=170, font=("Segoe UI", 11),
                                         fg_color=C.PANEL, button_color=C.BORDER)
            field_om.pack(side="left", padx=4)
            _ft = FIELD_TIPS.get(field, "Change this if the prediction is wrong.")
            tip(field_om,
                f"Predicted field: {field}\n\n{_ft}\n\n"
                "Use the dropdown to correct it, or choose '(ignore)' to skip.")

            cat_color = CATEGORY_COLORS.get(cat, C.MUTED)
            ctk.CTkLabel(row, text=cat, text_color=cat_color,
                         font=("Segoe UI", 10), width=100, anchor="w"
                         ).pack(side="left", padx=4)

            ctk.CTkLabel(row, text=f"{conf:.0%}", text_color=conf_color,
                         font=("Segoe UI", 11, "bold"), width=60
                         ).pack(side="left", padx=4)

            ctk.CTkLabel(row, text=samp, text_color=C.MUTED,
                         font=("Consolas", 10), anchor="w"
                         ).pack(side="left", padx=4, fill="x", expand=True)

    def _load_raw_df(self):
        if not self.path:
            return
        try:
            ext = os.path.splitext(self.path)[1].lower()
            if ext in (".xlsx", ".xls"):
                self.df_raw = pd.read_excel(self.path, header=self.header_row)
            else:
                self.df_raw = pd.read_csv(
                    self.path, sep=self.delimiter, encoding=self.encoding,
                    header=self.header_row, low_memory=False, on_bad_lines="skip")
            self.df_raw.columns = [str(c).strip() for c in self.df_raw.columns]
        except Exception as e:
            messagebox.showerror("Load error", str(e), parent=self)
            self.df_raw = None

    def _bulk_select(self, state: bool):
        for v in self.col_vars.values():
            v.set(state)

    def _select_core_only(self):
        for col, v in self.col_vars.items():
            v.set(bool(self.predictions.get(col, {}).get("field")))

    # ── Step 3 — Mapping review ────────────────────────────────────────────────

    def _build_step3(self, frame):
        ctk.CTkLabel(frame, text="Review your column mapping before import.",
                     text_color=C.MUTED, font=("Segoe UI", 12)
                     ).pack(anchor="w", padx=20, pady=(14, 4))

        scroll = ctk.CTkScrollableFrame(frame, fg_color=C.BG)
        scroll.pack(fill="both", expand=True, padx=14, pady=6)

        hdr = ctk.CTkFrame(scroll, fg_color=C.PANEL, corner_radius=4, height=28)
        hdr.pack(fill="x", pady=(0, 4))
        hdr.pack_propagate(False)
        for text, w in [("Original Column", 260), ("Mapped To", 200),
                        ("Category", 130), ("Sample", 0)]:
            ctk.CTkLabel(hdr, text=text, text_color=C.MUTED,
                         font=("Segoe UI", 9, "bold"), width=w, anchor="w"
                         ).pack(side="left", padx=8)

        included = [(c, self.map_vars[c].get()) for c in self.col_vars
                    if self.col_vars[c].get()]
        for col, field in included:
            row = ctk.CTkFrame(scroll, fg_color=C.CARD, corner_radius=4, height=28)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)
            cat   = CORE_COLUMNS.get(field, {}).get("category", "—")
            color = CATEGORY_COLORS.get(cat, C.MUTED)
            samp  = ", ".join(str(v)[:20] for v in get_samples(self.df_raw, col)[:2])
            ctk.CTkLabel(row, text=col, text_color=C.TEXT,
                         font=("Segoe UI", 11), width=250, anchor="w"
                         ).pack(side="left", padx=8)
            ctk.CTkLabel(row, text=field if field != "(ignore)" else "—",
                         text_color=color, font=("Segoe UI", 11), width=190, anchor="w"
                         ).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=cat, text_color=color,
                         font=("Segoe UI", 10), width=120, anchor="w"
                         ).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=samp, text_color=C.MUTED,
                         font=("Consolas", 10), anchor="w"
                         ).pack(side="left", padx=4, fill="x", expand=True)

        if not included:
            ctk.CTkLabel(scroll, text="No columns selected. Go back.",
                         text_color=C.DANGER).pack(pady=20)

    # ── Step 4 — Quality report ────────────────────────────────────────────────

    def _build_step4(self, frame):
        df = self._build_final_df()
        if df is None:
            ctk.CTkLabel(frame, text="Error building dataset.", text_color=C.DANGER
                         ).pack(pady=40)
            return
        self._final_df = df

        scroll = ctk.CTkScrollableFrame(frame, fg_color=C.BG)
        scroll.pack(fill="both", expand=True, padx=14, pady=10)

        cards = ctk.CTkFrame(scroll, fg_color="transparent")
        cards.pack(fill="x", pady=(0, 10))
        for label, value in [("Rows", str(len(df))),
                              ("Columns", str(len(df.columns))),
                              ("Duplicates", str(df.duplicated().sum()))]:
            c = ctk.CTkFrame(cards, fg_color=C.CARD, corner_radius=8, width=140)
            c.pack(side="left", padx=6, pady=4)
            c.pack_propagate(False)
            ctk.CTkLabel(c, text=value, font=("Segoe UI", 22, "bold"),
                         text_color=C.ACCENT).pack(pady=(8, 0))
            ctk.CTkLabel(c, text=label, font=("Segoe UI", 11),
                         text_color=C.MUTED).pack(pady=(0, 8))

        ctk.CTkLabel(scroll, text="Missing values per column",
                     font=("Segoe UI", 12, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", pady=(4, 4))

        for col in df.columns:
            if col.startswith("_"):
                continue
            pct = df[col].isna().mean()
            color = C.SUCCESS if pct < 0.2 else C.WARN if pct < 0.8 else C.DANGER
            row = ctk.CTkFrame(scroll, fg_color=C.CARD, corner_radius=4, height=26)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)
            ctk.CTkLabel(row, text=col, text_color=C.TEXT,
                         font=("Segoe UI", 11), width=260, anchor="w"
                         ).pack(side="left", padx=8)
            bar_bg = ctk.CTkFrame(row, fg_color=C.BORDER, corner_radius=3,
                                  width=200, height=10)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)
            if pct > 0:
                ctk.CTkFrame(bar_bg, fg_color=color, corner_radius=3,
                             width=int(200 * pct), height=10).place(x=0, y=0)
            ctk.CTkLabel(row, text=f"{pct:.0%}", text_color=color,
                         font=("Segoe UI", 11), width=50).pack(side="left", padx=4)

    def _build_final_df(self):
        if self.df_raw is None:
            return None
        included = [(c, self.map_vars[c].get()) for c in self.col_vars
                    if self.col_vars[c].get()]
        if not included:
            return None
        col_map = {}
        cols    = {}
        for orig, field in included:
            if field == "(ignore)":
                cols[orig] = self.df_raw[orig]
            else:
                cols[field] = self.df_raw[orig]
                col_map[field] = orig
        df = pd.DataFrame(cols)
        df["_tags"] = ""
        if "created_at" in df.columns:
            df["_date"] = safe_date_parse(df["created_at"])
        else:
            df["_date"] = pd.NaT
        self._col_map_result = col_map
        return df

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _next(self):
        if self._step == 1:
            if not self.path:
                self._status_var.set("Please select a file first.")
                return
            self._status_var.set("")
            self._show_step(2)
        elif self._step == 2:
            included = [c for c, v in self.col_vars.items() if v.get()]
            if not included:
                self._status_var.set("Select at least one column.")
                return
            self._status_var.set("")
            if 3 in self._frames:
                self._frames[3].destroy()
                del self._frames[3]
            self._show_step(3)
        elif self._step == 3:
            if 4 in self._frames:
                self._frames[4].destroy()
                del self._frames[4]
            self._show_step(4)
        elif self._step == 4:
            self._finish()

    def _back(self):
        if self._step > 1:
            self._show_step(self._step - 1)

    def _finish(self):
        df = getattr(self, "_final_df", None)
        if df is None:
            return
        col_map = getattr(self, "_col_map_result", {})
        name = os.path.splitext(os.path.basename(self.path))[0]
        entry = DatasetEntry(name, df, col_map, self.path)
        self.dm.add(entry)
        self.on_finish(entry)
        self.destroy()


# ── Merge Wizard ───────────────────────────────────────────────────────────────

class MergeWizard(ctk.CTkToplevel):
    """
    Dialog for concatenating multiple datasets into one aligned table.
    Shows canonical-field coverage per dataset and lets the user choose fields.
    """

    def __init__(self, master, dm: DatasetManager, on_merge):
        super().__init__(master)
        self.title("Merge Datasets — Data Dominion")
        self.geometry("960x680")
        self.configure(fg_color=C.BG)
        self.grab_set()

        self.dm       = dm
        self.on_merge = on_merge
        self._field_vars: dict = {}

        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Merge / Concatenate Datasets",
                     font=("Segoe UI", 16, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=20, pady=(16, 4))
        ctk.CTkLabel(self,
                     text="Select canonical fields to include. "
                          "Columns missing in a dataset will be filled with N/A.",
                     text_color=C.MUTED, font=("Segoe UI", 11)
                     ).pack(anchor="w", padx=20, pady=(0, 10))

        suggestion = self.dm.suggest_merge()
        coverage   = suggestion.get("field_coverage", {})
        unmatched  = suggestion.get("unmatched", {})
        ds_names   = [e.name for e in self.dm.entries]

        hdr = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=38)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Incl.", text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold"), width=50).pack(side="left", padx=6)
        ctk.CTkLabel(hdr, text="Canonical Field", text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold"), width=160).pack(side="left")
        ctk.CTkLabel(hdr, text="Category", text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold"), width=110).pack(side="left")
        for name in ds_names:
            ctk.CTkLabel(hdr, text=name[:20], text_color=C.MUTED,
                         font=("Segoe UI", 9, "bold"), width=160
                         ).pack(side="left", padx=2)

        scroll = ctk.CTkScrollableFrame(self, fg_color=C.BG)
        scroll.pack(fill="both", expand=True)

        for field, info in CORE_COLUMNS.items():
            cat     = info["category"]
            color   = CATEGORY_COLORS.get(cat, C.MUTED)
            present = field in coverage

            row = ctk.CTkFrame(scroll, fg_color=C.CARD, corner_radius=4, height=30)
            row.pack(fill="x", padx=6, pady=2)
            row.pack_propagate(False)

            var = tk.BooleanVar(value=present)
            self._field_vars[field] = var
            cb = ctk.CTkCheckBox(row, text="", variable=var, width=40,
                                 checkbox_width=16, checkbox_height=16)
            cb.pack(side="left", padx=(8, 0))
            _fdesc = FIELD_TIPS.get(field, info.get("description", ""))
            tip(cb, f"{field}\n\n{_fdesc}")

            req_star = " *" if info.get("required") else ""
            ctk.CTkLabel(row, text=field + req_star, text_color=C.TEXT,
                         font=("Segoe UI", 11), width=160, anchor="w"
                         ).pack(side="left", padx=2)
            ctk.CTkLabel(row, text=cat, text_color=color,
                         font=("Segoe UI", 10), width=100, anchor="w"
                         ).pack(side="left", padx=4)

            for name in ds_names:
                orig = coverage.get(field, {}).get(name, "")
                ctk.CTkLabel(row, text=orig[:20] if orig else "— N/A —",
                             text_color=C.SUCCESS if orig else C.MUTED,
                             font=("Segoe UI", 10), width=150, anchor="w"
                             ).pack(side="left", padx=2)

        if any(unmatched.values()):
            ctk.CTkLabel(scroll, text="Unmatched columns (will not appear in merge):",
                         text_color=C.MUTED, font=("Segoe UI", 10)
                         ).pack(anchor="w", padx=10, pady=(8, 2))
            for ds_name, cols in unmatched.items():
                if cols:
                    ctk.CTkLabel(scroll,
                                 text=f"  {ds_name}: {', '.join(cols[:8])}",
                                 text_color=C.DIM, font=("Consolas", 10)
                                 ).pack(anchor="w", padx=14)

        ftr = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=54)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)
        ctk.CTkButton(ftr, text="Cancel", fg_color=C.DIM, hover_color=C.BORDER,
                      width=100, command=self.destroy).pack(side="left", padx=16, pady=10)

        self._preview_label = ctk.CTkLabel(ftr, text="", text_color=C.MUTED,
                                           font=("Segoe UI", 11))
        self._preview_label.pack(side="left", padx=12)

        ctk.CTkButton(ftr, text="Merge Now", fg_color=C.SUCCESS, width=140,
                      command=self._do_merge).pack(side="right", padx=16, pady=10)

        for var in self._field_vars.values():
            var.trace_add("write", self._update_preview)
        self._update_preview()

    def _update_preview(self, *_):
        fields = [f for f, v in self._field_vars.items() if v.get()]
        total  = sum(e.row_count for e in self.dm.entries)
        self._preview_label.configure(
            text=f"Preview: {total} rows × {len(fields) + 1} cols "
                 f"(+_source_dataset) from {len(self.dm.entries)} datasets")

    def _do_merge(self):
        fields = [f for f, v in self._field_vars.items() if v.get()]
        if not fields:
            messagebox.showwarning("No fields", "Select at least one field.", parent=self)
            return
        merged = self.dm.merge(fields)
        self.on_merge(merged)
        self.destroy()
