"""Slicer frame — build, save and export named research datasets."""

from __future__ import annotations
from typing import TYPE_CHECKING
import math
import os
import re
from tkinter import filedialog, messagebox, ttk

import tkinter as tk
import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip, _style_treeview, DateRangeSlider

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App


# ── Brandwatch-style Boolean parser ────────────────────────────────────────────

def _tokenize(query: str) -> list:
    tokens = []
    i = 0
    while i < len(query):
        if query[i].isspace():
            i += 1
        elif query[i] == "(":
            tokens.append(("LPAREN", "("));  i += 1
        elif query[i] == ")":
            tokens.append(("RPAREN", ")"));  i += 1
        elif query[i] == '"':
            j = query.find('"', i + 1)
            j = len(query) if j == -1 else j
            tokens.append(("PHRASE", query[i + 1:j]));  i = j + 1
        else:
            j = i
            while j < len(query) and not query[j].isspace() and query[j] not in '()"':
                j += 1
            word = query[i:j]
            upper = word.upper()
            if   upper == "AND":  tokens.append(("AND",  "AND"))
            elif upper == "OR":   tokens.append(("OR",   "OR"))
            elif upper == "NOT":  tokens.append(("NOT",  "NOT"))
            elif upper == "NEAR": tokens.append(("AND",  "AND"))   # treat NEAR as AND
            elif word:            tokens.append(("WORD", word))
            i = j
    return tokens


class _BoolParser:
    """Recursive descent parser for Brandwatch-style boolean queries.

    Grammar:
        expr     = or_expr
        or_expr  = and_expr  ('OR' and_expr)*
        and_expr = not_expr  ('AND' not_expr | implicit-AND not_expr)*
        not_expr = 'NOT' not_expr | atom
        atom     = '(' expr ')' | PHRASE | WORD
    """

    def __init__(self, tokens: list, series: pd.Series):
        self.tokens = tokens
        self.pos    = 0
        self.series = series.fillna("").astype(str).str.lower()

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected=None):
        tok = self.tokens[self.pos]
        if expected and tok[0] != expected:
            raise ValueError(f"Expected {expected}, got {tok}")
        self.pos += 1
        return tok

    def _true(self):
        return pd.Series(True, index=self.series.index)

    def parse(self):
        return self._or() if self.tokens else self._true()

    def _or(self):
        left = self._and()
        while self._peek() and self._peek()[0] == "OR":
            self._consume("OR")
            left = left | self._and()
        return left

    def _and(self):
        left = self._not()
        while self._peek():
            tok = self._peek()
            if tok[0] == "AND":
                self._consume("AND")
                left = left & self._not()
            elif tok[0] in ("WORD", "PHRASE", "LPAREN", "NOT"):  # implicit AND
                left = left & self._not()
            else:
                break
        return left

    def _not(self):
        if self._peek() and self._peek()[0] == "NOT":
            self._consume("NOT")
            return ~self._not()
        return self._atom()

    def _atom(self):
        tok = self._peek()
        if tok is None:
            return self._true()
        if tok[0] == "LPAREN":
            self._consume("LPAREN")
            result = self._or()
            if self._peek() and self._peek()[0] == "RPAREN":
                self._consume("RPAREN")
            return result
        if tok[0] == "PHRASE":
            self._consume("PHRASE")
            return self.series.str.contains(re.escape(tok[1].lower()), na=False)
        if tok[0] == "WORD":
            self._consume("WORD")
            word = tok[1].lower()
            pattern = re.escape(word).replace(r"\*", r"\w*") if "*" in word else re.escape(word)
            return self.series.str.contains(pattern, na=False)
        self.pos += 1   # skip unexpected token
        return self._true()


def apply_boolean_query(series: pd.Series, query: str) -> pd.Series:
    """
    Apply a Brandwatch-style boolean query to a text Series.
    Returns a boolean mask (True = row matches).
    """
    query = query.strip()
    if not query:
        return pd.Series(True, index=series.index)
    try:
        return _BoolParser(_tokenize(query), series).parse()
    except Exception as exc:
        messagebox.showerror(
            "Boolean query error",
            f"Could not parse query:\n{exc}\n\n"
            "Check AND / OR / NOT / quotes / parentheses.")
        return pd.Series(True, index=series.index)


# ── Help text ──────────────────────────────────────────────────────────────────

BOOL_HELP = (
    "Brandwatch-style boolean query applied to the text column.\n\n"
    "Operators (case-insensitive):\n"
    "  AND  — both terms must appear\n"
    "  OR   — either term must appear\n"
    "  NOT  — term must NOT appear\n\n"
    'Quoting:  "exact phrase"\n\n'
    "Wildcards:  climat*  → climate, climatic, …\n\n"
    "Grouping:  (A OR B) AND C\n\n"
    "Adjacent words without an operator = implicit AND.\n\n"
    "Examples:\n"
    "  climate AND (change OR warming)\n"
    '  "fake news" OR disinformation\n'
    "  covid NOT (vaccine OR mask)\n"
    "  (ISD OR extremism) AND NOT satire"
)


# ── Column type detection ─────────────────────────────────────────────────────

def _detect_col_type(df: pd.DataFrame, col: str) -> str:
    """Return 'text', 'numeric', 'boolean', 'datetime', or 'categorical'."""
    if col == "_date" or pd.api.types.is_datetime64_any_dtype(df[col]):
        return "datetime"
    if pd.api.types.is_bool_dtype(df[col]):
        return "boolean"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    nunique = df[col].nunique()
    if 0 < nunique <= 25:
        return "categorical"
    return "text"


FILTER_OPS = {
    "text":        ["contains", "not contains", "equals", "not equals", "is empty", "is not empty"],
    "numeric":     ["=", ">", "<", ">=", "<=", "between"],
    "boolean":     ["is True", "is False"],
    "datetime":    ["after", "before", "between"],
    "categorical": ["equals", "not equals", "is empty", "is not empty"],
}

# Ops that need no value entry
_NO_VALUE_OPS = {"is True", "is False", "is empty", "is not empty"}


# ── Column filter row widget ──────────────────────────────────────────────────

class _ColumnFilterRow(ctk.CTkFrame):
    """One filter rule: [column] [operation] [value] [×]."""

    def __init__(self, master, columns: list, df: pd.DataFrame,
                 on_remove, on_change=None):
        super().__init__(master, fg_color="transparent")
        self._df = df
        self._on_change = on_change

        self._col_var = tk.StringVar(value=columns[0] if columns else "")
        self._op_var  = tk.StringVar()
        self._val_var = tk.StringVar()
        self._val2_var = tk.StringVar()   # for "between"

        # Column dropdown
        self._col_om = ctk.CTkOptionMenu(
            self, variable=self._col_var, values=columns, width=160, height=26,
            command=self._on_col_change)
        self._col_om.pack(side="left", padx=(0, 4))

        # Operation dropdown
        self._op_om = ctk.CTkOptionMenu(
            self, variable=self._op_var, values=["—"], width=120, height=26,
            command=self._on_op_change)
        self._op_om.pack(side="left", padx=(0, 4))

        # Value entry (or dropdown for categorical)
        self._val_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._val_frame.pack(side="left", padx=(0, 4))
        self._val_entry = ctk.CTkEntry(self._val_frame, textvariable=self._val_var,
                                       width=160, height=26,
                                       placeholder_text="value…")
        self._val_entry.pack(side="left")

        # Second value entry for "between"
        self._val2_entry = ctk.CTkEntry(self._val_frame, textvariable=self._val2_var,
                                        width=100, height=26,
                                        placeholder_text="to…")

        # Remove button
        ctk.CTkButton(self, text="✕", width=26, height=26, fg_color=C.BTN,
                      command=lambda: on_remove(self)).pack(side="left", padx=2)

        self._on_col_change(self._col_var.get())

    def _on_col_change(self, col_name):
        """Update operations dropdown based on column type."""
        col = col_name
        if col and col in self._df.columns:
            ctype = _detect_col_type(self._df, col)
        else:
            ctype = "text"
        ops = FILTER_OPS.get(ctype, FILTER_OPS["text"])
        self._op_om.configure(values=ops)
        self._op_var.set(ops[0])
        self._on_op_change(ops[0])

        # For categorical, replace value entry with dropdown
        for w in self._val_frame.winfo_children():
            w.pack_forget()
        self._val2_entry.pack_forget()

        if ctype == "categorical" and col in self._df.columns:
            uniques = sorted(self._df[col].dropna().astype(str).unique().tolist())
            if uniques:
                self._val_var.set(uniques[0])
                cat_om = ctk.CTkOptionMenu(self._val_frame, variable=self._val_var,
                                           values=uniques, width=160, height=26)
                cat_om.pack(side="left")
                return

        self._val_entry.pack(side="left")
        if self._on_change:
            self._on_change()

    def _on_op_change(self, op):
        """Show/hide value entries based on operation."""
        self._val2_entry.pack_forget()
        if op in _NO_VALUE_OPS:
            self._val_entry.pack_forget()
        else:
            if not self._val_entry.winfo_ismapped():
                self._val_entry.pack(side="left")
            if op == "between":
                self._val2_entry.pack(side="left", padx=(4, 0))
        if self._on_change:
            self._on_change()

    def get_config(self) -> dict:
        return {
            "column":    self._col_var.get(),
            "operation": self._op_var.get(),
            "value":     self._val_var.get(),
            "value2":    self._val2_var.get(),
        }

    def set_config(self, cfg: dict):
        col = cfg.get("column", "")
        if col:
            self._col_var.set(col)
            self._on_col_change(col)
        op = cfg.get("operation", "")
        if op:
            self._op_var.set(op)
            self._on_op_change(op)
        self._val_var.set(cfg.get("value", ""))
        self._val2_var.set(cfg.get("value2", ""))


# ── SlicerFrame ────────────────────────────────────────────────────────────────

class SlicerFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._col_vars: dict[str, tk.BooleanVar] = {}
        self._filter_rows: list[_ColumnFilterRow] = []
        self._init_vars()
        self._build()

    def _init_vars(self):
        self._df_var          = tk.StringVar()    # date from (slider)
        self._dt_var          = tk.StringVar()    # date to   (slider)
        self._date_col_var    = tk.StringVar()    # which date column
        self._text_col_var    = tk.StringVar()    # column for keyword/boolean
        self._kw_var          = tk.StringVar()
        self._bool_var        = tk.StringVar()
        self._dedup_var       = tk.BooleanVar(value=False)
        self._sample_var      = tk.BooleanVar(value=False)
        self._sample_n_var    = tk.StringVar(value="500")
        self._seed_var        = tk.StringVar(value="42")
        self._chunk_var       = tk.BooleanVar(value=False)
        self._chunk_k_var     = tk.StringVar(value="2")
        self._overlap_var     = tk.BooleanVar(value=False)
        self._overlap_pct_var = tk.StringVar(value="10")
        self._row_id_var      = tk.BooleanVar(value=True)
        self._codebook_var    = tk.StringVar()
        self._slice_name_var  = tk.StringVar()
        self._slice_notes_var = tk.StringVar()
        self._col_vars        = {}
        self._filter_rows     = []

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        # Title bar
        tb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=48)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="Slicer", font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(side="left", padx=16, pady=14)
        ctk.CTkLabel(tb, text="Build custom research datasets",
                     text_color=C.MUTED, font=("Segoe UI", 11)
                     ).pack(side="left", padx=4)

        # Always-visible status bar with live count
        status = ctk.CTkFrame(self, fg_color=C.CARD, corner_radius=0, height=42)
        status.pack(fill="x")
        status.pack_propagate(False)
        self._status_label = ctk.CTkLabel(
            status, text="  Load a dataset to begin slicing.",
            text_color=C.TEXT, font=("Segoe UI", 12, "bold"))
        self._status_label.pack(side="left", padx=16)
        self._live_count = ctk.CTkLabel(
            status, text="", text_color=C.ACCENT,
            font=("Segoe UI", 11, "bold"))
        self._live_count.pack(side="left", padx=8)
        btn = ctk.CTkButton(status, text="Apply & Preview", width=130,
                            fg_color=C.ACCENT, command=self._update_preview)
        btn.pack(side="right", padx=12)
        tip(btn, "Apply all filters and sampling, then refresh the preview below.")

        # Scrollable body
        scroll = ctk.CTkScrollableFrame(self, fg_color=C.BG)
        scroll.pack(fill="both", expand=True)

        self._build_filters(scroll)
        self._build_sampling(scroll)
        self._build_columns(scroll)
        self._build_preview_table(scroll)
        self._build_export(scroll)
        self._build_slices_panel(scroll)

    # ── Filters ──────────────────────────────────────────────────────────────

    def _build_filters(self, parent):
        sec = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=8)
        sec.pack(fill="x", padx=14, pady=(10, 6))

        # ── Date range slider ────────────────────────────────────────
        date_hdr = ctk.CTkFrame(sec, fg_color="transparent")
        date_hdr.pack(fill="x", padx=14, pady=(10, 2))
        ctk.CTkLabel(date_hdr, text="DATE RANGE",
                     font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(side="left")

        dcl = ctk.CTkLabel(date_hdr, text="Date column:", text_color=C.MUTED,
                           font=("Segoe UI", 10))
        dcl.pack(side="left", padx=(16, 4))
        tip(dcl, "Choose which date column to use for the range filter\n"
                 "(e.g. created_at, posted_at, collected_at).")
        self._date_col_om = ctk.CTkOptionMenu(
            date_hdr, variable=self._date_col_var, values=["—"],
            width=150, height=24, command=self._on_date_col_change)
        self._date_col_om.pack(side="left", padx=(0, 12))

        clr = ctk.CTkButton(date_hdr, text="Reset dates", width=80,
                            height=22, fg_color=C.BTN, font=("Segoe UI", 9),
                            command=self._reset_dates)
        clr.pack(side="left", padx=4)
        tip(clr, "Clear the date filter and show all dates.")

        self._date_slider = DateRangeSlider(
            sec, from_var=self._df_var, to_var=self._dt_var, width=400)
        self._date_slider.pack(anchor="w", padx=14, pady=(0, 8))

        # ── Column filters ───────────────────────────────────────────
        sep1 = ctk.CTkFrame(sec, fg_color=C.DIM, height=1)
        sep1.pack(fill="x", padx=14, pady=4)

        hdr = ctk.CTkFrame(sec, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(4, 4))
        ctk.CTkLabel(hdr, text="COLUMN FILTERS", font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(side="left")
        add_btn = ctk.CTkButton(hdr, text="+ Add Filter", fg_color=C.ACCENT,
                                width=100, height=26,
                                command=self._add_filter_row)
        add_btn.pack(side="left", padx=12)
        tip(add_btn, "Add a column-specific filter.\n"
                     "Pick any column, choose an operation\n"
                     "(contains, equals, >, <, etc.), and enter a value.")
        count_btn = ctk.CTkButton(hdr, text="Count", fg_color=C.BTN,
                                  width=60, height=26,
                                  command=self._quick_count)
        count_btn.pack(side="left", padx=2)
        tip(count_btn, "Quickly count how many rows match\n"
                       "the current filters without refreshing\n"
                       "the full preview.")

        # Dynamic filter rows container
        self._filter_container = ctk.CTkFrame(sec, fg_color="transparent")
        self._filter_container.pack(fill="x", padx=12, pady=4)

        # Hint when empty
        self._filter_hint = ctk.CTkLabel(
            self._filter_container,
            text="No column filters. Click '+ Add Filter' to filter by any column.",
            text_color=C.MUTED, font=("Segoe UI", 10))
        self._filter_hint.pack(pady=4)

        # ── Text search (keyword + boolean) ──────────────────────────
        sep2 = ctk.CTkFrame(sec, fg_color=C.DIM, height=1)
        sep2.pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(sec, text="TEXT SEARCH", font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(anchor="w", padx=14, pady=(4, 2))

        # Text column selector
        tcol_row = ctk.CTkFrame(sec, fg_color="transparent")
        tcol_row.pack(fill="x", padx=12, pady=(0, 4))
        tcl = ctk.CTkLabel(tcol_row, text="Search in column:",
                           text_color=C.MUTED, font=("Segoe UI", 11), width=130)
        tcl.pack(side="left")
        tip(tcl, "Choose which column the Keyword and Boolean\n"
                 "searches below will apply to.")
        self._text_col_om = ctk.CTkOptionMenu(
            tcol_row, variable=self._text_col_var,
            values=["— select —"], width=200)
        self._text_col_om.pack(side="left", padx=(0, 8))

        # Keyword
        krow = ctk.CTkFrame(sec, fg_color="transparent")
        krow.pack(fill="x", padx=12, pady=4)
        kl = ctk.CTkLabel(krow, text="Keyword:", text_color=C.MUTED,
                          font=("Segoe UI", 11), width=130)
        kl.pack(side="left")
        tip(kl, "Case-insensitive substring search in the column\n"
                "selected above.")
        ctk.CTkEntry(krow, textvariable=self._kw_var, width=400,
                     placeholder_text="e.g. climate change").pack(side="left")

        # Boolean query
        brow = ctk.CTkFrame(sec, fg_color="transparent")
        brow.pack(fill="x", padx=12, pady=4)
        bl = ctk.CTkLabel(brow, text="Boolean:", text_color=C.MUTED,
                          font=("Segoe UI", 11), width=130)
        bl.pack(side="left")
        tip(bl, BOOL_HELP)
        ctk.CTkEntry(brow, textvariable=self._bool_var, width=400,
                     placeholder_text='e.g. (climate OR warming) AND NOT hoax'
                     ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(brow, text="?", width=28, height=28, fg_color=C.BTN,
                      font=("Segoe UI", 11, "bold"),
                      command=lambda: messagebox.showinfo("Boolean Syntax", BOOL_HELP)
                      ).pack(side="left")

        # Dedup
        opt_row = ctk.CTkFrame(sec, fg_color="transparent")
        opt_row.pack(fill="x", padx=12, pady=(4, 10))
        dd = ctk.CTkCheckBox(opt_row, text="Remove exact duplicate texts",
                             variable=self._dedup_var)
        dd.pack(side="left")
        tip(dd, "Remove rows with identical text content\n"
                "(based on the search column selected above).\n"
                "Keeps the first occurrence of each unique text.")

    def _reset_dates(self):
        self._df_var.set("")
        self._dt_var.set("")
        self._date_slider.reset()

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
        # Clear any existing date filter so slider shows full range
        self._df_var.set("")
        self._dt_var.set("")

    def _refresh_filter_options(self):
        """Populate date slider, date column, text column from current data."""
        df = self.app.df
        if df.empty:
            return
        # Date column dropdown
        candidates = self.app.date_column_candidates()
        if candidates:
            self._date_col_om.configure(values=candidates)
            cur_dc = self._date_col_var.get()
            if not cur_dc or cur_dc == "—" or cur_dc not in candidates:
                src = self.app.date_source_col
                if src and src in candidates:
                    self._date_col_var.set(src)
                else:
                    self._date_col_var.set(candidates[0])
        else:
            self._date_col_om.configure(values=["—"])
            self._date_col_var.set("—")
        # Date slider
        if "_date" in df.columns:
            dates = df["_date"].dropna()
            if not dates.empty:
                self._date_slider.set_date_range(dates.min(), dates.max())
        # Text column dropdown
        cols = [c for c in df.columns if not c.startswith("_")]
        self._text_col_om.configure(values=cols if cols else ["— select —"])
        cur = self._text_col_var.get()
        if not cur or cur == "— select —" or cur not in cols:
            text_col = self.app.resolve_col("content_text", df)
            if text_col and text_col in cols:
                self._text_col_var.set(text_col)
            elif cols:
                self._text_col_var.set(cols[0])

    def _add_filter_row(self):
        df = self.app.df
        if df.empty:
            messagebox.showinfo("Filter", "Load a dataset first.")
            return
        # Hide hint
        self._filter_hint.pack_forget()

        cols = list(df.columns)
        row = _ColumnFilterRow(self._filter_container, cols, df,
                               self._remove_filter_row,
                               on_change=self._on_filter_change)
        row.pack(fill="x", pady=2)
        self._filter_rows.append(row)

    def _remove_filter_row(self, row):
        row.destroy()
        if row in self._filter_rows:
            self._filter_rows.remove(row)
        if not self._filter_rows:
            self._filter_hint.pack(pady=4)
        self._on_filter_change()

    def _on_filter_change(self):
        """Called when any filter row changes — update live count."""
        self._quick_count()

    def _quick_count(self):
        """Show how many rows match the current filters without full preview."""
        if self.app.df.empty:
            return
        try:
            df = self._get_sliced_df()
            total = len(self.app.df)
            n = len(df)
            pct = f"{n / total * 100:.1f}%" if total else "--"
            self._live_count.configure(
                text=f"→ {n:,} of {total:,} rows ({pct})")
        except Exception:
            self._live_count.configure(text="")

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _build_sampling(self, parent):
        sec = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=8)
        sec.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(sec, text="SAMPLING", font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(anchor="w", padx=14, pady=(10, 4))

        # Random sample
        srow = ctk.CTkFrame(sec, fg_color="transparent")
        srow.pack(fill="x", padx=12, pady=4)
        sc = ctk.CTkCheckBox(srow, text="Random sample",
                             variable=self._sample_var)
        sc.pack(side="left")
        tip(sc, "Draw a random subset of the filtered data.\n"
                "Set a seed for reproducible results.")
        ctk.CTkLabel(srow, text="N:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(16, 4))
        ne = ctk.CTkEntry(srow, textvariable=self._sample_n_var, width=80)
        ne.pack(side="left")
        tip(ne, "Number of rows to sample. If larger than the\n"
                "filtered set, all rows are returned.")
        ctk.CTkLabel(srow, text="Seed:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(16, 4))
        se = ctk.CTkEntry(srow, textvariable=self._seed_var, width=60)
        se.pack(side="left")
        tip(se, "Random seed for reproducibility.\n"
                "Same seed + same data = same sample every time.")

        # Chunks
        crow = ctk.CTkFrame(sec, fg_color="transparent")
        crow.pack(fill="x", padx=12, pady=4)
        cc = ctk.CTkCheckBox(crow, text="Split into equal chunks",
                             variable=self._chunk_var)
        cc.pack(side="left")
        tip(cc, "Divide the slice into K equal parts for distributing\n"
                "to different coders or annotators.\n"
                "Each chunk is exported as a separate file.")
        ctk.CTkLabel(crow, text="K:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(16, 4))
        ke = ctk.CTkEntry(crow, textvariable=self._chunk_k_var, width=60)
        ke.pack(side="left")
        tip(ke, "Number of chunks to split into (2–20).")

        # Overlap for ICR
        orow = ctk.CTkFrame(sec, fg_color="transparent")
        orow.pack(fill="x", padx=12, pady=(0, 10))
        oc = ctk.CTkCheckBox(orow, text="Add overlap for inter-coder reliability",
                             variable=self._overlap_var)
        oc.pack(side="left", padx=(24, 0))
        tip(oc, "Add a shared subset of rows to every chunk.\n"
                "This lets you calculate inter-coder reliability\n"
                "(e.g. Krippendorff's alpha, Cohen's kappa).")
        ctk.CTkLabel(orow, text="Overlap %:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(12, 4))
        oe = ctk.CTkEntry(orow, textvariable=self._overlap_pct_var, width=50)
        oe.pack(side="left")
        tip(oe, "Percentage of total rows shared across all chunks.\n"
                "These rows appear in every chunk for reliability checks.")

    # ── Columns ──────────────────────────────────────────────────────────────

    def _build_columns(self, parent):
        sec = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=8)
        sec.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(sec, text="COLUMNS", font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(anchor="w", padx=14, pady=(10, 4))

        # Select all / deselect all
        btn_row = ctk.CTkFrame(sec, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 4))
        sa = ctk.CTkButton(btn_row, text="Select All", width=90, fg_color=C.BTN,
                      command=self._select_all_cols)
        sa.pack(side="left", padx=(0, 6))
        tip(sa, "Check all column checkboxes.")
        da = ctk.CTkButton(btn_row, text="Deselect All", width=100, fg_color=C.BTN,
                      command=self._deselect_all_cols)
        da.pack(side="left")
        tip(da, "Uncheck all column checkboxes.")

        # Column checkboxes (scrollable, populated dynamically)
        self._col_frame = ctk.CTkScrollableFrame(sec, fg_color="transparent", height=110)
        self._col_frame.pack(fill="x", padx=8, pady=4)

        # Row ID
        extra_row = ctk.CTkFrame(sec, fg_color="transparent")
        extra_row.pack(fill="x", padx=12, pady=(4, 4))
        rid = ctk.CTkCheckBox(extra_row, text="Add row ID column",
                              variable=self._row_id_var)
        rid.pack(side="left")
        tip(rid, "Prepend a sequential ID column (1, 2, 3, …)\n"
                 "for easy reference in codebooks and annotation.")

        # Codebook columns
        cb_row = ctk.CTkFrame(sec, fg_color="transparent")
        cb_row.pack(fill="x", padx=12, pady=(0, 10))
        cbl = ctk.CTkLabel(cb_row, text="Codebook columns:", text_color=C.MUTED,
                           font=("Segoe UI", 11), width=140)
        cbl.pack(side="left")
        tip(cbl, "Comma-separated names of empty columns to add.\n"
                 "These appear in the export for coders to fill in.\n\n"
                 "Example: relevance, sentiment, notes")
        cbe = ctk.CTkEntry(cb_row, textvariable=self._codebook_var, width=360,
                           placeholder_text="e.g. relevance, sentiment, notes")
        cbe.pack(side="left")
        tip(cbe, "Empty columns added to every exported file.\n"
                 "Coders fill these in during annotation.")

    # ── Preview table ────────────────────────────────────────────────────────

    def _build_preview_table(self, parent):
        sec = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=8)
        sec.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(sec, text="PREVIEW (first 20 rows)",
                     font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(anchor="w", padx=14, pady=(10, 4))

        self._tree_frame = ctk.CTkFrame(sec, fg_color="transparent")
        self._tree_frame.pack(fill="x", padx=8, pady=(0, 10))

        self._tree = None
        self._tree_placeholder = ctk.CTkLabel(
            self._tree_frame,
            text="Click 'Apply & Preview' to see a preview of your slice.",
            text_color=C.MUTED, font=("Segoe UI", 11))
        self._tree_placeholder.pack(pady=20)

    # ── Export ───────────────────────────────────────────────────────────────

    def _build_export(self, parent):
        sec = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=8)
        sec.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(sec, text="EXPORT", font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(anchor="w", padx=14, pady=(10, 4))

        brow = ctk.CTkFrame(sec, fg_color="transparent")
        brow.pack(fill="x", padx=12, pady=(0, 6))
        for label, fmt, tt in [
            ("Excel (.xlsx)", "excel", "Export the current slice as an Excel workbook."),
            ("CSV",           "csv",   "Export the current slice as a CSV file."),
            ("JSON",          "json",  "Export the current slice as a JSON file (records)."),
        ]:
            btn = ctk.CTkButton(brow, text=label, fg_color=C.ACCENT, width=120,
                                command=lambda f=fmt: self._export_single(f))
            btn.pack(side="left", padx=(0, 8))
            tip(btn, tt)

        # Chunk export
        chunk_row = ctk.CTkFrame(sec, fg_color="transparent")
        chunk_row.pack(fill="x", padx=12, pady=(0, 10))
        cb = ctk.CTkButton(chunk_row, text="Export Chunks…", fg_color=C.SUCCESS,
                           width=140, command=self._export_chunks)
        cb.pack(side="left")
        tip(cb, "Export each chunk as a separate file into a folder.\n"
                "Enable 'Split into equal chunks' in Sampling first.\n"
                "Files are named: slicename_chunk1.xlsx, slicename_chunk2.xlsx, …")
        self._chunk_info = ctk.CTkLabel(chunk_row, text="", text_color=C.MUTED,
                                        font=("Segoe UI", 10))
        self._chunk_info.pack(side="left", padx=12)

    # ── Saved Slices ─────────────────────────────────────────────────────────

    def _build_slices_panel(self, parent):
        sec = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=8)
        sec.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(sec, text="SAVED SLICES", font=("Segoe UI", 10, "bold"),
                     text_color=C.MUTED).pack(anchor="w", padx=14, pady=(10, 4))

        save_row = ctk.CTkFrame(sec, fg_color="transparent")
        save_row.pack(fill="x", padx=12, pady=(0, 4))
        ne = ctk.CTkEntry(save_row, textvariable=self._slice_name_var,
                          placeholder_text="Slice name…", width=180)
        ne.pack(side="left", padx=(0, 8))
        tip(ne, "Name for this slice configuration. Auto-names if empty.")
        note_e = ctk.CTkEntry(save_row, textvariable=self._slice_notes_var,
                              placeholder_text="Notes (optional)…", width=260)
        note_e.pack(side="left", padx=(0, 8))
        tip(note_e, "Free-text note, e.g. 'English subset for RQ1'.")
        sb = ctk.CTkButton(save_row, text="Save Slice", fg_color=C.SUCCESS,
                           width=110, command=self._save)
        sb.pack(side="left")
        tip(sb, "Save the current filter, sampling, and column configuration.")

        self._slices_frame = ctk.CTkScrollableFrame(sec, fg_color="transparent",
                                                    height=220)
        self._slices_frame.pack(fill="x", padx=8, pady=(4, 8))
        self._refresh_slices()

    # ── Column management ────────────────────────────────────────────────────

    def _populate_columns(self):
        """Rebuild column checkboxes from the current dataset."""
        for w in self._col_frame.winfo_children():
            w.destroy()
        self._col_vars.clear()

        if self.app.df.empty:
            ctk.CTkLabel(self._col_frame, text="No dataset loaded.",
                         text_color=C.MUTED, font=("Segoe UI", 10)).pack()
            return

        cols = [c for c in self.app.df.columns if not c.startswith("_")]
        # Grid layout: 3 columns
        for i, col in enumerate(cols):
            var = tk.BooleanVar(value=True)
            self._col_vars[col] = var
            cb = ctk.CTkCheckBox(self._col_frame, text=col[:30], variable=var,
                                 font=("Segoe UI", 10))
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=6, pady=2)

    def _select_all_cols(self):
        for var in self._col_vars.values():
            var.set(True)

    def _deselect_all_cols(self):
        for var in self._col_vars.values():
            var.set(False)

    def _selected_columns(self) -> list[str]:
        """Return list of columns the user has checked."""
        selected = [c for c, v in self._col_vars.items() if v.get()]
        if not selected:
            # Fallback: all non-internal columns
            return [c for c in self.app.df.columns if not c.startswith("_")]
        return selected

    # ── Core slicing logic ───────────────────────────────────────────────────

    def _current_config(self) -> dict:
        """Read all UI fields into a config dict."""
        return {
            "column_filters": [r.get_config() for r in self._filter_rows],
            "date_from":    self._df_var.get().strip(),
            "date_to":      self._dt_var.get().strip(),
            "date_column":  self._date_col_var.get().strip(),
            "text_column":  self._text_col_var.get().strip(),
            "keyword":      self._kw_var.get().strip(),
            "boolean":      self._bool_var.get().strip(),
            "dedup":        self._dedup_var.get(),
            "sample":       self._sample_var.get(),
            "sample_n":     self._sample_n_var.get().strip(),
            "seed":         self._seed_var.get().strip(),
            "chunk":        self._chunk_var.get(),
            "chunk_k":      self._chunk_k_var.get().strip(),
            "overlap":      self._overlap_var.get(),
            "overlap_pct":  self._overlap_pct_var.get().strip(),
            "row_id":       self._row_id_var.get(),
            "codebook":     self._codebook_var.get().strip(),
            "notes":        self._slice_notes_var.get().strip(),
        }

    def _apply_column_filters(self, df: pd.DataFrame,
                              filters: list[dict]) -> pd.DataFrame:
        """Apply a list of column filter dicts to df."""
        for filt in filters:
            col = filt.get("column", "")
            op  = filt.get("operation", "")
            val = filt.get("value", "")
            val2 = filt.get("value2", "")

            if col not in df.columns or not op:
                continue

            try:
                if op == "contains":
                    df = df[df[col].astype(str).str.contains(
                        re.escape(val), case=False, na=False)]
                elif op == "not contains":
                    df = df[~df[col].astype(str).str.contains(
                        re.escape(val), case=False, na=False)]
                elif op == "equals":
                    df = df[df[col].astype(str) == val]
                elif op == "not equals":
                    df = df[df[col].astype(str) != val]
                elif op == "is empty":
                    df = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
                elif op == "is not empty":
                    df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
                elif op == "is True":
                    df = df[df[col].astype(str).str.lower().isin(
                        ["true", "1", "yes"])]
                elif op == "is False":
                    df = df[~df[col].astype(str).str.lower().isin(
                        ["true", "1", "yes"])]
                elif op in ("=", ">", "<", ">=", "<="):
                    nv = float(val)
                    series = pd.to_numeric(df[col], errors="coerce")
                    if   op == "=":  df = df[series == nv]
                    elif op == ">":  df = df[series > nv]
                    elif op == "<":  df = df[series < nv]
                    elif op == ">=": df = df[series >= nv]
                    elif op == "<=": df = df[series <= nv]
                elif op == "between":
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        if val:
                            df = df[df[col] >= pd.Timestamp(val, tz="UTC")]
                        if val2:
                            df = df[df[col] <= pd.Timestamp(val2, tz="UTC")]
                    else:
                        lo, hi = float(val), float(val2)
                        series = pd.to_numeric(df[col], errors="coerce")
                        df = df[(series >= lo) & (series <= hi)]
                elif op == "after":
                    df = df[df[col] >= pd.Timestamp(val, tz="UTC")]
                elif op == "before":
                    df = df[df[col] <= pd.Timestamp(val, tz="UTC")]
            except Exception:
                continue

        return df

    def _get_sliced_df(self, cfg: dict | None = None) -> pd.DataFrame:
        """Apply filters, dedup, and sampling to produce the sliced DataFrame."""
        if cfg is None:
            cfg = self._current_config()

        # Ensure correct date column is active before copying
        date_col_name = cfg.get("date_column", "")
        if date_col_name and date_col_name != "—":
            self.app.switch_date_column(date_col_name)

        df = self.app.df.copy()
        if df.empty:
            return df

        # ── Column-specific filters ──────────────────────────────────
        col_filters = cfg.get("column_filters", [])
        if col_filters:
            df = self._apply_column_filters(df, col_filters)

        # ── Date range (from slider) ─────────────────────────────────
        if "_date" in df.columns:
            if cfg.get("date_from"):
                try:
                    df = df[df["_date"] >= pd.Timestamp(
                        cfg["date_from"], tz="UTC")]
                except Exception:
                    pass
            if cfg.get("date_to"):
                try:
                    df = df[df["_date"] <= pd.Timestamp(
                        cfg["date_to"], tz="UTC")]
                except Exception:
                    pass

        # ── Backward compat: old-style lang/platform filters ─────────
        if cfg.get("language") and cfg.get("language") != "All":
            lang_col = self.app.resolve_col("language", df)
            if lang_col:
                df = df[df[lang_col].astype(str) == cfg["language"]]
        if cfg.get("platform") and cfg.get("platform") != "All":
            plat_col = (self.app.resolve_col("platform", df) or
                        ("_source_dataset" if "_source_dataset" in df.columns
                         else None))
            if plat_col:
                df = df[df[plat_col].astype(str) == cfg["platform"]]

        # ── Keyword + Boolean (use selected text column) ─────────────
        text_col = cfg.get("text_column", "")
        if not text_col or text_col == "— select —" or text_col not in df.columns:
            text_col = self.app.resolve_col("content_text", df)

        kw = cfg.get("keyword", "")
        if kw and text_col:
            df = df[df[text_col].astype(str).str.contains(
                re.escape(kw), case=False, na=False)]

        bq = cfg.get("boolean", "")
        if bq and text_col:
            mask = apply_boolean_query(df[text_col], bq)
            df = df[mask]

        # ── Dedup ────────────────────────────────────────────────────
        if cfg.get("dedup") and text_col:
            df = df.drop_duplicates(subset=[text_col], keep="first")

        # ── Random sample ────────────────────────────────────────────
        if cfg.get("sample"):
            try:
                n = int(cfg.get("sample_n", 500))
                seed = int(cfg.get("seed", 42))
                if 0 < n < len(df):
                    df = df.sample(n=n, random_state=seed)
            except (ValueError, TypeError):
                pass

        return df

    # ── Preview ──────────────────────────────────────────────────────────────

    def _update_preview(self):
        """Apply filters and update the status bar, preview table, chunk info."""
        df = self._get_sliced_df()
        total = len(self.app.df)
        n = len(df)
        pct = f"{n / total * 100:.1f}%" if total else "--"
        self._status_label.configure(
            text=f"  {n:,} of {total:,} rows  ({pct})")
        self._live_count.configure(text="")

        # Ensure column checkboxes are populated
        if not self._col_vars:
            self._populate_columns()

        # Preview table
        self._refresh_preview_tree(df)

        # Chunk info
        if self._chunk_var.get():
            try:
                k = int(self._chunk_k_var.get())
                if k < 2:
                    raise ValueError
                chunk_size = math.ceil(n / k) if k > 0 else n
                overlap_info = ""
                if self._overlap_var.get():
                    try:
                        pct_val = int(self._overlap_pct_var.get())
                        n_overlap = max(1, int(n * pct_val / 100))
                        overlap_info = f" + {n_overlap} shared overlap rows each"
                    except (ValueError, TypeError):
                        pass
                self._chunk_info.configure(
                    text=f"{k} chunks of ~{chunk_size} rows{overlap_info}")
            except (ValueError, TypeError):
                self._chunk_info.configure(text="K must be an integer >= 2")
        else:
            self._chunk_info.configure(text="")

    def _refresh_preview_tree(self, df: pd.DataFrame):
        """Show the first 20 rows of the slice in a treeview."""
        for w in self._tree_frame.winfo_children():
            w.destroy()

        if df.empty:
            ctk.CTkLabel(self._tree_frame, text="No matching rows.",
                         text_color=C.MUTED, font=("Segoe UI", 11)).pack(pady=20)
            return

        # Determine columns to show
        sel_cols = self._selected_columns()
        show_cols = sel_cols[:15]  # cap for readability
        preview = df.head(20)

        container = tk.Frame(self._tree_frame, bg=C.CARD)
        container.pack(fill="x")

        tree = ttk.Treeview(container, columns=show_cols, show="headings",
                            height=min(len(preview), 20))
        _style_treeview(tree)

        for col in show_cols:
            display = col[:25]
            tree.heading(col, text=display)
            tree.column(col, width=130, minwidth=60)

        for _, row in preview.iterrows():
            vals = [str(row.get(c, ""))[:80] for c in show_cols]
            tree.insert("", "end", values=vals)

        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=hsb.set)
        tree.pack(fill="x")
        hsb.pack(fill="x")

        self._tree = tree

    # ── Export helpers ────────────────────────────────────────────────────────

    def _prepare_export_df(self, df: pd.DataFrame,
                           cfg: dict | None = None) -> pd.DataFrame:
        """Apply column selection, row ID, and codebook columns."""
        if cfg is None:
            cfg = self._current_config()

        # Column selection
        sel_cols = cfg.get("columns") or self._selected_columns()
        out = df[[c for c in sel_cols if c in df.columns]].copy()

        # Strip timezone for Excel compatibility
        for col in out.select_dtypes(include=["datetimetz"]).columns:
            out[col] = out[col].dt.tz_localize(None)

        # Row ID
        if cfg.get("row_id", False):
            out.insert(0, "id", range(1, len(out) + 1))

        # Codebook columns (empty)
        cb = cfg.get("codebook", "")
        if cb:
            for name in [s.strip() for s in cb.split(",") if s.strip()]:
                if name not in out.columns:
                    out[name] = ""

        return out

    def _write_file(self, df: pd.DataFrame, path: str, fmt: str) -> bool:
        """Write a DataFrame to disk in the given format."""
        if fmt == "excel":
            try:
                df.to_excel(path, index=False)
            except ImportError:
                messagebox.showerror("Error", "pip install openpyxl")
                return False
        elif fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", force_ascii=False, indent=2)
        return True

    def _export_single(self, fmt: str):
        """Export the current slice as a single file."""
        df = self._get_sliced_df()
        if df.empty:
            messagebox.showinfo("Export", "The slice is empty — nothing to export.")
            return

        out = self._prepare_export_df(df)

        ext_map = {"excel": ".xlsx", "csv": ".csv", "json": ".json"}
        ft_map  = {"excel": [("Excel", "*.xlsx")], "csv": [("CSV", "*.csv")],
                   "json": [("JSON", "*.json")]}

        path = filedialog.asksaveasfilename(
            defaultextension=ext_map[fmt], filetypes=ft_map[fmt])
        if not path:
            return
        if self._write_file(out, path, fmt):
            messagebox.showinfo("Exported", f"Saved {len(out):,} rows to:\n{path}")

    def _export_chunks(self):
        """Open a dialog to name each chunk and pick its format, then export."""
        if not self._chunk_var.get():
            messagebox.showinfo(
                "Chunks",
                "Enable 'Split into equal chunks' in the Sampling section first.")
            return

        try:
            k = int(self._chunk_k_var.get())
            if k < 2:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror("Error", "K must be an integer >= 2.")
            return

        df = self._get_sliced_df()
        if df.empty:
            messagebox.showinfo("Export", "The slice is empty.")
            return

        cfg = self._current_config()

        try:
            seed = int(cfg.get("seed", 42))
        except (ValueError, TypeError):
            seed = 42

        # Shuffle for random distribution
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Overlap
        overlap_df = pd.DataFrame()
        main_df = df
        use_overlap = cfg.get("overlap", False)
        if use_overlap:
            try:
                pct = int(cfg.get("overlap_pct", 10))
                pct = max(1, min(pct, 50))
                n_overlap = max(1, int(len(df) * pct / 100))
                overlap_df = df.iloc[:n_overlap]
                main_df = df.iloc[n_overlap:]
            except (ValueError, TypeError):
                pass

        # Split into K chunks
        chunks = []
        chunk_size = math.ceil(len(main_df) / k) if k > 0 else len(main_df)
        for i in range(k):
            start = i * chunk_size
            end = min(start + chunk_size, len(main_df))
            chunk = main_df.iloc[start:end]
            if not overlap_df.empty:
                chunk = pd.concat([overlap_df, chunk], ignore_index=True)
            chunks.append(chunk)

        # ── Open dialog with a row per chunk ─────────────────────────
        name_base = self._slice_name_var.get().strip() or "slice"

        win = ctk.CTkToplevel(self)
        win.title("Export Chunks")
        win.geometry("560x" + str(min(180 + k * 42, 600)))
        win.configure(fg_color=C.BG)
        win.grab_set()
        win.focus_force()

        ctk.CTkLabel(win, text="Name and format for each chunk",
                     font=("Segoe UI", 14, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=16,
                                             pady=(12, 4))
        overlap_note = ""
        if use_overlap and not overlap_df.empty:
            overlap_note = (f"  ({len(overlap_df)} shared overlap rows "
                            "included in every chunk)")
        ctk.CTkLabel(win,
                     text=f"{k} chunks, ~{chunk_size} rows each.{overlap_note}",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=16, pady=(0, 8))

        sf = ctk.CTkScrollableFrame(win, fg_color=C.PANEL, height=min(k * 42, 340))
        sf.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # Header row
        hdr = ctk.CTkFrame(sf, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(hdr, text="#", width=30, text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold")).pack(side="left")
        ctk.CTkLabel(hdr, text="Rows", width=50, text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        ctk.CTkLabel(hdr, text="File name", width=250, text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        ctk.CTkLabel(hdr, text="Format", width=100, text_color=C.MUTED,
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)

        chunk_rows = []   # list of (name_var, fmt_var)
        for i, chunk in enumerate(chunks, 1):
            row = ctk.CTkFrame(sf, fg_color=C.CARD, corner_radius=4)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=str(i), width=30, text_color=C.TEXT,
                         font=("Segoe UI", 10)).pack(side="left", padx=(4, 0))
            ctk.CTkLabel(row, text=f"{len(chunk):,}", width=50,
                         text_color=C.MUTED,
                         font=("Segoe UI", 10)).pack(side="left", padx=4)
            nv = tk.StringVar(value=f"{name_base}_chunk{i}")
            ctk.CTkEntry(row, textvariable=nv, width=250,
                         height=26).pack(side="left", padx=4)
            fv = tk.StringVar(value="Excel (.xlsx)")
            ctk.CTkOptionMenu(row, variable=fv, width=120, height=26,
                              values=["Excel (.xlsx)", "CSV (.csv)",
                                      "JSON (.json)"]
                              ).pack(side="left", padx=4)
            chunk_rows.append((nv, fv))

        # Folder selector + export button
        ftr = ctk.CTkFrame(win, fg_color="transparent")
        ftr.pack(fill="x", padx=16, pady=(0, 12))

        folder_var = tk.StringVar()
        folder_lbl = ctk.CTkLabel(ftr, text="No folder selected",
                                  text_color=C.MUTED, font=("Segoe UI", 10))
        folder_lbl.pack(side="left", padx=(0, 8))

        def pick_folder():
            f = filedialog.askdirectory(title="Choose export folder")
            if f:
                folder_var.set(f)
                folder_lbl.configure(
                    text=f if len(f) < 40 else "…" + f[-37:],
                    text_color=C.TEXT)

        ctk.CTkButton(ftr, text="Choose Folder…", fg_color=C.BTN,
                      width=110, command=pick_folder).pack(side="left", padx=4)

        def do_export():
            folder = folder_var.get()
            if not folder:
                messagebox.showinfo("Export", "Select a folder first.",
                                    parent=win)
                return
            fmt_map = {"Excel (.xlsx)": ("excel", ".xlsx"),
                       "CSV (.csv)":    ("csv",   ".csv"),
                       "JSON (.json)":  ("json",  ".json")}
            exported = 0
            for i, (chunk, (nv, fv)) in enumerate(zip(chunks, chunk_rows)):
                name = nv.get().strip() or f"chunk{i+1}"
                fmt_key, ext = fmt_map.get(fv.get(), ("excel", ".xlsx"))
                out = self._prepare_export_df(chunk, cfg)
                path = os.path.join(folder, name + ext)
                if self._write_file(out, path, fmt_key):
                    exported += 1
            win.destroy()
            messagebox.showinfo(
                "Chunks Exported",
                f"Exported {exported} chunk(s) to:\n{folder}")

        ctk.CTkButton(ftr, text="Export All", fg_color=C.SUCCESS,
                      width=100, command=do_export).pack(side="right")

    # ── Saved slices ─────────────────────────────────────────────────────────

    def _save(self):
        name = (self._slice_name_var.get().strip()
                or f"Slice {len(self.app.slices) + 1}")
        cfg = self._current_config()
        # Store selected columns by name
        cfg["columns"] = self._selected_columns()
        self.app.slices[name] = cfg
        self._slice_name_var.set("")
        self._slice_notes_var.set("")
        self._refresh_slices()

    def _refresh_slices(self):
        for w in self._slices_frame.winfo_children():
            w.destroy()

        self._populate_columns()
        self._refresh_filter_options()

        # Update status if dataset available
        if not self.app.df.empty:
            total = len(self.app.df)
            self._status_label.configure(
                text=f"  {total:,} rows available")

        if not self.app.slices:
            ctk.CTkLabel(self._slices_frame,
                         text="No saved slices yet. Configure filters above "
                              "and click 'Save Slice'.",
                         text_color=C.MUTED, font=("Segoe UI", 11)).pack(pady=12)
            return

        for name, cfg in list(self.app.slices.items()):
            card = ctk.CTkFrame(self._slices_frame, fg_color=C.CARD,
                                corner_radius=6)
            card.pack(fill="x", pady=3, padx=2)

            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(6, 2))
            ctk.CTkLabel(top, text=name, font=("Segoe UI", 12, "bold"),
                         text_color=C.TEXT).pack(side="left")

            # Summary line
            parts = []
            for f in cfg.get("column_filters", []):
                val_str = f.get("value", "")[:20]
                parts.append(f"{f.get('column', '?')} {f.get('operation', '?')}"
                             + (f" {val_str}" if val_str else ""))
            # Backward compat
            if cfg.get("date_from") or cfg.get("date_to"):
                parts.append(
                    f"date: {cfg.get('date_from', '…')} → {cfg.get('date_to', '…')}")
            if cfg.get("language") and cfg["language"] != "All":
                parts.append(f"lang: {cfg['language']}")
            if cfg.get("platform") and cfg["platform"] != "All":
                parts.append(f"platform: {cfg['platform']}")
            if cfg.get("keyword"):
                parts.append(f'kw: "{cfg["keyword"]}"')
            if cfg.get("boolean"):
                bq = cfg["boolean"]
                parts.append(f'bool: {bq[:40]}{"…" if len(bq) > 40 else ""}')
            if cfg.get("dedup"):
                parts.append("dedup")
            if cfg.get("sample"):
                parts.append(f"sample: {cfg.get('sample_n', '?')}")
            if cfg.get("chunk"):
                parts.append(f"chunks: {cfg.get('chunk_k', '?')}")
            summary = "  |  ".join(parts) if parts else "No filters"
            ctk.CTkLabel(top, text=summary, font=("Segoe UI", 10),
                         text_color=C.MUTED).pack(side="left", padx=12)

            # Notes
            notes = cfg.get("notes", "")
            if notes:
                ctk.CTkLabel(card, text=notes, font=("Segoe UI", 10),
                             text_color=C.MUTED).pack(anchor="w", padx=10,
                                                      pady=(0, 2))

            # Action buttons
            btn_row = ctk.CTkFrame(card, fg_color="transparent")
            btn_row.pack(fill="x", padx=10, pady=(0, 6))

            lb = ctk.CTkButton(btn_row, text="Load", width=65,
                               fg_color=C.ACCENT,
                               command=lambda c=cfg: self._load_slice(c))
            lb.pack(side="left", padx=(0, 4))
            tip(lb, "Load this slice's settings into the filter fields.")

            vb = ctk.CTkButton(btn_row, text="View", width=65,
                               fg_color=C.ACCENT,
                               command=lambda c=cfg: self._view_slice(c))
            vb.pack(side="left", padx=(0, 4))
            tip(vb, "Apply this slice and open it in the Table view.")

            db = ctk.CTkButton(btn_row, text="Delete", width=65, fg_color=C.BTN,
                               command=lambda n=name: self._delete_slice(n))
            db.pack(side="left", padx=(0, 12))
            tip(db, "Remove this saved slice.")

            ctk.CTkLabel(btn_row, text="Export:", text_color=C.MUTED,
                         font=("Segoe UI", 10)).pack(side="left", padx=(0, 4))
            for label, fmt in [("Excel", "excel"), ("CSV", "csv"),
                               ("JSON", "json")]:
                eb = ctk.CTkButton(
                    btn_row, text=label, width=55, height=26,
                    fg_color=C.BTN, font=("Segoe UI", 10),
                    command=lambda c=cfg, f=fmt: self._export_saved(c, f))
                eb.pack(side="left", padx=2)

    def _load_slice(self, cfg: dict):
        """Load a saved slice's settings back into the filter fields."""
        # Clear existing filter rows
        for row in self._filter_rows[:]:
            self._remove_filter_row(row)

        # Restore column filters
        for filt_cfg in cfg.get("column_filters", []):
            self._add_filter_row()
            self._filter_rows[-1].set_config(filt_cfg)

        # Backward compat: migrate old date/lang/platform to column filters
        if "column_filters" not in cfg:
            self._migrate_old_filters(cfg)

        # Restore date column + slider
        dc = cfg.get("date_column", "")
        if dc:
            self._date_col_var.set(dc)
            self._on_date_col_change(dc)
        self._df_var.set(cfg.get("date_from", ""))
        self._dt_var.set(cfg.get("date_to", ""))
        # Restore text column
        tc = cfg.get("text_column", "")
        if tc:
            self._text_col_var.set(tc)

        self._kw_var.set(cfg.get("keyword", ""))
        self._bool_var.set(cfg.get("boolean", ""))
        self._dedup_var.set(cfg.get("dedup", False))
        self._sample_var.set(cfg.get("sample", False))
        self._sample_n_var.set(str(cfg.get("sample_n", "500")))
        self._seed_var.set(str(cfg.get("seed", "42")))
        self._chunk_var.set(cfg.get("chunk", False))
        self._chunk_k_var.set(str(cfg.get("chunk_k", "2")))
        self._overlap_var.set(cfg.get("overlap", False))
        self._overlap_pct_var.set(str(cfg.get("overlap_pct", "10")))
        self._codebook_var.set(cfg.get("codebook", ""))
        self._row_id_var.set(cfg.get("row_id", True))
        self._slice_notes_var.set(cfg.get("notes", ""))

        # Restore column selection if saved
        saved_cols = cfg.get("columns", [])
        if saved_cols:
            for col, var in self._col_vars.items():
                var.set(col in saved_cols)

        self._update_preview()

    def _migrate_old_filters(self, cfg: dict):
        """Convert old-format date/lang/platform to column filter rows."""
        if cfg.get("date_from"):
            self._add_filter_row()
            self._filter_rows[-1].set_config({
                "column": "_date", "operation": "after",
                "value": cfg["date_from"]})
        if cfg.get("date_to"):
            self._add_filter_row()
            self._filter_rows[-1].set_config({
                "column": "_date", "operation": "before",
                "value": cfg["date_to"]})
        if cfg.get("language") and cfg["language"] != "All":
            lang_col = self.app.resolve_col("language", self.app.df) or "language"
            self._add_filter_row()
            self._filter_rows[-1].set_config({
                "column": lang_col, "operation": "equals",
                "value": cfg["language"]})
        if cfg.get("platform") and cfg["platform"] != "All":
            plat_col = (self.app.resolve_col("platform", self.app.df) or
                        ("_source_dataset" if "_source_dataset" in
                         self.app.df.columns else "platform"))
            self._add_filter_row()
            self._filter_rows[-1].set_config({
                "column": plat_col, "operation": "equals",
                "value": cfg["platform"]})

    def _view_slice(self, cfg: dict):
        """Apply slice and jump to Table view."""
        self.app.filtered_df = self._get_sliced_df(cfg)
        self.app._show("table")

    def _delete_slice(self, name: str):
        if not messagebox.askyesno("Delete Slice",
                                   f"Delete saved slice '{name}'?",
                                   parent=self):
            return
        self.app.slices.pop(name, None)
        self._refresh_slices()

    def _export_saved(self, cfg: dict, fmt: str):
        """Export a saved slice."""
        df = self._get_sliced_df(cfg)
        if df.empty:
            messagebox.showinfo("Export", "The slice is empty.")
            return
        out = self._prepare_export_df(df, cfg)

        ext_map = {"excel": ".xlsx", "csv": ".csv", "json": ".json"}
        ft_map  = {"excel": [("Excel", "*.xlsx")], "csv": [("CSV", "*.csv")],
                   "json": [("JSON", "*.json")]}

        path = filedialog.asksaveasfilename(
            defaultextension=ext_map[fmt], filetypes=ft_map[fmt])
        if not path:
            return
        if self._write_file(out, path, fmt):
            messagebox.showinfo("Exported", f"Saved {len(out):,} rows to:\n{path}")

    # ── Theme rebuild ────────────────────────────────────────────────────────

    def rebuild(self):
        # Save current UI state
        saved_filters = [r.get_config() for r in self._filter_rows]
        saved = {}
        for attr in ("_df_var", "_dt_var", "_date_col_var", "_text_col_var",
                     "_kw_var", "_bool_var", "_dedup_var", "_sample_var",
                     "_sample_n_var", "_seed_var", "_chunk_var", "_chunk_k_var",
                     "_overlap_var", "_overlap_pct_var", "_row_id_var",
                     "_codebook_var", "_slice_name_var", "_slice_notes_var"):
            saved[attr] = getattr(self, attr).get()

        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._init_vars()
        self._build()

        # Restore state
        for attr, val in saved.items():
            getattr(self, attr).set(val)

        # Restore filter rows
        for filt_cfg in saved_filters:
            self._add_filter_row()
            self._filter_rows[-1].set_config(filt_cfg)
