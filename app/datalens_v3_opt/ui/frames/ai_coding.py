"""AI Coding frame — lightweight NLI-based zero-shot column classification.

Uses Natural Language Inference (NLI) models via the transformers
zero-shot-classification pipeline.  NLI models are purpose-built for this exact
task and are 5–20× smaller than generative LLMs while matching or exceeding
their accuracy on classification tasks.

Three model choices:
  • DeBERTa-v3-small   ~90 MB  /  <300 MB RAM  — fastest, English
  • mDeBERTa-v3-base  ~560 MB  /  ~800 MB RAM  — multilingual (100+ languages)
  • DeBERTa-v3-large  ~900 MB  /  ~1.5 GB RAM  — highest accuracy, English

Each row is processed individually with a small CPU yield between rows so the
machine stays responsive throughout a long coding run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import threading
import time
import gc
import re
import os
import pathlib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip, _style_treeview

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App

# ── Local model cache (~/.datalens/models/) ──────────────────────────────────
# Models are downloaded once and stored here — no internet needed after that.
import datalens_v3_opt.constants as C
MODEL_CACHE_DIR = C.get_model_cache()

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS = {
    "MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli": {
        "label": "DeBERTa-small  ·  Fast & Light",
        "info":  "~90 MB download  ·  <300 MB RAM  ·  English  ·  Fastest option",
    },
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli": {
        "label": "mDeBERTa-base  ·  Multilingual",
        "info":  "~560 MB download  ·  ~800 MB RAM  ·  100+ languages  ·  Best for multilingual data",
    },
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": {
        "label": "DeBERTa-large  ·  High Accuracy",
        "info":  "~900 MB download  ·  ~1.5 GB RAM  ·  English  ·  Highest classification accuracy",
    },
}

_ID_BY_LABEL   = {v["label"]: k for k, v in MODELS.items()}
_MODEL_LABELS  = [v["label"] for v in MODELS.values()]
_DEFAULT_LABEL = _MODEL_LABELS[1]   # multilingual as default

# Runtime state (thread-safe via _model_lock)
_classifier    = None
_active_id     = None
_model_error   = None
_model_ready   = threading.Event()
_model_loading = False
_model_lock    = threading.Lock()


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(model_id: str, status_cb=None):
    global _classifier, _active_id, _model_error, _model_loading
    with _model_lock:
        _model_loading = True
        _model_ready.clear()

        # Unload previous model
        if _classifier is not None:
            del _classifier
            _classifier = None
            gc.collect()

    try:
        from transformers import pipeline  # type: ignore
        label = MODELS[model_id]["label"]
        if status_cb:
            status_cb(f"⟳ Downloading {label} (first run only)…")
        clf = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=-1,       # CPU
            model_kwargs={"cache_dir": str(MODEL_CACHE_DIR)},
        )
        with _model_lock:
            _classifier  = clf
            _active_id   = model_id
            _model_error = None
        if status_cb:
            status_cb(f"● {label}")
    except ImportError:
        with _model_lock:
            _model_error = "transformers is not installed.\nRun: pip install transformers torch"
        if status_cb:
            status_cb("✕ transformers not installed")
    except Exception as exc:
        with _model_lock:
            _model_error = str(exc)
        if status_cb:
            status_cb(f"✕ {str(exc)[:70]}")
    finally:
        with _model_lock:
            _model_loading = False
        _model_ready.set()


# ── Classification helpers ─────────────────────────────────────────────────────

def _build_hypotheses(criterion: str):
    """
    Convert a user criterion into NLI hypothesis strings and return mode.

    Returns (hypotheses: list[str], display_labels: list[str], mode: str)
      mode = 'binary'   → hypotheses[0] = Yes hypothesis, [1] = No hypothesis
      mode = 'category' → hypotheses[i] = sentence for each category option

    NLI works by measuring entailment between the text (premise) and each
    hypothesis sentence. We construct grammatically natural hypotheses so the
    model can score them reliably.

    Examples
    --------
    "mentions climate change"
        → ["This text mentions climate change.",
           "This text does not mention climate change."]   binary

    "Is it a question or a statement"
        → ["This text is a question.", "This text is a statement."]   category

    "positive or negative sentiment"
        → ["This text has positive sentiment.",
           "This text has negative sentiment."]   category
    """
    c    = criterion.strip().rstrip("?").strip()
    body = re.sub(
        r"^(is it|is this|does it|does this|can it|is the text|does the text)\s+",
        "", c, flags=re.IGNORECASE,
    ).strip()

    has_q = bool(re.match(
        r"^(is it|is this|does it|does this|can it|is the text|does the text)\b",
        c, flags=re.IGNORECASE,
    ))

    # ── Category: question form + "X or Y" ───────────────────────────────────
    if has_q:
        parts = re.split(r"\s+or\s+", body, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            labels = [p.strip() for p in parts]
            hyps   = [f"This text is {l}." for l in labels]
            return hyps, labels, "category"

    # ── Category: short bare "X or Y" (e.g. "positive or negative") ──────────
    parts = re.split(r"\s+or\s+", c, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2 and all(len(p.split()) <= 5 for p in parts):
        labels = [p.strip() for p in parts]
        hyps   = [f"This text expresses or represents {l}." for l in labels]
        return hyps, labels, "category"

    # ── Binary: build a positive + negated hypothesis ─────────────────────────
    pos_hyp = _positive_hyp(body)
    neg_hyp = _negative_hyp(body)
    return [pos_hyp, neg_hyp], ["Yes", "No"], "binary"


def _positive_hyp(body: str) -> str:
    """Wrap the criterion body in a natural positive hypothesis sentence."""
    # Already a full sentence?
    if re.match(r".+\.$", body):
        return body
    return f"This text {body}."


def _negative_hyp(body: str) -> str:
    """Produce a grammatical negation of the criterion body."""
    # "is/are/was/were …"  →  "is not …"
    m = re.match(r"^(is|are|was|were)\b", body, re.IGNORECASE)
    if m:
        return f"This text {m.group()} not {body[m.end():].strip()}."
    # "has/have/had …"  →  "has not …"
    m = re.match(r"^(has|have|had)\b", body, re.IGNORECASE)
    if m:
        return f"This text {m.group()} not {body[m.end():].strip()}."
    # Default: "does not <body>"
    return f"This text does not {body}."


def _classify_one(text: str, criterion: str) -> str:
    """Classify a single text string.  Returns the winning display label."""
    hypotheses, display_labels, mode = _build_hypotheses(criterion)

    result = _classifier(
        str(text)[:512],
        candidate_labels=hypotheses,
        hypothesis_template="{}",   # labels ARE the full hypothesis sentences
        multi_label=False,
    )

    # The top-ranked hypothesis is the model's prediction
    top_hyp = result["labels"][0]
    idx     = hypotheses.index(top_hyp)
    return display_labels[idx]


# ── AICodingFrame ──────────────────────────────────────────────────────────────

class AICodingFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app

        self._cancel_event  = threading.Event()
        self._coding_active = False
        self._last_col_name = None
        self._progress_win  = None
        self._test_win      = None

        self._build()
        self._trigger_load(_DEFAULT_LABEL)

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self):
        # Title bar
        tb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=48)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="AI Coding", font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(side="left", padx=16)
        ctk.CTkLabel(tb,
                     text="Classify any column with a plain-English description",
                     text_color=C.MUTED, font=("Segoe UI", 11)).pack(side="left", padx=4)
        self._status_var = tk.StringVar(value="⟳ Loading model…")
        ctk.CTkLabel(tb, textvariable=self._status_var,
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(side="right", padx=12)

        # Data preview
        pf = ctk.CTkFrame(self, fg_color=C.CARD, corner_radius=0, height=240)
        pf.pack(fill="x")
        pf.pack_propagate(False)
        self._tv = ttk.Treeview(pf, show="headings", selectmode="browse", height=9)
        _style_treeview(self._tv)
        ysb = ttk.Scrollbar(pf, orient="vertical",   command=self._tv.yview)
        xsb = ttk.Scrollbar(pf, orient="horizontal", command=self._tv.xview)
        self._tv.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self._tv.pack(fill="both", expand=True)

        # Config panel
        cfg   = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0)
        cfg.pack(fill="x")
        inner = ctk.CTkFrame(cfg, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=10)

        # Row 0 — Model selector
        r0 = ctk.CTkFrame(inner, fg_color="transparent")
        r0.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(r0, text="Model:", text_color=C.MUTED,
                     font=("Segoe UI", 12), width=140).pack(side="left")
        self._model_var = tk.StringVar(value=_DEFAULT_LABEL)
        self._model_om  = ctk.CTkOptionMenu(
            r0, variable=self._model_var, values=_MODEL_LABELS,
            width=310, command=self._on_model_change)
        self._model_om.pack(side="left", padx=(0, 8))
        tip(self._model_om,
            "DeBERTa-small  — 90 MB, fastest, English only\n"
            "mDeBERTa-base  — 560 MB, multilingual (100+ languages)\n"
            "DeBERTa-large  — 900 MB, highest accuracy, English only\n\n"
            "All models use NLI (Natural Language Inference) — purpose-built\n"
            "for classification, much lighter than generative LLMs.")
        self._load_btn = ctk.CTkButton(
            r0, text="Load Model", fg_color=C.BTN, width=110,
            command=lambda: self._trigger_load(self._model_var.get()))
        self._load_btn.pack(side="left", padx=(0, 10))
        tip(self._load_btn, "Download (first time only) and load the selected model.")
        self._info_var = tk.StringVar(
            value=MODELS[_ID_BY_LABEL[_DEFAULT_LABEL]]["info"])
        ctk.CTkLabel(r0, textvariable=self._info_var,
                     text_color=C.MUTED, font=("Segoe UI", 9)
                     ).pack(side="left", padx=4)

        # Row 1 — Column selector
        r1 = ctk.CTkFrame(inner, fg_color="transparent")
        r1.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(r1, text="Column to code:", text_color=C.MUTED,
                     font=("Segoe UI", 12), width=140).pack(side="left")
        self._col_var = tk.StringVar(value="")
        self._col_om  = ctk.CTkOptionMenu(r1, variable=self._col_var,
                                          values=["— load a dataset first —"],
                                          width=260)
        self._col_om.pack(side="left", padx=(0, 12))
        tip(self._col_om, "The column the model will read and classify.")

        # Row 2 — Criterion
        r2 = ctk.CTkFrame(inner, fg_color="transparent")
        r2.pack(fill="x", pady=(0, 8))
        lbl = ctk.CTkLabel(r2, text="What to look for:", text_color=C.MUTED,
                           font=("Segoe UI", 12), width=140)
        lbl.pack(side="left")
        tip(lbl,
            "Describe what to classify in plain English.\n\n"
            "Yes / No output:\n"
            "  mentions climate change\n"
            "  contains a URL or hashtag\n"
            "  expresses anger or frustration\n"
            "  is written in formal language\n\n"
            "Category output  (X or Y):\n"
            "  Is it a question or a statement\n"
            "  positive or negative sentiment\n"
            "  formal or informal language")
        self._criterion_var = tk.StringVar()
        entry = ctk.CTkEntry(
            r2, textvariable=self._criterion_var, width=540,
            placeholder_text=(
                "e.g.  mentions computer programs   |   "
                "Is it a question or a statement"))
        entry.pack(side="left")
        tip(entry,
            "Describe a topic / property → Yes / No column\n"
            "Use 'X or Y' phrasing → category column")

        # Row 3 — Buttons + progress
        r3 = ctk.CTkFrame(inner, fg_color="transparent")
        r3.pack(fill="x")

        self._test_btn = ctk.CTkButton(
            r3, text="Run Test  (10 rows)", fg_color=C.BTN, width=160,
            command=self._run_test)
        self._test_btn.pack(side="left", padx=(0, 8))
        tip(self._test_btn,
            "Classify the first 10 rows and show results in a popup window.\n"
            "Verify the criterion is correct before running on all rows.")

        self._code_btn = ctk.CTkButton(
            r3, text="Code Rows", fg_color=C.ACCENT, width=120,
            command=self._run_full)
        self._code_btn.pack(side="left", padx=(0, 12))
        tip(self._code_btn,
            "Classify every row one by one and add a new column to the dataset.")

        self._cancel_btn = ctk.CTkButton(
            r3, text="Cancel", fg_color=C.BTN, width=80,
            state="disabled", command=self._cancel_coding)
        self._cancel_btn.pack(side="left", padx=(0, 10))

        self._progress_var = tk.DoubleVar(value=0)
        prog = ctk.CTkProgressBar(r3, variable=self._progress_var,
                                  width=200, height=12)
        prog.pack(side="left", padx=(0, 6))
        prog.set(0)

        self._prog_lbl = ctk.CTkLabel(r3, text="", text_color=C.MUTED,
                                      font=("Segoe UI", 10))
        self._prog_lbl.pack(side="left")

        # Export bar (shown after coding completes)
        self._export_bar = ctk.CTkFrame(self, fg_color=C.CARD,
                                        corner_radius=0, height=44)
        ei = ctk.CTkFrame(self._export_bar, fg_color="transparent")
        ei.pack(side="left", padx=12, pady=8)
        ctk.CTkLabel(ei, text="Export results:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(0, 10))
        ctk.CTkButton(ei, text="CSV", fg_color=C.ACCENT, width=90,
                      command=lambda: self._export("csv")).pack(side="left", padx=(0, 6))
        ctk.CTkButton(ei, text="Excel (.xlsx)", fg_color=C.ACCENT, width=120,
                      command=lambda: self._export("excel")).pack(side="left")
        self._export_info = ctk.CTkLabel(self._export_bar, text="",
                                         text_color=C.MUTED, font=("Segoe UI", 10))
        self._export_info.pack(side="left", padx=16)

        # ML install button (shown if packages are missing)
        from datalens_v3_opt.ui.ml_installer import MLInstallButton, ml_installed
        if not ml_installed():
            MLInstallButton(self).pack(fill="x", padx=12, pady=8)

    # ── Model management ───────────────────────────────────────────────────────

    def _on_model_change(self, label: str):
        mid = _ID_BY_LABEL.get(label)
        if mid:
            self._info_var.set(MODELS[mid]["info"])

    def _trigger_load(self, label: str):
        mid = _ID_BY_LABEL.get(label)
        if not mid:
            return
        if _active_id == mid and _classifier is not None:
            return
        self._set_status(f"⟳ Loading {label}…")
        self._load_btn.configure(state="disabled")
        threading.Thread(target=_load_model,
                         args=(mid, self._set_status), daemon=True).start()
        self._poll_ready()

    def _poll_ready(self):
        if _model_ready.is_set():
            self._load_btn.configure(state="normal")
            if _model_error:
                self._set_status(f"✕ {_model_error.splitlines()[0]}")
            elif _classifier:
                label = MODELS.get(_active_id, {}).get("label", "")
                self._set_status(f"● {label}")
        else:
            self.after(400, self._poll_ready)

    def _set_status(self, msg: str):
        self.after(0, lambda m=msg: self._status_var.set(m))

    # ── Refresh ────────────────────────────────────────────────────────────────

    def refresh(self):
        df = self.app.df
        self._tv.delete(*self._tv.get_children())

        if df.empty:
            self._tv.configure(columns=["info"])
            self._tv.heading("info", text="No dataset loaded — use Import first")
            self._col_om.configure(values=["— load a dataset first —"])
            return

        cols = [c for c in df.columns if not c.startswith("_")]
        self._col_om.configure(values=cols)
        if not self._col_var.get() or self._col_var.get() not in cols:
            text_col = self.app.resolve_col("content_text", df)
            self._col_var.set(text_col or cols[0])

        ai_cols    = [c for c in df.columns if c.startswith("ai_")]
        other_cols = [c for c in cols if not c.startswith("ai_")]
        display    = (ai_cols + other_cols)[:18]

        self._tv.configure(columns=display)
        for col in display:
            self._tv.heading(col, text=col)
            self._tv.column(col, width=80 if col.startswith("ai_") else 140, minwidth=50)

        for row in df.head(300)[display].values:
            vals = [str(v)[:100] if pd.notna(v) else "" for v in row]
            self._tv.insert("", "end", values=vals)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self):
        if not _classifier:
            msg = _model_error or "The model is still loading — please wait."
            messagebox.showerror("Model not ready", msg)
            return None, None
        if self.app.df.empty:
            messagebox.showinfo("No data", "Please import a dataset first.")
            return None, None
        col = self._col_var.get()
        if not col or col not in self.app.df.columns:
            messagebox.showinfo("No column", "Please select a column to code.")
            return None, None
        criterion = self._criterion_var.get().strip()
        if not criterion:
            messagebox.showinfo("No criterion",
                                "Please describe what you are looking for.")
            return None, None
        return col, criterion

    # ── Run Test ──────────────────────────────────────────────────────────────

    def _run_test(self):
        col, criterion = self._validate()
        if not col:
            return
        self._test_btn.configure(state="disabled")
        self._code_btn.configure(state="disabled")
        rows = self.app.df.head(10)
        self._open_test_progress_window(len(rows))
        threading.Thread(target=self._test_thread,
                         args=(rows, col, criterion), daemon=True).start()

    def _open_test_progress_window(self, total: int):
        win = ctk.CTkToplevel(self)
        win.title("AI Coding — Test Run")
        win.geometry("400x160")
        win.resizable(False, False)
        win.configure(fg_color=C.PANEL)
        win.wm_attributes("-topmost", True)
        win.protocol("WM_DELETE_WINDOW", lambda: None)

        ctk.CTkLabel(win, text="Running test on first 10 rows…",
                     text_color=C.TEXT, font=("Segoe UI", 12, "bold")
                     ).pack(pady=(20, 8))

        self._tp_count_var = tk.StringVar(value=f"0 / {total} rows")
        ctk.CTkLabel(win, textvariable=self._tp_count_var,
                     text_color=C.TEXT, font=("Segoe UI", 18, "bold")
                     ).pack(pady=(0, 8))

        self._tp_prog_var = tk.DoubleVar(value=0)
        bar = ctk.CTkProgressBar(win, variable=self._tp_prog_var, width=340, height=12)
        bar.pack(pady=(0, 12))
        bar.set(0)

        self._test_win = win

    def _close_test_progress_window(self):
        if hasattr(self, "_test_win") and self._test_win:
            try:
                self._test_win.destroy()
            except Exception:
                pass
            self._test_win = None

    def _test_thread(self, rows, col: str, criterion: str):
        total   = len(rows)
        results = []
        texts = rows[col].values if col in rows.columns else [""] * total
        for i, text in enumerate(texts):
            text = str(text) if pd.notna(text) else ""
            try:
                results.append(_classify_one(text, criterion))
            except Exception as exc:
                results.append(f"ERR: {str(exc)[:40]}")
            time.sleep(0.01)
            pct = (i + 1) / total
            self.after(0, lambda p=pct, d=i+1, t=total: (
                self._tp_prog_var.set(p) if hasattr(self, "_tp_prog_var") else None,
                self._tp_count_var.set(f"{d} / {t} rows") if hasattr(self, "_tp_count_var") else None,
            ))

        _, display_labels, mode = _build_hypotheses(criterion)
        self.after(0, lambda: self._finish_test(rows, col, criterion,
                                                results, display_labels, mode))

    def _finish_test(self, rows, col, criterion, results, display_labels, mode):
        self._close_test_progress_window()
        self._test_btn.configure(state="normal")
        self._code_btn.configure(state="normal")
        self._show_test_window(rows, col, criterion, results, display_labels, mode)

    def _show_test_window(self, rows, col, criterion, results,
                          display_labels, mode):
        win = ctk.CTkToplevel(self)
        win.title("Test Results — AI Coding (10 rows)")
        win.geometry("980x520")
        win.configure(fg_color=C.BG)
        win.grab_set()

        model_label = MODELS.get(_active_id, {}).get("label", "")
        result_hdr  = (
            "AI Result  (Yes / No)" if mode == "binary"
            else "AI Category  (" + " / ".join(display_labels) + ")"
        )

        ctk.CTkLabel(
            win,
            text=f"Model: {model_label}   |   Column: '{col}'   |   Criterion: {criterion}",
            text_color=C.MUTED, font=("Segoe UI", 11),
        ).pack(anchor="w", padx=16, pady=(12, 4))

        tf = ctk.CTkFrame(win, fg_color=C.CARD, corner_radius=0)
        tf.pack(fill="both", expand=True, padx=12, pady=8)
        tv = ttk.Treeview(tf, show="headings", columns=[col, "result"])
        _style_treeview(tv)
        ysb = ttk.Scrollbar(tf, orient="vertical",   command=tv.yview)
        xsb = ttk.Scrollbar(tf, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        tv.pack(fill="both", expand=True)
        tv.heading(col,      text=col)
        tv.heading("result", text=result_hdr)
        tv.column(col,      width=680, minwidth=200)
        tv.column("result", width=160, minwidth=80)

        texts = rows[col].values if col in rows.columns else [""] * len(results)
        for text, label in zip(texts, results):
            text = str(text)[:400] if pd.notna(text) else ""
            tv.insert("", "end", values=(text, label))

        ctk.CTkButton(win, text="Close", fg_color=C.BTN, width=100,
                      command=win.destroy).pack(pady=(0, 12))

    # ── Progress window ───────────────────────────────────────────────────────

    def _open_progress_window(self, total: int, col: str, criterion: str):
        """Create a floating progress window that stays on top during coding."""
        win = ctk.CTkToplevel(self)
        win.title("AI Coding — Running")
        win.geometry("480x200")
        win.resizable(False, False)
        win.configure(fg_color=C.PANEL)
        win.wm_attributes("-topmost", True)
        # Prevent closing via X button — use Cancel instead
        win.protocol("WM_DELETE_WINDOW", lambda: None)

        ctk.CTkLabel(
            win,
            text=f"Coding column:  {col}",
            text_color=C.TEXT, font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=20, pady=(18, 2))
        ctk.CTkLabel(
            win,
            text=f"Criterion:  {criterion[:60]}{'…' if len(criterion) > 60 else ''}",
            text_color=C.MUTED, font=("Segoe UI", 11),
        ).pack(anchor="w", padx=20, pady=(0, 12))

        self._pw_count_var = tk.StringVar(value=f"0 / {total:,} rows")
        ctk.CTkLabel(
            win,
            textvariable=self._pw_count_var,
            text_color=C.TEXT, font=("Segoe UI", 20, "bold"),
        ).pack(pady=(0, 6))

        self._pw_eta_var = tk.StringVar(value="Starting…")
        ctk.CTkLabel(
            win,
            textvariable=self._pw_eta_var,
            text_color=C.MUTED, font=("Segoe UI", 11),
        ).pack()

        self._pw_prog_var = tk.DoubleVar(value=0)
        bar = ctk.CTkProgressBar(win, variable=self._pw_prog_var,
                                 width=420, height=14)
        bar.pack(pady=(10, 8))
        bar.set(0)

        ctk.CTkButton(
            win, text="Cancel", fg_color=C.BTN, width=100,
            command=self._cancel_coding,
        ).pack(pady=(0, 12))

        self._progress_win = win

    def _close_progress_window(self):
        if hasattr(self, "_progress_win") and self._progress_win:
            try:
                self._progress_win.destroy()
            except Exception:
                pass
            self._progress_win = None

    # ── Code Rows ─────────────────────────────────────────────────────────────

    def _run_full(self):
        col, criterion = self._validate()
        if not col or self._coding_active:
            return
        self._coding_active = True
        self._cancel_event.clear()
        self._code_btn.configure(state="disabled", fg_color=C.BTN)
        self._test_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._progress_var.set(0)
        self._prog_lbl.configure(text="")
        self._export_bar.pack_forget()
        self._open_progress_window(len(self.app.df), col, criterion)
        threading.Thread(target=self._coding_thread,
                         args=(col, criterion), daemon=True).start()

    def _coding_thread(self, col: str, criterion: str):
        df    = self.app.df
        n     = len(df)
        texts = df[col].values if col in df.columns else [""] * n
        indices = df.index.tolist()
        res   = {}
        t0    = time.time()

        for i in range(n):
            if self._cancel_event.is_set():
                self.after(0, lambda: self._finish(res, col, criterion, True))
                return

            text = str(texts[i]) if pd.notna(texts[i]) else ""
            try:
                label = _classify_one(text, criterion)
            except Exception:
                label = "?"
            res[indices[i]] = label

            # Yield CPU so the machine stays responsive
            time.sleep(0.01)

            pct      = (i + 1) / n
            elapsed  = time.time() - t0
            rps      = (i + 1) / elapsed if elapsed > 0 else 0
            eta_s    = int((n - i - 1) / rps) if rps > 0 else 0
            eta_str  = f"ETA  {eta_s // 60}m {eta_s % 60:02d}s remaining" if eta_s > 5 else "Almost done…"

            self.after(0, lambda p=pct, d=i + 1, t=n, e=eta_str:
                       self._update_progress(p, d, t, e))

        self.after(0, lambda: self._finish(res, col, criterion, False))

    def _update_progress(self, pct, done, total, eta):
        self._progress_var.set(pct)
        self._prog_lbl.configure(text=f"{done:,} / {total:,}")
        # Update floating window
        if hasattr(self, "_pw_count_var"):
            self._pw_count_var.set(f"{done:,} / {total:,} rows  ({pct*100:.1f}%)")
        if hasattr(self, "_pw_eta_var"):
            self._pw_eta_var.set(eta)
        if hasattr(self, "_pw_prog_var"):
            self._pw_prog_var.set(pct)

    def _finish(self, results, col, criterion, cancelled):
        self._coding_active = False
        self._close_progress_window()
        self._cancel_btn.configure(state="disabled")
        self._code_btn.configure(state="normal", fg_color=C.ACCENT)
        self._test_btn.configure(state="normal")

        if cancelled:
            self._prog_lbl.configure(
                text=f"Cancelled — {len(results):,} rows coded")
            return

        base = _col_name_from(criterion)
        name = base
        i = 2
        while name in self.app.df.columns:
            name = f"{base}_{i}"; i += 1
        self._last_col_name = name

        for idx, label in results.items():
            self.app.df.at[idx, name] = label

        self.app.apply_filters()

        counts  = pd.Series(results.values()).value_counts()
        summary = "  ".join(f"{lbl}: {cnt:,}" for lbl, cnt in counts.items())
        self._progress_var.set(1.0)
        self._prog_lbl.configure(
            text=f"Done — column '{name}' added  |  {summary}")

        self.refresh()
        self._export_info.configure(text=f"Column '{name}'  |  {summary}")
        self._export_bar.pack(fill="x")

        try:
            self.app._frames["table"].refresh()
        except Exception:
            pass

    def _cancel_coding(self):
        self._cancel_event.set()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, fmt: str):
        df = self.app.df
        if df.empty:
            messagebox.showinfo("Export", "No data to export.")
            return
        out = df[[c for c in df.columns if not c.startswith("_")]].copy()
        for c in out.select_dtypes(include=["datetimetz"]).columns:
            out[c] = out[c].dt.tz_localize(None)

        if fmt == "csv":
            p = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
            if p:
                out.to_csv(p, index=False)
                messagebox.showinfo("Saved", f"Saved {len(out):,} rows to:\n{p}")
        elif fmt == "excel":
            p = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel", "*.xlsx")])
            if p:
                try:
                    out.to_excel(p, index=False)
                    messagebox.showinfo("Saved", f"Saved {len(out):,} rows to:\n{p}")
                except ImportError:
                    messagebox.showerror("Error", "pip install openpyxl")

    # ── Theme rebuild ──────────────────────────────────────────────────────────

    def rebuild(self):
        col       = self._col_var.get()       if hasattr(self, "_col_var")       else ""
        criterion = self._criterion_var.get() if hasattr(self, "_criterion_var") else ""
        model_lbl = self._model_var.get()     if hasattr(self, "_model_var")     else _DEFAULT_LABEL

        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._cancel_event  = threading.Event()
        self._coding_active = False
        self._progress_win  = None
        self._test_win      = None
        self._build()

        self._model_var.set(model_lbl)
        mid = _ID_BY_LABEL.get(model_lbl)
        if mid:
            self._info_var.set(MODELS[mid]["info"])
        if col and col in self.app.df.columns:
            self._col_var.set(col)
        if criterion:
            self._criterion_var.set(criterion)

        if _model_error:
            self._status_var.set(f"✕ {_model_error.splitlines()[0]}")
        elif _classifier:
            label = MODELS.get(_active_id, {}).get("label", "")
            self._status_var.set(f"● {label}")
        else:
            self._status_var.set("⟳ Loading model…")
            self._poll_ready()

        self.refresh()


# ── Utility ───────────────────────────────────────────────────────────────────

def _col_name_from(criterion: str) -> str:
    words = re.sub(r"[^\w\s]", "", criterion).split()[:4]
    slug  = "_".join(w.lower() for w in words if w)
    return f"ai_{slug}" if slug else "ai_result"
