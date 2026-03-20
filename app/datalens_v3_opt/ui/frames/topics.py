"""Topic Modelling frame — discover themes in your text data using BERTopic.

Uses sentence-transformer embeddings with UMAP dimensionality reduction and
HDBSCAN clustering to automatically find topics.  Runs entirely on CPU.

Includes a full preprocessing pipeline (language-aware stopwords, URL/mention
removal, etc.) and a pop-out dashboard with interactive topic refinement
(merge, delete, rename) plus five exploration screens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import threading
import time
import os
import pathlib
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import customtkinter as ctk
import pandas as pd
import numpy as np

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip, _style_treeview
from datalens_v3_opt.data.preprocessing import (
    LANGUAGE_NAMES, preprocess_corpus, preview_cleaning, build_stopword_set,
)

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App

# ── Model cache ───────────────────────────────────────────────────────────────
MODEL_CACHE = C.get_model_cache()

MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "label": "MiniLM-L6  ·  English  ·  Fast",
        "info":  "22M params  ·  256 token limit  ·  Best for English text",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "label": "Multilingual MiniLM-L12",
        "info":  "118M params  ·  128 token limit  ·  50+ languages",
    },
}
_MODEL_LABELS = [v["label"] for v in MODELS.values()]
_ID_BY_LABEL  = {v["label"]: k for k, v in MODELS.items()}

# Runtime (thread-safe via _model_lock)
_embedder      = None
_embedder_id   = None
_model_ready   = threading.Event()
_model_loading = False
_model_error   = None
_model_lock    = threading.Lock()


def _load_embedder(model_id: str, status_cb=None):
    global _embedder, _embedder_id, _model_error, _model_loading
    with _model_lock:
        if _embedder is not None and _embedder_id == model_id:
            if status_cb:
                status_cb(f"Model ready  ·  {MODELS[model_id]['label']}")
            _model_ready.set()
            return
        _model_loading = True
        _model_ready.clear()
    try:
        import gc
        with _model_lock:
            if _embedder is not None:
                del _embedder
                _embedder = None
                gc.collect()
        from sentence_transformers import SentenceTransformer
        if status_cb:
            status_cb("Downloading model (first run only)…")
        model = SentenceTransformer(model_id, cache_folder=str(MODEL_CACHE))
        with _model_lock:
            _embedder = model
            _embedder_id = model_id
            _model_error = None
        if status_cb:
            status_cb(f"Model ready  ·  {MODELS[model_id]['label']}")
    except ImportError:
        with _model_lock:
            _model_error = "sentence-transformers not installed.\npip install sentence-transformers"
        if status_cb:
            status_cb("Error: missing package")
    except Exception as exc:
        with _model_lock:
            _model_error = str(exc)
        if status_cb:
            status_cb(f"Error: {str(exc)[:60]}")
    finally:
        with _model_lock:
            _model_loading = False
        _model_ready.set()


TOPIC_COLOURS = [
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
    "#06b6d4", "#e11d48", "#10b981", "#a855f7", "#eab308",
    "#0ea5e9", "#d946ef", "#64748b", "#f43f5e", "#2dd4bf",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Main frame (config + launch)
# ══════════════════════════════════════════════════════════════════════════════

class TopicModellingFrame(ctk.CTkFrame):
    def __init__(self, master, app: "App"):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._cancel_event = threading.Event()
        self._running = False
        self._dashboard_win = None
        self._build()
        # Auto-install ML packages if missing, then load the model
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is not None:
            self._trigger_load()
        else:
            self._auto_install_ml()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        # Title bar
        tb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=48)
        tb.pack(fill="x"); tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="Topic Modelling", font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(side="left", padx=16)
        ctk.CTkLabel(tb, text="Discover themes and patterns in your text data",
                     text_color=C.MUTED, font=("Segoe UI", 11)
                     ).pack(side="left", padx=4)
        self._status_var = tk.StringVar(value="Loading model…")
        ctk.CTkLabel(tb, textvariable=self._status_var,
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(side="right", padx=12)

        # Config
        cfg = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0)
        cfg.pack(fill="x")
        inner = ctk.CTkFrame(cfg, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=10)

        # Row 0 — Column + Language + Model
        r0 = ctk.CTkFrame(inner, fg_color="transparent"); r0.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(r0, text="Text column:", text_color=C.MUTED,
                     font=("Segoe UI", 12), width=100).pack(side="left")
        self._col_var = tk.StringVar(value="")
        self._col_om = ctk.CTkOptionMenu(r0, variable=self._col_var,
                                         values=["— load a dataset —"], width=220)
        self._col_om.pack(side="left", padx=(0, 16))
        tip(self._col_om, "The column containing text to analyse.")

        ctk.CTkLabel(r0, text="Language:", text_color=C.MUTED,
                     font=("Segoe UI", 12)).pack(side="left")
        self._lang_var = tk.StringVar(value="Auto-detect")
        lang_om = ctk.CTkOptionMenu(
            r0, variable=self._lang_var,
            values=list(LANGUAGE_NAMES.values()), width=140)
        lang_om.pack(side="left", padx=(4, 16))
        tip(lang_om,
            "Language for stopword removal.\n\n"
            "'Auto-detect' samples your data and picks the best match.\n"
            "Set manually if detection is wrong or your data is mixed.")

        ctk.CTkLabel(r0, text="Model:", text_color=C.MUTED,
                     font=("Segoe UI", 12)).pack(side="left")
        self._model_var = tk.StringVar(value=_MODEL_LABELS[0])
        model_om = ctk.CTkOptionMenu(
            r0, variable=self._model_var, values=_MODEL_LABELS, width=240,
            command=self._on_model_change)
        model_om.pack(side="left", padx=(4, 0))
        tip(model_om,
            "MiniLM-L6 — 22M params, fastest, best for English.\n"
            "Multilingual MiniLM-L12 — 118M params, 50+ languages.\n\n"
            "If your data is not English, use the multilingual model.")

        # Row 1 — Core params
        r1 = ctk.CTkFrame(inner, fg_color="transparent"); r1.pack(fill="x", pady=(0, 4))
        for lbl, var_name, default, w, tt in [
            ("Documents:", "_n_docs_var", "500", 80,
             "Number of documents to analyse.\nUse 'All' for every row.\n"
             "Start with 300–500, increase for deeper analysis."),
            ("Min topic size:", "_min_topic_var", "10", 60,
             "Minimum docs for a topic.\nLower = more granular.\n"
             "Higher = broader, more reliable."),
            ("Max topics:", "_nr_topics_var", "Auto", 60,
             "Set a number or leave 'Auto'.\n"
             "Useful to force merging into fewer topics."),
        ]:
            ctk.CTkLabel(r1, text=lbl, text_color=C.MUTED,
                         font=("Segoe UI", 12)).pack(side="left", padx=(12, 0) if lbl != "Documents:" else 0)
            v = tk.StringVar(value=default)
            setattr(self, var_name, v)
            e = ctk.CTkEntry(r1, textvariable=v, width=w, placeholder_text=default)
            e.pack(side="left", padx=(4, 0))
            tip(e, tt)

        # ── Expandable: Preprocessing ─────────────────────────────────────────
        self._preproc_btn = ctk.CTkButton(
            inner, text="▶  Preprocessing  —  click to expand",
            fg_color=C.SELECT, hover_color=C.BTN, text_color=C.TEXT,
            font=("Segoe UI", 11), height=28, anchor="w",
            command=self._toggle_preproc)
        self._preproc_btn.pack(fill="x", pady=(6, 0))

        self._preproc_frame = ctk.CTkFrame(inner, fg_color="transparent")
        # starts hidden

        # Cleaning checkboxes
        chk_row = ctk.CTkFrame(self._preproc_frame, fg_color="transparent")
        chk_row.pack(fill="x", pady=(6, 4))
        self._clean_urls     = tk.BooleanVar(value=True)
        self._clean_mentions = tk.BooleanVar(value=True)
        self._clean_hashtags = tk.BooleanVar(value=True)
        self._clean_emojis   = tk.BooleanVar(value=True)
        self._clean_numbers  = tk.BooleanVar(value=True)
        self._clean_lower    = tk.BooleanVar(value=True)
        for text, var in [
            ("URLs", self._clean_urls),
            ("@mentions", self._clean_mentions),
            ("#hashtags (strip symbol)", self._clean_hashtags),
            ("Emojis", self._clean_emojis),
            ("Numbers", self._clean_numbers),
            ("Lowercase", self._clean_lower),
        ]:
            ctk.CTkCheckBox(chk_row, text=text, variable=var,
                            text_color=C.TEXT, font=("Segoe UI", 11),
                            width=20, height=20, checkbox_width=18, checkbox_height=18
                            ).pack(side="left", padx=(0, 14))

        # Min word/doc length
        len_row = ctk.CTkFrame(self._preproc_frame, fg_color="transparent")
        len_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(len_row, text="Min word length:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left")
        self._min_token_var = tk.StringVar(value="3")
        e = ctk.CTkEntry(len_row, textvariable=self._min_token_var, width=40)
        e.pack(side="left", padx=(4, 16))
        tip(e, "Words shorter than this are ignored in topic keywords.\n"
               "Kills noise like 'da', 'ar', 'um'. Default: 3.")
        ctk.CTkLabel(len_row, text="Min doc words:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left")
        self._min_doc_var = tk.StringVar(value="5")
        e = ctk.CTkEntry(len_row, textvariable=self._min_doc_var, width=40)
        e.pack(side="left", padx=(4, 16))
        tip(e, "Documents with fewer words after cleaning are discarded.\n"
               "Removes empty or very short texts. Default: 5.")

        # Custom stopwords
        sw_row = ctk.CTkFrame(self._preproc_frame, fg_color="transparent")
        sw_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(sw_row, text="Custom stopwords:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", anchor="n", pady=(4, 0))
        self._custom_sw_var = tk.StringVar(value="")
        sw_entry = ctk.CTkEntry(sw_row, textvariable=self._custom_sw_var, width=500,
                                placeholder_text="Comma-separated: siga, envie, curtiu, convidado …")
        sw_entry.pack(side="left", padx=(4, 0))
        tip(sw_entry,
            "Add extra words to remove from topic keywords.\n"
            "Separate with commas. These are added on top of\n"
            "the built-in language stopwords.\n\n"
            "Useful for removing engagement bait, brand names,\n"
            "influencer handles, or domain-specific noise.")

        # Preview button
        preview_row = ctk.CTkFrame(self._preproc_frame, fg_color="transparent")
        preview_row.pack(fill="x", pady=(0, 2))
        ctk.CTkButton(preview_row, text="Preview Cleaning (5 samples)",
                      fg_color=C.BTN, width=200, height=28,
                      command=self._preview_cleaning).pack(side="left")

        # ── Expandable: Advanced ──────────────────────────────────────────────
        self._adv_btn = ctk.CTkButton(
            inner, text="▶  Advanced  —  UMAP, HDBSCAN, outlier reduction",
            fg_color=C.SELECT, hover_color=C.BTN, text_color=C.TEXT,
            font=("Segoe UI", 11), height=28, anchor="w",
            command=self._toggle_advanced)
        self._adv_btn.pack(fill="x", pady=(4, 0))

        self._adv_frame = ctk.CTkFrame(inner, fg_color="transparent")
        # starts hidden

        adv_r0 = ctk.CTkFrame(self._adv_frame, fg_color="transparent")
        adv_r0.pack(fill="x", pady=(6, 4))
        for lbl, var_name, default, w, tt in [
            ("UMAP neighbours:", "_umap_n_var", "15", 50,
             "n_neighbors — controls local vs global structure.\n"
             "Lower (5–10) = tighter clusters.\n"
             "Higher (20–50) = broader structure. Default: 15."),
            ("UMAP min_dist:", "_umap_dist_var", "0.0", 50,
             "How tightly points are packed.\n"
             "0.0 = tight clusters. 0.5 = spread out. Default: 0.0."),
            ("HDBSCAN min_samples:", "_hdb_samples_var", "auto", 60,
             "'auto' = min_topic_size / 3.\n"
             "Lower = more topics, fewer outliers.\n"
             "Higher = stricter, more outliers."),
        ]:
            ctk.CTkLabel(adv_r0, text=lbl, text_color=C.MUTED,
                         font=("Segoe UI", 11)).pack(side="left", padx=(12, 0) if lbl != "UMAP neighbours:" else 0)
            v = tk.StringVar(value=default)
            setattr(self, var_name, v)
            e = ctk.CTkEntry(adv_r0, textvariable=v, width=w)
            e.pack(side="left", padx=(4, 0))
            tip(e, tt)

        adv_r1 = ctk.CTkFrame(self._adv_frame, fg_color="transparent")
        adv_r1.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(adv_r1, text="Outlier reduction:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left")
        self._outlier_var = tk.StringVar(value="None")
        om = ctk.CTkOptionMenu(adv_r1, variable=self._outlier_var,
                                values=["None", "Embeddings", "c-TF-IDF"], width=140)
        om.pack(side="left", padx=(4, 16))
        tip(om,
            "After clustering, reassign outlier documents:\n\n"
            "None — keep outliers as-is.\n"
            "Embeddings — assign by semantic similarity.\n"
            "c-TF-IDF — assign by keyword similarity.\n\n"
            "Reduces the outlier count but may assign noisy docs.")

        ctk.CTkLabel(adv_r1, text="Seed:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left")
        self._seed_var = tk.StringVar(value="42")
        e = ctk.CTkEntry(adv_r1, textvariable=self._seed_var, width=50)
        e.pack(side="left", padx=(4, 0))
        tip(e, "Random seed for reproducibility.\nSame seed = same results.")

        # ── Run / Cancel / Progress ───────────────────────────────────────────
        r_btn = ctk.CTkFrame(inner, fg_color="transparent")
        r_btn.pack(fill="x", pady=(8, 0))
        self._run_btn = ctk.CTkButton(
            r_btn, text="Find Topics", fg_color=C.ACCENT, width=140,
            font=("Segoe UI", 12, "bold"), command=self._run)
        self._run_btn.pack(side="left", padx=(0, 8))
        tip(self._run_btn, "Run the full pipeline: clean → embed → cluster → display.")
        self._cancel_btn = ctk.CTkButton(
            r_btn, text="Cancel", fg_color=C.BTN, width=80,
            state="disabled", command=self._cancel)
        self._cancel_btn.pack(side="left", padx=(0, 12))
        self._progress_var = tk.DoubleVar(value=0)
        ctk.CTkProgressBar(r_btn, variable=self._progress_var,
                           width=200, height=12).pack(side="left", padx=(0, 8))
        self._prog_lbl = ctk.CTkLabel(r_btn, text="", text_color=C.MUTED,
                                      font=("Segoe UI", 10))
        self._prog_lbl.pack(side="left")

        # Welcome
        self._welcome = ctk.CTkFrame(self, fg_color=C.BG, corner_radius=0)
        self._welcome.pack(fill="both", expand=True)
        self._show_welcome()

    # ── Expandable toggles ────────────────────────────────────────────────────

    def _toggle_preproc(self):
        if self._preproc_frame.winfo_manager():
            self._preproc_frame.pack_forget()
            self._preproc_btn.configure(text="▶  Preprocessing  —  click to expand")
        else:
            self._preproc_frame.pack(fill="x", after=self._preproc_btn, pady=(2, 0))
            self._preproc_btn.configure(text="▼  Preprocessing")

    def _toggle_advanced(self):
        if self._adv_frame.winfo_manager():
            self._adv_frame.pack_forget()
            self._adv_btn.configure(text="▶  Advanced  —  UMAP, HDBSCAN, outlier reduction")
        else:
            self._adv_frame.pack(fill="x", after=self._adv_btn, pady=(2, 0))
            self._adv_btn.configure(text="▼  Advanced")

    # ── Preview ───────────────────────────────────────────────────────────────

    def _preview_cleaning(self):
        col = self._col_var.get()
        if not col or col not in self.app.df.columns or self.app.df.empty:
            messagebox.showinfo("No data", "Import a dataset first.")
            return
        texts = self.app.df[col].dropna().astype(str).tolist()
        pairs = preview_cleaning(
            texts,
            remove_urls=self._clean_urls.get(),
            remove_mentions=self._clean_mentions.get(),
            strip_hashtag_symbol=self._clean_hashtags.get(),
            remove_emojis=self._clean_emojis.get(),
            remove_numbers=self._clean_numbers.get(),
            lowercase=self._clean_lower.get())

        win = ctk.CTkToplevel(self)
        win.title("Cleaning Preview — Before / After")
        win.geometry("900x520")
        win.configure(fg_color=C.BG)
        win.grab_set()

        ctk.CTkLabel(win, text="Cleaning Preview",
                     font=("Segoe UI", 14, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=16, pady=(12, 4))
        ctk.CTkLabel(win,
                     text="These are 5 random documents showing what the cleaning pipeline does.",
                     font=("Segoe UI", 11), text_color=C.MUTED
                     ).pack(anchor="w", padx=16, pady=(0, 8))

        sf = ctk.CTkScrollableFrame(win, fg_color=C.BG)
        sf.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        for i, (before, after) in enumerate(pairs):
            card = ctk.CTkFrame(sf, fg_color=C.CARD, corner_radius=8)
            card.pack(fill="x", pady=4)
            ctk.CTkLabel(card, text=f"Document {i+1}",
                         font=("Segoe UI", 10, "bold"), text_color=C.MUTED
                         ).pack(anchor="w", padx=12, pady=(8, 2))
            ctk.CTkLabel(card, text=f"BEFORE:  {before[:300]}",
                         font=("Segoe UI", 10), text_color="#ef4444",
                         wraplength=800, justify="left", anchor="w"
                         ).pack(anchor="w", padx=12, pady=(0, 2))
            ctk.CTkLabel(card, text=f"AFTER:   {after[:300]}",
                         font=("Segoe UI", 10), text_color="#22c55e",
                         wraplength=800, justify="left", anchor="w"
                         ).pack(anchor="w", padx=12, pady=(0, 8))

        ctk.CTkButton(win, text="Close", fg_color=C.BTN, width=100,
                      command=win.destroy).pack(pady=(0, 12))

    # ── Welcome ───────────────────────────────────────────────────────────────

    def _show_welcome(self):
        for w in self._welcome.winfo_children():
            w.destroy()
        wrap = ctk.CTkFrame(self._welcome, fg_color="transparent")
        wrap.pack(expand=True)
        ctk.CTkLabel(wrap, text="How Topic Modelling Works",
                     font=("Segoe UI", 18, "bold"), text_color=C.TEXT
                     ).pack(pady=(30, 14))
        for title, desc in [
            ("1.  Choose your text column and language",
             "Select the column to analyse. Set the language for accurate "
             "stopword removal, or use auto-detect."),
            ("2.  Configure preprocessing (optional)",
             "Expand the Preprocessing section to toggle URL/mention/emoji "
             "removal and add custom stopwords for cleaner results."),
            ("3.  Adjust parameters or use defaults",
             "Start with 300–500 documents and default settings. "
             "Expand Advanced for fine-grained clustering control."),
            ("4.  Click 'Find Topics'",
             "The pipeline cleans your text, computes embeddings, clusters "
             "them, and opens a full-size dashboard with the results."),
            ("5.  Explore and refine",
             "Use the dashboard to browse topics, view the map, inspect "
             "keywords, merge duplicates, delete junk, and rename topics."),
        ]:
            sf = ctk.CTkFrame(wrap, fg_color=C.CARD, corner_radius=8)
            sf.pack(fill="x", padx=50, pady=3)
            ctk.CTkLabel(sf, text=title, font=("Segoe UI", 12, "bold"),
                         text_color=C.TEXT, anchor="w"
                         ).pack(anchor="w", padx=16, pady=(8, 2))
            ctk.CTkLabel(sf, text=desc, font=("Segoe UI", 11),
                         text_color=C.MUTED, anchor="w", wraplength=600
                         ).pack(anchor="w", padx=16, pady=(0, 8))

        # Auto-install progress card OR manual install button
        from datalens_v3_opt.ui.ml_installer import MLInstallButton, ml_installed
        if not ml_installed():
            if getattr(self, "_ml_auto_installing", False):
                # Show progress card during auto-install
                card = ctk.CTkFrame(wrap, fg_color=C.CARD, corner_radius=8)
                card.pack(fill="x", padx=50, pady=(12, 0))
                ctk.CTkLabel(card, text="Setting up Topic Modelling…",
                             font=("Segoe UI", 14, "bold"), text_color=C.ACCENT
                             ).pack(anchor="w", padx=16, pady=(12, 2))
                self._install_detail_var = tk.StringVar(
                    value="Downloading required packages (first launch only)…")
                ctk.CTkLabel(card, textvariable=self._install_detail_var,
                             font=("Segoe UI", 11), text_color=C.MUTED,
                             wraplength=600, justify="left"
                             ).pack(anchor="w", padx=16, pady=(0, 4))
                prog = ctk.CTkProgressBar(card, width=500, height=10,
                                          mode="indeterminate")
                prog.pack(padx=16, pady=(0, 14))
                prog.start()
                self._install_prog_bar = prog
            else:
                MLInstallButton(wrap).pack(fill="x", padx=50, pady=(12, 0))

    # ── Auto-install ML packages ─────────────────────────────────────────────

    def _auto_install_ml(self):
        """Download ML packages in the background, then load the model."""
        import logging
        log = logging.getLogger("datalens.ml_install")
        log.info("Auto-install: starting ML package download")

        from datalens_v3_opt.ui.ml_installer import run_auto_install

        # Flag so _show_welcome shows the progress card instead of the button
        self._ml_auto_installing = True
        self._show_welcome()
        self._set_status("Downloading ML packages (first launch only)…")

        def _on_status(msg):
            log.info("Auto-install status: %s", msg)
            self._set_status(msg)
            # Also update the detail label on the progress card
            if hasattr(self, "_install_detail_var"):
                self.after(0, lambda m=msg: self._install_detail_var.set(m))

        def _on_done(ok):
            self._ml_auto_installing = False
            if ok:
                log.info("Auto-install: success, loading model")
                self._set_status("ML packages ready — loading model…")
                self.after(0, self._show_welcome)
                self.after(0, self._trigger_load)
            else:
                log.error("Auto-install: failed")
                self._set_status("ML install failed — click Install below to retry")
                self.after(0, self._show_welcome)

        run_auto_install(status_cb=_on_status, done_cb=_on_done)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _get_model_id(self) -> str:
        return _ID_BY_LABEL.get(self._model_var.get(),
                                "sentence-transformers/all-MiniLM-L6-v2")

    def _trigger_load(self):
        mid = self._get_model_id()
        if _embedder is not None and _embedder_id == mid:
            self._status_var.set(f"Model ready  ·  {MODELS[mid]['label']}")
            return
        threading.Thread(target=_load_embedder,
                         args=(mid, self._set_status), daemon=True).start()
        self._poll_ready()

    def _on_model_change(self, _label: str):
        self._trigger_load()

    def _poll_ready(self):
        if _model_ready.is_set():
            if _model_error:
                self._set_status(f"Error: {_model_error.splitlines()[0]}")
            elif _embedder:
                self._set_status(f"Model ready  ·  {MODELS.get(_embedder_id, {}).get('label', '')}")
        else:
            self.after(400, self._poll_ready)

    def _set_status(self, msg):
        self.after(0, lambda m=msg: self._status_var.set(m))

    # ── Refresh / Validate ────────────────────────────────────────────────────

    def refresh(self):
        df = self.app.df
        if df.empty:
            self._col_om.configure(values=["— load a dataset —"]); return
        cols = [c for c in df.columns if not c.startswith("_")]
        self._col_om.configure(values=cols)
        if not self._col_var.get() or self._col_var.get() not in cols:
            text_col = self.app.resolve_col("content_text", df)
            self._col_var.set(text_col or cols[0])

    def _validate(self):
        mid = self._get_model_id()
        if not _embedder or _embedder_id != mid:
            if _model_loading:
                messagebox.showinfo("Please wait",
                                    "The embedding model is still loading.\n"
                                    "Check the status in the top-right corner.")
            else:
                messagebox.showerror("Model not ready",
                                     _model_error or "Model failed to load.\n"
                                     "Check the log or restart the app.")
            return None
        if self.app.df.empty:
            messagebox.showinfo("No data",
                                "Import a dataset first using the '+ Import' button.")
            return None
        col = self._col_var.get()
        if not col or col == "— load a dataset —" or col not in self.app.df.columns:
            messagebox.showinfo("No column",
                                "Select a text column from the dropdown.")
            return None
        return col

    # ── Run ───────────────────────────────────────────────────────────────────

    def _parse_int(self, var, default, minimum=None):
        try:
            v = int(var.get().strip())
            return max(v, minimum) if minimum else v
        except ValueError:
            return default

    def _parse_float(self, var, default):
        try:
            return float(var.get().strip())
        except ValueError:
            return default

    def _run(self):
        col = self._validate()
        if not col:
            return
        if self._running:
            # Safety: if stuck from a previous crash, reset
            self._running = False
            self._run_btn.configure(state="normal", fg_color=C.ACCENT)
            self._cancel_btn.configure(state="disabled")
            return

        n_str = self._n_docs_var.get().strip()
        if n_str.lower() == "all":
            n_docs = len(self.app.df)
        else:
            try:
                n_docs = int(n_str)
                if n_docs < 20:
                    messagebox.showinfo("Too few", "Use at least 20 documents."); return
            except ValueError:
                messagebox.showinfo("Invalid", "'Documents' must be a number or 'All'."); return

        # Resolve language
        lang_label = self._lang_var.get()
        lang_code = "auto"
        for code, name in LANGUAGE_NAMES.items():
            if name == lang_label:
                lang_code = code; break

        params = dict(
            col=col, n_docs=n_docs,
            min_topic=self._parse_int(self._min_topic_var, 10, 2),
            nr_topics_str=self._nr_topics_var.get().strip(),
            language=lang_code,
            remove_urls=self._clean_urls.get(),
            remove_mentions=self._clean_mentions.get(),
            strip_hashtags=self._clean_hashtags.get(),
            remove_emojis=self._clean_emojis.get(),
            remove_numbers=self._clean_numbers.get(),
            lowercase=self._clean_lower.get(),
            min_token_length=self._parse_int(self._min_token_var, 3, 1),
            min_doc_words=self._parse_int(self._min_doc_var, 5, 1),
            custom_stopwords=[w.strip() for w in self._custom_sw_var.get().split(",") if w.strip()],
            umap_n=self._parse_int(self._umap_n_var, 15, 2),
            umap_dist=self._parse_float(self._umap_dist_var, 0.0),
            hdb_samples_str=self._hdb_samples_var.get().strip(),
            outlier_strategy=self._outlier_var.get(),
            seed=self._parse_int(self._seed_var, 42),
            model_id=self._get_model_id(),
        )

        self._running = True
        self._cancel_event.clear()
        self._run_btn.configure(state="disabled", fg_color=C.DIM)
        self._cancel_btn.configure(state="normal")
        self._progress_var.set(0)
        self._prog_lbl.configure(text="Preparing…")
        threading.Thread(target=self._worker, args=(params,), daemon=True).start()

    def _worker(self, p):
        try:
            df = self.app.df
            # 1. Sample raw texts
            raw = df[p["col"]].dropna().astype(str)
            raw = raw[raw.str.strip() != ""]
            if len(raw) > p["n_docs"]:
                raw = raw.sample(p["n_docs"], random_state=p["seed"])
            raw_texts = raw.tolist()

            if len(raw_texts) < 20:
                self.after(0, lambda: self._finish_error(
                    "Not enough text. Need at least 20 non-empty rows."))
                return

            self._update_prog(0.05, "Preprocessing…")
            if self._cancel_event.is_set(): self.after(0, self._finish_cancelled); return

            # 2. Preprocess
            cleaned, lang, stopword_list, clean_stats = preprocess_corpus(
                raw_texts,
                language=p["language"],
                remove_urls=p["remove_urls"],
                remove_mentions=p["remove_mentions"],
                strip_hashtag_symbol=p["strip_hashtags"],
                remove_emojis=p["remove_emojis"],
                remove_numbers=p["remove_numbers"],
                lowercase=p["lowercase"],
                min_token_length=p["min_token_length"],
                min_doc_words=p["min_doc_words"],
                custom_stopwords=p["custom_stopwords"],
            )

            if len(cleaned) < 20:
                self.after(0, lambda: self._finish_error(
                    f"Only {len(cleaned)} docs left after cleaning (need 20+). "
                    "Lower 'Min doc words' or add more data."))
                return

            self._update_prog(0.10, f"Embedding {len(cleaned):,} docs ({clean_stats['language']})…")
            if self._cancel_event.is_set(): self.after(0, self._finish_cancelled); return

            # 3. Embed
            t0 = time.time()
            embeddings = _embedder.encode(cleaned, show_progress_bar=False, batch_size=64)

            self._update_prog(0.40, "Clustering…")
            if self._cancel_event.is_set(): self.after(0, self._finish_cancelled); return

            # 4. BERTopic
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer

            min_topic = p["min_topic"]
            hdb_str = p["hdb_samples_str"]
            hdb_samples = max(1, min_topic // 3) if hdb_str.lower() == "auto" else self._parse_int(
                type("V", (), {"get": lambda s: hdb_str, "strip": lambda s: hdb_str})(), max(1, min_topic // 3), 1)

            nr_str = p["nr_topics_str"]
            nr_topics = None
            if nr_str.lower() not in ("auto", ""):
                try:
                    nr_topics = max(2, int(nr_str))
                except ValueError:
                    pass

            token_pat = r"\b\w{" + str(p["min_token_length"]) + r",}\b"

            vectorizer = CountVectorizer(
                stop_words=stopword_list, ngram_range=(1, 2),
                min_df=2, token_pattern=token_pat)

            topic_model = BERTopic(
                embedding_model=_embedder,
                umap_model=UMAP(n_neighbors=p["umap_n"], n_components=5,
                                min_dist=p["umap_dist"], metric="cosine",
                                random_state=p["seed"]),
                hdbscan_model=HDBSCAN(min_cluster_size=min_topic,
                                      min_samples=hdb_samples,
                                      metric="euclidean", prediction_data=True),
                vectorizer_model=vectorizer,
                nr_topics=nr_topics if nr_topics else "auto",
                top_n_words=10, verbose=False)

            self._update_prog(0.50, "Fitting model…")
            try:
                topics, probs = topic_model.fit_transform(cleaned, embeddings)
            except ValueError:
                # min_df=2 can fail when topics have very few docs; retry with 1
                vectorizer = CountVectorizer(
                    stop_words=stopword_list, ngram_range=(1, 2),
                    min_df=1, token_pattern=token_pat)
                topic_model = BERTopic(
                    embedding_model=_embedder,
                    umap_model=UMAP(n_neighbors=p["umap_n"], n_components=5,
                                    min_dist=p["umap_dist"], metric="cosine",
                                    random_state=p["seed"]),
                    hdbscan_model=HDBSCAN(min_cluster_size=min_topic,
                                          min_samples=hdb_samples,
                                          metric="euclidean",
                                          prediction_data=True),
                    vectorizer_model=vectorizer,
                    nr_topics=nr_topics if nr_topics else "auto",
                    top_n_words=10, verbose=False)
                topics, probs = topic_model.fit_transform(cleaned, embeddings)

            # 5. Outlier reduction
            if p["outlier_strategy"] != "None":
                self._update_prog(0.70, "Reducing outliers…")
                strategy = p["outlier_strategy"].lower().replace("-", "_")
                if strategy == "embeddings":
                    new_topics = topic_model.reduce_outliers(
                        cleaned, topics, strategy="embeddings",
                        embeddings=embeddings)
                else:
                    new_topics = topic_model.reduce_outliers(
                        cleaned, topics, strategy="c-tf-idf")
                topic_model.update_topics(cleaned, topics=new_topics)
                topics = new_topics

            self._update_prog(0.80, "Building visualisation…")
            if self._cancel_event.is_set(): self.after(0, self._finish_cancelled); return

            # 6. 2D projections
            emb_2d = UMAP(n_neighbors=p["umap_n"], n_components=2,
                          min_dist=0.1, metric="cosine",
                          random_state=p["seed"]).fit_transform(embeddings)

            topic_info = topic_model.get_topic_info()
            total_time = time.time() - t0
            self._update_prog(1.0, "Done")

            self.after(0, lambda: self._finish_success(
                topic_model, topic_info, topics, embeddings, emb_2d,
                cleaned, total_time, clean_stats))

        except Exception as exc:
            import traceback; traceback.print_exc()
            self.after(0, lambda e=str(exc): self._finish_error(e))

    def _update_prog(self, pct, msg):
        self.after(0, lambda: (self._progress_var.set(pct),
                               self._prog_lbl.configure(text=msg)))

    def _cancel(self):
        self._cancel_event.set()

    def _finish_cancelled(self):
        self._running = False
        self._run_btn.configure(state="normal", fg_color=C.ACCENT)
        self._cancel_btn.configure(state="disabled")
        self._prog_lbl.configure(text="Cancelled"); self._progress_var.set(0)

    def _finish_error(self, msg):
        self._running = False
        self._run_btn.configure(state="normal", fg_color=C.ACCENT)
        self._cancel_btn.configure(state="disabled")
        self._prog_lbl.configure(text=""); self._progress_var.set(0)
        messagebox.showerror("Topic Modelling Error", msg)

    def _finish_success(self, topic_model, topic_info, topics, embeddings,
                        emb_2d, texts, total_time, clean_stats):
        self._running = False
        self._run_btn.configure(state="normal", fg_color=C.ACCENT)
        self._cancel_btn.configure(state="disabled")

        n_topics = len([t for t in set(topics) if t != -1])
        outliers = sum(1 for t in topics if t == -1)
        self._prog_lbl.configure(
            text=f"{n_topics} topics  ·  {len(texts):,} docs  ·  {total_time:.1f}s")

        if self._dashboard_win and self._dashboard_win.winfo_exists():
            self._dashboard_win.destroy()

        self._dashboard_win = TopicDashboard(
            self, topic_model, topic_info, topics, embeddings, emb_2d,
            texts, n_topics, outliers, len(texts), total_time, clean_stats)

    # ── Theme rebuild ─────────────────────────────────────────────────────────

    def rebuild(self):
        saved = {}
        for attr in ("_col_var", "_n_docs_var", "_min_topic_var", "_nr_topics_var",
                      "_lang_var", "_model_var", "_custom_sw_var", "_seed_var",
                      "_min_token_var", "_min_doc_var", "_umap_n_var",
                      "_umap_dist_var", "_hdb_samples_var", "_outlier_var"):
            if hasattr(self, attr):
                saved[attr] = getattr(self, attr).get()

        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._cancel_event = threading.Event()
        self._running = False
        self._build()

        for attr, val in saved.items():
            if hasattr(self, attr):
                getattr(self, attr).set(val)

        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            self._auto_install_ml()
        elif _model_error:
            self._status_var.set(f"Error: {_model_error.splitlines()[0]}")
        elif _embedder:
            self._status_var.set(f"Model ready  ·  {MODELS.get(_embedder_id, {}).get('label', '')}")
        else:
            self._status_var.set("Loading model…"); self._poll_ready()
        self.refresh()


# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard window
# ══════════════════════════════════════════════════════════════════════════════

class TopicDashboard(ctk.CTkToplevel):
    SCREENS = [
        ("overview",   "Overview"),
        ("map",        "Topic Map"),
        ("keywords",   "Keywords"),
        ("similarity", "Similarity"),
        ("documents",  "Document Explorer"),
    ]

    def __init__(self, parent, topic_model, topic_info, topics, embeddings,
                 emb_2d, texts, n_topics, outliers, n_docs, total_time,
                 clean_stats):
        super().__init__(parent)
        self.title("Topic Modelling — Results Dashboard")
        self.geometry("1340x880")
        self.minsize(1000, 650)
        self.configure(fg_color=C.BG)

        self._tm           = topic_model
        self._topic_info   = topic_info
        self._topics       = list(topics)
        self._embeddings   = embeddings
        self._emb_2d       = emb_2d
        self._texts        = texts
        self._n_topics     = n_topics
        self._outliers     = outliers
        self._n_docs       = n_docs
        self._total_time   = total_time
        self._clean_stats  = clean_stats
        self._labels       = {}       # {tid: custom label}
        self._hidden       = set()    # hidden topic ids

        self._active  = None
        self._cache   = {}
        self._btns    = {}
        self._figures: list = []  # track matplotlib figures for cleanup

        self._build()
        self._switch("overview")
        self.after(100, self.focus_force)

    def destroy(self):
        """Close all matplotlib figures before destroying the window."""
        import sys
        if "matplotlib.pyplot" in sys.modules:
            import matplotlib.pyplot as _plt
            for fig in self._figures:
                try:
                    _plt.close(fig)
                except Exception:
                    pass
        self._figures.clear()
        super().destroy()

    def _build(self):
        top = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=100)
        top.pack(fill="x"); top.pack_propagate(False)

        # Stats
        sr = ctk.CTkFrame(top, fg_color="transparent")
        sr.pack(fill="x", padx=20, pady=(10, 0))
        ctk.CTkLabel(sr, text="Topic Modelling Results",
                     font=("Segoe UI", 16, "bold"), text_color=C.TEXT
                     ).pack(side="left", padx=(0, 24))
        for v, l in [
            (str(self._n_topics), "Topics"),
            (f"{self._n_docs:,}", "Documents"),
            (f"{self._outliers:,}", "Outliers"),
            (f"{self._total_time:.1f}s", "Time"),
            (self._clean_stats.get("language", "?"), "Language"),
        ]:
            f = ctk.CTkFrame(sr, fg_color="transparent"); f.pack(side="left", padx=(0, 20))
            ctk.CTkLabel(f, text=v, font=("Segoe UI", 14, "bold"),
                         text_color=C.ACCENT).pack(side="left", padx=(0, 4))
            ctk.CTkLabel(f, text=l, font=("Segoe UI", 11),
                         text_color=C.MUTED).pack(side="left")

        # Tab bar
        tabs = ctk.CTkFrame(top, fg_color="transparent")
        tabs.pack(fill="x", padx=16, pady=(8, 0), side="bottom")
        for key, label in self.SCREENS:
            btn = ctk.CTkButton(tabs, text=label, width=130, height=32,
                                corner_radius=6, font=("Segoe UI", 12),
                                fg_color="transparent", hover_color=C.SELECT,
                                text_color=C.MUTED,
                                command=lambda k=key: self._switch(k))
            btn.pack(side="left", padx=(0, 4))
            self._btns[key] = btn

        self._content = ctk.CTkFrame(self, fg_color=C.BG, corner_radius=0)
        self._content.pack(fill="both", expand=True)

    def _switch(self, key):
        if self._active == key:
            return
        for k, btn in self._btns.items():
            btn.configure(fg_color=C.ACCENT if k == key else "transparent",
                          text_color=C.TEXT if k == key else C.MUTED)
        for c in self._content.winfo_children():
            c.pack_forget()
        if key not in self._cache:
            self._cache[key] = getattr(self, f"_build_{key}")()
        self._cache[key].pack(fill="both", expand=True)
        self._active = key

    def _invalidate_screens(self):
        """Clear cached screens after a merge/delete/rename."""
        # Close matplotlib figures before destroying screens
        import sys
        if "matplotlib.pyplot" in sys.modules:
            import matplotlib.pyplot as _plt
            for fig in self._figures:
                try:
                    _plt.close(fig)
                except Exception:
                    pass
        self._figures.clear()
        for key in list(self._cache.keys()):
            self._cache[key].destroy()
        self._cache.clear()
        self._topic_info = self._tm.get_topic_info()
        self._n_topics = len([t for t in set(self._topics) if t != -1 and t not in self._hidden])
        self._outliers = sum(1 for t in self._topics if t == -1)
        current = self._active
        self._active = None
        self._switch(current)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _kw(self, row, n=8):
        if "Representation" in (row.index if hasattr(row, "index") else []):
            rep = row["Representation"]
            if isinstance(rep, list):
                return ", ".join(rep[:n])
        if "Name" in (row.index if hasattr(row, "index") else []):
            return str(row["Name"])
        return ""

    def _short_kw(self, tid):
        r = self._topic_info[self._topic_info["Topic"] == tid]
        return self._kw(r.iloc[0], 3) if not r.empty else ""

    def _topic_label(self, tid):
        if tid in self._labels:
            return self._labels[tid]
        return "Outliers" if tid == -1 else f"Topic {tid}"

    def _colour(self, tid):
        if tid == -1:
            return "#374151"
        ids = sorted(t for t in set(self._topics) if t != -1 and t not in self._hidden)
        try:
            return TOPIC_COLOURS[ids.index(tid) % len(TOPIC_COLOURS)]
        except ValueError:
            return "#64748b"

    # ── Screen 1: Overview ────────────────────────────────────────────────────

    def _build_overview(self):
        frame = ctk.CTkFrame(self._content, fg_color=C.BG)

        # Info
        info = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        info.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(info, text="Topics Overview",
                     font=("Segoe UI", 14, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=16, pady=(10, 4))
        ctk.CTkLabel(info,
                     text="Each row is a topic discovered by the model. "
                          "Select one or more rows, then use the buttons below "
                          "to merge duplicates, remove junk topics, or add a custom label. "
                          "Double-click a row to jump to its documents.",
                     font=("Segoe UI", 11), text_color=C.MUTED,
                     wraplength=1100, justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 10))

        # Table
        tv_frame = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        tv_frame.pack(fill="both", expand=True, padx=16, pady=(0, 4))

        cols = ("topic", "count", "pct", "keywords")
        self._ov_tv = tv = ttk.Treeview(tv_frame, show="headings", columns=cols,
                                        selectmode="extended", height=20)
        _style_treeview(tv)
        tv.heading("topic", text="Topic")
        tv.heading("count", text="Docs")
        tv.heading("pct",   text="Share")
        tv.heading("keywords", text="Key Terms")
        tv.column("topic",    width=120, minwidth=80, stretch=False)
        tv.column("count",    width=70,  minwidth=60, stretch=False)
        tv.column("pct",      width=70,  minwidth=60, stretch=False)
        tv.column("keywords", width=800, minwidth=300)

        ysb = ttk.Scrollbar(tv_frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=ysb.set)
        ysb.pack(side="right", fill="y", padx=(0, 4), pady=4)
        tv.pack(fill="both", expand=True, padx=4, pady=4)

        total = self._n_docs
        for _, row in self._topic_info.iterrows():
            tid = row["Topic"]
            if tid in self._hidden:
                continue
            count = row["Count"]
            tv.insert("", "end",
                      values=(self._topic_label(tid), f"{count:,}",
                              f"{count/total*100:.1f}%", self._kw(row, 10)),
                      tags=(str(tid),))

        tv.bind("<Double-1>", lambda e: self._jump_to_docs())

        # Action buttons
        ab = ctk.CTkFrame(frame, fg_color=C.PANEL, corner_radius=8)
        ab.pack(fill="x", padx=16, pady=(4, 12))
        ai = ctk.CTkFrame(ab, fg_color="transparent")
        ai.pack(fill="x", padx=12, pady=8)

        for text, cmd, tt in [
            ("Merge Selected", self._merge_topics,
             "Merge 2+ selected topics into one.\nUseful for near-duplicate topics."),
            ("Remove Selected", self._delete_topics,
             "Hide selected topics from results.\nTheir docs become outliers."),
            ("Rename Selected", self._rename_topic,
             "Give a human-readable label to the selected topic."),
        ]:
            b = ctk.CTkButton(ai, text=text, fg_color=C.BTN, width=150, height=32,
                              command=cmd)
            b.pack(side="left", padx=(0, 8))
            tip(b, tt)

        ctk.CTkLabel(ai,
                     text="Tip: Hold Ctrl/Shift to select multiple rows for merging",
                     font=("Segoe UI", 10), text_color=C.MUTED
                     ).pack(side="left", padx=(16, 0))

        return frame

    def _jump_to_docs(self):
        sel = self._ov_tv.selection()
        if sel:
            tid = int(self._ov_tv.item(sel[0])["tags"][0])
            self._preselect_topic = tid
            if "documents" in self._cache:
                self._cache["documents"].destroy()
                del self._cache["documents"]
            self._switch("documents")

    def _merge_topics(self):
        sel = self._ov_tv.selection()
        if len(sel) < 2:
            messagebox.showinfo("Merge", "Select at least 2 topics to merge.")
            return
        tids = [int(self._ov_tv.item(s)["tags"][0]) for s in sel]
        tids = [t for t in tids if t != -1]
        if len(tids) < 2:
            messagebox.showinfo("Merge", "Cannot merge outliers. Select 2+ real topics.")
            return
        labels = [self._topic_label(t) for t in tids]
        if not messagebox.askyesno("Merge Topics",
                                   f"Merge these topics into one?\n\n{', '.join(labels)}"):
            return
        self._tm.merge_topics(self._texts, [tids])
        self._topics = list(self._tm.topics_)
        self._invalidate_screens()

    def _delete_topics(self):
        sel = self._ov_tv.selection()
        if not sel:
            messagebox.showinfo("Remove", "Select topics to remove."); return
        tids = [int(self._ov_tv.item(s)["tags"][0]) for s in sel]
        tids = [t for t in tids if t != -1]
        if not tids:
            messagebox.showinfo("Remove", "Cannot remove the outlier group."); return
        labels = [self._topic_label(t) for t in tids]
        if not messagebox.askyesno("Remove Topics",
                                   f"Remove these topics?\nTheir documents become outliers.\n\n"
                                   f"{', '.join(labels)}"):
            return
        for t in tids:
            self._hidden.add(t)
        self._topics = [-1 if t in self._hidden else t for t in self._topics]
        self._invalidate_screens()

    def _rename_topic(self):
        sel = self._ov_tv.selection()
        if len(sel) != 1:
            messagebox.showinfo("Rename", "Select exactly one topic."); return
        tid = int(self._ov_tv.item(sel[0])["tags"][0])
        current = self._topic_label(tid)
        new = simpledialog.askstring("Rename Topic", f"New label for '{current}':",
                                     initialvalue=current, parent=self)
        if new and new.strip():
            self._labels[tid] = new.strip()
            self._invalidate_screens()

    # ── Screen 2: Topic Map ───────────────────────────────────────────────────

    def _build_map(self):
        frame = ctk.CTkFrame(self._content, fg_color=C.BG)

        info = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        info.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(info, text="Topic Map",
                     font=("Segoe UI", 14, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=16, pady=(10, 4))
        ctk.CTkLabel(info,
                     text="Each dot is a document positioned by UMAP so that semantically "
                          "similar texts sit close together. Colours represent topics. Tight, "
                          "well-separated clusters indicate strong themes. Overlap suggests "
                          "shared vocabulary. Grey dots are outliers. Use the toolbar at the "
                          "bottom to zoom and pan.",
                     font=("Segoe UI", 11), text_color=C.MUTED,
                     wraplength=1100, justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 10))

        chart = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        chart.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        C.ensure_mpl()
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        fig.patch.set_facecolor(C.CARD); ax.set_facecolor(C.CARD)
        for s in ax.spines.values(): s.set_visible(False)

        arr = np.array(self._topics)
        emb = self._emb_2d
        unique = sorted(set(self._topics))

        if -1 in unique:
            m = arr == -1
            ax.scatter(emb[m, 0], emb[m, 1], c="#374151", s=10, alpha=0.15,
                       label="Outliers", edgecolors="none", zorder=1)

        for tid in unique:
            if tid == -1 or tid in self._hidden: continue
            m = arr == tid
            col = self._colour(tid)
            kw = self._short_kw(tid)
            lab = f"{self._topic_label(tid)}: {kw}" if kw else self._topic_label(tid)
            if len(lab) > 35: lab = lab[:33] + "…"
            ax.scatter(emb[m, 0], emb[m, 1], c=col, s=18, alpha=0.6,
                       label=lab, edgecolors="none", zorder=2)
            cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
            ax.annotate(self._topic_label(tid), (cx, cy), fontsize=8,
                        fontweight="bold", color=col, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc=C.CARD,
                                  ec=col, alpha=0.85), zorder=3)

        ax.set_xticks([]); ax.set_yticks([])
        if self._n_topics <= 20:
            ncol = 1 if self._n_topics <= 10 else 2
            ax.legend(loc="upper right", fontsize=8, ncol=ncol,
                      framealpha=0.9, facecolor=C.PANEL,
                      edgecolor=C.BORDER, labelcolor=C.TEXT, markerscale=1.8)
        fig.tight_layout(pad=0.8)

        canvas = FigureCanvasTkAgg(fig, master=chart)
        canvas.draw()
        tb_frame = ctk.CTkFrame(chart, fg_color=C.PANEL, height=36)
        tb_frame.pack(fill="x", side="bottom")
        toolbar = NavigationToolbar2Tk(canvas, tb_frame); toolbar.update()
        try:
            toolbar.configure(background=C.PANEL)
            for w in toolbar.winfo_children():
                w.configure(background=C.PANEL)
        except Exception: pass
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._map_canvas, self._map_fig = canvas, fig
        self._figures.append(fig)
        return frame

    # ── Screen 3: Keywords ────────────────────────────────────────────────────

    def _build_keywords(self):
        frame = ctk.CTkFrame(self._content, fg_color=C.BG)

        info = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        info.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(info, text="Topic Keywords",
                     font=("Segoe UI", 14, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=16, pady=(10, 4))
        ctk.CTkLabel(info,
                     text="Horizontal bar charts showing the most important words per topic. "
                          "Longer bars = more representative. Compare across topics to "
                          "understand what makes each theme distinct. If stopwords appear, "
                          "add them to your custom stopwords list and re-run.",
                     font=("Segoe UI", 11), text_color=C.MUTED,
                     wraplength=1100, justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 10))

        chart_outer = ctk.CTkFrame(frame, fg_color=C.BG)
        chart_outer.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        C.ensure_mpl()
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        real = [r for _, r in self._topic_info.iterrows()
                if r["Topic"] != -1 and r["Topic"] not in self._hidden]
        n = len(real)
        if n == 0:
            ctk.CTkLabel(chart_outer, text="No topics found.",
                         text_color=C.MUTED).pack(pady=40)
            return frame

        ncols = min(3, n); nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(4, nrows * 2.8)), dpi=95)
        fig.patch.set_facecolor(C.BG)
        if n == 1: axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, row in enumerate(real):
            ax = axes[i]; tid = row["Topic"]
            col = self._colour(tid)
            tw = self._tm.get_topic(tid)
            if not tw: ax.set_visible(False); continue
            words  = [w for w, _ in tw[:10]][::-1]
            scores = [s for _, s in tw[:10]][::-1]
            ax.barh(words, scores, color=col, height=0.7, alpha=0.85)
            ax.set_title(f"{self._topic_label(tid)}  ({row['Count']:,} docs)",
                         fontsize=10, fontweight="bold", color=C.TEXT, pad=8)
            ax.set_facecolor(C.CARD)
            ax.tick_params(axis="y", labelsize=9, colors=C.TEXT)
            ax.tick_params(axis="x", labelsize=7, colors=C.MUTED)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(C.BORDER); ax.spines["left"].set_color(C.BORDER)

        for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        fig.tight_layout(pad=1.5)

        cf = ctk.CTkFrame(chart_outer, fg_color=C.CARD, corner_radius=8)
        cf.pack(fill="both", expand=True)
        canvas = FigureCanvasTkAgg(fig, master=cf); canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._kw_canvas, self._kw_fig = canvas, fig
        self._figures.append(fig)
        return frame

    # ── Screen 4: Similarity ──────────────────────────────────────────────────

    def _build_similarity(self):
        frame = ctk.CTkFrame(self._content, fg_color=C.BG)

        info = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        info.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(info, text="Topic Similarity",
                     font=("Segoe UI", 14, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=16, pady=(10, 4))
        ctk.CTkLabel(info,
                     text="This heatmap shows how similar each pair of topics is, "
                          "measured by cosine similarity of their average document "
                          "embeddings. Bright cells mean high similarity — those topics "
                          "might be candidates for merging. Use this to decide which "
                          "topics overlap and should be combined.",
                     font=("Segoe UI", 11), text_color=C.MUTED,
                     wraplength=1100, justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 10))

        chart = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        chart.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        C.ensure_mpl()
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from sklearn.metrics.pairwise import cosine_similarity

        tids = sorted(t for t in set(self._topics) if t != -1 and t not in self._hidden)
        if len(tids) < 2:
            ctk.CTkLabel(chart, text="Need at least 2 topics for similarity.",
                         text_color=C.MUTED).pack(pady=40)
            return frame

        arr = np.array(self._topics)
        centroids = []
        for tid in tids:
            m = arr == tid
            centroids.append(self._embeddings[m].mean(axis=0))
        centroids = np.array(centroids)
        sim = cosine_similarity(centroids)

        n = len(tids)
        fig_size = max(6, min(12, n * 0.6 + 3))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=95)
        fig.patch.set_facecolor(C.BG); ax.set_facecolor(C.CARD)

        im = ax.imshow(sim, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

        labels = [self._topic_label(t) for t in tids]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color=C.TEXT)
        ax.set_yticklabels(labels, fontsize=9, color=C.TEXT)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = sim[i, j]
                colour = "white" if val > 0.6 else C.TEXT
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=colour)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cosine Similarity", color=C.TEXT, fontsize=10)
        cbar.ax.tick_params(colors=C.MUTED)

        ax.set_title("Topic Similarity Matrix", fontsize=13, fontweight="bold",
                     color=C.TEXT, pad=12)
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=chart); canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._sim_canvas, self._sim_fig = canvas, fig
        self._figures.append(fig)
        return frame

    # ── Screen 5: Document Explorer ───────────────────────────────────────────

    def _build_documents(self):
        frame = ctk.CTkFrame(self._content, fg_color=C.BG)

        info = ctk.CTkFrame(frame, fg_color=C.CARD, corner_radius=8)
        info.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(info, text="Document Explorer",
                     font=("Segoe UI", 14, "bold"), text_color=C.TEXT
                     ).pack(anchor="w", padx=16, pady=(10, 4))
        ctk.CTkLabel(info,
                     text="Browse the actual documents assigned to each topic. "
                          "Use this to verify whether topics make sense and to "
                          "understand what kind of content each theme captures. "
                          "Select a topic from the dropdown to see its documents.",
                     font=("Segoe UI", 11), text_color=C.MUTED,
                     wraplength=1100, justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 10))

        # Topic selector
        sel_row = ctk.CTkFrame(frame, fg_color=C.PANEL, corner_radius=8)
        sel_row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(sel_row, text="Select topic:", text_color=C.MUTED,
                     font=("Segoe UI", 12)).pack(side="left", padx=(16, 8), pady=10)

        choices, ids = [], []
        for _, row in self._topic_info.iterrows():
            tid = row["Topic"]
            if tid in self._hidden: continue
            cnt = row["Count"]
            kw = self._kw(row, 4)
            lab = f"{self._topic_label(tid)}  ({cnt:,} docs)"
            if kw: lab += f"  —  {kw}"
            choices.append(lab); ids.append(tid)

        self._doc_ids = ids
        self._doc_sel = tk.StringVar(value=choices[0] if choices else "")

        # Pre-select from overview
        if hasattr(self, "_preselect_topic"):
            tid = self._preselect_topic
            del self._preselect_topic
            if tid in ids:
                self._doc_sel.set(choices[ids.index(tid)])

        ctk.CTkOptionMenu(sel_row, variable=self._doc_sel, values=choices,
                          width=600, command=lambda _: self._load_docs()
                          ).pack(side="left", padx=(0, 16), pady=10)

        self._doc_count = ctk.CTkLabel(sel_row, text="", text_color=C.MUTED,
                                       font=("Segoe UI", 11))
        self._doc_count.pack(side="left")

        self._doc_list = ctk.CTkScrollableFrame(frame, fg_color=C.CARD, corner_radius=8)
        self._doc_list.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self._load_docs()
        return frame

    def _load_docs(self):
        for w in self._doc_list.winfo_children(): w.destroy()

        sel = self._doc_sel.get()
        idx = 0
        for i, (_, row) in enumerate(self._topic_info.iterrows()):
            tid = row["Topic"]
            if tid in self._hidden: continue
            cnt = row["Count"]
            kw = self._kw(row, 4)
            lab = f"{self._topic_label(tid)}  ({cnt:,} docs)"
            if kw: lab += f"  —  {kw}"
            if lab == sel: idx = i; break

        tid = self._doc_ids[idx] if idx < len(self._doc_ids) else -1
        indices = [i for i, t in enumerate(self._topics) if t == tid]
        total = len(indices)
        show = min(50, total)
        examples = [self._texts[i] for i in indices[:show]]

        self._doc_count.configure(text=f"Showing {show} of {total:,} documents")

        if tid != -1:
            kw = self._kw(self._topic_info[self._topic_info["Topic"] == tid].iloc[0], 10)
            if kw:
                kf = ctk.CTkFrame(self._doc_list, fg_color=C.PANEL, corner_radius=6)
                kf.pack(fill="x", padx=4, pady=(4, 8))
                ctk.CTkLabel(kf, text=f"Key terms:  {kw}",
                             font=("Segoe UI", 11, "bold"), text_color=C.ACCENT
                             ).pack(anchor="w", padx=12, pady=8)

        for i, text in enumerate(examples):
            card = ctk.CTkFrame(self._doc_list, fg_color=C.PANEL, corner_radius=6)
            card.pack(fill="x", padx=4, pady=2)
            h = ctk.CTkFrame(card, fg_color="transparent"); h.pack(fill="x", padx=12, pady=(6, 0))
            ctk.CTkLabel(h, text=f"Document {i+1}", font=("Segoe UI", 10, "bold"),
                         text_color=C.MUTED).pack(side="left")
            ctk.CTkLabel(h, text=f"{len(text):,} chars", font=("Segoe UI", 9),
                         text_color=C.MUTED).pack(side="right")
            display = text[:500] + ("…" if len(text) > 500 else "")
            ctk.CTkLabel(card, text=display, font=("Segoe UI", 10),
                         text_color=C.TEXT, anchor="w", wraplength=1100,
                         justify="left").pack(anchor="w", padx=12, pady=(2, 8))

        if total > show:
            ctk.CTkLabel(self._doc_list,
                         text=f"… and {total - show:,} more not shown",
                         font=("Segoe UI", 10), text_color=C.MUTED
                         ).pack(pady=(8, 4))
