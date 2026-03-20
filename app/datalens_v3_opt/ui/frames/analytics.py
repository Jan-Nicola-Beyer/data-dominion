"""Analytics frame — interactive charts with click-to-filter, exports, and data tables."""

from __future__ import annotations
from typing import TYPE_CHECKING
import math
import re
import ast
from collections import Counter

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip, _style_treeview

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App

# Lazy-loaded on first chart render
plt = None
FigureCanvasTkAgg = None
SpanSelector = None
num2date = None

BIN_OPTIONS = {"Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "MS"}


def _ensure_mpl():
    """Lazy-load matplotlib globals on first chart render."""
    global plt, FigureCanvasTkAgg, SpanSelector, num2date
    if plt is not None:
        return
    C.ensure_mpl()
    import matplotlib.pyplot as _plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FCA
    from matplotlib.widgets import SpanSelector as _SS
    from matplotlib.dates import num2date as _n2d
    plt = _plt
    FigureCanvasTkAgg = _FCA
    SpanSelector = _SS
    num2date = _n2d


class AnalyticsFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._figures: list = []
        self._span_selectors: list = []
        self._chart_filter: dict | None = None
        self._pins: list[str] = []
        self._time_bin = tk.StringVar(value="Weekly")
        self._pin_var = tk.StringVar()
        self._candidates: list = []   # [(title, icon, desc, draw_fn), ...]
        self._active_btns: dict = {}  # title -> button widget
        self._build()

    def _build(self):
        # Title bar
        tb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=48)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="Analytics", font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(side="left", padx=16, pady=14)
        ctk.CTkButton(tb, text="Refresh", width=100, fg_color=C.BTN,
                      command=self.refresh).pack(side="right", padx=12)

        # Controls bar: time binning + event pins
        ctrl = ctk.CTkFrame(self, fg_color=C.CARD, corner_radius=0, height=38)
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)

        tl = ctk.CTkLabel(ctrl, text="Time granularity:", text_color=C.MUTED,
                          font=("Segoe UI", 10))
        tl.pack(side="left", padx=(12, 4))
        tip(tl, "Controls the X-axis binning for all time-series charts.\n"
                "Hourly is useful for breaking-news events;\n"
                "Monthly gives a long-term overview.")
        ctk.CTkOptionMenu(ctrl, variable=self._time_bin,
                          values=list(BIN_OPTIONS.keys()), width=100,
                          command=lambda _: self._on_time_bin_change()
                          ).pack(side="left", padx=(0, 16))

        pl = ctk.CTkLabel(ctrl, text="Pin event date:", text_color=C.MUTED,
                          font=("Segoe UI", 10))
        pl.pack(side="left", padx=(0, 4))
        tip(pl, "Add a vertical dashed line to all time-series charts.\n"
                "Useful for marking events (elections, product launches, etc).")
        ctk.CTkEntry(ctrl, textvariable=self._pin_var, width=100,
                     placeholder_text="YYYY-MM-DD"
                     ).pack(side="left", padx=(0, 4))
        ab = ctk.CTkButton(ctrl, text="+ Pin", width=55, height=26,
                           fg_color=C.BTN, font=("Segoe UI", 9),
                           command=self._add_pin)
        ab.pack(side="left")
        tip(ab, "Add this date as a vertical marker on time charts.")

        self._pin_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        self._pin_frame.pack(side="left", padx=8)

        # Filter badge (hidden by default)
        self._badge_frame = ctk.CTkFrame(self, fg_color=C.SELECT,
                                         corner_radius=0, height=32)
        self._badge_label = ctk.CTkLabel(self._badge_frame, text="",
                                         text_color=C.TEXT,
                                         font=("Segoe UI", 11))
        self._badge_label.pack(side="left", padx=12)
        ctk.CTkButton(self._badge_frame, text="Clear filter", width=90,
                      height=24, fg_color=C.BTN, font=("Segoe UI", 9),
                      command=self._clear_chart_filter
                      ).pack(side="left", padx=(0, 12))
        self._badge_frame.pack_forget()

        # Scrollable content area
        self._canvas_frame = ctk.CTkScrollableFrame(self, fg_color=C.BG)
        self._canvas_frame.pack(fill="both", expand=True)

        # Summary cards row
        self._stat_row = ctk.CTkFrame(self._canvas_frame, fg_color="transparent")
        self._stat_row.pack(fill="x", padx=8, pady=6)

        # Chart selector buttons
        self._selector_frame = ctk.CTkFrame(self._canvas_frame,
                                            fg_color="transparent")
        self._selector_frame.pack(fill="x", padx=8, pady=(0, 4))

        # Chart display area
        self._chart_area = ctk.CTkFrame(self._canvas_frame,
                                        fg_color="transparent")
        self._chart_area.pack(fill="both", expand=True, padx=4)

    # ── Time bin change (re-render visible charts, not full refresh) ──────

    def _on_time_bin_change(self):
        """When time granularity changes, refresh available charts + re-render any visible chart."""
        self.refresh()

    # ── Refresh ──────────────────────────────────────────────────────────────

    def refresh(self):
        # Close old matplotlib figures
        if plt is not None:
            for fig in self._figures:
                try:
                    plt.close(fig)
                except Exception:
                    pass
        self._figures.clear()
        self._span_selectors.clear()

        # Clear content areas
        for w in self._stat_row.winfo_children():
            w.destroy()
        for w in self._selector_frame.winfo_children():
            w.destroy()
        for w in self._chart_area.winfo_children():
            w.destroy()
        self._active_btns.clear()

        df = self.app.filtered_df
        if df.empty:
            ctk.CTkLabel(self._stat_row,
                         text="Load a dataset and click Refresh to see analytics.",
                         text_color=C.MUTED, font=("Segoe UI", 13)).pack(pady=40)
            return

        # Apply chart-level filter
        if self._chart_filter:
            df = self._apply_local_filter(df)
            self._show_badge()
        else:
            self._badge_frame.pack_forget()

        self._draw_summary_cards(df)
        self._build_chart_selector(df)

    # ── Chart-level filter ───────────────────────────────────────────────

    def _apply_local_filter(self, df):
        cf = self._chart_filter
        if cf.get("type") == "date":
            try:
                if "_date" in df.columns:
                    df = df[df["_date"] >= pd.Timestamp(cf["from"], tz="UTC")]
                    df = df[df["_date"] <= pd.Timestamp(cf["to"], tz="UTC")]
            except Exception:
                pass
        else:
            col = cf.get("col")
            val = cf.get("val")
            if col and col in df.columns:
                df = df[df[col].astype(str) == str(val)]
        return df

    def _show_badge(self):
        cf = self._chart_filter
        if cf.get("type") == "date":
            text = f"Filtered by date: {cf['from']}  to  {cf['to']}"
        else:
            text = f"Filtered by: {cf.get('col')} = {cf.get('val')}"
        self._badge_label.configure(text=text)
        self._badge_frame.pack(fill="x", before=self._canvas_frame)

    def _clear_chart_filter(self):
        self._chart_filter = None
        self.refresh()

    # ── Summary cards ────────────────────────────────────────────────────

    def _draw_summary_cards(self, df):
        stats = [("Rows", f"{len(df):,}"),
                 ("Columns", len([c for c in df.columns
                                  if not c.startswith("_")]))]

        src_col = ("_source_dataset" if "_source_dataset" in df.columns
                   else self._col("platform", df))
        if src_col and src_col in df.columns:
            stats.append(("Sources", df[src_col].nunique()))

        if "_date" in df.columns:
            valid = df["_date"].dropna()
            if len(valid):
                stats.append(("Date range",
                              f"{valid.min().date()} – {valid.max().date()}"))

        # ── Missingness score ────────────────────────────────────────────────
        key_fields = ["content_text", "language", "platform",
                      "author_username", "like_count", "sentiment"]
        missing_parts = []
        for canonical in key_fields:
            col = self._col(canonical, df)
            if col:
                pct = df[col].isna().mean() * 100
                if pct > 5:
                    missing_parts.append(f"{canonical}: {pct:.0f}%")
        if missing_parts:
            stats.append(("Missing Data", "\n".join(missing_parts[:3])))
        else:
            stats.append(("Data Quality", "Good"))

        # ── Bot / high-volume signal ─────────────────────────────────────
        auth_col = self._col("author_username", df)
        if auth_col:
            counts = df[auth_col].value_counts()
            if len(counts) >= 10:
                top1pct = max(1, int(len(counts) * 0.01))
                top_volume = counts.head(top1pct).sum()
                share = top_volume / len(df) * 100
                stats.append(("Top 1% Authors",
                              f"{share:.0f}% of posts"))

        for label, val in stats:
            card = ctk.CTkFrame(self._stat_row, fg_color=C.CARD,
                                corner_radius=8)
            card.pack(side="left", padx=6)
            val_str = str(val)
            if "\n" in val_str:
                ctk.CTkLabel(card, text=val_str, font=("Segoe UI", 9),
                             text_color=C.ACCENT, justify="left"
                             ).pack(padx=10, pady=(8, 0))
            else:
                ctk.CTkLabel(card, text=val_str, font=("Segoe UI", 18, "bold"),
                             text_color=C.ACCENT).pack(padx=10, pady=(8, 0))
            ctk.CTkLabel(card, text=label, text_color=C.MUTED,
                         font=("Segoe UI", 10)).pack(padx=10, pady=(0, 8))

    # ── Helpers ──────────────────────────────────────────────────────────

    def _col(self, canonical, df):
        """Resolve a canonical column name to an actual column in *df*."""
        if canonical in df.columns:
            return canonical
        mapped = self.app.col_map.get(canonical)
        return mapped if mapped and mapped in df.columns else None

    def _time_freq(self):
        return BIN_OPTIONS.get(self._time_bin.get(), "W")

    def _freq_label(self):
        return self._time_bin.get().lower()

    # ── Chart selector — build buttons for available charts ──────────────

    def _cat_score(self, series):
        clean = series.dropna().astype(str)
        clean = clean[~clean.str.lower().isin(["nan", "none", ""])]
        if clean.nunique() < 2:
            return 0, None
        vc = clean.value_counts(normalize=True)
        if vc.iloc[0] > 0.97:
            return 0, None
        entropy = -sum(p * math.log(p + 1e-10) for p in vc)
        score = min(100, entropy * 22 + clean.nunique() * 3)
        return score, clean.value_counts()

    def _build_chart_selector(self, df: pd.DataFrame):
        """Scan the dataset and create a button for each available chart."""
        self._candidates.clear()
        freq_lbl = self._freq_label()

        # 1 — Volume over time
        if "_date" in df.columns:
            dated = df.dropna(subset=["_date"])
            if len(dated) > 10:
                desc = f"Post volume ({freq_lbl}) — {len(dated):,} posts."
                self._candidates.append((
                    "Volume Over Time", "◷",  desc,
                    lambda _df=df: self._render_chart(
                        lambda p: self._chart_volume(p, _df))))

        # 2 — Platform / source
        src = (self._col("platform", df) or
               ("_source_dataset" if "_source_dataset" in df.columns
                else None))
        if src:
            sc, vc = self._cat_score(df[src])
            if sc > 0:
                desc = f"{vc.nunique()} platform(s). Click a bar to filter."
                self._candidates.append((
                    "Platform / Source", "▥", desc,
                    lambda _df=df, _c=src: self._render_chart(
                        lambda p: self._chart_bar(p, _df, _c, "Posts by Platform / Source"))))

        # 3 — Sentiment
        sent = self._col("sentiment", df)
        if sent:
            sc, vc = self._cat_score(df[sent])
            if sc > 0:
                desc = f"Sentiment across {int(df[sent].notna().sum()):,} posts."
                self._candidates.append((
                    "Sentiment", "◕", desc,
                    lambda _df=df, _c=sent: self._render_chart(
                        lambda p: self._chart_pie(p, _df, _c, "Sentiment Distribution"))))

        # 4 — Avg engagement over time
        eng = self._col("like_count", df) or self._col("engagement_total", df)
        if eng and "_date" in df.columns:
            num = pd.to_numeric(df[eng], errors="coerce").dropna()
            if len(num) > 10 and num.mean() > 0:
                label = eng.replace("_count", "").replace("_", " ")
                desc = f"Average {label} per {freq_lbl}."
                self._candidates.append((
                    f"Engagement ({label})", "◈", desc,
                    lambda _df=df, _e=eng: self._render_chart(
                        lambda p: self._chart_engagement(p, _df, _e))))

        # 5 — Engagement by platform
        like = self._col("like_count", df)
        cmt = self._col("comment_count", df)
        shr = self._col("share_count", df)
        vw = self._col("view_count", df)
        eng_cols = [c for c in [like, cmt, shr, vw] if c]
        if src and eng_cols and df[src].nunique() >= 2:
            desc = "Average engagement compared across sources."
            self._candidates.append((
                "Engagement by Platform", "▦", desc,
                lambda _df=df, _src=src, _ec=eng_cols: self._render_chart(
                    lambda p: self._chart_eng_by_platform(p, _df, _src, _ec))))

        # 6 — Top hashtags
        ht = self._col("hashtags", df)
        if ht:
            has_data = df[ht].dropna().astype(str)
            has_data = has_data[~has_data.str.lower().isin(["nan", "[]", ""])]
            if len(has_data) > 5:
                desc = f"Most frequent hashtags from {len(has_data):,} posts."
                self._candidates.append((
                    "Top Hashtags", "#", desc,
                    lambda _df=df, _h=ht: self._render_chart(
                        lambda p: self._chart_hashtags(p, _df, _h))))

        # 7 — Language
        lang = self._col("language", df)
        if lang:
            sc, vc = self._cat_score(df[lang])
            if sc > 0:
                desc = f"{vc.nunique()} languages detected."
                self._candidates.append((
                    "Language Breakdown", "⊞", desc,
                    lambda _df=df, _c=lang: self._render_chart(
                        lambda p: self._chart_bar(p, _df, _c, "Language Breakdown", 12))))

        # 8 — Media type
        med = self._col("media_type", df)
        if med:
            sc, vc = self._cat_score(df[med])
            if sc > 0:
                desc = f"{vc.nunique()} media types."
                self._candidates.append((
                    "Media Type", "◫", desc,
                    lambda _df=df, _c=med: self._render_chart(
                        lambda p: self._chart_bar(p, _df, _c, "Media Type Breakdown"))))

        # 9 — Top authors
        auth = self._col("author_username", df)
        if auth and df[auth].nunique() >= 4:
            desc = f"Top authors ({df[auth].nunique():,} unique)."
            self._candidates.append((
                "Top Authors", "◉", desc,
                lambda _df=df, _c=auth: self._render_chart(
                    lambda p: self._chart_bar(p, _df, _c, "Top Authors", 15))))

        # 10 — Sentiment by Platform
        if sent and src and df[src].nunique() >= 2:
            sc2, _ = self._cat_score(df[sent])
            if sc2 > 0:
                desc = "How sentiment varies across platforms."
                self._candidates.append((
                    "Sentiment by Platform", "◑", desc,
                    lambda _df=df, _s=sent, _p=src: self._render_chart(
                        lambda p: self._chart_sentiment_platform(p, _df, _s, _p))))

        # 11 — Hashtag co-occurrence
        if ht:
            tag_lists = self._parse_hashtags(df, ht)
            if len(tag_lists) > 20:
                desc = "Which hashtags frequently appear together."
                self._candidates.append((
                    "Hashtag Co-occurrence", "⊡", desc,
                    lambda _df=df, _h=ht: self._render_chart(
                        lambda p: self._chart_cooccurrence(p, _df, _h))))

        # 12 — Engagement vs Sentiment
        eng_col = (self._col("like_count", df)
                   or self._col("engagement_total", df))
        if eng_col and sent:
            desc = "Does sentiment correlate with engagement?"
            self._candidates.append((
                "Engagement vs Sentiment", "⧉", desc,
                lambda _df=df, _e=eng_col, _s=sent: self._render_chart(
                    lambda p: self._chart_eng_sentiment(p, _df, _e, _s))))

        # Build the button panel
        if not self._candidates:
            ctk.CTkLabel(
                self._selector_frame,
                text="No charts available for this dataset.\n"
                     "Try loading data with dates, platform, sentiment, "
                     "or hashtag columns.",
                text_color=C.MUTED, font=("Segoe UI", 12), wraplength=500
            ).pack(pady=20, padx=20)
            return

        # Header
        ctk.CTkLabel(self._selector_frame,
                     text="Select a chart to display:",
                     text_color=C.MUTED, font=("Segoe UI", 11)
                     ).pack(anchor="w", padx=4, pady=(4, 6))

        # Button grid
        btn_grid = ctk.CTkFrame(self._selector_frame, fg_color="transparent")
        btn_grid.pack(fill="x")

        for title, icon, desc, draw_fn in self._candidates:
            btn = ctk.CTkButton(
                btn_grid, text=f"{icon}  {title}",
                fg_color=C.BTN, hover_color=C.SELECT,
                text_color=C.TEXT, font=("Segoe UI", 11),
                height=34, corner_radius=6,
                command=lambda t=title, fn=draw_fn: self._on_chart_btn(t, fn))
            btn.pack(side="left", padx=3, pady=3)
            tip(btn, desc)
            self._active_btns[title] = btn

    def _on_chart_btn(self, title: str, draw_fn):
        """Handle a chart button click — highlight it and render."""
        # Highlight the active button
        for t, btn in self._active_btns.items():
            btn.configure(fg_color=C.ACCENT if t == title else C.BTN)
        draw_fn()

    def _render_chart(self, draw_fn):
        """Clear the chart area, lazy-load matplotlib, and call draw_fn."""
        _ensure_mpl()

        # Close old figures
        for fig in self._figures:
            try:
                plt.close(fig)
            except Exception:
                pass
        self._figures.clear()
        self._span_selectors.clear()

        # Clear chart area
        for w in self._chart_area.winfo_children():
            w.destroy()

        try:
            draw_fn(self._chart_area)
        except Exception as exc:
            ctk.CTkLabel(self._chart_area,
                         text=f"Chart error: {exc}",
                         text_color=C.DANGER, font=("Segoe UI", 11)
                         ).pack(pady=20, padx=20)
            print(f"[Analytics] chart error: {exc}")

    # ── Chart renderers ──────────────────────────────────────────────────

    def _chart_volume(self, parent, df):
        dated = df.dropna(subset=["_date"])
        if dated.empty:
            return
        freq = self._time_freq()
        s = dated.set_index("_date").resample(freq).size()
        if s.empty:
            return
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.fill_between(s.index, s.values, alpha=0.25, color=C.ISD_RED)
        ax.plot(s.index, s.values, color=C.ISD_RED, linewidth=1.8)
        ax.set_title(f"Post Volume ({self._freq_label()})", fontsize=10)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3)
        self._draw_pins(ax)
        fig.tight_layout()

        data_df = s.reset_index()
        data_df.columns = ["Period", "Count"]
        self._embed(parent, fig, "Volume Over Time",
                    data_df=data_df, time_chart=True)

    def _chart_bar(self, parent, df, col, title, top=20):
        counts = df[col].dropna().astype(str)
        counts = counts[counts.str.lower() != "nan"].value_counts().head(top)
        if counts.empty:
            return
        sorted_c = counts.sort_values()

        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(range(len(sorted_c)), sorted_c.values,
                       color=C.ISD_PURPLE)
        ax.set_yticks(range(len(sorted_c)))
        ax.set_yticklabels(sorted_c.index, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Posts")

        categories = list(sorted_c.index)
        for i, bar in enumerate(bars):
            bar.set_picker(True)
            bar.set_gid(categories[i])

        fig.tight_layout()

        data_df = counts.reset_index()
        data_df.columns = [col, "Count"]
        self._embed(parent, fig, title, data_df=data_df, click_col=col)

    def _chart_pie(self, parent, df, col, title):
        counts = df[col].dropna().astype(str)
        counts = counts[counts.str.lower() != "nan"].value_counts()
        if counts.empty:
            return
        SENT_COLORS = {"positive": "#22c55e", "neutral": "#6b7280",
                       "negative": C.ISD_RED, "mixed": C.ISD_CORAL}
        colors = [SENT_COLORS.get(k.lower(), C.ISD_BLUE) for k in counts.index]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            autopct="%1.0f%%", colors=colors, startangle=90)

        txt_color = "#e6edf3" if C.CURRENT_THEME == "dark" else "#111827"
        for at in autotexts:
            at.set_color(txt_color)
            at.set_fontsize(9)

        for i, wedge in enumerate(wedges):
            wedge.set_picker(True)
            wedge.set_gid(str(counts.index[i]))

        ax.set_title(title, fontsize=10)
        fig.tight_layout()

        data_df = counts.reset_index()
        data_df.columns = [col, "Count"]
        self._embed(parent, fig, title, data_df=data_df, click_col=col)

    def _chart_engagement(self, parent, df, col):
        if "_date" not in df.columns:
            return
        tmp = df[["_date", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=["_date", col])
        if tmp.empty:
            return
        freq = self._time_freq()
        series = tmp.set_index("_date")[col].resample(freq).mean()

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.fill_between(series.index, series.values, alpha=0.2, color=C.ISD_BLUE)
        ax.plot(series.index, series.values, color=C.ISD_BLUE, linewidth=1.8)
        ax.set_title(f"Avg {col} ({self._freq_label()})", fontsize=10)
        ax.set_xlabel("")
        self._draw_pins(ax)
        fig.tight_layout()

        data_df = series.reset_index()
        data_df.columns = ["Period", f"Avg {col}"]
        self._embed(parent, fig, f"Engagement {col}",
                    data_df=data_df, time_chart=True)

    def _chart_eng_by_platform(self, parent, df, src_col, eng_cols):
        tmp = df[[src_col] + eng_cols].copy()
        for c in eng_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        grouped = tmp.groupby(src_col)[eng_cols].mean().dropna(how="all")
        if grouped.empty:
            return
        grouped.columns = [c.replace("_count", "").replace("_", " ")
                           for c in eng_cols]
        COLORS = [C.ISD_RED, C.ISD_BLUE, C.ISD_CORAL, C.ISD_PURPLE]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        grouped.plot(kind="bar", ax=ax, color=COLORS[:len(eng_cols)])
        ax.set_title("Avg Engagement by Platform", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Average count")
        ax.legend(fontsize=8, loc="upper right")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        fig.tight_layout()
        self._embed(parent, fig, "Engagement by Platform",
                    data_df=grouped.reset_index())

    def _chart_hashtags(self, parent, df, col):
        tag_lists = self._parse_hashtags(df, col)
        flat = [t for tags in tag_lists for t in tags]
        if not flat:
            return
        top = pd.Series(Counter(flat)).nlargest(15)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        top.sort_values().plot(kind="barh", ax=ax, color=C.ISD_CORAL)
        ax.set_title("Top Hashtags", fontsize=10)
        ax.set_xlabel("Mentions")
        fig.tight_layout()

        data_df = top.reset_index()
        data_df.columns = ["Hashtag", "Count"]
        self._embed(parent, fig, "Top Hashtags", data_df=data_df)

    def _chart_sentiment_platform(self, parent, df, sent_col, plat_col):
        tmp = df[[sent_col, plat_col]].dropna()
        if tmp.empty or len(tmp) < 10:
            return
        ct = pd.crosstab(tmp[plat_col], tmp[sent_col],
                         normalize="index") * 100
        if ct.empty:
            return

        SENT_COLORS = {"positive": "#22c55e", "neutral": "#6b7280",
                       "negative": C.ISD_RED, "mixed": C.ISD_CORAL}
        colors = [SENT_COLORS.get(c.lower(), C.ISD_BLUE) for c in ct.columns]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ct.plot(kind="bar", ax=ax, color=colors, width=0.75)
        ax.set_title("Sentiment by Platform", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("% of posts")
        ax.legend(fontsize=8, loc="upper right")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        fig.tight_layout()

        data_df = ct.reset_index()
        self._embed(parent, fig, "Sentiment by Platform", data_df=data_df)

    def _chart_cooccurrence(self, parent, df, col):
        tag_lists = self._parse_hashtags(df, col)
        flat = [t for tags in tag_lists for t in tags]
        if len(flat) < 20:
            return
        top_tags = pd.Series(Counter(flat)).nlargest(15).index.tolist()

        matrix = pd.DataFrame(0, index=top_tags, columns=top_tags, dtype=int)
        for tags in tag_lists:
            present = [t for t in tags if t in top_tags]
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    matrix.loc[a, b] += 1
                    matrix.loc[b, a] += 1

        np.fill_diagonal(matrix.values, 0)
        if matrix.values.max() == 0:
            return

        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(top_tags)))
        ax.set_yticks(range(len(top_tags)))
        ax.set_xticklabels(top_tags, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(top_tags, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Hashtag Co-occurrence", fontsize=10)
        fig.tight_layout()

        self._embed(parent, fig, "Hashtag Co-occurrence",
                    data_df=matrix.reset_index())

    def _chart_eng_sentiment(self, parent, df, eng_col, sent_col):
        tmp = df[[sent_col, eng_col]].copy()
        tmp[eng_col] = pd.to_numeric(tmp[eng_col], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp) < 10:
            return

        groups = tmp.groupby(sent_col)[eng_col]
        labels = sorted(groups.groups.keys())
        data = [groups.get_group(label).values for label in labels]

        SENT_COLORS = {"positive": "#22c55e", "neutral": "#6b7280",
                       "negative": C.ISD_RED, "mixed": C.ISD_CORAL}

        fig, ax = plt.subplots(figsize=(6, 3.5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            color = SENT_COLORS.get(labels[i].lower(), C.ISD_BLUE)
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"{eng_col.replace('_', ' ').title()} by Sentiment",
                     fontsize=10)
        ax.set_ylabel(eng_col.replace("_", " "))
        fig.tight_layout()

        summary = tmp.groupby(sent_col)[eng_col].describe()
        self._embed(parent, fig, "Engagement vs Sentiment",
                    data_df=summary.reset_index())

    # ── Hashtag parser ───────────────────────────────────────────────────

    def _parse_hashtags(self, df, col) -> list[list[str]]:
        """Parse hashtag column into list of tag lists per row."""
        result = []
        for val in df[col].dropna():
            s = str(val).strip()
            if not s or s.lower() in ("nan", "[]", ""):
                continue
            tags = []
            try:
                items = ast.literal_eval(s)
                if isinstance(items, list):
                    tags = [str(x).strip("#").lower() for x in items if x]
            except Exception:
                for t in re.split(r"[,\s]+", s):
                    t = t.strip("#").strip()
                    if t and t.lower() != "nan":
                        tags.append(t.lower())
            if tags:
                result.append(tags)
        return result

    # ── Event pins ───────────────────────────────────────────────────────

    def _add_pin(self):
        date_str = self._pin_var.get().strip()
        if not date_str:
            return
        try:
            pd.Timestamp(date_str)
        except Exception:
            return
        if date_str not in self._pins:
            self._pins.append(date_str)
        self._pin_var.set("")
        self._refresh_pins()

    def _refresh_pins(self):
        for w in self._pin_frame.winfo_children():
            w.destroy()
        for p in self._pins:
            f = ctk.CTkFrame(self._pin_frame, fg_color=C.DIM, corner_radius=4)
            f.pack(side="left", padx=2)
            ctk.CTkLabel(f, text=p, font=("Segoe UI", 9),
                         text_color=C.TEXT).pack(side="left", padx=(6, 2))
            ctk.CTkButton(f, text="x", width=18, height=18,
                          fg_color="transparent", font=("Segoe UI", 8),
                          command=lambda d=p: self._remove_pin(d)
                          ).pack(side="left", padx=(0, 2))

    def _remove_pin(self, date_str):
        if date_str in self._pins:
            self._pins.remove(date_str)
        self._refresh_pins()

    def _draw_pins(self, ax):
        """Draw vertical dashed lines for pinned event dates."""
        for p in self._pins:
            try:
                ts = pd.Timestamp(p)
                ax.axvline(ts, color=C.ISD_CORAL, linestyle="--",
                           linewidth=1.5, alpha=0.8, zorder=5)
                ax.text(ts, ax.get_ylim()[1] * 0.95, f"  {p}",
                        fontsize=7, color=C.ISD_CORAL, rotation=90,
                        va="top", ha="left")
            except Exception:
                pass

    # ── Embed (enhanced with export + data toggle + click support) ───────

    def _embed(self, parent, fig, title, data_df=None,
               click_col=None, time_chart=False):
        card = ctk.CTkFrame(parent, fg_color=C.CARD, corner_radius=8)
        card.pack(fill="x", padx=6, pady=6)

        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # Click-to-filter: categorical charts (bars / pie wedges)
        if click_col:
            def on_pick(event):
                artist = event.artist
                gid = artist.get_gid()
                if gid:
                    self._chart_filter = {"col": click_col, "val": gid}
                    self.refresh()
            canvas.mpl_connect("pick_event", on_pick)

        # Click-to-filter: time charts (drag to select date range)
        if time_chart and fig.axes:
            ax = fig.axes[0]
            try:
                span = SpanSelector(
                    ax, self._on_span_select, "horizontal",
                    useblit=False,
                    props=dict(alpha=0.3, facecolor=C.ISD_RED))
                self._span_selectors.append(span)  # prevent GC
            except Exception:
                pass

        # Button row: export formats + view data
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(0, 6))

        for fmt_name, fmt in [("SVG", "svg"), ("PDF", "pdf"), ("PNG", "png")]:
            eb = ctk.CTkButton(
                btn_row, text=fmt_name, width=45, height=24,
                fg_color=C.BTN, font=("Segoe UI", 9),
                command=lambda f=fig, t=title, fm=fmt:
                    self._export_chart(f, t, fm))
            eb.pack(side="left", padx=2)
            tip(eb, f"Export this chart as {fmt_name} "
                    f"({'vector — crisp in papers' if fmt != 'png' else 'raster'}).")

        # View Data toggle
        if data_df is not None and not data_df.empty:
            data_container = ctk.CTkFrame(card, fg_color="transparent")
            visible = [False]

            def toggle_data():
                if visible[0]:
                    data_container.pack_forget()
                    visible[0] = False
                else:
                    if not data_container.winfo_children():
                        self._build_data_table(data_container, data_df)
                    data_container.pack(fill="x", padx=8, pady=(0, 8))
                    visible[0] = True

            vd = ctk.CTkButton(btn_row, text="View Data", width=75, height=24,
                               fg_color=C.BTN, font=("Segoe UI", 9),
                               command=toggle_data)
            vd.pack(side="left", padx=6)
            tip(vd, "Show / hide the raw numbers behind this chart.\n"
                    "Useful for copying values into a manuscript.")

        self._figures.append(fig)

    def _build_data_table(self, parent, data_df):
        """Build a small treeview showing the data behind a chart."""
        cols = list(data_df.columns)[:10]
        preview = data_df.head(25)

        container = tk.Frame(parent, bg=C.CARD)
        container.pack(fill="x")

        tree = ttk.Treeview(container, columns=cols, show="headings",
                            height=min(len(preview), 10))
        _style_treeview(tree)

        for col in cols:
            tree.heading(col, text=str(col)[:25])
            tree.column(col, width=110, minwidth=50)

        for _, row in preview.iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v)[:60])
            tree.insert("", "end", values=vals)

        hsb = ttk.Scrollbar(container, orient="horizontal",
                            command=tree.xview)
        tree.configure(xscrollcommand=hsb.set)
        tree.pack(fill="x")
        hsb.pack(fill="x")

    # ── SpanSelector callback ────────────────────────────────────────────

    def _on_span_select(self, xmin, xmax):
        """Called when the user drags across a time chart."""
        try:
            d1 = num2date(xmin).strftime("%Y-%m-%d")
            d2 = num2date(xmax).strftime("%Y-%m-%d")
            if d1 != d2:
                self._chart_filter = {"type": "date", "from": d1, "to": d2}
                self.refresh()
        except Exception:
            pass

    # ── Chart export ─────────────────────────────────────────────────────

    def _export_chart(self, fig, title, fmt):
        """Save a chart to disk as SVG, PDF, or PNG."""
        ext_map = {"svg": ".svg", "pdf": ".pdf", "png": ".png"}
        ft_map = {
            "svg": [("SVG", "*.svg")],
            "pdf": [("PDF", "*.pdf")],
            "png": [("PNG", "*.png")],
        }
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

        path = filedialog.asksaveasfilename(
            defaultextension=ext_map[fmt],
            filetypes=ft_map[fmt],
            initialfile=f"{safe_title}{ext_map[fmt]}")
        if not path:
            return
        try:
            fig.savefig(path, format=fmt, dpi=200, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Export Error", str(exc))

    # ── Theme rebuild ────────────────────────────────────────────────────

    def rebuild(self):
        pins = list(self._pins)
        time_bin = self._time_bin.get()
        chart_filter = self._chart_filter

        for fig in self._figures:
            try:
                if plt is not None:
                    plt.close(fig)
            except Exception:
                pass
        self._figures.clear()
        self._span_selectors.clear()

        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)

        self._time_bin = tk.StringVar(value=time_bin)
        self._pin_var = tk.StringVar()
        self._pins = pins
        self._chart_filter = chart_filter
        self._candidates = []
        self._active_btns = {}
        self._build()
        self._refresh_pins()
