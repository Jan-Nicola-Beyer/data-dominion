"""
Shared UI helpers:

  - Tooltip / tip()          — lightweight hover tooltip for any tk widget
  - _style_treeview()        — applies the current theme to a ttk.Treeview
  - get_samples()            — returns non-null sample values from a DataFrame column
  - DatePickerButton         — button that opens a calendar popup for date selection
  - NAV_TIPS / FIELD_TIPS    — tooltip text dictionaries
"""

import calendar
import datetime
import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C


# ── Tooltip ────────────────────────────────────────────────────────────────────

class Tooltip:
    """Lightweight hover tooltip for any tkinter / customtkinter widget."""

    DELAY = 650   # ms before the tooltip appears

    def __init__(self, widget, text: str, wraplength: int = 300):
        self._w  = widget
        self._tx = text
        self._wl = wraplength
        self._id = None
        self._tw = None

        widget.bind("<Enter>",       self._schedule, add="+")
        widget.bind("<Leave>",       self._cancel,   add="+")
        widget.bind("<ButtonPress>", self._cancel,   add="+")
        widget.bind("<Destroy>",     self._cancel,   add="+")

    def _schedule(self, _=None):
        self._cancel()
        self._id = self._w.after(self.DELAY, self._show)

    def _cancel(self, _=None):
        if self._id:
            self._w.after_cancel(self._id)
            self._id = None
        if self._tw:
            try:
                self._tw.destroy()
            except Exception:
                pass
            self._tw = None

    def _show(self):
        if self._tw:
            return
        try:
            wx = self._w.winfo_rootx()
            wy = self._w.winfo_rooty()
            wh = self._w.winfo_height()
            sw = self._w.winfo_screenwidth()
            sh = self._w.winfo_screenheight()
        except Exception:
            return

        tw = tk.Toplevel(self._w)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        tw.configure(bg="#1e293b")

        tk.Label(
            tw, text=self._tx, justify="left",
            bg="#1e293b", fg="#cbd5e1",
            font=("Segoe UI", 10),
            wraplength=self._wl,
            padx=12, pady=8,
        ).pack()

        tw.update_idletasks()
        tw_w = tw.winfo_reqwidth()
        tw_h = tw.winfo_reqheight()

        x = min(wx, sw - tw_w - 10)
        y = wy + wh + 6
        if y + tw_h > sh - 10:
            y = wy - tw_h - 6
        x = max(10, x)

        tw.wm_geometry(f"+{x}+{y}")
        self._tw = tw


def tip(widget, text: str, wraplength: int = 300) -> Tooltip:
    """Attach a hover tooltip to *widget* and return the Tooltip instance."""
    return Tooltip(widget, text, wraplength)


# ── Treeview styling ───────────────────────────────────────────────────────────

_tv_style_theme = None  # track which theme the style was last configured for

def _style_treeview(tv: ttk.Treeview):
    """Apply the current theme style to a ttk.Treeview widget.

    The ttk.Style is only reconfigured when the theme changes, not per-widget.
    """
    global _tv_style_theme
    if _tv_style_theme != C.CURRENT_THEME:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.Treeview",
            background=C.CARD, foreground=C.TEXT, fieldbackground=C.CARD,
            borderwidth=0, font=("Segoe UI", 10), rowheight=26)
        style.configure("Custom.Treeview.Heading",
            background=C.PANEL, foreground=C.MUTED, font=("Segoe UI", 9, "bold"),
            borderwidth=0)
        style.map("Custom.Treeview",
            background=[("selected", C.SELECT)],
            foreground=[("selected", C.TEXT)])
        _tv_style_theme = C.CURRENT_THEME
    tv.configure(style="Custom.Treeview")


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def get_samples(df: pd.DataFrame, col: str, n: int = 5) -> list:
    """Return up to *n* non-null, non-empty sample values from a column."""
    s = df[col].dropna()
    s = s[s.astype(str).str.strip() != ""]
    return s.head(n).tolist()


# ── Date picker ────────────────────────────────────────────────────────────────

class _CalendarPopup(tk.Toplevel):
    """A small month-view calendar popup window."""

    WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    MONTHS   = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]

    def __init__(self, parent, initial: datetime.date, on_select):
        super().__init__(parent)
        self.wm_overrideredirect(True)
        self.wm_attributes("-topmost", True)
        self.configure(bg=C.PANEL)
        self._on_select = on_select
        self._year  = initial.year
        self._month = initial.month
        self._today = datetime.date.today()
        self._draw()

    def position_below(self, widget):
        """Position the popup below *widget* and grab all events so buttons work."""
        self.update_idletasks()
        x = widget.winfo_rootx()
        y = widget.winfo_rooty() + widget.winfo_height() + 2
        sw = self.winfo_screenwidth()
        w  = self.winfo_reqwidth()
        if x + w > sw:
            x = sw - w - 4
        self.wm_geometry(f"+{x}+{y}")
        # On Windows, overrideredirect windows need grab_set + focus_force
        # to receive mouse click events reliably.
        self.grab_set()
        self.focus_force()
        # Defer the global click binding by one event-loop tick so the button
        # click that opened the popup finishes propagating first.
        self.after(50, lambda: self.bind_all("<Button-1>", self._on_global_click, add=True))

    def _draw(self):
        for w in self.winfo_children():
            w.destroy()

        # ── Header: prev / month+year / next ──────────────────────────────────
        hdr = tk.Frame(self, bg=C.PANEL, padx=4, pady=4)
        hdr.pack(fill="x")
        tk.Button(hdr, text="◀", bg=C.PANEL, fg=C.TEXT, bd=0, font=("Segoe UI", 10),
                  activebackground=C.SELECT, cursor="hand2",
                  command=self._prev).pack(side="left")
        tk.Label(hdr, text=f"{self.MONTHS[self._month - 1]}  {self._year}",
                 bg=C.PANEL, fg=C.TEXT, font=("Segoe UI", 10, "bold"),
                 width=18, anchor="center").pack(side="left", expand=True)
        tk.Button(hdr, text="▶", bg=C.PANEL, fg=C.TEXT, bd=0, font=("Segoe UI", 10),
                  activebackground=C.SELECT, cursor="hand2",
                  command=self._next).pack(side="right")

        # Thin separator
        tk.Frame(self, bg=C.BORDER, height=1).pack(fill="x")

        # ── Weekday headers ───────────────────────────────────────────────────
        wrow = tk.Frame(self, bg=C.PANEL, padx=6, pady=2)
        wrow.pack(fill="x")
        for d in self.WEEKDAYS:
            tk.Label(wrow, text=d, bg=C.PANEL, fg=C.MUTED,
                     font=("Segoe UI", 8), width=3, anchor="center").pack(side="left")

        # ── Day grid ──────────────────────────────────────────────────────────
        for week in calendar.monthcalendar(self._year, self._month):
            row = tk.Frame(self, bg=C.PANEL, padx=6, pady=1)
            row.pack(fill="x")
            for day in week:
                if day == 0:
                    tk.Label(row, text="", bg=C.PANEL, width=3).pack(side="left")
                else:
                    d      = datetime.date(self._year, self._month, day)
                    is_today = (d == self._today)
                    fg     = C.ACCENT if is_today else C.TEXT
                    weight = "bold" if is_today else "normal"
                    btn = tk.Button(
                        row, text=str(day), width=3, bd=0, relief="flat",
                        bg=C.PANEL, fg=fg, font=("Segoe UI", 9, weight),
                        activebackground=C.SELECT, activeforeground=C.TEXT,
                        cursor="hand2",
                        command=lambda _d=d: self._pick(_d))
                    btn.pack(side="left")

        # ── Footer: Clear / Today ─────────────────────────────────────────────
        tk.Frame(self, bg=C.BORDER, height=1).pack(fill="x")
        ftr = tk.Frame(self, bg=C.PANEL, padx=6, pady=4)
        ftr.pack(fill="x")
        tk.Button(ftr, text="Clear", bd=0, bg=C.PANEL, fg=C.MUTED,
                  font=("Segoe UI", 9), activebackground=C.SELECT,
                  cursor="hand2", command=lambda: self._pick(None)
                  ).pack(side="left")
        tk.Button(ftr, text="Today", bd=0, bg=C.PANEL, fg=C.ACCENT,
                  font=("Segoe UI", 9), activebackground=C.SELECT,
                  cursor="hand2", command=lambda: self._pick(self._today)
                  ).pack(side="right")

    def _prev(self):
        if self._month == 1:
            self._month, self._year = 12, self._year - 1
        else:
            self._month -= 1
        self._draw()

    def _next(self):
        if self._month == 12:
            self._month, self._year = 1, self._year + 1
        else:
            self._month += 1
        self._draw()

    def _on_global_click(self, event):
        """Close the popup when the user clicks outside it."""
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
            # Close only if the click was outside this popup's widget tree
            popup_path = str(self)
            if widget is None or not str(widget).startswith(popup_path):
                self._pick(None, cancelled=True)
        except Exception:
            pass

    def _pick(self, date, cancelled=False):
        try:
            self.unbind_all("<Button-1>")
            self.grab_release()
        except Exception:
            pass
        if not cancelled:
            self._on_select(date)
        self.destroy()


class DatePickerButton(ctk.CTkFrame):
    """
    A compact button that opens a month-view calendar popup.
    Stores the selected date in *textvariable* as an ISO string (YYYY-MM-DD).
    """

    def __init__(self, master, textvariable: tk.StringVar,
                 placeholder: str = "Pick date…", width: int = 120, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._var     = textvariable
        self._placeholder = placeholder
        self._popup   = None

        self._btn = ctk.CTkButton(
            self, text=placeholder, width=width,
            fg_color=C.PANEL, hover_color=C.SELECT,
            text_color=C.MUTED, font=("Segoe UI", 11),
            command=self._toggle)
        self._btn.pack()

        self._var.trace_add("write", self._sync_label)
        self._sync_label()

    def _sync_label(self, *_):
        val = self._var.get()
        self._btn.configure(text=val if val else self._placeholder,
                            text_color=C.TEXT if val else C.MUTED)

    def _toggle(self):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
            self._popup = None
            return
        val = self._var.get()
        try:
            initial = datetime.date.fromisoformat(val)
        except Exception:
            initial = datetime.date.today()
        self._popup = _CalendarPopup(self, initial, self._on_pick)
        self._popup.position_below(self._btn)

    def _on_pick(self, date):
        self._var.set(date.isoformat() if date else "")
        self._popup = None


# ── Date range slider ─────────────────────────────────────────────────────────

class DateRangeSlider(ctk.CTkFrame):
    """Two-handle date range slider.  Maps slider 0.0–1.0 to a min–max date range.

    Stores selected dates as ISO strings (YYYY-MM-DD) in *from_var* / *to_var*.
    Call ``set_date_range(min_ts, max_ts)`` after loading data.
    """

    def __init__(self, master, from_var: tk.StringVar, to_var: tk.StringVar,
                 width: int = 320, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._from_var = from_var
        self._to_var   = to_var
        self._min_epoch: float = 0
        self._max_epoch: float = 1
        self._range_set = False
        self._suppress  = False          # prevent feedback loops

        # ── From row ─────────────────────────────────────────────────
        r1 = ctk.CTkFrame(self, fg_color="transparent")
        r1.pack(fill="x")
        ctk.CTkLabel(r1, text="From", text_color=C.MUTED,
                     font=("Segoe UI", 9), width=34).pack(side="left")
        self._from_lbl = ctk.CTkLabel(r1, text="—", text_color=C.TEXT,
                                      font=("Segoe UI", 10), width=82,
                                      anchor="w")
        self._from_lbl.pack(side="right")
        self._from_sl = ctk.CTkSlider(r1, from_=0, to=1, width=width - 120,
                                      number_of_steps=1000,
                                      command=self._on_from)
        self._from_sl.set(0)
        self._from_sl.pack(side="left", padx=4)

        # ── To row ───────────────────────────────────────────────────
        r2 = ctk.CTkFrame(self, fg_color="transparent")
        r2.pack(fill="x")
        ctk.CTkLabel(r2, text="To", text_color=C.MUTED,
                     font=("Segoe UI", 9), width=34).pack(side="left")
        self._to_lbl = ctk.CTkLabel(r2, text="—", text_color=C.TEXT,
                                    font=("Segoe UI", 10), width=82,
                                    anchor="w")
        self._to_lbl.pack(side="right")
        self._to_sl = ctk.CTkSlider(r2, from_=0, to=1, width=width - 120,
                                    number_of_steps=1000,
                                    command=self._on_to)
        self._to_sl.set(1)
        self._to_sl.pack(side="left", padx=4)

        # Sync when vars are externally cleared
        self._from_var.trace_add("write", self._sync_from_var)
        self._to_var.trace_add("write", self._sync_to_var)

    # ── Public API ────────────────────────────────────────────────────

    def set_date_range(self, min_ts, max_ts):
        """Configure slider bounds from pandas Timestamps."""
        try:
            mn = pd.Timestamp(min_ts)
            mx = pd.Timestamp(max_ts)
            if pd.isna(mn) or pd.isna(mx) or mn >= mx:
                return
            self._min_epoch = mn.timestamp()
            self._max_epoch = mx.timestamp()
            self._range_set = True
            self.reset()
        except Exception:
            pass

    def reset(self):
        """Move both sliders to extremes and clear vars."""
        self._suppress = True
        self._from_sl.set(0)
        self._to_sl.set(1)
        self._update_label(self._from_lbl, 0)
        self._update_label(self._to_lbl, 1)
        self._from_var.set("")
        self._to_var.set("")
        self._suppress = False

    # ── Internal ──────────────────────────────────────────────────────

    def _frac_to_date(self, frac: float) -> str:
        epoch = self._min_epoch + frac * (self._max_epoch - self._min_epoch)
        return pd.Timestamp(epoch, unit="s").strftime("%Y-%m-%d")

    def _update_label(self, lbl, frac):
        if not self._range_set:
            lbl.configure(text="—")
            return
        lbl.configure(text=self._frac_to_date(frac))

    def _on_from(self, val):
        if self._suppress or not self._range_set:
            return
        val = float(val)
        to_val = self._to_sl.get()
        if val > to_val:
            val = to_val
            self._from_sl.set(val)
        self._update_label(self._from_lbl, val)
        self._suppress = True
        self._from_var.set(self._frac_to_date(val) if val > 0.001 else "")
        self._suppress = False

    def _on_to(self, val):
        if self._suppress or not self._range_set:
            return
        val = float(val)
        from_val = self._from_sl.get()
        if val < from_val:
            val = from_val
            self._to_sl.set(val)
        self._update_label(self._to_lbl, val)
        self._suppress = True
        self._to_var.set(self._frac_to_date(val) if val < 0.999 else "")
        self._suppress = False

    def _sync_from_var(self, *_):
        if self._suppress or not self._range_set:
            return
        val = self._from_var.get()
        if not val:
            self._suppress = True
            self._from_sl.set(0)
            self._update_label(self._from_lbl, 0)
            self._suppress = False

    def _sync_to_var(self, *_):
        if self._suppress or not self._range_set:
            return
        val = self._to_var.get()
        if not val:
            self._suppress = True
            self._to_sl.set(1)
            self._update_label(self._to_lbl, 1)
            self._suppress = False


# ── Tooltip text dictionaries ──────────────────────────────────────────────────

NAV_TIPS: dict = {
    "home":
        "Start here.\n\n"
        "See a summary of all loaded datasets and quickly launch the import wizard.",
    "import":
        "Load a new file (CSV, TSV or Excel).\n\n"
        "A 4-step wizard guides you through selecting the file, letting the app "
        "auto-detect your columns, reviewing the mapping, and seeing a quality report.",
    "datasets":
        "Manage all your loaded datasets in one place.\n\n"
        "See which standard social-media fields are present in each dataset, "
        "then merge them all into a single comparable table — missing columns "
        "are automatically filled with N/A.",
    "table":
        "Browse your data row by row.\n\n"
        "Use the filter bar to search by keyword, filter by platform or language, "
        "and narrow down by date range. Click any column header to sort.",
    "analytics":
        "Auto-generated charts for the currently filtered data.\n\n"
        "Includes: post volume over time, platform breakdown, language breakdown, "
        "and average engagement over time. Hit Refresh after changing filters.",
    "coding":
        "Qualitative coding — attach colour-coded labels to individual rows.\n\n"
        "Create tags (e.g. 'relevant', 'positive', 'misinformation'), select rows "
        "in the table, and apply tags. Use keyboard shortcuts 1–9 for speed.",
    "topics":
        "Topic Modelling — discover themes in your text data.\n\n"
        "Uses all-MiniLM-L6-v2 embeddings with BERTopic to automatically\n"
        "group similar documents into topics. Includes a visual topic map\n"
        "and adjustable parameters for iterative exploration.",
    "slicer":
        "Create and save named subsets of your data.\n\n"
        "Combine a keyword filter with an advanced boolean query "
        "(e.g. like_count > 500) and save the combination as a reusable 'slice'.",
    "export":
        "Save the current filtered view to a file.\n\n"
        "Options: plain CSV, CSV with tag annotations, Excel (.xlsx), "
        "or all loaded datasets bundled in a ZIP archive.",
    "settings":
        "Adjust display preferences.\n\n"
        "The row limit controls how many rows appear in the Table view "
        "(filtering and exports always use all rows, regardless of this setting).",
}

FIELD_TIPS: dict = {
    "post_id":               "Unique identifier of the post (e.g. tweet ID, Instagram media ID).",
    "platform":              "Which social network the post came from (Twitter, Instagram, TikTok …).",
    "post_url":              "Direct link / permalink to the original post.",
    "content_text":          "The full text or caption of the post.",
    "language":              "Language of the post — usually a 2-letter ISO code (en, de, fr …).",
    "created_at":            "Date and time when the post was originally published.",
    "collected_at":          "Date and time when this data was collected / scraped.",
    "author_id":             "Unique numeric or string identifier of the author's account.",
    "author_username":       "Username or handle of the author (e.g. @example).",
    "author_display_name":   "Full display name shown on the author's profile.",
    "author_verified":       "Whether the author has a verified / blue-check badge.",
    "author_followers_count": "Number of followers the author has.",
    "author_following_count": "Number of accounts the author is following.",
    "author_posts_count":    "Total number of posts the author has published.",
    "author_location":       "Geographic location listed on the author's profile.",
    "like_count":            "Number of likes, favorites, or hearts on the post.",
    "comment_count":         "Number of comments or replies.",
    "share_count":           "Number of shares, retweets, or reposts.",
    "view_count":            "Number of views, impressions, or plays.",
    "engagement_total":      "Total interactions (likes + comments + shares + clicks).",
    "engagement_rate":       "Engagement as a percentage of reach or impressions.",
    "quote_count":           "Number of quote-tweets or quote-posts.",
    "hashtags":              "Hashtags or topic tags used in the post.",
    "in_reply_to_id":        "ID of the post this is a reply to (if applicable).",
    "is_reply":              "Whether this post is a reply to another post.",
    "is_repost":             "Whether this is a retweet, repost, or reshare.",
    "mentioned_handles":     "Other accounts mentioned or tagged in the post.",
    "sentiment":             "Sentiment classification: positive, negative, or neutral.",
    "media_type":            "Type of media attached — photo, video, reel, story, link …",
}
