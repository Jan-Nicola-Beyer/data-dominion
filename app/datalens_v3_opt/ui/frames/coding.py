"""Qualitative coding frame — research-grade annotation tool.

Features: keyboard shortcuts 1–9, reading pane, undo, tag descriptions
with group-based mutual exclusivity, rename/delete/merge, bulk tagging
by search, regex auto-coding, progress bar, agreement statistics,
tag audit trail, and semicolon-normalised storage.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import datetime
import re

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import customtkinter as ctk
import pandas as pd

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip, _style_treeview

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App

MAX_UNDO = 50


def _normalize_tags(s: str) -> str:
    """Alphabetise, deduplicate, strip whitespace."""
    parts = sorted(set(t.strip() for t in s.split(";") if t.strip()))
    return ";".join(parts)


def _remove_tag_from_column(df: "pd.DataFrame", tag: str) -> None:
    """Remove *tag* from the _tags column using vectorised str ops."""
    col = df["_tags"].astype(str)
    esc = re.escape(tag)
    # Remove "tag;" or ";tag" or standalone "tag"
    col = col.str.replace(rf"(?:^|;)\s*{esc}\s*(?:;|$)", ";", regex=True)
    # Clean up leading/trailing semicolons and double semicolons
    col = col.str.strip(";").str.replace(r";{2,}", ";", regex=True)
    df["_tags"] = col


def _bulk_add_tag(df: "pd.DataFrame", mask, tag: str,
                  exclude_tags: set | None = None) -> None:
    """Add *tag* to rows where *mask* is True, using vectorised ops."""
    col = df.loc[mask, "_tags"].astype(str)
    # Remove excluded tags first
    if exclude_tags:
        for ex in exclude_tags:
            esc = re.escape(ex)
            col = col.str.replace(rf"(?:^|;)\s*{esc}\s*(?:;|$)", ";", regex=True)
            col = col.str.strip(";")
    # Append the new tag and normalize
    col = col + ";" + tag
    df.loc[mask, "_tags"] = col.apply(_normalize_tags)


def _rename_tag_in_column(df: "pd.DataFrame", old: str, new: str) -> None:
    """Rename *old* to *new* in the _tags column using vectorised str ops."""
    col = df["_tags"].astype(str)
    esc_old = re.escape(old)
    # Replace exact tag matches between semicolons / boundaries
    col = col.str.replace(
        rf"((?:^|;)\s*){esc_old}(\s*(?:;|$))", rf"\1{new}\2", regex=True)
    # Normalize to deduplicate and sort
    df["_tags"] = col.apply(_normalize_tags)


class CodingFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._undo_stack: list[tuple] = []   # (df_idx, old_tags_str)
        self._auto_rules: list[dict] = []    # [{"pattern": ..., "tag": ...}]
        self._display_cols: list[str] = []   # user-chosen columns for the table
        self._text_col: str = ""             # user-chosen text column for reading pane
        self._build()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _tag_color(self, name: str) -> str:
        info = self.app.tags.get(name, {})
        if isinstance(info, str):
            return info
        return info.get("color", C.DIM)

    def _tag_desc(self, name: str) -> str:
        info = self.app.tags.get(name, {})
        if isinstance(info, dict):
            return info.get("desc", "")
        return ""

    def _tag_group(self, name: str) -> str:
        info = self.app.tags.get(name, {})
        if isinstance(info, dict):
            return info.get("group", "")
        return ""

    def _tag_exclusive(self, name: str) -> bool:
        info = self.app.tags.get(name, {})
        if isinstance(info, dict):
            return info.get("exclusive", False)
        return False

    def _group_members(self, group: str) -> list[str]:
        """Return all tag names in the given group."""
        if not group:
            return []
        return [n for n in self.app.tags if self._tag_group(n) == group]

    def _existing_groups(self) -> list[str]:
        groups = set()
        for info in self.app.tags.values():
            if isinstance(info, dict) and info.get("group"):
                groups.add(info["group"])
        return sorted(groups)

    def _tag_names(self) -> list[str]:
        return list(self.app.tags.keys())

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        # Toolbar
        tb = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=48)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="Qualitative Coding",
                     font=("Segoe UI", 15, "bold"),
                     text_color=C.TEXT).pack(side="left", padx=16)

        new_btn = ctk.CTkButton(tb, text="+ New Tag", fg_color=C.ACCENT,
                                width=100, command=self._new_tag)
        new_btn.pack(side="left", padx=8, pady=8)
        tip(new_btn, "Create a new colour-coded tag with an optional\n"
                     "description and group for mutual exclusivity.")

        mgr_btn = ctk.CTkButton(tb, text="Manage Tags", fg_color=C.BTN,
                                width=110, command=self._manage_tags_dialog)
        mgr_btn.pack(side="left", padx=4, pady=8)
        tip(mgr_btn, "Rename, delete, merge tags, or edit descriptions.")

        auto_btn = ctk.CTkButton(tb, text="Auto-Code", fg_color=C.BTN,
                                 width=90, command=self._auto_code_dialog)
        auto_btn.pack(side="left", padx=4, pady=8)
        tip(auto_btn, "Define regex rules to auto-tag rows in bulk.")

        agr_btn = ctk.CTkButton(tb, text="Agreement", fg_color=C.BTN,
                                width=90, command=self._agreement_dialog)
        agr_btn.pack(side="left", padx=4, pady=8)
        tip(agr_btn, "Calculate inter-coder agreement (Cohen's Kappa)\n"
                     "by comparing your tags against another column.")

        # Progress bar
        prog = ctk.CTkFrame(self, fg_color=C.CARD, corner_radius=0, height=28)
        prog.pack(fill="x")
        prog.pack_propagate(False)
        self._prog_label = ctk.CTkLabel(prog, text="",
                                        text_color=C.MUTED,
                                        font=("Segoe UI", 10))
        self._prog_label.pack(side="left", padx=12)
        self._shortcut_label = ctk.CTkLabel(
            prog, text="", text_color=C.MUTED, font=("Segoe UI", 10))
        self._shortcut_label.pack(side="right", padx=12)

        # Tag bar
        self._tag_bar = ctk.CTkFrame(self, fg_color=C.CARD, corner_radius=0,
                                     height=44)
        self._tag_bar.pack(fill="x")

        # Setup bar: text column + display columns
        setup_bar = ctk.CTkFrame(self, fg_color=C.CARD, corner_radius=0,
                                 height=36)
        setup_bar.pack(fill="x")
        setup_bar.pack_propagate(False)

        # Text column selector (primary — used by reading pane, bulk tag, auto-code)
        ctk.CTkLabel(setup_bar, text="Text column:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(12, 4))
        self._text_col_var = tk.StringVar(value="— select —")
        self._text_col_om = ctk.CTkOptionMenu(
            setup_bar, variable=self._text_col_var,
            values=["— select —"], width=180, height=26,
            command=self._on_text_col_change)
        self._text_col_om.pack(side="left", padx=(0, 16), pady=4)
        tip(self._text_col_om,
            "Choose which column contains the main text to code.\n"
            "Used by the reading pane, bulk tagging, and auto-code.")

        # Display columns selector
        sep = ctk.CTkLabel(setup_bar, text="|", text_color=C.DIM)
        sep.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(setup_bar, text="Table columns:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(0, 4))
        self._col_summary = ctk.CTkLabel(
            setup_bar, text="(default)", text_color=C.TEXT,
            font=("Segoe UI", 10))
        self._col_summary.pack(side="left", padx=4)
        sel_btn = ctk.CTkButton(setup_bar, text="Select…",
                                fg_color=C.BTN, width=70, height=26,
                                command=self._open_column_picker)
        sel_btn.pack(side="left", padx=4, pady=4)
        tip(sel_btn, "Choose which columns to display in the coding table.")
        reset_btn = ctk.CTkButton(setup_bar, text="Reset", fg_color=C.BTN,
                                  width=55, height=26,
                                  command=self._reset_columns)
        reset_btn.pack(side="left", padx=2, pady=4)
        tip(reset_btn, "Reset to default column selection.")

        # Main area: table (left) + reading pane (right)
        main = ctk.CTkFrame(self, fg_color=C.BG, corner_radius=0)
        main.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        # Table
        tv_frame = ctk.CTkFrame(main, fg_color=C.CARD, corner_radius=0)
        tv_frame.pack(side="left", fill="both", expand=True)
        self._tv = ttk.Treeview(tv_frame, show="headings",
                                selectmode="extended")
        _style_treeview(self._tv)
        ysb = ttk.Scrollbar(tv_frame, orient="vertical",
                            command=self._tv.yview)
        self._tv.configure(yscrollcommand=ysb.set)
        ysb.pack(side="right", fill="y")
        self._tv.pack(fill="both", expand=True)
        self._tv.bind("<<TreeviewSelect>>", self._on_select)

        # Keyboard shortcuts 1-9
        for i in range(9):
            self._tv.bind(f"<Key-{i+1}>",
                          lambda e, idx=i: self._shortcut(idx))
        self._tv.bind("<Control-z>", lambda e: self._undo())

        # Reading pane (right)
        rp = ctk.CTkFrame(main, fg_color=C.PANEL, corner_radius=8, width=340)
        rp.pack(side="right", fill="y", padx=(8, 0))
        rp.pack_propagate(False)
        ctk.CTkLabel(rp, text="Reading Pane", font=("Segoe UI", 11, "bold"),
                     text_color=C.MUTED).pack(anchor="w", padx=12,
                                               pady=(10, 4))
        self._rp_meta = ctk.CTkLabel(rp, text="Select a row to view",
                                     text_color=C.MUTED,
                                     font=("Segoe UI", 10),
                                     wraplength=310, justify="left")
        self._rp_meta.pack(anchor="w", padx=12, pady=(0, 6))

        self._rp_text = ctk.CTkTextbox(rp, fg_color=C.CARD,
                                       text_color=C.TEXT,
                                       font=("Segoe UI", 12),
                                       wrap="word", state="disabled")
        self._rp_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Footer: apply + bulk tag
        ftr = ctk.CTkFrame(self, fg_color=C.PANEL, corner_radius=0, height=44)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)

        ctk.CTkLabel(ftr, text="Apply:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(12, 4))
        self._tag_apply_var = tk.StringVar()
        self._tag_apply_om = ctk.CTkOptionMenu(
            ftr, variable=self._tag_apply_var, values=["—"], width=140)
        self._tag_apply_om.pack(side="left", padx=4)

        ab = ctk.CTkButton(ftr, text="Apply", fg_color=C.ACCENT, width=70,
                           command=self._apply_tag)
        ab.pack(side="left", padx=4)
        tip(ab, "Apply the chosen tag to all selected rows.")

        ub = ctk.CTkButton(ftr, text="Undo", fg_color=C.BTN, width=60,
                           command=self._undo)
        ub.pack(side="left", padx=4)
        tip(ub, "Undo the last tag action (Ctrl+Z).")

        # Separator
        ctk.CTkLabel(ftr, text="|", text_color=C.DIM).pack(side="left",
                                                            padx=8)

        # Bulk tag
        ctk.CTkLabel(ftr, text="Bulk:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(side="left", padx=(0, 4))
        self._bulk_var = tk.StringVar()
        be = ctk.CTkEntry(ftr, textvariable=self._bulk_var, width=180,
                          placeholder_text="Search keyword…")
        be.pack(side="left", padx=4)
        tip(be, "Enter a keyword. Click 'Tag All' to apply the\n"
                "selected tag to every matching row at once.")
        bb = ctk.CTkButton(ftr, text="Tag All", fg_color=C.SUCCESS, width=70,
                           command=self._bulk_tag)
        bb.pack(side="left", padx=4)
        tip(bb, "Apply the selected tag to all rows matching\n"
                "the search keyword. Case-insensitive.")

        self._refresh_tags()
        self._refresh_table()
        self._refresh_progress()

    # ── Public refresh ───────────────────────────────────────────────────────

    def refresh(self):
        self._refresh_tags()
        self._refresh_table()
        self._refresh_progress()
        self._refresh_text_col_options()

    # ── Progress ─────────────────────────────────────────────────────────────

    def _refresh_progress(self):
        df = self.app.df
        if df.empty or "_tags" not in df.columns:
            self._prog_label.configure(text="No data loaded")
            self._shortcut_label.configure(text="")
            return
        total = len(df)
        coded = int((df["_tags"].astype(str).str.strip() != "").sum())
        pct = coded / total * 100 if total else 0
        self._prog_label.configure(
            text=f"{coded:,} / {total:,} coded  ({pct:.0f}%)")

        # Shortcut hints
        names = self._tag_names()
        hints = [f"{i+1}={n}" for i, n in enumerate(names[:9])]
        self._shortcut_label.configure(
            text="Shortcuts: " + "  ".join(hints) if hints else "")

    # ── Tag bar ──────────────────────────────────────────────────────────────

    def _refresh_tags(self):
        for w in self._tag_bar.winfo_children():
            w.destroy()
        names = self._tag_names()
        self._tag_apply_om.configure(values=["—"] + names)
        active_filter = self.app.filter_state.get("tag", "")

        all_fg = C.ACCENT if not active_filter else C.DIM
        all_btn = ctk.CTkButton(self._tag_bar, text="All", width=52,
                                height=28, fg_color=all_fg,
                                font=("Segoe UI", 11),
                                command=self._clear_tag_filter)
        all_btn.pack(side="left", padx=4, pady=6)
        tip(all_btn, "Show all rows (clear the tag filter).")

        for i, name in enumerate(names):
            color = self._tag_color(name)
            is_active = (active_filter == name)
            label = f"● {name}" if is_active else name
            shortcut = f" [{i+1}]" if i < 9 else ""

            tag_btn = ctk.CTkButton(
                self._tag_bar, text=label + shortcut, width=100, height=28,
                fg_color=color, hover_color=color,
                font=("Segoe UI", 11),
                command=lambda t=name: self._filter_by_tag(t))
            tag_btn.pack(side="left", padx=4, pady=6)

            # Tooltip with codebook description
            desc = self._tag_desc(name)
            group = self._tag_group(name)
            tt_parts = [f"Tag: {name}"]
            if desc:
                tt_parts.append(f"\n{desc}")
            if group:
                excl = " (exclusive)" if self._tag_exclusive(name) else ""
                tt_parts.append(f"\nGroup: {group}{excl}")
            tt_parts.append("\nClick to filter. Click again or 'All' to clear.")
            tip(tag_btn, "".join(tt_parts))

    # ── Column picker ─────────────────────────────────────────────────────

    def _available_columns(self) -> list[str]:
        """All columns in the current dataset (including internal ones)."""
        df = self.app.filtered_df
        if df.empty:
            return []
        return list(df.columns)

    def _default_columns(self) -> list[str]:
        """Sensible default columns for coding."""
        df = self.app.filtered_df
        if df.empty:
            return ["_tags"]

        def _resolve(canonical):
            if canonical in df.columns:
                return canonical
            mapped = self.app.col_map.get(canonical)
            return mapped if mapped and mapped in df.columns else None

        cols = ["_tags"]
        for c in ["content_text", "platform", "author_username",
                  "created_at", "like_count", "sentiment"]:
            r = _resolve(c)
            if r and r not in cols:
                cols.append(r)
        if len(cols) < 4:
            for c in df.columns:
                if not c.startswith("_") and c not in cols:
                    cols.append(c)
                if len(cols) >= 6:
                    break
        return cols

    def _active_columns(self) -> list[str]:
        """Return user-chosen columns, falling back to defaults."""
        if self._display_cols:
            # Filter out columns that no longer exist
            avail = set(self._available_columns())
            valid = [c for c in self._display_cols if c in avail]
            if valid:
                return valid
        return self._default_columns()

    def _update_col_summary(self):
        cols = self._active_columns()
        if not self._display_cols:
            self._col_summary.configure(text=f"(default — {len(cols)} cols)")
        else:
            names = ", ".join(cols[:4])
            extra = f" +{len(cols)-4}" if len(cols) > 4 else ""
            self._col_summary.configure(text=f"{names}{extra}")

    def _open_column_picker(self):
        avail = self._available_columns()
        if not avail:
            messagebox.showinfo("Columns", "Load a dataset first.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Select Columns")
        win.geometry("400x520")
        win.configure(fg_color=C.BG)
        win.grab_set()
        win.focus_force()

        ctk.CTkLabel(win, text="Choose columns to display",
                     font=("Segoe UI", 14, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=16,
                                             pady=(12, 4))
        ctk.CTkLabel(win,
                     text="Check the columns you want to see in the\n"
                          "coding table. _tags is always included.",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=16, pady=(0, 8))

        # Select all / deselect all
        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 4))

        active = set(self._active_columns())
        checks: dict[str, tk.BooleanVar] = {}

        sf = ctk.CTkScrollableFrame(win, fg_color=C.PANEL, height=340)
        sf.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        for col in avail:
            var = tk.BooleanVar(value=(col in active))
            checks[col] = var
            ctk.CTkCheckBox(sf, text=col, variable=var,
                            font=("Segoe UI", 11),
                            text_color=C.TEXT).pack(anchor="w", padx=8,
                                                     pady=2)

        def select_all():
            for v in checks.values():
                v.set(True)

        def deselect_all():
            for v in checks.values():
                v.set(False)
            checks.get("_tags", tk.BooleanVar()).set(True)

        ctk.CTkButton(btn_row, text="Select All", fg_color=C.BTN,
                      width=80, height=24, command=select_all
                      ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row, text="Deselect All", fg_color=C.BTN,
                      width=90, height=24, command=deselect_all
                      ).pack(side="left")

        def apply():
            chosen = [c for c in avail if checks[c].get()]
            if "_tags" not in chosen:
                chosen.insert(0, "_tags")
            self._display_cols = chosen
            win.destroy()
            self._update_col_summary()
            self._refresh_table()

        ctk.CTkButton(win, text="Apply", fg_color=C.ACCENT, width=100,
                      command=apply).pack(pady=(0, 12))

    def _reset_columns(self):
        self._display_cols = []
        self._update_col_summary()
        self._refresh_table()

    # ── Table ────────────────────────────────────────────────────────────────

    def _refresh_table(self):
        df = self.app.filtered_df
        self._tv.delete(*self._tv.get_children())
        self._update_col_summary()
        if df.empty:
            return

        cols = self._active_columns()

        self._tv.configure(columns=cols)
        widths = {"_tags": 120, "content_text": 320, "platform": 110,
                  "author_username": 140, "created_at": 160}
        for col in cols:
            self._tv.heading(col, text=col)
            self._tv.column(col, width=widths.get(col, 180))
        head = df.head(self.app.row_limit)
        for iid, row in enumerate(head[cols].values):
            self._tv.insert("", "end", iid=str(iid),
                            values=[str(v)[:200] if pd.notna(v) else "" for v in row])

    # ── Reading pane ─────────────────────────────────────────────────────────

    def _on_select(self, _event=None):
        """Update reading pane when selection changes."""
        sel = self._tv.selection()
        if not sel:
            return
        pos = int(sel[0])
        df = self.app.filtered_df
        if pos >= len(df):
            return
        row = df.iloc[pos]

        # Metadata line
        parts = []
        tags_str = str(row.get("_tags", "")).strip()
        if tags_str:
            parts.append(f"Tags: {tags_str}")

        def _resolve(c):
            if c in df.columns:
                return c
            mapped = self.app.col_map.get(c)
            return mapped if mapped and mapped in df.columns else None

        for field in ["platform", "author_username", "created_at",
                      "like_count", "sentiment", "language"]:
            col = _resolve(field)
            if col:
                val = row.get(col, "")
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{field}: {str(val)[:80]}")

        self._rp_meta.configure(text="\n".join(parts) if parts
                                else "No metadata")

        # Full text — use user-selected column or auto-detect
        text_col = self._resolve_text_col()
        text = str(row.get(text_col, "")) if text_col else "(no text column — select one above)"

        self._rp_text.configure(state="normal")
        self._rp_text.delete("1.0", "end")
        self._rp_text.insert("1.0", text)
        self._rp_text.configure(state="disabled")

    # ── Text column helpers ──────────────────────────────────────────────────

    def _resolve_text_col(self) -> "str | None":
        """Return the text column chosen by the user.

        This is the single source of truth used by the reading pane,
        bulk tagging, and auto-code.
        """
        choice = self._text_col_var.get() if hasattr(self, "_text_col_var") else ""
        df = self.app.filtered_df if not self.app.filtered_df.empty else self.app.df
        if choice and choice != "— select —" and choice in df.columns:
            return choice
        return None

    def _refresh_text_col_options(self):
        """Populate the text-column dropdown with all available columns."""
        df = self.app.filtered_df if not self.app.filtered_df.empty else self.app.df
        if df.empty:
            self._text_col_om.configure(values=["— select —"])
            return
        cols = [c for c in df.columns if not c.startswith("_")]
        self._text_col_om.configure(values=cols if cols else ["— select —"])
        # Auto-select on first load if nothing chosen yet
        cur = self._text_col_var.get()
        if cur == "— select —" or cur not in cols:
            # Try to auto-detect a sensible default
            for candidate in ["content_text",
                              self.app.col_map.get("content_text", "")]:
                if candidate and candidate in cols:
                    self._text_col_var.set(candidate)
                    return
            # Fallback: pick first column (user can change)
            if cols:
                self._text_col_var.set(cols[0])

    def _on_text_col_change(self, _value=None):
        """Re-render reading pane when user picks a different text column."""
        self._on_select()

    # ── Tag application ─────────────────────────────────────────────────────

    def _apply_tag(self):
        tag = self._tag_apply_var.get()
        if not tag or tag == "—":
            return
        self._apply_tag_to_selection(tag)

    def _apply_tag_to_selection(self, tag: str):
        """Apply *tag* to all selected rows, respecting group exclusivity."""
        if "_tags" not in self.app.df.columns:
            return
        sel = self._tv.selection()
        if not sel:
            return

        group = self._tag_group(tag)
        exclusive = self._tag_exclusive(tag)
        exclude_tags = set()
        if group and exclusive:
            exclude_tags = set(self._group_members(group)) - {tag}

        df = self.app.filtered_df
        now = datetime.datetime.now().isoformat(timespec="seconds")

        for iid in sel:
            pos = int(iid)
            if pos >= len(df):
                continue
            df_idx = df.index[pos]
            old_val = str(self.app.df.at[df_idx, "_tags"])

            # Save for undo
            if len(self._undo_stack) >= MAX_UNDO:
                self._undo_stack.pop(0)
            self._undo_stack.append((df_idx, old_val))

            # Build new tag string
            existing = set(t.strip() for t in old_val.split(";") if t.strip())
            if exclude_tags:
                existing -= exclude_tags
            existing.add(tag)
            self.app.df.at[df_idx, "_tags"] = _normalize_tags(
                ";".join(existing))

            # Audit trail
            if "_tagged_at" in self.app.df.columns:
                self.app.df.at[df_idx, "_tagged_at"] = now

        self.app.apply_filters()
        self._refresh_table()
        self._refresh_progress()

        # Re-select and show same position
        if sel:
            try:
                self._tv.selection_set(sel[0])
                self._tv.focus(sel[0])
                self._on_select()
            except Exception:
                pass

    def _shortcut(self, index: int):
        """Keyboard shortcut: apply tag by index and advance."""
        names = self._tag_names()
        if index >= len(names):
            return
        tag = names[index]
        self._apply_tag_to_selection(tag)
        # Advance to next row
        sel = self._tv.selection()
        if sel:
            next_iid = self._tv.next(sel[-1])
            if next_iid:
                self._tv.selection_set(next_iid)
                self._tv.see(next_iid)
                self._tv.focus(next_iid)
                self._on_select()

    def _undo(self):
        """Undo the last tag application."""
        if not self._undo_stack:
            return
        df_idx, old_val = self._undo_stack.pop()
        try:
            self.app.df.at[df_idx, "_tags"] = old_val
        except Exception:
            return
        self.app.apply_filters()
        self._refresh_table()
        self._refresh_progress()

    # ── Bulk tag ─────────────────────────────────────────────────────────────

    def _bulk_tag(self):
        tag = self._tag_apply_var.get()
        if not tag or tag == "—":
            messagebox.showinfo("Bulk Tag", "Select a tag first.")
            return
        keyword = self._bulk_var.get().strip()
        if not keyword:
            messagebox.showinfo("Bulk Tag", "Enter a search keyword.")
            return
        if "_tags" not in self.app.df.columns:
            return

        text_col = self._resolve_text_col()
        if not text_col or text_col not in self.app.df.columns:
            messagebox.showinfo("Bulk Tag",
                                "No text column selected.\n\n"
                                "Choose one from the 'Text column' dropdown\n"
                                "at the top of the Coding page.")
            return

        mask = self.app.df[text_col].astype(str).str.contains(
            re.escape(keyword), case=False, na=False)
        n_match = int(mask.sum())

        if n_match == 0:
            messagebox.showinfo("Bulk Tag",
                                f"No rows match '{keyword}'.")
            return

        if not messagebox.askyesno(
                "Bulk Tag",
                f"Apply tag '{tag}' to {n_match:,} rows matching "
                f"'{keyword}'?"):
            return

        group = self._tag_group(tag)
        exclusive = self._tag_exclusive(tag)
        exclude_tags = set()
        if group and exclusive:
            exclude_tags = set(self._group_members(group)) - {tag}

        now = datetime.datetime.now().isoformat(timespec="seconds")
        _bulk_add_tag(self.app.df, mask, tag, exclude_tags)
        if "_tagged_at" in self.app.df.columns:
            self.app.df.loc[mask, "_tagged_at"] = now

        self.app.apply_filters()
        self._refresh_table()
        self._refresh_progress()
        self._bulk_var.set("")
        messagebox.showinfo("Bulk Tag",
                            f"Tagged {n_match:,} rows as '{tag}'.")

    # ── New tag dialog ───────────────────────────────────────────────────────

    def _new_tag(self):
        win = ctk.CTkToplevel(self)
        win.title("New Tag")
        win.geometry("420x300")
        win.configure(fg_color=C.BG)
        win.grab_set()
        win.focus_force()

        ctk.CTkLabel(win, text="Create a New Tag",
                     font=("Segoe UI", 14, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=16,
                                             pady=(12, 8))

        # Name
        ctk.CTkLabel(win, text="Name:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=16)
        name_var = tk.StringVar()
        ctk.CTkEntry(win, textvariable=name_var, width=300,
                     placeholder_text="e.g. misinformation"
                     ).pack(anchor="w", padx=16, pady=(0, 6))

        # Description
        ctk.CTkLabel(win, text="Description (codebook):", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=16)
        desc_var = tk.StringVar()
        ctk.CTkEntry(win, textvariable=desc_var, width=300,
                     placeholder_text="e.g. Contains demonstrably false claims"
                     ).pack(anchor="w", padx=16, pady=(0, 6))

        # Group
        ctk.CTkLabel(win, text="Group (optional):", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=16)
        group_var = tk.StringVar()
        existing = self._existing_groups()
        vals = ["(none)"] + existing + ["+ New group…"]
        ctk.CTkOptionMenu(win, variable=group_var, values=vals, width=200
                          ).pack(anchor="w", padx=16, pady=(0, 4))
        group_var.set("(none)")

        # New group entry (shown when "+ New group…" selected)
        new_group_var = tk.StringVar()
        new_group_entry = ctk.CTkEntry(win, textvariable=new_group_var,
                                       width=200,
                                       placeholder_text="New group name")

        def on_group_change(*_):
            if group_var.get() == "+ New group…":
                new_group_entry.pack(anchor="w", padx=16, pady=(0, 4))
            else:
                new_group_entry.pack_forget()
        group_var.trace_add("write", on_group_change)

        # Exclusive
        excl_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(win, text="Mutual exclusivity within group",
                        variable=excl_var
                        ).pack(anchor="w", padx=16, pady=(4, 8))

        def create():
            name = name_var.get().strip()
            if not name:
                return
            if name in self.app.tags:
                messagebox.showinfo("Exists", f"Tag '{name}' already exists.",
                                    parent=win)
                return
            group = group_var.get()
            if group == "(none)":
                group = ""
            elif group == "+ New group…":
                group = new_group_var.get().strip()

            color = C.TAG_PALETTE[
                len(self.app.tags) % len(C.TAG_PALETTE)]
            self.app.tags[name] = {
                "color": color,
                "desc": desc_var.get().strip(),
                "group": group,
                "exclusive": excl_var.get() if group else False,
            }
            win.destroy()
            self._refresh_tags()
            self._refresh_progress()

        ctk.CTkButton(win, text="Create Tag", fg_color=C.ACCENT, width=120,
                      command=create).pack(pady=(0, 12))

    # ── Manage tags dialog ───────────────────────────────────────────────────

    def _manage_tags_dialog(self):
        if not self.app.tags:
            messagebox.showinfo("Manage Tags", "No tags created yet.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Manage Tags")
        win.geometry("560x480")
        win.configure(fg_color=C.BG)
        win.grab_set()
        win.focus_force()

        ctk.CTkLabel(win, text="Manage Tags",
                     font=("Segoe UI", 14, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=16,
                                             pady=(12, 8))

        sf = ctk.CTkScrollableFrame(win, fg_color=C.BG, height=280)
        sf.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        def rebuild_list():
            for w in sf.winfo_children():
                w.destroy()
            for name in list(self.app.tags.keys()):
                card = ctk.CTkFrame(sf, fg_color=C.CARD, corner_radius=6)
                card.pack(fill="x", pady=2)
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.pack(fill="x", padx=8, pady=6)

                color = self._tag_color(name)
                ctk.CTkLabel(row, text="●", text_color=color,
                             font=("Segoe UI", 14)).pack(side="left",
                                                          padx=(0, 6))
                ctk.CTkLabel(row, text=name, font=("Segoe UI", 12, "bold"),
                             text_color=C.TEXT).pack(side="left")

                desc = self._tag_desc(name)
                group = self._tag_group(name)
                meta_parts = []
                if desc:
                    meta_parts.append(desc[:50])
                if group:
                    excl = " (excl)" if self._tag_exclusive(name) else ""
                    meta_parts.append(f"group: {group}{excl}")
                if meta_parts:
                    ctk.CTkLabel(row, text="  |  ".join(meta_parts),
                                 text_color=C.MUTED,
                                 font=("Segoe UI", 10)
                                 ).pack(side="left", padx=12)

                ctk.CTkButton(
                    row, text="Delete", width=55, height=24,
                    fg_color=C.BTN, font=("Segoe UI", 9),
                    command=lambda n=name: do_delete(n)
                ).pack(side="right", padx=2)
                ctk.CTkButton(
                    row, text="Rename", width=60, height=24,
                    fg_color=C.BTN, font=("Segoe UI", 9),
                    command=lambda n=name: do_rename(n)
                ).pack(side="right", padx=2)
                ctk.CTkButton(
                    row, text="Edit", width=45, height=24,
                    fg_color=C.BTN, font=("Segoe UI", 9),
                    command=lambda n=name: do_edit(n)
                ).pack(side="right", padx=2)

        def do_delete(name):
            if not messagebox.askyesno("Delete Tag",
                                       f"Delete tag '{name}'?\n\n"
                                       "This removes it from all rows.",
                                       parent=win):
                return
            self.app.tags.pop(name, None)
            if "_tags" in self.app.df.columns:
                _remove_tag_from_column(self.app.df, name)
            self.app.apply_filters()
            rebuild_list()
            self._refresh_tags()
            self._refresh_table()
            self._refresh_progress()

        def do_rename(name):
            new = simpledialog.askstring("Rename Tag", f"New name for '{name}':",
                                        parent=win)
            if not new or new == name:
                return
            new = new.strip()
            if new in self.app.tags:
                messagebox.showinfo("Exists", f"'{new}' already exists.",
                                    parent=win)
                return
            info = self.app.tags.pop(name)
            self.app.tags[new] = info
            if "_tags" in self.app.df.columns:
                _rename_tag_in_column(self.app.df, name, new)
            self.app.apply_filters()
            rebuild_list()
            self._refresh_tags()
            self._refresh_table()

        def do_edit(name):
            info = self.app.tags.get(name, {})
            if isinstance(info, str):
                info = {"color": info, "desc": "", "group": "",
                        "exclusive": False}
            new_desc = simpledialog.askstring(
                "Edit Description",
                f"Description for '{name}':",
                initialvalue=info.get("desc", ""),
                parent=win)
            if new_desc is not None:
                info["desc"] = new_desc.strip()
                self.app.tags[name] = info
                rebuild_list()
                self._refresh_tags()

        # Merge section
        ctk.CTkLabel(win, text="Merge Tags", font=("Segoe UI", 12, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=16,
                                             pady=(8, 4))
        merge_row = ctk.CTkFrame(win, fg_color="transparent")
        merge_row.pack(fill="x", padx=16, pady=(0, 12))
        names = self._tag_names()
        m1_var = tk.StringVar(value=names[0] if names else "")
        m2_var = tk.StringVar(value=names[1] if len(names) > 1 else "")
        ctk.CTkOptionMenu(merge_row, variable=m1_var, values=names,
                          width=140).pack(side="left", padx=(0, 4))
        ctk.CTkLabel(merge_row, text="→", text_color=C.MUTED
                     ).pack(side="left", padx=4)
        ctk.CTkOptionMenu(merge_row, variable=m2_var, values=names,
                          width=140).pack(side="left", padx=(0, 8))

        def do_merge():
            src = m1_var.get()
            dst = m2_var.get()
            if not src or not dst or src == dst:
                return
            if not messagebox.askyesno(
                    "Merge Tags",
                    f"Merge '{src}' into '{dst}'?\n\n"
                    f"All rows tagged '{src}' will be retagged as '{dst}'.\n"
                    f"Tag '{src}' will be deleted.",
                    parent=win):
                return
            self.app.tags.pop(src, None)
            if "_tags" in self.app.df.columns:
                _rename_tag_in_column(self.app.df, src, dst)
            self.app.apply_filters()
            rebuild_list()
            self._refresh_tags()
            self._refresh_table()
            self._refresh_progress()

        ctk.CTkButton(merge_row, text="Merge", fg_color=C.SUCCESS, width=70,
                      command=do_merge).pack(side="left")

        rebuild_list()

    # ── Auto-code dialog ─────────────────────────────────────────────────────

    def _auto_code_dialog(self):
        if not self.app.tags:
            messagebox.showinfo("Auto-Code", "Create at least one tag first.")
            return
        text_col = self._resolve_text_col()
        if not text_col or text_col not in self.app.df.columns:
            messagebox.showinfo("Auto-Code",
                                "No text column selected.\n\n"
                                "Choose one from the 'Text column' dropdown\n"
                                "at the top of the Coding page.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Auto-Code Rules")
        win.geometry("540x420")
        win.configure(fg_color=C.BG)
        win.grab_set()
        win.focus_force()

        ctk.CTkLabel(win, text="Auto-Code Rules",
                     font=("Segoe UI", 14, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=16,
                                             pady=(12, 4))
        ctk.CTkLabel(win,
                     text="Define regex patterns to auto-tag matching rows.\n"
                          "Case-insensitive. Applied to the text column.",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=16, pady=(0, 8))

        # Add rule row
        add_row = ctk.CTkFrame(win, fg_color="transparent")
        add_row.pack(fill="x", padx=16, pady=(0, 8))
        pat_var = tk.StringVar()
        ctk.CTkEntry(add_row, textvariable=pat_var, width=250,
                     placeholder_text="Regex pattern, e.g. vaccin"
                     ).pack(side="left", padx=(0, 8))
        rule_tag_var = tk.StringVar(value=self._tag_names()[0])
        ctk.CTkOptionMenu(add_row, variable=rule_tag_var,
                          values=self._tag_names(), width=120
                          ).pack(side="left", padx=(0, 8))

        rules_frame = ctk.CTkScrollableFrame(win, fg_color=C.PANEL,
                                             height=180)
        rules_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        def refresh_rules():
            for w in rules_frame.winfo_children():
                w.destroy()
            if not self._auto_rules:
                ctk.CTkLabel(rules_frame, text="No rules defined.",
                             text_color=C.MUTED,
                             font=("Segoe UI", 10)).pack(pady=12)
                return
            for i, rule in enumerate(self._auto_rules):
                r = ctk.CTkFrame(rules_frame, fg_color=C.CARD,
                                 corner_radius=4)
                r.pack(fill="x", pady=2)
                ctk.CTkLabel(
                    r, text=f"/{rule['pattern']}/  →  {rule['tag']}",
                    text_color=C.TEXT, font=("Segoe UI", 11)
                ).pack(side="left", padx=8, pady=4)
                ctk.CTkButton(
                    r, text="X", width=24, height=24, fg_color=C.BTN,
                    command=lambda idx=i: remove_rule(idx)
                ).pack(side="right", padx=4)

        def add_rule():
            pattern = pat_var.get().strip()
            tag = rule_tag_var.get()
            if not pattern or not tag:
                return
            try:
                re.compile(pattern)
            except re.error as e:
                messagebox.showerror("Invalid Regex", str(e), parent=win)
                return
            self._auto_rules.append({"pattern": pattern, "tag": tag})
            pat_var.set("")
            refresh_rules()

        def remove_rule(idx):
            if 0 <= idx < len(self._auto_rules):
                self._auto_rules.pop(idx)
                refresh_rules()

        ctk.CTkButton(add_row, text="+ Add", fg_color=C.ACCENT, width=70,
                      command=add_rule).pack(side="left")

        def run_rules():
            if not self._auto_rules:
                return
            total = 0
            now = datetime.datetime.now().isoformat(timespec="seconds")
            for rule in self._auto_rules:
                mask = self.app.df[text_col].astype(str).str.contains(
                    rule["pattern"], case=False, na=False, regex=True)
                tag = rule["tag"]
                group = self._tag_group(tag)
                exclusive = self._tag_exclusive(tag)
                exclude = set()
                if group and exclusive:
                    exclude = set(self._group_members(group)) - {tag}

                # Only tag rows that don't already have this tag
                already = self.app.df.loc[mask, "_tags"].astype(str).str.contains(
                    rf"(?:^|;)\s*{re.escape(tag)}\s*(?:;|$)", regex=True, na=False)
                new_mask = mask & ~already
                n_new = int(new_mask.sum())
                if n_new > 0:
                    _bulk_add_tag(self.app.df, new_mask, tag, exclude)
                    if "_tagged_at" in self.app.df.columns:
                        self.app.df.loc[new_mask, "_tagged_at"] = now
                total += n_new

            self.app.apply_filters()
            self._refresh_table()
            self._refresh_progress()
            messagebox.showinfo("Auto-Code",
                                f"Applied {total:,} tags across all rules.",
                                parent=win)

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(btn_row, text="Run All Rules", fg_color=C.SUCCESS,
                      width=130, command=run_rules).pack(side="left")
        ctk.CTkButton(btn_row, text="Close", fg_color=C.BTN, width=70,
                      command=win.destroy).pack(side="right")

        refresh_rules()

    # ── Agreement statistics ─────────────────────────────────────────────────

    def _agreement_dialog(self):
        df = self.app.df
        if df.empty or "_tags" not in df.columns:
            messagebox.showinfo("Agreement", "Load a dataset first.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Inter-Coder Agreement")
        win.geometry("480x400")
        win.configure(fg_color=C.BG)
        win.grab_set()
        win.focus_force()

        ctk.CTkLabel(win, text="Inter-Coder Agreement",
                     font=("Segoe UI", 14, "bold"),
                     text_color=C.TEXT).pack(anchor="w", padx=16,
                                             pady=(12, 4))
        ctk.CTkLabel(win,
                     text="Compare your tags against another coder's column.\n"
                          "Import a second coder's tags as a column in your dataset,\n"
                          "then select it here to compute Cohen's Kappa.",
                     text_color=C.MUTED, font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=16, pady=(0, 12))

        # Column selector
        other_cols = [c for c in df.columns
                      if c != "_tags" and not c.startswith("_")]
        if not other_cols:
            ctk.CTkLabel(win, text="No comparison column available.",
                         text_color=C.MUTED).pack(pady=20)
            return

        ctk.CTkLabel(win, text="Other coder's column:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=16)
        col_var = tk.StringVar(value=other_cols[0])
        ctk.CTkOptionMenu(win, variable=col_var, values=other_cols,
                          width=200).pack(anchor="w", padx=16, pady=(0, 8))

        # Tag to compare
        names = self._tag_names()
        ctk.CTkLabel(win, text="Tag to compare:", text_color=C.MUTED,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=16)
        tag_var = tk.StringVar(value=names[0] if names else "")
        ctk.CTkOptionMenu(win, variable=tag_var, values=names or ["—"],
                          width=200).pack(anchor="w", padx=16, pady=(0, 12))

        result_label = ctk.CTkLabel(win, text="", text_color=C.TEXT,
                                    font=("Segoe UI", 11), justify="left",
                                    wraplength=400)
        result_label.pack(anchor="w", padx=16, pady=(0, 12))

        def calculate():
            other_col = col_var.get()
            tag = tag_var.get()
            if not other_col or not tag:
                return

            # Build binary arrays: does each row have the tag?
            coder1 = df["_tags"].astype(str).apply(
                lambda s: int(tag in [t.strip() for t in s.split(";")]))
            coder2 = df[other_col].astype(str).apply(
                lambda s: int(tag in [t.strip() for t in s.split(";")]))

            n = len(df)
            agree = int((coder1 == coder2).sum())
            both_yes = int(((coder1 == 1) & (coder2 == 1)).sum())
            c1_yes = int(coder1.sum())
            c2_yes = int(coder2.sum())

            # Cohen's Kappa (manual to avoid sklearn dependency)
            po = agree / n if n else 0
            pe1 = (c1_yes / n) * (c2_yes / n) if n else 0
            pe0 = ((n - c1_yes) / n) * ((n - c2_yes) / n) if n else 0
            pe = pe1 + pe0
            kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0

            # Interpretation
            if kappa >= 0.81:
                interp = "almost perfect"
            elif kappa >= 0.61:
                interp = "substantial"
            elif kappa >= 0.41:
                interp = "moderate"
            elif kappa >= 0.21:
                interp = "fair"
            else:
                interp = "slight / poor"

            result_label.configure(
                text=f"Cohen's Kappa: {kappa:.3f}  ({interp})\n"
                     f"Agreement: {agree:,} / {n:,}  "
                     f"({po*100:.1f}%)\n\n"
                     f"Your tags ('{tag}'): {c1_yes:,} rows\n"
                     f"Other coder: {c2_yes:,} rows\n"
                     f"Both agree (yes): {both_yes:,} rows")

        ctk.CTkButton(win, text="Calculate", fg_color=C.ACCENT, width=120,
                      command=calculate).pack(anchor="w", padx=16)

    # ── Filter ───────────────────────────────────────────────────────────────

    def _filter_by_tag(self, tag: str):
        current = self.app.filter_state.get("tag", "")
        if current == tag:
            self._clear_tag_filter()
            return
        self.app.filter_state["tag"] = tag
        self.app.apply_filters()
        self._refresh_tags()
        self._refresh_table()

    def _clear_tag_filter(self):
        self.app.filter_state["tag"] = ""
        self.app.apply_filters()
        self._refresh_tags()
        self._refresh_table()

    # ── Theme rebuild ────────────────────────────────────────────────────────

    def rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._build()
