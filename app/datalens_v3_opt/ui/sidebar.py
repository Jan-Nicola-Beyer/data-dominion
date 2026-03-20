"""Sidebar navigation panel with ISD logo and theme toggle."""

from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import customtkinter as ctk

import datalens_v3_opt.constants as C
from datalens_v3_opt.ui.widgets import tip, NAV_TIPS

try:
    from PIL import Image
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

def _has_sbert() -> bool:
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None

HAS_SBERT = _has_sbert()

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App


class Sidebar(ctk.CTkFrame):
    """Left-hand navigation panel with ISD logo and icon buttons for each view."""

    ICONS = {
        "home": "⌂", "import": "⤒", "datasets": "⬡",
        "table": "▤", "analytics": "◈", "coding": "⊞",
        "topics": "◉", "slicer": "⧉", "export": "↓", "settings": "⚙",
    }

    def __init__(self, master, navigate_cb: Callable[[str], None], app: "App"):
        super().__init__(master, fg_color=C.PANEL, corner_radius=0, width=200)
        self.pack_propagate(False)
        self._nav_cb = navigate_cb
        self._app    = app
        self._btns: dict = {}
        self._build()

    def _build(self):
        # ── Logo ──────────────────────────────────────────────────────────────
        self._build_logo()

        ctk.CTkFrame(self, fg_color=C.BORDER, height=1, corner_radius=0
                     ).pack(fill="x", padx=12, pady=4)

        # ── Nav buttons ───────────────────────────────────────────────────────
        for label, key in C.NAV_ITEMS:
            icon = self.ICONS.get(key, "•")
            btn = ctk.CTkButton(
                self, text=f"{icon}   {label}", anchor="w",
                fg_color="transparent", hover_color=C.SELECT,
                text_color=C.TEXT, font=("Segoe UI", 12),
                height=40, corner_radius=6,
                command=lambda k=key: self._nav_cb(k))
            btn.pack(fill="x", padx=8, pady=2)
            self._btns[key] = btn
            if key in NAV_TIPS:
                tip(btn, NAV_TIPS[key], wraplength=260)

        # ── Footer ────────────────────────────────────────────────────────────
        ctk.CTkFrame(self, fg_color=C.BORDER, height=1, corner_radius=0
                     ).pack(fill="x", padx=12, pady=8)

        model_txt = "Semantic" if HAS_SBERT else "Fuzzy"
        ctk.CTkLabel(self, text=f"Prediction: {model_txt}",
                     text_color=C.MUTED, font=("Segoe UI", 9)).pack(pady=2)

        # ── Theme toggle ──────────────────────────────────────────────────────
        is_dark = (C.CURRENT_THEME == "dark")
        toggle_btn = ctk.CTkButton(
            self,
            text="☀  Light Mode" if is_dark else "🌙  Dark Mode",
            fg_color=C.DIM, hover_color=C.SELECT, text_color=C.MUTED,
            font=("Segoe UI", 10), height=32, corner_radius=6,
            command=self._app.toggle_theme)
        toggle_btn.pack(fill="x", padx=8, pady=(4, 10))
        tip(toggle_btn, "Switch between dark and light colour theme.")

    def _build_logo(self):
        """Load and display the ISD logo scaled to fit the sidebar."""
        if _PIL_OK:
            try:
                light_path = C.LOGO_DIR / "ISD-logo-Red mixed.png"
                dark_path  = C.LOGO_DIR / "ISD-logo-ISD Red transp.png"

                light_img = Image.open(light_path).convert("RGBA")
                dark_img  = Image.open(dark_path).convert("RGBA")

                # Scale both images to the same target width, then pad to a
                # shared height so CTkImage doesn't reject mismatched sizes.
                target_w = 164
                def _resize(img):
                    w, h = img.size
                    return img.resize((target_w, max(1, int(h * target_w / w))),
                                      Image.LANCZOS)

                light_img = _resize(light_img)
                dark_img  = _resize(dark_img)

                # Unify height by padding the shorter image with transparency
                h = max(light_img.size[1], dark_img.size[1])
                def _pad(img):
                    if img.size[1] == h:
                        return img
                    canvas = Image.new("RGBA", (target_w, h), (0, 0, 0, 0))
                    canvas.paste(img, (0, (h - img.size[1]) // 2))
                    return canvas

                light_img = _pad(light_img)
                dark_img  = _pad(dark_img)

                self._logo_img = ctk.CTkImage(
                    light_image=light_img,
                    dark_image=dark_img,
                    size=(target_w, h))

                ctk.CTkLabel(self, text="", image=self._logo_img,
                             fg_color="transparent").pack(pady=(14, 6))
                return
            except Exception:
                pass

        # Fallback: text logo
        ctk.CTkLabel(self, text="Data Dominion", font=("Segoe UI", 22, "bold"),
                     text_color=C.ACCENT).pack(pady=(18, 0))
        ctk.CTkLabel(self, text="Unified Data Analysis System",
                     font=("Segoe UI", 10), text_color=C.MUTED).pack(pady=(0, 6))

    def set_active(self, key: str):
        for k, btn in self._btns.items():
            btn.configure(fg_color=C.ACCENT if k == key else "transparent")

    def rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self._btns = {}
        self.configure(fg_color=C.PANEL)
        self._build()
        self.set_active(self._app.active_nav)
