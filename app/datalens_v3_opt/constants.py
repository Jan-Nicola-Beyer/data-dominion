"""
Theme constants, navigation items, and global matplotlib style for DataLens V3.
All colour tokens, palette definitions and nav configuration live here.
"""

import pathlib
import sys
import customtkinter as ctk


# ── Exe-safe path helpers ────────────────────────────────────────────────────────

def _asset_root() -> pathlib.Path:
    """Return the package root, works in dev, PyInstaller, and Nuitka."""
    if getattr(sys, "frozen", False):
        # PyInstaller: assets extracted to _MEIPASS temp dir
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return pathlib.Path(meipass) / "datalens_v3_opt"
        # Nuitka --onefile/--standalone: __file__ resolves correctly
        return pathlib.Path(__file__).parent
    return pathlib.Path(__file__).parent


def get_model_cache() -> pathlib.Path:
    """Return a user-writable directory for downloaded ML models."""
    d = pathlib.Path.home() / ".datalens" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Asset paths ─────────────────────────────────────────────────────────────────
LOGO_DIR = _asset_root() / "PNG" / "PNG"

# ── ISD Brand colours ───────────────────────────────────────────────────────────
ISD_RED        = "#C7074D"
ISD_DARK_GREY  = "#5C6771"
ISD_BLUE       = "#0068B2"
ISD_CORAL      = "#E76863"
ISD_PURPLE     = "#4C4193"
ISD_LIGHT_GREY = "#B4B2B1"

# ── Theme definitions ───────────────────────────────────────────────────────────
DARK_THEME: dict = {
    "BG":      "#0d1117",
    "PANEL":   "#111827",
    "CARD":    "#161b22",
    "BORDER":  "#21262d",
    "ACCENT":  ISD_RED,        # ISD Red as primary action colour
    "SELECT":  "#3d0f28",      # dark red tint for selection
    "SUCCESS": "#059669",
    "WARN":    "#d97706",
    "DANGER":  "#dc2626",
    "TEXT":    "#e6edf3",
    "MUTED":   "#6b7280",
    "DIM":     "#374151",
    "BTN":     "#374151",      # secondary button face (visible on dark panels)
    "MPL_BG":      "#161b22",
    "MPL_AX":      "#0d1117",
    "MPL_EDGE":    "#30363d",
    "MPL_LABEL":   "#e6edf3",
    "MPL_TICK":    "#7d8590",
    "MPL_GRID":    "#21262d",
}

LIGHT_THEME: dict = {
    "BG":      "#ffffff",
    "PANEL":   "#f3f4f6",
    "CARD":    "#ffffff",
    "BORDER":  "#d1d5db",
    "ACCENT":  ISD_RED,        # ISD Red stays the same in both themes
    "SELECT":  "#fce7ef",      # light red tint
    "SUCCESS": "#16a34a",
    "WARN":    "#b45309",
    "DANGER":  "#dc2626",
    "TEXT":    "#111827",
    "MUTED":   ISD_DARK_GREY,  # ISD Dark Grey for muted text
    "DIM":     "#e5e7eb",
    "BTN":     ISD_DARK_GREY,  # secondary button face (visible on light panels)
    "MPL_BG":      "#ffffff",
    "MPL_AX":      "#f9fafb",
    "MPL_EDGE":    "#d1d5db",
    "MPL_LABEL":   "#111827",
    "MPL_TICK":    "#6b7280",
    "MPL_GRID":    "#e5e7eb",
}

# ── Current theme (mutable module-level tokens) ─────────────────────────────────
CURRENT_THEME = "dark"

BG      = DARK_THEME["BG"];      PANEL   = DARK_THEME["PANEL"]
CARD    = DARK_THEME["CARD"];    BORDER  = DARK_THEME["BORDER"]
ACCENT  = DARK_THEME["ACCENT"];  SELECT  = DARK_THEME["SELECT"]
SUCCESS = DARK_THEME["SUCCESS"]; WARN    = DARK_THEME["WARN"]
DANGER  = DARK_THEME["DANGER"];  TEXT    = DARK_THEME["TEXT"]
MUTED   = DARK_THEME["MUTED"];   DIM     = DARK_THEME["DIM"]
BTN     = DARK_THEME["BTN"]

# ── Apply theme ─────────────────────────────────────────────────────────────────

def _apply_mpl_style(theme_dict: dict) -> None:
    """Update matplotlib rcParams if matplotlib has been imported."""
    import sys
    if "matplotlib.pyplot" not in sys.modules:
        return
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": theme_dict["MPL_BG"],
        "axes.facecolor":   theme_dict["MPL_AX"],
        "axes.edgecolor":   theme_dict["MPL_EDGE"],
        "axes.labelcolor":  theme_dict["MPL_LABEL"],
        "xtick.color":      theme_dict["MPL_TICK"],
        "ytick.color":      theme_dict["MPL_TICK"],
        "text.color":       theme_dict["MPL_LABEL"],
        "grid.color":       theme_dict["MPL_GRID"],
        "grid.linestyle":   "--",
        "grid.alpha":       0.6,
    })


def ensure_mpl() -> None:
    """Import and configure matplotlib on first use (lazy init)."""
    import sys
    if "matplotlib.pyplot" in sys.modules:
        return
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as _plt  # noqa: F811
    t = DARK_THEME if CURRENT_THEME == "dark" else LIGHT_THEME
    _apply_mpl_style(t)


def apply_theme(name: str) -> None:
    """Update all module-level colour tokens and matplotlib style for *name* ('dark'/'light')."""
    global CURRENT_THEME
    global BG, PANEL, CARD, BORDER, ACCENT, SELECT, SUCCESS, WARN, DANGER, TEXT, MUTED, DIM, BTN
    t = DARK_THEME if name == "dark" else LIGHT_THEME
    CURRENT_THEME = name
    BG = t["BG"]; PANEL = t["PANEL"]; CARD = t["CARD"]; BORDER = t["BORDER"]
    ACCENT = t["ACCENT"]; SELECT = t["SELECT"]; SUCCESS = t["SUCCESS"]
    WARN = t["WARN"]; DANGER = t["DANGER"]; TEXT = t["TEXT"]
    MUTED = t["MUTED"]; DIM = t["DIM"]; BTN = t["BTN"]

    ctk.set_appearance_mode(name)
    _apply_mpl_style(t)


# ── Initial customtkinter style ────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Tag palette ─────────────────────────────────────────────────────────────────
TAG_PALETTE = [
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
    "#3b82f6", "#8b5cf6", "#ec4899", "#64748b", "#84cc16",
]

# ── Navigation items ────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("Home",      "home"),
    ("Import",    "import"),
    ("Datasets",  "datasets"),
    ("Table",     "table"),
    ("Analytics", "analytics"),
    ("Coding",    "coding"),
    ("Topics",    "topics"),
    ("Slicer",    "slicer"),
    ("Export",    "export"),
    ("Settings",  "settings"),
]
