"""Top-level entry point for Data Dominion.

Works in three modes:
  - Dev mode:      python datadominion.py
  - Embedded dist: runtime/python.exe app/datadominion.py
  - Nuitka exe:    DataDominion.exe (legacy)
"""

import sys
import os
from pathlib import Path

# Find the app root directory (where DataDominion.exe / runtime/ / app/ live)
_script_dir = Path(os.path.abspath(sys.argv[0])).parent  # .../app/
_app_root = _script_dir.parent  # .../DataDominion/

# Optional ML packages — check all possible locations
for _ml_candidate in [
    _app_root / "ml_packages",               # shipped alongside (embedded dist)
    _script_dir / "ml_packages",             # next to script (dev mode)
    Path(sys.executable).resolve().parent / "ml_packages",  # next to exe (Nuitka)
    Path.home() / ".datalens" / "ml_packages",  # user-installed
]:
    if _ml_candidate.is_dir():
        _p = str(_ml_candidate)
        if _p not in sys.path:
            sys.path.insert(0, _p)

from datalens_v3_opt.ui.app import App

if __name__ == "__main__":
    App().mainloop()
