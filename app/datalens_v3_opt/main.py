"""Entry point — run Data Dominion."""

import sys
import os
from pathlib import Path

# Ensure the parent directory is on sys.path so 'datalens_v3_opt' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional ML packages: if the user installed them to ~/.datalens/ml_packages/,
# add that directory to sys.path so they are importable without polluting the
# system Python installation.
_ml_path = Path.home() / ".datalens" / "ml_packages"
if _ml_path.exists():
    sys.path.insert(0, str(_ml_path))

from datalens_v3_opt.ui.app import App

if __name__ == "__main__":
    App().mainloop()
