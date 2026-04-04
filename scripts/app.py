"""Legacy Streamlit dashboard entrypoint.

This file is kept for backward compatibility. The canonical dashboard lives in
`apps/dashboard.py`.

Preferred:
  streamlit run apps/dashboard.py

Legacy:
  streamlit run scripts/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_path()

# Import runs the Streamlit app top-level.
import apps.dashboard  # noqa: F401
