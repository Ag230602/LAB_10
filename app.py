"""Convenience Streamlit entrypoint.

This repo's canonical Streamlit dashboard is `apps/dashboard.py`.

This file exists so you can run either of these and get the same app:
  - streamlit run app.py
  - streamlit run apps/dashboard.py

(There is also a legacy wrapper at `scripts/app.py`.)
"""

from __future__ import annotations

# Import runs the Streamlit app at module import time.
import apps.dashboard  # noqa: F401
