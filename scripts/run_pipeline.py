"""Legacy pipeline entrypoint.

This file is kept for backward compatibility. The canonical entrypoint lives in
`apps/pipeline.py`.

Preferred (no PYTHONPATH needed):
  python -m apps.pipeline --state KS --use-reddit false --use-trends false

Legacy:
  PYTHONPATH=. python scripts/run_pipeline.py --state KS --use-reddit false --use-trends false
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _ensure_repo_root_on_path()
    from apps.pipeline import main as apps_main

    apps_main()


if __name__ == "__main__":
    main()
