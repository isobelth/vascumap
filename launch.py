"""Convenience launcher so you can run from anywhere::

    python C:\\Users\\taylorhearn\\git_repos\\vascumap\\launch.py

This avoids the name clash that occurs when ``python -m vascumap`` is
invoked from *inside* the ``vascumap`` package directory (where Python
would otherwise pick up ``vascumap.py`` instead of the package).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vascumap.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
