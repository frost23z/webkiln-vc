"""Entrypoint for Vercel, which looks for `module:object` relative to this folder.

The real app lives in `src/webkiln_api`. Putting `src` on the path keeps this working even when
the package itself was not installed into the function's environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from webkiln_api.main import app

__all__ = ["app"]
