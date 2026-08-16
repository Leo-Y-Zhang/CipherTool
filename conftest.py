"""Make ``src/`` importable so the test suite runs without installing.

The toolkit has no runtime dependencies, and we did not want running the tests
to require a build step either. ``pytest`` picks this file up automatically;
for the stdlib runner use ``python run_tests.py``, which does the same thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
