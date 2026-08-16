#!/usr/bin/env python3
"""Run the whole test suite with nothing but the standard library.

    python run_tests.py                 # every test
    python run_tests.py test_caesar     # one module
    python run_tests.py -v              # verbose

``pytest`` also works if you have it (``python -m pytest -q``), but it is not
required: the tests are plain ``unittest`` cases on purpose, so that the
toolkit and its tests together depend on nothing outside the standard library.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    verbosity = 2 if "-v" in argv else 1
    names = [arg for arg in argv if not arg.startswith("-")]

    loader = unittest.TestLoader()
    if names:
        suite = unittest.TestSuite()
        for name in names:
            module = name if name.startswith("tests.") else f"tests.{name}"
            suite.addTests(loader.loadTestsFromName(module))
    else:
        suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py",
                                top_level_dir=str(ROOT))

    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
