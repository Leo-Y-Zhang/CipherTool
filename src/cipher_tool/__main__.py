"""Allow ``python -m cipher_tool`` as well as the installed script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
