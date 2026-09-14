#!/bin/bash
set -euo pipefail

# Only do work in Claude Code on the web; local sessions are untouched.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# The toolkit itself needs nothing beyond the stdlib. The venv exists only so
# pytest (the "dev" extra) is available; run_tests.py works with no venv at
# all.
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'

echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
