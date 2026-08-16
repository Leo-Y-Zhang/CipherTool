#!/usr/bin/env bash
# ===================================================================
#  cipher_tool -- one-click launcher for macOS and Linux.
#
#  On a Mac: double-click this file in Finder to open the interactive
#  shell. (The first time, macOS may ask you to confirm -- right-click
#  and choose Open.)
#
#  From a terminal, arguments are passed straight through:
#      ./cipher_tool.command analyse message.txt
#      ./cipher_tool.command auto message.txt --fast
#
#  Nothing needs installing. There are no dependencies.
# ===================================================================
set -u

cd "$(dirname "$0")" || exit 1

find_python() {
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

PY="$(find_python)" || {
    cat <<'MESSAGE'

  No suitable Python was found.

  cipher_tool needs Python 3.10 or newer.

    macOS   : install from https://www.python.org/downloads/
              or run:  brew install python
    Ubuntu  : sudo apt install python3
    Fedora  : sudo dnf install python3

MESSAGE
    # Keep the window open when double-clicked from Finder.
    [ -t 0 ] && read -r -p "Press Return to close. " _
    exit 1
}

# Run straight from src/ so the toolkit works without being installed.
export PYTHONPATH="$PWD/src"

if [ "$#" -gt 0 ]; then
    exec "$PY" -m cipher_tool "$@"
fi

cat <<'BANNER'

  cipher_tool -- offline cryptanalysis toolkit
  -------------------------------------------
  Type 'help' for commands, or 'quit' to leave.

  Quick start:
    load /path/to/message.txt
    analyse
    auto fast

BANNER

"$PY" -m cipher_tool shell

echo
[ -t 0 ] && read -r -p "Press Return to close. " _
exit 0
