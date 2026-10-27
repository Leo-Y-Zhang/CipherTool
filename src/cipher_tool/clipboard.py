"""Best-effort copying to the local system clipboard, using installed tools."""

from __future__ import annotations

import shutil
import subprocess
import sys


def copy_to_clipboard(text: str) -> bool:
    """Copy exact text without a trailing newline; unavailable tools are OK."""
    if sys.platform == "win32":
        commands = (("clip",),)
        payload = text.encode("utf-16le")
    elif sys.platform == "darwin":
        commands = (("pbcopy",),)
        payload = text.encode("utf-8")
    elif sys.platform.startswith("linux"):
        commands = (("wl-copy",), ("xclip", "-selection", "clipboard"))
        payload = text.encode("utf-8")
    else:
        return False

    for command in commands:
        try:
            executable = shutil.which(command[0])
            if executable is None:
                continue
            completed = subprocess.run(
                [executable, *command[1:]], input=payload,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=2, check=False,
            )
            if completed.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False
