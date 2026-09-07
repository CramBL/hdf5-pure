"""Tooling that runs from the repository root, whichever directory invokes it."""

import subprocess
from pathlib import Path


def repo_root() -> Path:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, text=True, capture_output=True
    ).stdout.strip()
    return Path(top)
