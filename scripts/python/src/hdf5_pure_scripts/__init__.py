"""Tooling that runs from the repository root, whichever directory invokes it."""

import subprocess
from pathlib import Path

# `git diff` with its algorithm, hunk merging, renames, paths, color, and drivers set as arguments.
REPRODUCIBLE_DIFF = (
    "-c",
    "core.attributesFile=",
    "-c",
    "core.quotePath=false",
    "diff",
    "--diff-algorithm=myers",
    "--indent-heuristic",
    "--inter-hunk-context=0",
    "--find-renames",
    "-l0",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


def repo_root() -> Path:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, text=True, capture_output=True
    ).stdout.strip()
    return Path(top)
