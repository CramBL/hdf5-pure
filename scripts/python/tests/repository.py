"""Helpers that build the git repositories the tests read."""

import subprocess
from collections.abc import Sequence
from pathlib import Path


def git(root: Path, args: Sequence[str]) -> None:
    """Runs `git` in `root` under a test identity, raising `CalledProcessError` on a failure."""
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def commit_files(root: Path, files: dict[str, str], message: str) -> None:
    """Writes each of `files` under `root` and commits every change in the working tree."""
    for path, text in files.items():
        (root / path).write_text(text)
    git(root, ["add", "--all"])
    git(root, ["commit", "--quiet", "-m", message])
