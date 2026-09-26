import subprocess
import sys

from hdf5_pure_scripts import repo_root


def require() -> str:
    pinned = toolchain()
    installed = subprocess.run(
        ["rustup", "toolchain", "list"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    if not any(line.startswith(f"{pinned}-") for line in installed):
        sys.exit(f"needs the {pinned} toolchain: just nightly")
    return pinned


def toolchain() -> str:
    return (repo_root() / "scripts" / "nightly.txt").read_text().strip()
