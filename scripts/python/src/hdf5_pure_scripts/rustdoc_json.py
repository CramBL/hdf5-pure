import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from hdf5_pure_scripts import nightly


def build(manifest: Path, target_dir: Path, features: list[str], locked: bool = True) -> dict:
    toolchain = nightly.require()
    crate = tomllib.loads(manifest.read_text())["package"]["name"].replace("-", "_")
    path = target_dir / "doc" / f"{crate}.json"
    path.unlink(missing_ok=True)
    subprocess.run(
        [
            "cargo",
            f"+{toolchain}",
            "rustdoc",
            "--manifest-path",
            str(manifest),
            *(["--locked"] if locked else []),
            "--target-dir",
            str(target_dir),
            "--lib",
            *features,
            "--",
            "-Zunstable-options",
            "--output-format",
            "json",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        env={
            **os.environ,
            "RUSTDOCFLAGS": "-A rustdoc::private_doc_tests "
            "-A rustdoc::broken_intra_doc_links "
            "-A rustdoc::private_intra_doc_links",
        },
    )
    if not path.is_file():
        sys.exit(f"rustdoc wrote no JSON at {path}")
    return json.loads(path.read_text())
