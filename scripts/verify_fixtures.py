# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy"]
#
# [tool.uv]
# # h5py's PyPI wheels vendor their own libhdf5. Checking this crate's output
# # against that one would say nothing about the version under test, so h5py is
# # built here against whatever HDF5_DIR points at.
# no-binary-package = ["h5py"]
# ///
"""Check the fixture files against the manifest `libver_fixtures` wrote.

    uv run scripts/verify_fixtures.py <fixture-dir> [--file-prefix repacked-]

Every expectation comes from `fixtures.json`, next to the files, so the values
checked here are the ones the writer used. Reading through h5py rather than
h5dump's text means a reference is followed rather than recognized, and a
version that prints differently is not a difference at all.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np


def scalars(value):
    """A dataset or attribute value as a flat list of Python scalars."""
    array = np.asarray(value)
    if array.dtype.names:  # compound: each member, in declaration order
        return [item for element in array.ravel() for item in element]
    return list(array.ravel())


def normalize(value):
    """Bytes to str, numpy scalars to Python ones, char sequences to one string.

    `MATLAB_fields` is a vlen of one-character strings per field name, so an
    element of it arrives as an array rather than a scalar.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        parts = [normalize(item) for item in value.ravel()]
        return "".join(parts) if all(isinstance(p, str) for p in parts) else parts
    return value


def equal(got, want) -> bool:
    got, want = normalize(got), normalize(want)
    if isinstance(got, float) or isinstance(want, float):
        return math.isclose(float(got), float(want), rel_tol=1e-12, abs_tol=0.0)
    return got == want


class Checker:
    def __init__(self, directory: Path, prefix: str) -> None:
        self.directory = directory
        self.prefix = prefix
        self.failures = 0

    def report(self, ok: bool, description: str, detail: str) -> None:
        if ok:
            print(f"  ok   {description} — {detail}")
        else:
            print(f"  FAIL {description} — {detail}")
            self.failures += 1

    def run(self, check: dict) -> None:
        name = self.prefix + check["file"]
        path = check["path"]
        kind = check["kind"]
        description = f"{name} {path}" + (f" {check['name']}" if "name" in check else "")

        try:
            with h5py.File(self.directory / name, "r") as handle:
                getattr(self, f"check_{kind}")(handle, check, description)
        except Exception as error:  # a missing object is a failure, not a crash
            self.report(False, f"{description} [{kind}]", f"{type(error).__name__}: {error}")

    def check_data(self, handle, check, description) -> None:
        dataset = handle[check["path"]]
        got = [normalize(v) for v in scalars(dataset[...])]
        want = check["data"]
        ok = len(got) == len(want) and all(equal(g, w) for g, w in zip(got, want))
        if ok and "dims" in check:
            ok = list(dataset.shape) == check["dims"]
            self.report(ok, description, f"shape {list(dataset.shape)}")
            return
        self.report(ok, description, f"read {got}")

    def check_attr(self, handle, check, description) -> None:
        value = handle[check["path"]].attrs[check["name"]]
        got = [normalize(v) for v in scalars(value)]
        want = check["data"] if isinstance(check["data"], list) else [check["data"]]
        ok = len(got) == len(want) and all(equal(g, w) for g, w in zip(got, want))
        self.report(ok, description, f"read {got}")

    def check_no_attr(self, handle, check, description) -> None:
        present = check["name"] in handle[check["path"]].attrs
        self.report(not present, description, "absent" if not present else "present")

    def check_refs(self, handle, check, description) -> None:
        dataset = handle[check["path"]]
        got = [handle[reference].name for reference in dataset[...].ravel()]
        self.report(got == check["targets"], description, f"resolved {got}")

    def check_links(self, handle, check, description) -> None:
        got = len(handle[check["path"]])
        self.report(got == check["count"], description, f"{got} links")

    def check_attrs(self, handle, check, description) -> None:
        got = len(handle[check["path"]].attrs)
        self.report(got == check["count"], description, f"{got} attributes")

    def check_named_type(self, handle, check, description) -> None:
        got = isinstance(handle[check["path"]], h5py.Datatype)
        self.report(got, description, "a committed datatype" if got else "not a datatype")

    def check_members(self, handle, check, description) -> None:
        names = handle[check["path"]].dtype.names or ()
        self.report(len(names) == check["count"], description, f"members {list(names)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest", default="fixtures.json")
    parser.add_argument("--file-prefix", default="")
    parser.add_argument(
        "--expect-hdf5",
        metavar="VERSION",
        help="fail unless h5py is linked against this library, so a cached "
        "build cannot stand in for the version under test",
    )
    parser.add_argument(
        "--skip-file",
        action="append",
        default=[],
        metavar="NAME",
        help="a fixture this library cannot open, so its checks do not apply",
    )
    args = parser.parse_args()

    checks = json.loads((args.directory / args.manifest).read_text())
    checks = [check for check in checks if check["file"] not in args.skip_file]
    linked = h5py.version.hdf5_version
    print(f"==> h5py {h5py.__version__} on HDF5 {linked}")

    if args.expect_hdf5 and linked != args.expect_hdf5:
        print(
            f"h5py is linked against HDF5 {linked}, not the {args.expect_hdf5} "
            f"under test: rebuild it with `uv run --refresh-package h5py`",
            file=sys.stderr,
        )
        return 2

    checker = Checker(args.directory, args.file_prefix)
    for check in checks:
        checker.run(check)

    print()
    if checker.failures:
        print(f"{checker.failures} of {len(checks)} checks failed")
        return 1
    print(f"all {len(checks)} checks passed against HDF5 {h5py.version.hdf5_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
