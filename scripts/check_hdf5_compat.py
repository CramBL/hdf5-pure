# /// script
# requires-python = ">=3.11"
# ///
"""Check what a real HDF5 library makes of the files this crate writes.

    uv run scripts/check_hdf5_compat.py [--prefix <hdf5-install-dir>]

With --prefix, use the HDF5 installed there (conda, apt, brew). Without one,
build HDF5 1.8.23 — the last 1.8 release — under `tmp/`, which takes a few
minutes the first time and is then reused.

Nothing in the test suite can cover this ground: every libhdf5 reachable as a
dependency is modern, and the difference that matters is a superblock version
below everything they parse. That is why `mat::Options` defaults to the 1.8
format, and why this is a command you run rather than a test.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "tmp" / "hdf5-compat-check"
FIXTURES = WORK / "fixtures"

SOURCE_VERSION = "1_8_23"
SOURCE_URL = (
    "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-{v}.tar.gz"
).format(v=SOURCE_VERSION)
CONFIG_SCRIPTS_URL = "https://git.savannah.gnu.org/cgit/config.git/plain/"

# Current h5py requires HDF5 1.10.7 or newer, which conda-forge's 1.10 series
# never reached. Anything below takes the pinned entry point beside this one,
# where h5py 3.9 covers HDF5 back to 1.8.4.
H5PY_FLOOR = (1, 10, 7)

V18_FIXTURES = ["mat_v18.mat", "mat_string_v18.mat", "plain_v18.h5"]
V110_FIXTURES = ["mat_v110.mat", "plain_v110.h5"]


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def build_hdf5(prefix: Path) -> None:
    """Build HDF5 1.8.23 into `prefix`, unless it is already there."""
    if find_tool(prefix, "h5dump"):
        return

    source = WORK / f"hdf5-hdf5-{SOURCE_VERSION}"
    if not source.is_dir():
        print(f"==> downloading HDF5 {SOURCE_VERSION}")
        archive = WORK / "hdf5.tar.gz"
        urllib.request.urlretrieve(SOURCE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(WORK, filter="data")

    refresh_config_scripts(source)

    # 1.8.23 predates GCC 14, which turned several of its warnings into errors.
    relax = (
        "-std=gnu17 -Wno-error=incompatible-pointer-types "
        "-Wno-error=implicit-function-declaration -Wno-error=int-conversion"
    )
    # The high-level library and the shared objects are h5py's build
    # requirements; the tools alone would need neither.
    configure = [
        "./configure",
        f"--prefix={prefix}",
        "--disable-fortran",
        "--disable-cxx",
        "--enable-hl",
        "--enable-shared",
        "--enable-tools",
        f"CFLAGS={os.environ.get('CFLAGS', '')} {relax}",
    ]
    for step, command in [
        ("configuring", configure),
        ("building", ["make", f"-j{os.cpu_count() or 4}"]),
        ("installing", ["make", "install"]),
    ]:
        log = WORK / f"{step}.log"
        print(f"==> {step} (log: {log})")
        result = run(command, cwd=source)
        log.write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            sys.exit(f"{step} failed, see {log}:\n{result.stderr[-2000:]}")


def refresh_config_scripts(source: Path) -> None:
    """Teach an old autotools tree about hosts it predates, such as arm64 macOS.

    Only when it needs teaching: every host the bundled scripts already know
    stays off the network.
    """
    config_sub = source / "bin" / "config.sub"
    if (source / "bin" / "config.sub.orig").exists():
        return

    guess = run([str(source / "bin" / "config.guess")])
    host = guess.stdout.strip()
    if host and run([str(config_sub), host]).returncode == 0:
        return

    print("==> refreshing config.guess / config.sub for this host")
    for name in ("config.sub", "config.guess"):
        script = source / "bin" / name
        shutil.copy(script, script.with_suffix(".orig"))
        urllib.request.urlretrieve(CONFIG_SCRIPTS_URL + name, script)
        script.chmod(0o755)


def find_python(prefix: Path) -> Path | None:
    """A conda environment's own interpreter, which h5py has to be built with.

    Building under another Python loads this environment's HDF5 into a process
    holding another OpenSSL, which fails before h5py's build finds the library.
    """
    for candidate in (prefix / "bin" / "python3", prefix / "bin" / "python",
                      prefix / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


def find_tool(prefix: Path, name: str) -> Path | None:
    """conda puts the tools under `Library/bin` on Windows and `bin` elsewhere."""
    for directory in (prefix / "bin", prefix / "Library" / "bin"):
        for candidate in (directory / name, directory / f"{name}.exe"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def write_fixtures() -> None:
    print("==> writing fixtures")
    shutil.rmtree(FIXTURES, ignore_errors=True)
    result = run(
        ["cargo", "run", "--quiet", "--example", "libver_fixtures",
         "--features", "serde", "--", str(FIXTURES)],
        cwd=REPO,
    )
    if result.returncode != 0:
        sys.exit(f"writing the fixtures failed:\n{result.stderr}")


class Checks:
    def __init__(self) -> None:
        self.failures = 0

    def report(self, ok: bool, description: str, detail: str) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {description} — {detail}")
        self.failures += not ok

    def opens(self, h5dump: Path, name: str, expected: bool) -> None:
        """Whether the library can open a fixture at all — the format boundary."""
        result = run([str(h5dump), "-n", str(FIXTURES / name)])
        opened = result.returncode == 0
        detail = "lists its objects" if opened else "refused"
        self.report(opened == expected, name, detail)

    def signature(self, name: str, signature: bytes, expected: int) -> None:
        """A structure named by its on-disk signature, counted in the bytes."""
        found = (FIXTURES / name).read_bytes().count(signature)
        label = signature.decode()
        self.report(found == expected, f"{name} {label}", f"{found} block(s)")

    def repack(self, h5repack: Path, name: str) -> None:
        output = FIXTURES / f"repacked-{name}"
        output.unlink(missing_ok=True)
        result = run([str(h5repack), str(FIXTURES / name), str(output)])
        self.report(result.returncode == 0, f"{name} repacked", result.stderr.strip() or "ok")

    def verify(self, script: str, environment: dict, arguments: list[str],
               python: Path | None, reinstall: bool = False) -> None:
        """The h5py-backed content checks, which print their own lines.

        A reinstall on the build pass, because uv keys its cache on the
        lockfile and would otherwise carry a build against another HDF5 across.
        """
        command = ["uv", "run"]
        if reinstall:
            command += ["--reinstall-package", "h5py"]
        if python:
            command += ["--python", str(python)]
        command += [str(REPO / "scripts" / script), str(FIXTURES), *arguments]
        result = subprocess.run(command, env={**os.environ, **environment})
        self.failures += result.returncode != 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, help="an HDF5 installation to check against")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="build h5py against the library and report the pair, then stop",
    )
    arguments = parser.parse_args()

    # The verifier writes straight to the terminal; keep this in step with it.
    sys.stdout.reconfigure(line_buffering=True)

    WORK.mkdir(parents=True, exist_ok=True)
    prefix = arguments.prefix
    if prefix is None:
        prefix = WORK / "install"
        build_hdf5(prefix)

    h5dump = find_tool(prefix, "h5dump")
    h5repack = find_tool(prefix, "h5repack")
    if not h5dump or not h5repack:
        sys.exit(f"no h5dump and h5repack under {prefix}")

    version = run([str(h5dump), "--version"]).stdout.split()[-1]
    numbers = tuple(int(part) for part in version.split("."))
    print(f"==> using HDF5 {version}")

    home = prefix / "Library" if (prefix / "Library" / "include").is_dir() else prefix
    environment = {"HDF5_DIR": str(home)}
    if numbers >= H5PY_FLOOR:
        script = "verify_fixtures.py"
    else:
        # Compiling against 1.8 headers trips warnings GCC 14 turned into
        # errors, in h5py's C as it does in HDF5's own.
        script = "verify_fixtures_hdf5_18.py"
        environment["CFLAGS"] = (
            f"{os.environ.get('CFLAGS', '')} -Wno-error=incompatible-pointer-types "
            "-Wno-error=implicit-function-declaration -Wno-error=int-conversion"
        )
    python = find_python(prefix)

    checks = Checks()

    if arguments.prepare:
        print(f"==> building h5py against HDF5 {version}")
        checks.verify(script, environment, [f"--expect-hdf5={version}", "--link-only"],
                      python, reinstall=True)
        return 1 if checks.failures else 0

    write_fixtures()

    # A version 3 superblock is a 1.10 addition, so 1.8 cannot open the 1.10
    # fixtures at all. That refusal is what `LibVer::V18` exists to avoid.
    reads_v110 = numbers >= (1, 10)
    print(f"==> the 1.10 fixtures must {'open' if reads_v110 else 'refuse'}")
    for name in V110_FIXTURES:
        checks.opens(h5dump, name, reads_v110)

    print("==> the 1.8 format must be readable")
    for name in V18_FIXTURES:
        checks.opens(h5dump, name, True)

    # Nine or more attributes on one object move out of the object header into a
    # fractal heap indexed by a version-2 B-tree. Read from the file itself, so
    # the values below are known to come from dense storage.
    print("==> the dense attribute set must live in a fractal heap")
    checks.signature("plain_v18.h5", b"FRHP", 1)
    checks.signature("plain_v18.h5", b"BTHD", 1)

    skips = [] if reads_v110 else [f"--skip-file={name}" for name in V110_FIXTURES]

    print("==> it must read the values, not merely open the files")
    checks.verify(script, environment, [f"--expect-hdf5={version}", *skips], python)

    # h5repack rewrites every object through the library's own writer, so the
    # same manifest read back says the copy kept what the original held.
    print("==> an h5repack round trip must preserve what the file holds")
    for name in V18_FIXTURES:
        checks.repack(h5repack, name)
    checks.verify(
        script,
        environment,
        [f"--expect-hdf5={version}", "--file-prefix=repacked-",
         *[f"--skip-file={name}" for name in V110_FIXTURES]],
        python,
    )

    print()
    if checks.failures:
        print(f"{checks.failures} check(s) failed against HDF5 {version}")
        return 1
    print(f"all checks passed against HDF5 {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
