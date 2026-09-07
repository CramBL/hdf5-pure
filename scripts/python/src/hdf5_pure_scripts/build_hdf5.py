"""Build a release of the HDF5 C library and its tools under `tmp/hdf5`.

    just interop::hdf5-build <version>    build, unless already built
    just interop::hdf5-env <version>      print the exports that link it

The result is an install prefix with `include/`, `lib/` and `bin/`, the layout
`hdf5-metno` reads from `HDF5_DIR`. `--env` prints `export` lines for
`eval "$(...)"`.

Windows needs zlib from vcpkg (`vcpkg install zlib:x64-windows`) and
`VCPKG_INSTALLATION_ROOT` set, as on the GitHub runners.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from hdf5_pure_scripts import repo_root

WORK = repo_root() / "tmp" / "hdf5"

# The last patch release of each series: its git tag, and the sha256 of the
# archive GitHub serves for the tag.
RELEASES = {
    "1.8.23": ("hdf5-1_8_23", "e5f575403ae05c080950dba8d2cb3d06e7ccbe566b69b20cba08bec41fd4fa5f"),
    "1.10.11": ("hdf5-1_10_11", "4ef6375fc7d8c54dcd66e9bc35a7a3580d33cd8878bdf21ad1eb388a43863159"),
    "1.12.3": ("hdf5-1_12_3", "39e2e3e25f2263f2dc3370556c174952675d7ac51c4d6f7d4f5e8b3a8bd10ac6"),
    "1.14.6": ("hdf5_1.14.6", "09ee1c671a87401a5201c06106650f62badeea5a3b3941e9b1e2e1e08317357f"),
    "2.2.0": ("2.2.0", "5b8d75125ae7b4fef55d3b39e8d3dbfd238cdf63b6518afd67fa69ac80f06542"),
}
ARCHIVE_URL = "https://github.com/HDFGroup/hdf5/archive/refs/tags/{tag}.tar.gz"

# GCC 14 and recent clang make these errors. The older releases trip all three.
RELAXED_C_FLAGS = (
    "-Wno-error=incompatible-pointer-types "
    "-Wno-error=implicit-function-declaration "
    "-Wno-error=int-conversion"
)

WINDOWS = platform.system() == "Windows"


def prefix_of(version: str) -> Path:
    return WORK / version


def tool(prefix: Path, name: str) -> Path | None:
    for candidate in (prefix / "bin" / name, prefix / "bin" / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    return None


def download(tag: str, sha256: str) -> Path:
    archive = WORK / "src" / f"{tag}.tar.gz"
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"==> downloading {tag}")
        urllib.request.urlretrieve(ARCHIVE_URL.format(tag=tag), archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != sha256:
        archive.unlink()
        sys.exit(f"{archive.name}: sha256 {digest}, expected {sha256}")
    return archive


def extract(archive: Path) -> Path:
    with tarfile.open(archive) as tar:
        top = tar.getnames()[0].split("/")[0]
        source = WORK / "src" / top
        if not source.is_dir():
            print(f"==> extracting {archive.name}")
            tar.extractall(WORK / "src", filter="data")
    return source


def run(step: str, command: list[str], cwd: Path) -> None:
    log = WORK / f"{step}.log"
    print(f"==> {step} (log: {log})")
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        text = log.read_text(errors="replace")
        errors = [
            line
            for line in text.splitlines()
            if "error" in line.lower() and "warning" not in line.lower()
        ]
        sys.exit(f"{step} failed, see {log}:\n" + "\n".join(errors[:40]) + "\n...\n" + text[-1500:])


def build(version: str) -> Path:
    prefix = prefix_of(version)
    if tool(prefix, "h5dump"):
        return prefix
    tag, sha256 = RELEASES[version]
    source = extract(download(tag, sha256))
    build_dir = WORK / "src" / f"build-{version}"
    shutil.rmtree(build_dir, ignore_errors=True)

    # Both spellings of the zlib option, 2.0 renamed it. Static libraries stay
    # on: 1.10 and 1.12 name the tools `h5dump-shared` without them. No
    # high-level library, nothing uses it.
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build_dir),
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DCMAKE_INSTALL_RPATH={prefix}/lib",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTING=OFF",
        "-DHDF5_BUILD_TOOLS=ON",
        "-DHDF5_BUILD_EXAMPLES=OFF",
        "-DHDF5_BUILD_HL_LIB=OFF",
        "-DHDF5_BUILD_UTILS=OFF",
        "-DHDF5_ENABLE_Z_LIB_SUPPORT=ON",
        "-DHDF5_ENABLE_ZLIB_SUPPORT=ON",
        "-DHDF5_ENABLE_SZIP_SUPPORT=OFF",
    ]
    if WINDOWS:
        # Named directly rather than through the vcpkg toolchain file: the
        # releases before 1.14 ask for a static zlib component that the
        # dynamic vcpkg package does not have.
        installed = vcpkg_installed()
        configure += [
            f"-DZLIB_INCLUDE_DIR={installed / 'include'}",
            f"-DZLIB_LIBRARY={vcpkg_zlib_import_library(installed)}",
        ]
    else:
        configure.append(f"-DCMAKE_C_FLAGS={os.environ.get('CFLAGS', '')} {RELAXED_C_FLAGS}")

    jobs = str(os.cpu_count() or 4)
    run("configure", configure, source)
    run(
        "build",
        ["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", jobs],
        source,
    )
    run("install", ["cmake", "--install", str(build_dir), "--config", "Release"], source)
    if not tool(prefix, "h5dump"):
        sys.exit(f"the install under {prefix} has no h5dump")
    return prefix


def vcpkg_installed() -> Path:
    vcpkg = os.environ.get("VCPKG_INSTALLATION_ROOT")
    if not vcpkg:
        sys.exit("set VCPKG_INSTALLATION_ROOT to a vcpkg checkout with zlib installed")
    return Path(vcpkg) / "installed" / "x64-windows"


def vcpkg_zlib_import_library(installed: Path) -> Path:
    """`z.lib` from zlib 1.3.2 on, `zlib.lib` before."""
    for name in ("z.lib", "zlib.lib"):
        candidate = installed / "lib" / name
        if candidate.is_file():
            return candidate
    found = sorted(p.name for p in (installed / "lib").glob("*.lib"))
    sys.exit(f"no zlib import library under {installed / 'lib'}: {found}")


def installed_version(prefix: Path) -> str:
    """The release under `prefix`, from its header rather than its tools."""
    for line in (prefix / "include" / "H5pubconf.h").read_text().splitlines():
        if line.startswith("#define H5_VERSION "):
            return line.split('"')[1]
    sys.exit(f"no H5_VERSION in {prefix}/include/H5pubconf.h")


def posix(path: Path) -> str:
    """A path for bash on Windows: `D:\\a\\x` as `/d/a/x`."""
    text = str(path).replace("\\", "/")
    if WINDOWS and len(text) > 1 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


def exports(prefix: Path) -> list[str]:
    """Run-time lookup of the shared library.

    Linux: the loader path. macOS: an rpath at link time, because the system
    strips `DYLD_*` from the environment of every system binary, `sh` under
    `just` included. RUSTFLAGS replaces `.cargo/config.toml`, so the x86_64
    frame pointers are repeated. Windows: `PATH`.
    """
    lines = [f'export HDF5_DIR="{str(prefix).replace(chr(92), "/")}"']
    system = platform.system()
    if system == "Linux":
        lines.append(
            f'export LD_LIBRARY_PATH="{prefix}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"'
        )
    elif system == "Darwin":
        lines.append(
            f'export RUSTFLAGS="-C force-frame-pointers=yes -C link-args=-Wl,-rpath,{prefix}/lib"'
        )
    binaries = posix(prefix / "bin")
    if WINDOWS:
        binaries += ":" + posix(vcpkg_installed() / "bin")
    lines.append(f'export PATH="{binaries}:$PATH"')
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=sorted(RELEASES))
    parser.add_argument("--env", action="store_true", help="print shell exports for the build")
    arguments = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    if arguments.env:
        prefix = prefix_of(arguments.version)
        if not tool(prefix, "h5dump"):
            sys.exit(
                f"no build under {prefix}: run `just interop::hdf5-build {arguments.version}` first"
            )
        print("\n".join(exports(prefix)))
        return

    sys.stdout.reconfigure(line_buffering=True)
    prefix = build(arguments.version)
    print(f"==> HDF5 {installed_version(prefix)} under {prefix}")
