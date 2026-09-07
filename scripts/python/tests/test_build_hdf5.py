from pathlib import Path

import pytest

from hdf5_pure_scripts import build_hdf5


def test_exports_link_the_prefix_per_platform(monkeypatch):
    prefix = Path("/opt/hdf5/1.8.23")
    monkeypatch.setattr(build_hdf5, "WINDOWS", False)

    monkeypatch.setattr(build_hdf5.platform, "system", lambda: "Linux")
    assert build_hdf5.exports(prefix) == [
        'export HDF5_DIR="/opt/hdf5/1.8.23"',
        'export LD_LIBRARY_PATH="/opt/hdf5/1.8.23/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        'export PATH="/opt/hdf5/1.8.23/bin:$PATH"',
    ]

    monkeypatch.setattr(build_hdf5.platform, "system", lambda: "Darwin")
    lines = build_hdf5.exports(prefix)
    assert lines[1] == (
        'export RUSTFLAGS="-C force-frame-pointers=yes'
        ' -C link-args=-Wl,-rpath,/opt/hdf5/1.8.23/lib"'
    )


def test_windows_paths_are_spelled_for_bash(monkeypatch):
    monkeypatch.setattr(build_hdf5, "WINDOWS", True)
    assert build_hdf5.posix(Path("D:\\a\\hdf5\\bin")) == "/d/a/hdf5/bin"


def test_installed_version_comes_from_the_header(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    (include / "H5pubconf.h").write_text('#define H5_VERSION "1.14.6"\n')
    assert build_hdf5.installed_version(tmp_path) == "1.14.6"


@pytest.mark.parametrize("version", ["1.8.23", "1.10.11", "1.12.3", "1.14.6", "2.2.0"])
def test_every_release_has_a_tag_and_a_digest(version):
    tag, sha256 = build_hdf5.RELEASES[version]
    assert tag
    assert len(sha256) == 64
