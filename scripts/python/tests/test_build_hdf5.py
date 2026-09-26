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


@pytest.mark.parametrize("entry", build_hdf5.series().values(), ids=build_hdf5.series())
def test_every_release_has_a_tag_and_a_digest(entry):
    assert entry["tag"]
    assert len(entry["sha256"]) == 64


@pytest.mark.parametrize(
    ("name", "version", "feature"),
    [
        ("1.8", "1.8.23", ""),
        ("1.8.23", "1.8.23", ""),
        ("1.14", "1.14.6", "__hdf5-1.14"),
        ("2", "2.2.0", "__hdf5-2"),
        ("2.2.0", "2.2.0", "__hdf5-2"),
    ],
)
def test_a_series_or_its_release_names_the_release(name, version, feature):
    entry = build_hdf5.release(name)
    assert (entry["version"], entry["feature"]) == (version, feature)


def test_an_unlisted_release_is_refused():
    with pytest.raises(SystemExit, match="no HDF5 release '1.14.5'"):
        build_hdf5.release("1.14.5")


def test_the_series_run_from_the_oldest_release_to_the_newest():
    versions = [entry["version"] for entry in build_hdf5.series().values()]
    assert versions == sorted(versions, key=lambda version: tuple(map(int, version.split("."))))
