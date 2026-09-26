from unittest.mock import patch

import pytest

from hdf5_pure_scripts import semver
from hdf5_pure_scripts.api_profiles import cargo_feature_args, feature_args, profile
from hdf5_pure_scripts.semver import checks, comparison_baseline

FULL_PUBLIC = "fast-deflate,ndarray,provenance,serde,zfp,num-complex,checksum"


def test_release_replay_compares_with_the_last_tag_after_the_candidate_was_published():
    assert comparison_baseline("hdf5-pure", "0.48.0", "0.47.0", "0.48.0") == "0.47.0"
    with pytest.raises(SystemExit, match="differs from release tag"):
        comparison_baseline("hdf5-pure", "0.49.0", "0.47.0", "0.48.0")


@pytest.mark.parametrize(
    ("package", "name", "semver_checks", "cargo"),
    [
        ("hdf5-pure", "default", ["--default-features"], []),
        (
            "hdf5-pure",
            "no-std",
            ["--only-explicit-features", "--features", "checksum"],
            ["--no-default-features", "--features", "checksum"],
        ),
        (
            "hdf5-pure",
            "full-public",
            ["--default-features", "--features", FULL_PUBLIC],
            ["--features", FULL_PUBLIC],
        ),
        ("hdf5-pure-core", "no-std", ["--only-explicit-features"], ["--no-default-features"]),
        (
            "hdf5-pure-core",
            "std",
            ["--only-explicit-features", "--features", "std"],
            ["--no-default-features", "--features", "std"],
        ),
    ],
)
def test_profiles_select_their_feature_sets(package, name, semver_checks, cargo):
    selected = profile(package, name)
    assert feature_args(selected) == semver_checks
    assert cargo_feature_args(selected) == cargo


@pytest.mark.parametrize(
    ("package", "name", "selected"),
    [
        (
            None,
            None,
            [
                ("hdf5-pure", "default"),
                ("hdf5-pure", "no-std"),
                ("hdf5-pure", "full-public"),
                ("hdf5-pure-core", "no-std"),
                ("hdf5-pure-core", "std"),
            ],
        ),
        ("hdf5-pure-core", None, [("hdf5-pure-core", "no-std"), ("hdf5-pure-core", "std")]),
        ("hdf5-pure", "no-std", [("hdf5-pure", "no-std")]),
        ("hdf5-pure-core", "full-public", []),
    ],
)
def test_checks_select_the_crates_and_profiles(package, name, selected):
    assert checks(package, name) == selected


@pytest.mark.parametrize(
    ("migration", "checked"),
    [
        (
            False,
            [
                (
                    "hdf5-pure-core",
                    "0.48.0",
                    "minor",
                    ["--only-explicit-features", "--features", "std"],
                )
            ],
        ),
        (True, []),
    ],
)
def test_core_is_checked_against_its_own_release_once_the_migration_ends(migration, checked):
    selected = profile("hdf5-pure-core", "std")
    manifest = {"package": {"metadata": {}}, "workspace": {"package": {"version": "0.48.0"}}}
    with (
        patch.object(semver, "profile", return_value=selected),
        patch.object(semver, "released_version", return_value="0.48.0"),
        patch.object(semver, "migration_baseline", return_value=migration),
        patch.object(semver.tomllib, "loads", return_value=manifest),
        patch.object(semver, "check_package") as check,
    ):
        semver.check_api("hdf5-pure-core", "std", "minor")
    assert [call.args for call in check.call_args_list] == checked
