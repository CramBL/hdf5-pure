from unittest.mock import patch

import pytest

from hdf5_pure_scripts import semver
from hdf5_pure_scripts.api_profiles import profile
from hdf5_pure_scripts.semver import comparison_baseline


def test_release_replay_compares_with_the_last_tag_after_the_candidate_was_published():
    assert comparison_baseline("hdf5-pure", "0.48.0", "0.47.0", "0.48.0") == "0.47.0"
    with pytest.raises(SystemExit, match="differs from release tag"):
        comparison_baseline("hdf5-pure", "0.49.0", "0.47.0", "0.48.0")


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
