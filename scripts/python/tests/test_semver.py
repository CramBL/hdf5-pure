from unittest.mock import patch

import pytest

from hdf5_pure_scripts import semver
from hdf5_pure_scripts.api_profiles import profile
from hdf5_pure_scripts.semver import comparison_baseline


def test_release_replay_compares_with_the_last_tag_after_the_candidate_was_published():
    assert comparison_baseline("hdf5-pure", "0.48.0", "0.47.0", "0.48.0") == "0.47.0"
    with pytest.raises(SystemExit, match="differs from release tag"):
        comparison_baseline("hdf5-pure", "0.49.0", "0.47.0", "0.48.0")


def test_core_is_checked_against_its_own_release():
    selected = profile("hdf5-pure-core", "std")
    with (
        patch.object(semver, "released_version", return_value="0.48.0") as released,
        patch.object(semver, "check_package") as check,
    ):
        semver.check_api("hdf5-pure-core", "std", "minor")
    released.assert_called_once_with("hdf5-pure-core")
    check.assert_called_once_with(
        "hdf5-pure-core", "0.48.0", "minor", semver.feature_args(selected)
    )
