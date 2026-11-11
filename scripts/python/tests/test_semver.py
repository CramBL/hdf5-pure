from argparse import Namespace
from unittest.mock import call, patch

import pytest

from hdf5_pure_scripts import release, semver
from hdf5_pure_scripts.semver import comparison_baseline
from repository import commit_files, git


def test_release_replay_compares_with_the_last_tag_after_the_candidate_was_published():
    assert comparison_baseline("hdf5-pure", "0.48.0", "0.47.0", "0.48.0") == "0.47.0"
    with pytest.raises(SystemExit, match="differs from release tag"):
        comparison_baseline("hdf5-pure", "0.49.0", "0.47.0", "0.48.0")


def test_each_package_baseline_is_resolved_once_before_profiles_run():
    selected = semver.checks()
    published = {"hdf5-pure": "0.48.0", "hdf5-pure-core": "0.48.1"}

    def check_profile(*args):
        published.update(dict.fromkeys(published, "0.49.0"))

    with (
        patch.object(semver, "released_version", side_effect=published.__getitem__) as released,
        patch.object(semver, "check_api", side_effect=check_profile) as check,
    ):
        semver.check_all("minor")
    assert released.call_args_list == [call("hdf5-pure"), call("hdf5-pure-core")]
    assert check.call_args_list == [
        call(name, profile, "minor", "0.48.0" if name == "hdf5-pure" else "0.48.1")
        for name, profile in selected
    ]


@pytest.mark.parametrize("publication", ["before", "during", "after"])
def test_release_checks_keep_previous_stable_across_publication(tmp_path, monkeypatch, publication):
    selected = semver.checks()
    git(tmp_path, ["init", "--quiet"])
    commit_files(
        tmp_path,
        {
            "Cargo.toml": '[workspace.package]\nversion = "0.48.0"\n',
            "CHANGELOG.md": "## [Unreleased]\n\n**Breaking:** API changes.\n",
        },
        "Previous stable release",
    )
    git(tmp_path, ["tag", "v0.48.0"])
    commit_files(
        tmp_path,
        {
            "Cargo.toml": '[workspace.package]\nversion = "0.49.0"\n',
            "CHANGELOG.md": "## [Unreleased]\n\n## [0.49.0] - 2026-11-11\n\n"
            "**Breaking:** API changes.\n",
        },
        "Release v0.49.0",
    )
    if publication == "after":
        git(tmp_path, ["tag", "v0.49.0"])
    monkeypatch.chdir(tmp_path)
    published = "0.49.0" if publication == "after" else "0.48.0"

    def check_profile(*args):
        nonlocal published
        if publication == "during":
            published = "0.49.0"

    with (
        patch.object(semver, "checks", return_value=selected),
        patch.object(semver, "released_version", side_effect=lambda _: published) as released,
        patch.object(semver, "check_api", side_effect=check_profile) as check,
    ):
        release.check_api(Namespace(base="HEAD~1", package=None, profile=None))
    assert released.call_args_list == [call("hdf5-pure"), call("hdf5-pure-core")]
    assert check.call_args_list == [
        call(name, profile, "major", "0.48.0") for name, profile in selected
    ]
