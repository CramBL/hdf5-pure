import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hdf5_pure_scripts import release
from hdf5_pure_scripts.release import (
    Version,
    bump_allows,
    changelog_section,
    cycle_release_type,
    is_release_subject,
    previous_release_baseline,
    promote_changelog,
    pull_request_release_type,
    semver_checks_release_type,
    set_version,
    set_workspace_version,
)


def v(text):
    return Version.parse(text)


def test_candidates_come_before_their_final_and_order_by_number():
    assert v("0.45.0-rc.1") < v("0.45.0-rc.2") < v("0.45.0") < v("0.45.1-rc.1")
    assert not v("0.45.0") < v("0.45.0-rc.9")
    assert v("0.44.0") < v("0.45.0-rc.1")
    assert v("0.4.5") < v("0.45.0")


@pytest.mark.parametrize("text", ["0.45", "v0.45.0", "0.45.0-rc1", "0.45.0-beta.1", "0.45.0-rc.1x"])
def test_malformed_versions_are_rejected(text):
    with pytest.raises(ValueError):
        Version.parse(text)


def test_final_and_rc_round_trip():
    assert str(v("0.45.0-rc.3")) == "0.45.0-rc.3"
    assert v("0.45.0-rc.3").final == v("0.45.0")
    assert v("0.45.0-rc.3").is_rc and not v("0.45.0").is_rc


def test_release_type_is_the_first_component_that_changed():
    assert v("1.0.0").release_type(v("0.45.0")) == "major"
    assert v("0.45.0").release_type(v("0.44.3")) == "minor"
    assert v("0.44.4").release_type(v("0.44.3")) == "patch"


def test_set_version_touches_only_the_crate_after_its_name_line():
    manifest = (
        '[dependencies]\nversion = "0.1.0"\n\n'
        '[package]\nname = "hdf5-pure"\nversion = "0.44.0"\n\n'
        '[[package]]\nname = "other"\nversion = "0.44.0"\n'
    )
    updated = set_version(manifest, v("0.45.0-rc.1"))
    assert updated.count('version = "0.45.0-rc.1"') == 1
    assert updated.count('version = "0.44.0"') == 1
    assert updated.startswith('[dependencies]\nversion = "0.1.0"')
    with pytest.raises(ValueError):
        set_version('name = "other"\nversion = "1"\n', v("0.45.0"))


def test_workspace_version_updates_the_package_version_and_the_exact_pins():
    manifest = (
        '[package]\nname = "hdf5-pure"\nversion.workspace = true\n'
        '[workspace.package]\nversion = "0.47.0"\nedition = "2024"\n'
        "[workspace.dependencies]\n"
        'hdf5-pure = { path = "." }\n'
        'hdf5-pure-core = { path = "crates/hdf5-pure-core", version = "=0.47.0" }\n'
        'rstest = "0.27"\n'
        "[dependencies]\n"
        'serde = { version = "=1.0.0" }\n'
    )
    assert set_workspace_version(manifest, v("0.48.0"), ["hdf5-pure-core", "hdf5-pure"]) == (
        manifest.replace('version = "0.47.0"', 'version = "0.48.0"').replace(
            'version = "=0.47.0"', 'version = "=0.48.0"'
        )
    )
    with pytest.raises(ValueError, match="workspace.package"):
        set_workspace_version('[package]\nversion = "0.47.0"\n', v("0.48.0"), [])


@pytest.mark.parametrize("local_tag", ["", "v0.48.0"])
def test_publish_resumes_tagging_after_every_package_was_published(local_tag):
    args = SimpleNamespace(version=v("0.48.0"), skip_api_delta=False, dry_run=False)

    def git_output(*command):
        if command == ("git", "rev-parse", "HEAD"):
            return "release-sha"
        if command == ("git", "log", "-1", "--format=%s"):
            return "Release v0.48.0"
        if command == ("git", "tag", "--list", "v0.48.0"):
            return local_tag
        if command == ("git", "rev-parse", "refs/tags/v0.48.0^{}"):
            return "release-sha"
        if command[:3] == ("git", "ls-remote", "--tags"):
            return ""
        raise AssertionError(command)

    with (
        patch.object(release, "manifest", return_value={"version": "0.48.0"}),
        patch.object(release, "worktree_clean", return_value=True),
        patch.object(release, "output", side_effect=git_output),
        patch.object(release, "run") as run,
        patch.object(release.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
        patch.object(release, "CHANGELOG") as changelog,
        patch.object(
            release,
            "tagged_versions",
            return_value=[v("0.47.0"), v("0.48.0")] if local_tag else [v("0.47.0")],
        ),
        patch.object(release, "semver_verdict", return_value="API compatibility passed") as semver,
        patch.object(
            release,
            "published_packages",
            return_value=[
                {"name": "hdf5-pure-core", "version": "0.48.0"},
                {"name": "hdf5-pure", "version": "0.48.0"},
            ],
        ),
        patch.object(release.urllib.request, "urlopen"),
    ):
        changelog.read_text.return_value = "## [0.48.0] - 2026-09-25\n"
        release.publish(args)

    semver.assert_called_once_with(v("0.47.0"), "major")
    assert (
        ("git", "tag", "-a", "v0.48.0", "-m", "Release v0.48.0")
        in [call.args for call in run.call_args_list]
    ) == (local_tag == "")
    assert ("git", "push", "-q", "origin", "v0.48.0") in [call.args for call in run.call_args_list]
    assert not any(
        call.args[:2] == ("cargo", "publish") and "--dry-run" not in call.args
        for call in run.call_args_list
    )


@pytest.mark.parametrize("subject", ["Release v0.48.0", "Release v0.48.0 (#636)"])
def test_the_release_subject_survives_a_rebase_or_a_squash_merge(subject):
    assert is_release_subject(subject, "v0.48.0")


@pytest.mark.parametrize(
    "subject",
    [
        "Release v0.48.1",
        "Release v0.48.0 fixups",
        "Release v0.48.0 (#636) extra",
        "Release v0.48.0 (#)",
        "Merge pull request #636 from release/v0.48.0",
        "Release v0.48.0-rc.1",
    ],
)
def test_any_other_subject_is_not_the_release_commit(subject):
    assert not is_release_subject(subject, "v0.48.0")


def test_published_packages_come_after_their_dependencies():
    def package(name, *dependencies, publish=None):
        return {
            "name": name,
            "publish": publish,
            "dependencies": [
                {"name": dependency, "kind": kind} for dependency, kind in dependencies
            ],
        }

    metadata = {
        "packages": [
            package("hdf5-pure", ("hdf5-pure-format", None), ("test-util", "dev")),
            package("hdf5-pure-format", ("hdf5-pure-core", None), ("test-util", "dev")),
            package("hdf5-pure-core"),
            package("test-util", ("hdf5-pure", None), publish=[]),
        ]
    }
    with patch.object(release, "output", return_value=json.dumps(metadata)):
        names = [package["name"] for package in release.published_packages()]
    assert names == ["hdf5-pure-core", "hdf5-pure-format", "hdf5-pure"]


def test_release_baseline_precedes_the_candidate_and_refuses_later_tags():
    assert previous_release_baseline([v("0.47.0"), v("0.48.0")], v("0.48.0")) == v("0.47.0")
    with pytest.raises(ValueError, match="after v0.48.0"):
        previous_release_baseline([v("0.47.0"), v("0.49.0")], v("0.48.0"))


CHANGELOG = """# Changelog

## [Unreleased]

### Fixed

- A thing.

## [0.44.0] - 2026-09-04

- **Breaking:** `open` takes a path.

[Unreleased]: https://example/compare/v0.44.0...HEAD
[0.44.0]: https://example/compare/v0.43.1...v0.44.0
"""


def test_changelog_section_is_the_body_up_to_the_next_header():
    assert changelog_section(CHANGELOG, "Unreleased") == "\n### Fixed\n\n- A thing.\n\n"
    assert changelog_section(CHANGELOG, "0.44.0") == "\n- **Breaking:** `open` takes a path.\n\n"
    assert changelog_section(CHANGELOG, "0.45.0") is None


def test_promotion_dates_the_section_opens_it_with_the_summary_and_moves_the_links():
    promoted = promote_changelog(
        CHANGELOG, v("0.45.0"), v("0.44.0"), "New summary.\n", "https://example", date(2026, 9, 7)
    )
    assert (
        promoted
        == """# Changelog

## [Unreleased]

## [0.45.0] - 2026-09-07

New summary.

### Fixed

- A thing.

## [0.44.0] - 2026-09-04

- **Breaking:** `open` takes a path.

[Unreleased]: https://example/compare/v0.45.0...HEAD
[0.45.0]: https://example/compare/v0.44.0...v0.45.0
[0.44.0]: https://example/compare/v0.43.1...v0.44.0
"""
    )
    assert changelog_section(promoted, "Unreleased") == "\n"


def test_promotion_without_a_summary_opens_with_the_entries():
    promoted = promote_changelog(
        CHANGELOG, v("0.45.0"), v("0.44.0"), None, "https://example", date(2026, 9, 7)
    )
    assert "## [0.45.0] - 2026-09-07\n\n### Fixed\n" in promoted


def test_promotion_needs_the_section_and_the_link_line():
    with pytest.raises(ValueError):
        promote_changelog("# Changelog\n", v("0.45.0"), v("0.44.0"), "s", "u", date(2026, 9, 7))
    with pytest.raises(ValueError):
        promote_changelog(
            "## [Unreleased]\n\n- x\n", v("0.45.0"), v("0.44.0"), "s", "u", date(2026, 9, 7)
        )


def test_release_type_order():
    assert (
        bump_allows("major", "major")
        and bump_allows("minor", "minor")
        and bump_allows("patch", None)
    )
    assert bump_allows("major", "minor")
    assert not bump_allows("minor", "major")
    assert not bump_allows("patch", "minor")


@pytest.mark.parametrize(
    ("release_type", "expected"),
    [("patch", "minor"), ("minor", "major"), ("major", "major")],
)
def test_a_pre_1_0_baseline_translates_the_release_type_to_the_word_semver_checks_takes(
    release_type, expected
):
    assert semver_checks_release_type(release_type, v("0.46.0")) == expected


@pytest.mark.parametrize("release_type", ["patch", "minor", "major"])
def test_from_1_0_on_the_release_type_is_the_word_semver_checks_takes(release_type):
    assert semver_checks_release_type(release_type, v("1.2.3")) == release_type


def test_the_cycle_is_a_minor_only_with_a_breaking_entry_under_unreleased():
    plain = (
        "## [Unreleased]\n\n- Read a thing.\n\n## [0.44.0] - 2026-09-04\n\n- **Breaking:** old.\n"
    )
    assert cycle_release_type(plain) == "patch"
    marked = plain.replace("- Read a thing.", "- **Breaking:** `read` returns a handle.")
    assert cycle_release_type(marked) == "minor"
    assert cycle_release_type("## [0.44.0]\n\n- **Breaking:** old.\n") == "patch"


MID_CYCLE_DIFF = """diff --git a/CHANGELOG.md b/CHANGELOG.md
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -5,3 +5,5 @@
 ### Fixed

+- A thing.
+
 ## [0.44.0] - 2026-09-04
"""

RELEASE_DIFF = """diff --git a/CHANGELOG.md b/CHANGELOG.md
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -2,6 +2,8 @@
 ## [Unreleased]

+## [0.45.0] - 2026-09-07
+
 ### Fixed

@@ -12,3 +14,4 @@
-[Unreleased]: https://example/compare/v0.44.0...HEAD
+[Unreleased]: https://example/compare/v0.45.0...HEAD
+[0.45.0]: https://example/compare/v0.44.0...v0.45.0
"""

BREAKING_CHANGELOG = CHANGELOG.replace("- A thing.", "- **Breaking:** `read` returns a handle.")


def released(changelog):
    return promote_changelog(
        changelog, v("0.45.0"), v("0.44.0"), None, "https://example", date(2026, 9, 7)
    )


def test_a_mid_cycle_pull_request_with_a_breaking_entry_is_a_minor():
    assert pull_request_release_type(BREAKING_CHANGELOG, MID_CYCLE_DIFF) == "minor"


def test_a_mid_cycle_pull_request_without_a_breaking_entry_is_a_patch():
    assert pull_request_release_type(CHANGELOG, MID_CYCLE_DIFF) == "patch"


def test_a_release_pull_request_reads_the_section_it_promotes():
    assert pull_request_release_type(released(BREAKING_CHANGELOG), RELEASE_DIFF) == "minor"


def test_a_release_pull_request_promoting_an_unmarked_section_is_a_patch():
    assert pull_request_release_type(released(CHANGELOG), RELEASE_DIFF) == "patch"


def test_a_pull_request_that_leaves_the_changelog_alone_reads_unreleased():
    assert pull_request_release_type(BREAKING_CHANGELOG, "") == "minor"
    assert pull_request_release_type(CHANGELOG, "") == "patch"
