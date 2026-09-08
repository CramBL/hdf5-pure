from datetime import date

import pytest

from hdf5_pure_scripts.release import (
    Version,
    bump_allows,
    changelog_section,
    promote_changelog,
    required_bump,
    set_version,
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


CHANGELOG = """# Changelog

## [Unreleased]

### Fixed

- A thing.

## [0.44.0] - 2026-09-04

Old summary.

[Unreleased]: https://example/compare/v0.44.0...HEAD
[0.44.0]: https://example/compare/v0.43.1...v0.44.0
"""


def test_changelog_section_is_the_body_up_to_the_next_header():
    assert changelog_section(CHANGELOG, "Unreleased") == "\n### Fixed\n\n- A thing.\n\n"
    assert changelog_section(CHANGELOG, "0.44.0") == "\nOld summary.\n\n"
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

Old summary.

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


def test_semver_verdict_gates_the_release_type():
    assert (
        required_bump(
            "Summary semver requires new major version: 2 major and 0 minor checks failed"
        )
        == "major"
    )
    assert (
        required_bump("Summary semver requires new minor version: 1 minor check failed") == "minor"
    )
    assert required_bump("Summary no semver update required") is None
    assert (
        bump_allows("major", "major")
        and bump_allows("minor", "minor")
        and bump_allows("patch", None)
    )
    assert bump_allows("major", "minor")
    assert not bump_allows("minor", "major")
    assert not bump_allows("patch", "minor")
