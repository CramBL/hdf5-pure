# Releases

## Versioning

`0.x.0` may break, `0.x.y` may not. `Cargo.toml` on `main` reads the last release, no pull request modifies it. A pull request that breaks the public API marks its changelog entry `**Breaking:**`, and the release decides the bump from that: a cycle with a marked entry releases as the next minor, and any other as a patch. `prepare` rejects a patch version over a marked cycle. CI verifies that the accumulated public API delta is compatible with the release type the current cycle requires. Before the first breaking entry in a cycle, an unmarked API break therefore fails the semver gate. Once the cycle is marked breaking, a later breaking change still marks its entry `**Breaking:**` for accurate release notes, but review enforces that, not the semver gate.

## Changelog

One or two sentences per entry: the capability, the public API name, one caveat clause, then the pull request link, `([#NN](url))`. Prefix breaking changes with `**Breaking:**` and list them first in their section.

A released section may open with a summary paragraph, written by hand.

`CHANGELOG.md` is the source.

## Cutting one

1. `just ci-local`, the broad local validation gate. GitHub Actions additionally runs CI-specific matrix and cross-target jobs.
2. `just release::prepare 0.45.0`, or `just release::prepare 0.45.0-rc.1` for a candidate. `--summary-file notes.md` opens the section with a summary paragraph.
3. `just release::pr 0.45.0`, and merge the pull request it opens. The Release workflow runs on the merge.

Re-running the workflow's job resumes after a failure.

`just release::publish` runs the same steps from a checkout of the merged commit on a machine with `cargo login` and `gh auth login` done, if Actions is unavailable.

## Release candidates

`X.Y.Z-rc.N`, published as a pre-release. `[Unreleased]` stays open until the final release promotes it. While a candidate cycle is open, no other version can be prepared.

## The API delta

`prepare` prints the public-API delta since the last release. Read it against the `[Unreleased]` section. The Release workflow stops a release whose delta needs a larger bump than the release makes, which catches a break in a cycle that no entry marked. `just api::release-type` prints the release type the `[Unreleased]` section calls for, or, on a release pull request, the one the dated section it promotes calls for. `just api::semver-release-type` prints that type in the word `cargo semver-checks --release-type` takes, which is what CI checks a pull request against.

`--skip-api-delta` skips the report on `prepare`, and on `publish` run by hand it releases without the check.

After publishing the first release containing `hdf5-pure-core`, the release maintainer opens a cleanup pull request that removes the temporary 0.47 ownership migration bridge:

- `migration.py`, `rustdoc_json.py`, `nightly.py` and their tests, `test_migration.py` and `test_nightly.py`
- the migration branches in `semver.py` and their cases in `test_semver.py`
- the `core_features` keys in `scripts/api-profiles.toml` and their handling in `api_profiles.py`
- `[package.metadata.hdf5-pure-api-migration]` and the `[package.metadata.cargo-semver-checks.lints]` downgrades in `Cargo.toml`
- the rustdoc JSON nightly step in `api.yml` and `release.yml`, if nothing else there needs it

Check the list against the code at that point. CI rejects the bridge once the published `hdf5-pure` baseline advances.
