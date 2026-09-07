# Releases

## Versioning

`0.x.0` may break, `0.x.y` may not. `Cargo.toml` holds the version being developed: a PR that breaks the public API bumps it to the next minor, with `Cargo.lock` and its changelog entry. One bump covers the cycle.

## Changelog

One or two sentences per entry: the capability, the public API name, one caveat clause, then `([#NN](url))`. Prefix breaking changes with `**Breaking:**`. `d5a966b` is the model.

Write the summary paragraph that leads each released section by hand.

`CHANGELOG.md` is the source; `docs/reference/changelog.md` includes it.

## Cutting one

1. `just ci`
2. Write the summary paragraph into `notes.md`.
3. `scripts/release.sh 0.25.0 --summary-file notes.md` — reports the cycle's API delta, bumps the version, promotes `[Unreleased]` into a dated section, refreshes the compare links, and packages with `cargo publish --dry-run`.
4. Read the diff it leaves in your tree.
5. `scripts/release.sh 0.25.0 --summary-file notes.md --gh-release --publish` to commit, tag, push, release and publish.

Run `just release::check-release-script` after changing the script.

## The API delta

Step 3's report is the cycle's only full public-API check: CI's `SemVer` job derives what to check from the manifest, so a cycle that has already bumped is checked here alone. Read it against the `[Unreleased]` section you are promoting, and treat an empty report on a cycle that claims a breaking change as a finding.

The verdict is the `Summary` line — 0.48.0 exits 1 for both findings and failure-to-run, 0.50.0 splits those into 100 and 101 (#337). `cargo install cargo-semver-checks --locked` fixes a tool too old for the toolchain; `--skip-api-delta` releases without the check, for the window after a rustc release.

`cargo-semver-checks` matches items by importable path. `just api::api-surface` covers types reachable through the API without one — `Superblock::base_address` went from `u64` to `BaseAddress` in 0.40.0 with no finding — and `tests/public_api_surface.rs` covers a retyped public field.
