# Releases

## Versioning

`0.x.0` may break, `0.x.y` may not. `Cargo.toml` holds the version being developed: a PR that breaks the public API bumps it to the next minor, with `Cargo.lock` and its changelog entry. One bump covers the cycle.

## Changelog

One or two sentences per entry: the capability, the public API name, one caveat clause, then `([#NN](url))`. Prefix breaking changes with `**Breaking:**`. `d5a966b` is the model.

Write the summary paragraph that leads each released section by hand.

`CHANGELOG.md` is the source; `docs/reference/changelog.md` includes it.

## Cutting one

1. `just ci`
2. For a final release, write the summary paragraph into `notes.md`.
3. `just release::prepare 0.45.0 --summary-file notes.md`, or `just release::prepare 0.45.0-rc.1` for a candidate. This reports the cycle's API delta, sets the version, promotes `[Unreleased]` into a dated section for a final, packages with `cargo publish --dry-run`, commits on `release/v0.45.0`, pushes it and opens the pull request.
4. Review and merge the pull request. Every guard runs on it.
5. Run the Release workflow from the Actions tab with the version. It gates on the API delta, then publishes to crates.io, tags, and creates the GitHub release. A final release also rebuilds the documentation site.

Re-running the workflow with the same version resumes after a failure: each public step is skipped once it has happened.

`just release::publish 0.45.0` runs the same steps from a machine with `cargo login` and `gh auth login` done, if Actions is unavailable.

## Release candidates

`X.Y.Z-rc.N`, tagged `vX.Y.Z-rc.N` and published as a pre-release, which no `^` requirement selects. A candidate leaves `CHANGELOG.md` alone: `[Unreleased]` stays open until the final release promotes it, and the candidate's release notes are its body. While a candidate cycle is open, a breaking pull request bumps to the next candidate form, and no other version can be prepared until the cycle ends in its final release.

## The API delta

Step 3's report and step 5's gate are the cycle's only full public-API checks: CI's `SemVer` job derives what to check from the manifest, so a cycle that has already bumped is checked here alone. Read the report against the `[Unreleased]` section being promoted, and treat an empty report on a cycle that claims a breaking change as a finding. The gate stops a release whose delta needs a larger bump than the release makes.

The verdict is the `Summary` line. 0.48.0 exits 1 for both findings and failure-to-run, 0.50.0 splits those into 100 and 101 (#337). `cargo binstall cargo-semver-checks` fixes a tool too old for the toolchain. `--skip-api-delta` on the prepare and the workflow's checkbox release without the check, for the window after a rustc release.

`cargo-semver-checks` matches items by importable path. `just api::api-surface` covers types reachable through the API without one — `Superblock::base_address` went from `u64` to `BaseAddress` in 0.40.0 with no finding — and `tests/public_api_surface.rs` covers a retyped public field.
