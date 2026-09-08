# Releases

## Versioning

`0.x.0` may break, `0.x.y` may not. `Cargo.toml` holds the version being developed: a PR that breaks the public API bumps it to the next minor, with `Cargo.lock` and its changelog entry. One bump covers the cycle.

## Changelog

One or two sentences per entry: the capability, the public API name, one caveat clause, then `([#NN](url))`. Prefix breaking changes with `**Breaking:**`. `d5a966b` is the model.

A released section may open with a summary paragraph, written by hand.

`CHANGELOG.md` is the source; `docs/reference/changelog.md` includes it.

## Cutting one

1. `just ci`
2. `just release::prepare 0.45.0`, or `just release::prepare 0.45.0-rc.1` for a candidate. `--summary-file notes.md` opens the section with a summary paragraph.
3. `just release::pr 0.45.0`, and merge the pull request it opens.
4. Run the Release workflow from the Actions tab with the version.

Re-running the workflow with the same version resumes after a failure.

`just release::publish 0.45.0` runs the same steps from a machine with `cargo login` and `gh auth login` done, if Actions is unavailable.

## Release candidates

`X.Y.Z-rc.N`, published as a pre-release. `[Unreleased]` stays open until the final release promotes it. While a candidate cycle is open, no other version can be prepared.

## The API delta

`prepare` prints the public-API delta since the last release. Read it against the `[Unreleased]` section. The Release workflow stops a release whose delta needs a larger bump than the release makes.

`--skip-api-delta` on `prepare`, and the workflow's checkbox, release without the check.
