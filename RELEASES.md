# Releases

## Versioning

`0.x.0` may break, `0.x.y` may not. `Cargo.toml` on `main` reads the last release, and no pull request touches it. A pull request that breaks the public API marks its changelog entry `**Breaking:**`, and the release decides the bump from that: a cycle with a marked entry releases as the next minor, and any other as a patch. `prepare` rejects a patch version over a marked cycle, and CI checks each pull request's API delta against the release type the marker implies, so a break without the marker fails there.

## Changelog

One or two sentences per entry: the capability, the public API name, one caveat clause, then `([#NN](url))`. Prefix breaking changes with `**Breaking:**`. `d5a966b` is the model.

A released section may open with a summary paragraph, written by hand.

`CHANGELOG.md` is the source.

## Cutting one

1. `just ci`
2. `just release::prepare 0.45.0`, or `just release::prepare 0.45.0-rc.1` for a candidate. `--summary-file notes.md` opens the section with a summary paragraph.
3. `just release::pr 0.45.0`, and merge the pull request it opens. The Release workflow runs on the merge.

Re-running the workflow's job resumes after a failure.

`just release::publish` runs the same steps from a checkout of the merged commit on a machine with `cargo login` and `gh auth login` done, if Actions is unavailable.

## Release candidates

`X.Y.Z-rc.N`, published as a pre-release. `[Unreleased]` stays open until the final release promotes it. While a candidate cycle is open, no other version can be prepared.

## The API delta

`prepare` prints the public-API delta since the last release. Read it against the `[Unreleased]` section. The Release workflow stops a release whose delta needs a larger bump than the release makes, which catches a break that no entry marked. `just api::release-type` prints the release type the section calls for, which is what CI checks a pull request against.

`--skip-api-delta` on `prepare`, and the workflow's checkbox, release without the check.
