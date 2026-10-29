# Releases

## Versioning

`0.x.0` may break, `0.x.y` may not. Bump versions only in release pull requests.
Between releases, `Cargo.toml` on `main` keeps the last published version.

Mark every public API break `**Breaking:**` in the changelog. A marked cycle requires a minor release,
otherwise a patch. CI checks the accumulated API delta against that bump. Review checks each entry.

## Changelog

Edit `CHANGELOG.md`. One or two sentences per entry: capability, public API name, caveat if needed,
then `([#NN](url))`. Put `**Breaking:**` entries first in each section.

Start fixes with "Fixed <the bug>". Describe replacement behavior only where users need to act.
Use `### Performance` for execution time, memory, allocation or I/O improvements without API changes.
Name the operation. Back numerical claims with benchmarks.

## Cutting one

1. Start from clean `main`. Run `just ci-local`.
2. `just release::prepare VERSION`. Optional final-release summary: `--summary-file PATH`.
3. `just release::pr VERSION`, using the same version.
4. Wait for approval and passing CI, including the workspace publication dry run.
5. For unpublished crate names, follow the first-publication steps below.
6. Merge. The Release workflow publishes packages, tags the commit and creates the GitHub release.

Rerun a failed workflow job to resume. Already-published versions are skipped.

Manual fallback: check out the merged release commit. Run `cargo login`, `gh auth login`,
then `just release::publish`.

## A crate's first publication

Before merging its first release PR:

1. Check out the approved PR revision. Keep the tree clean.
2. `cargo login` with an API token that permits creating crates.
3. Publish new crates with `cargo publish`, one `-p NAME` per crate. Cargo orders uploads by dependency.
4. Add each crate's Trusted Publishing entry on crates.io: `CramBL/hdf5-pure`, workflow `release.yml`.

For the first workspace release:

```sh
cargo publish \
  -p hdf5-pure-core \
  -p hdf5-pure-filter \
  -p hdf5-pure-format \
  -p hdf5-pure-space
```

Retry only unpublished versions. Changes to published contents require a new version.

## Release candidates

Use `X.Y.Z-rc.N` with the same release steps. `[Unreleased]` stays open until the final release.
Finish the candidate cycle before preparing a different target version.

Announce candidates to dependents. Collect feedback. Publish another candidate for fixes.
Prepare the final release explicitly.

## The API delta

`prepare` reports the delta from the previous stable release. Check it against the changelog.
`publish` blocks an insufficient version bump.

`just api::release-type` reads the bump from `[Unreleased]` or the section promoted by the release PR.
`just api::semver-release-type` translates it for `cargo semver-checks`, as CI uses it.

Manual override: `--skip-api-delta` on `prepare` or `publish`.

## Migration cleanup

Keep the ownership migration bridge through candidates. After the first stable workspace release,
open a cleanup PR to remove:

- `migration.py`, `rustdoc_json.py`, `nightly.py` and their tests, `test_migration.py` and `test_nightly.py`
- the migration branches in `semver.py` and their cases in `test_semver.py`
- the `core_features` keys in `scripts/api-profiles.toml` and their handling in `api_profiles.py`
- `[package.metadata.hdf5-pure-api-migration]` and the
  `[package.metadata.cargo-semver-checks.lints]` downgrades in `Cargo.toml`
- the rustdoc JSON nightly step in `api.yml` and `release.yml`, if nothing else there needs it

Check the list against the code. CI rejects the bridge once the stable facade baseline advances.
