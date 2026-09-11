set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default: semver api-surface msrv

semver:
    cargo semver-checks -p hdf5-pure --default-features --features serde,zfp,provenance,ndarray

# The release type the changelog calls for: minor with a **Breaking:** entry
# under [Unreleased], or in the dated section the diff against BASE promotes.
release-type BASE="origin/main":
    {{ uv }} release release-type --base {{ BASE }}

# The library builds on the `rust-version` Cargo.toml declares, with every feature.
msrv:
    cargo msrv verify -- cargo check --locked --lib --all-features

# Types reachable through the public API that cannot be named. Needs nightly.
api-surface:
    {{ uv }} api-surface
