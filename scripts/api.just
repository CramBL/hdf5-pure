set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default: semver api-surface msrv

semver:
    cargo semver-checks -p hdf5-pure --default-features --features serde,zfp,provenance,ndarray

# The library builds on the `rust-version` Cargo.toml declares, with every feature.
msrv:
    cargo msrv verify -- cargo check --locked --lib --all-features

# Types reachable through the public API that cannot be named. Needs nightly.
api-surface:
    {{ uv }} api-surface
