set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default: semver api-surface

semver:
    cargo semver-checks -p hdf5-pure --default-features --features serde,zfp,provenance,ndarray

# Types reachable through the public API that cannot be named. Needs nightly.
api-surface:
    {{ uv }} api-surface
