set working-directory := ".."

default: semver api-surface

semver:
    cargo semver-checks --default-features --features serde,zfp,provenance,ndarray

api-surface:
    ./scripts/check-api-surface.sh
