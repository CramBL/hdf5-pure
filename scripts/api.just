import "constants.just"
set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default: semver api-surface msrv

# The public API delta against the last release, for the release type the changelog calls for.
semver BASE="origin/main":
    cargo semver-checks -p hdf5-pure --default-features --features "{{ USER_FACING_FEATURES }}" \
        --release-type "$({{ just_executable() }} api::semver-release-type {{ BASE }})"

# The release type the changelog calls for: minor with a **Breaking:** entry
# under [Unreleased], or in the dated section the diff against BASE promotes.
release-type BASE="origin/main":
    {{ uv }} release release-type --base {{ BASE }}

# The same release type in the word `cargo semver-checks --release-type` takes.
semver-release-type BASE="origin/main":
    {{ uv }} release release-type --base {{ BASE }} --for-semver-checks

# The library builds on the `rust-version` Cargo.toml declares, with every feature.
msrv:
    cargo msrv verify -- cargo check --locked --lib --all-features

# Types reachable through the public API that cannot be named. Needs nightly.
api-surface:
    {{ uv }} api-surface
