set lazy

import "constants.just"
set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default: semver msrv

# Each guaranteed crate's API delta against its last release, or PACKAGE's under PROFILE alone.
semver BASE="origin/main" PACKAGE="" PROFILE="":
    {{ uv }} semver {{ if PACKAGE == "" { "" } else { "--package " + PACKAGE } }} \
        {{ if PROFILE == "" { "" } else { "--profile " + PROFILE } }} \
        --release-type "$({{ just_executable() }} api::semver-release-type {{ BASE }})"

# The release type the changelog calls for: minor with a **Breaking:** entry
# under [Unreleased], or in the dated section the diff against BASE promotes.
release-type BASE="origin/main":
    {{ uv }} release release-type --base {{ BASE }}

# The same release type in the word `cargo semver-checks --release-type` takes.
semver-release-type BASE="origin/main":
    {{ uv }} release release-type --base {{ BASE }} --for-semver-checks

# Every published crate builds on the `rust-version` Cargo.toml declares, with every feature.
msrv:
    cargo msrv verify -- cargo check --locked {{ PUBLISHED }} --lib --all-features
