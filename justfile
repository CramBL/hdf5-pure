set lazy

import "scripts/constants.just"

# Test on 32-bit and big-endian via QEMU
mod portability "scripts/portability.just"
# Check unsafe with miri, and allocations with heapscope
mod soundness "scripts/soundness.just"
# Check API with `semver` and `msrv`
mod api "scripts/api.just"
# Check for unused dependencies, typos, dead links, and bad GitHub actions usage
mod hygiene "scripts/hygiene.just"
mod fuzz "scripts/fuzz.just"
# Test the facade and external interoperability suite with libhdf5 and libmatio
mod interop "scripts/interop.just"
mod release "scripts/release.just"
# format, lint, test python scripts
mod python "scripts/python.just"
# Recreate the hdf5 test files
mod test-data "scripts/test-data.just"
# Lints prose in comments & commit messages, using vale
mod prose "scripts/prose.just"

[default]
_default:
    @{{ quote(just_executable()) }} --list

CI_TEST_PROFILE := "test-optimized"

ci-essentials: fmt-check clippy-full doc (test-full "--cargo-profile" CI_TEST_PROFILE) (doctest-full "--profile" CI_TEST_PROFILE)

# Broad local validation gate. GitHub Actions additionally runs CI-specific matrix and cross-target checks.
ci-local: nightly ci-essentials clippy-feature-powerset clippy-no-std-feature-powerset docs-rs check-release examples portability::default hygiene::default python::default prose::default api::default soundness::default (interop::default CI_TEST_PROFILE) (test "--cargo-profile" CI_TEST_PROFILE) (doctest "--profile" CI_TEST_PROFILE)

# Nextest over every workspace crate but the crosschecks, with `ARGS`.
test *ARGS:
    cargo nextest run --locked {{ WORKSPACE }} {{ ARGS }}

# The same with the facade's full-public features and `ARGS`.
test-full *ARGS:
    cargo nextest run --locked {{ WORKSPACE }} {{ FULL_PUBLIC }} {{ ARGS }}

# The facade's tests with std and deflate without defaults, followed by
# each full-public feature beside the defaults.
test-each-feature *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    for features in {{ FEATURE_TEST_CONFIGURATIONS }}; do
        cargo hack -p hdf5-pure --features "$features" nextest run --locked --no-default-features {{ ARGS }}
    done

# The doctests of every workspace crate but the crosschecks, with `ARGS`.
doctest *ARGS:
    cargo test --locked --doc {{ WORKSPACE }} {{ ARGS }}

# The same with the facade's full-public features and `ARGS`.
doctest-full *ARGS:
    cargo test --locked --doc {{ WORKSPACE }} {{ FULL_PUBLIC }} {{ ARGS }}

# The facade's doctests with std and deflate without defaults, followed by
# each full-public feature beside the defaults.
doctest-each-feature *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    for features in {{ FEATURE_TEST_CONFIGURATIONS }}; do
        cargo hack -p hdf5-pure --features "$features" test --locked --doc --no-default-features {{ ARGS }}
    done

fmt:
    cargo fmt --all
    rustfmt --edition 2024 src/edit/group_inspection.rs src/edit/native_crosschecks.rs src/edit/native_crosschecks/*.rs src/repack/native_crosschecks.rs src/reader_native_crosschecks.rs src/group_v2_native_crosschecks.rs src/fractal_heap_storage/native_crosschecks.rs src/chunked_write/native_crosschecks.rs

# Check Rust formatting
fmt-check:
    cargo fmt --all -- --check
    rustfmt --check --edition 2024 src/edit/group_inspection.rs src/edit/native_crosschecks.rs src/edit/native_crosschecks/*.rs src/repack/native_crosschecks.rs src/reader_native_crosschecks.rs src/group_v2_native_crosschecks.rs src/fractal_heap_storage/native_crosschecks.rs src/chunked_write/native_crosschecks.rs

# Clippy over every workspace crate but the crosschecks, with the default features.
clippy *ARGS:
    cargo clippy --locked {{ WORKSPACE }} --all-targets {{ ARGS }} -- -D warnings

# The same with the facade's full-public features, as CI and the agent gates run it.
clippy-full *ARGS:
    cargo clippy --locked {{ WORKSPACE }} {{ FULL_PUBLIC }} --all-targets {{ ARGS }} -- -D warnings

# Pairwise Clippy over the facade's public features with defaults and dev-dependencies disabled.
clippy-feature-powerset *ARGS:
    cargo hack clippy -p hdf5-pure --feature-powerset --depth 2 --include-features {{ PUBLIC_FEATURES }} --exclude-features default --no-dev-deps --lib {{ ARGS }} -- -D warnings

# Full Clippy coverage of public feature combinations that remain `no_std` on bare metal.
clippy-no-std-feature-powerset *ARGS:
    cargo hack clippy -p hdf5-pure --target thumbv7em-none-eabi --feature-powerset --include-features {{ NO_STD_PUBLIC_FEATURES }} --exclude-features default --no-dev-deps --lib {{ ARGS }} -- -D warnings

# The published crates' documentation, then the same with the private items contributors read.
doc *ARGS:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked {{ PUBLISHED }} --no-deps {{ FULL_PUBLIC }} {{ ARGS }}
    RUSTDOCFLAGS="-D warnings" cargo doc --locked {{ PUBLISHED }} --no-deps --document-private-items {{ FULL_PUBLIC }} {{ ARGS }}

# The facade's documentation with its `[package.metadata.docs.rs]` settings, then the other published crates' with their defaults.
docs-rs *ARGS:
    RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +{{ NIGHTLY }} doc --locked -p hdf5-pure --no-deps --features provenance,zfp,ndarray,serde,num-complex {{ ARGS }}
    RUSTDOCFLAGS="-D warnings" cargo +{{ NIGHTLY }} doc --locked -p hdf5-pure-core -p hdf5-pure-filter -p hdf5-pure-format -p hdf5-pure-space --no-deps {{ ARGS }}

# Cargo check over every workspace crate but the crosschecks, with the default features.
check *ARGS:
    cargo check --locked {{ WORKSPACE }} --all-targets {{ ARGS }}

# The same in release mode with the facade's full-public features, as CI runs it.
check-release *ARGS:
    cargo check --locked {{ WORKSPACE }} --release --all-targets {{ FULL_PUBLIC }} {{ ARGS }}

# Every example of every workspace package, each with the features it requires.
examples:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "requires 'jq'" >&2; exit 1; }
    cargo metadata --locked --no-deps --format-version 1 \
        | jq -r '.packages[] | .name as $package | .targets[] | select(.kind[] == "example") | [$package, .name, ((."required-features" // []) | join(","))] | @tsv' \
        | while IFS=$'\t' read -r package example features; do
            cargo run --locked -p "$package" --example "$example" ${features:+--features "$features"} </dev/null
        done

# Install the pinned nightly the rustdoc JSON, docs.rs and fuzzing checks use.
nightly:
    rustup toolchain install --no-self-update --profile minimal {{ NIGHTLY }}

# Everything cargo built for the workspace.
clean:
    cargo clean

alias c := check
alias l := clippy
alias t := test
alias tf := test-full
alias e := examples
alias d := doc
alias cie := ci-essentials
alias cil := ci-local
