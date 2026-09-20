import "scripts/constants.just"

# Test on 32-bit and big-endian via QEMU
mod portability "scripts/portability.just"
# Check unsafe with miri, and allocations with heapscope
mod soundness "scripts/soundness.just"
# Check API with `semver`, `msrv`, and a python api-surface checking script
mod api "scripts/api.just"
# Check for unused dependencies, typos, dead links, and bad GitHub actions usage
mod hygiene "scripts/hygiene.just"
mod fuzz "scripts/fuzz.just"
# Test with libhdf5 and libmatio in the `hdf5-pure-crosstest` crate
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
    @just --list

ci-essentials: fmt-check clippy-full doc test-full doctest-full

# Everything CI runs except fuzzing.
ci: ci-essentials docs-rs check-release examples portability::default hygiene::default python::default prose::default api::default soundness::default interop::default test doctest

# run nextest with `ARGS`
test *ARGS:
    cargo nextest run --locked {{ ARGS }}

# run nextest with all user-facing features and `ARGS`
test-full *ARGS:
    cargo nextest run --locked --features "{{ USER_FACING_FEATURES }}" {{ ARGS }}

# Each optional feature on its own beside the defaults, one run per feature.
test-each-feature *ARGS:
    cargo hack --each-feature --include-features serde,zfp,ndarray --exclude-no-default-features --features default nextest run --locked {{ ARGS }}

# Run rustdoc tests with `ARGS`
doctest *ARGS:
    cargo test --locked --doc {{ ARGS }}

# Run rustdoc tests with all user-facing features and `ARGS`
doctest-full *ARGS:
    cargo test --locked --doc --features "{{ USER_FACING_FEATURES }}" {{ ARGS }}

doctest-each-feature *ARGS:
    cargo hack --each-feature --include-features serde,zfp,ndarray --exclude-no-default-features --features default test --locked --doc {{ ARGS }}

fmt:
    cargo fmt --all

# Check Rust formatting
fmt-check:
    cargo fmt --all -- --check

# clippy with the default features, sharing a build cache with `check` and `test`
clippy *ARGS:
    cargo clippy --locked --all-targets {{ ARGS }} -- -D warnings

# clippy with all user-facing features, as CI and the agent gates run it
clippy-full *ARGS:
    cargo clippy --locked --features "{{ USER_FACING_FEATURES }}" --all-targets {{ ARGS }} -- -D warnings

# The published documentation, then the same with the private items contributors read.
doc *ARGS:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --features "{{ USER_FACING_FEATURES }}" {{ ARGS }}
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items --features "{{ USER_FACING_FEATURES }}" {{ ARGS }}

# The published documentation as docs.rs builds it: nightly, the feature badges on.
docs-rs *ARGS:
    RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc --locked --no-deps --features "{{ USER_FACING_FEATURES }}" {{ ARGS }}

# cargo check with the default features, sharing a build cache with `clippy` and `test`
check *ARGS:
    cargo check --locked --all-targets {{ ARGS }}

# cargo check --release, with all user-facing features, as CI runs it
check-release *ARGS:
    cargo check --locked --release --all-targets --features "{{ USER_FACING_FEATURES }}" {{ ARGS }}

# Run all examples
examples:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq || (echo "requires 'jq'"; exit 1)
    for ex in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].targets[] | select(.kind[] == "example") | .name'); do
        cargo run --locked --features "serde ndarray" --example "$ex"
    done

# cargo clean
clean:
    cargo clean -p hdf5-pure
    cargo clean -p hdf5-pure-crosscheck

alias c := check
alias l := clippy
alias t := test
alias tf := test-full
alias e := examples
alias d := doc
