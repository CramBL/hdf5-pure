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
# Test with libhdf5 and libmatio in the `hdf5-pure-crosscheck` crate
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

ci-essentials: fmt-check clippy-full doc test-full doctest-full

# Broad local validation gate. GitHub Actions additionally runs CI-specific matrix and cross-target checks.
ci-local: nightly ci-essentials docs-rs check-release examples portability::default hygiene::default python::default prose::default api::default soundness::default interop::default test doctest

# Nextest over every workspace crate but the crosschecks, with `ARGS`.
test *ARGS:
    cargo nextest run --locked {{ WORKSPACE }} {{ ARGS }}

# The same with the facade's full-public features and `ARGS`.
test-full *ARGS:
    cargo nextest run --locked {{ WORKSPACE }} {{ FULL_PUBLIC }} {{ ARGS }}

# The facade's tests once per feature of its full-public profile, each beside the defaults.
test-each-feature *ARGS:
    cargo hack -p hdf5-pure --each-feature --include-features {{ FULL_PUBLIC_FEATURES }} --exclude-no-default-features --features default nextest run --locked {{ ARGS }}

# The doctests of every workspace crate but the crosschecks, with `ARGS`.
doctest *ARGS:
    cargo test --locked --doc {{ WORKSPACE }} {{ ARGS }}

# The same with the facade's full-public features and `ARGS`.
doctest-full *ARGS:
    cargo test --locked --doc {{ WORKSPACE }} {{ FULL_PUBLIC }} {{ ARGS }}

# The facade's doctests once per feature of its full-public profile, each beside the defaults.
doctest-each-feature *ARGS:
    cargo hack -p hdf5-pure --each-feature --include-features {{ FULL_PUBLIC_FEATURES }} --exclude-no-default-features --features default test --locked --doc {{ ARGS }}

fmt:
    cargo fmt --all

# Check Rust formatting
fmt-check:
    cargo fmt --all -- --check

# Clippy over every workspace crate but the crosschecks, with the default features.
clippy *ARGS:
    cargo clippy --locked {{ WORKSPACE }} --all-targets {{ ARGS }} -- -D warnings

# The same with the facade's full-public features, as CI and the agent gates run it.
clippy-full *ARGS:
    cargo clippy --locked {{ WORKSPACE }} {{ FULL_PUBLIC }} --all-targets {{ ARGS }} -- -D warnings

# The published crates' documentation, then the same with the private items contributors read.
doc *ARGS:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked {{ PUBLISHED }} --no-deps {{ FULL_PUBLIC }} {{ ARGS }}
    RUSTDOCFLAGS="-D warnings" cargo doc --locked {{ PUBLISHED }} --no-deps --document-private-items {{ FULL_PUBLIC }} {{ ARGS }}

# The facade's documentation with its `[package.metadata.docs.rs]` settings, then the other published crates' with their defaults.
docs-rs *ARGS:
    RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +{{ NIGHTLY }} doc --locked -p hdf5-pure --no-deps --features provenance,zfp,ndarray,serde,num-complex {{ ARGS }}
    RUSTDOCFLAGS="-D warnings" cargo +{{ NIGHTLY }} doc --locked -p hdf5-pure-core -p hdf5-pure-filter -p hdf5-pure-format --no-deps {{ ARGS }}

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
