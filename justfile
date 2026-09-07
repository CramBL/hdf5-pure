mod portability "scripts/portability.just"
mod soundness "scripts/soundness.just"
mod api "scripts/api.just"
mod hygiene "scripts/hygiene.just"
mod fuzz "scripts/fuzz.just"
mod interop "scripts/interop.just"
mod release "scripts/release.just"
mod docs "scripts/docs.just"

default:
    @just --list

ci-essentials: fmt-check clippy doc test-full doctest-full

# Everything CI runs, against the last 1.8 release only for interop.
ci: ci-essentials check-release examples portability::default hygiene::default api::default soundness::default (interop::test-hdf5 "1.8.23") test doctest

test *ARGS:
    cargo nextest run --locked --features __hdf5-bundled {{ ARGS }}

test-full *ARGS:
    cargo nextest run --locked --features __hdf5-bundled --features "serde zfp fast-deflate provenance ndarray" {{ ARGS }}

test-lib *ARGS:
    cargo test --lib --features __hdf5-bundled {{ ARGS }}

doctest *ARGS:
    cargo test --locked --doc --features __hdf5-bundled {{ ARGS }}

doctest-full *ARGS:
    cargo test --locked --doc --features __hdf5-bundled --features "serde zfp fast-deflate provenance ndarray" {{ ARGS }}

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy *ARGS:
    cargo clippy --locked --features __hdf5-bundled --features "serde ndarray" --all-targets {{ ARGS }} -- -D warnings

doc *ARGS:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --features "provenance zfp ndarray serde" {{ ARGS }}

check-release *ARGS:
    cargo check --locked --release --all-targets --features __hdf5-bundled --features "serde ndarray" {{ ARGS }}

examples:
    #!/usr/bin/env bash
    set -euo pipefail
    for ex in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].targets[] | select(.kind[] == "example") | .name'); do
        cargo run --locked --features __hdf5-bundled --features "serde ndarray" --example "$ex"
    done

clean:
    cargo clean -p hdf5-pure
