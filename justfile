default:
    @just --list

ci-essentials: fmt-check clippy doc test-full doctest-full

test-full:
    cargo nextest run --locked --features "serde zfp fast-deflate provenance ndarray"

doctest-full:
    cargo test --locked --doc --features "serde zfp fast-deflate provenance ndarray"

ci: ci-essentials check-release examples check-no-std check-wasm shear semver api-surface clippy-32bit cast-gate miri hdf5-compat test doctest

test *ARGS:
    cargo nextest run --locked {{ ARGS }}

test-matio *ARGS:
    cargo nextest run --locked --features "serde matio-crosscheck" --test serde_matio_crosscheck {{ ARGS }}

test-lib *ARGS:
    cargo test --lib {{ ARGS }}

doctest *ARGS:
    cargo test --locked --doc {{ ARGS }}

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy *ARGS:
    cargo clippy --locked --features "serde ndarray" --all-targets {{ ARGS }} -- -D warnings

doc *ARGS:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --features "provenance zfp ndarray serde" {{ ARGS }}

check-release:
    cargo check --locked --release --all-targets --features "serde ndarray"

check-no-std:
    cargo check --locked --no-default-features

check-wasm:
    cargo check --locked --target wasm32-unknown-unknown --no-default-features
    cargo check --locked --target wasm32-unknown-unknown

check-bare-metal:
    cargo build --locked --no-default-features --target thumbv7em-none-eabi

clippy-32bit:
    cargo clippy --locked --target i686-unknown-linux-gnu --lib -- -D warnings
    cargo clippy --locked --target i686-unknown-linux-gnu --lib --features "serde zfp provenance ndarray" -- -D warnings

cast-gate:
    cargo clean -p hdf5-pure --target i686-unknown-linux-gnu
    cargo clippy --locked --target i686-unknown-linux-gnu --lib --features "serde zfp provenance ndarray" -- -D clippy::cast_possible_truncation -D clippy::cast_possible_wrap -D unfulfilled_lint_expectations

test-32bit:
    cross test --locked --target i686-unknown-linux-gnu --no-default-features --features "std checksum deflate zfp serde ndarray provenance" --lib --test write_read_roundtrip --test serde_roundtrip --test zfp_roundtrip --test streaming_reader --test dense_attr_limits --test dense_attr_vlen --test attr_message_size_limit --test userblock_dense_attrs --test c_1_8_read_compat --test fixed_strings

test-big-endian:
    cross test --locked --target s390x-unknown-linux-gnu --no-default-features --features "std checksum deflate zfp serde ndarray provenance" --lib --test write_read_roundtrip --test serde_roundtrip --test complex_bulk_array --test complex_integer --test streaming_reader --test vlen_strings --test fixed_strings

miri:
    MIRIFLAGS=-Zmiri-strict-provenance cargo miri test --locked --no-default-features --features std,serde --lib mat::transpose
    MIRIFLAGS=-Zmiri-strict-provenance cargo miri test --locked --no-default-features --features std,serde --lib mat::complex

fuzz TARGET DURATION="30":
    cargo +nightly fuzz run {{ TARGET }} --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096

fuzz-all DURATION="30":
    cargo +nightly fuzz run streaming_differential fuzz/corpus/parse_file --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096
    cargo +nightly fuzz run parse_file --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096
    cargo +nightly fuzz run roundtrip_simple --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096

shear:
    cargo shear

semver:
    cargo semver-checks --default-features --features serde,zfp,provenance,ndarray

examples:
    #!/usr/bin/env bash
    set -euo pipefail
    for ex in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].targets[] | select(.kind[] == "example") | .name'); do
        cargo run --locked --features "serde ndarray" --example "$ex"
    done

heap-baseline:
    cargo nextest run --locked --features heap-baseline --test allocation_baseline

heap-baseline-record:
    HEAPSCOPE_UPDATE_BASELINE=1 cargo test --features heap-baseline --test allocation_baseline

heap-profile:
    cargo run --release --example heap_profile

api-surface:
    ./scripts/check-api-surface.sh

hdf5-compat *ARGS: (hdf5-build ARGS) (hdf5-check ARGS)

hdf5-build *ARGS:
    uv run scripts/check_hdf5_compat.py --prepare {{ ARGS }}

hdf5-check *ARGS:
    uv run scripts/check_hdf5_compat.py {{ ARGS }}

# The libhdf5 the crosscheck tests link when HDF5_DIR is unset.
hdf5-bundled-version:
    #!/usr/bin/env bash
    set -euo pipefail
    manifest=$(cargo metadata --locked --format-version 1 | jq -r '.packages[] | select(.name == "hdf5-metno-src") | .manifest_path')
    awk '/^#define H5_VERS_(MAJOR|MINOR|RELEASE) / { v[++n] = $3 } END { print v[1] "." v[2] "." v[3] }' "$(dirname "$manifest")/ext/hdf5/src/H5public.h"

verify-fixtures *ARGS:
    uv run scripts/verify_fixtures.py {{ ARGS }}

lock-python:
    uv lock --script scripts/verify_fixtures.py

check-release-script:
    ./scripts/check-release-script.sh

release *ARGS:
    ./scripts/release.sh {{ ARGS }}

clean:
    cargo clean -p hdf5-pure
