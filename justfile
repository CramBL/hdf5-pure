default:
    @just --list

ci-essentials: fmt-check clippy doc test-full doctest-full

test-full:
    cargo nextest run --locked --features __hdf5-bundled --features "serde zfp fast-deflate provenance ndarray"

doctest-full:
    cargo test --locked --doc --features __hdf5-bundled --features "serde zfp fast-deflate provenance ndarray"

ci: ci-essentials check-release examples check-no-std check-wasm shear semver api-surface clippy-32bit cast-gate miri (test-hdf5 "1.8.23") test doctest

test *ARGS:
    cargo nextest run --locked --features __hdf5-bundled {{ ARGS }}

# The test suite linking the HDF5 installed under HDF5_DIR.
test-external-hdf5 *ARGS:
    cargo nextest run --locked {{ ARGS }}

test-matio *ARGS:
    cargo nextest run --locked --features __hdf5-bundled --features "serde matio-crosscheck" --test serde_matio_crosscheck {{ ARGS }}

test-lib *ARGS:
    cargo test --lib --features __hdf5-bundled {{ ARGS }}

doctest *ARGS:
    cargo test --locked --doc --features __hdf5-bundled {{ ARGS }}

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy *ARGS:
    cargo clippy --locked --features __hdf5-bundled --features "serde ndarray" --all-targets {{ ARGS }} -- -D warnings

doc *ARGS:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --features "provenance zfp ndarray serde" {{ ARGS }}

check-release:
    cargo check --locked --release --all-targets --features __hdf5-bundled --features "serde ndarray"

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

# Miri over the crate's unsafe code. The recipe fails when a file no longer
# holds unsafe code, when a filter selects no tests, or when the selected tests
# do not run. Each of those would otherwise pass without checking anything, as
# happened in #113.
miri:
    #!/usr/bin/env bash
    set -euo pipefail
    log="$(mktemp)"
    for entry in "src/mat/transpose.rs:mat::transpose" "src/mat/complex.rs:mat::complex"; do
        file="${entry%%:*}"
        filter="${entry##*:}"
        if ! grep -qE '^[^/]*\bunsafe[[:space:]]*(\{|fn |impl )' "$file"; then
            echo "$file holds no unsafe code; point this recipe at the code that does." >&2
            exit 1
        fi
        MIRIFLAGS=-Zmiri-strict-provenance cargo miri test --locked --no-default-features \
            --features std,serde,__hdf5-bundled --lib "$filter" 2>&1 | tee "$log"
        if grep -q "running 0 tests" "$log"; then
            echo "Miri ran no tests for $filter: the --lib filter matched nothing." >&2
            exit 1
        fi
        if ! grep -qE '[1-9][0-9]* passed' "$log"; then
            echo "Miri passed no tests for $filter: they were selected but ignored." >&2
            exit 1
        fi
    done

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
        cargo run --locked --features __hdf5-bundled --features "serde ndarray" --example "$ex"
    done

heap-baseline:
    cargo nextest run --locked --features __hdf5-bundled --features heap-baseline --test allocation_baseline

heap-baseline-record:
    HEAPSCOPE_UPDATE_BASELINE=1 cargo test --features __hdf5-bundled --features heap-baseline --test allocation_baseline

heap-profile:
    cargo run --release --features __hdf5-bundled --example heap_profile

api-surface:
    ./scripts/check-api-surface.sh

# Build libhdf5 VERSION and its tools under tmp/hdf5/VERSION.
hdf5-build VERSION:
    uv run scripts/build_hdf5.py {{ VERSION }}

# Shell exports that make a build link the libhdf5 under tmp/hdf5/VERSION:
# eval "$(just hdf5-env 1.8.23)" && just test-external-hdf5
hdf5-env VERSION:
    @uv run scripts/build_hdf5.py {{ VERSION }} --env

# The test suite linking libhdf5 VERSION, built under tmp/hdf5 on first use.
# FEATURES names the release series from 1.10 up, as in `serde,__hdf5-1.10`.
test-hdf5 VERSION FEATURES="serde" *ARGS: (hdf5-build VERSION)
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just hdf5-env {{ VERSION }})"
    cargo nextest run --locked --features {{ FEATURES }} {{ ARGS }}

# The libhdf5 the crosscheck tests link when HDF5_DIR is unset.
hdf5-bundled-version:
    #!/usr/bin/env bash
    set -euo pipefail
    manifest=$(cargo metadata --locked --format-version 1 | jq -r '.packages[] | select(.name == "hdf5-metno-src") | .manifest_path')
    awk '/^#define H5_VERS_(MAJOR|MINOR|RELEASE) / { v[++n] = $3 } END { print v[1] "." v[2] "." v[3] }' "$(dirname "$manifest")/ext/hdf5/src/H5public.h"

check-release-script:
    ./scripts/check-release-script.sh

release *ARGS:
    ./scripts/release.sh {{ ARGS }}

clean:
    cargo clean -p hdf5-pure
