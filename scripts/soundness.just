set lazy

import "constants.just"
set working-directory := ".."

default: miri heap-baseline

# Check MAT buffers, single-threaded cache synchronization, and dense output assembly with Miri.
miri $MIRIFLAGS="-Zmiri-strict-provenance": miri-toolchain
    # Miri artifacts contain compiler-specific metadata. Store them under the pinned nightly so
    # artifacts from another Miri toolchain cannot share this target directory. cargo-miri adds its
    # own `miri` subdirectory below this path.
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std,serde --lib mat::complex
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std,serde --lib mat::transpose
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std --lib nosync
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std --lib chunked_read::tests::dense_assembly
    # Check exact primitive copies into initialized vectors and their spare capacity.
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std --lib data_read::primitive::tests
    # Check primitive encoding and its dataset, attribute, and append callers.
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std --lib primitive_bytes::tests

# Install the pinned nightly with Miri and the standard library sources it builds.
miri-toolchain:
    if [ -z "${MIRI_SYSROOT:-}" ]; then rustup toolchain install --no-self-update --profile minimal {{ NIGHTLY }} --component miri,rust-src; fi

# Check exact allocation figures in the default feature configuration.
heap-baseline:
    cargo nextest run --locked -p hdf5-pure --test allocation_baseline --run-ignored only --no-tests fail

# Record exact allocation figures in the default feature configuration.
heap-baseline-record:
    HEAPSCOPE_UPDATE_BASELINE=1 cargo nextest run --locked -p hdf5-pure --test allocation_baseline --run-ignored only --no-tests fail

heap-profile:
    cargo run --release --example heap_profile

# Measure raw staging, retained encoding, and completed output for filtered creation.
filtered-chunk-memory:
    cargo test --locked -p hdf5-pure --lib chunked_write::allocation_tests -- --nocapture

# Measure filtered dataset creation throughput for the shared chunk workloads.
filtered-chunk-bench *ARGS:
    cargo bench --locked --bench filtered_chunks -- {{ ARGS }}
