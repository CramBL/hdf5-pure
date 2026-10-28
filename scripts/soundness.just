set lazy

import "constants.just"
set working-directory := ".."

default: miri heap-baseline

# Check MAT buffers, cache synchronization, dense assembly, and primitive byte operations with Miri.
miri $MIRIFLAGS="-Zmiri-strict-provenance": miri-toolchain
    # Miri artifacts contain compiler-specific metadata. Store them under the pinned nightly so
    # artifacts from another Miri toolchain cannot share this target directory. cargo-miri adds its
    # own `miri` subdirectory below this path.
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked \
        -p hdf5-pure --no-default-features --features std,serde --lib -- \
        mat::complex mat::transpose
    # Check primitive copies into initialized vectors and their spare capacity, and primitive encoding
    # through dataset, attribute, and append callers.
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked \
        -p hdf5-pure --no-default-features --features std --lib -- \
        nosync chunked_read::tests::dense_assembly data_read::primitive::tests primitive_bytes::tests

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

# Profile metadata-heavy repacks or time them without allocation recording.
repack-profile MODE="memory" SUBTREES="512" ROUNDS="5":
    CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo run --locked --release --example repack_profile -- {{ MODE }} {{ SUBTREES }} {{ ROUNDS }}

# Measure raw staging, retained encoding, and completed output for filtered creation.
filtered-chunk-memory:
    cargo test --locked -p hdf5-pure --lib chunked_write::allocation_tests -- --nocapture

# Measure filtered dataset creation throughput for the shared chunk workloads.
filtered-chunk-bench *ARGS:
    cargo bench --locked --bench filtered_chunks -- {{ ARGS }}
