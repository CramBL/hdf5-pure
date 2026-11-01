set lazy

import "constants.just"
set working-directory := ".."

# Features shared by native checks and the MAT Miri batch.
MIRI_MAT_FEATURES := "std,serde,num-complex"
# Test families in the MAT Miri batch.
MIRI_MAT_FAMILIES := "mat::complex mat::transpose"
# Features shared by native checks and the memory Miri batch.
MIRI_MEMORY_FEATURES := "std"
# Cache, assembly, read, and encoding families in the memory Miri batch.
MIRI_MEMORY_FAMILIES := "nosync chunked_read::tests::dense_assembly data_read::primitive::tests primitive_bytes::tests"

default: miri-selection miri allocation-coverage heap-baseline

# Execute each Miri family natively and fail if its selection is empty.
miri-selection:
    #!/usr/bin/env bash
    set -euo pipefail
    for family in {{ MIRI_MAT_FAMILIES }}; do
        cargo nextest run --locked -p hdf5-pure --no-default-features \
            --features {{ MIRI_MAT_FEATURES }} --lib --no-tests fail --ignore-default-filter \
            -E "test(/^$family/)"
    done
    for family in {{ MIRI_MEMORY_FAMILIES }}; do
        cargo nextest run --locked -p hdf5-pure --no-default-features \
            --features {{ MIRI_MEMORY_FEATURES }} --lib --no-tests fail --ignore-default-filter \
            -E "test(/^$family/)"
    done

# Check MAT buffers, cache synchronization, dense assembly, and primitive byte operations with Miri.
miri $MIRIFLAGS="-Zmiri-strict-provenance": miri-toolchain
    # Miri artifacts contain compiler-specific metadata. Store them under the pinned nightly so
    # artifacts from another Miri toolchain cannot share this target directory. cargo-miri adds its
    # own `miri` subdirectory below this path.
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked \
        -p hdf5-pure --no-default-features --features {{ MIRI_MAT_FEATURES }} --lib -- \
        {{ MIRI_MAT_FAMILIES }}
    # Check primitive copies into initialized vectors and their spare capacity, and primitive encoding
    # through dataset, attribute, and append callers.
    CARGO_TARGET_DIR="target/{{ NIGHTLY }}" cargo +{{ NIGHTLY }} miri test --locked \
        -p hdf5-pure --no-default-features --features {{ MIRI_MEMORY_FEATURES }} --lib -- \
        {{ MIRI_MEMORY_FAMILIES }}

# Install the pinned nightly with Miri and the standard library sources it builds.
miri-toolchain:
    if [ -z "${MIRI_SYSROOT:-}" ]; then rustup toolchain install --no-self-update --profile minimal {{ NIGHTLY }} --component miri,rust-src; fi

# Check exact allocation figures in the default feature configuration.
heap-baseline:
    cargo nextest run --locked -p hdf5-pure --test allocation_baseline --run-ignored only --no-tests fail

# Record exact allocation figures in the default feature configuration.
heap-baseline-record:
    HEAPSCOPE_UPDATE_BASELINE=1 INSTA_UPDATE=always cargo nextest run --locked -p hdf5-pure --test allocation_baseline --run-ignored only --no-tests fail

heap-profile:
    cargo run --release --example heap_profile

# Profile metadata-heavy repacks or time them without allocation recording.
repack-profile MODE="memory" SUBTREES="512" ROUNDS="5":
    CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo run --locked --release --example repack_profile -- {{ MODE }} {{ SUBTREES }} {{ ROUNDS }}

# Measure filtered creation and direct pipeline buffers.
filtered-chunk-memory:
    cargo nextest run --locked -p hdf5-pure --features std --lib --no-tests fail --ignore-default-filter --success-output immediate -E 'test(/^chunked_write::tests::allocation::/)'
    cargo nextest run --locked -p hdf5-pure-filter --features deflate --lib --no-tests fail --ignore-default-filter --success-output immediate -E 'test(/^pipeline::tests::allocation::/)'

# Execute each allocation suite on std x86_64/aarch64 hosts and fail if its selection is empty.
allocation-coverage: filtered-chunk-memory
    cargo nextest run --locked -p hdf5-pure --features std --test allocation_bounds --no-tests fail --ignore-default-filter
    cargo nextest run --locked -p hdf5-pure --features std --lib --no-tests fail --ignore-default-filter -E 'test(/^edit::object_header::tests::a_growing_extension_rewrite_keeps_the_allocation_count/)'
    cargo nextest run --locked -p hdf5-pure --features std --lib --no-tests fail --ignore-default-filter -E 'test(/^file_writer::tests::allocation::/)'
    cargo nextest run --locked -p hdf5-pure --features std --lib --no-tests fail --ignore-default-filter -E 'test(/^chunked_read::tests::btree_v1_cached_acquisition_allocates_no_second_payload/)'

# Measure filtered creation and direct pipeline throughput.
filtered-chunk-bench *ARGS:
    cargo bench --locked --bench filtered_chunks -- {{ ARGS }}

# Record direct pipeline instruction references.
filtered-pipeline-profile CASE="shuffle_deflate" DIRECTION="decode" PATTERN="cycling" SCRATCH="reused" ELEMENTS="1024" ROUNDS="100" OUTPUT="target/filter-pipeline.callgrind.%p":
    CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --locked --release --example filter_pipeline_profile
    "${VALGRIND:-valgrind}" --tool=callgrind --collect-atstart=no --toggle-collect='*profile_batch*' --callgrind-out-file={{ quote(OUTPUT) }} "${CARGO_TARGET_DIR:-target}/release/examples/filter_pipeline_profile" {{ quote(CASE) }} {{ quote(DIRECTION) }} {{ quote(PATTERN) }} {{ quote(SCRATCH) }} {{ quote(ELEMENTS) }} {{ quote(ROUNDS) }}

# Measure bounded writable opening of native fragmented PAGE files.
fragmented-recovery:
    cargo bench --locked -p {{ CROSSCHECK }} --features __hdf5-bundled --bench fragmented_recovery
