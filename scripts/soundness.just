set lazy

import "constants.just"
set working-directory := ".."

default: miri heap-baseline

# Miri over the tests of selected unsafe and high-risk paths: the MAT complex and transpose buffers, and the single-threaded mutex.
miri $MIRIFLAGS="-Zmiri-strict-provenance": miri-toolchain
    cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std,serde --lib mat::complex
    cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std,serde --lib mat::transpose
    cargo +{{ NIGHTLY }} miri test --locked -p hdf5-pure --no-default-features --features std --lib nosync

# Install the pinned nightly with Miri and the standard library sources it builds.
miri-toolchain:
    rustup toolchain install --no-self-update --profile minimal {{ NIGHTLY }} --component miri,rust-src

heap-baseline:
    cargo nextest run --locked --features __heap-baseline --test allocation_baseline

heap-baseline-record:
    HEAPSCOPE_UPDATE_BASELINE=1 cargo test --features __heap-baseline --test allocation_baseline

heap-profile:
    cargo run --release --example heap_profile
