set working-directory := ".."

# The cross-target tests need `cross` and Docker, so they are not run here.
default: check-no-std check-wasm check-bare-metal clippy-32bit cast-gate

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

test-32bit FEATURES="std checksum deflate zfp serde ndarray provenance":
    cross test \
        --config 'build.rustc-wrapper=""' \
        --config 'target."i686-unknown-linux-gnu".linker="i686-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target i686-unknown-linux-gnu \
        --no-default-features \
        --features "{{ FEATURES }}"

test-big-endian:
    cross test --locked --target s390x-unknown-linux-gnu --no-default-features --features "std checksum deflate zfp serde ndarray provenance" --lib --test write_read_roundtrip --test serde_roundtrip --test complex_bulk_array --test complex_integer --test streaming_reader --test vlen_strings --test fixed_strings
