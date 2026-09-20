import "constants.just"

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
    cargo clippy --locked --target i686-unknown-linux-gnu --lib --features "{{ USER_FACING_FEATURES_PORTABILITY }}" -- -D warnings

cast-gate:
    cargo clean -p hdf5-pure --target i686-unknown-linux-gnu
    cargo clippy --locked --target i686-unknown-linux-gnu --lib --features "{{ USER_FACING_FEATURES_PORTABILITY }}" -- -D clippy::cast_possible_truncation -D clippy::cast_possible_wrap -D unfulfilled_lint_expectations

test-32bit FEATURES=USER_FACING_FEATURES:
    cross test \
        --config 'build.rustc-wrapper=""' \
        --config 'target."i686-unknown-linux-gnu".linker="i686-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target i686-unknown-linux-gnu \
        --no-default-features \
        --features "{{ FEATURES }}"

test-big-endian:
    cross test \
        --config 'build.rustc-wrapper=""' \
        --config 'target."s390x-unknown-linux-gnu".linker="s390x-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target s390x-unknown-linux-gnu \
        --no-default-features \
        --features "{{ USER_FACING_FEATURES }}"

# The crosschecks against the C library, cross-compiled and linked for the
# same 32-bit target `test-32bit` checks. `Cross.toml` installs the newer
# cmake the bundled build needs.
test-32bit-hdf5 FEATURES="serde,ndarray":
    cross test \
        --config 'build.rustc-wrapper=""' \
        --config 'target."i686-unknown-linux-gnu".linker="i686-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target i686-unknown-linux-gnu \
        -p hdf5-pure-crosscheck \
        --no-default-features \
        --features "hdf5,__hdf5-bundled,{{ FEATURES }}"

# The crosschecks against the C library, cross-compiled and linked for the
# same big-endian target `test-big-endian` checks.
test-big-endian-hdf5 FEATURES="serde,ndarray":
    cross test \
        --config 'build.rustc-wrapper=""' \
        --config 'target."s390x-unknown-linux-gnu".linker="s390x-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target s390x-unknown-linux-gnu \
        -p hdf5-pure-crosscheck \
        --no-default-features \
        --features "hdf5,__hdf5-bundled,{{ FEATURES }}"
