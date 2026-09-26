set lazy

import "constants.just"

set working-directory := ".."

# The full-public features but `fast-deflate`, whose zlib-ng C build the i686 lints have no toolchain for.
FULL_PUBLIC_WITHOUT_ZLIB_NG := "--features checksum,ndarray,num-complex,provenance,serde,zfp"
# The features the 32-bit and big-endian tests build the portable crates with, in place of their defaults.
PORTABILITY_FEATURES := "checksum,fast-deflate,ndarray,num-complex,provenance,serde,zfp"

# The cross-target tests need `cross` and Docker, so they are not run here.
default: check-no-std check-wasm check-bare-metal clippy-32bit cast-gate

# The portable crates on the host without their default features.
check-no-std:
    cargo check --locked {{ PORTABLE }} --no-default-features

# The portable crates without their default features for wasm32, then the facade with its defaults.
check-wasm:
    cargo check --locked {{ PORTABLE }} --target wasm32-unknown-unknown --no-default-features
    cargo check --locked -p hdf5-pure --target wasm32-unknown-unknown

# The portable crates without their default features, then the facade with `checksum` alone, without `std`.
check-bare-metal:
    cargo build --locked {{ PORTABLE }} --no-default-features --target thumbv7em-none-eabi
    cargo build --locked -p hdf5-pure --no-default-features --features checksum --target thumbv7em-none-eabi

# Clippy over the portable crates on i686, with the default and the full-public features.
clippy-32bit:
    cargo clippy --locked {{ PORTABLE }} --target i686-unknown-linux-gnu --lib -- -D warnings
    cargo clippy --locked {{ PORTABLE }} --target i686-unknown-linux-gnu --lib {{ FULL_PUBLIC_WITHOUT_ZLIB_NG }} -- -D warnings

# The truncating casts in the portable crates, denied on i686 where they can fire.
cast-gate:
    cargo clippy --locked {{ PORTABLE }} --target i686-unknown-linux-gnu --lib {{ FULL_PUBLIC_WITHOUT_ZLIB_NG }} -- -D clippy::cast_possible_truncation -D clippy::cast_possible_wrap -D unfulfilled_lint_expectations

test-32bit FEATURES=PORTABILITY_FEATURES:
    cross test \
        --config 'build.rustc-wrapper=""' \
        --config 'target."i686-unknown-linux-gnu".linker="i686-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        {{ PORTABLE }} \
        --target i686-unknown-linux-gnu \
        --no-default-features \
        --features "{{ FEATURES }}"

test-big-endian:
    cross test \
        --config 'build.rustc-wrapper=""' \
        --config 'target."s390x-unknown-linux-gnu".linker="s390x-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        {{ PORTABLE }} \
        --target s390x-unknown-linux-gnu \
        --no-default-features \
        --features "{{ PORTABILITY_FEATURES }}"

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
        -p {{ CROSSCHECK }} \
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
        -p {{ CROSSCHECK }} \
        --no-default-features \
        --features "hdf5,__hdf5-bundled,{{ FEATURES }}"
