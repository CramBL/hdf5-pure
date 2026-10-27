set lazy

import "constants.just"

set working-directory := ".."

# The full-public features but `fast-deflate`, whose zlib-ng C build the i686 lints have no toolchain for.
FULL_PUBLIC_WITHOUT_ZLIB_NG := "--features checksum,ndarray,num-complex,provenance,serde,zfp"
# The features the 32-bit and big-endian tests build the portable crates with, in place of their defaults.
PORTABILITY_FEATURES := "checksum,fast-deflate,ndarray,num-complex,provenance,serde,zfp"
CROSS_VERBOSE := if env("CI_CROSS_CANDIDATE", "false") == "true" { "-vv" } else { "" }
# The package whose tests run the in-memory API on wasm32.
WASM_TEST := "hdf5-pure-wasm-test"

# The cross-target tests need `cross` and Docker, and the wasm32 tests a `wasm-bindgen-test-runner`, so they are not run here.
default: check-no-std check-wasm check-bare-metal clippy-32bit cast-gate

# The portable crates on the host without their default features.
check-no-std:
    cargo check --locked {{ PORTABLE }} --no-default-features

# The portable crates without their default features for wasm32, the facade with its defaults, then Clippy over the wasm32 tests.
check-wasm:
    cargo check --locked {{ PORTABLE }} --target wasm32-unknown-unknown --no-default-features
    cargo check --locked -p hdf5-pure --target wasm32-unknown-unknown
    cargo clippy --locked -p {{ WASM_TEST }} --target wasm32-unknown-unknown --all-targets -- -D warnings

# The wasm32 tests under Node, which need the `wasm-bindgen-test-runner` of the wasm-bindgen release `Cargo.lock` pins.
test-wasm:
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test --locked -p {{ WASM_TEST }} --target wasm32-unknown-unknown

# The wasm-bindgen release `Cargo.lock` pins, which the runner that `test-wasm` calls must match.
wasm-bindgen-version:
    @cargo pkgid --locked wasm-bindgen | sed 's/.*@//'

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
    cross test {{ CROSS_VERBOSE }} \
        --config 'build.rustc-wrapper=""' \
        --config 'target."i686-unknown-linux-gnu".linker="i686-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        {{ PORTABLE }} \
        --target i686-unknown-linux-gnu \
        --no-default-features \
        --features "{{ FEATURES }}"

test-big-endian:
    cross test {{ CROSS_VERBOSE }} \
        --config 'build.rustc-wrapper=""' \
        --config 'target."s390x-unknown-linux-gnu".linker="s390x-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        {{ PORTABLE }} \
        --target s390x-unknown-linux-gnu \
        --no-default-features \
        --features "{{ PORTABILITY_FEATURES }}"

# The crosschecks against the C library, cross-compiled for the same 32-bit target `test-32bit` checks.
test-32bit-hdf5 FEATURES="serde,ndarray,provenance":
    cross test {{ CROSS_VERBOSE }} \
        --config 'build.rustc-wrapper=""' \
        --config 'target."i686-unknown-linux-gnu".linker="i686-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target i686-unknown-linux-gnu \
        -p {{ CROSSCHECK }} \
        --no-default-features \
        --features "hdf5,__hdf5-bundled,{{ FEATURES }}"

    cross test {{ CROSS_VERBOSE }} \
        --config 'build.rustc-wrapper=""' \
        --config 'target."i686-unknown-linux-gnu".linker="i686-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target i686-unknown-linux-gnu \
        -p {{ CROSSCHECK }} -p hdf5-pure --lib native_crosschecks \
        --no-default-features \
        --features "hdf5,__hdf5-bundled,{{ FEATURES }}"

# The crosschecks against the C library, cross-compiled and linked for the
# same big-endian target `test-big-endian` checks. `QEMU_TEST_SHARD` defaults to `all`.
# `free-space` selects names containing `file_space`, and `other` excludes them.
test-big-endian-hdf5 FEATURES="serde,ndarray,provenance":
    #!/usr/bin/env bash
    set -euo pipefail
    test_args=()
    case "${QEMU_TEST_SHARD-all}" in
        all) ;;
        free-space) test_args=(-- file_space) ;;
        other) test_args=(-- --skip file_space) ;;
        *) printf 'Unknown QEMU_TEST_SHARD: %q\n' "$QEMU_TEST_SHARD" >&2; exit 1 ;;
    esac
    cross test {{ CROSS_VERBOSE }} \
        --config 'build.rustc-wrapper=""' \
        --config 'target."s390x-unknown-linux-gnu".linker="s390x-linux-gnu-gcc"' \
        --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
        --locked \
        --target s390x-unknown-linux-gnu \
        -p {{ CROSSCHECK }} \
        --no-default-features \
        --features "hdf5,__hdf5-bundled,{{ FEATURES }}" \
        "${test_args[@]}"

    test_native() {
        cross test {{ CROSS_VERBOSE }} \
            --config 'build.rustc-wrapper=""' \
            --config 'target."s390x-unknown-linux-gnu".linker="s390x-linux-gnu-gcc"' \
            --config 'target."x86_64-unknown-linux-gnu".linker="cc"' \
            --locked \
            --target s390x-unknown-linux-gnu \
            -p {{ CROSSCHECK }} -p hdf5-pure --lib \
            --no-default-features \
            --features "hdf5,__hdf5-bundled,{{ FEATURES }}" \
            "$@"
    }
    case "${QEMU_TEST_SHARD-all}" in
        all) test_native -- native_crosschecks ;;
        other) test_native -- native_crosschecks --skip file_space ;;
        free-space)
            # The test harness combines positional filters with OR. List native crosschecks first
            # to select only those whose names also contain `file_space`.
            native_tests="$(test_native -- native_crosschecks --list)"
            native_args=(-- --exact)
            while IFS= read -r test; do
                if [[ "$test" == *file_space*": test" ]]; then
                    native_args+=("${test%: test}")
                fi
            done <<< "$native_tests"
            if (( ${#native_args[@]} > 2 )); then
                test_native "${native_args[@]}"
            fi
            ;;
    esac
