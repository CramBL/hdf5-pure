#!/usr/bin/env bash
set -euo pipefail
: "${MIRI_IMAGE:?missing selected Miri image}"
: "${CARGO_HOME:?missing Cargo cache path}"
: "${RUNNER_TEMP:?missing runner temporary directory}"
nightly=$(cat scripts/nightly.txt)
mkdir -p "$CARGO_HOME" "$RUNNER_TEMP/miri-home" "$RUNNER_TEMP/miri-tmp"
exec docker run --rm --init --sig-proxy=true --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$PWD,dst=$PWD" --workdir "$PWD" \
    --mount "type=bind,src=$CARGO_HOME,dst=$CARGO_HOME" \
    --mount "type=bind,src=$RUNNER_TEMP,dst=$RUNNER_TEMP" \
    --env "HOME=$RUNNER_TEMP/miri-home" --env "TMPDIR=$RUNNER_TEMP/miri-tmp" \
    --env "RUSTUP_TOOLCHAIN=$nightly" \
    --env CARGO_HOME --env CARGO_TARGET_DIR --env CARGO_TERM_COLOR --env CARGO_INCREMENTAL \
    --env CARGO_PROFILE_TEST_OPT_LEVEL --env CARGO_PROFILE_TEST_DEBUG_ASSERTIONS \
    --env CARGO_PROFILE_TEST_OVERFLOW_CHECKS --env CARGO_PROFILE_TEST_LTO \
    --env CARGO_PROFILE_TEST_CODEGEN_UNITS --env CARGO_PROFILE_TEST_DEBUG \
    --env CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER \
    --env CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS \
    --env CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTDOCFLAGS \
    --env RUSTFLAGS --env RUSTDOCFLAGS --env CARGO_ENCODED_RUSTFLAGS --env CARGO_ENCODED_RUSTDOCFLAGS \
    --env MIRIFLAGS --env RUST_TEST_THREADS --env RUST_BACKTRACE \
    --env JUST_COLOR --env JUST_VERBOSE --env JUST_TIMESTAMP --env CI --env GITHUB_ACTIONS \
    "$MIRI_IMAGE" bash -c 'exec "$@"' bash "$@"
