#!/usr/bin/env bash
set -euo pipefail
: "${MATIO_IMAGE:?missing selected matio image}"
: "${CARGO_HOME:?missing Cargo cache path}"
: "${RUNNER_TEMP:?missing runner temporary directory}"
for variable in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS; do
    if [[ -v $variable ]]; then
        echo "$variable overrides the target linker flags" >&2
        exit 1
    fi
done
mkdir -p "$CARGO_HOME" "$RUNNER_TEMP/matio-home" "$RUNNER_TEMP/matio-tmp"
exec docker run --rm --init --sig-proxy=true --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$PWD,dst=$PWD" --workdir "$PWD" \
    --mount "type=bind,src=$CARGO_HOME,dst=$CARGO_HOME" \
    --mount "type=bind,src=$RUNNER_TEMP,dst=$RUNNER_TEMP" \
    --env "HOME=$RUNNER_TEMP/matio-home" --env "TMPDIR=$RUNNER_TEMP/matio-tmp" \
    --env CARGO_HOME --env CARGO_TARGET_DIR --env CARGO_TERM_COLOR --env CARGO_INCREMENTAL \
    --env CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER \
    --env CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS \
    --env RUST_BACKTRACE --env JUST_COLOR --env JUST_VERBOSE --env JUST_TIMESTAMP \
    --env NEXTEST_PROFILE --env NEXTEST_NO_TESTS --env CI --env GITHUB_ACTIONS \
    "$MATIO_IMAGE" bash -c 'exec "$@"' bash "$@"
