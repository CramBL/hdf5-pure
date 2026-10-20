set lazy

import "constants.just"
set working-directory := ".."

# A short run of every target.
default: fuzz-all

# One target for DURATION seconds, from its corpus or from the INPUTS given, such as a crash file.
fuzz TARGET DURATION="30" *INPUTS:
    cargo +{{ NIGHTLY }} fuzz run {{ TARGET }} {{ INPUTS }} --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096

# DURATION seconds of every target `cargo fuzz list` names, streaming-differential from the parse-file corpus.
fuzz-all DURATION="30":
    #!/usr/bin/env bash
    set -euo pipefail
    targets=$(cargo +{{ NIGHTLY }} fuzz list)
    cargo +{{ NIGHTLY }} fuzz build --target x86_64-unknown-linux-gnu
    pids=()
    cleanup() {
        result=$?
        trap - EXIT INT TERM
        for pid in "${pids[@]}"; do
            kill -TERM -- "-$pid" 2>/dev/null || true
        done
        wait || true
        exit "$result"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    set -m
    (
        trap - EXIT INT TERM
        set +m
        for target in $targets; do
            if [[ "$target" == roundtrip-simple ]]; then
                {{ quote(just_executable()) }} fuzz::fuzz "$target" {{ DURATION }}
            fi
        done
    ) &
    pids+=("$!")
    (
        trap - EXIT INT TERM
        set +m
        result=0
        for target in $targets; do
            case "$target" in
                roundtrip-simple | streaming-differential) continue ;;
            esac
            {{ quote(just_executable()) }} fuzz::fuzz "$target" {{ DURATION }} || result=1
        done
        for target in $targets; do
            if [[ "$target" == streaming-differential ]]; then
                {{ quote(just_executable()) }} fuzz::fuzz "$target" {{ DURATION }} fuzz/corpus/parse-file || result=1
            fi
        done
        exit "$result"
    ) &
    pids+=("$!")
    set +m
    result=0
    for pid in "${pids[@]}"; do
        wait "$pid" || result=1
    done
    exit "$result"
