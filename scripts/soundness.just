set working-directory := ".."

default: miri heap-baseline

# Miri over the crate's unsafe code. The recipe fails when a file no longer
# holds unsafe code, when a filter selects no tests, or when the selected tests
# do not run. Each of those would otherwise pass without checking anything, as
# happened in #113.
miri:
    #!/usr/bin/env bash
    set -euo pipefail
    log="$(mktemp)"
    for entry in "src/mat/transpose.rs:mat::transpose" "src/mat/complex.rs:mat::complex"; do
        file="${entry%%:*}"
        filter="${entry##*:}"
        if ! grep -qE '^[^/]*\bunsafe[[:space:]]*(\{|fn |impl )' "$file"; then
            echo "$file holds no unsafe code; point this recipe at the code that does." >&2
            exit 1
        fi
        MIRIFLAGS=-Zmiri-strict-provenance cargo miri test --locked --no-default-features \
            --features std,serde --lib "$filter" 2>&1 | tee "$log"
        if grep -q "running 0 tests" "$log"; then
            echo "Miri ran no tests for $filter: the --lib filter matched nothing." >&2
            exit 1
        fi
        if ! grep -qE '[1-9][0-9]* passed' "$log"; then
            echo "Miri passed no tests for $filter: they were selected but ignored." >&2
            exit 1
        fi
    done

heap-baseline:
    cargo nextest run --locked --features heap-baseline --test allocation_baseline

heap-baseline-record:
    HEAPSCOPE_UPDATE_BASELINE=1 cargo test --features heap-baseline --test allocation_baseline

heap-profile:
    cargo run --release --example heap_profile
