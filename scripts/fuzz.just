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
    for target in $(cargo +{{ NIGHTLY }} fuzz list); do
        case "$target" in
            streaming-differential) inputs=(fuzz/corpus/parse-file) ;;
            *) inputs=() ;;
        esac
        {{ quote(just_executable()) }} fuzz::fuzz "$target" {{ DURATION }} ${inputs[@]+"${inputs[@]}"}
    done
