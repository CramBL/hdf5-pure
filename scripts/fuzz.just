set working-directory := ".."

# A short run of every target.
default: fuzz-all

fuzz TARGET DURATION="30":
    cargo +nightly fuzz run {{ TARGET }} --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096

fuzz-all DURATION="30":
    cargo +nightly fuzz run streaming-differential fuzz/corpus/parse-file --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096
    cargo +nightly fuzz run parse-file --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096
    cargo +nightly fuzz run roundtrip-simple --target x86_64-unknown-linux-gnu -- -max_total_time={{ DURATION }} -rss_limit_mb=4096
