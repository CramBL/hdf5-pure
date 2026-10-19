#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

family=${1:?missing image family}
candidate=${2:?missing construction-change flag}
case "$family" in i686|s390x) ;; *) exit 1 ;; esac
case "$candidate" in true|false) ;; *) exit 1 ;; esac
target="$family-unknown-linux-gnu"
if [[ $candidate == true ]]; then
    for variable in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS; do
        if [[ -v $variable ]]; then
            echo "$variable overrides the target linker flags" >&2
            exit 1
        fi
    done
    identity=$(bash scripts/ci/cross-image.sh fingerprint "$family")
    image="hdf5-pure-cross-$family:$identity"
    bash scripts/ci/cross-image.sh build "$family" "$image"
    config="$RUNNER_TEMP/cross-$family.toml"
    awk -v table="[target.$target]" -v image="$image" '
        /^\[/ { selected = ($0 == table) }
        selected && /^pre-build =/ { next }
        selected && /^image =/ { print "image = \"" image "\""; next }
        { print }
    ' Cross.toml > "$config"
    if ! grep -qxF "image = \"$image\"" "$config"; then
        echo "No $target image in Cross.toml; candidate was not selected" >&2
        exit 1
    fi
    {
        echo "CROSS_CONFIG=$config"
        echo 'CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_RUSTFLAGS=-C link-arg=-fuse-ld=mold'
        echo 'CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS=-C link-arg=-fuse-ld=mold'
    } >> "$GITHUB_ENV"
else
    image=$(awk -v table="[target.$target]" '
        /^\[/ { selected = ($0 == table) }
        selected && /^image =/ { gsub(/"/, "", $3); print $3 }
    ' Cross.toml)
    test -n "$image"
    docker pull "$image"
    identity=$(sha256sum Cross.toml | cut -d' ' -f1)
fi
printf 'image=%s\ncandidate=%s\nidentity=%s\n' "$image" "$candidate" "$identity" >> "$GITHUB_OUTPUT"
printf 'Cross image: %s\nCache environment: %s\n' "$image" "$identity"
