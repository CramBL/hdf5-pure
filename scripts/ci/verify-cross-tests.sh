#!/usr/bin/env bash
set -euo pipefail

target=$1
image=$2
directory="${CARGO_TARGET_DIR:-target}/$target/debug"
binary=$(find "$directory/deps" -maxdepth 1 -type f -executable ! -name '*.so' -print -quit)
test -n "$binary"
case "$target" in
    i686-unknown-linux-gnu) machine='Intel 80386' ;;
    s390x-unknown-linux-gnu) machine='IBM S/390' ;;
    *) exit 1 ;;
esac
readelf -h "$binary"
readelf -h "$binary" | grep -F "$machine"
readelf -p .comment "$binary"
mold_version=$(docker run --rm "$image" mold --version)
readelf -p .comment "$binary" | grep -F "$mold_version"
if [[ ${3:-} == bundled ]]; then
    test ! -v HDF5_DIR
    archive=$(find "$directory/build" -path '*/out/*' -name libhdf5.a -print -quit)
    test -n "$archive"
    echo "Bundled static HDF5: $archive"
    readelf -h "$archive" | grep -F "$machine" > /dev/null
    find "$directory/build" -name CMakeCache.txt -print
fi
