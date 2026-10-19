#!/usr/bin/env bash
set -euo pipefail

test ! -v HDF5_DIR
test "$RUST_TEST_THREADS" = 1
mold --version
ld.mold --version
cc --version
"${CROSS_TOOLCHAIN_PREFIX}gcc" --version
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
printf 'int main(void) { return 0; }\n' > "$work/main.c"
cc -fuse-ld=mold "$work/main.c" -o "$work/host"
readelf -h "$work/host" | grep -F 'Advanced Micro Devices X86-64'
readelf -p .comment "$work/host" | grep -F "$(mold --version)"
"$work/host"
"${CROSS_TOOLCHAIN_PREFIX}gcc" -fuse-ld=mold "$work/main.c" -o "$work/target"
readelf -h "$work/target"
case "$CROSS_TOOLCHAIN_PREFIX" in
    i686-linux-gnu-) machine='Intel 80386' ;;
    s390x-linux-gnu-) machine='IBM S/390' ;;
    *) exit 1 ;;
esac
readelf -h "$work/target" | grep -F "$machine"
readelf -p .comment "$work/target" | grep -F "$(mold --version)"
read -ra runner <<< "$CROSS_TARGET_RUNNER"
CROSS_DEBUG=1 "${runner[@]}" "$work/target"
