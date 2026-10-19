#!/usr/bin/env bash
set -euo pipefail
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
rustc --version --verbose
cargo --version
just --version
cargo nextest --version
mold --version
nm -D /opt/matio/lib/libmatio.so | grep -w Mat_VarCreateStruct2
cat > "$work/probe.c" <<'C'
#include <matio.h>
int main(void) {
    const size_t dims[] = {1, 1};
    const char *fields[] = {"value", 0};
    matvar_t *var = Mat_VarCreateStruct2("probe", 2, dims, fields);
    if (!var) return 1;
    Mat_VarFree(var);
    return 0;
}
C
cc -I/opt/matio/include "$work/probe.c" -L/opt/matio/lib -lmatio \
    -Wl,-rpath,/opt/matio/lib -fuse-ld=mold -o "$work/probe"
readelf -h "$work/probe"
readelf -p .comment "$work/probe" | grep -F "$(mold --version)"
ldd "$work/probe"
"$work/probe"
test -d /opt/matio-licenses/libmatio-1.5.30-he0a2e19_0
