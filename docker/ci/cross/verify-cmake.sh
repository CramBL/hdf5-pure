#!/usr/bin/env bash
set -euo pipefail

expected=$1
test "$(printf '%s\n' 3.26 "$expected" | sort --version-sort | head -1)" = 3.26
for tool in cmake ctest cpack; do
    test "$(readlink -f "$(command -v "$tool")")" = "/opt/cmake/bin/$tool"
    test "$(readlink -f "/usr/local/bin/$tool")" = "/opt/cmake/bin/$tool"
    "$tool" --version
    test "$("$tool" --version | head -1)" = "$tool version $expected"
done
