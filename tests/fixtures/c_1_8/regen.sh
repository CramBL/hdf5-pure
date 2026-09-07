#!/usr/bin/env bash
#
# Regenerate the HDF5 1.8-era read fixtures in this directory.
#
# Needs the HDF5 1.8.23 install that `just hdf5-build` builds; run that
# first if it is not there. The fixtures are committed, so this is a developer
# tool rather than part of any test run — nothing in `cargo test` invokes it.
#
#   just hdf5-build                     # once, builds 1.8.23
#   ./tests/fixtures/c_1_8/regen.sh
#
# Rerun it only to change what the fixtures contain. The committed files are the
# ground truth the tests read; regenerating them for no reason replaces measured
# bytes with freshly measured bytes and reviews as a diff nobody can check.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
H5CC="$REPO/tmp/hdf5-compat-check/install/bin/h5cc"

if [ ! -x "$H5CC" ]; then
  echo "no HDF5 1.8.23 install at $H5CC" >&2
  echo "run `just hdf5-build` first, it builds one" >&2
  exit 1
fi

# Refuse to run against anything but 1.8. `H5F_LIBVER_LATEST` in regen.c means
# "the 1.8 format" only here; against 1.10 or newer the same source writes a
# version 3 superblock into v2_superblock.h5 and a version 2 one into
# v1_superblock.h5, and both filenames become lies.
VERSION="$("$REPO/tmp/hdf5-compat-check/install/bin/h5dump" --version | awk '{print $NF}')"
case "$VERSION" in
  1.8.*) echo "==> using h5dump $VERSION" ;;
  *)
    echo "refusing to regenerate with HDF5 $VERSION: these fixtures are 1.8 output" >&2
    echo "run `just hdf5-build`, which builds 1.8.23" >&2
    exit 1
    ;;
esac

cd "$HERE"
"$H5CC" -O2 -o regen regen.c
./regen
rm -f regen regen.o

# And check what was actually written. Printing the version without comparing it
# is how a generator quietly produces the wrong bytes: every fixture here is
# named for its superblock version, so that is the thing to assert.
status=0
check_version() {  # check_version <file> <expected superblock version>
  local file="$1" want="$2" got
  got="$(python3 -c "
import sys
b = open(sys.argv[1], 'rb').read()
i = b.find(b'\x89HDF\r\n\x1a\n')
print(b[i + 8] if i >= 0 else 'no signature')" "$file")"
  if [ "$got" = "$want" ]; then
    echo "  ok   $file superblock $got"
  else
    echo "  FAIL $file superblock $got, expected $want" >&2
    status=1
  fi
}

check_version v1_superblock.h5 1
check_version v2_superblock.h5 2

if [ "$status" -ne 0 ]; then
  echo "fixtures not regenerated as expected; the working copies are wrong" >&2
  exit 1
fi
