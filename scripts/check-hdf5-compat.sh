#!/usr/bin/env bash
#
# Check this crate's HDF5 1.8 output format against an actual HDF5 1.8 library.
#
# Why this exists, and why it is not a CI job
# -------------------------------------------
# MATLAB linked HDF5 1.8.12 before R2021b — a different, older library than
# the 1.10.7 behind its own `h5read`/`h5disp`/`h5info` family. A version 3
# superblock is a 1.10 addition, so a file carrying one reads fine under
# `h5disp` and fails under `load`. That is the whole reason `mat::Options`
# defaults to `LibVer::V18`.
#
# Nothing in the test suite can catch a regression here. The `hdf5-metno`
# dev-dependency builds a current libhdf5, `h5py` links a current libhdf5, and
# every third-party MAT v7.3 reader (mat73, hdf5storage, pymatreader, MAT.jl,
# matio) delegates to whichever libhdf5 *it* links. All of them read a version 3
# superblock without complaint. The failure is one byte below everything they
# parse, so only an old library can see it.
#
# So this builds HDF5 1.8.23 — the last 1.8 release — and points its tools at
# both formats. It takes a couple of minutes and needs network access the first
# time, which is why it is a script you run rather than a gate that runs itself.
# Run it when changing anything about superblock or message versions.
#
#   ./scripts/check-hdf5-18.sh
#
# Measured on 1.8.23 when the 1.8 output format landed in 0.34.0:
#   superblock 3 (through 0.33.0) -> h5dump: "unable to open file", exit 1
#   superblock 2 (0.34.0 default) -> h5dump reads data, groups and attributes
#   h5repack round trip           -> preserves every attribute (it dropped all
#                                    of them before the Attribute Info fix)
#   committed (H5Tcommit) type    -> listed as a named datatype, and the dataset
#                                    and attribute typed through it both read
#
# And on 1.8.23 when the cell and string-class fixtures were added in 0.35.0:
#   cell of vectors, of structs,  -> 1.8 follows every object reference to its
#   and of a `None`                  `#refs#` target — datasets and groups alike
#                                    — and reads each one's data
#   H5PATH on an interned object  -> reads back as the object's own path, on a
#                                    group as well as a dataset
#   the canonical empty           -> carries no H5PATH, as in MATLAB's own files
#   MATLAB_fields                 -> the vlen string array decodes
#   empty value, `struct([])`     -> the dimension payload reads as `0 0`
#   MCOS subsystem                -> the opaque handle, the templates' reference
#                                    arrays and the alias all read
#   dense attributes              -> 12 on one object move into a fractal heap
#                                    and a v2 B-tree; 1.8 reads them all, and
#                                    h5repack carries them across
#   compound (complex) dataset    -> both members read, in order
#   rank-3 dataspace              -> reads
#   a group of 64 links           -> every member is listed and reads
#
# Those two fixtures are what put object references, a `#refs#` group, a vlen
# attribute and an MCOS subsystem in front of an old library at all; before them
# this was scalars, vectors, a struct, a string and an empty. Nor had 1.8 ever
# read the builder-backed emitter's output: `to_file` goes through the walker,
# and the builder's only other fixture is the 1.10 file, which 1.8 refuses by
# design. `mat_string_v18.mat` is written through the builder for that reason as
# much as for the subsystem.
#
# One thing that measurement showed about this file's shape: 1.8's h5dump gives
# up on the *whole file* when a committed datatype does not decode, so the plain
# checks above fail alongside the committed ones. The committed fixture is what
# makes any of them sensitive to it — before it there was no named type in the
# file to get wrong — and the committed checks say which thing broke rather than
# leaving "the file will not open" to be interpreted. The exception is a type
# that is written and referenced but never linked: the file then reads correctly
# end to end and 1.8 lists the type as `/#324`, its address standing in for the
# name it lost. Only `check_named_type` sees that one.
#
# The checks below compare dataset and attribute *values*, not just exit status.
# A file whose headers all decode while its data resolves to the wrong offset
# opens cleanly and lists every object, so `h5dump -n` alone would report it as
# passing — which is the defect class this format work fixed on the read side.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO/tmp/hdf5-compat-check"   # under the gitignored scratch dir
PREFIX="$WORK/install"
SRC_VERSION="1_8_23"
FIXTURES="$WORK/fixtures"

# With --prefix, use an HDF5 already installed there (conda, apt, brew) instead
# of building 1.8.23. That is how CI covers a range of versions.
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a directory}"; SRC_VERSION=""; shift 2 ;;
    -h|--help) echo "usage: $0 [--prefix <hdf5-install-dir>]"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$WORK"

# ---------------------------------------------------------------- build 1.8.23

if [ -n "$SRC_VERSION" ] && [ ! -x "$PREFIX/bin/h5dump" ]; then
  if [ ! -d "$WORK/hdf5-hdf5-$SRC_VERSION" ]; then
    echo "==> downloading HDF5 $SRC_VERSION"
    curl -fsSL --max-time 300 -o "$WORK/hdf5.tar.gz" \
      "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-$SRC_VERSION.tar.gz"
    tar xzf "$WORK/hdf5.tar.gz" -C "$WORK"
  fi

  cd "$WORK/hdf5-hdf5-$SRC_VERSION"

  # 1.8.23 predates arm64 macOS, and its bundled config.sub rejects that host
  # outright ("machine `aarch64-apple' not recognized"). Refreshing the two
  # config scripts from upstream is the standard remedy, and asking them about
  # this host first keeps every host they already know off the network.
  host="$(bin/config.guess 2>/dev/null || true)"
  if [ ! -f bin/config.sub.orig ] &&
     { [ -z "$host" ] || ! bin/config.sub "$host" >/dev/null 2>&1; }; then
    echo "==> refreshing config.guess / config.sub for this host"
    cp bin/config.sub bin/config.sub.orig
    cp bin/config.guess bin/config.guess.orig
    base='https://git.savannah.gnu.org/cgit/config.git/plain/'
    curl -fsSL --max-time 120 -o bin/config.sub "${base}config.sub"
    curl -fsSL --max-time 120 -o bin/config.guess "${base}config.guess"
    chmod +x bin/config.sub bin/config.guess
  fi

  # 1.8.23 predates GCC 14, which turned several long-standing warnings into
  # errors by default; the tools do not build under it otherwise.
  RELAX="-std=gnu17 -Wno-error=incompatible-pointer-types"
  RELAX="$RELAX -Wno-error=implicit-function-declaration -Wno-error=int-conversion"

  echo "==> configuring (log: $WORK/configure.log)"
  ./configure --prefix="$PREFIX" \
    --disable-fortran --disable-cxx --disable-hl --disable-shared \
    --enable-tools CFLAGS="${CFLAGS:-} $RELAX" > "$WORK/configure.log" 2>&1 ||
    { echo "configure failed:"; tail -30 "$WORK/configure.log"; exit 1; }

  echo "==> building (log: $WORK/make.log)"
  make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" \
    > "$WORK/make.log" 2>&1 ||
    { echo "make failed:"; tail -40 "$WORK/make.log"; exit 1; }

  make install > "$WORK/install.log" 2>&1 ||
    { echo "install failed:"; tail -20 "$WORK/install.log"; exit 1; }
fi

H5DUMP="$PREFIX/bin/h5dump"
H5REPACK="$PREFIX/bin/h5repack"
[ -x "$H5DUMP" ] || { echo "no h5dump at $H5DUMP" >&2; exit 2; }
echo "==> using $("$H5DUMP" --version)"

# The library's own version decides what the 1.10 fixtures are expected to do:
# 1.8 cannot open a version 3 superblock, every later release reads it.
H5_VERSION="$("$H5DUMP" --version | awk '{print $NF}')"
H5_MAJOR="$(printf '%s' "$H5_VERSION" | cut -d. -f1)"
H5_MINOR="$(printf '%s' "$H5_VERSION" | cut -d. -f2)"
if [ "$H5_MAJOR" -gt 1 ] || [ "$H5_MINOR" -ge 10 ]; then
  V110_EXPECT=open
else
  V110_EXPECT=refuse
fi

# ------------------------------------------------------------------- fixtures

echo "==> writing fixtures"
rm -rf "$FIXTURES"
cd "$REPO"
cargo run --quiet --example libver_fixtures --features serde -- "$FIXTURES"

# --------------------------------------------------------------------- checks

failures=0
check() {  # check <description> <expected: open|refuse> <file>
  local desc="$1" expect="$2" file="$3" out rc
  out="$("$H5DUMP" -n "$file" 2>&1)" && rc=0 || rc=$?
  if [ "$expect" = open ] && [ "$rc" -eq 0 ]; then
    echo "  ok   $desc — lists its objects"
  elif [ "$expect" = refuse ] && [ "$rc" -ne 0 ]; then
    echo "  ok   $desc — refused (${out##*$'\n'})"
  else
    echo "  FAIL $desc — expected to $expect, exit $rc"
    echo "$out" | sed 's/^/       /'
    failures=$((failures + 1))
  fi
}

# The values in h5dump's first `DATA` block, as one space-separated line.
#
# Taking only the first block drops the dataset's own attributes, which
# `h5dump -d` prints after it. Done this way rather than with `-A 0` because
# that spelling is a 1.10 addition: 1.8's `-A` takes no argument, so passing one
# there dumps the wrong thing.
#
# The block ends where its braces balance rather than where the next `DATA`
# begins, and every line inside it counts rather than only the ones carrying an
# `(index):` prefix. A compound — which is how a complex array is stored — puts
# each element in a brace group of its own across several lines, so an extractor
# reading only the indexed lines sees `{` and nothing else. The index prefix,
# braces and commas are dropped, which leaves the same text for every other
# shape as taking the indexed lines did.
#
# A failing h5dump yields no values rather than killing the run: under `set -e`
# and `pipefail` its exit status would abort the script at the first bad file,
# and every check after it would go unanswered. This is run by hand and rarely,
# so one run has to report every check, not just the ones before the first
# failure.
dump_values() {  # dump_values <h5dump args...>
  { "$H5DUMP" "$@" 2>/dev/null || true; } |
    awk '
      /^[[:space:]]*DATA[[:space:]]*[{]/ && !started { started = 1; depth = 1; next }
      started {
        line = $0
        sub(/^[[:space:]]*[(][0-9,]*[)]:[[:space:]]*/, "", line)
        opened = gsub(/[{]/, "{", line); closed = gsub(/[}]/, "}", line)
        depth += opened - closed
        gsub(/[{},]/, " ", line)
        print line
        if (depth <= 0) exit
      }' |
    tr -s '[:space:]' ' ' |
    sed -e 's/^ //' -e 's/ $//'
}

# Compare what 1.8 *read* against what the fixture holds.
#
# `h5dump -n` lists object names and nothing else, so on its own it cannot see a
# file whose headers all decode while its data resolves to the wrong offset —
# the base-address defect class, among others. Every check below therefore names
# the values it expects rather than settling for exit 0.
check_data() {  # check_data <description> <expected values> <h5dump args...>
  local desc="$1" want="$2"; shift 2
  local got
  got="$(dump_values "$@")"
  if [ "$got" = "$want" ]; then
    echo "  ok   $desc — read [$got]"
  else
    echo "  FAIL $desc — read [$got], expected [$want]"
    failures=$((failures + 1))
  fi
}

# `h5dump -n` lists each object with its kind first, so a committed datatype
# appears as `datatype /reading_t`. It is listed as one only if the object header
# decoded as a named type — which `H5O__dtype_isa` decides from the messages
# present — so this sees a header that opens but is not what it claims.
check_named_type() {  # check_named_type <description> <name> <file>
  local desc="$1" name="$2" file="$3" out
  # As in `dump_values`: a failing h5dump must fail this check, not the run.
  out="$("$H5DUMP" -n "$file" 2>&1)" || true
  if printf '%s\n' "$out" | grep -qE "datatype[[:space:]]+$name\$"; then
    echo "  ok   $desc — lists $name as a named datatype"
  else
    echo "  FAIL $desc — $name is not listed as a named datatype"
    printf '%s\n' "$out" | sed 's/^/       /'
    failures=$((failures + 1))
  fi
}

# An object-reference dataset is the one shape whose data is a pointer. h5dump
# follows each reference and names the target it found before printing its
# values, so collecting those paths in order shows the references *resolving* —
# a dataset that opens with references that dangle would still dump, with no
# target lines.
#
# Two spellings of that line: 1.8 writes `DATASET <addr> "/path"`, 1.14 and
# later write `DATASET "<file>/path"`. Both reduce to the path by dropping an
# optional address and an optional leading copy of the file name.
check_ref_targets() {  # check_ref_targets <description> <expected paths> <file> <dataset>
  local desc="$1" want="$2" file="$3" ds="$4" got
  # `GROUP` as well as `DATASET`: a cell of structs interns a group per element,
  # and h5dump labels a resolved reference by the kind of object it found.
  got="$({ "$H5DUMP" -d "$ds" "$file" 2>/dev/null || true; } |
    sed -nE 's/^[[:space:]]*(DATASET|GROUP) ([0-9]+ )?"(.*)"[[:space:]]*$/\3/p' |
    sed -e "s|^${file}||" |
    tr '\n' ' ' | sed -e 's/ $//')"
  if [ "$got" = "$want" ]; then
    echo "  ok   $desc — resolved [$got]"
  else
    echo "  FAIL $desc — resolved [$got], expected [$want]"
    failures=$((failures + 1))
  fi
}

# An attribute this crate deliberately does *not* write. h5dump exits 1 for a
# missing attribute and for a missing object alike, so every use of this pairs
# with a value check on the same object: absence has to mean the attribute is
# gone, not that the walk went somewhere that does not exist.
check_absent_attr() {  # check_absent_attr <description> <file> <object/attribute>
  local desc="$1" file="$2" attr="$3"
  if "$H5DUMP" -a "$attr" "$file" > /dev/null 2>&1; then
    echo "  FAIL $desc — the attribute is present"
    failures=$((failures + 1))
  else
    echo "  ok   $desc — absent"
  fi
}

# How many lines of a dump match a pattern — the count of objects or attributes
# 1.8 found, where naming each one would say no more than the total does.
check_count() {  # check_count <description> <expected> <pattern> <h5dump args...>
  local desc="$1" want="$2" pattern="$3"; shift 3
  local got
  got="$({ "$H5DUMP" "$@" 2>/dev/null || true; } | grep -c "$pattern" || true)"
  if [ "$got" = "$want" ]; then
    echo "  ok   $desc — $got"
  else
    echo "  FAIL $desc — $got, expected $want"
    failures=$((failures + 1))
  fi
}

# An on-disk structure named by its four-byte signature, counted in the raw
# file. This says the fixture *reaches* a structure before anything reads
# through it: dense attribute storage is a threshold behavior, and a fixture
# that drifted back under the threshold would still pass every check below
# while testing the object header it was written to avoid.
check_signature() {  # check_signature <description> <expected count> <signature> <file>
  local desc="$1" want="$2" sig="$3" file="$4" got
  got="$(LC_ALL=C grep -a -o "$sig" "$file" 2>/dev/null | wc -l | tr -d ' ' || true)"
  if [ "$got" = "$want" ]; then
    echo "  ok   $desc — $got $sig block(s)"
  else
    echo "  FAIL $desc — $got $sig block(s), expected $want"
    failures=$((failures + 1))
  fi
}

echo "==> the 1.10 fixtures must $V110_EXPECT under $H5_VERSION"
check "mat_v110.mat"  "$V110_EXPECT" "$FIXTURES/mat_v110.mat"
check "plain_v110.h5" "$V110_EXPECT" "$FIXTURES/plain_v110.h5"
if [ "$V110_EXPECT" = open ]; then
  check_data "plain_v110.h5 /values"    "1 2 3" -d /values    "$FIXTURES/plain_v110.h5"
  check_data "plain_v110.h5 /grp/inner" "7 8"   -d /grp/inner "$FIXTURES/plain_v110.h5"
  check_data "mat_v110.mat /values"     "1 2 3" -d /values    "$FIXTURES/mat_v110.mat"
fi

echo "==> the 1.8 format must be readable by $H5_VERSION"
check "mat_v18.mat"        open "$FIXTURES/mat_v18.mat"
check "mat_string_v18.mat" open "$FIXTURES/mat_string_v18.mat"
check "plain_v18.h5"       open "$FIXTURES/plain_v18.h5"

# Both fixtures are written by `examples/libver_fixtures.rs`; these are the
# values it puts in them. Reading the *data* is the half `-n` cannot do.
echo "==> it must read the right bytes, not merely open the file"
check_data "plain_v18.h5 /values"     "1 2 3" -d /values     "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /grp/inner"  "7 8"   -d /grp/inner  "$FIXTURES/plain_v18.h5"
check_data "mat_v18.mat /values"      "1 2 3" -d /values     "$FIXTURES/mat_v18.mat"
check_data "mat_v18.mat /nested/count" "7"    -d /nested/count "$FIXTURES/mat_v18.mat"
# An empty value's payload *is* its dimension vector, so this reads the `0x0`
# rule itself rather than a dataset that happens to be empty.
check_data "mat_v18.mat /empty"       "0 0"   -d /empty       "$FIXTURES/mat_v18.mat"

# A cell array is the only shape that interns objects under `#refs#`: the parent
# dataset holds object references instead of data, and each interned object
# carries an `H5PATH` attribute alongside its `MATLAB_class`. Neither had ever
# been put in front of an old library — the fixture had no cell in it — so a
# regression in either was invisible here.
echo "==> it must resolve a cell array's interned objects"
MAT="$FIXTURES/mat_v18.mat"
# `demo()` interns in field order: ragged's two vectors, records' two structs,
# then optional's scalar and its `struct([])`.
r() { printf '/#refs#/ref_%016x' "$1"; }
check_ref_targets "mat_v18.mat /ragged references"   "$(r 0) $(r 1)" "$MAT" /ragged
check_ref_targets "mat_v18.mat /records references"  "$(r 2) $(r 3)" "$MAT" /records
check_ref_targets "mat_v18.mat /optional references" "$(r 4) $(r 5)" "$MAT" /optional

# Each interned object's own data and the H5PATH it carries. A cell of vectors
# interns datasets; a cell of structs interns groups, which is where the
# attribute lands on a group rather than a dataset.
check_data "mat_v18.mat $(r 0)"        "1"   -d "$(r 0)" "$MAT"
check_data "mat_v18.mat $(r 1)"        "2 3" -d "$(r 1)" "$MAT"
check_data "mat_v18.mat $(r 2)/count"  "11"  -d "$(r 2)/count" "$MAT"
check_data "mat_v18.mat $(r 3)/count"  "13"  -d "$(r 3)/count" "$MAT"
check_data "mat_v18.mat $(r 4)"        "1.5" -d "$(r 4)" "$MAT"
# The `struct([])` a `None` slot lowers to: an empty marker under `#refs#`,
# whose payload is its own dimension vector.
check_data "mat_v18.mat $(r 5) struct([])" "0 0" -d "$(r 5)" "$MAT"
for ref in 0 1 2 3 4 5; do
  check_data "mat_v18.mat $(r "$ref") H5PATH" "\"$(r "$ref")\"" \
    -a "$(r "$ref")/H5PATH" "$MAT"
done

# `MATLAB_fields` is the only variable-length string array this crate writes —
# a vlen of one-character strings, held in the global heap rather than inline.
# Nothing else here asks an old library to decode that shape.
check_data "mat_v18.mat $(r 2) MATLAB_fields" \
  '("c" "o" "u" "n" "t") ("f" "l" "a" "g")' \
  -a "$(r 2)/MATLAB_fields" "$MAT"

# The `string` class builds an MCOS subsystem: a `#subsystem#/MCOS` blob, a
# FileWrapper metadata ref, two reference-array templates, an alias, and the
# canonical empty. This is also the only 1.8 file here written by the
# builder-backed emitter — `to_file` uses the walker, and the builder's other
# output is the 1.10 file 1.8 refuses.
echo "==> it must read the MCOS subsystem the string class builds"
STR="$FIXTURES/mat_string_v18.mat"
# The opaque handle MATLAB resolves through the subsystem: 0xDD000000, then the
# class and object ids.
check_data "mat_string_v18.mat /label" "3707764736 2 1 1 1 1" -d /label "$STR"
check_data "mat_string_v18.mat MCOS alias"    "0 0" -d "$(r 12)" "$STR"
check_ref_targets "mat_string_v18.mat template B" "$(r 13) $(r 14)" "$STR" "$(r 15)"
# The canonical empty is the one object under `#refs#` MATLAB leaves without an
# H5PATH, so it is the one we leave without one too. The class check is what
# keeps the absence check honest: it names the object as the one intended.
check_data "mat_string_v18.mat canonical empty" '"canonical empty"' \
  -a "$(r 8)/MATLAB_class" "$STR"
check_absent_attr "mat_string_v18.mat canonical empty H5PATH" "$STR" "$(r 8)/H5PATH"

# A complex array is a `{real, imag}` compound, the only compound datatype this
# crate writes. MATLAB decides real from complex by this shape, so an old
# library reading the members in the wrong order — or not at all — is a
# difference that reaches the workspace.
echo "==> it must read a compound (complex) dataset"
check_data "mat_v18.mat /signal" "1 -2 0.5 0.25" -d /signal "$MAT"
check_count "mat_v18.mat /signal members" 2 'H5T_IEEE_F64LE "' -H -d /signal "$MAT"

# Nine or more attributes on one object move out of the object header into a
# fractal heap indexed by a version-2 B-tree, reached through an Attribute Info
# message — 1.8 additions all, and the storage this crate's own history has the
# most defects in. The signature counts come first: below the threshold every
# check under them would pass while reading the header instead.
echo "==> it must read attributes out of a fractal heap, not just a header"
check_signature "plain_v18.h5 dense attribute heap"  1 FRHP "$FIXTURES/plain_v18.h5"
check_signature "plain_v18.h5 dense attribute index" 1 BTHD "$FIXTURES/plain_v18.h5"
check_count "plain_v18.h5 /dense_attrs attributes" 12 'ATTRIBUTE' -A -d /dense_attrs "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /dense_attrs"     "1 2" -d /dense_attrs     "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /dense_attrs a00" "0"   -a /dense_attrs/a00 "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /dense_attrs a11" "11"  -a /dense_attrs/a11 "$FIXTURES/plain_v18.h5"

# Rank 3, and a group holding more links than one object header chunk. Where the
# C library would switch the group to dense link storage this crate keeps
# writing link messages, so the question an old library answers is whether the
# header it grows is one that still walks.
echo "==> it must read a rank-3 dataspace and a group of many links"
check_data "plain_v18.h5 /cube" \
  "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23" \
  -d /cube "$FIXTURES/plain_v18.h5"
check_count "plain_v18.h5 /wide members" 64 '/wide/m' -n "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /wide/m000" "0"  -d /wide/m000 "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /wide/m063" "63" -d /wide/m063 "$FIXTURES/plain_v18.h5"

# And the attribute values, not just the count the repack loop below compares.
echo "==> it must read attribute values on all three kinds of object"
check_data "plain_v18.h5 /values units" '"m/s"' -a /values/units "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 / root_attr"   '"r"'   -a /root_attr    "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /grp tag"      "7"     -a /grp/tag      "$FIXTURES/plain_v18.h5"

# A committed (`H5Tcommit`) datatype is an object of its own, and everything
# using it stores a reference to that object's header in place of an encoding.
# An old library that cannot follow the reference does not fail loudly: it reads
# the reference bytes as a datatype and reports the wrong element type, or drops
# the attribute list the reference hangs off. So name the object, then read
# through it — the dataset whose element type it is, and the attribute whose
# datatype it is.
echo "==> it must resolve a committed datatype, not merely list it"
check_named_type "plain_v18.h5" /reading_t "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /typed"          "3 1 4" -d /typed          "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /typed baseline" "9"     -a /typed/baseline "$FIXTURES/plain_v18.h5"

# The attribute-count fix, against the toolchain that lost the attributes: a
# header that declares no attributes makes h5repack copy the object without
# them, silently. Counting `ATTRIBUTE` blocks in the dump is enough to see it.
echo "==> an h5repack round trip must preserve every attribute"
for f in mat_v18.mat mat_string_v18.mat plain_v18.h5; do
  before=$("$H5DUMP" -A "$FIXTURES/$f" 2>/dev/null | grep -c 'ATTRIBUTE' || true)
  rm -f "$FIXTURES/repacked-$f"
  if ! "$H5REPACK" "$FIXTURES/$f" "$FIXTURES/repacked-$f" > /dev/null 2>&1; then
    echo "  FAIL $f — h5repack itself failed"
    failures=$((failures + 1))
    continue
  fi
  after=$("$H5DUMP" -A "$FIXTURES/repacked-$f" 2>/dev/null | grep -c 'ATTRIBUTE' || true)
  if [ "$before" -gt 0 ] && [ "$before" -eq "$after" ]; then
    echo "  ok   $f — $before attributes in, $after out"
  else
    echo "  FAIL $f — $before attributes in, $after out"
    failures=$((failures + 1))
  fi
done

echo
if [ "$failures" -eq 0 ]; then
  echo "all checks passed against HDF5 $H5_VERSION"
else
  echo "$failures check(s) failed"
  exit 1
fi
