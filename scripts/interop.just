set working-directory := ".."

uv := "uv run --project scripts/python --locked"

# The whole matrix: every release, built under tmp/hdf5 on first use.
default: (test-hdf5 "1.8.23") (test-hdf5 "1.10.11" "serde,__hdf5-1.10") (test-hdf5 "1.12.3" "serde,__hdf5-1.12") (test-hdf5 "1.14.6" "serde,__hdf5-1.14") (test-hdf5 "2.2.0" "serde,__hdf5-2") test-matio

# Build libhdf5 VERSION and its tools under tmp/hdf5/VERSION.
hdf5-build VERSION:
    {{ uv }} hdf5-build {{ VERSION }}

# For eval "$(just interop::hdf5-env 1.8.23)" && just interop::test-external-hdf5.
# Shell exports that make a build link the libhdf5 under tmp/hdf5/VERSION.
hdf5-env VERSION:
    @{{ uv }} hdf5-build {{ VERSION }} --env

# FEATURES names the release series from 1.10 up, as in `serde,__hdf5-1.10`.
# The test suite linking libhdf5 VERSION, built under tmp/hdf5 on first use.
test-hdf5 VERSION FEATURES="serde" *ARGS: (hdf5-build VERSION)
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just interop::hdf5-env {{ VERSION }})"
    cargo nextest run --locked --features {{ FEATURES }} {{ ARGS }}

# The test suite linking the HDF5 installed under HDF5_DIR.
test-external-hdf5 *ARGS:
    cargo nextest run --locked {{ ARGS }}

test-matio *ARGS:
    cargo nextest run --locked --features __hdf5-bundled --features "serde matio-crosscheck" --test serde_matio_crosscheck {{ ARGS }}

# The libhdf5 the crosscheck tests link when HDF5_DIR is unset.
hdf5-bundled-version:
    #!/usr/bin/env bash
    set -euo pipefail
    manifest=$(cargo metadata --locked --format-version 1 | jq -r '.packages[] | select(.name == "hdf5-metno-src") | .manifest_path')
    awk '/^#define H5_VERS_(MAJOR|MINOR|RELEASE) / { v[++n] = $3 } END { print v[1] "." v[2] "." v[3] }' "$(dirname "$manifest")/ext/hdf5/src/H5public.h"
