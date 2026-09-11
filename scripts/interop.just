set working-directory := ".."

uv := "uv run --project scripts/python --locked"

# The crosschecks live in the `hdf5-pure-crosscheck` package under crates/crosscheck/,
# the only one that links libhdf5.
crosscheck := "-p hdf5-pure-crosscheck"

# The whole matrix: every release, built under tmp/hdf5 on first use.
default: (test-hdf5 "1.8.23") (test-hdf5 "1.10.11" "serde,ndarray,__hdf5-1.10") (test-hdf5 "1.12.3" "serde,ndarray,__hdf5-1.12") (test-hdf5 "1.14.6" "serde,ndarray,__hdf5-1.14") (test-hdf5 "2.2.0" "serde,ndarray,__hdf5-2") test-matio

# Build libhdf5 VERSION and its tools under tmp/hdf5/VERSION.
hdf5-build VERSION:
    {{ uv }} hdf5-build {{ VERSION }}

# For eval "$(just interop::hdf5-env 1.8.23)" && just interop::test-external-hdf5.
# Shell exports that make a build link the libhdf5 under tmp/hdf5/VERSION.
hdf5-env VERSION:
    @{{ uv }} hdf5-build {{ VERSION }} --env

# FEATURES names the release series from 1.10 up, as in `serde,ndarray,__hdf5-1.10`.
# The crosschecks linking libhdf5 VERSION, built under tmp/hdf5 on first use.
# Passes `hdf5` and `hdf5_sys` to activate the C HDF5 bindings.
test-hdf5 VERSION FEATURES="serde,ndarray" *ARGS: (hdf5-build VERSION)
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just interop::hdf5-env {{ VERSION }})"
    cargo nextest run --locked {{ crosscheck }} --features hdf5,hdf5_sys,{{ FEATURES }} {{ ARGS }}

# Clippy over the crosschecks, linking libhdf5 VERSION.
lint-hdf5 VERSION FEATURES="serde,ndarray":
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just interop::hdf5-env {{ VERSION }})"
    cargo clippy --locked {{ crosscheck }} --all-targets --features hdf5,hdf5_sys,{{ FEATURES }} -- -D warnings

# The crosschecks linking the libhdf5 `hdf5-metno` bundles, built from source
# on first use. Needs no HDF5 on the host.
test-bundled *ARGS:
    cargo nextest run --locked {{ crosscheck }} --features __hdf5-bundled,serde,ndarray {{ ARGS }}

# The crosschecks linking the HDF5 installed under HDF5_DIR. Pass the feature
# naming its release series, as in `--features __hdf5-1.14`.
test-external-hdf5 *ARGS:
    cargo nextest run --locked {{ crosscheck }} --features hdf5,hdf5_sys,serde,ndarray {{ ARGS }}

test-matio *ARGS:
    cargo nextest run --locked --features "serde matio-crosscheck" --test serde_matio_crosscheck {{ ARGS }}

# The libhdf5 the crosschecks link when HDF5_DIR is unset.
hdf5-bundled-version:
    #!/usr/bin/env bash
    set -euo pipefail
    manifest=$(cargo metadata --locked --features hdf5-pure-crosscheck/__hdf5-bundled --format-version 1 | jq -r '.packages[] | select(.name == "hdf5-metno-src") | .manifest_path')
    awk '/^#define H5_VERS_(MAJOR|MINOR|RELEASE) / { v[++n] = $3 } END { print v[1] "." v[2] "." v[3] }' "$(dirname "$manifest")/ext/hdf5/src/H5public.h"
