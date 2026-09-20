set working-directory := ".."

uv := "uv run --project scripts/python --locked"

# The crosschecks live in the `hdf5-pure-crosscheck` package under crates/crosscheck/,
# the only one that links libhdf5.
crosscheck := "-p hdf5-pure-crosscheck"

# The whole matrix: every release, built under tmp/hdf5 on first use.
default: test-hdf5-all test-matio

# Build libhdf5 VERSION and its tools under tmp/hdf5/VERSION.
hdf5-build VERSION:
    {{ uv }} hdf5-build {{ if HDF5_VERSION_MAP =~ (".* " + VERSION + ":.*") { replace_regex(HDF5_VERSION_MAP, ".* " + VERSION + ":([^ ]+).*", "$1") } else { VERSION } }}

# For eval "$(just interop::hdf5-env 1.8.23)" && just interop::test-external-hdf5.
# Shell exports that make a build link the libhdf5 under tmp/hdf5/VERSION.
hdf5-env VERSION:
    @{{ uv }} hdf5-build {{ if HDF5_VERSION_MAP =~ (".* " + VERSION + ":.*") { replace_regex(HDF5_VERSION_MAP, ".* " + VERSION + ":([^ ]+).*", "$1") } else { VERSION } }} --env

# The crosscheck feature for the release series of libhdf5 VERSION, with the
# comma that appends it to a feature list. The 1.8 series has no such feature.
hdf5-release-feature VERSION:
    @echo '{{ if VERSION =~ "^1[.]8([.]|$)" { "" } else if VERSION =~ "^1[.]10([.]|$)" { ",__hdf5-1.10" } else if VERSION =~ "^1[.]12([.]|$)" { ",__hdf5-1.12" } else if VERSION =~ "^1[.]14([.]|$)" { ",__hdf5-1.14" } else if VERSION =~ "^2([.]|$)" { ",__hdf5-2" } else { error("no crosscheck feature for HDF5 " + VERSION + ": add its release series to hdf5-release-feature") } }}'

# Test all supported HDF5 versions with `FEATURES`
test-hdf5-all FEATURES="serde,ndarray": (test-hdf5 "1.8" FEATURES) (test-hdf5 "1.10" FEATURES) (test-hdf5 "1.12" FEATURES) (test-hdf5 "1.14" FEATURES) (test-hdf5 "2" FEATURES)

# Run cross tests on libhdf5 `VERSION` with `FEATURES` and `ARGS`. Built under tmp/hdf5 on first use.
test-hdf5 VERSION FEATURES="serde,ndarray" *ARGS: (hdf5-build VERSION)
    release_feature="$(just interop::hdf5-release-feature {{ VERSION }})" && \
    eval "$(just interop::hdf5-env {{ VERSION }})" && \
    cargo nextest run --locked {{ crosscheck }} --features "hdf5,{{ FEATURES }}$release_feature" {{ ARGS }}

# Clippy over the crosschecks, linking libhdf5 VERSION.
lint-hdf5 VERSION FEATURES="serde,ndarray":
    release_feature="$(just interop::hdf5-release-feature {{ VERSION }})" && \
    eval "$(just interop::hdf5-env {{ VERSION }})" && \
    cargo clippy --locked {{ crosscheck }} --all-targets --features "hdf5,{{ FEATURES }}$release_feature" -- -D warnings

# The crosschecks linking the libhdf5 `hdf5-metno` bundles, built from source
# on first use. Needs no HDF5 on the host.
test-bundled *ARGS:
    cargo nextest run --locked {{ crosscheck }} --features __hdf5-bundled,serde,ndarray {{ ARGS }}

# The crosschecks linking the HDF5 installed under HDF5_DIR. Pass the feature
# naming its release series, as in `--features __hdf5-1.14`.
test-external-hdf5 *ARGS:
    cargo nextest run --locked {{ crosscheck }} --features hdf5,serde,ndarray {{ ARGS }}

test-matio *ARGS:
    cargo nextest run --locked {{ crosscheck }} --features serde,__matio {{ ARGS }}

# The libhdf5 the crosschecks link when HDF5_DIR is unset.
hdf5-bundled-version:
    #!/usr/bin/env bash
    set -euo pipefail
    manifest=$(cargo metadata --locked --features hdf5-pure-crosscheck/__hdf5-bundled --format-version 1 | jq -r '.packages[] | select(.name == "hdf5-metno-src") | .manifest_path')
    awk '/^#define H5_VERS_(MAJOR|MINOR|RELEASE) / { v[++n] = $3 } END { print v[1] "." v[2] "." v[3] }' "$(dirname "$manifest")/ext/hdf5/src/H5public.h"

HDF5_1_8 := "1.8.23"
HDF5_1_10 := "1.10.11"
HDF5_1_12 := "1.12.3"
HDF5_1_14 := "1.14.6"
HDF5_2 := "2.2.0"
HDF5_VERSION_MAP := " 1.8:" + HDF5_1_8 + " 1.10:" + HDF5_1_10 + " 1.12:" + HDF5_1_12 + " 1.14:" + HDF5_1_14 + " 2:" + HDF5_2 + " "
