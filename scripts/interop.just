set lazy

import "constants.just"
set working-directory := ".."
set shell := ["bash", "-euo", "pipefail", "-c"]

uv := "uv run --project scripts/python --locked"

# The whole matrix: every release, built under tmp/hdf5 on first use.
default: test-hdf5-all test-matio

# Build libhdf5 VERSION, a series such as 1.14 or its release, and its tools under tmp/hdf5.
hdf5-build VERSION:
    {{ uv }} hdf5-build {{ VERSION }}

# For eval "$(just interop::hdf5-env 1.8.23)" && just interop::test-external-hdf5.
# Shell exports that make a build link the libhdf5 under tmp/hdf5/VERSION.
hdf5-env VERSION:
    @{{ uv }} hdf5-build {{ VERSION }} --env

# The crosscheck feature for the release series of libhdf5 VERSION, empty for 1.8.
hdf5-release-feature VERSION:
    @{{ uv }} hdf5-build {{ VERSION }} --feature

# Every release scripts/hdf5-releases.toml lists, oldest first, or with `--json` as a JSON array.
hdf5-versions *ARGS:
    @{{ uv }} hdf5-build --list {{ ARGS }}

# Test every release scripts/hdf5-releases.toml lists with `FEATURES`.
test-hdf5-all FEATURES="serde,ndarray":
    #!/usr/bin/env bash
    set -euo pipefail
    for version in $({{ uv }} hdf5-build --list); do
        {{ quote(just_executable()) }} interop::test-hdf5 "$version" "{{ FEATURES }}"
    done

# Run cross tests on libhdf5 `VERSION` with `FEATURES` and `ARGS`. Built under tmp/hdf5 on first use.
test-hdf5 VERSION FEATURES="serde,ndarray" *ARGS: (hdf5-build VERSION)
    release_feature="$({{ quote(just_executable()) }} interop::hdf5-release-feature {{ VERSION }})" && \
    eval "$({{ quote(just_executable()) }} interop::hdf5-env {{ VERSION }})" && \
    cargo nextest run --locked -p {{ CROSSCHECK }} --features "hdf5,{{ FEATURES }}${release_feature:+,$release_feature}" {{ ARGS }}

# Clippy over the crosschecks, linking libhdf5 VERSION.
lint-hdf5 VERSION FEATURES="serde,ndarray":
    release_feature="$({{ quote(just_executable()) }} interop::hdf5-release-feature {{ VERSION }})" && \
    eval "$({{ quote(just_executable()) }} interop::hdf5-env {{ VERSION }})" && \
    cargo clippy --locked -p {{ CROSSCHECK }} --all-targets --features "hdf5,{{ FEATURES }}${release_feature:+,$release_feature}" -- -D warnings

# The crosschecks linking the libhdf5 `hdf5-metno` bundles, built from source
# on first use. Needs no HDF5 on the host.
test-bundled *ARGS:
    cargo nextest run --locked -p {{ CROSSCHECK }} --features __hdf5-bundled,serde,ndarray {{ ARGS }}

# The crosschecks linking the HDF5 installed under HDF5_DIR. Pass the feature
# naming its release series, as in `--features __hdf5-1.14`.
test-external-hdf5 *ARGS:
    cargo nextest run --locked -p {{ CROSSCHECK }} --features hdf5,serde,ndarray {{ ARGS }}

test-matio *ARGS:
    cargo nextest run --locked -p {{ CROSSCHECK }} --features serde,__matio {{ ARGS }}

# The libhdf5 the crosschecks link when HDF5_DIR is unset.
hdf5-bundled-version:
    #!/usr/bin/env bash
    set -euo pipefail
    manifest=$(cargo metadata --locked --features {{ CROSSCHECK }}/__hdf5-bundled --format-version 1 | jq -r '.packages[] | select(.name == "hdf5-metno-src") | .manifest_path')
    awk '/^#define H5_VERS_(MAJOR|MINOR|RELEASE) / { v[++n] = $3 } END { print v[1] "." v[2] "." v[3] }' "$(dirname "$manifest")/ext/hdf5/src/H5public.h"
