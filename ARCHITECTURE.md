# Architecture

## The pipeline

The workspace separates HDF5 byte representations from operations on files.
`hdf5-pure-core` owns shared types exposed through the `hdf5-pure` API.
`hdf5-pure-format` defines extracted structures and byte encodings.
`hdf5-pure` depends on both and retains the file API.

| Package | Responsibility |
|---|---|
| `hdf5-pure-core` | Shared public definitions, including `FormatError`, datatype types, maximum extents, and message identifiers |
| `hdf5-pure-format` | Stored addresses, checked byte reads, checksums, and parsers and encoders for datatype, dataspace, layout, fill-value, and filter-pipeline messages |
| `hdf5-pure-filter` | Filter algorithms and execution primitives |
| `hdf5-pure` | Base-address conversion, file access, object navigation, chunk lookup, allocation, editing, caching, the public file API, and MATLAB v7.3 support |

The format crate parses bytes supplied by its caller and writes metadata
back into bytes. The file crate obtains those bytes from storage and uses the
parsed structures to decide what to read or write. The core crate holds
types whose public shape must remain compatible through the file crate's
re-exports. Each package supports `no_std` with `alloc` for its portable
paths.

## API compatibility

| Package | Compatibility |
|---|---|
| `hdf5-pure` | The primary stable public facade |
| `hdf5-pure-core` | Stable: a small portable crate that owns the canonical shared public types |
| `hdf5-pure-format`, `hdf5-pure-filter`, and later implementation crates such as `hdf5-pure-fs` | No independent compatibility guarantee at this time |

Public API exposed by `hdf5-pure` does not use types canonically owned by an
implementation crate. A shared type that needs a stable identity across crate
boundaries belongs in `hdf5-pure-core`, and the facade re-exports it directly
from there. Rust visibility, the `unnameable_types` lint and review keep to
this rule. The facade's re-exported types belong to its API even though
`hdf5-pure-core` owns their definitions.

`cargo-semver-checks` compares `hdf5-pure` and `hdf5-pure-core` with their
released versions under the profiles in `scripts/api-profiles.toml`. The
`default` and `full-public` profiles enable `std` in the core crate, and the
`no-std` profile builds it without `std`.

The first release containing the core crate compares against the last
release that defined those types in `hdf5-pure`. Three ownership-move lints are
downgraded for that baseline, and a rustdoc comparison checks the moved type
shapes and root paths. This bridge is temporary: after publishing the first
release containing `hdf5-pure-core`, the release maintainer removes it as
described in `RELEASES.md`. The checker rejects the bridge once the published
`hdf5-pure` baseline advances.
