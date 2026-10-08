# Architecture

## The pipeline

The workspace separates HDF5 semantic policy, byte representation, and file execution.
`hdf5-pure-core` owns shared types exposed through the `hdf5-pure` API.
`hdf5-pure-space` owns pure HDF5 file-space allocation and free-space policy.
`hdf5-pure-format` defines on-disk structures and their byte encodings.
`hdf5-pure` depends on those implementation crates and retains the file API, storage access, and
crash-safe publication.

| Package | Responsibility |
|---|---|
| `hdf5-pure-core` | Shared public definitions, including `FormatError`, datatype types, maximum extents, message identifiers, base and stored addresses, the superblock, and the File Space Info message |
| `hdf5-pure-space` | Semantic file-space allocation policy, reusable-space state, PAGE rules, and semantic free-space manager planning |
| `hdf5-pure-format` | Checked byte reads, checksums, message flags, object headers and the version 2 object header writer, the Link Info message parser, and parsers and encoders for the superblock, for datatype, dataspace, layout, fill-value, filter-pipeline, link, attribute info, and File Space Info messages, and for shared message references |
| `hdf5-pure-filter` | Filter algorithms and execution primitives |
| `hdf5-pure` | File access, object navigation, chunk lookup, editing, caching, free-space manager I/O and durability, the public file API, and MATLAB v7.3 support |

`hdf5-pure-space` depends only on `hdf5-pure-core`. It decides which ranges are reusable, which
ranges satisfy allocations, how PAGE transitions and returned space change semantic state, and
which free extents belong to each HDF5 free-space manager. It is permanently portable and uses
`no_std` with `alloc`. It does not represent file access provided by the operating system and owns no file access,
locking, synchronization, or durability protocol.

`hdf5-pure-format` parses bytes supplied by its caller and encodes HDF5 binary structures. It owns
the FSHD and FSSE byte representation, including structural parsing, field widths, and checksums.
It does not decide allocation policy.

`hdf5-pure` bridges those layers. It obtains bytes from storage, converts persisted manager data to
semantic space state, assigns manager slots and block addresses, passes manager blocks to
`hdf5-pure-format` for encoding, writes them, and owns synchronization, superblock publication,
reclamation, and crash ordering.

The core crate holds types whose public shape must remain compatible through the file crate's
re-exports. Each package supports `no_std` with `alloc` for its portable paths, and
`hdf5-pure-space` requires that portability for all of its file-space algorithms.

## API compatibility

| Package | Compatibility |
|---|---|
| `hdf5-pure` | The primary stable public facade |
| `hdf5-pure-core` | Stable: a small portable crate that owns the canonical shared public types |
| `hdf5-pure-format`, `hdf5-pure-filter`, and `hdf5-pure-space` | No independent compatibility guarantee at this time |

`hdf5-pure-format`, `hdf5-pure-filter`, and `hdf5-pure-space` export their implementation surfaces
to the other workspace crates through a hidden `__private` module each. Besides that module, the
format crate's root exports only `hdf5-pure-core` items, and the filter and space crate roots export
nothing.

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
