# Documentation style

This guide covers the comments and the documentation: doc comments, the comments beside the code, and the prose in this repository. [CODE_STYLE.md](CODE_STYLE.md) covers the code.

The Rust standard library is the model. Doc comments follow [Appendix A of Rust RFC 1574][1574-A]. Spelling is American English.

## The summary line and the description

### Open with one sentence in the third person, present tense

The summary line is one sentence in the third person singular present indicative, "Returns" and not "Return". It ends in a period and stands on its own, since a reader meets it in a listing, away from the item. See RFC 1574.

```rust
/// Opens the HDF5 file at `path` and reads it into memory.
```

Not this:

```rust
/// Open an HDF5 file from a filesystem path.
```

### Follow the summary with a blank line, then the description

A type's documentation is complete on its own, and a module's is a high-level summary. The description of an item says what it is, the invariant it holds, and what the format specification or libhdf5 calls it. See RFC 1574.

```rust
/// The width of a file address in bytes, the superblock's "Size of Offsets" field.
///
/// Every file address has this width:
///
/// - the addresses in the superblock, such as the root group address
/// - the addresses in object header messages
/// - the child and sibling addresses in B-tree nodes
/// - the chunk addresses in a chunk index
///
/// The C library calls this value `H5F_SIZEOF_ADDR` and sets it with `H5Pset_sizes`.
pub(crate) enum OffsetWidth {
```

### Use the common headings for their purposes, and a topical heading for a topic that needs its own section

The common headings are `# Examples`, `# Panics`, `# Errors`, `# Safety`, `# Aborts` and `# Undefined Behavior`. Write `# Errors` for a function that returns `Result`, opening "Returns [`Variant`] if …", `# Panics` where it can panic, and `# Safety` on an `unsafe` function. See RFC 1574.

A distinct topic that needs a section of its own takes a topical heading, such as `# Note`, `# Performance` or `# Platform-specific behavior`, placed where a reader who reads to the end finds it, as the standard library does. Never write a heading that narrates.

```rust
/// Parses the superblock's "Size of Offsets" byte.
///
/// # Errors
///
/// Returns [`FormatError::InvalidOffsetSize`] if `size` is not 2, 4, or 8.
```

Not this:

```rust
//! # Why this exists
```

### Give a public item an example where a call is not obvious from its signature

Write the examples under `# Examples`, plural, as the standard library does, and show more than one where the item has more than one calling convention. An example is a doctest and runs in CI. See RFC 1574.

### Document every public item, and a private item where its name and signature do not say enough

Every public item has a doc comment. A field or a variant has one where its name does not say what it holds.

## What never goes in a comment

### Describe the code as it stands

Leave out the change that introduced the item, what the code did before, the defect the change closes, how many call sites are converted, and what a later change will do. Those facts belong in the commit message, where a reader has the diff.

Write this:

```rust
//! Random-access byte sources for the reader: the [`Source`] trait and its backends.
//!
//! [`BytesSource`] reads from a buffer that holds the whole file, and [`ReadSeekSource`] from a
//! reader that seeks. [`MetadataCachingSource`] wraps either with a bounded cache of the metadata
//! reads and passes a dataset's payload through to the inner source.
```

Not this:

```rust
//! # Why this exists
//!
//! Today the reader holds the **entire file** in one `Vec<u8>` ([`crate::File`])
//! and threads a `&[u8]` of that whole buffer through every parser, indexing it
//! by absolute offset. That is simple and fast, but it has a hard ceiling: a
//! file larger than the process address space cannot be loaded at all.
```

### Do not restate the name or the signature

A summary line says the one fact the name and the signature do not.

Write this:

```rust
/// Returns the access properties this file was opened with.
///
/// An open without options, such as [`File::open`], has [`FileAccessProperties::new`].
pub fn access_properties(&self) -> FileAccessProperties {
```

Not this:

```rust
/// Return the access properties used when opening this file.
pub fn access_properties(&self) -> FileAccessProperties {
```

## Links

### Link every type, function and constant mentioned, and write other links reference style

Link the types a doc comment mentions, with intra-doc links (``[`Source`]``), and write a link to anything else reference style, with the definition at the end of the doc comment. The citation examples below have both. See RFC 1574.

## Citing the specification

### Name the section by its title, without the level number, and by the version

Write the link as a reference link whose definition sits at the end of the doc comment, with the section's anchor in the definition. Rendered documentation then shows the section title alone.

The level number shifts between releases of the documentation, and the anchor stays: `subsubsec_fmt4_dataobject_hdr_msg_simple` is "IV.A.2.b. The Dataspace Message" in the source pinned at `hdf5_2.1.0`, and "IV.A.3.b." on the page today.

```rust
/// The size in bytes of the prefix of a version 1 B-tree node that does not depend on the
/// offset width: the signature (4), the node type (1), the node level (1), and the entries
/// used (2). The two sibling addresses that follow are each `offset_size` bytes wide, see
/// [`btree_v1_node_header_size`]. The node is defined in "Version 1 B-trees" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
```

A module doc cites the same way:

```rust
//! Both superblock fields are defined in "Format Signature and Superblock" of the [format
//! specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
```

### Cite the newest version that documents the structure, and write the on-disk version out beside it

The on-disk version of a structure and the version of the document that describes it are separate numbers. Version 4.0 describes a version 1 object header, so an item that implements one cites version 4.0.

### Link an older version only where the older wording itself matters

The two cases: a description a later version dropped or changed, and a rule about the files that libraries of that era wrote.

### Cite in words where a private item encodes a detail of the format

Cite the source of a version-dependent field width or of a quirk libhdf5 writes. The comment need not be a doc comment.

```rust
// Version 3: bits 2-3 of the flags byte. 0x26 is Late alloc, Never, and Defined,
// and 0x2a the same with IfSet. The C library writes Defined for
// `H5D_FILL_TIME_NEVER` whenever a fill value is set (`H5Ofill.c`, HDF5 2.1.0).
```

### Link a specification section from a doc comment, or link nothing

Never link a page of this repository from a doc comment. Rustdoc is published on docs.rs, where such a link is dead. Never link a `latest` page without a document version in its address.

## Citing libhdf5

### Name the function, property or macro in backticks, and never link it

The anchors of the libhdf5 API reference change with every rebuild, and no versioned tree of it is published. Cite libhdf5 in a sentence of its own, with the function or the C library as the subject.

```rust
/// The C library calls this value `H5F_SIZEOF_ADDR` and sets it with `H5Pset_sizes`.
```

```rust
/// Returns [`Error::FileMarkedInUse`] if the status flags mark the file as open for writing.
/// `H5Fopen` makes the same check.
```

### Add the source file and the release where the behavior comes from the implementation

```rust
// The C library writes Defined for `H5D_FILL_TIME_NEVER` whenever a fill value is set
// (`H5Ofill.c`, HDF5 2.1.0).
```

## Appendix: specification versions

Each document version has its own page in the HDF Group's documentation tree and its own source file in the HDF5 repository.

| Version | What it adds | Source |
| --- | --- | --- |
| [1.0][fmt1] | The base format: the super block, B-link trees, group and symbol nodes, local heaps, the global heap, the free-space heap, and data object headers | [`H5.format.1.0.dox`][dox1] |
| [1.1][fmt11] | Superblock version 1 with its Indexed Storage Internal Node K field, and the Data Storage - Fill Value message beside Data Storage - Fill Value (Old) | [`H5.format.1.1.dox`][dox11] |
| [2.0][fmt2] | Superblock version 2, the Superblock Extension, version 2 B-trees, the Fractal Heap, the Shared Object Header Message Table, and the version 2 data object header prefix | [`H5.format.2.0.dox`][dox2] |
| [3.0][fmt3] | Its change list for HDF5 1.10: superblock version 3, version 2 B-tree types 10 and 11, the Global Heap Block for Virtual Datasets, data layout message version 4, the File Space Info message, and Appendix C with five indexing types | [`H5.format.3.0.dox`][dox3] |
| [4.0][fmt4] | Its change list for HDF5 2.0: datatype message version 5 and the Complex class (11) | [`H5.format.4.0.dox`][dox4] |

Versions 1.0, 1.1 and 2.0 have no change list of their own, and the cells above are read from their sections. Version 1.0 gives Release 1.4.4 (July 2002) as current and version 1.1 Release 1.6 (July 2003). Version 2.0 says the HDF5 1.10 format was not finalized as of October 2015. Versions 3.0 and 4.0 include the same three change lists, for HDF5 1.10, HDF5 1.12 and HDF5 2.0, and the cells above use one list each. `FileFormatDisc.dox` at the same tag describes version 4.0 as the current HDF5 file format.

A document version describes every on-disk version of the structures it still covers, so version 4.0 defines superblock versions 0 to 3 and both object header versions. A structure that a later version dropped stays in the older document alone: version 1.0 is the only one that describes the Data Storage - Compact message.

Each section anchor id contains the document version, as in `subsec_fmt4_boot_super` for the superblock section of version 4.0, and the source files above define the anchors. The sources are pinned to the `hdf5_2.1.0` tag, where all five documents sit under `doxygen/dox/`.

The crosschecks under `crates/crosscheck/` link the libhdf5 that `hdf5-metno` bundles, release 2.2.0, under the `__hdf5-bundled` feature that `just interop::test-bundled` passes, and its source is at [tag `2.2.0`][libhdf5]. `just interop::hdf5-bundled-version` prints that release. `just interop::default` links 1.8.23, 1.10.11, 1.12.3, 1.14.6 and 2.2.0 in turn, which `scripts/interop.just` lists, and `just interop::test-external-hdf5` links the installation under `HDF5_DIR`.

[1574-A]: https://rust-lang.github.io/rfcs/1574-more-api-documentation-conventions.html#appendix-a-full-conventions-text
[fmt1]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t1.html
[fmt11]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t11.html
[fmt2]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t2.html
[fmt3]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html
[fmt4]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html
[dox1]: https://github.com/HDFGroup/hdf5/blob/hdf5_2.1.0/doxygen/dox/H5.format.1.0.dox
[dox11]: https://github.com/HDFGroup/hdf5/blob/hdf5_2.1.0/doxygen/dox/H5.format.1.1.dox
[dox2]: https://github.com/HDFGroup/hdf5/blob/hdf5_2.1.0/doxygen/dox/H5.format.2.0.dox
[dox3]: https://github.com/HDFGroup/hdf5/blob/hdf5_2.1.0/doxygen/dox/H5.format.3.0.dox
[dox4]: https://github.com/HDFGroup/hdf5/blob/hdf5_2.1.0/doxygen/dox/H5.format.4.0.dox
[libhdf5]: https://github.com/HDFGroup/hdf5/tree/2.2.0
