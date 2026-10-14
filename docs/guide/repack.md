[`repack`](crate::repack()) rewrites a whole HDF5 file into a fresh, compact copy, optionally dropping objects on the way. It is the guaranteed shrink where in-place editing cannot give one: deleting an object cannot always return its bytes to the operating system.

This page is backed by [`examples/repack.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/repack.rs). Run it with:

```console
$ cargo run --example repack
```

## Why a delete cannot always shrink a file

Deleting an object inside a [`File::open_rw`](crate::File::open_rw) session, which [Editing files](crate::_guide::editing) covers, reuses the freed space *within that session*, and the file is truncated when the freed bytes happen to reach the very end. But a single delete-then-close cannot shrink a file whose freed region sits in the middle: an HDF5 file is a single address space, and a hole in the middle cannot be removed by truncating the tail. This is the same reason the HDF5 C library ships a separate `h5repack` tool.

[`repack`](crate::repack()) solves this by reading every surviving object and rewriting the whole file from scratch through [`FileBuilder`](crate::FileBuilder), so the result has no dead space and is strictly smaller when objects are dropped.

## Basic usage

[`repack(src, dst, &RepackOptions)`](crate::repack()) reads every object of `src` not excluded by the options and writes them into a fresh, compact file at `dst`.

```rust
# let dir = tempfile::tempdir()?;
# let input = dir.path().join("input.h5");
# let compact = dir.path().join("compact.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("keep").with_f64_data(&[1.0, 2.0, 3.0]);
# builder.write(&input)?;
use hdf5_pure::{repack, RepackOptions};

// Pure compaction copy: drop nothing, just remove dead space.
repack(&input, &compact, &RepackOptions::new())?;
# assert_eq!(
#     hdf5_pure::File::open(&compact)?.dataset("keep")?.read_f64()?,
#     vec![1.0, 2.0, 3.0]
# );
# Ok::<(), hdf5_pure::Error>(())
```

## Dropping objects

[`RepackOptions::new()`](crate::RepackOptions::new) starts from a pure-compaction copy. [`RepackOptions::drop_path(path)`](crate::RepackOptions::drop_path) adds a path to omit from the output and is chainable. Dropping a group drops its whole subtree.

```rust
# let dir = tempfile::tempdir()?;
# let input = dir.path().join("input.h5");
# let compact = dir.path().join("compact.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("keep").with_f64_data(&[1.0, 2.0, 3.0]);
# builder.create_dataset("scratch").with_f64_data(&vec![0.0; 1024]);
# let mut runs = builder.create_group("runs");
# runs.create_dataset("aborted").with_i32_data(&[1, 2, 3]);
# builder.add_group(runs.finish());
# builder.write(&input)?;
use hdf5_pure::{repack, RepackOptions};

// Drop a dataset and a whole group subtree, then write a fresh, compact file.
let options = RepackOptions::new()
    .drop_path("scratch")
    .drop_path("runs/aborted");
repack(&input, &compact, &options)?;
# assert!(std::fs::metadata(&compact)?.len() < std::fs::metadata(&input)?.len());
# Ok::<(), hdf5_pure::Error>(())
```

Leading and trailing slashes in a drop path are ignored, so `"grp/old"` and `"/grp/old"` are equivalent.

**Every drop path must exist.** A drop path that matches no object in the source fails the repack, since a no-op drop is treated as a mistake. The error is reported as [`Error::RepackUnsupported`](crate::Error::RepackUnsupported), and no output file is written.

## The fidelity guarantee

[`repack`](crate::repack()) never silently degrades data it understands. Every surviving object is reproduced faithfully: datatype, shape, max-shape, chunking, filters, element data, and attributes. Otherwise, the operation fails with [`Error::RepackUnsupported`](crate::Error::RepackUnsupported), which states the object and the reason. An object it cannot reproduce faithfully is rejected.

Unknown optional object-header messages follow the source's read-only semantics. A message flagged only as mandatory for a decoder with write access may be omitted when the object is rebuilt. [`RepackOptions::reject_unknown_messages_only_a_writer_must_understand`](crate::RepackOptions::reject_unknown_messages_only_a_writer_must_understand) turns that case into an error instead. Messages mandatory for every decoder are rejected. See the [compatibility guide](crate::_guide::compatibility#unknown-metadata) for the support matrix.

Fidelity checks and drop-path validation run before output bytes are written. Dataset chunk bytes are streamed from the source during the final write, so an I/O error while reading a chunk or writing the destination can leave a partial `dst`. Remove it before retrying.

### What it reproduces

| Aspect | Supported |
| --- | --- |
| Datatypes | fixed-point, floating-point, fixed-length string, time, bit-field, opaque, compound, enumeration and array, plus variable-length strings and sequences, and 8-byte object references (rewritten to their targets' new addresses) |
| Embedded addresses | a compound with a variable-length member, an object-reference member, or both, an array of such compounds, and nesting of either. The embedded addresses are rewritten, the surrounding bytes carried through untouched |
| Layout | contiguous / compact or chunked |
| Unallocated storage | a dataset created and never written stores nothing in the copy, though a read of it still returns the fill value |
| Filters | eligible fully allocated chunked datasets preserve any filter pipeline verbatim. Re-encoded paths support deflate, shuffle, Fletcher32, LZF, and lossless integer scale-offset |
| Structure | group hierarchy of arbitrary depth |
| Attributes | every datatype above, carried across with the source's own encoding (width, charset, string padding, and rank included) on datasets, groups, and root |
| File-space strategy | the source's strategy, page size, and threshold (carried forward as non-persistent) |

A repacked file has no free space to persist, so even when the source recorded a persistent file-space strategy the compact output carries that strategy forward as non-persistent. See [File-space strategy](crate::_guide::file_space) for what that controls.

Attributes keep their own encoding. An attribute is copied as the source encoded it, not as [`AttrValue`](crate::AttrValue) renders it. That matters because [`AttrValue`](crate::AttrValue) is a deliberately lossy convenience view: it has no byte order, no sub-width precision, no string padding rule, and no rank above one, so an attribute rebuilt from one would come back in this crate's own layout and flattened. Only an attribute whose element bytes hold a *location*, variable-length data or a reference, cannot be copied as-is. A variable-length string keeps its datatype and dataspace while its strings are written into the new file's heap, and a reference attribute is rejected. See [Attributes](crate::_guide::groups_attributes#attributes) for what a *read* still normalizes.

One deviation from byte-for-byte. A dataset that was never written stores nothing in the destination as it did in the source. A **resizable** one is the exception: it is given the eagerly built Extensible Array this crate gives every empty resizable dataset, because an in-place append needs the index to exist before the first chunk arrives. Its chunks stay absent either way, and the index costs a few hundred bytes the source did not spend. A dataset that stores only *some* of its chunks is a separate case and is unaffected: a sparse grid cannot take the verbatim path, so the destination re-encodes and stores every slot.

Fully allocated chunked datasets whose element bytes contain no variable-length or object-reference addresses take a **verbatim chunk-copy path**: each encoded chunk is copied without decoding or re-encoding it. That preserves lossy filters such as floating-point D-scale scale-offset and ZFP, filters this crate cannot decode or encode such as SZIP, and unknown filters byte-for-byte. The destination rebuilds the chunk index around those encoded chunks.

Two paths still re-encode data: contiguous or compact filtered datasets, and sparse chunked datasets whose chunk grid contains unallocated entries. Those paths require a filter pipeline this crate can reproduce losslessly: deflate, shuffle, Fletcher32, LZF, and lossless integer scale-offset. See [Compression](crate::_guide::compression) for filter behavior and the [compatibility guide](crate::_guide::compatibility#filters) for the support matrix.

### What it rejects (by name)

These are reported as [`Error::RepackUnsupported`](crate::Error::RepackUnsupported), which states the object, and never silently dropped or degraded:

| Rejected | Reason |
| --- | --- |
| chunked, filtered, or resizable datasets whose datatype **is or contains an object reference** | their object addresses are assigned as elements are re-staged, which a compressed chunk would need rewritten in place |
| variable-length sequences whose base type is itself variable-length, or a reference | the copied element bytes would carry addresses that go stale on rewrite |
| region references, and object references other than 8 bytes wide | their stored selections and addresses are not rewritten |
| object references in a file with a userblock, or to an object being dropped | the new target address cannot be resolved safely, or will not exist |
| a virtual data layout, and external data storage (`H5Pset_external`) | the element bytes live outside the file, and this crate does not read them |
| a lossy filter on a path that requires re-encoding | re-encoding is not guaranteed idempotent. Fully allocated chunked datasets preserve it verbatim |
| SZIP or an unknown filter on a path that requires re-encoding | this crate cannot encode it. Fully allocated chunked datasets preserve it verbatim |
| an attribute whose datatype is or contains a reference | its stored address is not rewritten, and no [`AttrValue`](crate::AttrValue) can re-encode it |
| file-space settings that do not parse | the source strategy cannot be reproduced faithfully |

## Verifying the result

After a repack, the surviving objects open exactly as before and the dropped objects are gone. Adapting the example:

```rust
# let dir = tempfile::tempdir()?;
# let input = dir.path().join("input.h5");
# let compact = dir.path().join("compact.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("keep").with_f64_data(&[1.0, 2.0, 3.0]);
# builder.create_dataset("scratch").with_f64_data(&vec![0.0; 1024]);
# let mut runs = builder.create_group("runs");
# runs.create_dataset("aborted").with_i32_data(&[1, 2, 3]);
# builder.add_group(runs.finish());
# builder.write(&input)?;
# let options = hdf5_pure::RepackOptions::new().drop_path("scratch").drop_path("runs");
# hdf5_pure::repack(&input, &compact, &options)?;
use hdf5_pure::{Error, File, FormatError};

let file = File::open(&compact)?;
let keep = file.dataset("keep")?.read_f64()?;
assert_eq!(keep, vec![1.0, 2.0, 3.0]);

// Dropped objects are absent.
let err = file.dataset("scratch").unwrap_err();
let Error::Format(FormatError::PathNotFound(missing)) = &err else {
    panic!("expected PathNotFound, got {err:?}");
};
assert_eq!(missing, "scratch");

let err = file.group("runs").unwrap_err();
let Error::Format(FormatError::PathNotFound(missing)) = &err else {
    panic!("expected PathNotFound, got {err:?}");
};
assert_eq!(missing, "runs");
# Ok::<(), hdf5_pure::Error>(())
```

## Repack vs. in-place editing

| | [`File::open_rw`](crate::File::open_rw) delete | [`repack`](crate::repack()) |
| --- | --- | --- |
| Reclaims space mid-session | Yes (reused for later writes) | n/a |
| Shrinks a closed file | Only if freed bytes reach the end | Always |
| Spans a reopen | No | Yes (writes a new file) |
| Output | edits the same file | a fresh file at `dst` |

For incremental edits where add/delete churn stays bounded, prefer a [`File::open_rw`](crate::File::open_rw) session, which [Editing files](crate::_guide::editing) covers. For guaranteed compaction across a reopen, or to drop objects and reclaim their space unconditionally, use [`repack`](crate::repack()).
