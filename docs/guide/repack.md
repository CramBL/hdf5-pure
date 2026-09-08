# Reclaiming Space (repack)

`repack` rewrites a whole HDF5 file into a fresh, compact copy, optionally dropping objects on the way. It is the guaranteed-shrink answer to a fundamental limitation of in-place editing: deleting an object cannot always return its bytes to the operating system.

!!! tip "Runnable example"
    This page is backed by [`examples/repack.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/repack.rs). Run it with:

    ```bash
    cargo run --example repack
    ```

## Why a delete cannot always shrink a file

Deleting an object inside an [`File::open_rw`](editing.md) session reuses the freed space *within that session*, and the file is truncated when the freed bytes happen to reach the very end. But a single delete-then-close cannot shrink a file whose freed region sits in the middle: an HDF5 file is a single address space, and a hole in the middle cannot be removed by truncating the tail. This is the same reason the HDF5 C library ships a separate `h5repack` tool rather than relying on deletion alone.

`repack` solves this by reading every surviving object and rewriting the whole file from scratch through [`FileBuilder`](writing.md), so the result has no dead space and is strictly smaller when objects are dropped.

## Basic usage

`repack(src, dst, &RepackOptions)` reads every object of `src` not excluded by the options and writes them into a fresh, compact file at `dst`.

```rust
use hdf5_pure::{repack, RepackOptions};

// Pure compaction copy: drop nothing, just remove dead space.
repack("input.h5", "compact.h5", &RepackOptions::new()).unwrap();
```

## Dropping objects

`RepackOptions::new()` starts from a pure-compaction copy. `RepackOptions::drop_path(path)` adds a path to omit from the output and is chainable. Dropping a group drops its whole subtree.

```rust
use hdf5_pure::{repack, RepackOptions};

// Drop a dataset and a whole group subtree, then write a fresh, compact file.
let options = RepackOptions::new()
    .drop_path("scratch")
    .drop_path("runs/aborted");
repack("input.h5", "compact.h5", &options).unwrap();
```

Leading and trailing slashes in a drop path are ignored, so `"grp/old"` and `"/grp/old"` are equivalent.

!!! warning "Every drop path must exist"
    A drop path that does not match any object in the source fails the repack rather than being silently ignored — a no-op drop is treated as a mistake. The error is reported as `Error::RepackUnsupported`, and no output file is written.

## The fidelity guarantee

`repack` never silently degrades data. Every surviving object is reproduced byte-for-byte — datatype, shape, max-shape, chunking, supported filters, raw element data, and attributes — or the whole operation fails with `Error::RepackUnsupported` naming the object and the reason. It refuses rather than approximate.

The operation is all-or-nothing: the entire source is validated and staged in memory before the first byte is committed, so on any failure nothing is written to `dst` and no partial output file is left behind.

### What it reproduces

| Aspect | Supported |
| --- | --- |
| Datatypes | fixed-point, floating-point, fixed-length string, time, bit-field, opaque, compound, enumeration, array; variable-length strings and sequences, and 8-byte object references (rewritten to their targets' new addresses) |
| Embedded addresses | a compound with a variable-length member, an object-reference member, or both; an array of such compounds; and nesting of either. The embedded addresses are rewritten, the surrounding bytes carried through untouched |
| Layout | contiguous / compact or chunked |
| Unallocated storage | a dataset created and never written comes out storing nothing, rather than materialized full of the fill value a read of it answers with |
| Filters | deflate, shuffle, fletcher32, LZF, and/or lossless integer scale-offset |
| Structure | group hierarchy of arbitrary depth |
| Attributes | every datatype above, carried across with the source's own encoding — width, charset, string padding, and rank included — on datasets, groups, and root |
| File-space strategy | the source's strategy, page size, and threshold (carried forward as non-persistent) |

A repacked file has no free space to persist, so even when the source recorded a persistent file-space strategy the compact output carries that strategy forward as non-persistent. See [File-space strategy](file-space.md) for what that controls.

!!! note "Attributes keep their own encoding"
    An attribute is copied as the source encoded it, not as `AttrValue` renders it. That matters because `AttrValue` is a deliberately lossy convenience view: it has no byte order, no sub-width precision, no string padding rule, and no rank above one, so an attribute rebuilt from one would come back in this crate's own layout and flattened. Only an attribute whose element bytes hold a *location* — variable-length data, or a reference — cannot be copied as-is; a variable-length string keeps its datatype and dataspace while its strings are restaged into the new file's heap, and a reference attribute is refused. See [Attributes](groups-attributes.md#attributes) for what a *read* still normalizes.

!!! note "One deviation from byte-for-byte"
    A dataset that was never written stores nothing in the destination as it did in the source — except a **resizable** one, which is given the eagerly built Extensible Array this crate gives every empty resizable dataset, because an in-place append needs the index to exist before the first chunk arrives. It stores no chunk either way; the index costs a few hundred bytes the source did not spend. A dataset that stores only *some* of its chunks is a separate case and is unaffected: a sparse grid cannot take the verbatim path, so the destination re-encodes and stores every slot.

!!! note "Lossless filters only"
    `repack` reads each dataset's *decompressed* bytes and re-applies its filters. It can therefore reproduce only **lossless** filters, where the re-encoded chunks decompress to the exact same bytes. This includes deflate, shuffle, fletcher32, LZF, and lossless integer scale-offset. See [Compression](compression.md) for the full filter list.

### What it refuses (by name)

These are reported as `Error::RepackUnsupported` naming the object, never silently dropped or degraded:

| Refused | Reason |
| --- | --- |
| chunked, filtered, or resizable datasets whose datatype **is or contains an object reference** | their object addresses are assigned as elements are re-staged, which a compressed chunk would need rewritten in place |
| variable-length sequences whose base type is itself variable-length, or a reference | the copied element bytes would carry addresses that go stale on rewrite |
| region references, and object references other than 8 bytes wide | their stored selections and addresses are not rewritten yet |
| object references in a file with a userblock, or to an object being dropped | the new target address cannot be resolved safely, or will not exist |
| a virtual data layout, and external data storage (`H5Pset_external`) | the element bytes live outside the file, and this crate does not read them |
| lossy filters: float D-scale scale-offset and ZFP | re-encoding is not guaranteed idempotent |
| SZIP filter | this crate cannot write it |
| an attribute whose datatype is or contains a reference | its stored address is not rewritten yet, and no `AttrValue` can re-encode it |

## Verifying the result

After a repack, the surviving objects open exactly as before and the dropped objects are gone. Adapting the example:

```rust
use hdf5_pure::File;

let file = File::open("compact.h5").unwrap();
let keep = file.dataset("keep").unwrap().read_f64().unwrap();
assert_eq!(keep, vec![1.0, 2.0, 3.0]);

// Dropped objects are absent.
assert!(file.dataset("scratch").is_err());
assert!(file.group("runs").is_err());
```

## Repack vs. in-place editing

| | `File::open_rw` delete | `repack` |
| --- | --- | --- |
| Reclaims space mid-session | Yes (reused for later writes) | n/a |
| Shrinks a closed file | Only if freed bytes reach the end | Always |
| Spans a reopen | No | Yes (writes a new file) |
| Output | edits the same file | a fresh file at `dst` |

For incremental edits where add/delete churn stays bounded, prefer an [`File::open_rw`](editing.md) session. For guaranteed compaction across a reopen, or to drop objects and reclaim their space unconditionally, use `repack`.
