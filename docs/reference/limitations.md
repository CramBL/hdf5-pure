# Limitations & Unsupported Features

`hdf5-pure` reads and writes a broad, interoperable subset of the HDF5 format. Where it cannot yet handle something, it returns a **clear typed error** rather than producing a wrong result — every gap below is a deliberate, well-messaged refusal, not a silent misread.

The refusals fall into two kinds:

- **[Deliberately unsupported](#deliberately-unsupported)** — by-design constraints or guards against file formats outside the range `hdf5-pure` models. These are not planned to change.
- **[Planned support](#planned-support)** — features refused *for now*, each tracked by a GitHub issue. The error messages for these read `... not supported yet` / `... cannot be ... yet`.

!!! note "Ordinary errors are not on this page"
    Malformed-file errors (truncated or garbled headers, an address that exceeds the platform's pointer width) and API-contract errors (deleting or copying the root group, conflicting edits in a single commit) are normal runtime errors, not capability limits, so they are not catalogued here.

## Deliberately unsupported

### Reading non-modeled formats

| Refused | Error | Why |
|---|---|---|
| Superblock version > 3 | `FormatError::UnsupportedVersion` | Superblock versions 0–3 are read; no higher version exists in the released format |
| An unrecognized object-header message flagged *must-understand* | `FormatError::UnsupportedMessage` | Refusing is the format-required behavior for a must-understand message a reader does not know |
| File Space Info message version other than 1 | `FormatError::UnsupportedFileSpaceInfoVersion` | Only version 1 is defined for the layouts this crate emits and reads |

These guard against files outside the format-version range `hdf5-pure` models; they are not features to add.

### Numeric element width

| Refused | Error | Why |
|---|---|---|
| An integer or float element wider than 8 bytes | `FormatError::NumericElementTooWide` | The typed numeric readers model an element as a 64-bit word taken from its leading eight bytes, and which part of a wider element that is depends on its byte order ([#361](https://github.com/CramBL/hdf5-pure/issues/361)) |

The refusal keeps a partial value from passing for a whole one: a 9-byte integer holding 2^64 would otherwise read back as `0`, and big-endian it would read back as 2^56, since the bytes kept are the *most* significant ones. It covers datasets and attributes alike, and an attribute this refuses is omitted from `attrs()` while `attr_datatypes()` still reports its width. `read_raw` and `read_u8`/`read_i8` hand back the bytes either way. Decoding these widths is a separate feature rather than a gap here — it means 128-bit variants throughout the `AttrValue` and `read_*` surfaces.

### Compression

| Refused | Error | Why |
|---|---|---|
| A filter whose backend is not compiled in | `FormatError::UnsupportedFilter` | Enable the `deflate` (or `zfp`) Cargo feature — see [Cargo Features](features.md) |
| ZFP outside fixed-rate, ranks 1–4, dtypes `f32`/`f64`/`i32`/`i64` | `FormatError::UnsupportedZfp` | The supported scope of the bundled ZFP codec — see [Compression](../guide/compression.md) |

### Producer-backed datasets

A dataset staged with [`MatBuilder::write_blocks`](../interop/matlab.md#writing-more-data-than-fits-in-memory) is always stored **uncompressed**, and requesting one on a builder configured for deflate is refused with `MatError::CompressionUnsupportedForBlocks` rather than silently stored unfiltered. The writer places every object before it emits a byte, so it needs the data region's exact size up front: unfiltered that is pure geometry, compressed it is not knowable without compressing — which would buffer the data the path exists to avoid. Supporting it would need either a two-pass producer contract (compress to measure, then compress again to emit, requiring the producer to be deterministic) or a spill file, so it is a separate design rather than a gap here.

Relatedly, a producer that fails partway leaves a **partial file** on the sink. That is inherent to a non-seekable destination, not a defect: write to a temporary path and rename on success if you need all-or-nothing.

### External data files

A dataset created with `H5Pset_external` keeps its elements in one or more files beside the HDF5 file. Its layout message records a contiguous layout with an **undefined** data address — byte-for-byte what a never-written dataset records — and a separate header message names the files. Reading one is refused with `FormatError::UnsupportedExternalStorage` rather than answering the fill value for data that exists but lives elsewhere, and `repack` refuses it rather than writing a copy with none of it ([#331](https://github.com/CramBL/hdf5-pure/issues/331), [#293](https://github.com/CramBL/hdf5-pure/issues/293)).

Writing is refused for the same reason, with `Error::EditUnsupported`: `Dataset::write`, `write_staged`, and `append_staged` would otherwise take the address-less layout for never-allocated storage, append the new elements to the HDF5 file and point the layout at them, leaving the file disagreeing with the external files about where its data lives. Delete the dataset and create it again in the same commit to replace it.

Copying is refused too, by name: `File::copy` and `File::copy_from` reproduce a dataset whose storage was never allocated as the empty storage it is, and an externally stored dataset carries that same address-less layout over data that exists — so a copy that treated the two alike would report success having written the schema and none of the elements ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).

What still works is reading the metadata — the shape, the datatype, and the address-less layout, since that layout is the evidence a caller needs — and setting and removing attributes. Following the external files is a separate feature rather than a gap here — it means resolving each named file against the reading process's own filesystem, which is outside what a self-contained HDF5 file describes.

### Repack faithfulness

`repack` rewrites a file and refuses **lossy filter re-encoding** (lossy float scale-offset, ZFP) rather than silently altering data: only *lossless* integer scale-offset can be re-encoded faithfully, since re-compressing lossy data would change the values. The dataset's fill value comes with it, in whichever of the two forms the source's filter recorded. (Repack instead copies already-compressed chunks **verbatim** wherever it can, which preserves lossy filters byte-exact without re-encoding.)

### SWMR (single-writer / multiple-reader)

SWMR append requires a **latest-format** file (v3 superblock) and **no userblock**. This mirrors the HDF5 SWMR model, which is only defined for the latest format; the C library refuses an older superblock for SWMR writing too, and neither library reads the SWMR-write flag back on one.

### In-place editing

In-place editing operates on files with **8-byte offsets and lengths** (what the writer emits and what modern files use). Other offset/length widths are not editable in place.

### Bounded-memory read-write

A file [`File::open_rw` edits bounded](../guide/editing.md#bounded-memory-appends) offers the same edit surface as one it mirrors — reads, immediate `Dataset::append`, and the staged surface (`write`, attribute edits, `create_*`/`delete`, `copy`, `commit`, `space_accounting`) — with a commit whose resident memory is bounded by the edit rather than by the file (`File::copy` excepted: copying an object reads the whole of it into memory). The bounded backing needs a latest-format file with 8-byte offsets and no userblock, and anything else is mirrored instead; reads have the [streaming backend's capabilities](../guide/streaming.md). It grows a file that persists its free space, including a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`); a paged file that does *not* persist its free space is refused with `Error::EditUnsupported` (recreate it with `persist = true`), whichever backing holds it.

Adding an **object-reference dataset** (`Group::create_dataset` with `with_path_references(...)`) resolves a path target against every object this commit places, but only once that object has actually been placed: `commit()` processes groups deepest-first and, within a group, non-reference datasets before reference ones, so a target that is itself still being written when the reference is resolved — an ancestor group, a same-depth sibling group ordered later, a copy destination (or its interior), or a `Dataset::write` target — is refused rather than resolved to a stale or wrong address. A target the same commit **deletes** — the deleted path or anything under it — is refused too, since a deletion reclaims the whole subtree and the reference would name space the file is about to hand out again. A target supplied as an **address** rather than as a path — `with_reference_data`, or `with_raw_data` over a datatype that holds a reference — is screened by address instead of by name, wherever the commit writes one: a new dataset, a `Dataset::write_staged` overwrite, and an in-file `copy`, which re-emits its source's references verbatim ([#317](https://github.com/CramBL/hdf5-pure/issues/317)). A **supplied** address is screened against both halves of what a commit can vacate — the objects it deletes, and the headers it rewrites elsewhere (a group it dirties, a dataset whose write relocates it) — and the second refusal points at the path form, since the object still exists — though only a dirty *group* is resolvable that way, once this commit has placed it; a relocating dataset write is refused by name outright, which is why the message names separate commits too. A **copied** address is screened against deletions only: a target that merely *moves* needs no refusal on either side, because the commit repoints every reachable stored reference once it has published its tree ([#324](https://github.com/CramBL/hdf5-pure/issues/324)), the copy's included.

Three things are refused rather than screened, all only in a commit that deletes: copying a **chunked** object-reference dataset, whose addresses sit inside chunks the copy path carries compressed and never decodes — the same limit that makes `repack` refuse one outright; copying an object carrying a **shared (SOHM) attribute message**, whose bytes this path cannot reach; and copying an object whose datatype or attribute this parser cannot read, since an unreadable datatype cannot be shown to be free of references. A **committed** (`H5Tcommit`) datatype is not one of these: it is resolved, so an object with a named type still copies. Separately, a datatype whose references this screen cannot read out of the element bytes — one wider than 8 bytes, a dataset-region reference, a variable length *of* references (whose addresses live in the global heap the elements point at), or a compound holding one of those beside a reference it *can* read — is refused rather than written unscreened. A **staged** dataset is refused in any commit that rebuilds a header — which is every commit except one whose only staged edit is a same-length in-place overwrite, and that one vacates nothing to dangle into; a **copied** one is refused only beside a deletion, like the three above. Nothing here builds such a type; `with_raw_data` is the door it comes through, and an in-file `copy` of a file the reference C library wrote is the other. A path the commit deletes and then [replaces](../guide/editing.md#replacing-an-object) resolves to the replacement once the replacement has been placed, so the placement-order rule above still governs it. A target untouched by the commit resolves against the pre-commit file; a path that resolves nowhere at all becomes an undefined reference, matching `FileBuilder`'s resolution convention for the same builder type. This is a permanent scope line (not a `... yet` gap): reproducing the whole-file writer's two-pass dummy/real-address scheme inside the editor's single-pass commit would be a large rewrite of the core apply loop for a narrow benefit.

### Object header message size

A version 2 object header describes each message's length in a 2-byte field, so no message it carries may exceed `OBJECT_HEADER_MESSAGE_MAX` (65,535 bytes). The whole-file writer refuses rather than truncating:

- A **compact attribute** past the limit would be refused with `FormatError::AttributeMessageTooLarge`, naming the attribute. The limit is on the *message* — name, datatype, dataspace, and data — not on the element count.
- Any **other** oversized message — most reachably a Link message from a very long dataset or group name — is refused with `FormatError::ObjectHeaderMessageTooLarge`, carrying the message type.

An attribute that would exceed the limit is not refused, though: it selects fractal-heap storage instead, where no such field bounds it. See [dense attribute storage](#dense-attribute-storage) below. `AttributeMessageTooLarge` is therefore a backstop no input reaches today — the writer sends exactly those attributes to a heap — kept because the limit it describes is a real property of the object header. Only `ObjectHeaderMessageTooLarge`, which covers messages with no heap alternative, is reachable on size.

The [in-place editor](../guide/editing.md) answers the limit the same way: an attribute whose message would exceed it is written to a heap rather than refused, and the messages with no heap alternative are refused with `Error::EditUnsupported`. What no storage lifts is the 2-byte *field* inside the attribute message that describes its name, datatype or dataspace — `FormatError::AttributeFieldTooLong`, on either path.

### Dense attribute storage

An object stores its attributes in a fractal heap when it has more than eight of them, **or** when any one of them is too large for an object-header message — the same disjunction the reference C library uses, and the reason a single large attribute is written rather than refused.

The writer emits the same heap geometry the reference C library uses for an attribute heap: a doubling table of direct blocks from 1 KiB up to 64 KiB, reached through a root indirect block once one block no longer holds everything, and indexed by B-trees of fixed 512-byte nodes that grow internal levels as the record count rises. An attribute serializing past **65,514 bytes** does not fit a heap *managed* object, so it is written as a **huge** object instead: its bytes go outside the managed blocks and a huge-objects B-tree maps a generated ID to them. There is no limit on an individual attribute's size, and none on how many attributes an object may carry. What is still refused:

- An attribute whose **name, datatype, or dataspace** serializes past **65,535 bytes** is refused with `FormatError::AttributeFieldTooLong`, naming the attribute and the field. Each has a 2-byte length field in the attribute message, and huge storage lifts the limit on an attribute's data, not on the fields that describe it.
- About a terabyte of *managed* attributes on one object is refused with `FormatError::DenseAttributeHeapTooLarge`: the heap's offsets are 40 bits wide, so its blocks cannot span more than that between them.

The [in-place editor](../guide/editing.md) emits the same storage through the same builder: an attribute edit that outgrows the object header moves the whole set to a heap, an object already using one is rebuilt around the edit, and a dataset or group created in place may carry one ([#102](https://github.com/CramBL/hdf5-pure/issues/102)). Three things are refused there rather than written: an attribute holding an object reference a commit could still repoint is not moved out of the header, a shared (SOHM) attribute message is not rewritten, and the heap's own bounds above apply as they do to the whole-file writer. The heap a rebuild supersedes is left as dead bytes for `repack`, like every dense heap this editor replaces.

Eight is fixed, on both paths. A version 2 object header may store its own `H5Pset_attr_phase_change` thresholds, and the in-place editor preserves that block across a rewrite so the reference C library still reports the pair a caller set — but the switch to a heap is still made at eight attributes whatever the stored pair says ([#422](https://github.com/CramBL/hdf5-pure/pull/422)).

The *total* is otherwise not limited, and it no longer rounds the whole heap up to a power of two: the table adds blocks rather than growing one. Space is still lost per block, though. An attribute that does not fit what remains of a block moves to the next one, and an attribute just over half of the largest block's size leaves most of a block unused, so a set of such attributes can still occupy close to twice its own size. The writer only ever appends to the block it filled last; it does not go back and reuse an earlier block's remainder the way the reference C library's free-space manager does.

### Group creation property list (GCPL)

There is no property-list API for group creation, and none of its settings are configurable — every group `hdf5-pure` writes (including the root group) has exactly one fixed shape: a new-style (v2 object header) group with compact link storage and no stored timestamps. An object header the [in-place editor](../guide/editing.md) *rewrites* keeps whatever times it already stored, with the modification and change times moved to the edit; only a header created from nothing gets this shape. This is equivalent to always creating every group with `obj_track_times = false`, and never switching to old-style (symbol-table) or dense (fractal-heap) link storage, regardless of file version or child count. Unlike the reference library, whose GCPL defaults vary by version, this shape is fixed on purpose: it keeps output byte-for-byte reproducible, which is exactly what makes `hdf5-pure` a good fit for stable snapshot files. See [#131](https://github.com/CramBL/hdf5-pure/issues/131).

## Planned support

Refused today with a `... yet` message, intended to land. Each row links to its tracking issue.

### In-place editing

| Capability | Tracking |
|---|---|
| **Dense** (fractal-heap) **link** storage — a group with more links than it keeps compactly | [#102](https://github.com/CramBL/hdf5-pure/issues/102) |
| Editing across **soft / external links** | [#103](https://github.com/CramBL/hdf5-pure/issues/103) |
| Copying a **version-1** object (attribute creation-order tracking is edited as of [#416](https://github.com/CramBL/hdf5-pure/issues/416)) | [#104](https://github.com/CramBL/hdf5-pure/issues/104) |
| Adding a **link** to a group that tracks **link creation order**, as netCDF-4 writes, once the addition would take the group past the compact-storage threshold its group-info message declares (8 links by default). Past it the links move to a fractal heap with a creation-order B-tree beside the name index, which is dense link storage — the row above, tracked by [#102](https://github.com/CramBL/hdf5-pure/issues/102). Below it such a group is edited like any other as of [#416](https://github.com/CramBL/hdf5-pure/issues/416): an added link takes the next creation index, a removed one leaves a gap without lowering the group's counter, and a copy of the group keeps the order it had | [#102](https://github.com/CramBL/hdf5-pure/issues/102) |
| Changing a **shared (SOHM) message's reference count**. A file with a shared-message table is read and edited, and a commit refuses what it would strand: an object whose attributes are a shared message is not rewritten, and an object header the file's shared-message index names is neither moved nor removed. A commit that *deletes* an object using a shared message leaves the count one too high — the message stays readable, and the entry is never reclaimed. `repack` rewrites the file with every message inline | [#417](https://github.com/CramBL/hdf5-pure/issues/417) |
| Adding **chunked/extensible variable-length-string** datasets | [#105](https://github.com/CramBL/hdf5-pure/issues/105) |
| **Overwriting** (`Dataset::write_staged`) with `with_path_references`, whose staged element bytes are placeholder addresses only the dataset-*creation* path resolves. A refusal on that builder, not on the datatype: a reference dataset still overwrites through `with_reference_data` (which supplies resolved addresses) or `with_raw_data`. `with_vlen_strings` overwrites as of [#321](https://github.com/CramBL/hdf5-pure/issues/321); the global heap collections it can prove are its own are reclaimed, and `repack` recovers the rest (see [Editing in place](../guide/editing.md#staging-and-committing-edits)) | [#321](https://github.com/CramBL/hdf5-pure/issues/321) |
| **Cross-file copy** of variable-length / reference / shared data, including an attribute naming a committed datatype | [#106](https://github.com/CramBL/hdf5-pure/issues/106) |
| Keeping an **object reference already stored in the file** valid when its target's header is rewritten, where the address cannot be reached. By *storage*: inside a **chunked** dataset's chunks, in a **dense** (fractal-heap) attribute or one held as a **shared (SOHM) record**, and in an attribute of a **version 1** object header (such a header's *element data* is reached, through the parser that reads both versions; version 1 is what the reference C library and h5py write by default). By *datatype*: an object reference **wider than 8 bytes**, a **dataset-region** reference, a **variable length** of references (the encoding the dimension-scale attribute `DIMENSION_LIST` uses), an **enumeration** over one, and any compound or array reaching one of those. What a commit does repoint as its last act is the rest: a contiguous or compact dataset's elements and an attribute held in the header, holding an 8-byte object reference at any depth of compound or array nesting, including through a committed (`H5Tcommit`) datatype. An unreached reference is left as it was, which is where every one of them stood before this; nothing is refused for being unreachable, since every commit rebuilds its root group and so would refuse every commit on such a file | [#324](https://github.com/CramBL/hdf5-pure/issues/324) |

### Repack

| Capability | Tracking |
|---|---|
| Repack of **region references**, non-8-byte object references, chunked/filtered/resizable **reference** datasets, and unrecognized filter pipelines (time, variable-length sequences, and 8-byte object references now repack faithfully; chunked, filtered, and resizable variable-length datasets repack as of [#109](https://github.com/CramBL/hdf5-pure/issues/109)) | [#107](https://github.com/CramBL/hdf5-pure/issues/107) |
| Dropping a **committed** (`H5Tcommit`) datatype a surviving dataset or attribute still names, and a committed datatype no hard link reaches — refused rather than left pointing at nothing (a committed datatype otherwise repacks: the named object is recreated and its users still name it) | [#254](https://github.com/CramBL/hdf5-pure/issues/254) |

### Reading

| Capability | Tracking |
|---|---|
| **Filter-encoded fractal-heap** objects | [#108](https://github.com/CramBL/hdf5-pure/issues/108) |
| **Virtual (VDS)** datasets | [#111](https://github.com/CramBL/hdf5-pure/issues/111) |
| **Writing** shared (SOHM) messages: this crate's writer stores every message privately, and `repack` rewrites a shared-message file that way. Reading one is supported as of [#417](https://github.com/CramBL/hdf5-pure/issues/417) — the shared-message table, both index kinds, and the heap lookup — for the datatype, dataspace, fill value, filter pipeline and attribute messages `H5Pset_shared_mesg_index` can share | [#417](https://github.com/CramBL/hdf5-pure/issues/417) |

A virtual dataset is refused by `repack` as well, since it cannot relocate data living outside the file; that lifts together with VDS read support ([#111](https://github.com/CramBL/hdf5-pure/issues/111)).

### Writing

| Capability | Tracking |
|---|---|
| **Append** to a variable-length dataset (writing one resizable is supported; growing it is not) | [#109](https://github.com/CramBL/hdf5-pure/issues/109) |
| More than one **unlimited** dimension in a `maxshape`. The reference library indexes that dataspace with a version-2 B-tree, which this crate neither writes nor reads; one unlimited dimension is supported at any rank | [#299](https://github.com/CramBL/hdf5-pure/issues/299) |
| A `maxshape` whose chunk index would spend more than 32 MiB on elements describing no chunk. An Extensible Array allocates only the blocks its chunks land in, as the reference library does, so this bites only where the slack inside those blocks is itself large; a **Fixed** Array is dense by format and costs the same there as in the reference library, so for a fixed `maxshape` this is a bound on what this writer will hold in memory while building the index in one pass | [#299](https://github.com/CramBL/hdf5-pure/issues/299) |
| Add a **chunked variable-length** dataset to an *existing* file in place | [#109](https://github.com/CramBL/hdf5-pure/issues/109) |
| Add a dataset or attribute naming a **committed** (`H5Tcommit`) datatype to an *existing* file in place — the in-place engine appends into a fixed layout and has nowhere to place the named type object. `FileBuilder::commit_datatype` writes one when the whole file is written, and `repack` carries them across | [#254](https://github.com/CramBL/hdf5-pure/issues/254) |

A dataset whose storage was never allocated keeps that state through `repack` ([#293](https://github.com/CramBL/hdf5-pure/issues/293)), so a schema-only file stays one instead of being written out full of its fill value. Three things bound that. A dataset that stores only *some* of its chunks is unaffected: a sparse grid cannot take the verbatim path, so it falls back to read-and-re-encode and the destination stores every slot — measured, one chunk in and ten out. A *resizable* destination is given the eagerly built Extensible Array this crate gives every empty resizable dataset, because an in-place append needs the index to exist before the first chunk arrives; it stores no chunk either way, and the index costs a few hundred bytes the source did not spend. And the `maxshape` bound in the table above does not apply to a dataset that stores nothing, since both halves of it are sized from the slots an index spans and an unallocated one spans none — such a dataset is written, and refused when its first chunk arrives instead.

Chunked, filtered, and resizable variable-length datasets now write, so the entries that were here for those have gone ([#109](https://github.com/CramBL/hdf5-pure/issues/109)). Two gaps remain. A resizable one can be created but not grown: `Dataset::append` is typed and `H5Element` covers only numeric scalars, and `append_raw` refuses a variable-length datatype because it cannot encode the heap references. And adding such a dataset to an existing file through the in-place edit engine is refused, since the engine appends into a fixed layout with nowhere to place the heap collections ahead of the chunks.

### SWMR

| Capability | Tracking |
|---|---|
| Append to **multi-dimensional** and **filtered** datasets | [#110](https://github.com/CramBL/hdf5-pure/issues/110) |

This gap is specific to SWMR (concurrent-reader) append. Appending to a **filtered** 1-D unlimited dataset without concurrent readers is already supported by any length, via [`Dataset::append_staged`](../guide/editing.md#appending-to-an-unlimited-dataset) and [streaming `Dataset::append`](../guide/editing.md#streaming-appends) alike — the streaming one under a lossless pipeline, since growing a lossy dataset's trailing chunk would re-encode committed values.
