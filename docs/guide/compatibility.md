# HDF5 compatibility

This page lists HDF5 features supported by `hdf5-pure`.

The tables cover file-format compatibility and HDF5 behavior. Features not listed here have not been audited for this page. Section placement reflects typical reader relevance.

| Status | Meaning |
|:---:|---|
| ✅ | Supported |
| 🟡 | Partially supported. See the detail column. |
| ❌ | Unsupported |
| ⬆️ | Not emitted. Output uses another supported encoding. |

## Core HDF5 compatibility

### Groups

| Storage | Read | Create | Modify | Detail |
|---|:---:|:---:|:---:|---|
| Version 1 symbol table | ✅ | ⬆️ | ✅ | Modified groups use compact link storage. |
| Compact link storage | ✅ | ✅ | ✅ | |
| Dense link storage | ✅ | ❌ | ❌ | Deleting a group's last hard link reclaims its v2 B-tree indexes, its fractal heap, and its subtree. A filtered link heap is reclaimed too. A group whose link heap is filtered and has huge objects or a filter without a decoder stays allocated with its subtree. Managed-space free-space-manager metadata is reclaimed with the heap when that metadata is valid and supported. Invalid or unsupported manager metadata leaves the heap allocated. |
| Link creation-order tracking | ✅ | ❌ | 🟡 | New links can be added while compact storage remains valid. |

### Attributes

| Feature | Read | Create | Modify | Detail |
|---|:---:|:---:|:---:|---|
| Compact storage | ✅ | ✅ | ✅ | |
| Dense storage | ✅ | ✅ | ✅ | Deleting an object's last hard link, or editing its dense attributes, reclaims the set's v2 B-tree indexes and its fractal heap. A filtered attribute heap is reclaimed too. One with huge objects or a filter without a decoder stays allocated, and so does its owner unless the owner is a group with dense link storage. Managed-space free-space-manager metadata is reclaimed with the heap when that metadata is valid and supported. Invalid or unsupported manager metadata leaves the heap allocated. |
| Attribute creation-order tracking | ✅ | ❌ | ✅ | |
| Attribute phase-change thresholds | ✅ | ❌ | ✅ | Existing thresholds are preserved. |
| Null dataspace | ✅ | ❌ | ❌ | Reading returns an empty value. |
| Variable-length strings | ✅ | ✅ | ✅ | |

### Dataspaces

| Feature | Read | Write | Detail |
|---|:---:|:---:|---|
| Scalar dataspace | ✅ | ✅ | |
| Simple dataspace | ✅ | ✅ | |
| Zero-length dimension | ✅ | ✅ | |
| Maximum dimensions | ✅ | ✅ | |
| Unlimited dimension | ✅ | ✅ | |
| Multiple unlimited dimensions | ✅ | ❌ | Writing requires a version 2 B-tree chunk index. |

### Dataset layouts

`Inspect` means that the layout metadata can be read without reading dataset values.

| Layout | Inspect | Read data | Write |
|---|:---:|:---:|:---:|
| Compact | ✅ | ✅ | ✅ |
| Contiguous | ✅ | ✅ | ✅ |
| Chunked | ✅ | ✅ | ✅ |
| Virtual | ✅ | ❌ | ❌ |
| External raw storage | ✅ | ❌ | ❌ |

### Dataset extension

| Feature | Status | Detail |
|---|:---:|---|
| Staged append | ✅ | |
| In-place append | 🟡 | Requires an Extensible Array chunk index. |
| Filtered append | ✅ | |
| Non-chunk-aligned filtered append | ✅ | |
| Lossy filtered partial-chunk append | ❌ | Existing values would require re-encoding. |
| Multiple unlimited dimensions | ❌ | Requires version 2 B-tree chunk-index writing. |

### Chunk semantics

| Feature | Read | Configure | Detail |
|---|:---:|:---:|---|
| Unallocated chunks | ✅ | ✅ | |
| Sparse chunk grids | ✅ | ✅ | |
| Partial edge chunks | ✅ | ✅ | |
| Per-chunk filter masks | ✅ | ✅ | |
| `H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS` | ✅ | ❌ | |
| Fill value | ✅ | ✅ | |
| `H5D_FILL_TIME_NEVER` | ✅ | ✅ | |

### Filters

`Verbatim repack` means that encoded chunk bytes can be copied without decoding the filter.

| Filter | Decode | Encode | Verbatim repack | Detail |
|---|:---:|:---:|:---:|---|
| Deflate | ✅ | ✅ | ✅ | Requires the `deflate` feature. |
| Shuffle | ✅ | ✅ | ✅ | |
| Fletcher32 | ✅ | ✅ | ✅ | |
| SZIP | ❌ | ❌ | ✅ | |
| Scale-offset, integer | ✅ | ✅ | ✅ | |
| Scale-offset, floating-point D-scale | ✅ | ✅ | ✅ | |
| LZF | ✅ | ✅ | ✅ | |
| ZFP | 🟡 | 🟡 | ✅ | Requires the `zfp` feature. Fixed-rate `f32`, `f64`, `i32`, and `i64` are supported for ranks 1 through 4. |
| Unknown filter | ❌ | ❌ | ✅ | |

Verbatim repack requires a fully allocated chunk grid.

### Datatypes

| Datatype class | Read | Write | Detail |
|---|:---:|:---:|---|
| Fixed-point | ✅ | ✅ | Element sizes up to 64 bits. |
| Floating-point | ✅ | ✅ | IEEE 754 binary32 and binary64. |
| Time | ✅ | ✅ | |
| Fixed-length string | ✅ | ✅ | |
| Variable-length string | ✅ | ✅ | |
| Bit field | ✅ | ✅ | |
| Opaque | ✅ | ✅ | |
| Compound | ✅ | ✅ | |
| Enumeration | ✅ | ✅ | |
| Variable-length sequence | ✅ | 🟡 | Repack writes sequences. `DatasetBuilder` has no general sequence setter. |
| Array | ✅ | ✅ | |
| Object reference | ✅ | ✅ | 8-byte object references. |
| Dataset-region reference | ❌ | ❌ | |
| Complex class | ❌ | ❌ | HDF5 2.0 complex datatypes are unsupported. Complex helpers use compound `{real, imag}` values. |

#### Fixed-point layouts

| Feature | Read | Write |
|---|:---:|:---:|
| Signed integers | ✅ | ✅ |
| Unsigned integers | ✅ | ✅ |
| Little-endian byte order | ✅ | ✅ |
| Big-endian byte order | ✅ | ✅ |
| Sub-byte precision | ✅ | ✅ |
| Nonzero bit offset | ✅ | ✅ |

#### Floating-point layouts

| Feature | Read | Write |
|---|:---:|:---:|
| IEEE 754 binary32 | ✅ | ✅ |
| IEEE 754 binary64 | ✅ | ✅ |
| Little-endian byte order | ✅ | ✅ |
| Big-endian byte order | ✅ | ✅ |
| Arbitrary HDF5 floating-point layouts | ❌ | ❌ |

### Compound datatypes

| Feature | Read | Write | Detail |
|---|:---:|:---:|---|
| Explicit member offsets | ✅ | ✅ | |
| Padding between members | ✅ | ✅ | |
| Nested compound types | ✅ | ✅ | |
| Array members | ✅ | ✅ | |
| Variable-length members | ✅ | 🟡 | Repack rewrites embedded heap addresses. |
| Object-reference members | ✅ | 🟡 | Repack rewrites embedded object addresses. |

### Strings

| Feature | Read | Write | Detail |
|---|:---:|:---:|---|
| Fixed-length ASCII | ✅ | ✅ | |
| Fixed-length UTF-8 | ✅ | ✅ | |
| Variable-length ASCII | ✅ | ✅ | |
| Variable-length UTF-8 | ✅ | ✅ | |
| Null-padded fixed strings | ✅ | ✅ | |
| Null-terminated fixed strings | ✅ | 🟡 | Writing requires raw datatype and data APIs. |
| Space-padded fixed strings | ✅ | 🟡 | Writing requires raw datatype and data APIs. |

## Extended HDF5 compatibility

### Enumeration datatypes

| Feature | Read | Write |
|---|:---:|:---:|
| Signed integer base | ✅ | ✅ |
| Unsigned integer base | ✅ | ✅ |
| Arbitrary integer base width up to 64 bits | ✅ | ✅ |

### Committed datatypes

| Feature | Read | Create | Repack |
|---|:---:|:---:|:---:|
| Named datatype object | ✅ | ✅ | ✅ |
| Dataset using a committed datatype | ✅ | ✅ | ✅ |
| Attribute using a committed datatype | ✅ | ✅ | ✅ |

### References

| Feature | Read | Write | Dereference |
|---|:---:|:---:|:---:|
| 8-byte object reference | ✅ | ✅ | ✅ |
| Dataset-region reference | ❌ | ❌ | ❌ |
| Object reference wider than 8 bytes | ❌ | ❌ | ❌ |

### Variable-length data

| Feature | Read | Write | Detail |
|---|:---:|:---:|---|
| Variable-length UTF-8 string | ✅ | ✅ | |
| Variable-length ASCII string | ✅ | ✅ | |
| Chunked variable-length string | ✅ | ✅ | |
| Filtered variable-length string | ✅ | ✅ | |
| Resizable variable-length string | ✅ | ✅ | |
| Variable-length sequence | ✅ | 🟡 | Repack supports sequence values. |
| Variable-length member in a compound type | ✅ | 🟡 | Repack rewrites embedded heap addresses. |

### Datatype conversion

| Feature | Status | Detail |
|---|:---:|---|
| Integer to integer | ✅ | |
| Integer to floating-point | ✅ | |
| Floating-point to integer | ✅ | |
| Floating-point to floating-point | ✅ | |
| Enumeration through integer base type | ✅ | |
| Soft numeric clamping | ✅ | Matches `libhdf5` numeric read conversion. |
| User-registered conversion function | ❌ | |
| Conversion exception callback | ❌ | |
| Hard conversion registration | ❌ | |
| Soft conversion registration | ❌ | |

## Advanced HDF5 compatibility

### SWMR

| Feature | Status | Detail |
|---|:---:|---|
| SWMR reader | 🟡 | Supports refreshing reads for the append-oriented subset. |
| SWMR writer | 🟡 | Requires an unfiltered chunked dataset with one unlimited dimension and chunk-aligned appends. |

### File-space management

| Feature | Read | Configure | Detail |
|---|:---:|:---:|---|
| `H5F_FSPACE_STRATEGY_FSM_AGGR` | ✅ | 🟡 | Edits reuse freed space through the free-space managers, without aggregators. |
| `H5F_FSPACE_STRATEGY_PAGE` | ✅ | ✅ | |
| `H5F_FSPACE_STRATEGY_AGGR` | ✅ | 🟡 | Staged edits and immediate appends use separate metadata and raw aggregates. Freed extents are discarded. |
| `H5F_FSPACE_STRATEGY_NONE` | ✅ | 🟡 | Edits allocate at the end of the file, and a deletion leaves the file at its length. |
| Aggregators | ✅ | 🟡 | Staged edits and immediate appends under `H5F_FSPACE_STRATEGY_AGGR` share session aggregates. Whole-file writes and other strategies use no classic aggregates. |
| Persistent free-space managers | ✅ | ✅ | |
| Free-space threshold | ✅ | ✅ | Edits track a freed region shorter than the threshold only where it adjoins tracked free space, which under `H5F_FSPACE_STRATEGY_PAGE` is free space in its own page for a region shorter than a page. Page tails are tracked whatever the threshold. |
| File-space page size | ✅ | ✅ | The writer rejects a page size outside 512 bytes to 1 GiB under every strategy, as libhdf5 does. `File::open` reads a file that stores one without its file-space settings, and `File::open_rw` rejects it. |
| Persisted free-space reuse | ✅ | ✅ | |

## HDF5 file-format compatibility

### File structure

| Feature | Read | Emit | Detail |
|---|:---:|:---:|---|
| User block | ✅ | ✅ | |
| Superblock v0 | ✅ | ⬆️ | New files use v2 or v3. |
| Superblock v1 | ✅ | ⬆️ | New files use v2 or v3. |
| Superblock v2 | ✅ | ✅ | |
| Superblock v3 | ✅ | ✅ | |
| Object header v1 | ✅ | ⬆️ | New object headers use v2. |
| Object header v2 | ✅ | ✅ | |
| Object-header continuation blocks | ✅ | ✅ | |
| Shared object-header messages | ✅ | ⬆️ | Repack writes messages inline. Edits that invalidate shared-message tables are rejected. |
| Metadata checksums | ✅ | ✅ | General metadata checksum validation requires the `checksum` feature. Ownership-sensitive free-space-manager parsing validates the free-space-manager header and section-list checksums even when that feature is disabled. |

Shared object-header messages are supported for datatypes, dataspaces, fill values, filter pipelines, and attributes.

### Chunk indexes

| Index | Read data | Emit | Enumerate | Detail |
|---|:---:|:---:|:---:|---|
| Version 1 B-tree | ✅ | ⬆️ | ✅ | Rewritten with a version 4 chunk index. |
| Single chunk | ✅ | ✅ | ✅ | |
| Implicit | ✅ | ⬆️ | ✅ | Rewritten with another version 4 chunk index. |
| Fixed array | ✅ | ✅ | ✅ | |
| Extensible array | ✅ | ✅ | ✅ | |
| Version 2 B-tree | ✅ | ❌ | ✅ | Read/enumerate and deletion reclaim are supported. The editor does not create or mutate the index. |

`Enumerate` refers to `Dataset::chunks()`.

### Unknown metadata

| Feature | Read-only | Read-write | Repack | Detail |
|---|:---:|:---:|:---:|---|
| Unknown optional object-header message | ✅ | 🟡 | 🟡 | Writer requirements encoded in the message flags are enforced. |
| Unknown message mandatory for readers | ❌ | ❌ | ❌ | |
| Unknown message mandatory for writers only | ✅ | ❌ | 🟡 | Repack can be configured to reject these messages. |
