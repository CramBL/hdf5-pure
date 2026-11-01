# HDF5 compatibility

This page lists HDF5 features supported by `hdf5-pure`.

The tables cover file-format compatibility and HDF5 behavior. Unlisted features have not been audited for this page.

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
| Version 1 symbol table | ✅ | ⬆️ | 🟡 | Edits convert storage to compact/dense. Soft links prevent modification. |
| Compact link storage | ✅ | ✅ | ✅ | |
| Dense link storage | ✅ | 🟡 | 🟡 | Edits: unfiltered managed heaps with seven-byte IDs. Encoded links up to 4096 bytes. |
| Link creation-order tracking | ✅ | ❌ | ✅ | Preserved across compact/dense transitions. |
| Link phase-change thresholds | ✅ | ❌ | ✅ | Existing thresholds preserved. New groups use 8/6. |

### Link resolution

| Link | Resolve | Detail |
|---|:---:|---|
| Hard link | ✅ | |
| Soft link | ✅ | |
| External link | ❌ | |

### Attributes

| Feature | Read | Create | Modify | Detail |
|---|:---:|:---:|:---:|---|
| Compact storage | ✅ | ✅ | ✅ | |
| Dense storage | ✅ | ✅ | ✅ | Deletion and attribute edits may leave heap storage allocated. |
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
| Multiple unlimited dimensions | ✅ | ❌ | |

### Dataset layouts

`Inspect` means that the layout metadata can be read without reading dataset values.

| Layout | Inspect | Read data | Write | Detail |
|---|:---:|:---:|:---:|---|
| Compact | ✅ | ✅ | ⬆️ | Output uses contiguous or chunked storage. |
| Contiguous | ✅ | ✅ | ✅ | |
| Chunked | ✅ | ✅ | ✅ | |
| Unallocated contiguous/chunked | ✅ | ✅ | ✅ | Fill-value reads. |
| Virtual | ✅ | ❌ | ❌ | |
| External raw storage | ✅ | ❌ | ❌ | |

### Dataset selections

| Selection | Read | Write | Detail |
|---|:---:|:---:|---|
| Whole dataset | ✅ | ✅ | Overwrites require the existing shape and datatype. |
| Leading-dimension rows | ✅ | ❌ | Clamps to the current extent. |
| Unit-stride multidimensional region | ✅ | ❌ | |
| Strided or repeated-block selection | ❌ | ❌ | |
| Point selection | ❌ | ❌ | |

### Dataset extension

| Feature | Status | Detail |
|---|:---:|---|
| Dataset geometry | 🟡 | Rank-one, chunked, unlimited. Extensible Array indexes only. |
| Staged append | ✅ | Allocated or unallocated indexes. |
| In-place append | ✅ | Allocated or unallocated indexes. |
| Buffered append | 🟡 | No SWMR. |
| Filtered append | ✅ | Re-encodable pipelines only. |
| Non-chunk-aligned lossless filtered append | ✅ | |
| Lossy filtered partial-chunk append | 🟡 | Opt-in with `LossyTailReencoding::Allow`. |
| Unfiltered partial tails with lossy filters | ✅ | `PartialEdgeFiltering::Skip`. |
| Multiple unlimited dimensions | ❌ | |

### Chunk semantics

| Feature | Read | Configure | Detail |
|---|:---:|:---:|---|
| Unallocated chunks | ✅ | ✅ | |
| Sparse chunk grids | ✅ | ✅ | |
| Partial edge chunks | ✅ | ✅ | |
| Per-chunk filter masks | ✅ | ✅ | |
| `H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS` | ✅ | ✅ | |
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

| Chunk allocation | Repack | Detail |
|---|:---:|---|
| Fully allocated | ✅ | Verbatim copy for types without VLEN or object references. |
| Partially allocated | 🟡 | Lossless re-encoding only. |
| Unallocated | ✅ | Filter metadata preserved, even for unsupported filters. |

### Datatypes

| Datatype class | Read | Write | Detail |
|---|:---:|:---:|---|
| Fixed-point | ✅ | ✅ | Numeric conversion: element sizes up to 64 bits. |
| Floating-point | ✅ | ✅ | Numeric conversion: IEEE 754 binary32 and binary64 only. |
| Time | ✅ | ✅ | |
| Fixed-length string | ✅ | ✅ | |
| Variable-length string | ✅ | ✅ | |
| Bit field | ✅ | ✅ | |
| Opaque | ✅ | ✅ | |
| Compound | ✅ | ✅ | |
| Enumeration | ✅ | ✅ | |
| Variable-length sequence | ✅ | 🟡 | General sequence writing requires repack. |
| Array | ✅ | ✅ | |
| Object reference | ✅ | ✅ | 8-byte object references. |
| Dataset-region reference | ❌ | ❌ | |
| Native complex (class 11) | 🟡 | 🟡 | Homogeneous rectangular form. Metadata inspection and raw dataset access only. |

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

| Feature | Read | Write | Detail |
|---|:---:|:---:|---|
| IEEE 754 binary32 | ✅ | ✅ | |
| IEEE 754 binary64 | ✅ | ✅ | |
| Little-endian byte order | ✅ | ✅ | |
| Big-endian byte order | ✅ | ✅ | |
| Arbitrary HDF5 floating-point layouts | 🟡 | 🟡 | Raw access only. No numeric conversion. |

#### Native complex datatypes

`Datatype::Complex` represents the homogeneous rectangular form introduced in HDF5 2.0.
Dataset and attribute inspection preserve the floating-point component layout and byte order.
The parser returns a structured format error for other complex forms.

Dataset raw reads and writes preserve both components. Scalar numeric readers return
`TypeMismatch`. Fixed-size datasets and attributes can be repacked, including complex types
nested in arrays or compounds. Raw append requires little-endian components. Repacking
variable-length complex attributes returns `RepackUnsupported`.

The MATLAB complex helpers use the class-6 compound `{real, imag}` convention.

### Compound datatypes

| Feature | Read | Write | Repack | Detail |
|---|:---:|:---:|:---:|---|
| Explicit member offsets | ✅ | ✅ | ✅ | |
| Padding between members | ✅ | ✅ | ✅ | |
| Nested compound types | ✅ | ✅ | ✅ | |
| Array members | ✅ | ✅ | ✅ | |
| Variable-length members | ✅ | ❌ | 🟡 | Lossless, re-encodable filters. Object references impose additional restrictions. |
| Object-reference members | ✅ | ❌ | 🟡 | For allocated storage, repack supports contiguous and compact layouts, and rejects chunked, filtered, and resizable layouts. |

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

| Feature | Read | Write | Repack | Detail |
|---|:---:|:---:|:---:|---|
| Variable-length UTF-8 string | ✅ | ✅ | ✅ | |
| Variable-length ASCII string | ✅ | ✅ | ✅ | |
| Chunked variable-length string | ✅ | ✅ | ✅ | |
| Filtered variable-length string | ✅ | ✅ | ✅ | |
| Resizable variable-length string | ✅ | ✅ | ✅ | |
| Variable-length sequence | ✅ | ❌ | 🟡 | Lossless, re-encodable filters. No VLEN or reference base types. |
| Variable-length member in a compound type | ✅ | ❌ | 🟡 | Lossless, re-encodable filters. Object references impose additional restrictions. |

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
| SWMR writer | 🟡 | v3 superblock. Unfiltered rank-one Extensible Array datasets. Chunk-aligned length and appends. |

### File-space management

| Feature | Read | Configure | Detail |
|---|:---:|:---:|---|
| `H5F_FSPACE_STRATEGY_FSM_AGGR` | ✅ | ✅ | Aggregation is unavailable in SWMR sessions. |
| `H5F_FSPACE_STRATEGY_PAGE` | ✅ | ✅ | Without persistence: mirrored staged edits only, with supported attribute storage. Object references, page buffering, and space reuse are unsupported. |
| `H5F_FSPACE_STRATEGY_AGGR` | ✅ | ✅ | Deleted object storage remains unavailable for reuse. |
| `H5F_FSPACE_STRATEGY_NONE` | ✅ | ✅ | Freed storage remains allocated. |
| Aggregators | ✅ | 🟡 | `FsmAggr` and `Aggr` only. Disabled in SWMR under `FsmAggr`. |
| Persistent free-space managers | ✅ | ✅ | |
| Free-space threshold | ✅ | ✅ | Page tails are tracked regardless of threshold. |
| File-space page size | ✅ | ✅ | Writing and editing require a size from 512 bytes to 1 GiB. |
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
| Shared object-header messages | ✅ | ⬆️ | Edits that invalidate shared-message tables are rejected. |
| File Space Info v0 | ✅ | ⬆️ | New messages use v1. |
| File Space Info v1 | ✅ | ✅ | |
| Metadata checksums | ✅ | ✅ | General validation requires the `checksum` feature. |

Shared object-header messages are supported for datatypes, dataspaces, fill values, filter pipelines, and attributes.

### Chunk indexes

| Index | Read data | Emit | Enumerate | Detail |
|---|:---:|:---:|:---:|---|
| Version 1 B-tree | ✅ | ⬆️ | ✅ | Rewritten with a version 4 chunk index. |
| Single chunk | ✅ | ✅ | ✅ | |
| Implicit | ✅ | ⬆️ | ✅ | Rewritten with another version 4 chunk index. |
| Fixed array | ✅ | ✅ | ✅ | |
| Extensible array | ✅ | ✅ | ✅ | |
| Version 2 B-tree | ✅ | ❌ | ✅ | Index mutation is unsupported. |

`Enumerate` refers to `Dataset::chunks()`.

### Unknown metadata

| Feature | Read-only | Read-write | Repack | Detail |
|---|:---:|:---:|:---:|---|
| Unknown optional object-header message | ✅ | 🟡 | 🟡 | Write and repack support depends on the message flags. |
| Unknown message mandatory for readers | ❌ | ❌ | ❌ | |
| Unknown message mandatory for writers only | ✅ | ❌ | 🟡 | Repack can be configured to reject these messages. |
