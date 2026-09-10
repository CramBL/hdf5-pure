This page covers working with HDF5 files that are too large to buffer in memory, in both directions. Reading comes first: [`File::open`](crate::File::open) loads the whole file into RAM, while [`File::open_streaming`](crate::File::open_streaming) fetches metadata and dataset chunks from disk on demand, so peak memory tracks the data you actually read and not the size of the file. [Writing without buffering](#writing-without-buffering) covers the other direction.

## Why stream

[`File::open(path)`](crate::File::open) reads the entire file into memory before you touch any dataset. That is simple and fast for files that comfortably fit in RAM, but it does not scale to files that exceed available memory, for example a multi-gigabyte file produced on a 32-bit host where it exceeds the address space.

[`File::open_streaming(path)`](crate::File::open_streaming) opens the same file with a lazy backing store. It fetches metadata and dataset chunks from the file as they are needed and never holds the entire file in memory at once. Peak memory tracks what you actually read: one dataset, decompressed, with its chunks fetched on demand, plus the metadata being parsed.

Chunks that lie next to each other on disk are fetched together, in reads of at most 256 KiB. This is what makes a file written a row at a time readable at a sensible speed: such a file carries one small chunk per row, so fetching them one at a time costs thousands of reads where a handful would do. A recording of 73 datasets over 280 rows holds 20,440 chunks of 32 bytes, which the streaming reader fetches in 73 reads. A read never fetches bytes outside the chunks it needs, so what this trades is read count, not read volume. A chunk larger than 256 KiB is read on its own.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("huge.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("signal").with_f64_data(&[1.0, 2.0, 3.0]).with_chunks(&[2]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open_streaming(&path)?;
let ds = file.dataset("signal")?;
let values = ds.read_f64()?;  // only this dataset's chunks are read
# assert_eq!(values, vec![1.0, 2.0, 3.0]);
# Ok::<(), hdf5_pure::Error>(())
```

The reading API is identical to [`File::open`](crate::File::open), and only the backing store differs. Everything you can do with an in-memory file (see [Reading datasets](crate::_guide::reading)) applies here too.

[`File::open_streaming`](crate::File::open_streaming) needs the `std` filesystem, so a `no_std` build cannot reach it. [Cargo features](crate#cargo-features) has the feature matrix.

## Streaming from something that is not a path

[`File::from_source(source)`](crate::File::from_source) opens the same lazy backing store over anything that implements [`Source`](crate::Source), for a caller whose bytes are not a file: an object store addressed by HTTP range request, a WebAssembly guest that receives byte ranges from its host and has no filesystem at all, a decrypting layer.

[`Source`](crate::Source) has two methods, the total length and the bytes at an absolute offset, which is all the reader ever needs of a file:

```rust
# fn image() -> Vec<u8> {
#     let mut builder = hdf5_pure::FileBuilder::new();
#     builder.create_dataset("signal").with_f64_data(&[1.0, 2.0, 3.0]);
#     builder.finish().unwrap()
# }
# fn fetch_len() -> u64 {
#     u64::try_from(image().len()).unwrap()
# }
# fn fetch(offset: u64, len: usize) -> Result<Vec<u8>, String> {
#     let start = usize::try_from(offset).map_err(|_| "offset past this platform".to_string())?;
#     let end = start.checked_add(len).ok_or_else(|| "offset overflow".to_string())?;
#     image().get(start..end).map(<[u8]>::to_vec).ok_or_else(|| "read past the end".to_string())
# }
use hdf5_pure::{File, FormatError, Source};

struct Remote {
    len: u64,
}

impl Source for Remote {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        // However the bytes arrive: a range request, a host call, a decrypting
        // layer. Fill the whole request or fail — a short read is an error.
        let bytes = fetch(offset, buf.len()).map_err(FormatError::Source)?;
        buf.copy_from_slice(&bytes);
        Ok(())
    }
}

let file = File::from_source(Remote { len: fetch_len() })?;
let values = file.dataset("signal")?.read_f64()?;
# assert_eq!(values, vec![1.0, 2.0, 3.0]);
# Ok::<(), hdf5_pure::Error>(())
```

A `Read + Seek` needs no implementation of its own: [`ReadSeekSource`](crate::ReadSeekSource) wraps one. (Doing that to a `std::fs::File` is exactly [`File::open_streaming`](crate::File::open_streaming), which wraps it for you.)

Two things are worth knowing before pointing this at a remote store:

- **Turn the metadata cache on**, through [`File::from_source_with_options`](crate::File::from_source_with_options). It is off by default, and without one every read a parser makes is a round trip. See [`MetadataCacheConfig`](crate::MetadataCacheConfig).
- **A file the superblock marks as held by a writer is rejected here too**, with [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse). Some of the recoveries a path open would offer do not apply: [`File::open_swmr`](crate::File::open_swmr) and [`File::clear_swmr_flag`](crate::File::clear_swmr_flag) both need a path. What a source can take is [`FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)`](crate::FileAccessProperties::with_write_mark_policy), which reads a file marked by a non-SWMR writer that has flushed (see [Reading](crate::_guide::reading)), or [`File::from_bytes`](crate::File::from_bytes) with the whole file in memory. The error says which.

## What streaming supports

Dataset reads are fully supported across every storage layout:

| Layout | Supported when streaming |
| --- | --- |
| Contiguous | Yes |
| Compact | Yes |
| Chunked (B-tree v1 index) | Yes |
| Chunked (fixed array index) | Yes |
| Chunked (extensible array index) | Yes |

The streaming backend resolves both group forms (v2 and v1 symbol-table) and reads compact, dense, shared, and variable-length attributes, same as [`File::open`](crate::File::open). The remaining differences:

- [`File::as_bytes`](crate::File::as_bytes) returns an empty slice, since there is no whole-file buffer, and [`persisted_free_space`](crate::File::persisted_free_space) returns an empty vector, since the free-space-manager blocks stay unread.
- A streaming file cannot be the **source** of a cross-file [`copy_from`](crate::File::copy_from): that copy requires a buffered source.
- Chunk decompression is sequential.

Streaming opens are read-only. To **append** to a file with the same bounded-memory discipline, open it with [`File::open_rw`](crate::File::open_rw), which edits a latest-format file bounded, the read-write sibling of [`open_streaming`](crate::File::open_streaming), sharing this backend's read capabilities and the [`FileAccessProperties`](crate::FileAccessProperties) cache budgets below. [Bounded-memory appends](crate::_guide::editing#bounded-memory-appends) has the detail.

## Reading a large dataset a window at a time

The whole-dataset reads above materialize one dataset in full. When a single dataset is itself too large to hold decompressed, read it in **row windows**: [`read_raw_rows(start, count)`](crate::Dataset::read_raw_rows) and the typed [`read_f64_rows`](crate::Dataset::read_f64_rows) / … / [`read_string_rows`](crate::Dataset::read_string_rows) decode only the leading-dimension rows `[start, start + count)`, touching only the chunks that window overlaps. Peak memory then tracks the window (plus one chunk), not the dataset.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("huge.h5");
# let samples: Vec<f64> = (0..4_096).map(f64::from).collect();
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("signal").with_f64_data(&samples).with_chunks(&[512]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open_streaming(&path)?;
let ds = file.dataset("signal")?;
let rows = ds.shape()?[0];

// Process a million rows at a time, never holding the whole dataset.
for start in (0..rows).step_by(1_000_000) {
    let window = ds.read_f64_rows(start, 1_000_000)?;
    // ... process `window` ...
    let _ = window;
}
# Ok::<(), hdf5_pure::Error>(())
```

The window is clamped to the dataset, so the final short window needs no special-casing. See [Reading a row window](crate::_guide::reading#reading-a-row-window) for the full method list.

## Tuning retained memory

[`File::open_streaming_with_options(path, FileAccessProperties)`](crate::File::open_streaming_with_options) bounds the memory the streaming backend retains. [`FileAccessProperties::new()`](crate::FileAccessProperties::new) returns the crate's default access behavior, and its builder methods layer on two independent caches.

[`MetadataCacheConfig`](crate::MetadataCacheConfig) mirrors the memory-budget role of `H5Pset_mdc_config`: it caps the bytes retained for parsed metadata reads. [`MetadataCacheConfig::new(max_bytes)`](crate::MetadataCacheConfig::new) sets the total byte budget, and [`.with_max_entry_bytes(...)`](crate::MetadataCacheConfig::with_max_entry_bytes) caps the size of any single cached metadata read so one large heap or index block cannot monopolize the cache. It is disabled until a caller turns it on, which pays off on a file holding many datasets: opening one walks the root group's link storage, and every dataset repeats that walk. Entries are indexed and never searched, so raising the budget costs memory alone.

[`ChunkCacheConfig`](crate::ChunkCacheConfig) mirrors the raw-data chunk-cache settings from `H5Pset_cache`. [`ChunkCacheConfig::from_h5p_cache(rdcc_nslots, rdcc_nbytes)`](crate::ChunkCacheConfig::from_h5p_cache) builds one directly from the familiar HDF5 slot count and byte budget. It controls decompressed chunk data and whether parsed chunk indexes are retained between repeated reads of the same dataset.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("huge.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("signal").with_f64_data(&[1.0, 2.0, 3.0]).with_chunks(&[2]);
# builder.write(&path)?;
use hdf5_pure::{ChunkCacheConfig, File, FileAccessProperties, MetadataCacheConfig};

let access = FileAccessProperties::new()
    .with_metadata_cache(MetadataCacheConfig::new(8 * 1024 * 1024).with_max_entry_bytes(64 * 1024))
    .with_chunk_cache(ChunkCacheConfig::from_h5p_cache(521, 256 * 1024));
let file = File::open_streaming_with_options(&path, access)?;
# assert_eq!(file.dataset("signal")?.read_f64()?, vec![1.0, 2.0, 3.0]);
# Ok::<(), hdf5_pure::Error>(())
```

The chunk cache configured here is the file-wide default and applies to every dataset opened from this file. The metadata cache reaches the opens that read metadata lazily, the streaming opens here and a bounded [`File::open_rw`](crate::File::open_rw), since an in-memory open already holds the whole file in one buffer.

| Config | HDF5 analogue | Controls |
| --- | --- | --- |
| [`MetadataCacheConfig`](crate::MetadataCacheConfig) | `H5Pset_mdc_config` (memory budget) | Bytes retained for parsed metadata reads |
| [`ChunkCacheConfig`](crate::ChunkCacheConfig) (file-wide) | `H5Pset_cache` raw-data settings | Decompressed chunk bytes and retained chunk indexes, as the default for all datasets |
| [`ChunkCacheConfig`](crate::ChunkCacheConfig) (per dataset) | `H5Pset_chunk_cache` | Same, overridden for one dataset |

### Per-dataset overrides

To override the chunk cache for a single dataset, open it with [`dataset_with_options(name, DatasetAccessProperties)`](crate::File::dataset_with_options). This is the analogue of HDF5's per-dataset access property list (`H5Pset_chunk_cache`). The override replaces the file-wide default for that one dataset, and other datasets keep the default. A dataset that is read once front-to-back, for instance, gains nothing from caching its decompressed chunks, so you can disable the cache with [`ChunkCacheConfig::disabled()`](crate::ChunkCacheConfig::disabled):

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("data.h5");
# let samples: Vec<f64> = (0..256).map(f64::from).collect();
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("scan").with_f64_data(&samples).with_chunks(&[64]);
# builder.write(&path)?;
use hdf5_pure::{ChunkCacheConfig, DatasetAccessProperties, File};

let file = File::open(&path)?;
// This dataset is read once front-to-back: skip caching its decompressed chunks.
let dapl = DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled());
let ds = file.dataset_with_options("scan", dapl)?;
let values = ds.read_f64()?;
# assert_eq!(values.len(), 256);
# Ok::<(), hdf5_pure::Error>(())
```

[`dataset_with_options`](crate::File::dataset_with_options) is available on both [`File`](crate::File) and [`Group`](crate::Group). [`Dataset::chunk_cache_config()`](crate::Dataset::chunk_cache_config) reports the effective [`ChunkCacheConfig`](crate::ChunkCacheConfig) for an opened dataset (the analogue of `H5Pget_chunk_cache`): the per-dataset override when one was supplied, otherwise the file-wide default.

[`DatasetAccessProperties::new()`](crate::DatasetAccessProperties::new) inherits every file-wide access default, so you only set what you want to change.

## Confirming cache behavior

### The chunk cache

To confirm a cache is behaving as configured, [`Dataset::chunk_cache_stats()`](crate::Dataset::chunk_cache_stats) returns a read-only [`ChunkCacheStats`](crate::ChunkCacheStats) snapshot. Occupancy is a point-in-time view: whether the parsed index is loaded ([`index_loaded()`](crate::ChunkCacheStats::index_loaded)), how many decompressed chunks are retained ([`cached_chunks()`](crate::ChunkCacheStats::cached_chunks)), and how many bytes they occupy ([`cached_bytes()`](crate::ChunkCacheStats::cached_bytes)). The counters beside it are cumulative since the handle was opened, or since the last [`reset_chunk_cache_stats()`](crate::Dataset::reset_chunk_cache_stats).

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("data.h5");
# let samples: Vec<f64> = (0..256).map(f64::from).collect();
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("signal").with_f64_data(&samples).with_chunks(&[64]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let ds = file.dataset("signal")?;
let _ = ds.read_f64()?;
let stats = ds.chunk_cache_stats();
// "signal" here is a chunked dataset, so chunks are retained for reuse;
// a contiguous or compact dataset has no chunk cache and reports zero.
assert!(stats.cached_chunks() > 0);
# Ok::<(), hdf5_pure::Error>(())
```

**Read [`rejections()`](crate::ChunkCacheStats::rejections) and [`evictions()`](crate::ChunkCacheStats::evictions) together, not one or the other.** Which of the two moves depends on how the dataset is read, and each is structurally zero on the other's path:

| read | budget signal | stays zero |
|---|---|---|
| whole ([`read_f64`](crate::Dataset::read_f64) and friends) | [`rejections()`](crate::ChunkCacheStats::rejections) | [`evictions()`](crate::ChunkCacheStats::evictions) |
| row window (`read_*_rows`) | [`evictions()`](crate::ChunkCacheStats::evictions) | [`rejections()`](crate::ChunkCacheStats::rejections) |

A whole read visits each of its chunks exactly once, so a full cache rejects the chunks that follow and keeps the ones it has already placed, which no later chunk of that read would have been served from anyway: a dataset eight times the budget reports seven eighths of its chunks rejected and no evictions at all. Reading it again hits the retained chunks and rejects the rest, so a settled cache costs nothing to keep. A row window keeps the plain LRU rule, because the chunk its successor needs is the one it finished on. Either counter climbing while [`hit_rate()`](crate::ChunkCacheStats::hit_rate) stays low is the same finding.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("data.h5");
# let samples: Vec<f64> = (0..256).map(f64::from).collect();
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("signal").with_f64_data(&samples).with_chunks(&[64]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let ds = file.dataset("signal")?;

// The reads that fill a cache miss by definition, so a rate measured over the
// whole run charges the steady state for the warm-up.
let _ = ds.read_f64()?;
ds.reset_chunk_cache_stats();
let _ = ds.read_f64()?;

let stats = ds.chunk_cache_stats();
if stats.rejections() > 0 || stats.evictions() > 0 {
    // The working set did not fit: raise max_slots, max_bytes, or both.
}
if stats.oversize_chunks() > 0 {
    // One chunk is larger than the whole byte budget; more slots will not help.
}
println!("{:?} over {} lookups", stats.hit_rate(), stats.lookups());
# Ok::<(), hdf5_pure::Error>(())
```

Two figures need a word of care. [`hit_rate()`](crate::ChunkCacheStats::hit_rate) returns `None` before any lookup, where a cache that missed everything and a cache that was disabled both report `0.0`. Read it beside [`chunk_cache_config()`](crate::Dataset::chunk_cache_config). And [`invalidations()`](crate::ChunkCacheStats::invalidations) counts the whole session: a commit through *any* handle advances the file's content revision, so every dataset handle drops its chunks. A figure approaching [`misses()`](crate::ChunkCacheStats::misses) means the session is rewriting what it caches, and a larger budget will not change that.

### The metadata cache

A cache budget is a number chosen before a single read has happened. [`File::metadata_cache_stats`](crate::File::metadata_cache_stats) reports what it bought, so the next number is measured and not guessed. It is the counterpart of `H5Fget_mdc_hit_rate` and `H5Fget_mdc_size`, and returns `None` when the open has no metadata cache to report on.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("huge.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("a").with_f64_data(&[1.0, 2.0]);
# builder.create_dataset("b").with_f64_data(&[3.0, 4.0]);
# builder.write(&path)?;
use hdf5_pure::{File, FileAccessProperties, MetadataCacheConfig};

let access = FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(8 << 20));
let file = File::open_streaming_with_options(&path, access)?;

// The reads that fill a cache miss by definition, so a rate measured over the
// whole run charges the steady state for the warm-up.
for name in file.root().datasets()? {
    let _ = file.dataset(&name)?.read_raw()?;
}
file.reset_metadata_cache_stats();

for name in file.root().datasets()? {
    let _ = file.dataset(&name)?.read_raw()?;
}
let stats = file.metadata_cache_stats().unwrap();
println!("{:?} over {} reads, {} of {} bytes held", stats.hit_rate(), stats.reads(), stats.bytes(), 8 << 20);
# Ok::<(), hdf5_pure::Error>(())
```

Which figure to read depends on the question:

- **[`hit_rate`](crate::MetadataCacheStats::hit_rate)** is the headline: the fraction of eligible reads served without touching the file. `None` means no eligible read has happened yet, which is not the same as a cache that has missed everything.
- **[`evictions`](crate::MetadataCacheStats::evictions)** is what says a larger budget would help. A disappointing hit rate *with* evictions is a budget too small for the working set. The same hit rate with none is a workload that reads each piece of metadata once, and raising the budget will not change it.
- **[`oversize_reads`](crate::MetadataCacheStats::oversize_reads)** counts reads turned away, for exceeding [`max_entry_bytes`](crate::MetadataCacheConfig::max_entry_bytes) or the budget itself, before the cache saw them. A file with large fractal heaps or index blocks can be missing from the cache entirely for this reason while the hit rate looks healthy.
- **[`invalidations`](crate::MetadataCacheStats::invalidations)** applies to a read-write session: entries dropped because a write overlapped them. Approaching the miss count, it means the session is rewriting the metadata it is caching.

[`reset_metadata_cache_stats`](crate::File::reset_metadata_cache_stats) clears the counters and evicts nothing, so occupancy carries across it.

Set the budget generously. [The metadata cache](crate::FileAccessProperties#the-metadata-cache-h5pset_mdc_config) goes through `H5AC_cache_config_t`'s adaptive-resize policy field by field, and why none of it is modeled.

## Writing without buffering

[`FileBuilder::finish()`](crate::FileBuilder::finish) returns the assembled file, so writing an N-byte file costs N bytes of output on top of whatever the data already cost. [`FileBuilder::finish_to(w)`](crate::FileBuilder::finish_to) writes the same bytes onto any `io::Write` instead, and [`FileBuilder::write(path)`](crate::FileBuilder::write) is [`finish_to`](crate::FileBuilder::finish_to) onto a file.

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0, 3.0]);

let mut sink: Vec<u8> = Vec::new();
builder.finish_to(&mut sink)?;
# assert!(hdf5_pure::is_hdf5_bytes(&sink));
# Ok::<(), hdf5_pure::Error>(())
```

This works because the writer computes every object's address (object headers, data blocks, indexes) *before* it emits a byte, then writes the file in ascending-address order. It never seeks back to patch an address, which is what a backpatching writer would have to do. So the destination can be anything that accepts bytes forward-only: a socket, a pipe, a compressing wrapper, a hash. Both are the same internal assembly pass against different sinks, a `Vec<u8>` for [`finish`](crate::FileBuilder::finish) and the caller's writer for [`finish_to`](crate::FileBuilder::finish_to), so they are byte-identical by construction.

**A failure partway through leaves whatever was already written on the sink.** With a non-seekable destination there is nothing to roll back, so a caller who needs all-or-nothing writes to a temporary path and renames on success.

### Userblock content

A file with a userblock needs its header bytes to be part of what the writer emits, since a streaming write has nothing left to patch by the time it returns. [`with_userblock_content`](crate::FileBuilder::with_userblock_content) supplies them up front:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("wrapped.h5");
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.with_userblock(512);
builder.with_userblock_content(b"my wrapper format's header");
builder.create_dataset("x").with_f64_data(&[1.0]);
builder.write(&path)?;
# assert_eq!(&std::fs::read(&path)?[..26], b"my wrapper format's header");
# Ok::<(), hdf5_pure::Error>(())
```

The rest of the region stays zero-filled, and content longer than the userblock is rejected with [`FormatError::UserblockContentTooLarge`](crate::FormatError::UserblockContentTooLarge), never allowed to displace the superblock. This is how the MATLAB v7.3 writer emits its 512-byte header, as the MATLAB interop guide shows.

### Datasets that never become resident

[`finish_to`](crate::FileBuilder::finish_to) removes the assembled file from peak memory, but not the data: [`with_f64_data(&values)`](crate::DatasetBuilder::with_f64_data) still copies the slice into the builder. Two paths avoid that too.

**Repacking** an existing file streams each chunk from the source to the destination, verbatim and one at a time, without decoding or re-encoding it. [`repack`](crate::repack()) is the entry point, and the [repack](crate::_guide::repack) guide walks a repack end to end.

**Producing** a dataset's bytes at write time is available on the MATLAB writer as [`MatBuilder::write_blocks`](crate::mat::MatBuilder::write_blocks), which takes a [`DataProducer`](crate::mat::producer::DataProducer) the writer calls once per block during emission. Layout works from the shape alone, so the producer is never called before the write begins. Paired with [`MatBuilder::finish_to`](crate::mat::MatBuilder::finish_to), a `.mat` of any size is written in about one block of memory. The full API is in the MATLAB interop guide, with why it is uncompressed-only and which array shape an acquisition should choose.

## Related topics

- [Reading datasets](crate::_guide::reading) has the dataset read API shared between in-memory and streaming opens.
- [Writing files](crate::_guide::writing) walks the [`FileBuilder`](crate::FileBuilder) workflow these output paths finish.
- Reading string datasets is the subject of the [variable-length strings](crate::_guide::vlen_strings) guide.
- [Cargo features](crate#cargo-features) lists the `std` feature requirement.
