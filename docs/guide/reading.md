This page covers opening HDF5 files, navigating their group hierarchy, and reading datasets and attributes back into Rust. The reading API is the same regardless of how a file is opened, so the patterns here apply equally to in-memory, on-disk, streaming, and SWMR reads.

The [`quickstart`](https://github.com/CramBL/hdf5-pure/blob/main/examples/quickstart.rs) example builds a file in memory and reads it back, doubling as a self-check. Run it with:

```console
$ cargo run --example quickstart
```

The [`groups_and_attributes`](https://github.com/CramBL/hdf5-pure/blob/main/examples/groups_and_attributes.rs) example walks a nested hierarchy. Run it with `cargo run --example groups_and_attributes`.

## Opening a file

[`File`](crate::File) is the entry point for reading. These constructors produce one, and each returns a value with an identical reading API:

| Constructor | Source | Notes |
|---|---|---|
| [`File::open(path)`](crate::File::open) | A file on disk | Reads the whole file into memory. Requires the `std` filesystem. |
| [`File::from_bytes(bytes)`](crate::File::from_bytes) | An in-memory `Vec<u8>` | The in-memory path, for WebAssembly and anywhere else without a filesystem. |
| [`File::open_streaming(path)`](crate::File::open_streaming) | A file on disk, read on demand | Fetches metadata and chunks lazily and never buffers the whole file. See [Streaming](crate::_guide::streaming). |
| [`File::open_swmr(path)`](crate::File::open_swmr) | A file being appended to | Re-readable with [`refresh()`](crate::File::refresh) to pick up new data, the reader half of SWMR. |
| [`File::from_source(source)`](crate::File::from_source) | Any [`Source`](crate::Source), bytes with no path | The streaming reader over an object store, a sandboxed host, anything that reports a length and returns the bytes at an offset. See [Streaming](crate::_guide::streaming). |

[`File::open`](crate::File::open), [`File::open_streaming`](crate::File::open_streaming) and [`File::from_source`](crate::File::from_source) reject a file whose superblock marks it as held by a writer, with [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse). `H5Fopen` checks the same byte. Which reader gets at such a file depends on which mark it carries:

- **both bits, a SWMR writer**: follow it with [`File::open_swmr`](crate::File::open_swmr), which also picks up what the writer appends later.
- **the write bit alone**, which is what a [page-buffered](crate::_guide::editing#letting-pages-live-longer) writer leaves: no SWMR reader can follow it, so read a snapshot through any `*_with_options` open under [`FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)`](crate::FileAccessProperties::with_write_mark_policy). That asserts the writer has flushed, through [`File::sync`](crate::File::sync) or a close, which this crate cannot check for you.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("live.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# builder.write(&path)?;
use hdf5_pure::{File, FileAccessProperties, WriteMarkPolicy};

let props = FileAccessProperties::new().with_write_mark_policy(WriteMarkPolicy::AllowSnapshot);
let file = File::open_with_options(&path, props)?;
# assert_eq!(file.dataset("temperature")?.shape()?, vec![3]);
# Ok::<(), hdf5_pure::Error>(())
```

The opt-in unlocks nothing else: a SWMR pair still belongs to [`File::open_swmr`](crate::File::open_swmr), and [`File::open_rw`](crate::File::open_rw) rejects such a file whatever it says. [`File::from_bytes`](crate::File::from_bytes) does not consult the byte at all, at the cost of a copy of the file. [`File::clear_swmr_flag`](crate::File::clear_swmr_flag) recovers a file a writer left flagged when it exited, and the [SWMR](crate::_guide::swmr) guide walks the writer side that raises the flag.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let ds = file.dataset("temperature")?;

println!("shape: {:?}", ds.shape()?);    // [3]
println!("data:  {:?}", ds.read_f64()?);  // [22.5, 23.1, 21.8]
# Ok::<(), hdf5_pure::Error>(())
```

To read from bytes that were never written to disk, for example the output of [`FileBuilder::finish`](crate::FileBuilder::finish) (see [Writing Files](crate::_guide::writing)):

```rust
use hdf5_pure::{File, FileBuilder};

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);
let bytes = builder.finish()?;

let file = File::from_bytes(bytes)?;
let x = file.dataset("x")?.read_f64()?;
# assert_eq!(x, vec![1.0, 2.0]);
# Ok::<(), hdf5_pure::Error>(())
```

[`File::open_with_options`](crate::File::open_with_options), [`File::from_bytes_with_options`](crate::File::from_bytes_with_options) and [`File::open_swmr_with_options`](crate::File::open_swmr_with_options) accept a [`FileAccessProperties`](crate::FileAccessProperties) whose chunk-cache budget bounds the decompressed chunks a dataset retains. [`File::open_streaming_with_options`](crate::File::open_streaming_with_options) and [`File::from_source_with_options`](crate::File::from_source_with_options) take the metadata budget as well, since those two are the opens that read metadata lazily. See [Streaming](crate::_guide::streaming) for [`MetadataCacheConfig`](crate::MetadataCacheConfig) and [`ChunkCacheConfig`](crate::ChunkCacheConfig).

## Opening datasets

[`File::dataset(path)`](crate::File::dataset) resolves a dataset by its full path from the root, returning a [`Dataset`](crate::Dataset):

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# let mut sensors = builder.create_group("sensors");
# let mut imu = sensors.create_group("imu");
# imu.create_dataset("accel").with_f64_data(&[0.0, 0.0, 9.81]);
# sensors.add_group(imu.finish());
# builder.add_group(sensors.finish());
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let accel = file.dataset("sensors/imu/accel")?;
# assert_eq!(accel.read_f64()?, vec![0.0, 0.0, 9.81]);
# Ok::<(), hdf5_pure::Error>(())
```

A dataset can also be opened by name relative to its parent group via [`Group::dataset(name)`](crate::Group::dataset) (see [Navigating groups](#navigating-groups-and-attributes) below). To override the chunk cache for a single dataset, use [`File::dataset_with_options(path, DatasetAccessProperties)`](crate::File::dataset_with_options).

### Inspecting shape and datatype

[`Dataset::shape()`](crate::Dataset::shape) returns the dimensions as a `Vec<u64>`. Two accessors describe the datatype: [`Dataset::dtype()`](crate::Dataset::dtype) returns a simplified [`DType`](crate::DType) classification, while [`Dataset::datatype()`](crate::Dataset::datatype) returns the full [`Datatype`](crate::Datatype) with exact field offsets and layout, which is what a compound type needs. The [compound types](crate::_guide::compound_types) guide works through one.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let ds = file.dataset("temperature")?;

println!("shape: {:?}", ds.shape()?);
println!("dtype: {:?}", ds.dtype()?);
# Ok::<(), hdf5_pure::Error>(())
```

### Chunking, filters, and append eligibility

Four accessors describe a dataset's storage layout without reading any data. They also report whether a dataset can be grown in place with [`Dataset::append_staged`](crate::Dataset::append_staged) or a [streaming append](crate::_guide::editing#streaming-appends) before you attempt the append:

| Method | Returns |
|---|---|
| [`Dataset::is_chunked()`](crate::Dataset::is_chunked) | `bool`, `true` for chunked storage. A filtered dataset is always chunked |
| [`Dataset::maxshape()`](crate::Dataset::maxshape) | `Option<Vec<u64>>`, the maximum dimensions, with an unlimited axis reported as `u64::MAX`. `None` for a fixed-shape dataset |
| [`Dataset::chunk_shape()`](crate::Dataset::chunk_shape) | `Option<Vec<u64>>`, the chunk dimensions, one per rank. `None` for a dataset that is not chunked |
| [`Dataset::filters()`](crate::Dataset::filters) | `Vec<u16>`, the HDF5 filter IDs in pipeline order (1 = deflate, 2 = shuffle, 3 = fletcher32, 6 = scale-offset, 32000 = LZF). Empty for an unfiltered dataset |

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_i32_data(&[1, 2, 3])
#     .with_maxshape(&[u64::MAX])
#     .with_chunks(&[512]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let ds = file.dataset("samples")?;

// Broadly appendable: chunked and unlimited along axis 0. The full rules
// (rank 1, Extensible-Array index, single hard link) are in the editing guide.
let appendable = ds.is_chunked()
    && matches!(ds.maxshape()?.as_deref(), Some([u64::MAX, ..]));
# assert!(appendable);
# Ok::<(), hdf5_pure::Error>(())
```

## Reading dataset data

### Typed reads

The `read_*` family delivers a dataset's elements as a flat `Vec<T>` of the type the method names. Each method coerces the stored values to that type, so a call says how you want the data delivered and asserts nothing about the stored datatype.

| Method | Result |
|---|---|
| [`read_f64`](crate::Dataset::read_f64) | `Vec<f64>` |
| [`read_f32`](crate::Dataset::read_f32) | `Vec<f32>` |
| [`read_i8`](crate::Dataset::read_i8) / [`read_i16`](crate::Dataset::read_i16) / [`read_i32`](crate::Dataset::read_i32) / [`read_i64`](crate::Dataset::read_i64) | signed integers |
| [`read_u8`](crate::Dataset::read_u8) / [`read_u16`](crate::Dataset::read_u16) / [`read_u32`](crate::Dataset::read_u32) / [`read_u64`](crate::Dataset::read_u64) | unsigned integers |
| [`read_string`](crate::Dataset::read_string) | `Vec<String>` (fixed- and variable-length) |
| [`read_raw`](crate::Dataset::read_raw) | `Vec<u8>` of the complete unfiltered record bytes |

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let values = file.dataset("temperature")?.read_f64()?;
# assert_eq!(values, vec![22.5, 23.1, 21.8]);
# Ok::<(), hdf5_pure::Error>(())
```

A typed read costs about what its result costs. The numeric readers decode the dataset a row window at a time, so what stands beside the returned `Vec<T>` is one window of stored bytes, roughly a mebibyte, and never a second copy of the whole dataset. [`read_u8`](crate::Dataset::read_u8) and [`read_raw`](crate::Dataset::read_raw) hand back the stored bytes themselves and have nothing to decode.

### Generic reads

[`Dataset::read::<T>()`](crate::Dataset::read) is the generic counterpart to the typed `read_*` methods, bounded by the sealed [`H5Element`](crate::H5Element) trait (implemented for `f32`/`f64` and the 8/16/32/64-bit signed and unsigned integers). It lets you write code generic over the element type. Like [`read_f64`](crate::Dataset::read_f64), it requests delivery as `T` and coerces, so pick `T` to match the stored type for a lossless read.

```rust
use hdf5_pure::{File, FileBuilder, H5Element, Error};

fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, Error> {
    file.dataset(name)?.read::<T>()
}

let mut fb = FileBuilder::new();
fb.create_dataset("counts").with_data(&[1u32, 2, 3]);
let file = File::from_bytes(fb.finish()?)?;

let counts: Vec<u32> = load(&file, "counts")?;  // [1, 2, 3]
# assert_eq!(counts, vec![1, 2, 3]);
# Ok::<(), hdf5_pure::Error>(())
```

The [generic I/O](crate::_guide::generic_io) guide has the writing side. [`read_array`](crate::Dataset::read_array) and [`read_array_dyn`](crate::Dataset::read_array_dyn) deliver an N-dimensional dataset as an `ndarray` array under the `ndarray` feature, with the shapes and the feature gate in the [ndarray](crate::_guide::ndarray) guide.

### String reads

[`Dataset::read_string`](crate::Dataset::read_string) reads both fixed-length and variable-length HDF5 string datasets into a `Vec<String>`. To bound variable-length payload allocation before reading, or to consume strings one at a time, use [`read_vlen_strings(VlenStringReadOptions)`](crate::Dataset::read_vlen_strings) or [`visit_vlen_strings`](crate::Dataset::visit_vlen_strings). The bounding options are set out in the [variable-length strings](crate::_guide::vlen_strings) guide.

### Raw and compound reads

[`Dataset::read_raw`](crate::Dataset::read_raw) returns the complete unfiltered record bytes, and [`Dataset::read_compound::<T>()`](crate::Dataset::read_compound) decodes compound (struct-like) records. Their encodings are in the [compound types](crate::_guide::compound_types) guide and the data types reference.

### Reading a row window

The `read_*` methods above deliver a whole dataset. A **row window**, the leading-dimension rows `[start, start + count)`, is read without materializing the rest through [`read_raw_rows(start, count)`](crate::Dataset::read_raw_rows) and the typed [`read_f64_rows`](crate::Dataset::read_f64_rows) / [`read_f32_rows`](crate::Dataset::read_f32_rows) / [`read_i8_rows`](crate::Dataset::read_i8_rows) … [`read_u64_rows`](crate::Dataset::read_u64_rows) / [`read_string_rows`](crate::Dataset::read_string_rows) counterparts. Each decodes exactly like its whole-dataset form, so a window is that whole read sliced to that row range.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("frames.h5");
# let rows: Vec<f64> = (0..200).map(f64::from).collect();
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("frames").with_f64_data(&rows);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let ds = file.dataset("frames")?;
let window = ds.read_f64_rows(100, 50)?; // rows 100..150 only
# assert_eq!(window.len(), 50);
# assert_eq!(window[0], 100.0);
# Ok::<(), hdf5_pure::Error>(())
```

Only the storage the window touches is read: a single bounded sub-read for compact and contiguous layouts, and just the chunks whose first-dimension span overlaps the window for chunked layouts, so peak memory scales with the window (plus one chunk) and not with the dataset. The window is clamped to the leading dimension, so a read past the end returns only the rows that exist and a zero-row request returns an empty `Vec`. Variable-length strings keep the bound too: [`read_string_rows`](crate::Dataset::read_string_rows) resolves only the window's heap references, and [`read_raw_rows`](crate::Dataset::read_raw_rows) streams those fixed-size references like any other element.

Combined with a [streaming open](crate::_guide::streaming#reading-a-large-dataset-a-window-at-a-time), this reads a dataset too large to hold in memory a fixed number of rows at a time.

### Datasets with unwritten storage

HDF5 allocates a dataset's storage lazily, so a dataset can exist with a shape and nothing behind it: created and never written, or written in part, which leaves some chunks of a chunked dataset allocated and others not. A read of those regions returns the dataset's **fill value**, the value [`with_fill_value`](crate::DatasetBuilder::with_fill_value) set, or the type's zero when none was set. The reference C library does the same, so a dataset another tool created and has not filled in yet returns a full-size array of its fill value.

[`Dataset::fill_value::<T>()`](crate::Dataset::fill_value) reports the value itself, returning `None` when the dataset uses the library default.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("schema.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("pending").with_f64_data(&[1.0, 2.0]).with_fill_value(-1.0_f64);
# builder.create_dataset("plain").with_f64_data(&[1.0, 2.0]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
assert_eq!(file.dataset("pending")?.fill_value::<f64>()?, Some(-1.0));
assert_eq!(file.dataset("plain")?.fill_value::<f64>()?, None);
# Ok::<(), hdf5_pure::Error>(())
```

## Navigating groups and attributes

[`File::root()`](crate::File::root) returns the root [`Group`](crate::Group), and [`File::group(path)`](crate::File::group) resolves a subgroup by path. A [`Group`](crate::Group) lists its children with [`groups()`](crate::Group::groups) and [`datasets()`](crate::Group::datasets) (each returning `Vec<String>` of names), opens a child dataset with [`dataset(name)`](crate::Group::dataset), and opens a child subgroup with [`group(name)`](crate::Group::group).

[`iter_groups()`](crate::Group::iter_groups) and [`iter_datasets()`](crate::Group::iter_datasets) walk the members, yielding `(String, Group)` and `(String, Dataset)` pairs and enumerating the group once for the whole walk. They are for taking every member: reaching one you can already name stays cheaper through [`dataset(name)`](crate::Group::dataset).

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# let mut sensors = builder.create_group("sensors");
# sensors.create_dataset("pressure").with_f32_data(&[101.3, 101.5]);
# builder.add_group(sensors.finish());
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;

let sensors = file.group("sensors")?;
println!("child groups: {:?}", sensors.groups()?);
println!("datasets:     {:?}", sensors.datasets()?);

let pressure = sensors.dataset("pressure")?;
# assert_eq!(pressure.read_f32()?, vec![101.3, 101.5]);
# Ok::<(), hdf5_pure::Error>(())
```

Attributes are read with [`attrs()`](crate::Group::attrs), available on both [`Group`](crate::Group) and [`Dataset`](crate::Dataset). It returns a `HashMap<String, AttrValue>`:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature")
#     .with_f64_data(&[22.5])
#     .set_attr("unit", AttrValue::AsciiString("degC".into()));
# builder.set_attr("version", AttrValue::I64(2));
# builder.write(&path)?;
use hdf5_pure::{AttrValue, File};

let file = File::open(&path)?;

let root_attrs = file.root().attrs()?;
println!("version: {:?}", root_attrs.get("version"));  // Some(I64(2))

let ds = file.dataset("temperature")?;
println!("unit: {:?}", ds.attrs()?.get("unit"));
# assert_eq!(root_attrs["version"], AttrValue::I64(2));
# Ok::<(), hdf5_pure::Error>(())
```

The full [`AttrValue`](crate::AttrValue) set and the writing patterns are in the groups and attributes guide. Attributes read identically on every backend, a [streaming open](crate::_guide::streaming) included.

## Inspecting a file

Two free functions check whether bytes or a path look like an HDF5 file without fully opening it:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5]);
# builder.write(&path)?;
# let bytes = std::fs::read(&path)?;
use hdf5_pure::{is_hdf5, is_hdf5_bytes};

let on_disk: bool = is_hdf5(&path)?;  // io::Result<bool>
let in_memory: bool = is_hdf5_bytes(&bytes);
# assert!(on_disk && in_memory);
# Ok::<(), hdf5_pure::Error>(())
```

An open [`File`](crate::File) reports its size and the format version it requires:

| Method | Returns |
|---|---|
| [`File::file_size()`](crate::File::file_size) | `u64` total byte length |
| [`File::libver_bound()`](crate::File::libver_bound) | [`LibVer`](crate::LibVer) low bound implied by the superblock version |

[`libver_bound()`](crate::File::libver_bound) returns the minimum library version needed to read the file, derived from its superblock version. `H5Fget_libver_bounds` reports the same low bound. The [`LibVer`](crate::LibVer) enum lists the release boundaries at which the on-disk format changed:

| Variant | HDF5 release |
|---|---|
| [`LibVer::Earliest`](crate::LibVer::Earliest) | 1.0+ (version 0/1 superblock, v1 symbol-table groups) |
| [`LibVer::V18`](crate::LibVer::V18) | 1.8 (version 2 superblock, new-style object headers) |
| [`LibVer::V110`](crate::LibVer::V110) | 1.10 (version 3 superblock, SWMR, extensible/fixed array indices) |
| [`LibVer::V112`](crate::LibVer::V112) | 1.12 |
| [`LibVer::V114`](crate::LibVer::V114) | 1.14, the value of [`LibVer::LATEST`](crate::LibVer::LATEST) |

[`libver_bound()`](crate::File::libver_bound) never returns [`V112`](crate::LibVer::V112) or [`V114`](crate::LibVer::V114): it derives from the superblock version, whose newest value is the version 3 of HDF5 1.10, so it caps at [`V110`](crate::LibVer::V110). Both are still meaningful to [`with_libver_bounds`](crate::FileBuilder::with_libver_bounds), where a lower bound of [`V112`](crate::LibVer::V112), [`V114`](crate::LibVer::V114) or [`LATEST`](crate::LibVer::LATEST) selects the 1.10 format. A low bound licenses newer encodings and does not demand them.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5]);
# builder.write(&path)?;
use hdf5_pure::{File, LibVer};

let file = File::open(&path)?;
println!("{} bytes", file.file_size());
assert_eq!(file.libver_bound(), LibVer::V110);  // this crate's writer output
# Ok::<(), hdf5_pure::Error>(())
```
