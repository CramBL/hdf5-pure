# Reading Files

This page covers opening HDF5 files, navigating their group hierarchy, and reading datasets and attributes back into Rust. The reading API is the same regardless of how a file is opened, so the patterns here apply equally to in-memory, on-disk, streaming, and SWMR reads.

!!! tip "Runnable example"
    The [`quickstart`](https://github.com/CramBL/hdf5-pure/blob/main/examples/quickstart.rs) example builds a file in memory and reads it back, doubling as a self-check. Run it with:

    ```bash
    cargo run --example quickstart
    ```

    The [`groups_and_attributes`](https://github.com/CramBL/hdf5-pure/blob/main/examples/groups_and_attributes.rs) example walks a nested hierarchy. Run it with `cargo run --example groups_and_attributes`.

## Opening a file

`File` is the entry point for reading. There are four ways to obtain one, all of which produce a value with an identical reading API:

| Constructor | Source | Notes |
|---|---|---|
| `File::open(path)` | A file on disk | Reads the whole file into memory. Requires the `std` filesystem. |
| `File::from_bytes(bytes)` | An in-memory `Vec<u8>` | No filesystem needed — the in-memory path, e.g. for WebAssembly. |
| `File::open_streaming(path)` | A file on disk, read on demand | Fetches metadata and chunks lazily; never buffers the whole file. See [Streaming](streaming.md). |
| `File::open_swmr(path)` | A file being appended to | Re-readable with `refresh()` to pick up new data. See [SWMR](swmr.md). |
| `File::from_source(source)` | Any `Source` — bytes with no path | The streaming reader over an object store, a sandboxed host, anything that answers "how long" and "bytes at this offset". See [Streaming](streaming.md). |

`File::open`, `File::open_streaming` and `File::from_source` refuse a file whose superblock marks it as held by a writer, with `Error::FileMarkedInUse` — the check `H5Fopen` makes of the same byte. Which reader gets at such a file depends on which mark it carries:

- **both bits, a SWMR writer**: follow it with `File::open_swmr`, which also picks up what the writer appends later.
- **the write bit alone**, which is what a [page-buffered](editing.md#the-page-buffer) writer leaves: no SWMR reader can follow it, so read a snapshot through any `*_with_options` open under `FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)`. That asserts the writer has flushed — it called `File::sync`, or closed the file — which this crate cannot check for you.

```rust
use hdf5_pure::{File, FileAccessProperties, WriteMarkPolicy};

let props = FileAccessProperties::new().with_write_mark_policy(WriteMarkPolicy::AllowSnapshot);
let file = File::open_with_options("live.h5", props).unwrap();
```

The opt-in unlocks nothing else: a SWMR pair still belongs to `File::open_swmr`, and `File::open_rw` is refused whatever it says. `File::from_bytes` does not consult the byte at all, at the cost of a copy of the file. Recover a file a writer left flagged when it exited with `File::clear_swmr_flag`; see [SWMR](swmr.md#recovering-a-flagged-file).

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let ds = file.dataset("temperature").unwrap();

println!("shape: {:?}", ds.shape().unwrap());    // [3]
println!("data:  {:?}", ds.read_f64().unwrap());  // [22.5, 23.1, 21.8]
```

To read from bytes that were never written to disk, for example the output of `FileBuilder::finish` (see [Writing Files](writing.md)):

```rust
use hdf5_pure::{File, FileBuilder};

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);
let bytes = builder.finish().unwrap();

let file = File::from_bytes(bytes).unwrap();
let x = file.dataset("x").unwrap().read_f64().unwrap();
```

!!! note "Tuning streaming reads"
    `File::open_streaming_with_options`, `File::open_swmr_with_options`, `File::open_with_options`, and `File::from_bytes_with_options` accept a `FileAccessProperties` to bound retained metadata and chunk-cache memory. See [Streaming](streaming.md) for `MetadataCacheConfig` and `ChunkCacheConfig`.

## Opening datasets

`File::dataset(path)` resolves a dataset by its full path from the root, returning a `Dataset`:

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let accel = file.dataset("sensors/imu/accel").unwrap();
```

A dataset can also be opened by name relative to its parent group via `Group::dataset(name)` (see [Navigating groups](#navigating-groups-and-attributes) below). To override the chunk cache for a single dataset, use `File::dataset_with_options(path, DatasetAccessProperties)`.

### Inspecting shape and datatype

`Dataset::shape()` returns the dimensions as a `Vec<u64>`. Two accessors describe the datatype: `Dataset::dtype()` returns a simplified `DType` classification, while `Dataset::datatype()` returns the full `Datatype` with exact field offsets and layout, which is useful for compound types (see [Compound Types](compound-types.md)).

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let ds = file.dataset("temperature").unwrap();

println!("shape: {:?}", ds.shape().unwrap());
println!("dtype: {:?}", ds.dtype().unwrap());
```

### Chunking, filters, and append eligibility

Four accessors describe a dataset's storage layout without reading any data. They also tell you whether a dataset can be grown in place with [`Dataset::append_staged`](editing.md#appending-to-an-unlimited-dataset) or [streaming `Dataset::append`](editing.md#streaming-appends) before you attempt the append:

| Method | Returns |
|---|---|
| `Dataset::is_chunked()` | `bool` — `true` for chunked storage; filtered datasets are always chunked |
| `Dataset::maxshape()` | `Option<Vec<u64>>` — maximum dimensions, an unlimited axis reported as `u64::MAX`; `None` for a fixed-shape dataset |
| `Dataset::chunk_shape()` | `Option<Vec<u64>>` — chunk dimensions, one per rank; `None` when not chunked |
| `Dataset::filters()` | `Vec<u16>` — HDF5 filter IDs in pipeline order (1 = deflate, 2 = shuffle, 3 = fletcher32, 6 = scale-offset, 32000 = LZF); empty when unfiltered |

```rust
use hdf5_pure::File;

let file = File::open("log.h5").unwrap();
let ds = file.dataset("samples").unwrap();

// Broadly appendable: chunked and unlimited along axis 0. The full rules
// (rank 1, Extensible-Array index, single hard link) are in the editing guide.
let appendable = ds.is_chunked()
    && matches!(ds.maxshape().unwrap().as_deref(), Some([u64::MAX, ..]));
```

## Reading dataset data

### Typed reads

The `read_*` family delivers a dataset's elements as a flat `Vec<T>` of the requested type. Each method coerces the stored values to the requested type, so they are about how you want the data delivered rather than an assertion about the stored datatype.

| Method | Result |
|---|---|
| `read_f64` | `Vec<f64>` |
| `read_f32` | `Vec<f32>` |
| `read_i8` / `read_i16` / `read_i32` / `read_i64` | signed integers |
| `read_u8` / `read_u16` / `read_u32` / `read_u64` | unsigned integers |
| `read_string` | `Vec<String>` (fixed- and variable-length) |
| `read_raw` | `Vec<u8>` of the complete unfiltered record bytes |

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let values = file.dataset("temperature").unwrap().read_f64().unwrap();
```

A typed read costs about what its result costs. The numeric readers decode the dataset a row window at a time, so what stands beside the returned `Vec<T>` is one window of stored bytes — roughly a mebibyte — rather than a second copy of the whole dataset. `read_u8` and `read_raw` hand back the stored bytes themselves and have nothing to decode.

### Generic reads

`Dataset::read::<T>()` is the generic counterpart to the typed `read_*` methods, bounded by the sealed `H5Element` trait (implemented for `f32`/`f64` and the 8/16/32/64-bit signed and unsigned integers). It lets you write code generic over the element type. Like `read_f64`, it requests delivery as `T` and coerces, so pick `T` to match the stored type for a lossless read.

```rust
use hdf5_pure::{File, FileBuilder, H5Element, Error};

fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, Error> {
    file.dataset(name)?.read::<T>()
}

let mut fb = FileBuilder::new();
fb.create_dataset("counts").with_data(&[1u32, 2, 3]);
let file = File::from_bytes(fb.finish().unwrap()).unwrap();

let counts: Vec<u32> = load(&file, "counts").unwrap();  // [1, 2, 3]
```

For the writing side and more detail, see [Generic I/O](generic-io.md). For N-dimensional reads as `ndarray` arrays via `read_array` / `read_array_dyn`, see [ndarray Support](ndarray.md) (needs the `ndarray` feature).

### String reads

`Dataset::read_string` reads both fixed-length and variable-length HDF5 string datasets into a `Vec<String>`. When you need to bound variable-length payload allocation before reading, or to consume strings one at a time, use `read_vlen_strings(VlenStringReadOptions)` or `visit_vlen_strings`. See [Variable-Length Strings](vlen-strings.md).

### Raw and compound reads

`Dataset::read_raw` returns the complete unfiltered record bytes, and `Dataset::read_compound::<T>()` decodes compound (struct-like) records. See [Compound Types](compound-types.md) and the [data types reference](../reference/data-types.md).

### Reading a row window

The `read_*` methods above deliver a whole dataset. To read only a range of leading-dimension rows — a **row window** `[start, start + count)` — without materializing the rest, use `read_raw_rows(start, count)` and the typed `read_f64_rows` / `read_f32_rows` / `read_i8_rows` … `read_u64_rows` / `read_string_rows` counterparts. Each decodes exactly like its whole-dataset form, so a window is that whole read sliced to the given row range.

```rust
use hdf5_pure::File;

let file = File::open("frames.h5").unwrap();
let ds = file.dataset("frames").unwrap();
let window = ds.read_f64_rows(100, 50).unwrap(); // rows 100..150 only
```

Only the storage the window touches is read: a single bounded sub-read for compact and contiguous layouts, and just the chunks whose first-dimension span overlaps the window for chunked layouts, so peak memory scales with the window (plus one chunk) rather than the dataset. The window is clamped to the leading dimension, so a read past the end returns only the rows that exist and a zero-row request returns an empty `Vec`. Variable-length strings keep the bound too: `read_string_rows` resolves only the window's heap references, and `read_raw_rows` streams those fixed-size references like any other element.

Combined with a [streaming open](streaming.md#reading-a-large-dataset-a-window-at-a-time), this reads a dataset too large to hold in memory a fixed number of rows at a time.

### Datasets with unwritten storage

HDF5 allocates a dataset's storage lazily, so a dataset can exist with a shape and have nothing behind it — created and never written, or written in part, leaving some chunks of a chunked dataset allocated and others not. Those regions read as the dataset's **fill value**: the value `with_fill_value` set, or the type's zero when none was set. This matches the reference C library, and it means a dataset another tool created but has not filled in yet reads as a full-size array of its fill value rather than failing.

`Dataset::fill_value::<T>()` reports the value itself, returning `None` when the dataset uses the library default.

```rust
use hdf5_pure::File;

let file = File::open("schema.h5").unwrap();
let ds = file.dataset("pending").unwrap();
assert_eq!(ds.fill_value::<f64>().unwrap(), Some(-1.0));
let values = ds.read_f64().unwrap(); // every element is -1.0 until something writes
```

## Navigating groups and attributes

`File::root()` returns the root `Group`, and `File::group(path)` resolves a subgroup by path. A `Group` lists its children with `groups()` and `datasets()` (each returning `Vec<String>` of names), opens a child dataset with `dataset(name)`, and opens a child subgroup with `group(name)`.

To walk the members rather than list them, `iter_groups()` and `iter_datasets()` yield `(String, Group)` and `(String, Dataset)` pairs, enumerating the group once instead of once per member. They are for taking every member: reaching one you can already name stays cheaper through `dataset(name)`.

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();

let sensors = file.group("sensors").unwrap();
println!("child groups: {:?}", sensors.groups().unwrap());
println!("datasets:     {:?}", sensors.datasets().unwrap());

let pressure = sensors.dataset("pressure").unwrap();
```

Attributes are read with `attrs()`, available on both `Group` and `Dataset`. It returns a `HashMap<String, AttrValue>`:

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();

let root_attrs = file.root().attrs().unwrap();
println!("version: {:?}", root_attrs.get("version"));  // Some(I64(2))

let ds = file.dataset("temperature").unwrap();
println!("unit: {:?}", ds.attrs().unwrap().get("unit"));
```

See [Groups and Attributes](groups-attributes.md) for the full `AttrValue` set and writing patterns. Attributes read identically on every backend, including [`File::open_streaming`](streaming.md).

## Inspecting a file

Two free functions check whether bytes or a path look like an HDF5 file without fully opening it:

```rust
use hdf5_pure::{is_hdf5, is_hdf5_bytes};

let on_disk: bool = is_hdf5("output.h5").unwrap();  // io::Result<bool>
let in_memory: bool = is_hdf5_bytes(&bytes);
```

An open `File` reports its size and the format version it requires:

| Method | Returns |
|---|---|
| `File::file_size()` | `u64` total byte length |
| `File::libver_bound()` | `LibVer` low bound implied by the superblock version |

`libver_bound()` mirrors the low bound of HDF5's `H5Fget_libver_bounds`: it returns the minimum library version needed to read the file, derived from its superblock version. The `LibVer` enum names the release boundaries at which the on-disk format changed:

| Variant | HDF5 release |
|---|---|
| `LibVer::Earliest` | 1.0+ (version 0/1 superblock, v1 symbol-table groups) |
| `LibVer::V18` | 1.8 (version 2 superblock, new-style object headers) |
| `LibVer::V110` | 1.10 (version 3 superblock, SWMR, extensible/fixed array indices) |
| `LibVer::V112` | 1.12 |
| `LibVer::V114` | 1.14, the value of `LibVer::LATEST` |

`libver_bound()` never returns `V112` or `V114`: it derives from the superblock version, whose newest value is the version 3 of HDF5 1.10, so it caps at `V110`. Both are still meaningful to `with_libver_bounds`, where a lower bound of `V112`, `V114` or `LATEST` selects the 1.10 format — the low bound licenses newer encodings rather than demanding them.

```rust
use hdf5_pure::{File, LibVer};

let file = File::open("output.h5").unwrap();
println!("{} bytes", file.file_size());
assert_eq!(file.libver_bound(), LibVer::V110);  // this crate's writer output
```
