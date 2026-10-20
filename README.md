# hdf5-pure

Pure-Rust HDF5 reader, writer, and in-place editor. No C dependencies, no build scripts, WASM-compatible.

**📖 [Documentation (docs.rs)](https://docs.rs/hdf5-pure)** ·
[Compatibility](docs/guide/compatibility.md) ·
[Examples](examples) ·
[Changelog](CHANGELOG.md)

> [!NOTE]
> Compatible with HDF5 1.8, 1.10, 1.12, 1.14, and 2.x. Checked in CI against `libhdf5`. See the
> [HDF5 compatibility matrix](docs/guide/compatibility.md) for feature-level
> read and write support.

## Features

- **Write** HDF5 files with datasets, groups, attributes, and nested hierarchies
- **Read** HDF5 files (v0/v1/v2/v3 superblocks, v1/v2 object headers, contiguous/chunked/compact storage)
- **Edit in place** Incremental add, delete (`H5Ldelete`), and copy (`H5Ocopy`) for datasets and groups.
- **SWMR** (single-writer / multiple-reader) append and refreshing read for 1-D unlimited datasets, interoperable with the `libhdf5` and `h5py`
- **Free Space Management** persistent tracking and reuse of space from deleted objects across sessions to prevent file bloat during in-place edits
- **Tunable metadata cache** configure memory limits and eviction behavior to optimize I/O performance for heavy read/write workloads
- **No C dependencies** pure Rust, with support for WASM and bare-metal targets (`no_std` with `alloc`)
- **MATLAB v7.3 compatible** userblock support, fixed-length ASCII attributes, variable-length string arrays, object references
- **Filters** Deflate, shuffle, Fletcher32, LZF, scale-offset, and optional ZFP
- Compound types, enumerations, array types
- Complex number datasets (as compound `{real, imag}`)

## Examples

Runnable, self-checking examples live in [`examples/`](examples). The table highlights common workflows. See the directory for the complete set. Run any with `cargo run --example <name>`:

| Example                 | What it shows                                                            |
| ----------------------- | ------------------------------------------------------------------------ |
| `quickstart`            | Build a file in memory and read it back                                  |
| `generic_io`            | Read/write generically over the element type (`with_data` / `read::<T>`) |
| `groups_and_attributes` | Nested groups and attributes of several types                            |
| `compression`           | Deflate, shuffle, scale-offset, and LZF filters                          |
| `compound_types`        | Compound (struct-like) records and complex numbers                       |
| `ndarray_io`            | N-dimensional array I/O (needs `--features ndarray`)                     |
| `edit_in_place`         | Add, copy, and delete objects with `File::open_rw`                       |
| `repack`                | Shrink a file and drop objects with `repack`                             |
| `swmr`                  | Single-writer / multiple-reader append and refresh                       |
| `file_space`            | File-space strategy and persistent free-space reuse across sessions      |

The `matlab_fixtures` example (run with `--features serde`) writes `.mat` v7.3 files for verification in MATLAB/Octave.

## Installation

```console
$ cargo add hdf5-pure
```

Or add it to `Cargo.toml` by hand:

```toml
[dependencies]
hdf5-pure = "0.47.0"
```

That pulls the default feature set (`std`, `checksum`, and `deflate`), which covers file I/O, the high-level reader and writer API, and deflate compression. The [Cargo features](#cargo-features) section below points to the canonical feature documentation.

For WebAssembly, keep the default features and add the target:

```console
$ cargo build --target wasm32-unknown-unknown
```

A WASM build uses the in-memory API: `FileBuilder::finish` serializes the file into a `Vec<u8>` and `File::from_bytes` parses one.

## Quick start

### Writing

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

// Datasets
builder.create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3]);

// Groups with nested datasets
let mut grp = builder.create_group("sensors");
grp.create_dataset("pressure").with_f32_data(&[101.3, 101.5]);
grp.set_attr("location", AttrValue::AsciiString("lab_a".into()));
builder.add_group(grp.finish());

// Attributes on the root group
builder.set_attr("version", AttrValue::I64(2));

builder.write("output.h5").unwrap();
```

### Reading

```rust
use hdf5_pure::File;

let file = File::open("output.h5").unwrap();
let ds = file.dataset("temperature").unwrap();

println!("shape: {:?}", ds.shape().unwrap());    // [3]
println!("data:  {:?}", ds.read_f64().unwrap());  // [22.5, 23.1, 21.8]

let attrs = file.root().attrs().unwrap();
println!("version: {:?}", attrs.get("version"));  // Some(I64(2))
```

### Generic over the element type

`DatasetBuilder::with_data(&[T])` and `Dataset::read::<T>()` let code work over the
supported scalar types through the sealed `H5Element` trait. `read::<T>()` requests
a delivery type and may coerce the stored datatype.

See the [generic element I/O guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/generic_io/index.html)
for supported element types, coercion behavior, and examples.

### Editing in place

`File::open_rw` opens an existing file for staged object edits and dataset appends without
rewriting the file from scratch. It selects a bounded or mirrored backing according to the file
and access properties.

```rust,no_run
use hdf5_pure::File;

let file = File::open_rw("output.h5").unwrap();
let root = file.root();

let run2 = root.create_group("run2").unwrap();
run2.create_dataset("signal", |b| { b.with_f64_data(&[1.0, 2.0, 3.0]); }).unwrap();
file.copy("temperature", "temperature_backup").unwrap();
root.delete("sensors/pressure").unwrap();

file.commit().unwrap();
```

The [editing guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/editing/index.html)
covers the backing choice, append modes, durability, locking, space reuse, and format-specific
rules. See the [compatibility matrix](docs/guide/compatibility.md) for the exact support status of
on-disk structures such as dense storage and chunk indexes.

### Reclaiming space (`repack`)

Deleting objects can make their storage reusable without making the file itself smaller.
`repack` writes the surviving objects into a fresh compact file, optionally dropping paths.

```rust,no_run
use hdf5_pure::{repack, RepackOptions};

let options = RepackOptions::new()
    .drop_path("scratch")
    .drop_path("runs/aborted");
repack("input.h5", "compact.h5", &options).unwrap();
```

The [repack guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/repack/index.html)
documents the fidelity guarantee, verbatim chunk-copy path, re-encoding fallbacks, reference
handling, and the cases that are rejected.

### File-space strategy

`FileBuilder::with_file_space_strategy` records the allocation strategy, persistence choice, and
free-space threshold. `with_file_space_page_size` configures paged allocation. Paged files that
will later be edited normally need persisted page information: a `FileSpaceStrategy::Page` file
created with `persist = false` is normally rejected by `File::open_rw` because the editor cannot
preserve metadata/raw-data page segregation.

See the [file-space guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/file_space/index.html)
for strategy semantics, persisted free-space reuse, paged files, and editing behavior.

### Streaming large files

`File::open_streaming` reads metadata and dataset storage on demand while keeping the same
dataset-reading API as `File::open`.

```rust
use hdf5_pure::File;

let file = File::open_streaming("huge.h5").unwrap();
let values = file.dataset("signal").unwrap().read_f64().unwrap();
```

The [streaming guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/streaming/index.html)
covers row-window reads, cache budgets, caller-supplied `Source` implementations, and the
bounded-memory read/write paths.

### In-memory (WASM)

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

let bytes: Vec<u8> = builder.finish().unwrap(); // no filesystem needed
```

### N-dimensional arrays (`ndarray` feature)

The `ndarray` feature adds rank-generic dataset I/O through the `ndarray` crate:

```rust
use hdf5_pure::{File, FileBuilder};
use ndarray::{array, Array2};

let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
let mut builder = FileBuilder::new();
builder.create_dataset("m").with_ndarray(&a);
let file = File::from_bytes(builder.finish().unwrap()).unwrap();
let back: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();
assert_eq!(a, back);
```

See the [N-dimensional array guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/ndarray/index.html)
for feature setup, memory order, dynamic-rank reads, and compression.

## SWMR (single writer, multiple readers)

SWMR lets one process append to an unlimited dataset while other processes follow it with
refreshing reads. It works with the reference HDF5 C library and h5py.

See the [SWMR guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/swmr/index.html) for dataset
layout requirements, writer and reader usage, durability, crash recovery, and the exact supported
subset. The [compatibility matrix](docs/guide/compatibility.md) records the support status.

## Supported data types

### Datasets

| Method                                                               | HDF5 type                            |
| -------------------------------------------------------------------- | ------------------------------------ |
| `with_data` (generic, any scalar below)                              | Inferred from the element type       |
| `with_f64_data`                                                      | IEEE 64-bit float                    |
| `with_f32_data`                                                      | IEEE 32-bit float                    |
| `with_i8_data` / `with_i16_data` / `with_i32_data` / `with_i64_data` | Signed integers                      |
| `with_u8_data` / `with_u16_data` / `with_u32_data` / `with_u64_data` | Unsigned integers                    |
| `with_complex32_data`                                                | Compound `{real: f32, imag: f32}`    |
| `with_complex64_data`                                                | Compound `{real: f64, imag: f64}`    |
| `with_compound_data`                                                 | Arbitrary compound types             |
| `with_compound_values`                                               | Safely encoded numeric tuples        |
| `with_enum_i32_data` / `with_enum_u8_data`                           | Enumeration types                    |
| `with_array_data`                                                    | Fixed-size array types               |
| `with_path_references`                                               | Object references (resolved by path) |
| `with_dtype` + `with_shape`                                          | Empty/zero-dimension datasets        |

### Compound types

Compound records, complex-number helpers, enumerations, fixed-size arrays, and object references
have dedicated builders and readers. See the [compound types guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/compound_types/index.html)
for layouts, tuple helpers, complex values, enumerations, arrays, and references.

### Variable-length string reads

`Dataset::read_string` reads fixed- and variable-length HDF5 string datasets. For large
variable-length datasets, the API also supports measuring payload size, applying allocation
limits, and visiting values one at a time.

See the [variable-length string guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/vlen_strings/index.html)
for the bounded and streaming read APIs.

### Attributes

| Variant                                                   | HDF5 encoding                                                  |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `AttrValue::F32` / `F64` (+ `…Array`)                     | Float scalar/array, at that width                              |
| `AttrValue::I8` / `I16` / `I32` / `I64` (+ `…Array`)      | Signed integer scalar/array, at that width                     |
| `AttrValue::U8` / `U16` / `U32` / `U64` (+ `…Array`)      | Unsigned integer scalar/array, at that width                   |
| `AttrValue::String` / `StringArray`                       | UTF-8 null-padded string                                       |
| `AttrValue::AsciiString`                                  | Fixed-length ASCII string                                      |
| `AttrValue::VarLenString` / `VarLenStringArray`           | Variable-length UTF-8 string, scalar or array (global heap)    |
| `AttrValue::VarLenAsciiString` / `VarLenAsciiStringArray` | The same, with `CSET = ASCII`                                  |
| `AttrValue::VarLenAsciiCharArray`                         | MATLAB's VLEN sequence of one-byte ASCII strings (global heap) |

## Compression

Dataset filters include deflate, shuffle, Fletcher32, LZF, scale-offset, and optional ZFP.
Filters operate on chunked storage. Reads decode the configured pipeline automatically.

```rust
# use hdf5_pure::DeflateLevel;
builder.create_dataset("compressed")
    .with_f64_data(&data)
    .with_chunks(&[100])
    .with_shuffle()
    .with_deflate(DeflateLevel::Six);
```

See the [compression guide](https://docs.rs/hdf5-pure/latest/hdf5_pure/_guide/compression/index.html)
for filter semantics, chaining rules, scale-offset and ZFP behavior, feature flags, and
interoperability.

## Userblock (MATLAB v7.3)

```rust
let mut builder = FileBuilder::new();
builder.with_userblock(512);
builder.create_dataset("data").with_f64_data(&[1.0]);

let mut bytes = builder.finish().unwrap();
// Write MATLAB header into userblock
bytes[126] = b'I';
bytes[127] = b'M';
```

## MATLAB struct pattern

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();
let mut grp = builder.create_group("my_struct");

let mut fields = Vec::new();
for (name, data) in [("x", vec![1.0, 2.0]), ("y", vec![3.0, 4.0])] {
    fields.push(name.to_string());
    grp.create_dataset(name).with_f64_data(&data)
        .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
}

grp.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
grp.set_attr("MATLAB_fields", AttrValue::VarLenAsciiCharArray(fields));
builder.add_group(grp.finish());
```

## MATLAB v7.3 `.mat` via serde

With the `serde` feature, Rust structs can be serialized directly to `.mat`
v7.3 files and back:

> These files are written in the HDF5 1.8 on-disk format, which every MATLAB
> release can open. A version 3 superblock is an HDF5 1.10 addition, and MATLAB
> used HDF5 1.8.12 before R2021b, so the newer format is unreadable there. Set
> `mat::Options::libver` to `LibVer::V110` to opt into it, which compression
> requires. See the [`mat` module](https://docs.rs/hdf5-pure/latest/hdf5_pure/mat/).

```rust
use hdf5_pure::mat::{self, Complex64, Matrix};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Experiment {
    name: String,
    trial: u32,
    samples: Vec<f64>,
    data: Matrix<f64>,
    waveform: Vec<Complex64>,
    config: Config,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Config { threshold: f64, tag: String }

let e = Experiment {
    name: "run1".into(), trial: 3,
    samples: vec![1.0, 2.0, 3.0],
    data: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    waveform: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    config: Config { threshold: 0.5, tag: "prod".into() },
};

mat::to_file(&e, "experiment.mat").unwrap();
let back: Experiment = mat::from_file("experiment.mat").unwrap();
assert_eq!(back, e);
```

The top-level value must be a struct (or `HashMap<String, _>`); each field
becomes a MATLAB variable. Mapping:

| Rust                                                    | HDF5 / MATLAB encoding                                                                                                      |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `f64`, `f32`, `i*`, `u*`                                | scalar dataset `[1,1]`, `MATLAB_class = "double"` / `"single"` / `"int*"` / `"uint*"`                                       |
| `bool`                                                  | `uint8` scalar, `MATLAB_class = "logical"`                                                                                  |
| `String` / `&str`                                       | `uint16` `[1, N]` UTF-16LE, `MATLAB_class = "char"`                                                                         |
| `Vec<T>` of numeric `T`                                 | MATLAB `[N, 1]` column vector (HDF5 shape `[1, N]`) - `OneDimensionalMode::RowVector` makes it a MATLAB `[1, N]` row        |
| `Matrix<T>` or `Vec<Vec<T>>` of same length             | column-major 2-D dataset, HDF5 shape `[cols, rows]`                                                                         |
| `Complex64` / `Complex32` / `ComplexI16` / …            | compound `{real, imag}` dataset, `MATLAB_class` = the _component_ class                                                     |
| nested struct                                           | HDF5 group with `MATLAB_class = "struct"`, `MATLAB_fields`                                                                  |
| `Option<T>` (struct field)                              | `struct([])` if `None`, `NullPolicy::Omit` drops the field instead, `NullPolicy::Error` rejects it                          |
| `()`, unit struct, `serde_json::Value::Null`            | same as `None`, through `NullPolicy`                                                                                        |
| `None` / `()` / `Null` at the **root**                  | a valid file with no variables, byte-identical to what an empty root map writes, only `NullPolicy::Error` rejects it        |
| unit enum variant                                       | UTF-16 char dataset holding the variant name. `UnitVariantEncoding::Index` writes the declaration index as `uint32` instead |
| empty `Vec<T>`                                          | empty `double` (`[]`). `EmptySequencePolicy::Cell` writes `{}` instead                                                      |
| `Vec<Struct>` / `Vec<Option<T>>` / ragged `Vec<Vec<T>>` | cell array (`MATLAB_class = "cell"`, object references into `#refs#`). `None` slots become `struct([])`                     |

### Cell array pattern

Sequences that don't unify into a numeric matrix lower to a MATLAB cell array. Each element is interned under the conventional `#refs#` group and the parent dataset stores object references.

```rust
use hdf5_pure::mat;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Point { x: f64, y: f64 }

#[derive(Serialize, Deserialize)]
struct Capture {
    /// 3x1 cell array of struct.
    path: Vec<Point>,
    /// 3x1 cell array; the `None` slot becomes `struct([])`.
    optionals: Vec<Option<Point>>,
    /// Outer 2x1 cell of cells; rows-of-variable-length-records shape.
    grid: Vec<Vec<Option<Point>>>,
    /// Ragged numerics also fall back to cell rather than erroring.
    ragged: Vec<Vec<f64>>,
}
```

In MATLAB this loads as `iscell(path) == true`, `path{1}.x`, etc. Empty `None` slots load as `struct([])` (`isempty(fieldnames(...))`).

**Reader compatibility.** Cell arrays load correctly in MATLAB, libmatio (reference C library), Julia's `MAT.jl`, and Python via `pymatreader` / `hdf5storage`. GNU Octave 11's `load` does not yet follow object references for v7.3 cells (warns "unknown datatype"); load such files with one of the above instead.

Reading decodes the MATLAB opaque value classes `datetime`, `duration`, and `categorical` into `MatDatetime` / `MatDuration` / `MatCategorical`, and surfaces any other opaque class (`table`, `containers.Map`, …) losslessly as its raw property map. Not supported for _writing_: non-unit enum variants, MATLAB objects (`classdef`), datetime / duration / categorical types.

## Cargo features

The default feature set is `std`, `checksum`, and `deflate`. Optional features are
`fast-deflate`, `provenance`, `zfp`, `ndarray`, `serde`, and `num-complex`. The
[Cargo features section of the crate documentation](https://docs.rs/hdf5-pure/latest/hdf5_pure/#cargo-features)
is the canonical description of what each feature enables and which dependencies or other
features it implies.

For bare-metal `no_std`, disable default features and keep `checksum` for metadata validation:

```toml
[dependencies]
hdf5-pure = { version = "0.47.0", default-features = false, features = ["checksum"] }
```

The high-level `File` / `FileBuilder` API is `std`-gated, so a `no_std` build exposes only the
lower-level primitives.

## Limitations

Unsupported and partially supported HDF5 file-format features are listed in the [HDF5 compatibility matrix](docs/guide/compatibility.md). Operations that cannot preserve data faithfully return a typed error.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The [AI policy](AI_POLICY.md) and the [code style](CODE_STYLE.md) apply to every contribution.

## History

hdf5-pure was started by Stephen Berry, as a derived project of rustyhdf5 which was a ~25k LoC slop implementation of HDF5 with wild claims about performance over libhdf5. rustyhdf5 of course didn't support 98% of HDF5 features and was extremely buggy.

Under Stephen, hdf5-pure rapidly developed into an ambitious and respectable project, which read and wrote most HDF5 files without issues. Stephen eventually transferred the project ownership to Marc König who'd been aiding development with numerous bug reports and general recommendations to improvements. Marc König is a long time HDF5 user and contributor to hdf5-rust (hdf5-metno) and hdf5-pure development continued under him.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

### Third-party code

- The scale-offset filter in `crates/hdf5-pure-filter/src/scaleoffset.rs` is a port of `H5Zscaleoffset.c` from the HDF5 library, which The HDF Group licenses under the BSD 3-Clause license in [LICENSE-HDF5](LICENSE-HDF5). The crosscheck tests build that library to compare against, and nothing of it ships in the crate.
- The ZFP codec in `crates/hdf5-pure-filter/src/zfp.rs` is a port of the reference implementation by
  Lawrence Livermore National Security, LLC, under the BSD 3-Clause license in
  [LICENSE-ZFP](LICENSE-ZFP).
- The metadata checksum in `crates/hdf5-pure-format/src/checksum.rs` is Bob Jenkins' public-domain lookup3 hash.
- The HDF5 format parsing and low-level I/O modules are derived from rustyhdf5 by the RustyStack project, MIT licensed.
