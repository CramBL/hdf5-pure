# MATLAB v7.3 (.mat) Files

A MATLAB v7.3 `.mat` file is an HDF5 file dressed in MATLAB conventions: a 512-byte userblock carrying the `MATLAB 7.3 MAT-file` signature, a `MATLAB_class` attribute on every dataset and group, column-major 2-D arrays, and UTF-16 strings. This page covers the high-level serde path that writes and reads `.mat` files from ordinary Rust structs, the supported type mapping, MATLAB cell arrays, and the lower-level conventions for hand-built files.

!!! tip "Runnable example"
    The [`matlab_fixtures`](https://github.com/stephenberry/hdf5-pure/blob/main/examples/matlab_fixtures.rs) example writes a directory of `.mat` v7.3 fixtures (scalars, vectors, matrices, strings, nested structs, complex data, cell arrays, and edge shapes) for verification in MATLAB and Octave. Run it with:

    ```bash
    cargo run --example matlab_fixtures --features serde
    ```

## Requires the `serde` feature

The high-level `.mat` API is gated on the `serde` feature, which is off by default. The `mat::Matrix`, the `mat::Complex*` types, and `mat::MatElement`, along with `mat::to_file` / `mat::from_file`, are only available when it is enabled. See the [features reference](../reference/features.md) for the full list.

```toml
[dependencies]
hdf5-pure = { version = "0.14", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
```

## Serializing a struct to `.mat`

Any type deriving `serde::Serialize` / `Deserialize` round-trips through `mat::to_file` and `mat::from_file`. The top-level value must be a struct with named fields (or a `HashMap<String, _>`); each field becomes a top-level MATLAB variable.

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

To work with bytes instead of the filesystem, use `mat::to_bytes` and `mat::from_bytes`, which take and return a `Vec<u8>` / `&[u8]`. To write somewhere else entirely — a socket, a compressing wrapper — use `mat::to_writer` and `mat::to_writer_with_options`, which assemble the file straight onto any `io::Write` without holding it in memory. All are equally subject to the `serde` feature gate.

## Type mapping

The serializer maps Rust types to HDF5 datasets and the MATLAB classes MATLAB expects on read:

| Rust | HDF5 / MATLAB encoding |
|---|---|
| `f64`, `f32`, `i*`, `u*` | scalar dataset `[1,1]`, `MATLAB_class = "double"` / `"single"` / `"int*"` / `"uint*"` |
| `bool` | `uint8` scalar, `MATLAB_class = "logical"` |
| `String` / `&str` | `uint16` `[1, N]` UTF-16LE, `MATLAB_class = "char"` |
| `Vec<T>` of numeric `T` | MATLAB `[N, 1]` column vector (HDF5 shape `[1, N]`); see [1-D vector orientation](#1-d-vector-orientation) |
| `Matrix<T>` or `Vec<Vec<T>>` of same length | column-major 2-D dataset, HDF5 shape `[cols, rows]` |
| `Complex64` / `Complex32` / `ComplexI16` / … | compound `{real, imag}` dataset, `MATLAB_class` = the *component* class |
| nested struct | HDF5 group with `MATLAB_class = "struct"`, `MATLAB_fields` |
| `Option<T>` (struct field) | `struct([])` if `None`; `NullPolicy::Omit` drops the field instead, `NullPolicy::Error` refuses it |
| unit `()` / unit struct / `serde_json::Value::Null` (struct field) | same as `None` (see note below) |
| `None` / `()` / `Null` at the **root** | a valid file with no variables, byte-identical to what an empty root map or a fieldless struct writes; only `NullPolicy::Error` refuses it. It does not read back as `None`, since the deserializer presents the root as a struct |
| unit enum variant | UTF-16 char dataset holding the variant name; `UnitVariantEncoding::Index` writes the declaration index as `uint32` instead |
| empty `Vec<T>` | empty `double` (`[]`); `EmptySequencePolicy::Cell` writes `{}` instead |
| `Vec<Struct>` / `Vec<Option<T>>` / ragged `Vec<Vec<T>>` | cell array (`MATLAB_class = "cell"`, object references into `#refs#`); `None` slots become `struct([])` |

!!! note "Unit and `null` fields"
    A struct field that serializes as a Rust unit `()` is written exactly like `Option::None`. The most common case is a `serde_json::Value::Null` field, since `serde_json` serializes `Value::Null` via `serialize_unit`.

    Under the default `NullPolicy::EmptyStructArray` the field is present on disk as MATLAB `struct([])`, so `isfield` reports `true` and MATLAB code can reference it unconditionally and test it with `isempty(fieldnames(x))`. The two forms still differ for a struct with no fields, and only the `fieldnames` form is verified against MATLAB (see `matlab_fixtures/verify.m`). Since 0.34 a bare `isempty(x)` is also reliable: the default `EmptyMarkerEncoding::DataAsDims` stores the marker's dimensions in its payload, so the reference library recovers `0x0` with zero elements where the previous encoding recovered no dimensions and an element count of one.

    Reading it back is lenient but not universal. `struct([])` deserializes into `Option<T>` as `None`, `Vec<T>` as empty, `serde_json::Value` as `Null`, and `()` as `()`. It does **not** deserialize into a bare scalar, `String`, struct or map: those report a type error. Note `#[serde(default)]` does not rescue them, because the field is *present* with a struct value rather than missing, so the default is never consulted. This is the one migration hazard in 0.30: a reader with `#[serde(default)] count: u32` against a writer whose `count` is `None` worked under the pre-0.30 behavior and now fails. Give such a field type `Option<u32>`, or write with `NullPolicy::Omit`.

    Under `NullPolicy::Omit` (which is what this writer did unconditionally before 0.30) the field is absent instead, and reading it back needs `#[serde(default)]` on the field, or an `Option<T>` field, which serde defaults to `None` automatically. A non-`Option` field with no serde default fails to deserialize an omitted field with a missing-field error.

### 1-D vector orientation

A `Vec<T>` becomes a MATLAB **column** vector — MATLAB `[N, 1]`, stored as HDF5 shape `[1, N]`, since HDF5 storage is the transpose of the MATLAB shape. Every 1-D array follows this rule, complex ones and [cell arrays](#cell-arrays) included.

[`to_bytes`](https://docs.rs/hdf5-pure/latest/hdf5_pure/mat/fn.to_bytes.html) is fixed at that default. To get MATLAB `[1, N]` rows instead, write through `to_bytes_with_options` with `one_dimensional_mode: OneDimensionalMode::RowVector`:

```rust
use hdf5_pure::mat::{self, OneDimensionalMode, Options};
use serde::Serialize;

#[derive(Serialize)]
struct Capture { samples: Vec<f64> }

let mut opts = Options::default();
opts.one_dimensional_mode = OneDimensionalMode::RowVector;
let bytes = mat::to_bytes_with_options(&Capture { samples: vec![1.0, 2.0, 3.0] }, &opts).unwrap();
```

This matters when the same data reaches MATLAB by more than one route. [BEVE](https://github.com/beve-org/beve)'s MATLAB loader (`matlab/load_beve.m`) reconstructs a complex array as `complex(raw(1,:), raw(2,:))`, a `1×N` **row**; a pipeline that writes the same captures as both BEVE and `.mat` should set `RowVector` here so the two agree in the MATLAB workspace rather than differing by a transpose.

### Matrices and the column-major convention

Rust is row-major; MATLAB is column-major. The `mat::Matrix<T>` newtype carries the Rust-side `rows`/`cols` and a row-major `data` vector, and the serializer transposes to column-major byte order and stores the HDF5 dataset with shape `[cols, rows]` so MATLAB sees the intended `rows × cols` matrix. Build one with `Matrix::from_row_major(rows, cols, data)` (it panics if `data.len() != rows * cols`) or `Matrix::zeros(rows, cols)`; read the parts back with `rows()`, `cols()`, `data()`, and `into_data()`.

```rust
use hdf5_pure::mat::{self, Matrix};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Frame { a: Matrix<f64> }

let v = Frame {
    a: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
};
mat::to_file(&v, "matrix.mat").unwrap();
```

A bare `Vec<Vec<T>>` whose rows all share a length is also recognized as a 2-D matrix, but `Matrix` is the unambiguous API. The element type `T` is bounded by the sealed `mat::MatElement` trait, which is implemented for `f32`/`f64`, the 8/16/32/64-bit signed and unsigned integers, `bool`, and every complex type. The trait is sealed because MAT v7.3 admits only this fixed set of numeric classes; you cannot implement it for other types.

### Complex numbers

There is one complex newtype per component class — `Complex64` and `Complex32` for the float classes, `ComplexI8` / `ComplexI16` / `ComplexI32` / `ComplexI64` and the `ComplexU*` counterparts for the integer ones — each constructed with `ComplexI16::new(re, im)` (or the `re` / `im` fields directly). A bare value becomes a compound scalar of HDF5 shape `[1, 1]`; a `Vec<ComplexI16>` becomes a compound dataset of HDF5 shape `[1, N]`, which is a MATLAB column (see [1-D vector orientation](#1-d-vector-orientation)). The on-disk layout is the same `{real, imag}` compound MATLAB uses for complex arrays. For a deeper treatment of HDF5 compound datasets see the [compound types guide](../guide/compound-types.md).

`MATLAB_class` names the *component* class, not anything complex-specific, which is how MATLAB tells `complex(int16(re), int16(im))` from a complex `double`. Picking the component your data actually has is worth doing: a capture that samples as pairs of 16-bit integers takes four bytes per sample as `ComplexI16` and eight as `Complex32`, and the extra four carry nothing.

```rust
use hdf5_pure::mat::{self, ComplexI16};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Capture { samples: Vec<ComplexI16> }

let v = Capture {
    samples: vec![ComplexI16::new(-32768, 32767), ComplexI16::new(0, -1)],
};
mat::to_file(&v, "capture.mat").unwrap();
```

One consumer-side caveat worth knowing before choosing an integer component: MATLAB stores and loads complex integer arrays, but it refuses *arithmetic* on them — `a * b` on two complex `int16` values raises "Complex integer arithmetic is not supported", and the caller has to `double(...)` them first (or use Fixed-Point Designer's `fi` objects). That is a property of MATLAB, not of the file: the array arrives intact and `isa(x, 'int16')` and `~isreal(x)` both hold. It costs nothing for a capture format that is stored compactly and widened once at the point of use, which is the case this exists for.

Components are never converted between widths, in either direction. An `int16` complex dataset deserializes into `ComplexI16` and nothing else: reading it as `Complex64` would be lossless and is still refused, because the component width is part of what the file says it holds. Reading a `double` capture into `ComplexI16` is refused for the same reason, and would truncate. A caller holding float data that it knows to be exact integers is the one that can decide whether narrowing is meaningful, so do the conversion before serializing.

The same rule is enforced against the file itself: `MATLAB_class` names the component width, the `{real, imag}` compound carries the bytes, and a file where those two disagree is refused rather than decoded. This matters because the disagreement is not always visible — a complex `int64` array with no class attribute at all falls back to `double`, whose element size is identical, so the payload length alone cannot tell them apart.

### Large complex arrays

Serde has no bulk channel for a sequence, so a `Vec<ComplexI16>` costs one serializer dispatch per sample — around 26 ns, which for a capture-sized array is essentially the whole cost of writing the file. `mat::complex` carries a `serialize_with` helper per component class that hands the slice over whole instead:

```rust
use hdf5_pure::mat::{self, ComplexI16};
use serde::Serialize;

#[derive(Serialize)]
struct Capture {
    #[serde(serialize_with = "mat::complex::i16_array")]
    samples: Vec<ComplexI16>,
}
```

Measured on 64 Mi samples (256 MiB), one write per process: 1.78 s element-wise against 0.06 s. Repeat writes in a warm process reach 0.03 s, so quote the cold figure for a capture tool that writes once and exits. The bytes are identical either way — this changes how long a write takes, not what it produces.

The element type is any `mat::ComplexElement`: this module's `Complex*` types, `num_complex::Complex<T>` with the `num-complex` feature enabled, or your own. Implementing it is `unsafe` and asserts a layout — `#[repr(C)]`, two components, real first, every byte initialized data — because the helpers read the slice as raw bytes. The component type is part of the bound, so a same-width class cannot slip through: `i32` parts will not compile as `f32_array`.

Three things to know before annotating an existing field:

- **The output is MAT-specific.** The slice crosses serde as a byte string, so the same struct serialized to JSON or another format emits raw bytes rather than a sequence, and its own `Deserialize` will not read that back. Annotate a field only on a type written to `.mat` alone.
- **An empty slice keeps its component class.** See [empty complex arrays](#empty-complex-arrays) below; it is the only case where the annotation changes the file.
- **One-dimensional only.** A `Matrix<ComplexI16>` or `Vec<Vec<ComplexI16>>` has no bulk path and still pays per element.

Reading is untouched, which after this change makes it the slower half of a round trip: a `.mat` deserializes through the same per-element path it always did. If peak memory rather than speed is the constraint, [`write_blocks`](#writing-more-data-than-fits-in-memory) is the tool — these helpers still build the array in full.

### Empty complex arrays

An empty complex array is written here as a zero-element `{real, imag}` compound that keeps its component class, and it round-trips through this crate. MATLAB writes empties differently: `Mat_VarWriteEmpty` stores the dimensions *as data* under `MATLAB_empty = 1`, keeping the plain class name, and the `EmptyMarkerEncoding` option selects that form for real arrays. Complex arrays do not currently follow the option. libmatio reads both forms; if you need the MATLAB-native shape for an empty complex array specifically, write it as an empty real array of the component class instead.

An empty `Vec<ComplexI16>` is the one case where the plain and [bulk](#large-complex-arrays) paths differ. An empty sequence reveals no element, so the plain path cannot recover the component class and falls back on `empty_sequence_policy` — an empty `double` by default, an empty cell under `EmptySequencePolicy::Cell`. The helper was told the class by name and keeps it, matching `MatBuilder::write_complex_i16` and the `Matrix<Complex*>` sentinels. Both read back as an empty array; the shape is `0x0` either way.

## Cell arrays

A sequence whose elements do not unify into a single numeric matrix lowers to a MATLAB cell array rather than erroring. Each element is interned under the conventional `#refs#` group, and the parent dataset stores HDF5 object references with `MATLAB_class = "cell"`. This covers `Vec<Struct>`, `Vec<Option<T>>` with interspersed `None`, nested cells of cells, and ragged `Vec<Vec<T>>`. An `Option::None` slot inside a sequence becomes `struct([])` so every cell slot has a defined MATLAB type.

Each interned object also carries `H5PATH`, the absolute path of the object itself, which is what MATLAB writes on all but one of its own (the `canonical empty` placeholder in the MCOS subsystem carries none). Nothing here reads the attribute — an object reference resolves without it — and MATLAB stamps objects deeper than `#refs#`'s immediate children with a value that is *not* their path, which this crate does not reproduce.

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

In MATLAB this loads as `iscell(path) == true`, with elements addressed as `path{1}.x`, and so on. Empty `None` slots load as `struct([])` (`isempty(fieldnames(...))`).

!!! note "Reader compatibility"
    Cell arrays load correctly in MATLAB, libmatio (the reference C library), Julia's `MAT.jl`, and Python via `pymatreader` / `hdf5storage`. GNU Octave 11's `load` does not yet follow object references for v7.3 cells (it warns "unknown datatype"); load such files with one of the other tools instead.

## Struct arrays (reading)

A struct array authored in MATLAB (`s(1).x = …; s(2).x = …`) is stored as a `MATLAB_class = "struct"` group whose every field is a dataset of per-element object references — a struct-of-arrays. `mat::from_file` / `mat::from_bytes` transpose that into an array-of-structs: a `1×N` / `N×1` array reads into `Vec<T>`, and a true `M×N` array into `Vec<Vec<T>>`, where `T` is your own struct. A scalar struct still reads as a single struct.

!!! note "Write/read asymmetry"
    This is a read-only path. Writing a `Vec<Struct>` from Rust produces a MATLAB **cell array** (see [Cell arrays](#cell-arrays)), not a native struct array, so a `.mat` you write and one MATLAB writes from the same Rust type differ on disk. Both read back into `Vec<Struct>`.

## The on-disk format MATLAB's `load` needs

MATLAB has not always read HDF5 with the same library. [MathWorks documents the version per release](https://www.mathworks.com/help/matlab/hdf5-files.html):

| MATLAB release | HDF5 C library |
| --- | --- |
| R2024b and later | 1.14.4.3 |
| R2024a | 1.10.11 |
| R2023b | 1.10.10 |
| R2022a – R2023a | 1.10.8 |
| R2021b | 1.10.7 |
| R2021a and earlier | 1.8.12 |

**Read that table as which library the release links, not as which formats `load` accepts.** The two are not the same, and assuming they are is how a caller ships a file their own MATLAB cannot open.

The version 3 superblock is an HDF5 1.10 addition, so a `.mat` file carrying one cannot be opened by MATLAB before R2021b — not partially, not with a warning, at all. `mat::Options::libver` therefore defaults to `LibVer::V18` and the writer emits a version 2 superblock with version 3 data-layout messages: the newest encoding HDF5 1.8 understands, and one every later MATLAB reads as well. Choosing it costs nothing. The message bodies are identical between the two, so the files differ in two version bytes and are the same size.

That much follows from the table. What does not, and is the stronger reason to prefer the 1.8 format, is this measurement:

| MATLAB | `H5.get_libversion` | superblock 2 `load` | superblock 3 `load` |
| --- | --- | --- | --- |
| R2023a Update 1 | 1.10.8 | loads, decodes correctly | *"Not a binary MAT-file"* |

R2023a links HDF5 1.10.8, which reads a version 3 superblock perfectly well — and its `load` refuses one anyway. The two files behind that row are byte-identical through the 512-byte userblock and differ in 35 bytes, all of them the superblock version, the data-layout message versions, and the checksums that follow; so the format is the only variable, and it is decisive.

MathWorks documents the beginning of this. Around R2021b it shipped two libraries at once — 1.10.7 for the `h5read`/`h5disp`/`h5info` interface, with 1.8.12 kept on the MAT v7.3 path to avoid 1.10 regressions — which produces an unusual and recognizable symptom: a file `h5disp` prints happily and `load` refuses. The measurement above says that split, or something with the same effect, is still in force in R2023a, two years on.

What it does *not* say is why. An older library on the MAT path and a MAT reader that caps the superblock version it will accept are both consistent with this result, and telling them apart would take more than a `load`. The operational conclusion is the same either way: **write the 1.8 format, and do not infer from the reported library version that a 1.10 file is safe.**

`examples/octave/check_format.m` is what produced that row, and it puts the same question to whichever MATLAB you have. It prints its own inputs — release and library version — alongside the outcome, so a run on another release extends the table above rather than replacing it. Run it in MATLAB rather than Octave, whose HDF5 is modern and reads both formats, so it cannot tell them apart.

The one thing that cannot be written that way is compression, which needs chunked storage, whose chunk indices arrived in 1.10. Asking for both is refused with `MatError::CompressionNeedsNewerFormat` rather than resolved in either direction, since dropping the compression loses what you asked for and raising the format produces a file MATLAB cannot load:

```rust
use hdf5_pure::mat::{Compression, Options};
use hdf5_pure::LibVer;

let mut options = Options::default();
options.compression = Compression::Deflate { level: 6, shuffle: true };
options.libver = LibVer::V110; // required for compression; MATLAB `load` cannot read it
```

Real MATLAB writes an older format still: a version 0 superblock with v1 symbol-table groups and v1 B-tree chunk indices, which is what `save -v7.3` produces. This crate reads that format but does not write it, so its `.mat` output is not byte-identical to MATLAB's own.

## How an empty value is stored

MAT v7.3 is not publicly documented, so the particulars here are measured against the genuine MATLAB files vendored under `tests/data/matlab` rather than taken from a specification. An empty value is a two-element `uint64` dataset whose *payload* is the dimension vector, marked with `MATLAB_empty`, and it carries no other attribute — in particular no `MATLAB_int_decode`, which says how to read stored integers back as `char` or `logical` and has nothing to describe when the payload is a dimension vector. Not one of the 352 empty datasets in those fixtures carries it, for any class.

An empty Rust sequence is written as `0x0`, MATLAB's `[]`, whichever `OneDimensionalMode` is set: the mode orients a vector, and an empty one has no orientation to preserve. The difference is visible in MATLAB, where `[[], 1]` is `1` while `[zeros(0,1), 1]` is a dimension-mismatch error, and `0x0` is what MATLAB itself overwhelmingly writes. `MatBuilder::write_empty` still stores whatever dimensions you hand it, so `zeros(0,1)` remains expressible.

## Opaque value classes

Reading (`from_bytes`) decodes the MCOS opaque value classes `datetime`, `duration`, and `categorical` into the public `MatDatetime`, `MatDuration`, and `MatCategorical` types (Unix-epoch millisecond instants, durations in milliseconds, and category codes plus names). Any other opaque class (`table`, `containers.Map`, `dictionary`, user `classdef`s, …) is surfaced losslessly as its raw property map, so it still deserializes into a matching struct; function handles and legacy objects are refused by name with `MatError::UnsupportedMatlabClass`.

## Not supported (writing)

Writing (`to_bytes`) does not encode non-unit enum variants, MATLAB `classdef` objects, or `datetime` / `duration` / `categorical` types. Unit enum variants are supported and serialize to a UTF-16 char dataset holding the variant name.

## Writing more data than fits in memory

`MatBuilder::write_f64` and its siblings copy the slice they are given, and `finish` returns the assembled file, so writing a large array costs roughly three times its size at peak: the caller's copy, the builder's, and the file's. Two APIs remove those, independently.

`MatBuilder::finish_to` (and `write`, which is `finish_to` onto a file) assembles the file straight onto an `io::Write` in ascending-address order, never seeking and never holding the result. It produces byte-for-byte what `finish` returns.

`MatBuilder::write_blocks` stages a dataset whose bytes are produced during the write rather than handed over. It takes a `DataProducer`, which the writer calls once per block, in order, during emission — not during layout, which works from the shape alone. Together they write a `.mat` of any size in about one block of memory:

```rust
use hdf5_pure::mat::{Block, DataProducer, MatBuilder, MatError, Options};

// Each request carries its own contract, so the producer holds no state.
struct Samples;

impl DataProducer for Samples {
    fn block_bytes(&self, block: Block, out: &mut Vec<u8>) -> Result<(), MatError> {
        for i in 0..block.elements {
            let value = (block.first_element + i) as f64;
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

// [channels, samples]: see "element order" below for why this way round.
let dims = [4, 10_000_000];
let mut mb = MatBuilder::new(Options::default());
mb.write_blocks::<f64>("samples", &dims, Box::new(Samples)).unwrap();
mb.write("capture.mat").unwrap();
```

A `Block` carries its `index`, the `first_element` it starts at in MATLAB's linear order, and how many `elements` to write (`block.len()` in bytes). Handing that to the producer rather than making it fetch a copy of the split beforehand is deliberate: a producer built against one blocking and staged against a different dataset would otherwise fail only mid-write, with part of the file already on the sink.

`write_blocks` returns `&mut Self` like the other writers, so it chains. `Blocking::plan::<T>(dims)` computes the same split from the shape alone if you want it in advance — to size a buffer, or report progress — but no producer needs it.

The element type names the MATLAB class: `write_blocks::<i16>` writes an `int16` array, `write_blocks::<(i16, i16)>` a complex one, `write_blocks::<bool>` a `logical`. The dataset is byte-for-byte the one `write_f64` would have produced for the same content, so a file written this way is reviewable against fixtures built the ordinary way.

Two constraints are worth knowing before designing around this.

**Uncompressed only.** The writer places every object before it emits a byte, so it needs the data region's exact size up front — pure geometry when unfiltered, unknowable without compressing when not. A producer-backed dataset on a builder configured for deflate is refused rather than silently stored uncompressed.

**Element order.** MATLAB is column-major, so a producer emits elements in MATLAB's linear order, first index varying fastest, and each block continues where the previous stopped. That fixes which shape an acquisition should ask for: `[channels, samples]` puts a timestep's channels next to each other, so blocks run forward through time. The transpose, `[samples, channels]`, stores channel 0's entire history before channel 1's, and no producer can emit that as an acquisition proceeds.

A producer that fails partway leaves a partial file on the sink. With a non-seekable sink there is nothing to roll back, so write to a temporary path and rename on success if you need all-or-nothing.

### Errors

| Condition | Error | When |
|---|---|---|
| Builder configured for deflate | `MatError::CompressionUnsupportedForBlocks` | At `write_blocks`, before anything is staged |
| Producer wrote the wrong number of bytes | `MatError::BlockSizeMismatch { block, expected, actual }` | During the write |
| Producer returned an error | that error, verbatim | During the write |

A wrong block length is refused rather than written, because a short or long block shifts every address after it and the result would be a file that fails to open for reasons that no longer point back at the producer.

!!! tip
    The `mat_streaming` example is a complete working version of the above — an acquisition producer, a streamed write, a read-back, and a check that the bytes match the same content written the ordinary way. Run it with `cargo run --example mat_streaming` (no features needed: `MatBuilder` is not behind `serde`).

## Carrying your own error type

Every callback the builder takes returns `Result<(), MatError>`: `DataProducer::block_bytes`, and the nesting closures `MatBuilder::struct_`, `MatBuilder::cell`, `CellWriter::push_with` and their siblings. A crate that writes `.mat` as one of several output formats therefore has to put its own error type through that boundary, and `MatError::Custom(e.to_string())` gets it across as text alone.

`MatError::from_source` carries the error itself instead, so the caller's caller can recover it:

```rust,ignore
let value = self.encode(x).map_err(MatError::from_source)?;
```

It takes a concrete error type or an already-boxed one, and the result answers `Error::source` with what it was given, so `err.source().and_then(|e| e.downcast_ref::<EncodeError>())` reaches the original. `Display` prints the inner error, so nothing about the message changes.

This covers what a callback of yours returns. An error raised by a sink you supplied to `finish_to` or `write` is an `io::Error` the writer met on its own, and it arrives as `MatError::Hdf5(Error::Io(..))` rather than through this variant.

## Hand-built files (low-level conventions)

If you are not using serde, you can apply the MATLAB conventions yourself on top of `FileBuilder`. Two pieces matter: the userblock header and the `MATLAB_class` / `MATLAB_fields` attributes.

### Userblock header

MATLAB expects a 512-byte userblock beginning with the `MATLAB 7.3 MAT-file` signature. Reserve the block with `with_userblock(512)` and hand the header to `with_userblock_content`, which makes it part of the file the writer emits:

```rust
use hdf5_pure::FileBuilder;
use hdf5_pure::mat::userblock;

let mut builder = FileBuilder::new();
builder.with_userblock(512);
builder.with_userblock_content(&userblock::header_block(userblock::DEFAULT_DESCRIPTION));
builder.create_dataset("data").with_f64_data(&[1.0]);

builder.write("hand_built.mat").unwrap();
```

Nothing in a v7.3 userblock depends on the file that follows it, so it can be emitted first rather than patched in afterwards. That is what lets `write` and `finish_to` produce a `.mat` without buffering it: patching the returned bytes only works with `finish`, since the streaming paths have already written the region by the time they return.

### Struct pattern

A MATLAB struct is an HDF5 group carrying `MATLAB_class = "struct"` and a `MATLAB_fields` list naming its fields, with each field a child dataset that carries its own `MATLAB_class`. Use `AttrValue::AsciiString` for the fixed-length ASCII class names and `AttrValue::VarLenAsciiCharArray` for the variable-length field-name array. Not `AttrValue::VarLenAsciiStringArray`, which writes the *standard* variable-length string datatype: MATLAB expects the sequence-of-one-byte-strings shape here.

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

See the [groups and attributes guide](../guide/groups-attributes.md) for more on the `AttrValue` variants used here.
