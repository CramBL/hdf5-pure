//! MATLAB v7.3 (`.mat`) file conventions on top of HDF5.
//!
//! MAT v7.3 files are HDF5 files with MATLAB conventions:
//! - A 512-byte userblock starting with the `MATLAB 7.3 MAT-file...` signature
//! - Every dataset and group carries a `MATLAB_class` attribute (`"double"`,
//!   `"char"`, `"struct"`, ...)
//! - Strings are stored as `uint16` datasets encoding UTF-16LE (legacy `char`
//!   class) or as opaque-class objects via `#subsystem#/MCOS` (modern `string`
//!   class)
//! - 2-D arrays are laid out column-major (Fortran order); HDF5 shape is
//!   `[cols, rows]` so that MATLAB sees the intended `[rows, cols]`
//!
//! This module exposes three layers of API:
//!
//! 1. **Low-level conventions helpers** (always available): [`class::MatClass`],
//!    [`userblock`], [`utf16`], [`identifier`], [`dims`], [`string_object`].
//! 2. **Mid-level builder** (always available): [`builder::MatBuilder`] wraps
//!    [`crate::FileBuilder`] and applies MATLAB conventions automatically.
//!    Use this for one-pass writers that walk a custom value tree (e.g. a
//!    BEVE wire-format walker).
//! 3. **High-level serde** (gated on `feature = "serde"`): [`to_file`],
//!    [`to_bytes`], [`from_file`], [`from_bytes`] for `#[derive(Serialize,
//!    Deserialize)]` types.
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize, Debug, PartialEq)]
//! struct Experiment {
//!     name: String,
//!     trial: u32,
//!     samples: Vec<f64>,
//! }
//!
//! let e = Experiment {
//!     name: "run1".into(),
//!     trial: 3,
//!     samples: vec![1.0, 2.0, 3.0],
//! };
//!
//! let bytes = mat::to_bytes(&e).unwrap();
//! let back: Experiment = mat::from_bytes(&bytes).unwrap();
//! assert_eq!(back, e);
//! # }
//! ```
//!
//! The
//! [`matlab_fixtures`](https://github.com/CramBL/hdf5-pure/blob/main/examples/matlab_fixtures.rs)
//! example writes a directory of `.mat` v7.3 fixtures (scalars, vectors, matrices, strings, nested
//! structs, complex data, cell arrays, and edge shapes) for verification in MATLAB and Octave. Run
//! it with:
//!
//! ```console
//! $ cargo run --example matlab_fixtures --features serde
//! ```
//!
//! # Requires the `serde` feature
//!
//! The high-level `.mat` API is gated on the `serde` feature, which is off by default. [`Matrix`],
//! the `Complex*` types, and [`MatElement`], along with [`to_file`] / [`from_file`], are only
//! available when it is enabled. The features reference has the full list.
//!
//! ```toml
//! [dependencies]
//! hdf5-pure = { version = "0.44", features = ["serde"] }
//! serde = { version = "1", features = ["derive"] }
//! ```
//!
//! # Serializing a struct to `.mat`
//!
//! Any type deriving `serde::Serialize` / `Deserialize` round-trips through [`to_file`] and
//! [`from_file`]. The top-level value must be a struct with named fields, or a `HashMap<String,
//! _>`, and each field becomes a top-level MATLAB variable.
//!
//! ```rust
//! # #[cfg(feature = "serde")] {
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("experiment.mat");
//! use hdf5_pure::mat::{self, Complex64, Matrix};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize, PartialEq, Debug)]
//! struct Experiment {
//!     name: String,
//!     trial: u32,
//!     samples: Vec<f64>,
//!     data: Matrix<f64>,
//!     waveform: Vec<Complex64>,
//!     config: Config,
//! }
//!
//! #[derive(Serialize, Deserialize, PartialEq, Debug)]
//! struct Config { threshold: f64, tag: String }
//!
//! let e = Experiment {
//!     name: "run1".into(), trial: 3,
//!     samples: vec![1.0, 2.0, 3.0],
//!     data: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
//!     waveform: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
//!     config: Config { threshold: 0.5, tag: "prod".into() },
//! };
//!
//! mat::to_file(&e, &path).unwrap();
//! let back: Experiment = mat::from_file(&path).unwrap();
//! assert_eq!(back, e);
//! # }
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! For bytes in place of the filesystem, [`to_bytes`] and [`from_bytes`] take and return a
//! `Vec<u8>` and a `&[u8]`. To write somewhere else entirely, a socket or a compressing wrapper,
//! [`to_writer`] and [`to_writer_with_options`] assemble the file straight onto any `io::Write`
//! without holding it in memory. All are equally subject to the `serde` feature gate.
//!
//! # Type mapping
//!
//! The serializer maps Rust types to HDF5 datasets and the MATLAB classes MATLAB expects on read:
//!
//! | Rust | HDF5 / MATLAB encoding |
//! |---|---|
//! | `f64`, `f32`, `i*`, `u*` | scalar dataset `[1,1]`, `MATLAB_class = "double"` / `"single"` / `"int*"` / `"uint*"` |
//! | `bool` | `uint8` scalar, `MATLAB_class = "logical"` |
//! | `String` / `&str` | `uint16` `[1, N]` UTF-16LE, `MATLAB_class = "char"` |
//! | `Vec<T>` of numeric `T` | MATLAB `[N, 1]` column vector (HDF5 shape `[1, N]`), which [1-D vector orientation](#1-d-vector-orientation) covers |
//! | [`Matrix<T>`](Matrix) or `Vec<Vec<T>>` of same length | column-major 2-D dataset, HDF5 shape `[cols, rows]` |
//! | [`Complex64`] / [`Complex32`] / [`ComplexI16`] / … | compound `{real, imag}` dataset, `MATLAB_class` = the component class |
//! | nested struct | HDF5 group with `MATLAB_class = "struct"`, `MATLAB_fields` |
//! | `Option<T>` (struct field) | `struct([])` if `None`. [`NullPolicy::Omit`] drops the field, and [`NullPolicy::Error`] rejects it |
//! | unit `()` / unit struct / `serde_json::Value::Null` (struct field) | same as `None` (see the note below) |
//! | `None` / `()` / `Null` at the root | a valid file with no variables, byte-identical to what an empty root map or a fieldless struct writes, and [`NullPolicy::Error`] alone rejects it. It does not read back as `None`, since the deserializer presents the root as a struct |
//! | unit enum variant | a UTF-16 char dataset holding the variant name, and [`UnitVariantEncoding::Index`] writes the declaration index as `uint32` |
//! | empty `Vec<T>` | an empty `double` (`[]`), and [`EmptySequencePolicy::Cell`] writes `{}` |
//! | `Vec<Struct>` / `Vec<Option<T>>` / ragged `Vec<Vec<T>>` | a cell array (`MATLAB_class = "cell"`, object references into `#refs#`), where a `None` slot becomes `struct([])` |
//!
//! A struct field that serializes as a Rust unit `()` is written exactly like `Option::None`. The
//! most common case is a `serde_json::Value::Null` field, since `serde_json` serializes
//! `Value::Null` via `serialize_unit`.
//!
//! Under the default [`NullPolicy::EmptyStructArray`] the field is present on disk as MATLAB
//! `struct([])`, so `isfield` reports `true` and MATLAB code can reference it unconditionally and
//! test it with `isempty(fieldnames(x))`. The two forms still differ for a struct with no fields,
//! and the `fieldnames` form is the one verified against MATLAB (see `matlab_fixtures/verify.m`).
//! A bare `isempty(x)` is reliable too: the default [`EmptyMarkerEncoding::DataAsDims`] stores the
//! marker's dimensions in its payload, so the reference library recovers `0x0` with zero elements.
//!
//! Reading it back is lenient and not universal. `struct([])` deserializes into `Option<T>` as
//! `None`, `Vec<T>` as empty, `serde_json::Value` as `Null`, and `()` as `()`. It does not
//! deserialize into a bare scalar, `String`, struct or map: those report a type error.
//! `#[serde(default)]` does not rescue them, because the field is present with a struct value, so
//! the default is never consulted. Give such a field type `Option<u32>`, or write with
//! [`NullPolicy::Omit`].
//!
//! Under [`NullPolicy::Omit`] the field is absent, and reading it back needs `#[serde(default)]`
//! on the field, or an `Option<T>` field, which serde defaults to `None` automatically. A
//! non-`Option` field with no serde default fails to deserialize an omitted field with a
//! missing-field error.
//!
//! ## 1-D vector orientation
//!
//! A `Vec<T>` becomes a MATLAB column vector, MATLAB `[N, 1]`, stored as HDF5 shape `[1, N]`,
//! since HDF5 storage is the transpose of the MATLAB shape. Every 1-D array follows this rule,
//! complex ones and [cell arrays](#cell-arrays) included.
//!
//! [`to_bytes`] is fixed at that default. For MATLAB `[1, N]` rows, write through
//! [`to_bytes_with_options`] with `one_dimensional_mode: OneDimensionalMode::RowVector`:
//!
//! ```rust
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat::{self, OneDimensionalMode, Options};
//! use serde::Serialize;
//!
//! #[derive(Serialize)]
//! struct Capture { samples: Vec<f64> }
//!
//! let mut opts = Options::default();
//! opts.one_dimensional_mode = OneDimensionalMode::RowVector;
//! let capture = Capture { samples: vec![1.0, 2.0, 3.0] };
//! let bytes = mat::to_bytes_with_options(&capture, &opts).unwrap();
//! # assert!(hdf5_pure::is_hdf5_bytes(&bytes[512..]));
//! # }
//! ```
//!
//! This matters when the same data reaches MATLAB by more than one route.
//! [BEVE](https://github.com/beve-org/beve)'s MATLAB loader (`matlab/load_beve.m`) reconstructs a
//! complex array as `complex(raw(1,:), raw(2,:))`, a `1×N` row. A pipeline that writes the same
//! captures as both BEVE and `.mat` sets `RowVector` here, so that the two agree in the MATLAB
//! workspace and no transpose separates them.
//!
//! ## Matrices and the column-major convention
//!
//! Rust is row-major and MATLAB is column-major. The [`Matrix<T>`](Matrix) newtype carries the
//! Rust-side `rows`/`cols` and a row-major `data` vector, and the serializer transposes to
//! column-major byte order and stores the HDF5 dataset with shape `[cols, rows]` so MATLAB sees the
//! intended `rows × cols` matrix. Build one with
//! [`Matrix::from_row_major(rows, cols, data)`](Matrix::from_row_major) (it panics if `data.len()
//! != rows * cols`) or [`Matrix::zeros(rows, cols)`](Matrix::zeros). Read the parts back with
//! [`rows()`](Matrix::rows), [`cols()`](Matrix::cols), [`data()`](Matrix::data), and
//! [`into_data()`](Matrix::into_data).
//!
//! ```rust
//! # #[cfg(feature = "serde")] {
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("matrix.mat");
//! use hdf5_pure::mat::{self, Matrix};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Frame { a: Matrix<f64> }
//!
//! let v = Frame {
//!     a: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
//! };
//! mat::to_file(&v, &path).unwrap();
//! # let back: Frame = mat::from_file(&path).unwrap();
//! # assert_eq!(back.a.data(), v.a.data());
//! # }
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! A bare `Vec<Vec<T>>` whose rows all share a length is recognized as a 2-D matrix too, and
//! [`Matrix`] is the unambiguous API. The element type `T` is bounded by the sealed [`MatElement`]
//! trait, which is implemented for `f32` and `f64`, the 8, 16, 32 and 64-bit signed and unsigned
//! integers, `bool`, and every complex type. The trait is sealed because MAT v7.3 admits this
//! fixed set of numeric classes alone, so a caller cannot implement it for another type.
//!
//! ## Complex numbers
//!
//! There is one complex newtype per component class: [`Complex64`] and [`Complex32`] for the float
//! classes, and [`ComplexI8`], [`ComplexI16`], [`ComplexI32`], [`ComplexI64`] and the `ComplexU*`
//! counterparts for the integer ones. Each is constructed with `ComplexI16::new(re, im)`, or
//! through the `re` and `im` fields directly. A bare value becomes a compound scalar of HDF5 shape
//! `[1, 1]`, and a `Vec<ComplexI16>` a compound dataset of HDF5 shape `[1, N]`, which is a MATLAB
//! column (see [1-D vector orientation](#1-d-vector-orientation)). The on-disk layout is the same
//! `{real, imag}` compound MATLAB uses for complex arrays. The [compound types
//! guide](crate::_guide::compound_types) treats HDF5 compound datasets in depth.
//!
//! `MATLAB_class` states the component class, and nothing complex-specific, which is how MATLAB
//! separates `complex(int16(re), int16(im))` from a complex `double`. Pick the component the data
//! has: a capture that samples as pairs of 16-bit integers takes four bytes per sample as
//! [`ComplexI16`] and eight as [`Complex32`], and the extra four hold nothing.
//!
//! ```rust
//! # #[cfg(feature = "serde")] {
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("capture.mat");
//! use hdf5_pure::mat::{self, ComplexI16};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Capture { samples: Vec<ComplexI16> }
//!
//! let v = Capture {
//!     samples: vec![ComplexI16::new(-32768, 32767), ComplexI16::new(0, -1)],
//! };
//! mat::to_file(&v, &path).unwrap();
//! # let back: Capture = mat::from_file(&path).unwrap();
//! # assert_eq!(back.samples, v.samples);
//! # }
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! One consumer-side caveat bears on the choice of an integer component: MATLAB stores and
//! loads complex integer arrays, and rejects arithmetic on them. `a * b` on two complex `int16`
//! values raises "Complex integer arithmetic is not supported", and the caller has to `double(...)`
//! them first, or use Fixed-Point Designer's `fi` objects. That is a property of MATLAB and not of
//! the file: the array arrives intact, and `isa(x, 'int16')` and `~isreal(x)` both hold. It costs
//! nothing for a capture format stored compactly and widened once at the point of use, which is the
//! case this exists for.
//!
//! Components are never converted between widths, in either direction. An `int16` complex dataset
//! deserializes into [`ComplexI16`] and nothing else: reading it as [`Complex64`] would be lossless
//! and is rejected all the same, because the component width is part of what the file states it
//! holds. Reading a `double` capture into [`ComplexI16`] is rejected for the same reason, and would
//! truncate. A caller holding float data it has established to be exact integers is the one that
//! can decide whether narrowing is meaningful, so it converts before serializing.
//!
//! The same rule is enforced against the file itself: `MATLAB_class` states the component width,
//! the `{real, imag}` compound carries the bytes, and a file where those two disagree is rejected
//! and never decoded. The disagreement is not always visible: a complex `int64` array with no class
//! attribute at all falls back to `double`, whose element size is identical, so the payload length
//! alone does not separate them.
//!
//! ## Large complex arrays
//!
//! Serde has no bulk channel for a sequence, so a `Vec<ComplexI16>` costs one serializer dispatch
//! per sample, around 26 nanoseconds, which for a capture-sized array is close to the whole cost of
//! writing the file. [`complex`] provides a `serialize_with` helper per component class that hands
//! the slice over whole:
//!
//! ```rust
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat::{self, ComplexI16};
//! use serde::Serialize;
//!
//! #[derive(Serialize)]
//! struct Capture {
//!     #[serde(serialize_with = "mat::complex::i16_array")]
//!     samples: Vec<ComplexI16>,
//! }
//! # let bytes = mat::to_bytes(&Capture {
//! #     samples: vec![ComplexI16::new(1, -1), ComplexI16::new(2, -2)],
//! # }).unwrap();
//! # assert!(hdf5_pure::is_hdf5_bytes(&bytes[512..]));
//! # }
//! ```
//!
//! Measured on 64 Mi samples (256 MiB), one write per process: 1.78 s element-wise against 0.06 s.
//! Repeat writes in a warm process reach 0.03 s, so the cold figure is the one to quote for a
//! capture tool that writes once and exits. The bytes are identical either way, and what the
//! helper changes is how long a write takes.
//!
//! The element type is any [`ComplexElement`]: this module's `Complex*` types,
//! `num_complex::Complex<T>` with the `num-complex` feature enabled, or the caller's own.
//! Implementing it is `unsafe` and asserts a layout, `#[repr(C)]`, two components, real first,
//! every byte initialized data, because the helpers read the slice as raw bytes. The component type
//! is part of the bound, so a same-width class cannot slip through: `i32` parts will not compile as
//! `f32_array`.
//!
//! Three things bound where the annotation may be used:
//!
//! - The output is MAT-specific. The slice crosses serde as a byte string, so the same struct
//!   serialized to JSON or another format emits raw bytes and never a sequence, and its own
//!   `Deserialize` will not read that back. Annotate a field only on a type written to `.mat`
//!   alone.
//! - An empty slice keeps its component class. See [empty complex
//!   arrays](#empty-complex-arrays) below, the one case where the annotation changes the file.
//! - The helpers are one-dimensional. A `Matrix<ComplexI16>` or `Vec<Vec<ComplexI16>>` has no bulk
//!   path and still pays per element.
//!
//! Reading has no bulk path of its own, which makes it the slower half of a round trip: a `.mat`
//! deserializes element by element. Where peak memory is the constraint,
//! [`write_blocks`](#writing-more-data-than-fits-in-memory) is the tool, since these helpers build
//! the array in full.
//!
//! ## Empty complex arrays
//!
//! An empty complex array is written here as a zero-element `{real, imag}` compound that keeps its
//! component class, and it round-trips through this crate. MATLAB writes empties differently:
//! `Mat_VarWriteEmpty` stores the dimensions as data under `MATLAB_empty = 1`, keeping the plain
//! class name, and the [`EmptyMarkerEncoding`] option selects that form for real arrays. Complex
//! arrays do not follow the option. libmatio reads both forms, and the MATLAB-native shape for an
//! empty complex array is reached by writing it as an empty real array of the component class.
//!
//! An empty `Vec<ComplexI16>` is the one case where the plain and [bulk](#large-complex-arrays)
//! paths differ. An empty sequence reveals no element, so the plain path cannot recover the
//! component class and falls back on [`Options::empty_sequence_policy`](Options), an empty `double`
//! by default and an empty cell under [`EmptySequencePolicy::Cell`]. The helper is given the class
//! by name and keeps it, matching
//! [`MatBuilder::write_complex_i16`](MatBuilder::write_complex_i16) and the `Matrix<Complex*>`
//! sentinels. Both read back as an empty array, with the shape `0x0` either way.
//!
//! # Cell arrays
//!
//! A sequence whose elements do not unify into a single numeric matrix lowers to a MATLAB cell
//! array. Each element is interned under the conventional `#refs#` group, and
//! the parent dataset stores HDF5 object references with `MATLAB_class = "cell"`. This covers
//! `Vec<Struct>`, `Vec<Option<T>>` with interspersed `None`, nested cells of cells, and ragged
//! `Vec<Vec<T>>`. An `Option::None` slot inside a sequence becomes `struct([])` so every cell slot
//! has a defined MATLAB type.
//!
//! Each interned object also carries `H5PATH`, the absolute path of the object itself, which is
//! what MATLAB writes on all but one of its own: the `canonical empty` placeholder in the MCOS
//! subsystem carries none. Nothing here reads the attribute, since an object reference resolves
//! without it, and MATLAB stamps objects deeper than `#refs#`'s immediate children with a value
//! other than their path, which this crate does not reproduce.
//!
//! ```rust
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Point { x: f64, y: f64 }
//!
//! #[derive(Serialize, Deserialize)]
//! struct Capture {
//!     /// 3x1 cell array of struct.
//!     path: Vec<Point>,
//!     /// 3x1 cell array; the `None` slot becomes `struct([])`.
//!     optionals: Vec<Option<Point>>,
//!     /// Outer 2x1 cell of cells; rows-of-variable-length-records shape.
//!     grid: Vec<Vec<Option<Point>>>,
//!     /// Ragged numerics also fall back to cell rather than erroring.
//!     ragged: Vec<Vec<f64>>,
//! }
//! # let bytes = mat::to_bytes(&Capture {
//! #     path: vec![Point { x: 1.0, y: 2.0 }],
//! #     optionals: vec![Some(Point { x: 3.0, y: 4.0 }), None],
//! #     grid: vec![vec![None], vec![Some(Point { x: 5.0, y: 6.0 })]],
//! #     ragged: vec![vec![1.0], vec![2.0, 3.0]],
//! # }).unwrap();
//! # assert!(hdf5_pure::is_hdf5_bytes(&bytes[512..]));
//! # }
//! ```
//!
//! In MATLAB this loads as `iscell(path) == true`, with elements addressed as `path{1}.x`, and so
//! on. Empty `None` slots load as `struct([])` (`isempty(fieldnames(...))`).
//!
//! Cell arrays load correctly in MATLAB, libmatio (the reference C library), Julia's `MAT.jl`, and
//! Python through `pymatreader` and `hdf5storage`. GNU Octave 11's `load` does not follow object
//! references for v7.3 cells, warning "unknown datatype", so one of the other tools loads such a
//! file.
//!
//! # Struct arrays (reading)
//!
//! A struct array authored in MATLAB (`s(1).x = …; s(2).x = …`) is stored as a `MATLAB_class =
//! "struct"` group whose every field is a dataset of per-element object references, a
//! struct-of-arrays. [`from_file`] and [`from_bytes`] transpose that into an array-of-structs: a
//! `1×N` or `N×1` array reads into `Vec<T>`, and a true `M×N` array into `Vec<Vec<T>>`, where `T`
//! is the caller's own struct. A scalar struct deserializes into a single struct.
//!
//! This is a read-only path. Writing a `Vec<Struct>` from Rust produces a MATLAB cell array (see
//! [Cell arrays](#cell-arrays)) in place of a native struct array, so a `.mat` this crate writes
//! and one MATLAB writes from the same Rust type differ on disk. Both read back into
//! `Vec<Struct>`.
//!
//! # The on-disk format MATLAB's `load` needs
//!
//! MATLAB has not always read HDF5 with the same library. [MathWorks documents the version per
//! release][mathworks-hdf5]:
//!
//! | MATLAB release | HDF5 C library |
//! | --- | --- |
//! | R2024b and later | 1.14.4.3 |
//! | R2024a | 1.10.11 |
//! | R2023b | 1.10.10 |
//! | R2022a to R2023a | 1.10.8 |
//! | R2021b | 1.10.7 |
//! | R2021a and earlier | 1.8.12 |
//!
//! That table gives the library each release links, and it is a separate question from the formats
//! `load` accepts. Taking the two for one is how a caller ships a file their own MATLAB cannot
//! open.
//!
//! The version 3 superblock is an HDF5 1.10 addition, so MATLAB before R2021b cannot open a `.mat`
//! file carrying one at all, neither partially nor with a warning. [`Options::libver`](Options)
//! therefore defaults to `LibVer::V18` and the writer emits a version 2 superblock with version 3
//! data-layout messages: the newest encoding HDF5 1.8 understands, and one every later MATLAB
//! reads as well. Choosing it costs nothing. The message bodies are identical between the two, so
//! the files differ in two version bytes and are the same size.
//!
//! That much follows from the table. This measurement does not, and it is the stronger reason to
//! prefer the 1.8 format:
//!
//! | MATLAB | `H5.get_libversion` | superblock 2 `load` | superblock 3 `load` |
//! | --- | --- | --- | --- |
//! | R2023a Update 1 | 1.10.8 | loads, decodes correctly | *"Not a binary MAT-file"* |
//!
//! R2023a links HDF5 1.10.8, which reads a version 3 superblock perfectly well, and its `load`
//! rejects one all the same. The two files behind that row are byte-identical through the 512-byte
//! userblock and differ in 35 bytes, all of them the superblock version, the data-layout message
//! versions, and the checksums that follow. The format is therefore the only variable, and it is
//! decisive.
//!
//! MathWorks documents the beginning of this. Around R2021b it shipped two libraries at once,
//! 1.10.7 for the `h5read`, `h5disp` and `h5info` interface, with 1.8.12 kept on the MAT v7.3 path
//! to avoid 1.10 regressions, which produces an unusual and recognizable symptom: a file `h5disp`
//! prints happily and `load` rejects. The measurement above puts that split, or something with the
//! same effect, still in force in R2023a, two years on.
//!
//! Why, it does not say. An older library on the MAT path and a MAT reader that caps the superblock
//! version it will accept are both consistent with this result, and separating them would take more
//! than a `load`. The operational conclusion is the same either way: write the 1.8 format, and take
//! the reported library version as no evidence that a 1.10 file is safe.
//!
//! `examples/octave/check_format.m` is what produced that row, and it runs the same check under
//! whichever MATLAB is at hand. It prints its own inputs, the release and the library version,
//! beside the outcome, so a run on another release extends the table above. Run it in MATLAB and
//! not in Octave, whose HDF5 is modern and reads both formats, which leaves it unable to separate
//! them.
//!
//! The one thing that cannot be written that way is compression, which needs chunked storage, whose
//! chunk indices arrived in 1.10. Requesting both is rejected with
//! [`MatError::CompressionNeedsNewerFormat`], since dropping the compression loses what the caller
//! requested and raising the format produces a file MATLAB cannot load:
//!
//! ```rust
//! use hdf5_pure::mat::{Compression, Options};
//! use hdf5_pure::LibVer;
//!
//! let mut options = Options::default();
//! options.compression = Compression::Deflate { level: 6, shuffle: true };
//! options.libver = LibVer::V110; // required for compression; MATLAB `load` cannot read it
//! # assert_eq!(options.libver, LibVer::V110);
//! ```
//!
//! MATLAB itself writes an older format still: a version 0 superblock with v1 symbol-table groups
//! and v1 B-tree chunk indices, which is what `save -v7.3` produces. This crate reads that format
//! and does not write it, so its `.mat` output is not byte-identical to MATLAB's own.
//!
//! # How an empty value is stored
//!
//! MAT v7.3 is not publicly documented, so the particulars here are measured against the genuine
//! MATLAB files vendored under `tests/data/matlab`, and no specification stands behind them. An
//! empty
//! value is a two-element `uint64` dataset whose payload is the dimension vector, marked with
//! `MATLAB_empty`, and it carries no other attribute, `MATLAB_int_decode` included. That attribute
//! states how to read stored integers back as `char` or `logical`, and it has nothing to describe
//! when the payload is a dimension vector. Not one of the 352 empty datasets in those fixtures
//! carries it, for any class.
//!
//! An empty Rust sequence is written as `0x0`, MATLAB's `[]`, whichever [`OneDimensionalMode`] is
//! set: the mode orients a vector, and an empty one has no orientation to preserve. The difference
//! is visible in MATLAB, where `[[], 1]` is `1` while `[zeros(0,1), 1]` is a dimension-mismatch
//! error, and `0x0` is what MATLAB itself overwhelmingly writes. [`MatBuilder::write_empty`] stores
//! whatever dimensions the caller passes, so `zeros(0,1)` stays expressible.
//!
//! # Opaque value classes
//!
//! Reading ([`from_bytes`]) decodes the MCOS opaque value classes `datetime`, `duration`, and
//! `categorical` into the public [`MatDatetime`], [`MatDuration`], and [`MatCategorical`] types
//! (Unix-epoch millisecond instants, durations in milliseconds, and category codes plus names). Any
//! other opaque class (`table`, `containers.Map`, `dictionary`, user `classdef`s, …) is surfaced
//! losslessly as its raw property map, so it still deserializes into a matching struct. Function
//! handles and legacy objects are rejected by name with [`MatError::UnsupportedMatlabClass`].
//!
//! # Not supported (writing)
//!
//! Writing ([`to_bytes`]) does not encode non-unit enum variants, MATLAB `classdef` objects, or
//! `datetime` / `duration` / `categorical` types. Unit enum variants are supported and serialize to
//! a UTF-16 char dataset holding the variant name.
//!
//! # Writing more data than fits in memory
//!
//! [`MatBuilder::write_f64`] and its siblings copy the slice they are given, and
//! [`finish`](MatBuilder::finish) returns the assembled file, so writing a large array costs
//! roughly three times its size at peak: the caller's copy, the builder's, and the file's. Two
//! APIs remove those, independently of each other.
//!
//! [`MatBuilder::finish_to`] (and [`write`](MatBuilder::write), which is
//! [`finish_to`](MatBuilder::finish_to) onto a file) assembles the file straight onto an
//! `io::Write` in ascending-address order, never seeking and never holding the result. It produces
//! byte-for-byte what [`finish`](MatBuilder::finish) returns.
//!
//! [`MatBuilder::write_blocks`] stages a dataset whose bytes the producer emits during the write.
//! It takes a [`DataProducer`], which the writer calls once per block, in order,
//! during emission, and never during layout, which works from the shape alone. Together they write
//! a `.mat` of any size in about one block of memory:
//!
//! ```rust
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("capture.mat");
//! use hdf5_pure::mat::{Block, DataProducer, MatBuilder, MatError, Options};
//!
//! // Each request carries its own contract, so the producer holds no state.
//! struct Samples;
//!
//! impl DataProducer for Samples {
//!     fn block_bytes(&self, block: Block, out: &mut Vec<u8>) -> Result<(), MatError> {
//!         for i in 0..block.elements {
//!             let value = (block.first_element + i) as f64;
//!             out.extend_from_slice(&value.to_le_bytes());
//!         }
//!         Ok(())
//!     }
//! }
//!
//! // [channels, samples]: see "element order" below for why this way round.
//! let dims = [4, 10_000];
//! let mut mb = MatBuilder::new(Options::default());
//! mb.write_blocks::<f64>("samples", &dims, Box::new(Samples)).unwrap();
//! mb.write(&path).unwrap();
//! # assert_eq!(hdf5_pure::File::open(&path)?.dataset("samples")?.shape()?, vec![10_000, 4]);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! A [`Block`] carries its `index`, the `first_element` it starts at in MATLAB's linear order, and
//! how many `elements` to write ([`block.len()`](Block::len) in bytes). Handing that to the
//! producer is deliberate: a producer built against one blocking and staged against a different
//! dataset would otherwise fail only mid-write, with part of the file already on the sink.
//!
//! [`write_blocks`](MatBuilder::write_blocks) returns `&mut Self` like the other writers, so it
//! chains. [`Blocking::plan::<T>(dims)`](Blocking::plan) computes the same split from the shape
//! alone, for a caller that wants it in advance, to size a buffer or to report progress, and no
//! producer needs it.
//!
//! The element type selects the MATLAB class: `write_blocks::<i16>` writes an `int16` array,
//! `write_blocks::<(i16, i16)>` a complex one, `write_blocks::<bool>` a `logical`. The dataset is
//! byte-for-byte the one [`write_f64`](MatBuilder::write_f64) would have produced for the same
//! content, so a file written this way is reviewable against fixtures built the ordinary way.
//!
//! Two constraints bound a design around this.
//!
//! Uncompressed only. The writer places every object before it emits a byte, so it needs the data
//! region's exact size up front: pure geometry when unfiltered, and knowable only by compressing
//! when not. A producer-backed dataset on a builder configured for deflate is rejected.
//!
//! Element order. MATLAB is column-major, so a producer emits elements in MATLAB's linear
//! order, first index varying fastest, and each block continues where the previous stopped. That
//! fixes which shape an acquisition requests: `[channels, samples]` puts a timestep's
//! channels next to each other, so blocks run forward through time. The transpose, `[samples,
//! channels]`, stores channel 0's entire history before channel 1's, and no producer can emit that
//! as an acquisition proceeds.
//!
//! A producer that fails partway leaves a partial file on the sink. With a non-seekable sink there
//! is nothing to roll back, so write to a temporary path and rename on success where the result has
//! to be all-or-nothing.
//!
//! The `mat_streaming` example is a complete working version of the above: an acquisition producer,
//! a streamed write, a read-back, and a check that the bytes match the same content written the
//! ordinary way. Run it with `cargo run --example mat_streaming`, which needs no feature, since
//! [`MatBuilder`] is not behind `serde`.
//!
//! ## Errors
//!
//! | Condition | Error | When |
//! |---|---|---|
//! | Builder configured for deflate | [`MatError::CompressionUnsupportedForBlocks`] | At [`write_blocks`](MatBuilder::write_blocks), before anything is staged |
//! | Producer wrote the wrong number of bytes | [`MatError::BlockSizeMismatch`] | During the write |
//! | Producer returned an error | that error, verbatim | During the write |
//!
//! A wrong block length is rejected, because a short or long block shifts every address after it
//! and the result would be a file that fails to open for reasons that no longer point back at the
//! producer.
//!
//! # Carrying a caller's own error type
//!
//! Every callback the builder takes returns `Result<(), MatError>`: [`DataProducer::block_bytes`],
//! and the nesting closures [`MatBuilder::struct_`], [`MatBuilder::cell`],
//! [`CellWriter::push_with`] and their siblings. A crate that writes `.mat` as one of several
//! output formats therefore has to put its own error type through that boundary, and
//! `MatError::Custom(e.to_string())` gets it across as text alone.
//!
//! [`MatError::from_source`] stores the error itself, so the caller's caller can recover
//! it:
//!
//! ```rust
//! use std::error::Error as StdError;
//! use std::fmt;
//!
//! use hdf5_pure::mat::MatError;
//!
//! #[derive(Debug)]
//! struct EncodeError;
//!
//! impl fmt::Display for EncodeError {
//!     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//!         f.write_str("the value is out of range")
//!     }
//! }
//!
//! impl StdError for EncodeError {}
//!
//! fn encode(value: f64) -> Result<f64, EncodeError> {
//!     if value.is_finite() { Ok(value) } else { Err(EncodeError) }
//! }
//!
//! fn block(value: f64) -> Result<f64, MatError> {
//!     encode(value).map_err(MatError::from_source)
//! }
//!
//! let err = block(f64::NAN).unwrap_err();
//! assert!(err.source().and_then(|e| e.downcast_ref::<EncodeError>()).is_some());
//! ```
//!
//! It takes a concrete error type or an already-boxed one, and the result returns from
//! `Error::source` what it was given, so `err.source().and_then(|e| e.downcast_ref::<EncodeError>())`
//! reaches the original. `Display` prints the inner error, so the message stays as it was.
//!
//! This covers what a caller's own callback returns. An error raised by a sink the caller supplied
//! to [`finish_to`](MatBuilder::finish_to) or [`write`](MatBuilder::write) is an `io::Error` the
//! writer met on its own, and it arrives as `MatError::Hdf5(Error::Io(..))`, outside this
//! variant.
//!
//! # Hand-built files (low-level conventions)
//!
//! Without serde, the MATLAB conventions are applied by hand on top of
//! [`FileBuilder`](crate::FileBuilder). Two pieces matter: the userblock header and the
//! `MATLAB_class` and `MATLAB_fields` attributes.
//!
//! ## Userblock header
//!
//! MATLAB requires a 512-byte userblock beginning with the `MATLAB 7.3 MAT-file` signature. Reserve
//! the block with [`with_userblock(512)`](crate::FileBuilder::with_userblock) and hand the header
//! to [`with_userblock_content`](crate::FileBuilder::with_userblock_content), which makes it part
//! of the file the writer emits:
//!
//! ```rust
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("hand_built.mat");
//! use hdf5_pure::FileBuilder;
//! use hdf5_pure::mat::userblock;
//!
//! let mut builder = FileBuilder::new();
//! builder.with_userblock(512);
//! builder.with_userblock_content(&userblock::header_block(userblock::DEFAULT_DESCRIPTION));
//! builder.create_dataset("data").with_f64_data(&[1.0]);
//!
//! builder.write(&path).unwrap();
//! # assert_eq!(hdf5_pure::File::open(&path)?.dataset("data")?.read_f64()?, vec![1.0]);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! Nothing in a v7.3 userblock depends on the file that follows it, so the writer emits it first
//! and patches nothing in afterwards. That is what lets [`write`](crate::FileBuilder::write) and
//! [`finish_to`](crate::FileBuilder::finish_to) produce a `.mat` without buffering it: patching the
//! returned bytes only works with [`finish`](crate::FileBuilder::finish), since the streaming paths
//! have already written the region by the time they return.
//!
//! ## Struct pattern
//!
//! A MATLAB struct is an HDF5 group carrying `MATLAB_class = "struct"` and a `MATLAB_fields` list
//! of its field names, with each field a child dataset that carries its own `MATLAB_class`. Use
//! [`AttrValue::AsciiString`](crate::AttrValue::AsciiString) for the fixed-length ASCII class names
//! and [`AttrValue::VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray) for the
//! variable-length field-name array. Not
//! [`AttrValue::VarLenAsciiStringArray`](crate::AttrValue::VarLenAsciiStringArray), which writes
//! the standard variable-length string datatype: MATLAB reads the sequence-of-one-byte-strings
//! shape here.
//!
//! ```rust
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("struct.h5");
//! use hdf5_pure::{FileBuilder, AttrValue};
//!
//! let mut builder = FileBuilder::new();
//! let mut grp = builder.create_group("my_struct");
//!
//! let mut fields = Vec::new();
//! for (name, data) in [("x", vec![1.0, 2.0]), ("y", vec![3.0, 4.0])] {
//!     fields.push(name.to_string());
//!     grp.create_dataset(name).with_f64_data(&data)
//!         .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
//! }
//!
//! grp.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
//! grp.set_attr("MATLAB_fields", AttrValue::VarLenAsciiCharArray(fields));
//! builder.add_group(grp.finish());
//! # builder.write(&path)?;
//! # let file = hdf5_pure::File::open(&path)?;
//! # assert_eq!(file.group("my_struct")?.datasets()?, vec!["x", "y"]);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! The [groups and attributes guide](crate::_guide::groups_attributes) has more on the
//! [`AttrValue`](crate::AttrValue) variants used here.
//!
//! [mathworks-hdf5]: https://www.mathworks.com/help/matlab/hdf5-files.html

pub mod class;
pub mod error;
pub mod userblock;
pub mod utf16;

// Convention helpers and builder. Available without serde so downstream
// crates can drive a MatBuilder directly.
pub mod builder;
pub mod dims;
pub mod identifier;
pub mod options;
pub mod producer;
pub mod string_object;
#[cfg(feature = "serde")]
pub(crate) mod transpose;

// Complex/Matrix types and the serde-driven (de)serializer. Only meaningful
// when serde is in scope.
#[cfg(feature = "serde")]
pub mod complex;
#[cfg(feature = "serde")]
pub mod matrix;
// Public Rust views over decoded MCOS opaque value classes (datetime,
// duration, categorical). Deserialized from the reader's decoded form.
#[cfg(feature = "serde")]
pub mod opaque;
#[cfg(feature = "serde")]
pub mod table;
#[cfg(feature = "serde")]
pub(crate) mod value;

#[cfg(feature = "serde")]
pub mod de;
#[cfg(feature = "serde")]
pub mod ser;

pub use builder::{CellWriter, MatBuilder, StructWriter};
pub use class::MatClass;
pub use error::MatError;
pub use options::{
    Compression, EmptyMarkerEncoding, EmptySequencePolicy, InvalidNamePolicy, NullPolicy,
    OneDimensionalMode, Options, RowMajorPolicy, StringClass, UnitVariantEncoding,
    UnsupportedPolicy,
};
pub use producer::{Block, BlockElement, Blocking, DataProducer};

#[cfg(feature = "serde")]
pub use complex::{
    Complex32, Complex64, ComplexElement, ComplexI8, ComplexI16, ComplexI32, ComplexI64, ComplexU8,
    ComplexU16, ComplexU32, ComplexU64,
};
#[cfg(feature = "serde")]
pub use matrix::{MatElement, Matrix};
#[cfg(feature = "serde")]
pub use opaque::{MatCategorical, MatDatetime, MatDuration, MatEnum};
#[cfg(feature = "serde")]
pub use table::{MatColumn, MatTable, MatTimetable};

#[cfg(feature = "serde")]
use serde::Serialize;
#[cfg(feature = "serde")]
use serde::de::DeserializeOwned;

/// Serialize `value` to a MAT v7.3 byte vector.
///
/// The root value must be a struct with named fields. Each field becomes a
/// top-level MATLAB variable.
///
/// # Sequence handling
///
/// Numeric sequences whose elements share a class collapse to row/column vectors or 2-D matrices as before. Any sequence that doesn't (e.g. `Vec<MyStruct>`, `Vec<Option<T>>` with `None` interspersed, ragged `Vec<Vec<f64>>`, or mixed numeric tags) lowers to a MATLAB cell array; each element is interned under the conventional `#refs#` group and the parent dataset stores object references with `MATLAB_class="cell"`. `Option::None` inside a sequence becomes `struct([])` so each cell slot has a defined MATLAB type. Cases that previously errored on a ragged or mixed-type sequence now succeed via this fallback.
#[cfg(feature = "serde")]
pub fn to_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, MatError> {
    ser::to_bytes(value)
}

/// Serialize `value` to the given filesystem path as a MAT v7.3 file. See
/// [`to_bytes`] for details on how heterogeneous sequences are encoded.
///
/// Streams the file to disk, so it produces the same bytes as [`to_bytes`]
/// without ever holding them all at once.
///
/// A value this crate refuses is rejected before `path` is created, so it leaves
/// an existing file there untouched. A failure once writing has begun can still
/// leave a partial file, which is inherent to not buffering; write to a temporary
/// path and rename on success if you need all-or-nothing.
#[cfg(feature = "serde")]
pub fn to_file<T: Serialize + ?Sized, P: AsRef<std::path::Path>>(
    value: &T,
    path: P,
) -> Result<(), MatError> {
    ser::to_path(value, path)
}

/// Serialize `value` as a MAT v7.3 file written straight onto `w`.
///
/// Produces byte-for-byte what [`to_bytes`] returns, but assembles the file in
/// ascending-address order and never seeks, so `w` can be a socket. Peak memory
/// is the staged data plus the file's metadata rather than that plus the whole
/// file.
#[cfg(feature = "serde")]
pub fn to_writer<T: Serialize + ?Sized, W: std::io::Write>(
    value: &T,
    w: W,
) -> Result<(), MatError> {
    ser::to_writer(value, w)
}

/// Like [`to_bytes`] but with explicit options. Use for opting into the
/// modern `string` class, name sanitization, compression, etc.
#[cfg(feature = "serde")]
pub fn to_bytes_with_options<T: Serialize + ?Sized>(
    value: &T,
    options: &Options,
) -> Result<Vec<u8>, MatError> {
    ser::to_bytes_with_options(value, options)
}

/// Like [`to_file`] but with explicit options.
#[cfg(feature = "serde")]
pub fn to_file_with_options<T: Serialize + ?Sized, P: AsRef<std::path::Path>>(
    value: &T,
    path: P,
    options: &Options,
) -> Result<(), MatError> {
    ser::to_path_with_options(value, path, options)
}

/// Like [`to_writer`] but with explicit options.
#[cfg(feature = "serde")]
pub fn to_writer_with_options<T: Serialize + ?Sized, W: std::io::Write>(
    value: &T,
    options: &Options,
    w: W,
) -> Result<(), MatError> {
    ser::to_writer_with_options(value, options, w)
}

/// Deserialize a MAT v7.3 file from a byte slice.
///
/// Numeric arrays, `char` strings, complex values, structs, **cell arrays**, and
/// the modern **`string`** class are reconstructed. A cell array's element
/// references into the hidden `#refs#` group are resolved so the [`to_bytes`]
/// sequence fallbacks (`Vec<Struct>`, ragged `Vec<Vec<T>>`, `Vec<Option<T>>`,
/// nested cells) round-trip; a `string` object's payload is resolved against the
/// `#subsystem#/MCOS` store. MATLAB's reserved `#refs#` / `#subsystem#` groups
/// are not surfaced as variables.
///
/// MCOS opaque value classes are decoded from the `#subsystem#` store:
/// **`datetime`**, **`duration`**, and **`categorical`** deserialize into
/// [`MatDatetime`], [`MatDuration`], and [`MatCategorical`], an **enumeration**
/// deserializes into [`MatEnum`] (its member names), and a **`containers.Map`**
/// deserializes as a `key -> value` map (string/char keys verbatim, numeric keys
/// stringified) straight into a `HashMap<String, V>` / `BTreeMap<String, V>` or a
/// matching struct. Any other opaque class (`dictionary`, user `classdef`s, …) is
/// surfaced losslessly as its raw property map, so it still deserializes into a
/// matching struct. Function handles and legacy objects (`MATLAB_object_decode`
/// 1 / 2) are refused with the typed [`MatError::UnsupportedMatlabClass`].
#[cfg(feature = "serde")]
pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, MatError> {
    de::from_bytes(bytes)
}

/// Deserialize a MAT v7.3 file from the filesystem.
#[cfg(feature = "serde")]
pub fn from_file<T: DeserializeOwned, P: AsRef<std::path::Path>>(path: P) -> Result<T, MatError> {
    let bytes = std::fs::read(path).map_err(MatError::Io)?;
    from_bytes(&bytes)
}
