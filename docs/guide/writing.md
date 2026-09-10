This page covers building HDF5 files with [`FileBuilder`](crate::FileBuilder): creating datasets from typed Rust slices, attaching attributes, and serializing the result either to memory or to disk. It is the foundation for everything else you write to a file.

A complete, self-checking version of this workflow lives in [`examples/quickstart.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/quickstart.rs). Run it with:

```console
$ cargo run --example quickstart
```

## The `FileBuilder` workflow

A file is assembled with [`FileBuilder`](crate::FileBuilder). You start one with [`FileBuilder::new()`](crate::FileBuilder::new), add datasets and groups, attach attributes, and finally serialize. [`create_dataset(name)`](crate::FileBuilder::create_dataset) returns a [`DatasetBuilder`](crate::DatasetBuilder) whose typed setters supply both the data and (by default) the shape:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

builder
    .create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .set_attr("unit", AttrValue::AsciiString("degC".into()));

builder.set_attr("version", AttrValue::I64(2));

builder.write(&path)?;
# assert!(hdf5_pure::is_hdf5(&path)?);
# Ok::<(), hdf5_pure::Error>(())
```

[`create_dataset`](crate::FileBuilder::create_dataset) returns a `&mut DatasetBuilder`, so the typed setters chain. The builder owns the dataset until the file is serialized: a dataset needs no "commit" step of its own.

## Typed data setters and shape

Each scalar type has a dedicated setter. Calling one sets both the element datatype and the data. The shape defaults to `[len]`, the one-dimensional shape matching the slice length, so [`with_shape`](crate::DatasetBuilder::with_shape) is optional for flat 1-D data and only needed when you want a different rank:

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();

// 1-D: shape defaults to [6].
builder.create_dataset("flat").with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

// 2-D: same six values laid out row-major as [2, 3].
builder
    .create_dataset("grid")
    .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    .with_shape(&[2, 3]);
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("flat")?.shape()?, vec![6]);
# assert_eq!(file.dataset("grid")?.shape()?, vec![2, 3]);
# Ok::<(), hdf5_pure::Error>(())
```

The element type of a dataset comes from the setter you call:

| Method | HDF5 type |
|---|---|
| [`with_f64_data`](crate::DatasetBuilder::with_f64_data) | IEEE 64-bit float |
| [`with_f32_data`](crate::DatasetBuilder::with_f32_data) | IEEE 32-bit float |
| [`with_i8_data`](crate::DatasetBuilder::with_i8_data) / [`with_i16_data`](crate::DatasetBuilder::with_i16_data) / [`with_i32_data`](crate::DatasetBuilder::with_i32_data) / [`with_i64_data`](crate::DatasetBuilder::with_i64_data) | Signed integers (8/16/32/64-bit) |
| [`with_u8_data`](crate::DatasetBuilder::with_u8_data) / [`with_u16_data`](crate::DatasetBuilder::with_u16_data) / [`with_u32_data`](crate::DatasetBuilder::with_u32_data) / [`with_u64_data`](crate::DatasetBuilder::with_u64_data) | Unsigned integers (8/16/32/64-bit) |
| [`with_ascii_strings`](crate::DatasetBuilder::with_ascii_strings) / [`with_strings`](crate::DatasetBuilder::with_strings) | Fixed-width strings, ASCII or UTF-8 |

This is the common subset. Compound, enumeration, array, complex, and object-reference datatypes have their own setters, one class at a time, in the [compound and complex types](crate::_guide::compound_types) guide.

Data is stored row-major (C order), which is what HDF5 uses on disk. When you provide a multi-dimensional [`with_shape`](crate::DatasetBuilder::with_shape), the flat slice is interpreted in row-major order.

## Generic writing over the element type

The typed setters have a generic counterpart, [`with_data(&[T])`](crate::DatasetBuilder::with_data), bounded by the sealed [`H5Element`](crate::H5Element) trait. It infers the datatype from `T`, letting you write code that is generic over any supported scalar:

```rust
use hdf5_pure::{FileBuilder, H5Element};

fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
    fb.create_dataset(name).with_data(values);
}

let mut fb = FileBuilder::new();
store(&mut fb, "counts", &[1u32, 2, 3]);
# let file = hdf5_pure::File::from_bytes(fb.finish()?)?;
# assert_eq!(file.dataset("counts")?.read_u32()?, vec![1, 2, 3]);
# Ok::<(), hdf5_pure::Error>(())
```

The full [`with_data`](crate::DatasetBuilder::with_data) / [`read::<T>()`](crate::Dataset::read) round trip and the types implementing [`H5Element`](crate::H5Element) are in the [generic I/O](crate::_guide::generic_io) guide.

## Strings

[`with_ascii_strings(&[&str])`](crate::DatasetBuilder::with_ascii_strings) writes a fixed-width string dataset, sizing the datatype to the longest value and zero-padding the rest. [`with_strings`](crate::DatasetBuilder::with_strings) writes the same dataset under a UTF-8 charset, where [`with_ascii_strings`](crate::DatasetBuilder::with_ascii_strings) writes ASCII. Both return a `Result`, because not every value has a fixed-width HDF5 type to be stored in.

```rust
use hdf5_pure::FileBuilder;

let mut fb = FileBuilder::new();
fb.create_dataset("station")
    .with_ascii_strings(&["north", "s", "east"])?;
# let file = hdf5_pure::File::from_bytes(fb.finish()?)?;
# assert_eq!(file.dataset("station")?.read_string()?, vec!["north", "s", "east"]);
# Ok::<(), hdf5_pure::Error>(())
```

[`with_ascii_strings_sized(&[&str], width)`](crate::DatasetBuilder::with_ascii_strings_sized) and [`with_strings_sized`](crate::DatasetBuilder::with_strings_sized) take the width from the caller, where the two above derive it from the values. Reach for those when the values in hand are not all the values the dataset will hold: a dataset can be extended, and a width taken from the first batch leaves a later, longer string with nowhere to go. A value that does not fit the declared width is rejected with [`FormatError::FixedStringTooLong`](crate::FormatError::FixedStringTooLong), never stored as a truncated prefix, and a width of zero is rejected with [`FormatError::ZeroFixedStringWidth`](crate::FormatError::ZeroFixedStringWidth), since no HDF5 string datatype may be zero bytes wide.

The same values written as an attribute ([`AttrValue::AsciiString`](crate::AttrValue::AsciiString), [`AttrValue::AsciiStringArray`](crate::AttrValue::AsciiStringArray)) reach the file under the same encoding, so a value does not change shape by moving between the two.

For strings that should not share a width at all, [`with_vlen_strings`](crate::DatasetBuilder::with_vlen_strings) writes a variable-length dataset whose payloads live in the file's global heap. Either kind reads back through [`Dataset::read_string`](crate::Dataset::read_string), which dispatches on the datatype and trims the padding for you. The [variable-length strings](crate::_guide::vlen_strings) guide has the read side in full.

Other paddings (`NULLTERM`, `SPACEPAD`, as `H5T_C_S1` and `H5T_FORTRAN_S1` carry) are left to [`with_raw_data`](crate::DatasetBuilder::with_raw_data) with a hand-built [`Datatype::String`](crate::Datatype::String).

## Attributes

Attributes attach metadata to a dataset or to a group. On a dataset, [`set_attr`](crate::DatasetBuilder::set_attr) is part of the builder chain. On the file root, [`FileBuilder::set_attr`](crate::FileBuilder::set_attr) attaches an attribute to the root group:

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

builder
    .create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .set_attr("unit", AttrValue::AsciiString("degC".into()));

// Root-group attribute.
builder.set_attr("version", AttrValue::I64(2));
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.root().attrs()?["version"], AttrValue::I64(2));
# Ok::<(), hdf5_pure::Error>(())
```

Attribute values are [`AttrValue`](crate::AttrValue) variants ([`F64`](crate::AttrValue::F64), [`I64`](crate::AttrValue::I64), [`AsciiString`](crate::AttrValue::AsciiString), and others). The full set of variants and their HDF5 encodings is in the [groups and attributes](crate::_guide::groups_attributes) guide.

## Groups

[`create_group(name)`](crate::FileBuilder::create_group) returns a [`GroupBuilder`](crate::GroupBuilder) you populate the same way as the root, then hand back to the file with [`add_group`](crate::FileBuilder::add_group):

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

let mut grp = builder.create_group("sensors");
grp.create_dataset("pressure").with_f32_data(&[101.3, 101.5]);
grp.set_attr("location", AttrValue::AsciiString("lab_a".into()));
builder.add_group(grp.finish());
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("sensors/pressure")?.read_f32()?, vec![101.3, 101.5]);
# Ok::<(), hdf5_pure::Error>(())
```

[`GroupBuilder::finish()`](crate::GroupBuilder::finish) produces a [`FinishedGroup`](crate::FinishedGroup), which [`add_group`](crate::FileBuilder::add_group) inserts into the file. Nested hierarchies and group attributes are the subject of the [groups and attributes](crate::_guide::groups_attributes) guide.

## Committed (named) datatypes

[`commit_datatype`](crate::FileBuilder::commit_datatype) writes a datatype as an object of its own, as `H5Tcommit` does. Datasets and attributes then *name* it, so one encoding of the type serves them all:

```rust
use hdf5_pure::{AttrValue, FileBuilder, make_i32_type};

let mut builder = FileBuilder::new();
builder.commit_datatype("reading_t", make_i32_type());

builder
    .create_dataset("readings")
    .with_i32_data(&[3, 1, 4])
    .with_committed_datatype("reading_t")
    .set_attr_committed("baseline", AttrValue::I32(0), "reading_t");
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.root().named_datatypes()?, vec!["reading_t"]);
# assert_eq!(file.dataset("readings")?.read_i32()?, vec![3, 1, 4]);
# Ok::<(), hdf5_pure::Error>(())
```

`h5dump` reports such a dataset as `DATATYPE "/reading_t"`, and every object that refers to the type shares the one object. This is what netCDF-4 writes for a user-defined type, and what `h5py` writes for `create_dataset(..., dtype=f["reading_t"])`.

[`GroupBuilder::commit_datatype`](crate::GroupBuilder::commit_datatype) commits a type inside a group, which a dataset refers to by path, as in [`with_committed_datatype("sensors/reading_t")`](crate::DatasetBuilder::with_committed_datatype). A leading `/` is accepted.

The referring object still declares its own element type, and the two must agree: [`with_i32_data`](crate::DatasetBuilder::with_i32_data) above against a committed i32. A dataset whose element type differs from the type it refers to, or whose committed path the file holds nothing at, fails the write, so the element bytes and the declared type always agree in the file that lands.

Committed datatypes survive [`repack`](crate::repack()), but cannot be added to an existing file in place: the in-place engine appends into a fixed layout with nowhere to put the new object. Read them back with [`Group::named_datatypes`](crate::Group::named_datatypes) and [`Group::named_datatype`](crate::Group::named_datatype). A name that reaches anything else, a dataset included, is [`Error::NotANamedDatatype`](crate::Error::NotANamedDatatype).

## Empty and zero-dimension datasets

To create a dataset without supplying data, set the datatype and a zero-element shape with [`with_dtype`](crate::DatasetBuilder::with_dtype) and [`with_shape`](crate::DatasetBuilder::with_shape). A shape that has elements and no data behind it is rejected with [`FormatError::DatasetMissingData`](crate::FormatError::DatasetMissingData), a zero-dimension (scalar-shaped) dataset included:

```rust
use hdf5_pure::{FileBuilder, make_f64_type};

let mut builder = FileBuilder::new();

builder
    .create_dataset("placeholder")
    .with_dtype(make_f64_type())
    .with_shape(&[0]);
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("placeholder")?.shape()?, vec![0]);
# Ok::<(), hdf5_pure::Error>(())
```

[`with_dtype`](crate::DatasetBuilder::with_dtype) takes a [`Datatype`](crate::Datatype), which the crate's `make_*_type` constructors produce (for example [`make_f64_type()`](crate::make_f64_type)).

An empty dataset may also be **chunked and resizable**, which is how you declare a dataset up front and grow it later with [`Dataset::append_staged`](crate::Dataset::append_staged), as [Editing files](crate::_guide::editing#appending-to-an-unlimited-dataset) shows:

```rust
use hdf5_pure::{FileBuilder, make_f64_type};

let mut builder = FileBuilder::new();

builder
    .create_dataset("stream")
    .with_dtype(make_f64_type())
    .with_shape(&[0])
    .with_maxshape(&[u64::MAX])
    .with_chunks(&[512]);
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("stream")?.chunk_shape()?, Some(vec![512]));
# Ok::<(), hdf5_pure::Error>(())
```

[`with_chunks`](crate::DatasetBuilder::with_chunks) is required here: auto-chunking derives the chunk from the shape, and a zero-element shape has nothing to derive from. A dataset staged without it is rejected with [`FormatError::InvalidChunkGeometry`](crate::FormatError::InvalidChunkGeometry).

A zero-element shape and staged element data are rejected together, with [`FormatError::ShapeDataMismatch`](crate::FormatError::ShapeDataMismatch): the shape declares nowhere for the data to go. Pass an empty slice or leave the data out entirely, as both examples above do.

## Serializing: `finish()` vs `write(path)`

When the file is fully assembled, choose how to materialize it:

| Method | Returns | Use when |
|---|---|---|
| [`finish()`](crate::FileBuilder::finish) | `Result<Vec<u8>, Error>` | You want the file image in memory (WASM-friendly, no filesystem) |
| [`write(path)`](crate::FileBuilder::write) | `Result<(), Error>` | You want the file written to disk |
| [`finish_to(w)`](crate::FileBuilder::finish_to) | `Result<(), Error>` | You want the file on any `io::Write`: a socket, a pipe, a compressing wrapper |

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

// In memory: no filesystem touched, just the serialized bytes. `write(path)`
// and `finish_to(w)` consume the same builder and emit the same bytes.
let bytes: Vec<u8> = builder.finish()?;
# assert!(hdf5_pure::is_hdf5_bytes(&bytes));
# Ok::<(), hdf5_pure::Error>(())
```

The in-memory `Vec<u8>` is exactly the bytes that [`write`](crate::FileBuilder::write) would put on disk, so it round-trips through [`File::from_bytes`](crate::File::from_bytes). This is what makes writing usable in environments without a filesystem.

All three produce the same file. [`finish`](crate::FileBuilder::finish) is the only one that holds it: [`write`](crate::FileBuilder::write) and [`finish_to`](crate::FileBuilder::finish_to) assemble it front-to-back onto their destination, never seeking, so peak memory does not include the output. [`write`](crate::FileBuilder::write) is [`finish_to`](crate::FileBuilder::finish_to) onto a `std::fs::File`. See [writing without buffering](crate::_guide::streaming#writing-without-buffering) for what that makes possible.

[`FileBuilder`](crate::FileBuilder) is part of the high-level API gated behind the `std` feature (enabled by default), so both [`finish`](crate::FileBuilder::finish) and [`write`](crate::FileBuilder::write) require `std`. The difference is the filesystem: [`finish`](crate::FileBuilder::finish) returns the file image in memory and never touches disk, while [`write`](crate::FileBuilder::write) writes those same bytes to a path.

## Next steps

- [Reading files](crate::_guide::reading) loads what you wrote back, including from the in-memory bytes.
- Chunking, deflate, shuffle, LZF, and scale-offset filters are in the [compression](crate::_guide::compression) guide.
- The [portability guide](crate::_guide::portability) walks through how the reference HDF5 C library, h5py, and MATLAB read these files.
