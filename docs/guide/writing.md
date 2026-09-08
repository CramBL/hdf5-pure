# Writing Files

This page covers building HDF5 files with `FileBuilder`: creating datasets from typed Rust slices, attaching attributes, and serializing the result either to memory or to disk. It is the foundation for everything else you write to a file.

!!! tip "Runnable example"
    A complete, self-checking version of this workflow lives in [`examples/quickstart.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/quickstart.rs). Run it with:

    ```bash
    cargo run --example quickstart
    ```

## The `FileBuilder` workflow

A file is assembled with `FileBuilder`. You start one with `FileBuilder::new()`, add datasets and groups, attach attributes, and finally serialize. `create_dataset(name)` returns a `DatasetBuilder` whose typed setters supply both the data and (by default) the shape:

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

builder
    .create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .set_attr("unit", AttrValue::AsciiString("degC".into()));

builder.set_attr("version", AttrValue::I64(2));

builder.write("output.h5").unwrap();
```

`create_dataset` returns a `&mut DatasetBuilder`, so the typed setters chain. The builder owns the dataset until the file is serialized; there is no separate "commit" step per dataset.

## Typed data setters and shape

Each scalar type has a dedicated setter. Calling one sets both the element datatype and the data. The shape defaults to `[len]`, the one-dimensional shape matching the slice length, so `with_shape` is optional for flat 1-D data and only needed when you want a different rank:

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
```

The element type of a dataset comes from the setter you call:

| Method | HDF5 type |
|---|---|
| `with_f64_data` | IEEE 64-bit float |
| `with_f32_data` | IEEE 32-bit float |
| `with_i8_data` / `with_i16_data` / `with_i32_data` / `with_i64_data` | Signed integers (8/16/32/64-bit) |
| `with_u8_data` / `with_u16_data` / `with_u32_data` / `with_u64_data` | Unsigned integers (8/16/32/64-bit) |
| `with_ascii_strings` / `with_strings` | Fixed-width strings, ASCII or UTF-8 |

This is the common subset. Compound, enumeration, array, complex, and object-reference datatypes have their own setters; see [compound and complex types](compound-types.md) for those.

!!! note
    Data is stored row-major (C order), which is what HDF5 uses on disk. When you provide a multi-dimensional `with_shape`, the flat slice is interpreted in row-major order.

## Generic writing over the element type

The typed setters have a generic counterpart, `with_data(&[T])`, bounded by the sealed `H5Element` trait. It infers the datatype from `T`, letting you write code that is generic over any supported scalar:

```rust
use hdf5_pure::{FileBuilder, H5Element};

fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
    fb.create_dataset(name).with_data(values);
}

let mut fb = FileBuilder::new();
store(&mut fb, "counts", &[1u32, 2, 3]);
```

See [Generic I/O](generic-io.md) for the full `with_data` / `read::<T>()` round trip and the list of types implementing `H5Element`.

## Strings

`with_ascii_strings(&[&str])` writes a fixed-width string dataset, sizing the datatype to the longest value and zero-padding the rest. `with_strings` is the same under a UTF-8 charset rather than ASCII. Both return a `Result`, because not every value has a fixed-width HDF5 type to be stored in.

```rust
use hdf5_pure::FileBuilder;

let mut fb = FileBuilder::new();
fb.create_dataset("station")
    .with_ascii_strings(&["north", "s", "east"])
    .unwrap();
```

`with_ascii_strings_sized(&[&str], width)` and `with_strings_sized` declare the width instead of deriving it. Reach for those when the values in hand are not all the values the dataset will hold: a dataset can be extended, and a width taken from the first batch leaves a later, longer string with nowhere to go. A value that does not fit the declared width is refused with `FormatError::FixedStringTooLong` rather than stored as a truncated prefix, and a width of zero with `FormatError::ZeroFixedStringWidth`, since no HDF5 string datatype may be zero bytes wide.

The same values written as an attribute (`AttrValue::AsciiString`, `AttrValue::AsciiStringArray`) reach the file under the same encoding, so a value does not change shape by moving between the two.

For strings that should not share a width at all, `with_vlen_strings` writes a variable-length dataset whose payloads live in the file's global heap. Either kind reads back through `Dataset::read_string`, which dispatches on the datatype and trims the padding for you. See [Variable-length strings](vlen-strings.md).

Other paddings (`NULLTERM`, `SPACEPAD`, as `H5T_C_S1` and `H5T_FORTRAN_S1` carry) are left to `with_raw_data` with a hand-built `Datatype::String`.

## Attributes

Attributes attach metadata to a dataset or to a group. On a dataset, `set_attr` is part of the builder chain; on the file root, `FileBuilder::set_attr` attaches an attribute to the root group:

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

builder
    .create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .set_attr("unit", AttrValue::AsciiString("degC".into()));

// Root-group attribute.
builder.set_attr("version", AttrValue::I64(2));
```

Attribute values are `AttrValue` variants (`F64`, `I64`, `AsciiString`, and others). The full set of variants and their HDF5 encodings is covered under [groups and attributes](groups-attributes.md).

## Groups

`create_group(name)` returns a `GroupBuilder` you populate the same way as the root, then hand back to the file with `add_group`:

```rust
use hdf5_pure::{FileBuilder, AttrValue};

let mut builder = FileBuilder::new();

let mut grp = builder.create_group("sensors");
grp.create_dataset("pressure").with_f32_data(&[101.3, 101.5]);
grp.set_attr("location", AttrValue::AsciiString("lab_a".into()));
builder.add_group(grp.finish());
```

`GroupBuilder::finish()` produces a `FinishedGroup`, which `add_group` inserts into the file. Nested hierarchies and group attributes are covered in detail on the [groups and attributes](groups-attributes.md) page.

## Committed (named) datatypes

`commit_datatype` writes a datatype as an object of its own, the way HDF5's `H5Tcommit` does. Datasets and attributes then *name* it instead of each encoding the type again:

```rust
use hdf5_pure::{AttrValue, FileBuilder, make_i32_type};

let mut builder = FileBuilder::new();
builder.commit_datatype("reading_t", make_i32_type());

builder
    .create_dataset("readings")
    .with_i32_data(&[3, 1, 4])
    .with_committed_datatype("reading_t")
    .set_attr_committed("baseline", AttrValue::I32(0), "reading_t");
```

`h5dump` reports such a dataset as `DATATYPE "/reading_t"`, and every object naming the type shares one object rather than declaring an identical type of its own. This is what netCDF-4 writes for a user-defined type, and what `h5py` writes for `create_dataset(..., dtype=f["reading_t"])`.

`GroupBuilder::commit_datatype` commits a type inside a group; name it by path, as in `with_committed_datatype("sensors/reading_t")`. A leading `/` is accepted.

The naming object still declares its own element type, and the two must agree — `with_i32_data` above against a committed i32. A dataset naming a type it does not match, or a path the file commits nothing at, fails the write rather than producing a file whose element bytes and declared type disagree.

Committed datatypes survive [`repack`](repack.md), but cannot be added to an existing file in place: the in-place engine appends into a fixed layout with nowhere to put the new object. Read them back with `Group::named_datatypes` and `Group::named_datatype`. A name that reaches anything else, a dataset included, is `Error::NotANamedDatatype`.

## Empty and zero-dimension datasets

To create a dataset without supplying data, set the datatype and shape explicitly with `with_dtype` and `with_shape`. This is how you write an empty (zero-length) or zero-dimension (scalar-shaped) dataset:

```rust
use hdf5_pure::{FileBuilder, make_f64_type};

let mut builder = FileBuilder::new();

builder
    .create_dataset("placeholder")
    .with_dtype(make_f64_type())
    .with_shape(&[0]);
```

`with_dtype` takes a `Datatype`, which the crate's `make_*_type` constructors produce (for example `make_f64_type()`).

An empty dataset may also be **chunked and resizable**, which is how you declare a dataset up front and grow it later with [`Dataset::append_staged`](editing.md#appending-to-an-unlimited-dataset):

```rust
use hdf5_pure::{FileBuilder, make_f64_type};

let mut builder = FileBuilder::new();

builder
    .create_dataset("stream")
    .with_dtype(make_f64_type())
    .with_shape(&[0])
    .with_maxshape(&[u64::MAX])
    .with_chunks(&[512]);
```

`with_chunks` is required here rather than optional: auto-chunking derives the chunk from the shape, and a zero-element shape has nothing to derive from. Leaving it out is refused with `FormatError::InvalidChunkGeometry`.

A zero-element shape and staged element data are refused together, with `FormatError::ShapeDataMismatch`: the shape declares nowhere for the data to go. Pass an empty slice or leave the data out entirely, as both examples above do.

## Serializing: `finish()` vs `write(path)`

When the file is fully assembled, choose how to materialize it:

| Method | Returns | Use when |
|---|---|---|
| `finish()` | `Result<Vec<u8>, Error>` | You want the file image in memory (WASM-friendly, no filesystem) |
| `write(path)` | `Result<(), Error>` | You want the file written to disk |
| `finish_to(w)` | `Result<(), Error>` | You want the file on an arbitrary `io::Write` — a socket, a pipe, a compressing wrapper |

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

// In memory: no filesystem touched, just the serialized bytes.
let bytes: Vec<u8> = builder.finish().unwrap();

// Or straight to disk.
// builder.write("output.h5").unwrap();
```

The in-memory `Vec<u8>` is exactly the bytes that `write` would put on disk, so it round-trips through `File::from_bytes`. This is what makes writing usable in environments without a filesystem.

All three produce the same file. `finish` is the only one that holds it: `write` and `finish_to` assemble it front-to-back onto their destination, never seeking, so peak memory does not include the output. `write` is `finish_to` onto a `File`. See [writing without buffering](streaming.md#writing-without-buffering) for what that makes possible.

!!! note
    `FileBuilder` is part of the high-level API gated behind the `std` feature (enabled by default), so both `finish` and `write` require `std`. The difference is the filesystem: `finish` returns the file image in memory and never touches disk, while `write` writes those same bytes to a path.

## Next steps

- [Reading files](reading.md) to load what you wrote back, including from the in-memory bytes.
- [Compression](compression.md) for chunking, deflate, shuffle, LZF, and scale-offset filters.
- [Portability](../interop/portability.md) for how these files interoperate with the reference HDF5 C library, h5py, and MATLAB.
