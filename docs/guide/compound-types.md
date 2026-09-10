A compound dataset stores a record of named fields per element, like a C struct or an HDF5 `H5Tcreate(H5T_COMPOUND)` type. This page covers writing and reading compound (struct-like) datasets, the complex-number convention built on top of them, and the related enumeration, fixed-size array, and object-reference dataset kinds.

The patterns on this page come from [`examples/compound_types.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/compound_types.rs). Run it with:

```console
$ cargo run --example compound_types
```

## Numeric tuples as compound records

[`with_compound_values(&[tuple])`](crate::DatasetBuilder::with_compound_values) encodes a slice of numeric tuples field by field. Each field is written one at a time, so the on-disk layout does not depend on Rust's tuple layout, and no struct or tuple padding is copied into the file. Built-in implementations support numeric tuples with one through twelve fields. Field names are position-derived: `"0"`, `"1"`, `"2"`, and so on.

```rust
use hdf5_pure::{File, FileBuilder};

let records = [(1i8, 20u64, 3.5f32), (2, 30, 4.5), (3, 40, 5.5)];

let mut builder = FileBuilder::new();
builder
    .create_dataset("records")
    .with_compound_values(&records)?;

let file = File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("records")?.shape()?, vec![3]);
# Ok::<(), hdf5_pure::Error>(())
```

[`read_compound::<(...)>()`](crate::Dataset::read_compound) reads the records straight back into a matching tuple type. Tuple fields are matched by the same position-derived names (`"0"`, `"1"`, ...) that [`with_compound_values`](crate::DatasetBuilder::with_compound_values) writes:

```rust
# use hdf5_pure::{File, FileBuilder};
# let records = [(1i8, 20u64, 3.5f32), (2, 30, 4.5), (3, 40, 5.5)];
# let mut builder = FileBuilder::new();
# builder.create_dataset("records").with_compound_values(&records)?;
# let file = File::from_bytes(builder.finish()?)?;
let back = file.dataset("records")?.read_compound::<(i8, u64, f32)>()?;

assert_eq!(back, records);
# Ok::<(), hdf5_pure::Error>(())
```

The on-disk datatype carries the field names and exact byte offsets. [`Dataset::datatype`](crate::Dataset::datatype) returns the parsed [`Datatype`](crate::Datatype) (a [`Datatype::Compound`](crate::Datatype::Compound) for these records), exposing each field's offset, while [`Dataset::read_raw`](crate::Dataset::read_raw) returns the complete unfiltered record bytes:

```rust
# use hdf5_pure::{Datatype, File, FileBuilder};
# let records = [(1i8, 20u64, 3.5f32), (2, 30, 4.5), (3, 40, 5.5)];
# let mut builder = FileBuilder::new();
# builder.create_dataset("records").with_compound_values(&records)?;
# let file = File::from_bytes(builder.finish()?)?;
let dtype = file.dataset("records")?.datatype()?;
let raw = file.dataset("records")?.read_raw()?;
# assert!(matches!(dtype, Datatype::Compound { .. }));
# assert_eq!(raw.len(), 3 * 13);
# Ok::<(), hdf5_pure::Error>(())
```

## Arbitrary compound layouts

For records that are not plain numeric tuples, [`with_compound_data(datatype, raw_data, num_elements)`](crate::DatasetBuilder::with_compound_data) writes an explicit [`Datatype`](crate::Datatype) together with the matching little-endian record bytes. You build the layout with [`CompoundTypeBuilder`](crate::CompoundTypeBuilder), and [`CompoundTypeBuilder::with_size`](crate::CompoundTypeBuilder::with_size) switches to an `H5Tinsert`-style layout where you place each field at an explicit byte offset (allowing padding between fields):

```rust
use hdf5_pure::CompoundTypeBuilder;

// A 16-byte record: an i32 at offset 0, then an f64 at offset 8 (4 bytes of
// padding between them).
let datatype = CompoundTypeBuilder::with_size(16)
    .i32_field("id", 0)
    .f64_field("value", 8)
    .build()?;
# assert!(matches!(datatype, hdf5_pure::Datatype::Compound { size: 16, .. }));
# Ok::<(), hdf5_pure::FormatError>(())
```

The resulting `datatype` is then paired with raw bytes through [`with_compound_data`](crate::DatasetBuilder::with_compound_data). Because [`with_compound_data`](crate::DatasetBuilder::with_compound_data) writes the bytes verbatim, the caller is responsible for producing little-endian field values at the offsets the datatype declares.

[`CompoundTypeBuilder`](crate::CompoundTypeBuilder) and its explicit-offset form [`ExplicitCompoundTypeBuilder`](crate::ExplicitCompoundTypeBuilder) expose typed field helpers such as [`i32_field`](crate::CompoundTypeBuilder::i32_field), [`i64_field`](crate::CompoundTypeBuilder::i64_field), [`f32_field`](crate::CompoundTypeBuilder::f32_field), [`f64_field`](crate::CompoundTypeBuilder::f64_field), [`u8_field`](crate::CompoundTypeBuilder::u8_field) (and the other integer widths), as well as the generic [`field(name, byte_offset, datatype)`](crate::ExplicitCompoundTypeBuilder::field). The non-explicit [`CompoundTypeBuilder::field(name, datatype)`](crate::CompoundTypeBuilder::field) packs fields without manual offsets.

## Complex numbers

Complex numbers are stored as a compound `{real, imag}`, the convention MATLAB and h5py use. [`with_complex32_data(&[(f32, f32)])`](crate::DatasetBuilder::with_complex32_data) produces a `{real: f32, imag: f32}` record, and [`with_complex64_data(&[(f64, f64)])`](crate::DatasetBuilder::with_complex64_data) produces a `{real: f64, imag: f64}` record:

```rust
# use hdf5_pure::FileBuilder;
# let mut builder = FileBuilder::new();
let waveform = [(1.0f64, 0.0f64), (0.0, 1.0), (-1.0, 0.0)];
builder
    .create_dataset("waveform")
    .with_complex64_data(&waveform);
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("waveform")?.read_raw()?.len(), 3 * 16);
# Ok::<(), hdf5_pure::Error>(())
```

### Reading complex numbers back

The compound fields of a complex dataset are called `real` and `imag`, and [`read_compound::<(f64, f64)>()`](crate::Dataset::read_compound) matches fields by the position-derived names (`"0"`, `"1"`), so it does not reach them. Read the raw record bytes with [`Dataset::read_raw`](crate::Dataset::read_raw) and decode the little-endian pairs, which is what the crate's MATLAB reader does:

```rust
# use hdf5_pure::{File, FileBuilder};
# let waveform = [(1.0f64, 0.0f64), (0.0, 1.0), (-1.0, 0.0)];
# let mut builder = FileBuilder::new();
# builder.create_dataset("waveform").with_complex64_data(&waveform);
# let file = File::from_bytes(builder.finish()?)?;
let raw = file.dataset("waveform")?.read_raw()?;
let back_wave = decode_complex64(&raw);
assert_eq!(back_wave, waveform);

/// Decode a `{real: f64, imag: f64}` compound dataset's raw record bytes into
/// `(real, imag)` pairs (16 little-endian bytes per element).
fn decode_complex64(raw: &[u8]) -> Vec<(f64, f64)> {
    raw.chunks_exact(16)
        .map(|rec| {
            let re = f64::from_le_bytes(rec[0..8].try_into().expect("8 of 16 bytes"));
            let im = f64::from_le_bytes(rec[8..16].try_into().expect("8 of 16 bytes"));
            (re, im)
        })
        .collect()
}
# Ok::<(), hdf5_pure::Error>(())
```

**Inspect the on-disk datatype with [`Dataset::datatype`](crate::Dataset::datatype) before decoding**, so the field widths and offsets match your reader. A `complex32` record is 8 bytes wide with `f32` fields, and a `complex64` record 16 bytes wide with `f64` fields.

## Enumerations, arrays, and references

Several other structured dataset kinds round out the type system. The data types reference has the full set, and the writing helpers are summarized below.

| Method | HDF5 type |
|---|---|
| [`with_enum_i32_data(datatype, values)`](crate::DatasetBuilder::with_enum_i32_data) | Enumeration with an `i32` base type |
| [`with_enum_u8_data(datatype, values)`](crate::DatasetBuilder::with_enum_u8_data) | Enumeration with a `u8` base type |
| [`with_array_data(base_type, array_dims, raw_data, num_elements)`](crate::DatasetBuilder::with_array_data) | Fixed-size array elements |
| [`with_path_references(paths)`](crate::DatasetBuilder::with_path_references) | Object references, resolved by path |

Enumeration datatypes are constructed with [`EnumTypeBuilder`](crate::EnumTypeBuilder). Use [`EnumTypeBuilder::i32_based()`](crate::EnumTypeBuilder::i32_based) or [`EnumTypeBuilder::u8_based()`](crate::EnumTypeBuilder::u8_based), add named values with [`value(name, val)`](crate::EnumTypeBuilder::value) or [`u8_value(name, val)`](crate::EnumTypeBuilder::u8_value), and finish with [`build()`](crate::EnumTypeBuilder::build), which returns a `Result` because a member value has to fit its base type:

```rust
# use hdf5_pure::FileBuilder;
# let mut builder = FileBuilder::new();
use hdf5_pure::EnumTypeBuilder;

let datatype = EnumTypeBuilder::i32_based()
    .value("Red", 0)
    .value("Green", 1)
    .value("Blue", 2)
    .build()?;

builder
    .create_dataset("colors")
    .with_enum_i32_data(datatype, &[0, 1, 2, 1]);
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("colors")?.read_i32()?, vec![0, 1, 2, 1]);
# Ok::<(), hdf5_pure::Error>(())
```

For any other integer base type, use [`EnumTypeBuilder::with_base(datatype)`](crate::EnumTypeBuilder::with_base): the enumeration's element size comes from the base type, and [`value(name, val)`](crate::EnumTypeBuilder::value) encodes into that width:

```rust
use hdf5_pure::{EnumTypeBuilder, make_u16_type};

let datatype = EnumTypeBuilder::with_base(make_u16_type())
    .value("Low", 1)
    .value("High", 40_000)
    .build()?;
# assert!(matches!(datatype, hdf5_pure::Datatype::Enumeration { .. }));
# Ok::<(), hdf5_pure::FormatError>(())
```

[`raw_value(name, bytes)`](crate::EnumTypeBuilder::raw_value) takes a member's raw little-endian bytes, for values wider than an `i64` (a `u64` base near its maximum) or to reproduce stored bytes verbatim. [`build()`](crate::EnumTypeBuilder::build) rejects a non-integer base, a value that does not fit the base type, and raw bytes whose length disagrees with it. Each of the three would produce a datatype message the reference C library cannot read.

For more on the writing entry points used throughout this page, see [Writing files](crate::_guide::writing).
