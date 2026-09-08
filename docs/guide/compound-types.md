# Compound & Complex Types

A compound dataset stores a record of named fields per element, like a C struct or an HDF5 `H5Tcreate(H5T_COMPOUND)` type. This page covers writing and reading compound (struct-like) datasets, the complex-number convention built on top of them, and the related enumeration, fixed-size array, and object-reference dataset kinds.

!!! tip "Runnable example"
    The patterns on this page come from [`examples/compound_types.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/compound_types.rs). Run it with:

    ```bash
    cargo run --example compound_types
    ```

## Numeric tuples as compound records

`with_compound_values(&[tuple])` encodes a slice of numeric tuples field by field. Each field is written one at a time, so the on-disk layout does not depend on Rust's tuple layout, and no struct or tuple padding is copied into the file. Built-in implementations support numeric tuples with one through twelve fields. Field names are position-derived: `"0"`, `"1"`, `"2"`, and so on.

```rust
use hdf5_pure::{File, FileBuilder};

let records = [(1i8, 20u64, 3.5f32), (2, 30, 4.5), (3, 40, 5.5)];

let mut builder = FileBuilder::new();
builder
    .create_dataset("records")
    .with_compound_values(&records)
    .expect("encode compound");

let file = File::from_bytes(builder.finish().unwrap()).unwrap();
```

`read_compound::<(...)>()` reads the records straight back into a matching tuple type. Tuple fields are matched by the same position-derived names (`"0"`, `"1"`, ...) that `with_compound_values` writes:

```rust
let back = file
    .dataset("records")
    .unwrap()
    .read_compound::<(i8, u64, f32)>()
    .unwrap();

assert_eq!(back, records);
```

The on-disk datatype carries the field names and exact byte offsets. `Dataset::datatype` returns the parsed `Datatype` (a `Datatype::Compound` for these records), exposing each field's offset, while `Dataset::read_raw` returns the complete unfiltered record bytes:

```rust
let dtype = file.dataset("records").unwrap().datatype().unwrap();
let raw = file.dataset("records").unwrap().read_raw().unwrap();
```

## Arbitrary compound layouts

For records that are not plain numeric tuples, `with_compound_data(datatype, raw_data, num_elements)` writes an explicit `Datatype` together with the matching little-endian record bytes. You build the layout with `CompoundTypeBuilder`, and `CompoundTypeBuilder::with_size` switches to an `H5Tinsert`-style layout where you place each field at an explicit byte offset (allowing padding between fields):

```rust
use hdf5_pure::CompoundTypeBuilder;

// A 16-byte record: an i32 at offset 0, then an f64 at offset 8 (4 bytes of
// padding between them).
let datatype = CompoundTypeBuilder::with_size(16)
    .i32_field("id", 0)
    .f64_field("value", 8)
    .build()
    .unwrap();
```

The resulting `datatype` is then paired with raw bytes through `with_compound_data`. Because `with_compound_data` writes the bytes verbatim, the caller is responsible for producing little-endian field values at the offsets the datatype declares.

!!! note
    `CompoundTypeBuilder` and its explicit-offset form `ExplicitCompoundTypeBuilder` expose typed field helpers such as `i32_field`, `i64_field`, `f32_field`, `f64_field`, `u8_field` (and the other integer widths), as well as the generic `field(name, byte_offset, datatype)`. The non-explicit `CompoundTypeBuilder::field(name, datatype)` packs fields without manual offsets.

## Complex numbers

Complex numbers are stored as a compound `{real, imag}`, the convention MATLAB and h5py use. `with_complex32_data(&[(f32, f32)])` produces a `{real: f32, imag: f32}` record, and `with_complex64_data(&[(f64, f64)])` produces a `{real: f64, imag: f64}` record:

```rust
let waveform = [(1.0f64, 0.0f64), (0.0, 1.0), (-1.0, 0.0)];
builder
    .create_dataset("waveform")
    .with_complex64_data(&waveform);
```

### Reading complex back (current rough edge)

Reading complex data back is a rough edge worth knowing about. The compound fields are named `real` and `imag`, not the position-derived names (`"0"`, `"1"`) that `read_compound::<(f64, f64)>()` expects, and there is no `read_complex64` convenience yet. The portable path is to read the raw record bytes with `Dataset::read_raw` and decode the little-endian pairs yourself. This is exactly what the crate's [MATLAB reader](../interop/matlab.md) does internally:

```rust
let raw = file.dataset("waveform").unwrap().read_raw().unwrap();
let back_wave = decode_complex64(&raw);
assert_eq!(back_wave, waveform);

/// Decode a `{real: f64, imag: f64}` compound dataset's raw record bytes into
/// `(real, imag)` pairs (16 little-endian bytes per element).
fn decode_complex64(raw: &[u8]) -> Vec<(f64, f64)> {
    raw.chunks_exact(16)
        .map(|rec| {
            let re = f64::from_le_bytes(rec[0..8].try_into().unwrap());
            let im = f64::from_le_bytes(rec[8..16].try_into().unwrap());
            (re, im)
        })
        .collect()
}
```

!!! warning
    Inspect the on-disk datatype with `Dataset::datatype` before decoding so the field widths and offsets match your reader. For `complex32` the records are 8 bytes wide with `f32` fields; for `complex64` they are 16 bytes wide with `f64` fields.

## Enumerations, arrays, and references

Several other structured dataset kinds round out the type system. See the [data types reference](../reference/data-types.md) for the full set; the writing helpers are summarized below.

| Method | HDF5 type |
|---|---|
| `with_enum_i32_data(datatype, values)` | Enumeration with an `i32` base type |
| `with_enum_u8_data(datatype, values)` | Enumeration with a `u8` base type |
| `with_array_data(base_type, array_dims, raw_data, num_elements)` | Fixed-size array elements |
| `with_path_references(paths)` | Object references, resolved by path |

Enumeration datatypes are constructed with `EnumTypeBuilder`. Use `EnumTypeBuilder::i32_based()` or `EnumTypeBuilder::u8_based()`, add named values with `value(name, val)` or `u8_value(name, val)`, and finish with `build()`, which returns a `Result` because a member value has to fit its base type:

```rust
use hdf5_pure::EnumTypeBuilder;

let datatype = EnumTypeBuilder::i32_based()
    .value("Red", 0)
    .value("Green", 1)
    .value("Blue", 2)
    .build()
    .unwrap();

builder
    .create_dataset("colors")
    .with_enum_i32_data(datatype, &[0, 1, 2, 1]);
```

For any other integer base type, use `EnumTypeBuilder::with_base(datatype)` — the enumeration's element size comes from the base type, and `value(name, val)` encodes into that width:

```rust
use hdf5_pure::{EnumTypeBuilder, make_u16_type};

let datatype = EnumTypeBuilder::with_base(make_u16_type())
    .value("Low", 1)
    .value("High", 40_000)
    .build()
    .unwrap();
```

`raw_value(name, bytes)` takes a member's raw little-endian bytes instead, for values wider than an `i64` (a `u64` base near its maximum) or to reproduce stored bytes verbatim. `build()` refuses a non-integer base, a value that does not fit the base type, and raw bytes whose length disagrees with it, rather than writing a datatype message the reference C library cannot read.
```

For more on the writing entry points used throughout this page, see the [writing guide](writing.md).
