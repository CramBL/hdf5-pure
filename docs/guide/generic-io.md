[`DatasetBuilder::with_data`](crate::DatasetBuilder::with_data) and
[`Dataset::read`](crate::Dataset::read) support generic dataset I/O through the sealed
[`H5Element`](crate::H5Element) trait. The element type can be a numeric scalar, a fixed byte
array, or a [`FixedAscii`](crate::FixedAscii) string.

The patterns on this page come from [`examples/generic_io.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/generic_io.rs). Run it with:

```console
$ cargo run --example generic_io
```

## The `H5Element` trait

[`H5Element`](crate::H5Element) is implemented for the following types and cannot be implemented
outside this crate:

| Category | Types |
| --- | --- |
| Floating point | `f32`, `f64` |
| Signed integers | `i8`, `i16`, `i32`, `i64` |
| Unsigned integers | `u8`, `u16`, `u32`, `u64` |
| Fixed byte arrays | [`[u8; N]`](prim@array), with nonzero `N` that fits in [`u32`] |
| Fixed ASCII strings | [`FixedAscii<N>`](crate::FixedAscii), with nonzero `N` that fits in [`u32`] |

[`H5Element`](crate::H5Element) is available in the default build with no feature flags. The [ndarray](crate::_guide::ndarray) feature builds on the same bound for its N-dimensional [`with_ndarray`](crate::DatasetBuilder::with_ndarray) / [`read_array`](crate::Dataset::read_array) family.

For numeric types, a generic call uses the datatype, byte order, and conversions of the
per-type method for `T`. Fixed byte arrays use a one-dimensional HDF5 array datatype over unsigned bytes.
The array width is part of the element datatype. The writer stores
[`FixedAscii`](crate::FixedAscii) elements with a null-padded ASCII string datatype.
The constructor validates input bytes.

## Writing with `with_data`

[`DatasetBuilder::with_data(&[T])`](crate::DatasetBuilder::with_data) copies a flat slice of
elements and infers the datatype from `T`. If no shape has been set, the writer infers
`[data.len()]`. An existing shape is retained, including one inferred by an earlier data setter.
Each byte array or [`FixedAscii`](crate::FixedAscii) string is one element, with its width
stored in the datatype.

Because the element type is a generic parameter, one function can store any supported type:

```rust
use hdf5_pure::{FileBuilder, FixedAscii, H5Element};

fn store<T: H5Element>(builder: &mut FileBuilder, name: &str, values: &[T]) {
    builder.create_dataset(name).with_data(values);
}

let mut builder = FileBuilder::new();
store(&mut builder, "u32s", &[1u32, 2, 3]);
store(&mut builder, "i16s", &[-1i16, 0, 7]);
store(&mut builder, "f64s", &[1.5f64, 2.5, 3.5]);
store(&mut builder, "bytes", &[[0u8, 255], [128, 42]]);
let labels = [FixedAscii::<4>::new("ab")?, FixedAscii::new("wxyz")?];
store(&mut builder, "labels", &labels);
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.dataset("u32s")?.read_u32()?, vec![1, 2, 3]);
assert_eq!(file.dataset("bytes")?.read::<[u8; 2]>()?, [[0, 255], [128, 42]]);
assert_eq!(file.dataset("labels")?.read::<FixedAscii<4>>()?, labels);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Reading with `read::<T>()`

[`Dataset::read::<T>()`](crate::Dataset::read) reads elements in row-major order. The element
type is usually inferred from the binding, or can be specified with turbofish:

```rust
# use hdf5_pure::FileBuilder;
# let mut builder = FileBuilder::new();
# builder.create_dataset("u32s").with_data(&[1u32, 2, 3]);
# builder.create_dataset("i16s").with_data(&[-1i16, 0, 7]);
# builder.create_dataset("f64s").with_data(&[1.5f64, 2.5, 3.5]);
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
use hdf5_pure::{File, H5Element};

fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, hdf5_pure::Error> {
    file.dataset(name)?.read::<T>()
}

let counts: Vec<u32> = load(&file, "u32s")?;          // inferred from the binding
let readings = file.dataset("f64s")?.read::<f64>()?;  // named with turbofish
# assert_eq!(counts, vec![1, 2, 3]);
# assert_eq!(readings, vec![1.5, 2.5, 3.5]);
# Ok::<(), hdf5_pure::Error>(())
```

For numeric output types, the reader converts stored values using the rules of
[`read_f64`](crate::Dataset::read_f64) and the other scalar readers. These conversions can lose
precision. Reading into a fixed byte array requires a one-dimensional HDF5 array of full-width
unsigned bytes with the same element width. For [`FixedAscii`](crate::FixedAscii), the reader
requires a null-padded ASCII string datatype of the same width and validates every stored byte.
The reader rejects incompatible datatypes even for an empty dataset.

## Appending fixed-width elements

[`Dataset::append`](crate::Dataset::append), [`AppendBuilder::append`](crate::AppendBuilder::append),
and both buffered appenders accept fixed byte arrays and [`FixedAscii`](crate::FixedAscii) strings.
The writer checks datatype compatibility and preserves the stored datatype and element width.
Array base types and dimensions must match, with byte order ignored for one-byte fixed-point
leaves. ASCII strings require null-padded ASCII storage of the same width.

The following example stages an ASCII append, then appends through both buffered APIs:

```rust
use hdf5_pure::{File, FileBuilder, FixedAscii, MaxExtent};
# let directory = tempfile::tempdir()?;
# let path = directory.path().join("labels.h5");
let mut builder = FileBuilder::new();
builder.create_dataset("labels")
    .with_data(&[FixedAscii::<4>::new("ab")?])
    .with_maxshape(&[MaxExtent::Unlimited])
    .with_chunks(&[4]);
builder.write(&path)?;

let file = File::open_rw(&path)?;
let mut labels = file.dataset("labels")?;
let staged = FixedAscii::<4>::new("cd")?;
labels.append_staged(|append| {
    append.append(&[staged]);
})?;
file.commit()?;
let mut buffered = labels.buffered_appender()?;
buffered.append(&[FixedAscii::<4>::new("ef")?])?;
buffered.finish()?;
let mut owned = labels.into_buffered_appender()?;
owned.append(&[FixedAscii::<4>::new("gh")?])?;
owned.finish()?;
let values = file.dataset("labels")?.read::<FixedAscii<4>>()?;
assert_eq!(values, [
    FixedAscii::new("ab")?, FixedAscii::new("cd")?,
    FixedAscii::new("ef")?, FixedAscii::new("gh")?,
]);
drop(file);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For byte-array input, the immediate writer borrows the flattened slice. Chunk encoding and storage
require their own buffers. ASCII input is encoded per batch. The bounded-memory limits and append
policies are described in [Editing files](crate::_guide::editing#bounded-memory-appends).

## Cross-type coercion

Numeric reads through [`read::<T>()`](crate::Dataset::read) use the widening and truncation
rules of the per-type readers. This example reads an [`i16`] dataset as [`f64`] without losing
precision:

```rust
# use hdf5_pure::{File, FileBuilder, H5Element};
# fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, hdf5_pure::Error> {
#     file.dataset(name)?.read::<T>()
# }
# let mut builder = FileBuilder::new();
# builder.create_dataset("i16s").with_data(&[-1i16, 0, 7]);
# let file = File::from_bytes(builder.finish()?)?;
let offsets: Vec<i16> = load(&file, "i16s")?;   // exact: [-1, 0, 7]
let widened: Vec<f64> = load(&file, "i16s")?;   // widened: [-1.0, 0.0, 7.0]
# assert_eq!(offsets, vec![-1, 0, 7]);
# assert_eq!(widened, vec![-1.0, 0.0, 7.0]);
# Ok::<(), hdf5_pure::Error>(())
```

This mirrors what calling [`read_f64`](crate::Dataset::read_f64) on the same `i16` dataset would produce.

## Why write generic code

[`H5Element`](crate::H5Element) lets one routine accept every supported element type.
The `store` and `load` functions above share this bound.

For N-dimensional arrays built on the same trait, see [N-dimensional arrays](crate::_guide::ndarray). For the per-type write and read APIs, see [Writing files](crate::_guide::writing) and [Reading files](crate::_guide::reading).
