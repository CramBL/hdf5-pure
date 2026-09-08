# Generic Element I/O

This page covers writing and reading datasets generically over the scalar element type, so you can write one function that serves every supported type instead of reaching for the per-type `with_*_data` / `read_*` family. The entry points are `DatasetBuilder::with_data` and `Dataset::read`, both bounded by the sealed `H5Element` trait.

!!! tip "Runnable example"
    The patterns on this page come from [`examples/generic_io.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/generic_io.rs). Run it with:

    ```bash
    cargo run --example generic_io
    ```

## The `H5Element` trait

`H5Element` is the element bound that lets you read and write datasets generically over the scalar type. It is a *sealed* trait: it is implemented for a fixed set of scalar types and cannot be implemented for any other type. The supported types are:

| Category | Types |
| --- | --- |
| Floating point | `f32`, `f64` |
| Signed integers | `i8`, `i16`, `i32`, `i64` |
| Unsigned integers | `u8`, `u16`, `u32`, `u64` |

!!! note "Feature-independent"
    `H5Element` is available in the default build with no feature flags. It was previously gated behind the `ndarray` feature; it is now feature-independent. The [`ndarray`](ndarray.md) feature still builds on the same bound for its N-dimensional `with_ndarray` / `read_array` family.

Each implementation simply dispatches to the matching per-type method, so a generic read or write has exactly the same datatype, endianness, and conversion behavior as the corresponding `with_*_data` / `read_*` call. For the full set of Rust-to-HDF5 type mappings, see [Data Types](../reference/data-types.md).

## Writing with `with_data`

`DatasetBuilder::with_data(&[T])` sets the dataset's data and datatype from a flat slice of any supported scalar. It is the generic counterpart of the type-specific methods such as [`with_f64_data`](writing.md): it infers the datatype from `T` and, unless `with_shape` has already set one, takes the shape to be the 1-D `[data.len()]`. The builder is returned, so chunking, compression, and attributes can still be chained.

Because the element type is a generic parameter, one function can store any supported type:

```rust
use hdf5_pure::{FileBuilder, H5Element};

fn store<T: H5Element>(builder: &mut FileBuilder, name: &str, values: &[T]) {
    builder.create_dataset(name).with_data(values);
}

let mut builder = FileBuilder::new();
store(&mut builder, "u32s", &[1u32, 2, 3]);
store(&mut builder, "i16s", &[-1i16, 0, 7]);
store(&mut builder, "f64s", &[1.5f64, 2.5, 3.5]);
```

## Reading with `read::<T>()`

`Dataset::read::<T>()` reads the dataset into a `Vec<T>` for any supported scalar `T`, in row-major order. It is the generic counterpart of the type-specific [`read_*`](reading.md) methods. The element type is usually inferred from the binding, or you can name it with turbofish:

```rust
use hdf5_pure::{File, H5Element};

fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, hdf5_pure::Error> {
    file.dataset(name)?.read::<T>()
}

let counts: Vec<u32> = load(&file, "u32s")?;          // inferred from the binding
let readings = file.dataset("f64s")?.read::<f64>()?;  // named with turbofish
```

!!! warning "`T` is the delivery type, not an assertion about storage"
    `read::<T>()` requests delivery *as* `T`; it is not a check that the dataset is stored as `T`. The stored bytes are coerced into `T` using the same rules as `read_f64` and its siblings, so the conversion can be lossy: reading an `f64` dataset as `i32` truncates, and reading an `i32` dataset as `f64` widens. There is no check that `T` matches the on-disk datatype, so pick `T` to match the stored type when you need an exact, lossless read.

## Cross-type coercion

Because `read::<T>()` coerces, requesting a different `T` than the stored type follows the same widening and truncation rules as the per-type readers. The example reads an `i16` dataset back as `f64`, which widens losslessly:

```rust
let offsets: Vec<i16> = load(&file, "i16s")?;   // exact: [-1, 0, 7]
let widened: Vec<f64> = load(&file, "i16s")?;   // widened: [-1.0, 0.0, 7.0]
```

This mirrors what calling `read_f64` on the same `i16` dataset would produce.

## Why write generic code

The typed `with_*_data` / `read_*` methods each name a single concrete type, so a routine that needs to handle several element types must be duplicated per type or dispatched by hand. With `H5Element` as the bound, the same `store` and `load` functions above work for every supported scalar, letting library code stay parametric over the stored element type.

For N-dimensional arrays built on the same trait, see the [ndarray guide](ndarray.md). For the underlying per-type write and read APIs, see [Writing Datasets](writing.md) and [Reading Datasets](reading.md).
