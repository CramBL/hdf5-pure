# N-dimensional Arrays

The `ndarray` feature adds ergonomic, rank-generic dataset I/O on top of the [`ndarray`](https://docs.rs/ndarray) crate, so multi-dimensional data round-trips without manually flattening it or tracking shapes. Shape and datatype are taken directly from the array you pass in.

!!! tip
    A runnable example lives at [`examples/ndarray_io.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/ndarray_io.rs). Run it with:

    ```bash
    cargo run --example ndarray_io --features ndarray
    ```

## Enabling the feature

This page's API is gated behind the `ndarray` feature, which is off by default. Enable it in `Cargo.toml`:

```toml
[dependencies]
hdf5-pure = { version = "...", features = ["ndarray"] }
```

The crate depends on `ndarray` with a deliberately permissive version requirement, `>=0.16, <0.18`, so your project's existing `ndarray` unifies with the one this crate uses instead of compiling a second, incompatible copy. See the [features reference](../reference/features.md) for the full list of optional features.

## Writing arrays

`DatasetBuilder::with_ndarray(&arr)` sets a dataset's data, shape, and datatype from a single array. The dataset's rank and dimensions come from the array's shape and its on-disk datatype from the element type, making it the N-dimensional counterpart of the flat `with_*_data` methods covered in [Writing](writing.md).

```rust
use hdf5_pure::{File, FileBuilder};
use ndarray::{array, Array2};

let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

let mut fb = FileBuilder::new();
fb.create_dataset("m").with_ndarray(&a); // shape [2, 3], f64
let bytes = fb.finish().unwrap();
```

The array can be of any rank, and both owned arrays and views are accepted by reference (e.g. `&Array2<f64>` or `&arr.view()`).

### Memory order and non-standard layouts

HDF5 stores dataset elements in row-major (C) order, which is also `ndarray`'s default layout, so in the common case a write is a flat copy with no transpose. Inputs that are not in standard layout (transposed, Fortran-order, or strided views) are repacked once into row-major order on write; standard-layout inputs are used without copying. Either way, what you read back matches the logical array you passed in.

```rust
use hdf5_pure::{File, FileBuilder};
use ndarray::{array, Array2};

let m: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

let mut fb = FileBuilder::new();
fb.create_dataset("mt").with_ndarray(&m.t()); // transposed view
let file = File::from_bytes(fb.finish().unwrap()).unwrap();

let transposed: Array2<f64> = file.dataset("mt").unwrap().read_array().unwrap();
assert_eq!(transposed, m.t());
```

## Reading arrays

There are two read methods, distinguished by how the rank is determined.

| Method | Returns | Rank known at | Use when |
| --- | --- | --- | --- |
| `read_array::<T, D>()` | `Array<T, D>` | compile time | the dimensionality is fixed (usually inferred from the binding) |
| `read_array_dyn::<T>()` | `ArrayD<T>` | runtime | the rank is only known at runtime |

`read_array` infers the dimensionality `D` from the binding's type, so a call site reads naturally as `let m: Array2<f64> = ds.read_array()?;`. If the dataset's runtime rank does not match `D`, it returns `Error::Shape`; reach for `read_array_dyn` in that case.

```rust
use hdf5_pure::File;
use ndarray::{Array2, ArrayD};

let file = File::open("data.h5").unwrap();

// Statically known rank: inferred from the binding type.
let m: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();

// Rank only known at runtime.
let dynamic: ArrayD<f64> = file.dataset("m").unwrap().read_array_dyn().unwrap();
println!("runtime rank: {}", dynamic.ndim());
```

!!! note
    For both methods, `T` is the type you want the elements *delivered as*, not an assertion about the stored datatype. The bytes are coerced into `T` using the same rules as the scalar reads described in [Reading](reading.md) and [Generic I/O](generic-io.md), so the conversion can be lossy (reading an `f64` dataset as `i32` truncates). Pick `T` to match the stored type when you need an exact, lossless read.

## Chaining chunking and compression

`with_ndarray` returns the builder, so chunking and compression chain just like they do for the flat write methods. This is the natural way to write a large array compressed:

```rust
use hdf5_pure::FileBuilder;
use ndarray::Array2;

let a = Array2::<f64>::zeros((1024, 1024));

let mut fb = FileBuilder::new();
fb.create_dataset("big")
    .with_ndarray(&a)
    .with_chunks(&[64, 64])
    .with_deflate(6);
let bytes = fb.finish().unwrap();
```

See [Compression](compression.md) for the full set of available filters and how to combine them.
