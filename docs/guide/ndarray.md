The `ndarray` feature adds ergonomic, rank-generic dataset I/O on top of the [`ndarray`](https://docs.rs/ndarray) crate, so multi-dimensional data round-trips without manually flattening it or tracking shapes. Shape and datatype are taken directly from the array you pass in.

A runnable example lives at [`examples/ndarray_io.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/ndarray_io.rs). Run it with:

```console
$ cargo run --example ndarray_io --features ndarray
```

## Enabling the feature

This page's API is gated behind the `ndarray` feature, which is off by default. Enable it in `Cargo.toml`:

```toml
[dependencies]
hdf5-pure = { version = "0.44", features = ["ndarray"] }
```

The crate depends on `ndarray` with a deliberately permissive version requirement, `>=0.16, <0.18`, so your project's existing `ndarray` unifies with the one this crate uses, and cargo builds a single copy. [Cargo features](crate#cargo-features) has the full list of optional features.

## Writing arrays

[`DatasetBuilder::with_ndarray(&arr)`](crate::DatasetBuilder::with_ndarray) sets a dataset's data, shape, and datatype from a single array. The dataset's rank and dimensions come from the array's shape and its on-disk datatype from the element type, making it the N-dimensional counterpart of the flat `with_*_data` methods covered in [Writing files](crate::_guide::writing).

```rust
# #[cfg(feature = "ndarray")] {
use hdf5_pure::FileBuilder;
use ndarray::{array, Array2};

let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

let mut fb = FileBuilder::new();
fb.create_dataset("m").with_ndarray(&a); // shape [2, 3], f64
let bytes = fb.finish()?;
# assert_eq!(hdf5_pure::File::from_bytes(bytes)?.dataset("m")?.shape()?, vec![2, 3]);
# }
# Ok::<(), hdf5_pure::Error>(())
```

The array can be of any rank, and both owned arrays and views are accepted by reference (e.g. `&Array2<f64>` or `&arr.view()`).

### Memory order and non-standard layouts

HDF5 stores dataset elements in row-major (C) order, which is also `ndarray`'s default layout, so in the common case a write is a flat copy with no transpose. Inputs that are not in standard layout (transposed, Fortran-order, or strided views) are repacked once into row-major order on write, and standard-layout inputs are used without copying. Either way, what you read back matches the logical array you passed in.

```rust
# #[cfg(feature = "ndarray")] {
use hdf5_pure::{File, FileBuilder};
use ndarray::{array, Array2};

let m: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

let mut fb = FileBuilder::new();
fb.create_dataset("mt").with_ndarray(&m.t()); // transposed view
let file = File::from_bytes(fb.finish()?)?;

let transposed: Array2<f64> = file.dataset("mt")?.read_array()?;
assert_eq!(transposed, m.t());
# }
# Ok::<(), hdf5_pure::Error>(())
```

## Reading arrays

There are two read methods, distinguished by how the rank is determined.

| Method | Returns | Rank known at | Use when |
| --- | --- | --- | --- |
| [`read_array::<T, D>()`](crate::Dataset::read_array) | `Array<T, D>` | compile time | the dimensionality is fixed (usually inferred from the binding) |
| [`read_array_dyn::<T>()`](crate::Dataset::read_array_dyn) | `ArrayD<T>` | runtime | the rank is only known at runtime |

[`read_array`](crate::Dataset::read_array) infers the dimensionality `D` from the binding's type, so a call site reads naturally as `let m: Array2<f64> = ds.read_array()?;`. If the dataset's runtime rank does not match `D`, it returns [`Error::Shape`](crate::Error::Shape). Reach for [`read_array_dyn`](crate::Dataset::read_array_dyn) in that case.

```rust
# #[cfg(feature = "ndarray")] {
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("data.h5");
# {
#     use ndarray::{array, Array2};
#     let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
#     let mut fb = hdf5_pure::FileBuilder::new();
#     fb.create_dataset("m").with_ndarray(&a);
#     fb.write(&path)?;
# }
use hdf5_pure::File;
use ndarray::{Array2, ArrayD};

let file = File::open(&path)?;

// Statically known rank: inferred from the binding type.
let m: Array2<f64> = file.dataset("m")?.read_array()?;

// Rank only known at runtime.
let dynamic: ArrayD<f64> = file.dataset("m")?.read_array_dyn()?;
println!("runtime rank: {}", dynamic.ndim());
# assert_eq!(m.dim(), (2, 3));
# assert_eq!(dynamic.ndim(), 2);
# }
# Ok::<(), hdf5_pure::Error>(())
```

For both methods, `T` is the type you want the elements *delivered as*, not an assertion about the stored datatype. The bytes are coerced into `T` using the same rules as the scalar reads described in [Reading files](crate::_guide::reading) and [Generic element I/O](crate::_guide::generic_io), so the conversion can be lossy (reading an `f64` dataset as `i32` truncates). Pick `T` to match the stored type when you need an exact, lossless read.

## Chaining chunking and compression

[`with_ndarray`](crate::DatasetBuilder::with_ndarray) returns the builder, so chunking and compression chain just like they do for the flat write methods. This is the natural way to write a large array compressed:

```rust
# #[cfg(feature = "ndarray")] {
use hdf5_pure::FileBuilder;
use ndarray::Array2;

let a = Array2::<f64>::zeros((1024, 1024));

let mut fb = FileBuilder::new();
fb.create_dataset("big")
    .with_ndarray(&a)
    .with_chunks(&[64, 64])
    .with_deflate(6);
let bytes = fb.finish()?;
# assert!(bytes.len() < 1024 * 1024 * 8);
# }
# Ok::<(), hdf5_pure::Error>(())
```

See [Compression and filters](crate::_guide::compression) for the full set of available filters and how to combine them.
