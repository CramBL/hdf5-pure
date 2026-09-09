//! Ergonomic N-dimensional array I/O via the [`ndarray`] crate.
//!
//! Enabled by the `ndarray` feature. This module adds:
//!
//! * [`DatasetBuilder::with_ndarray`](crate::writer::FileBuilder) — write any
//!   [`ndarray`] array (owned or a view) of a supported scalar type, inferring
//!   both the dataset shape and datatype from the array.
//! * [`Dataset::read_array`](crate::Dataset::read_array) /
//!   [`Dataset::read_array_dyn`](crate::Dataset::read_array_dyn) — read a
//!   dataset back into a typed [`ndarray::Array`] (fixed rank) or
//!   [`ndarray::ArrayD`] (runtime rank).
//!
//! # Memory order
//!
//! HDF5 stores dataset elements in row-major (C) order, which is also
//! `ndarray`'s default layout, so reads and writes are a flat copy with no
//! transpose. On write, non-standard-layout inputs (Fortran-order, transposed,
//! or strided views) are repacked into row-major order via
//! [`ndarray::ArrayRef::as_standard_layout`]. Arrays that are already standard
//! layout are borrowed without copying.
//!
//! # Example
//!
//! ```
//! use hdf5_pure::{File, FileBuilder};
//! use ndarray::{array, Array2};
//!
//! let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//!
//! let mut fb = FileBuilder::new();
//! fb.create_dataset("m").with_ndarray(&a);
//! let bytes = fb.finish().unwrap();
//!
//! let file = File::from_bytes(bytes).unwrap();
//! let back: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();
//! assert_eq!(a, back);
//! ```

#![expect(
    rustdoc::private_doc_tests,
    reason = "rustdoc's test mode collects a private item, so this example is compiled and run by `cargo test --doc`"
)]

use ndarray::{Array, ArrayBase, ArrayD, Data, Dimension, IxDyn};

use crate::convert::TryToUsize;
use crate::element::H5Element;
use crate::error::Error;
use crate::reader::Dataset;
use crate::type_builders::DatasetBuilder;

impl Dataset {
    /// Read the dataset into a statically-ranked [`ndarray::Array`].
    ///
    /// The dimensionality `D` (e.g. [`ndarray::Ix2`](tyalias@ndarray::Ix2)) is usually inferred from
    /// the binding, so a call site reads as `let m: Array2<f64> =
    /// ds.read_array()?;`.
    ///
    /// `T` is the type you want the elements *delivered as*, not an assertion
    /// about the dataset's stored type. The stored bytes are coerced into `T`
    /// using the same rules as [`Dataset::read_f64`](crate::Dataset::read_f64)
    /// and its siblings, so the conversion can be lossy: reading an `f64`
    /// dataset as `i32` truncates, and reading an `i32` dataset as `f64`
    /// widens. There is no check that `T` matches the on-disk datatype, so pick
    /// `T` to match the stored type when you need an exact, lossless read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the dataset's runtime rank does not match
    /// `D`. Use [`read_array_dyn`](Self::read_array_dyn) when the rank is not
    /// known at compile time.
    pub fn read_array<T: H5Element, D: Dimension>(&self) -> Result<Array<T, D>, Error> {
        self.read_array_dyn::<T>()?
            .into_dimensionality::<D>()
            .map_err(|e| Error::Shape(e.to_string()))
    }

    /// Read the dataset into a dynamically-ranked [`ndarray::ArrayD`].
    ///
    /// The array's shape is taken from the dataset's dataspace, so this works
    /// for any rank without naming it at compile time. As with
    /// [`read_array`](Self::read_array), `T` is the requested element type and
    /// the stored bytes are coerced into it (possibly lossily) following the
    /// [`Dataset::read_f64`](crate::Dataset::read_f64) conversion rules, with
    /// no check that `T` matches the on-disk datatype.
    pub fn read_array_dyn<T: H5Element>(&self) -> Result<ArrayD<T>, Error> {
        let shape: Vec<usize> = self
            .shape()?
            .iter()
            .map(|&d| d.to_usize())
            .collect::<Result<Vec<_>, _>>()?;
        let data = T::read_from(self)?;
        ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| Error::Shape(e.to_string()))
    }
}

impl DatasetBuilder {
    /// Set the dataset's data, datatype, and shape from an [`ndarray`] array.
    ///
    /// Accepts any owned array or view of a supported scalar type by reference
    /// (e.g. `&Array2<f64>`, `&arr.view()`). The dataset's shape is taken from
    /// the array's shape and its datatype from the element type, so this is the
    /// N-dimensional counterpart of the flat `with_*_data` methods.
    ///
    /// Data is written in row-major (C) order. Inputs that are not already in
    /// standard layout (transposed, Fortran-order, or strided views) are
    /// repacked once into row-major order; standard-layout inputs are used
    /// without copying. The builder is returned so chunking and compression
    /// can be chained, e.g. `b.with_ndarray(&a).with_chunks(&[64, 64])`.
    pub fn with_ndarray<T, S, D>(&mut self, arr: &ArrayBase<S, D>) -> &mut Self
    where
        T: H5Element,
        S: Data<Elem = T>,
        D: Dimension,
    {
        let shape: Vec<u64> = arr.shape().iter().map(|&d| d as u64).collect();
        // `as_standard_layout` borrows when `arr` is already row-major
        // contiguous and copies into a fresh row-major buffer otherwise, so the
        // resulting slice is always the C-order data HDF5 expects.
        let standard = arr.as_standard_layout();
        let flat = standard
            .as_slice()
            .expect("as_standard_layout yields a contiguous standard-layout array");
        T::write_into(self, flat);
        self.shape = Some(shape);
        self
    }
}
