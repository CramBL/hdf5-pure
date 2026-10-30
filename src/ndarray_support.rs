//! Typed N-dimensional array I/O through [`ndarray`].
//!
//! The `ndarray` feature adds [`DatasetBuilder::with_ndarray`], [`Dataset::read_array`], and
//! [`Dataset::read_array_dyn`]. Element types and conversions follow [`H5Element`].
//!
//! The writer copies elements in row-major (C) order. For nonstandard layouts, it first copies
//! the input into a temporary row-major array.
//!
//! # Examples
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

use crate::convert::Narrow;
use crate::element::H5Element;
use crate::error::Error;
use crate::reader::Dataset;
use crate::type_builders::DatasetBuilder;

impl Dataset {
    /// Reads the dataset into an [`ndarray::Array`] with rank `D`.
    ///
    /// The rank `D`, such as [`ndarray::Ix2`](tyalias@ndarray::Ix2), is usually inferred
    /// from the binding: `let values: Array2<f64> = dataset.read_array()?;`.
    ///
    /// Element conversion follows [`Dataset::read`]. Numeric conversions can lose precision.
    /// Byte arrays require a matching array datatype and element width.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the dataset's rank differs from `D` or its dimensions
    /// cannot describe an array of the decoded elements. Propagates errors from
    /// [`Dataset::read`] and reading the shape.
    ///
    /// # Panics
    ///
    /// Panics under the conditions described by [`Dataset::read`].
    pub fn read_array<T: H5Element, D: Dimension>(&self) -> Result<Array<T, D>, Error> {
        self.read_array_dyn::<T>()?
            .into_dimensionality::<D>()
            .map_err(|e| Error::Shape(e.to_string()))
    }

    /// Reads the dataset into an [`ndarray::ArrayD`] with the stored shape.
    ///
    /// Element conversion follows [`Dataset::read`]. Numeric conversions can lose precision.
    /// Byte arrays require a matching array datatype and element width.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the stored dimensions cannot describe an array of the
    /// decoded elements. Propagates errors from [`Dataset::read`] and reading the shape.
    ///
    /// # Panics
    ///
    /// Panics under the conditions described by [`Dataset::read`].
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
    /// Sets the dataset's datatype and shape from an [`ndarray`] array.
    ///
    /// Accepts owned arrays and views by reference. The writer infers the datatype from the
    /// element type and the dataset's shape from the array's dimensions. Byte-array elements
    /// retain their width within the datatype.
    ///
    /// The writer copies elements in row-major (C) order. For nonstandard layouts, it first
    /// copies the input into a temporary row-major array.
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
