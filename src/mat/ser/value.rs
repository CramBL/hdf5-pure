//! The value tree the serializer builds and an emitter writes.
//!
//! A serialized input lowers to one [`Value`]: a struct group, a cell array, a
//! single dataset, or nothing at all. [`Leaf`] holds the values that write as one
//! dataset, so an emitter that has settled on a dataset takes a `Leaf` and has no
//! group case left to match.
//!
//! The reader builds a tree of its own, [`MatValue`], out of what it finds in a
//! file.
//!
//! [`MatValue`]: crate::mat::value::MatValue

use crate::mat::value::{ComplexNum, ComplexVec, NumVec, ScalarNum};

/// A node of the tree the serializer builds.
#[derive(Debug, PartialEq)]
pub(crate) enum Value {
    /// A MATLAB cell array. Each element is interned under `#refs#`, and the
    /// dataset stores object references in element order.
    ///
    /// A cell array comes from a sequence and carries no shape of its own, so
    /// its shape is the one every other 1-D value has: `[n, 1]` under the
    /// default [`OneDimensionalMode::ColumnVector`], `[1, n]` under
    /// [`RowVector`](crate::mat::OneDimensionalMode::RowVector), and `[0, 0]`
    /// when empty.
    ///
    /// [`OneDimensionalMode::ColumnVector`]: crate::mat::OneDimensionalMode::ColumnVector
    Cell(Vec<Value>),
    /// A value that writes as one dataset.
    Leaf(Leaf),
    /// Nothing to write: a struct drops the field and the root drops the
    /// variable. A `None` lowers to this under [`NullPolicy::Omit`] alone.
    ///
    /// [`NullPolicy::Omit`]: crate::mat::NullPolicy::Omit
    Omit,
    /// A MATLAB struct group, holding its fields in the order they were
    /// serialized.
    Struct(Vec<(String, Value)>),
}

impl Value {
    /// Returns a short description of this value, for an error message.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Value::Cell(_) => "cell array",
            Value::Leaf(leaf) => leaf.kind(),
            Value::Omit => "none",
            Value::Struct(_) => "struct",
        }
    }
}

/// A value that writes as one dataset.
#[derive(Debug, PartialEq)]
pub(crate) enum Leaf {
    /// A complex 2-D matrix in row-major order, written column-major at MATLAB
    /// shape `[rows, cols]`.
    ComplexMatrix {
        rows: usize,
        cols: usize,
        pairs: ComplexVec,
    },
    /// A complex scalar, written as a `[1, 1]` `{real, imag}` compound.
    ComplexScalar(ComplexNum),
    /// A complex 1-D array, oriented by [`Options::one_dimensional_mode`].
    ///
    /// [`Options::one_dimensional_mode`]: crate::mat::Options::one_dimensional_mode
    ComplexVec1D(ComplexVec),
    /// MATLAB's `struct([])`: a `[0, 0]` empty marker of class `struct`, which is
    /// what a `None` lowers to under the default [`NullPolicy::EmptyStructArray`].
    ///
    /// [`NullPolicy::EmptyStructArray`]: crate::mat::NullPolicy::EmptyStructArray
    EmptyStructArray,
    /// A numeric or logical 2-D matrix in row-major order, written column-major
    /// at MATLAB shape `[rows, cols]`.
    Matrix {
        rows: usize,
        cols: usize,
        vec: NumVec,
    },
    /// A numeric or logical scalar, written at MATLAB shape `[1, 1]`.
    Scalar(ScalarNum),
    /// A string, written as a `char` row vector of UTF-16 code units, or as a
    /// `string` object under [`StringClass::String`].
    ///
    /// [`StringClass::String`]: crate::mat::StringClass::String
    String(String),
    /// A numeric or logical 1-D array, oriented by
    /// [`Options::one_dimensional_mode`].
    ///
    /// [`Options::one_dimensional_mode`]: crate::mat::Options::one_dimensional_mode
    Vec1D(NumVec),
}

impl Leaf {
    fn kind(&self) -> &'static str {
        match self {
            Leaf::ComplexMatrix { .. } => "complex matrix",
            Leaf::ComplexScalar(_) => "complex scalar",
            Leaf::ComplexVec1D(_) => "complex vector",
            Leaf::EmptyStructArray => "empty struct array",
            Leaf::Matrix { .. } => "2-D matrix",
            Leaf::Scalar(_) => "scalar",
            Leaf::String(_) => "string",
            Leaf::Vec1D(_) => "1-D vector",
        }
    }
}
