//! Generic, type-parameterized dataset I/O.
//!
//! The [`H5Element`] trait is the element bound that lets you read and write
//! datasets generically over the scalar type, instead of reaching for a
//! type-specific method like [`DatasetBuilder::with_i64_data`] or
//! [`Dataset::read_i64`](crate::Dataset::read_i64). It powers two entry points:
//!
//! * [`DatasetBuilder::with_data`] — write a flat slice of any supported scalar,
//!   inferring the datatype and shape from the slice.
//! * [`Dataset::read`](crate::Dataset::read) — read a dataset back into a
//!   `Vec<T>` for any supported scalar `T`.
//!
//! Both dispatch to the existing per-type methods, so a generic read or write
//! has exactly the same datatype, endianness, and conversion behavior as the
//! corresponding `with_*_data` / `read_*` call.
//!
//! # Example
//!
//! ```
//! use hdf5_pure::{Dataset, Error, FileBuilder, H5Element};
//!
//! // A function generic over the element type — not possible with the
//! // type-specific `with_*_data` / `read_*` methods.
//! fn round_trip<T: H5Element + PartialEq + std::fmt::Debug>(name: &str, values: &[T]) {
//!     let mut fb = FileBuilder::new();
//!     fb.create_dataset(name).with_data(values);
//!     let bytes = fb.finish().unwrap();
//!
//!     let file = hdf5_pure::File::from_bytes(bytes).unwrap();
//!     let back: Vec<T> = file.dataset(name).unwrap().read().unwrap();
//!     assert_eq!(values, back.as_slice());
//! }
//!
//! round_trip("ints", &[1i64, 2, 3]);
//! round_trip("floats", &[1.0f32, 2.5, -3.0]);
//! ```

#![expect(
    rustdoc::private_doc_tests,
    reason = "rustdoc's test mode collects a private item, so this example is compiled and run by `cargo test --doc`"
)]

use crate::data_read::NumericDecoder;
use crate::datatype::Datatype;
use crate::edit::AppendBuilder;
use crate::error::Error;
use crate::reader::Dataset;
use crate::type_builders::DatasetBuilder;

mod sealed {
    use crate::{Datatype, Error};

    #[expect(
        unnameable_types,
        reason = "the supertrait that seals `H5Element` has no path outside the crate, so no other crate implements `H5Element`"
    )]
    pub trait Sealed: Sized {
        /// Converts exactly one stored numeric element to `Self`.
        ///
        /// # Errors
        ///
        /// Returns [`Error::Format`] if the datatype is unsupported or the source byte
        /// length differs from its element width.
        fn convert_scalar(raw: &[u8], datatype: &Datatype) -> Result<Self, Error>;
    }
}

/// A Rust scalar type that can be stored as an HDF5 dataset element.
///
/// This trait is sealed: it is implemented for the fixed set of scalar types
/// the crate's reader and writer support (`f32`, `f64`, the signed integers
/// `i8`/`i16`/`i32`/`i64`, and the unsigned integers `u8`/`u16`/`u32`/`u64`)
/// and cannot be implemented for other types. It is the element bound for the
/// generic [`DatasetBuilder::with_data`] / [`Dataset::read`](crate::Dataset::read)
/// helpers (and, with the `ndarray` feature, the
/// [`with_ndarray`](crate::FileBuilder)/[`read_array`](crate::Dataset::read_array)
/// family), and dispatches to the existing per-type reader/writer methods, so a
/// generic read or write has exactly the same datatype, endianness, and
/// conversion behavior as the corresponding
/// [`Dataset::read_f64`](crate::Dataset::read_f64) /
/// [`DatasetBuilder::with_f64_data`] call.
pub trait H5Element: sealed::Sealed + Copy {
    /// Read every element of `ds` as `Self`, in row-major order.
    #[doc(hidden)]
    fn read_from(ds: &Dataset) -> Result<Vec<Self>, Error>;

    /// Set `builder`'s data and datatype from a flat row-major slice.
    #[doc(hidden)]
    fn write_into(builder: &mut DatasetBuilder, data: &[Self]);

    /// Append a flat row-major slice to `builder`, recording the implied
    /// element datatype for the commit-time datatype-match check.
    #[doc(hidden)]
    fn append_into(builder: &mut AppendBuilder, data: &[Self]);

    /// Encode `self` as the little-endian bytes of one dataset element, the
    /// form a fill value is stored in.
    #[doc(hidden)]
    fn fill_bytes(self) -> Vec<u8>;

    /// Converts stored numeric elements using the same rules as typed dataset reads.
    ///
    /// The reader decodes `raw` using the byte order, signedness, and precision of `dt`, then
    /// converts each value to `Self`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if `dt` is not a supported numeric datatype or `raw` contains
    /// an incomplete element. The reader parses `dt` even when `raw` is empty.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{Datatype, DatatypeByteOrder, FixedPointType, H5Element};
    ///
    /// let datatype = Datatype::FixedPoint(
    ///     FixedPointType::new(2, DatatypeByteOrder::BigEndian, false, 0, 16)?,
    /// );
    /// assert_eq!(u8::convert_raw(&[0, 42], &datatype)?, [42]);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    #[doc(hidden)]
    fn convert_raw(raw: &[u8], dt: &Datatype) -> Result<Vec<Self>, Error>;
}

macro_rules! impl_h5_element {
    ($ty:ty, $read:ident, $write:ident, $append:ident) => {
        impl sealed::Sealed for $ty {
            fn convert_scalar(raw: &[u8], datatype: &Datatype) -> Result<Self, Error> {
                Ok(NumericDecoder::parse(datatype)?.scalar(raw)?.decode()?)
            }
        }
        impl H5Element for $ty {
            fn read_from(ds: &Dataset) -> Result<Vec<Self>, Error> {
                ds.$read()
            }
            fn write_into(builder: &mut DatasetBuilder, data: &[Self]) {
                builder.$write(data);
            }
            fn append_into(builder: &mut AppendBuilder, data: &[Self]) {
                builder.$append(data);
            }
            fn fill_bytes(self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }
            fn convert_raw(raw: &[u8], dt: &Datatype) -> Result<Vec<Self>, Error> {
                Ok(NumericDecoder::parse(dt)?.decode(raw)?)
            }
        }
    };
}

impl_h5_element!(f32, read_f32, with_f32_data, append_f32);
impl_h5_element!(f64, read_f64, with_f64_data, append_f64);
impl_h5_element!(i8, read_i8, with_i8_data, append_i8);
impl_h5_element!(i16, read_i16, with_i16_data, append_i16);
impl_h5_element!(i32, read_i32, with_i32_data, append_i32);
impl_h5_element!(i64, read_i64, with_i64_data, append_i64);
impl_h5_element!(u8, read_u8, with_u8_data, append_u8);
impl_h5_element!(u16, read_u16, with_u16_data, append_u16);
impl_h5_element!(u32, read_u32, with_u32_data, append_u32);
impl_h5_element!(u64, read_u64, with_u64_data, append_u64);

impl DatasetBuilder {
    /// Set the dataset's data and datatype from a flat slice of any supported
    /// scalar type.
    ///
    /// This is the generic counterpart of the type-specific `with_*_data`
    /// methods (e.g. [`with_f64_data`](Self::with_f64_data)): it infers the
    /// datatype from `T` and, unless [`with_shape`](Self::with_shape) has
    /// already set one, takes the shape to be the 1-D `[data.len()]`. The
    /// builder is returned so chunking, compression, and attributes can be
    /// chained.
    ///
    /// Because the element type is a generic parameter, you can write code that
    /// is generic over the stored type:
    ///
    /// ```
    /// use hdf5_pure::{FileBuilder, H5Element};
    ///
    /// fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
    ///     fb.create_dataset(name).with_data(values);
    /// }
    /// ```
    ///
    /// For N-dimensional arrays, see
    /// [`with_ndarray`](crate::FileBuilder) (the `ndarray` feature).
    pub fn with_data<T: H5Element>(&mut self, data: &[T]) -> &mut Self {
        T::write_into(self, data);
        self
    }

    /// Set the dataset's fill value — the value HDF5 reports for elements that
    /// have never been written (for example the unwritten regions of a chunked,
    /// extensible dataset).
    ///
    /// The fill value is stored in the dataset's datatype, so `T` must match the
    /// dataset's element type: its byte width is checked against the datatype
    /// when the file is written, and a mismatch (for example a `u8` fill value on
    /// an `i32` dataset) is refused with
    /// [`FormatError::FillValueSizeMismatch`](crate::FormatError::FillValueSizeMismatch).
    /// Set the data or datatype (e.g. via [`with_i32_data`](Self::with_i32_data)
    /// or [`with_data`](Self::with_data)) so the element type is known.
    ///
    /// Without this call the crate writes HDF5's library-default fill value
    /// (an implicit zero), unchanged from earlier releases.
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder};
    ///
    /// let mut fb = FileBuilder::new();
    /// fb.create_dataset("d")
    ///     .with_i32_data(&[1, 2, 3])
    ///     .with_fill_value(-1_i32);
    /// let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    /// let fv: Option<i32> = file.dataset("d").unwrap().fill_value().unwrap();
    /// assert_eq!(fv, Some(-1));
    /// ```
    pub fn with_fill_value<T: H5Element>(&mut self, value: T) -> &mut Self {
        self.fill = Some(T::fill_bytes(value));
        self
    }
}

impl Dataset {
    /// Read the dataset into a `Vec<T>` for any supported scalar type, in
    /// row-major order.
    ///
    /// This is the generic counterpart of the type-specific `read_*` methods
    /// (e.g. [`read_f64`](Self::read_f64)). The element type is usually inferred
    /// from the binding, so a call site reads as
    /// `let v: Vec<f64> = ds.read()?;`, or you can name it with turbofish:
    /// `ds.read::<i32>()?`.
    ///
    /// `T` is the type you want the elements *delivered as*, not an assertion
    /// about the dataset's stored type. The stored bytes are coerced into `T`
    /// using the same rules as [`read_f64`](Self::read_f64) and its siblings, so
    /// the conversion can be lossy: reading an `f64` dataset as `i32` truncates,
    /// and reading an `i32` dataset as `f64` widens. There is no check that `T`
    /// matches the on-disk datatype, so pick `T` to match the stored type when
    /// you need an exact, lossless read.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying typed read (see
    /// [`read_f64`](Self::read_f64)).
    pub fn read<T: H5Element>(&self) -> Result<Vec<T>, Error> {
        T::read_from(self)
    }

    /// Returns the dataset's user-defined fill value, converted to `T`.
    ///
    /// Returns `Ok(None)` when no user-defined fill value is stored. The reader converts
    /// a defined value using the same numeric rules as [`read`](Self::read), including the
    /// source byte order and lossy conversions. Pick `T` to match the stored datatype for
    /// an exact value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the Fill Value message is malformed. Returns
    /// [`Error::Format`] if a user-defined fill value is stored and the datatype is unsupported
    /// for numeric reads. Returns
    /// [`FormatError::DataSizeMismatch`](crate::FormatError::DataSizeMismatch) through
    /// [`Error::Format`] if the payload length differs from the stored element width.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder};
    ///
    /// let mut fb = FileBuilder::new();
    /// fb.create_dataset("with").with_f64_data(&[1.0]).with_fill_value(9.5_f64);
    /// fb.create_dataset("without").with_f64_data(&[1.0]);
    /// let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    ///
    /// assert_eq!(file.dataset("with").unwrap().fill_value::<f64>().unwrap(), Some(9.5));
    /// assert_eq!(file.dataset("with").unwrap().fill_value::<i32>().unwrap(), Some(9));
    /// assert_eq!(file.dataset("without").unwrap().fill_value::<f64>().unwrap(), None);
    /// ```
    pub fn fill_value<T: H5Element>(&self) -> Result<Option<T>, Error> {
        let Some(bytes) = self.defined_fill_bytes()? else {
            return Ok(None);
        };
        let dt = self.datatype()?;
        Ok(Some(T::convert_scalar(&bytes, &dt)?))
    }
}
