//! Typed dataset I/O for numeric scalars and fixed byte arrays.
//!
//! [`H5Element`] is the element bound for [`DatasetBuilder::with_data`] and [`Dataset::read`].
//! Numeric reads convert stored values to the output scalar type. Reads into fixed byte
//! arrays require a matching array datatype and preserve each element's bytes.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{File, FileBuilder, H5Element};
//!
//! fn round_trip<T: H5Element + PartialEq + std::fmt::Debug>(name: &str, values: &[T]) {
//!     let mut builder = FileBuilder::new();
//!     builder.create_dataset(name).with_data(values);
//!     let file = File::from_bytes(builder.finish().unwrap()).unwrap();
//!     let back: Vec<T> = file.dataset(name).unwrap().read().unwrap();
//!     assert_eq!(values, back.as_slice());
//! }
//!
//! round_trip("ints", &[1i64, 2, 3]);
//! round_trip("floats", &[1.0f32, 2.5, -3.0]);
//! round_trip("bytes", &[[0u8, 255], [128, 42]]);
//! ```

#![expect(
    rustdoc::private_doc_tests,
    reason = "rustdoc's test mode collects a private item, so this example is compiled and run by `cargo test --doc`"
)]

use crate::data_read::{NumericDecoder, NumericReadTarget};
use crate::datatype::Datatype;
use crate::edit::AppendBuilder;
use crate::error::Error;
use crate::reader::Dataset;
use crate::type_builders::DatasetBuilder;

mod fixed_bytes;

mod sealed {
    use crate::{Datatype, Error};

    #[expect(
        unnameable_types,
        reason = "the supertrait that seals `H5Element` has no path outside the crate, so no other crate implements `H5Element`"
    )]
    pub trait Sealed: Sized {
        /// Converts exactly one stored element to `Self`.
        ///
        /// # Errors
        ///
        /// Returns [`Error::Format`] if the datatype is unsupported or the source byte
        /// length differs from its element width.
        fn convert_scalar(raw: &[u8], datatype: &Datatype) -> Result<Self, Error>;
    }
}

/// An element type for typed dataset creation, reads, and appends.
///
/// Implemented for [`f32`], [`f64`], the signed integers [`i8`], [`i16`], [`i32`], and [`i64`],
/// the unsigned integers [`u8`], [`u16`], [`u32`], and [`u64`], and fixed byte arrays
/// ([`[u8; N]`](prim@array)). This trait is sealed and cannot be implemented outside this crate.
///
/// Numeric reads use the conversions of [`Dataset::read_f64`] and the other scalar readers.
/// A byte array is stored as a one-dimensional HDF5 array of unsigned bytes. Reading a byte
/// array requires the same width and base type, including on empty input. Array widths must
/// be nonzero and fit in [`u32`].
///
/// # Examples
///
/// ```
/// use hdf5_pure::{File, FileBuilder};
///
/// let values = [[0u8, 255, 128], [1, 2, 3]];
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("bytes").with_data(&values);
/// let file = File::from_bytes(builder.finish()?)?;
/// let dataset = file.dataset("bytes")?;
/// assert_eq!(dataset.read::<[u8; 3]>()?, values);
/// assert_eq!(dataset.shape()?, [2]);
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
pub trait H5Element: sealed::Sealed + Copy {
    /// Reads every element of `ds` as `Self`, in row-major order.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Dataset::read`].
    ///
    /// # Panics
    ///
    /// Panics under the conditions described by [`Dataset::read`].
    #[doc(hidden)]
    fn read_from(ds: &Dataset) -> Result<Vec<Self>, Error>;

    /// Sets `builder`'s data and datatype from a flat row-major slice.
    #[doc(hidden)]
    fn write_into(builder: &mut DatasetBuilder, data: &[Self]);

    /// Appends a flat row-major slice to `builder` with its element datatype.
    ///
    /// The editor checks the datatype against the dataset when preparing the append.
    #[doc(hidden)]
    fn append_into(builder: &mut AppendBuilder, data: &[Self]);

    /// Encodes one element for storage as a fill value.
    ///
    /// Numeric elements are encoded little-endian. Byte arrays are copied unchanged.
    #[doc(hidden)]
    fn fill_bytes(self) -> Vec<u8>;

    /// Converts stored elements using the same rules as typed dataset reads.
    ///
    /// Numeric values are converted using the byte order, signedness, and precision of `dt`.
    /// Byte arrays require a matching datatype and width. The reader checks `dt` even when
    /// `raw` is empty.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if `dt` is unsupported for `Self`, the byte-array width is
    /// invalid, or `raw` contains an incomplete element.
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
    /// Sets the dataset's datatype from `T` and copies its elements in row-major order.
    ///
    /// If a shape has not been set, the writer infers `[data.len()]`. An existing shape is
    /// retained, including one inferred by an earlier data setter. A byte array is one
    /// element, with its width stored in the datatype. The writer rejects zero-width arrays
    /// and widths that exceed [`u32::MAX`] when preparing the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, H5Element};
    ///
    /// fn store<T: H5Element>(builder: &mut FileBuilder, name: &str, values: &[T]) {
    ///     builder.create_dataset(name).with_data(values);
    /// }
    ///
    /// let mut builder = FileBuilder::new();
    /// store(&mut builder, "bytes", &[[1u8, 2], [3, 4]]);
    /// let file = File::from_bytes(builder.finish()?)?;
    /// assert_eq!(file.dataset("bytes")?.read::<[u8; 2]>()?, [[1, 2], [3, 4]]);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    ///
    /// With the `ndarray` feature, [`with_ndarray`](Self::with_ndarray) also infers the shape
    /// from an N-dimensional array.
    pub fn with_data<T: H5Element>(&mut self, data: &[T]) -> &mut Self {
        T::write_into(self, data);
        self
    }

    /// Sets the value used for unwritten elements of the dataset.
    ///
    /// The writer stores the value in the dataset's datatype and checks its byte width when
    /// preparing the dataset. A width mismatch is reported as
    /// [`FormatError::FillValueSizeMismatch`](crate::FormatError::FillValueSizeMismatch).
    /// Use the dataset's element type to encode the fill value. The default is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder};
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.create_dataset("bytes")
    ///     .with_data(&[[1u8, 2]])
    ///     .with_fill_value([255u8, 0]);
    /// let file = File::from_bytes(builder.finish()?)?;
    /// assert_eq!(file.dataset("bytes")?.fill_value::<[u8; 2]>()?, Some([255, 0]));
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn with_fill_value<T: H5Element>(&mut self, value: T) -> &mut Self {
        self.fill = Some(T::fill_bytes(value));
        self
    }
}

impl Dataset {
    /// Reads the dataset's elements as `T`, in row-major order.
    ///
    /// Numeric reads use the conversions of [`read_f64`](Self::read_f64) and its siblings.
    /// These conversions can lose precision. Reading into a fixed byte array requires a
    /// one-dimensional HDF5 array of full-width unsigned bytes with the same element width.
    /// The reader checks compatibility even for an empty dataset.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the stored datatype is unsupported for `T` or a byte-array
    /// width is invalid. Propagates errors from reading the dataset's metadata and bytes.
    ///
    /// # Panics
    ///
    /// Panics if the output capacity of a single-window read exceeds [`isize::MAX`] bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder};
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.create_dataset("bytes").with_data(&[[0u8, 255], [128, 42]]);
    /// let file = File::from_bytes(builder.finish()?)?;
    /// let values: Vec<[u8; 2]> = file.dataset("bytes")?.read()?;
    /// assert_eq!(values, [[0, 255], [128, 42]]);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn read<T: H5Element>(&self) -> Result<Vec<T>, Error> {
        T::read_from(self)
    }

    /// Returns the dataset's user-defined fill value, converted to `T`.
    ///
    /// Returns `Ok(None)` when no user-defined fill value is stored. The reader converts
    /// a defined value using the rules of [`read`](Self::read). Numeric conversions can lose
    /// precision. Byte arrays require a matching datatype and width.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the Fill Value message is malformed. Returns
    /// [`Error::Format`] if a user-defined fill value is stored and its datatype is unsupported
    /// for `T`. Returns
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

/// A parsed element encoding shared across row windows of a typed read.
pub(crate) trait ElementDecoder<T>: Sized {
    /// Parses the stored datatype for conversion to `T`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the datatype is unsupported for `T`.
    fn parse_decoder(datatype: &Datatype) -> Result<Self, Error>;

    /// Returns the stored byte width used to split input into elements.
    fn decoded_width(datatype: &Datatype) -> u32;

    /// Decodes complete elements and appends them to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if `raw` contains an incomplete or unsupported element.
    ///
    /// # Panics
    ///
    /// Panics if the new output capacity exceeds [`isize::MAX`] bytes.
    fn append_values(&self, raw: &[u8], output: &mut Vec<T>) -> Result<(), Error>;
}

impl<T: NumericReadTarget> ElementDecoder<T> for NumericDecoder<T> {
    fn parse_decoder(datatype: &Datatype) -> Result<Self, Error> {
        Ok(Self::parse(datatype)?)
    }

    fn decoded_width(datatype: &Datatype) -> u32 {
        crate::data_read::effective_numeric(datatype).type_size()
    }

    fn append_values(&self, raw: &[u8], output: &mut Vec<T>) -> Result<(), Error> {
        Ok(self.append(raw, output)?)
    }
}
