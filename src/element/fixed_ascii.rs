//! Fixed-width ASCII elements with null padding.

use core::mem;
use core::str;

use crate::convert::Narrow;
use crate::datatype::{self, CharacterSet, Datatype, StringPadding};
use crate::edit::AppendBuilder;
use crate::error::{Error, FormatError};
use crate::reader::Dataset;
use crate::type_builders::{self, DatasetBuilder};

use super::{ElementDecoder, H5Element, sealed};

/// A null-padded ASCII string with `N` bytes of storage.
///
/// Every stored byte is ASCII. The width must be nonzero and fit in [`u32`]. Embedded null
/// bytes are preserved. [`as_bytes`](Self::as_bytes) includes the padding, and
/// [`as_str`](Self::as_str) removes trailing null bytes.
///
/// The writer stores [`H5Element`] values of this type with a fixed-width HDF5 string datatype,
/// [`CharacterSet::Ascii`], and [`StringPadding::NullPad`]. Typed reads require that datatype
/// and the same width, including for an empty dataset, and reject non-ASCII stored bytes.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{File, FileBuilder, FixedAscii};
///
/// let values = [FixedAscii::<4>::new("ab")?, FixedAscii::new("wxyz")?];
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("labels").with_data(&values);
/// let file = File::from_bytes(builder.finish()?)?;
/// let labels = file.dataset("labels")?.read::<FixedAscii<4>>()?;
/// assert_eq!(labels, values);
/// assert_eq!(labels[0].as_bytes(), b"ab\0\0");
/// assert_eq!(labels[0].as_str(), "ab");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FixedAscii<const N: usize> {
    bytes: [u8; N],
}

impl<const N: usize> FixedAscii<N> {
    /// Copies ASCII bytes into the fixed-width storage and pads the remainder with nulls.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroFixedStringWidth`] if `N` is zero, or
    /// [`FormatError::ValueTooLargeForRepresentation`] if `N` exceeds [`u32::MAX`]. Returns
    /// [`FormatError::FixedStringTooLong`] if `value` exceeds `N` bytes, or
    /// [`FormatError::InvalidAscii`] if a byte is outside ASCII, with its zero-based offset
    /// in `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::FixedAscii;
    ///
    /// let text = FixedAscii::<4>::new("ab")?;
    /// let bytes = FixedAscii::<4>::new(b"ab")?;
    /// assert_eq!(text, bytes);
    /// assert_eq!(text.as_bytes(), b"ab\0\0");
    /// # Ok::<(), hdf5_pure::FormatError>(())
    /// ```
    pub fn new(value: impl AsRef<[u8]>) -> Result<Self, FormatError> {
        let width = width::<N>()?;
        let value = value.as_ref();
        if value.len() > N {
            return Err(FormatError::FixedStringTooLong {
                index: 0,
                len: value.len(),
                width,
            });
        }
        validate_ascii(value)?;
        let mut bytes = [0; N];
        bytes[..value.len()].copy_from_slice(value);
        Ok(Self { bytes })
    }

    /// Returns the stored bytes, including trailing null padding.
    pub fn as_bytes(&self) -> &[u8; N] {
        &self.bytes
    }

    /// Returns the text without trailing null bytes, preserving embedded null bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::FixedAscii;
    ///
    /// let value = FixedAscii::<5>::new(b"a\0b\0")?;
    /// assert_eq!(value.as_str(), "a\0b");
    /// # Ok::<(), hdf5_pure::FormatError>(())
    /// ```
    pub fn as_str(&self) -> &str {
        let len = self
            .bytes
            .iter()
            .rposition(|byte| *byte != 0)
            .map_or(0, |index| index + 1);
        // The private bytes are populated only after ASCII validation, and ASCII is valid UTF-8.
        str::from_utf8(&self.bytes[..len]).expect("validated ASCII")
    }
}

impl<const N: usize> sealed::Sealed for FixedAscii<N> {
    fn convert_scalar(raw: &[u8], datatype: &Datatype) -> Result<Self, Error> {
        AsciiDecoder::<N>::parse_decoder(datatype)?;
        if raw.len() != N {
            return Err(FormatError::DataSizeMismatch {
                expected: N,
                actual: raw.len(),
            }
            .into());
        }
        Ok(Self::new(raw)?)
    }
}

impl<const N: usize> H5Element for FixedAscii<N> {
    fn read_from(dataset: &Dataset) -> Result<Vec<Self>, Error> {
        dataset.read_whole_decoded::<Self, AsciiDecoder<N>>()
    }

    fn write_into(builder: &mut DatasetBuilder, data: &[Self]) {
        match datatype::<N>() {
            Ok(datatype) => {
                let mut bytes = Vec::with_capacity(mem::size_of_val(data));
                for value in data {
                    bytes.extend_from_slice(value.as_bytes());
                }
                builder.with_raw_data(datatype, bytes, data.len() as u64);
            }
            Err(error) => {
                builder.with_raw_data(type_builders::make_u8_type(), Vec::new(), data.len() as u64);
                builder.data = Some(Err(Box::new(error)));
            }
        }
    }

    fn append_into(builder: &mut AppendBuilder, data: &[Self]) {
        match datatype::<N>() {
            Ok(datatype) => builder.set_dt(datatype),
            Err(error) => {
                builder.set_error(error);
                return;
            }
        }
        for value in data {
            builder.append_raw(value.as_bytes());
        }
    }

    fn fill_bytes(self) -> Vec<u8> {
        self.bytes.to_vec()
    }

    fn convert_raw(raw: &[u8], datatype: &Datatype) -> Result<Vec<Self>, Error> {
        let decoder = AsciiDecoder::<N>::parse_decoder(datatype)?;
        let mut output = Vec::new();
        decoder.append_values(raw, &mut output)?;
        Ok(output)
    }
}

/// A decoder for null-padded ASCII strings of `N` bytes.
pub(super) struct AsciiDecoder<const N: usize>;

impl<const N: usize> ElementDecoder<FixedAscii<N>> for AsciiDecoder<N> {
    fn parse_decoder(stored: &Datatype) -> Result<Self, Error> {
        let expected = datatype::<N>()?;
        if stored != &expected {
            return Err(FormatError::TypeMismatch {
                expected: "null-padded ASCII string with matching width",
                actual: datatype::datatype_class_name(stored),
            }
            .into());
        }
        Ok(Self)
    }

    fn decoded_width(datatype: &Datatype) -> u32 {
        datatype.type_size()
    }

    fn append_values(&self, raw: &[u8], output: &mut Vec<FixedAscii<N>>) -> Result<(), Error> {
        let (elements, remainder) = raw.as_chunks::<N>();
        if !remainder.is_empty() {
            return Err(FormatError::DataSizeMismatch {
                expected: raw.len() - remainder.len(),
                actual: raw.len(),
            }
            .into());
        }
        validate_ascii(raw)?;
        output.extend(elements.iter().map(|bytes| FixedAscii { bytes: *bytes }));
        Ok(())
    }
}

/// Constructs a fixed-width string datatype with ASCII encoding and null padding.
///
/// The encoding and padding are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::ZeroFixedStringWidth`] if `N` is zero, or
/// [`FormatError::ValueTooLargeForRepresentation`] if `N` exceeds [`u32::MAX`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn datatype<const N: usize>() -> Result<Datatype, FormatError> {
    Ok(Datatype::String {
        size: width::<N>()?,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Ascii,
    })
}

/// Converts the storage width to a nonzero HDF5 string width.
///
/// # Errors
///
/// Returns [`FormatError::ZeroFixedStringWidth`] if `N` is zero, or
/// [`FormatError::ValueTooLargeForRepresentation`] if `N` exceeds [`u32::MAX`].
fn width<const N: usize>() -> Result<u32, FormatError> {
    let width: u32 = N.narrow()?;
    if width == 0 {
        return Err(FormatError::ZeroFixedStringWidth);
    }
    Ok(width)
}

/// Rejects the first byte outside ASCII.
///
/// # Errors
///
/// Returns [`FormatError::InvalidAscii`] with the byte and its zero-based offset in `bytes`.
fn validate_ascii(bytes: &[u8]) -> Result<(), FormatError> {
    if let Some((index, &byte)) = bytes.iter().enumerate().find(|(_, byte)| !byte.is_ascii()) {
        return Err(FormatError::InvalidAscii { index, byte });
    }
    Ok(())
}
