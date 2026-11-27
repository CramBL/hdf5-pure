use core::ops::Range;

use crate::codec::DecodePrefix;
use crate::error::FormatError;
use crate::width::LengthWidth;

use super::shape::Dataspace;
use super::stored::StoredDataspace;

/// A checked dataspace and the object header record body that contains it.
#[derive(Debug)]
pub struct DataspaceRecord<'a> {
    body: &'a [u8],
    consumed: usize,
    dimension_fields: Range<usize>,
    shape: Dataspace,
}

impl<'a> DataspaceRecord<'a> {
    /// Reads the dataspace at the start of an object header record body.
    ///
    /// Retains the bytes after the last declared field, accessible through [`Self::tail`].
    /// Message versions 1 and 2 are defined in "The Dataspace Message" of the
    /// [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns the header, dimension list, and shape errors of [`Dataspace::parse`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private as format, FormatError, LengthWidth, MaxExtent};
    ///
    /// let dataspace = format::Dataspace::simple(vec![3], Some(vec![MaxExtent::Fixed(7)]))?;
    /// let mut body = dataspace.prepare(LengthWidth::Two)?.to_bytes();
    /// body.push(0xaa);
    /// let record = format::DataspaceRecord::parse(&body, LengthWidth::Two)?;
    /// assert_eq!(record.dimension_fields(), 4..6);
    /// assert_eq!(record.tail(), &[0xaa]);
    /// assert_eq!(record.into_shape(), dataspace);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
    pub fn parse(body: &'a [u8], width: LengthWidth) -> Result<Self, FormatError> {
        let (stored, consumed) = StoredDataspace::decode_prefix(body, width)?.into_parts();
        Ok(Self {
            body,
            consumed,
            dimension_fields: stored.dimension_range(),
            shape: stored.into_dataspace()?,
        })
    }

    pub fn into_shape(self) -> Dataspace {
        self.shape
    }

    /// Borrows the checked shape.
    pub fn shape(&self) -> &Dataspace {
        &self.shape
    }

    /// Returns the byte range of the current dimensions relative to the start of the body.
    ///
    /// The range is empty for a scalar or null dataspace.
    pub fn dimension_fields(&self) -> Range<usize> {
        self.dimension_fields.clone()
    }

    /// Returns the record bytes following the last declared dataspace field.
    pub fn tail(&self) -> &'a [u8] {
        &self.body[self.consumed..]
    }
}
