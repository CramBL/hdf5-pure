//! Size checks and length calculation before writing a datatype.

use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private;
use hdf5_pure_core::{Datatype, FormatError};

use super::output::Output;
use super::source::{Body, Description};
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::datatype::header::DATATYPE_HEADER_SIZE;
use crate::datatype::nesting::MessageDepth;
use crate::width::OffsetWidth;

/// A datatype description checked for encoding at one file offset width.
///
/// Use [`Self::new`] for a semantic [`Datatype`] or
/// [`DatatypeMessage::prepare`](crate::datatype::message::DatatypeMessage::prepare) for a retained
/// description. Preparation determines the exact encoded size before writing any bytes. Call
/// [`Self::write_into`] for each destination slice.
#[derive(Debug)]
pub struct PreparedDatatype<'a> {
    description: Description<'a>,
    offset_width: OffsetWidth,
    computed_sizes: Vec<u32>,
    len: usize,
}

impl<'a> PreparedDatatype<'a> {
    /// Prepares a datatype message with canonical versions and the file's offset width.
    ///
    /// Counts the encoded bytes without allocating an output buffer. Uses `offset_width` to size
    /// variable-length descriptors at every level. Checks compound member ranges at that width.
    /// Checks each enumeration's value width against its base type's size at `offset_width`.
    /// Checks that any descriptors in the decoded base use that width.
    ///
    /// The writer uses at least version 3 for compounds, enumerations, and arrays, and at least
    /// version 1 for variable-length descriptions. The writer raises each container's version
    /// to the highest child version when necessary.
    /// See [`crate::datatype::serialize_datatype`] for the canonical encoding rules.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] for a zero-sized string or opaque type,
    /// [`FormatError::OpaqueTagTooLong`] for an opaque tag longer than 247 bytes, or
    /// [`FormatError::InvalidOpaqueTagByte`] if the tag contains a NUL byte.
    ///
    /// Returns [`FormatError::LengthTooLarge`] if an array size exceeds its four-byte size field,
    /// [`FormatError::ValueTooLargeForRepresentation`] if an array rank or member count exceeds
    /// its encoded field, or [`FormatError::AllocationFailed`] if allocation for computed
    /// array sizes fails.
    ///
    /// Returns [`FormatError::CompoundFieldOutOfBounds`] if a member extends past the end of
    /// its compound element or [`FormatError::CompoundFieldOverlap`] if nonempty member ranges
    /// overlap.
    ///
    /// Returns [`FormatError::EnumBaseSizeMismatch`] if an enumeration's value width differs
    /// from its base datatype's stored size at `offset_width`, or
    /// [`FormatError::EnumBaseOffsetWidthMismatch`] if a descriptor in its decoded base
    /// uses another offset width.
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the message body would exceed [`isize::MAX`]
    /// bytes, or [`FormatError::Internal`] for a datatype or reference kind without a canonical
    /// encoding.
    pub fn new(datatype: &'a Datatype, offset_width: OffsetWidth) -> Result<Self, FormatError> {
        Self::prepare(Description::Canonical(datatype), offset_width)
    }

    /// Records the byte length and the element sizes needed when writing the description.
    ///
    /// # Errors
    ///
    /// Propagates the errors from preparing the description. Applies the nesting limit
    /// when preparing a retained version.
    pub(super) fn prepare(
        description: Description<'a>,
        offset_width: OffsetWidth,
    ) -> Result<Self, FormatError> {
        let descriptor_size = __private::variable_length_descriptor_size(offset_width);
        let mut preparation = Preparation {
            len: 0,
            offset_width,
            descriptor_size,
            computed_sizes: Vec::new(),
        };
        let depth = description.version().map(|_| MessageDepth::ROOT);
        description.prepare(&mut preparation, depth)?;
        Ok(Self {
            description,
            offset_width,
            computed_sizes: preparation.computed_sizes,
            len: preparation.len,
        })
    }

    /// Compares the canonical encoding of `datatype` with this body at the prepared offset width.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`] when preparing `datatype` at this width.
    pub fn matches_datatype(&self, datatype: &Datatype) -> Result<bool, FormatError> {
        if matches!(&self.description, Description::Canonical(source) if *source == datatype) {
            return Ok(true);
        }
        let candidate = PreparedDatatype::new(datatype, self.offset_width)?;
        if self.len != candidate.len {
            return Ok(false);
        }
        let expected = self.to_bytes();
        let mut equal = true;
        Output::emit(&candidate, |position, bytes| {
            equal = equal && expected[position..position + bytes.len()] == *bytes;
        });
        Ok(equal)
    }

    /// Returns the exact body size in bytes, including all nested descriptions.
    pub fn encoded_size(&self) -> usize {
        self.len
    }

    /// Writes the entire body to the start of `output` and returns the number of bytes written.
    ///
    /// Writes without allocating and leaves bytes after the body untouched.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is shorter than [`Self::encoded_size`],
    /// leaving the entire slice untouched.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{Datatype, FixedPointType, FormatError, VariableLengthType};
    /// use hdf5_pure_format::__private::{OffsetWidth, PreparedDatatype};
    ///
    /// let sequence = Datatype::VariableLength(VariableLengthType::sequence(
    ///     Datatype::FixedPoint(FixedPointType::u16()),
    /// ));
    /// let prepared = PreparedDatatype::new(&sequence, OffsetWidth::Two)?;
    /// let mut output = [0xff; 21];
    ///
    /// assert_eq!(prepared.write_into(&mut output)?, prepared.encoded_size());
    /// assert_eq!(output, [
    ///     0x19, 0, 0, 0, 10, 0, 0, 0,
    ///     0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0,
    ///     0xff,
    /// ]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }

    /// Writes the body to a newly allocated vector of [`Self::encoded_size`] bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; self.len];
        self.write_body(&mut bytes);
        bytes
    }

    /// Writes the body into the first [`Self::encoded_size`] bytes of `bytes`.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is shorter than the encoded body.
    pub(crate) fn write_body(&self, bytes: &mut [u8]) {
        Output::emit(self, |position, field| {
            bytes[position..position + field.len()].copy_from_slice(field);
        });
    }

    /// Returns the source, destination offset width, and computed sizes in emission order.
    pub(super) fn emission_parts(&self) -> (&Description<'a>, OffsetWidth, &[u32]) {
        (&self.description, self.offset_width, &self.computed_sizes)
    }
}

impl Description<'_> {
    /// Counts the header and properties and returns the stored element size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the description has no supported encoding
    /// or [`FormatError::NestingDepthExceeded`] if a retained description exceeds the nesting
    /// limit. Propagates errors from accumulating the length and preparing the properties.
    pub(super) fn prepare(
        &self,
        preparation: &mut Preparation,
        depth: Option<MessageDepth>,
    ) -> Result<u32, FormatError> {
        preparation.add(DATATYPE_HEADER_SIZE)?;
        match self.body() {
            Body::Atomic(properties) => {
                let (size, property_size) = properties.prepare(self.version())?;
                preparation.add(property_size)?;
                Ok(size)
            }
            Body::Reference { size, .. } => Ok(size),
            Body::Container(properties) => {
                properties.prepare(preparation, depth.map(MessageDepth::child).transpose()?)
            }
            Body::Unsupported(reason) => Err(FormatError::Internal(reason.into())),
        }
    }
}

/// The body length and stored sizes computed at the selected offset width.
pub(super) struct Preparation {
    len: usize,
    offset_width: OffsetWidth,
    descriptor_size: u32,
    computed_sizes: Vec<u32>,
}

impl Preparation {
    /// Records the element size of an array or a retained complex description.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving space for the size fails.
    pub(super) fn record_size(&mut self, size: u32) -> Result<(), FormatError> {
        self.computed_sizes
            .try_reserve(1)
            .map_err(|source| FormatError::AllocationFailed {
                values: self.computed_sizes.len() + 1,
                source,
            })?;
        self.computed_sizes.push(size);
        Ok(())
    }

    pub(super) fn offset_width(&self) -> OffsetWidth {
        self.offset_width
    }

    pub(super) fn descriptor_size(&self) -> u32 {
        self.descriptor_size
    }

    /// Adds a field's length to the encoded byte count.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the total would exceed [`isize::MAX`] bytes.
    pub(super) fn add(&mut self, len: usize) -> Result<(), FormatError> {
        self.len = self
            .len
            .checked_add(len)
            .filter(|&end| end <= isize::MAX.cast_unsigned())
            .ok_or_else(|| FormatError::OffsetOverflow {
                offset: self.len.to_u64(),
                length: len.to_u64(),
            })?;
        Ok(())
    }
}
