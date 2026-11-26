//! Encoding entry points that preserve the version of each description.

use alloc::vec::Vec;

use hdf5_pure_core::FormatError;

use super::prepared::PreparedDatatype;
use super::source::Description;
use crate::datatype::message::DatatypeMessage;
use crate::width::OffsetWidth;

impl DatatypeMessage {
    /// Serializes a datatype description with its retained message versions.
    ///
    /// The writer derives each variable-length descriptor size from `offset_width`,
    /// including nested descriptions.
    /// The writer computes each array size from the dimensions and the base datatype's stored
    /// element size at that width.
    /// Atomic element sizes are independent of it. An enumeration's stored size must match
    /// the base type's size at that width. Descriptors in a decoded enumeration's base must
    /// use that width. The writer terminates each compound or enumeration member name with a
    /// NUL byte. It pads each terminated name to a multiple of eight bytes with zeros in
    /// versions 1 and 2. Raw enumeration values follow the names in member order.
    ///
    /// The writer preserves compound sizes and member offsets. It writes four-byte member
    /// offsets in versions 1 and 2 and uses the minimum width that fits the compound size in
    /// versions 3 through 5. The version 1 dimension block contains the rank, all four extent
    /// slots, and zeros for the permutation and reserved fields.
    /// To check compound member ranges, the writer sizes each nested datatype at
    /// `offset_width` and multiplies by the active extents for a version 1 inline array.
    /// The nesting limit is 64 descriptions on one parent chain, including the outer description.
    ///
    /// The writer appends a NUL terminator to each nonempty opaque tag and pads its property
    /// to a multiple of eight bytes. The maximum writable logical tag length is 247 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OpaqueTagTooLong`] for a tag longer than 247 bytes, and
    /// [`FormatError::InvalidOpaqueTagByte`] for a tag containing a NUL byte.
    /// Returns [`FormatError::NestingDepthExceeded`] if the parent chain exceeds the nesting
    /// limit, and [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte size
    /// field at `offset_width`. Propagates errors from checking floating-point version/order
    /// compatibility and calculating the stored element size. Returns
    /// [`FormatError::EnumBaseSizeMismatch`] if the enumeration's value width differs from its
    /// parent's size at `offset_width`, or [`FormatError::EnumBaseOffsetWidthMismatch`] if a
    /// descriptor in its decoded base uses another offset width. Returns
    /// [`FormatError::CompoundFieldOutOfBounds`] if a member exceeds its compound element
    /// at `offset_width`, [`FormatError::CompoundFieldOverlap`] if members overlap, or
    /// [`FormatError::CompoundArraySizeOverflow`] if an inline array exceeds [`u32::MAX`] bytes.
    /// Returns [`FormatError::AllocationFailed`] if allocation for computed array or
    /// complex sizes fails, or [`FormatError::OffsetOverflow`] if the message body would
    /// exceed [`isize::MAX`] bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let message = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::FixedPoint(FixedPointType::u8()),
    /// )?;
    /// assert_eq!(
    ///     message.serialize(OffsetWidth::Eight)?,
    ///     [0x50, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0],
    /// );
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// A variable-length sequence and its base can have different versions.
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let parent = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FixedPoint(FixedPointType::i16()),
    /// )?;
    /// let sequence = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::VariableLengthSequence { base_type: Box::new(parent) },
    /// )?;
    /// let bytes = sequence.serialize(OffsetWidth::Two)?;
    /// assert_eq!(&bytes[..8], &[0x59, 0, 0, 0, 10, 0, 0, 0]);
    /// assert_eq!(bytes[8], 0x20);
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, sequence);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// An array of complex numbers retains the version of the floating-point component.
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeByteOrder, FloatingPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let component = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary32(
    ///         DatatypeByteOrder::BigEndian,
    ///     )),
    /// )?;
    /// let complex = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::Complex { component: Box::new(component) },
    /// )?;
    /// let array = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::Array { base_type: Box::new(complex), dimensions: vec![2] },
    /// )?;
    /// let bytes = array.serialize(OffsetWidth::Eight)?;
    /// assert_eq!(&bytes[..8], &[0x5a, 0, 0, 0, 16, 0, 0, 0]);
    /// assert_eq!(bytes[13], 0x5b);
    /// assert_eq!(bytes[21], 0x21);
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, array);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// Compound members retain their name bytes, offsets, and individual versions.
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeMemberName, FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let integer = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FixedPoint(FixedPointType::u8()),
    /// )?;
    /// let compound = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::Compound {
    ///         size: 4,
    ///         members: vec![DatatypeMessageMember::new(
    ///             DatatypeMemberName::new(b"x\xff")?,
    ///             3,
    ///             integer,
    ///         )],
    ///     },
    /// )?;
    /// let bytes = compound.serialize(OffsetWidth::Eight)?;
    /// assert_eq!(&bytes[..12], &[0x56, 1, 0, 0, 4, 0, 0, 0, b'x', 0xff, 0, 3]);
    /// assert_eq!(bytes[12], 0x20);
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, compound);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// A version 1 compound stores inline array dimensions before the base datatype.
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     CompoundV1Member, CompoundV1Shape, DatatypeMessage, DatatypeMessageMember,
    ///     DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let byte = DatatypeMessage::new(
    ///     DatatypeVersion::One,
    ///     DatatypeProperties::FixedPoint(FixedPointType::u8()),
    /// )?;
    /// let member = CompoundV1Member::new(
    ///     DatatypeMessageMember::new(c"arr".into(), 0, byte),
    ///     CompoundV1Shape::new(1, [2, 0, 0, 0])?,
    /// );
    /// let compound = DatatypeMessage::new(
    ///     DatatypeVersion::One,
    ///     DatatypeProperties::CompoundV1 { size: 2, members: vec![member] },
    /// )?;
    /// assert_eq!(compound.serialize(OffsetWidth::Eight)?, [
    ///     0x16, 1, 0, 0, 2, 0, 0, 0,
    ///     b'a', b'r', b'r', 0, 0, 0, 0, 0,
    ///     0, 0, 0, 0,
    ///     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ///     2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ///     0x10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0,
    /// ]);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// The writer preserves the enumeration's name bytes and writes them before the values.
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeMemberName, EnumMember, FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let integer = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FixedPoint(FixedPointType::u8()),
    /// )?;
    /// let enumeration = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::Enumeration {
    ///         base_type: Box::new(integer),
    ///         members: vec![
    ///             EnumMember::new(DatatypeMemberName::new("low")?, vec![1]),
    ///             EnumMember::new(DatatypeMemberName::new(b"high\xff")?, vec![2]),
    ///         ],
    ///     },
    /// )?;
    /// let bytes = enumeration.serialize(OffsetWidth::Eight)?;
    /// assert_eq!(&bytes[..8], &[0x58, 2, 0, 0, 1, 0, 0, 0]);
    /// assert_eq!(bytes[8], 0x20);
    /// assert_eq!(&bytes[20..], b"low\0high\xff\0\x01\x02");
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, enumeration);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, FormatError> {
        Ok(self.prepare(offset_width)?.to_bytes())
    }

    /// Prepares a bounded encoding with the retained message versions.
    ///
    /// Uses `offset_width` to size variable-length descriptors at every level when checking
    /// compound member ranges and enumeration bases. Use [`PreparedDatatype::write_into`]
    /// to write the encoding into a byte slice.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::serialize`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::FormatError;
    /// use hdf5_pure_format::__private::{DatatypeMessage, OffsetWidth};
    ///
    /// let bytes = [
    ///     0x18, 1, 0, 0, 1, 0, 0, 0,
    ///     0x10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0,
    ///     b'x', 0xff, 0, 0, 0, 0, 0, 0,
    ///     7,
    /// ];
    /// let (message, _) = DatatypeMessage::parse(&bytes)?;
    /// let prepared = message.prepare(OffsetWidth::Eight)?;
    /// let mut output = [0xff; 30];
    ///
    /// assert_eq!(prepared.encoded_size(), bytes.len());
    /// assert_eq!(prepared.write_into(&mut output)?, bytes.len());
    /// assert_eq!(output[..bytes.len()], bytes);
    /// assert_eq!(output[bytes.len()], 0xff);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn prepare(&self, offset_width: OffsetWidth) -> Result<PreparedDatatype<'_>, FormatError> {
        PreparedDatatype::prepare(Description::Preserved(self), offset_width)
    }
}
