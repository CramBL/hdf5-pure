//! Datatype-message parsing and serialization with canonical or retained versions.
//!
//! The property layouts are defined in "The Datatype Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::slice;

use hdf5_pure_core::__private;
use hdf5_pure_core::__private::DatatypeOffsets;
use hdf5_pure_core::{
    CharacterSet, CompoundMember, Datatype, DatatypeByteOrder, EnumMember, FloatingPointType,
    FormatError, ReferenceType, StringPadding, VariableLengthType,
};

use super::atomic::{AtomicProperties, AtomicReservedFlagPolicy};
use super::compound_layout::CompoundLayout;
use super::decode::{self, Decoded, Layout, Member, Projection};
use super::{DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion};
use crate::bytes::{self, ByteCursor};
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::datatype::header::{Context, Header};
use crate::datatype::{
    self, ARRAY_CLASS, COMPLEX_CLASS, COMPLEX_HOMOGENEOUS_BIT, COMPOUND_CLASS,
    DATATYPE_HEADER_SIZE, DATATYPE_VERSION_SHIFT, ENUMERATION_CLASS, REFERENCE_CLASS,
    VARIABLE_LENGTH_CLASS, VARIABLE_LENGTH_PADDING_SHIFT, VARIABLE_LENGTH_SEQUENCE,
    VARIABLE_LENGTH_STRING, VariableLengthMetadata,
};
use crate::width::OffsetWidth;

impl DatatypeMessage {
    /// Parses a datatype message and returns the number of bytes consumed.
    ///
    /// Reads each complete eight-byte header before checking its version or class. Checks version
    /// before class, except in a complex datatype's parent, where it checks class first.
    /// Reads fixed-size atomic property blocks in full before checking class flags.
    /// Acquires all array dimensions before reading the base datatype. Checks the enumeration
    /// size against its base size before reading the member names, then acquires the complete
    /// value list.
    ///
    /// Supports fixed-point, floating-point, time, string, bitfield, opaque, and variable-length
    /// descriptions at versions 1 through 5, compounds, enumerations, and arrays at versions
    /// 3 through 5, and homogeneous rectangular complex descriptions at version 5. Each nested
    /// description retains its own version, at most the containing description's version.
    /// Complex parents are floating-point descriptions. VAX order requires version 3 or later.
    /// Floating-point order and reserved bits are checked through
    /// [`super::FloatingPointByteOrder::parse`].
    ///
    /// Variable-length descriptor sizes must be 10, 12, or 16 bytes, corresponding to heap
    /// address widths of 2, 4, or 8 bytes. The parser checks each array size against the stored
    /// base size multiplied by the dimension extents, and each complex size against twice the
    /// component size.
    /// Checks compound member bounds and overlaps using the sizes stored in the member
    /// descriptions. Reserves member capacity after reading a member's nested datatype and
    /// checking its byte range. A compound must have at least one member, with unique UTF-8 names.
    /// Retains the enumeration's parent and reads at least one member. Each member has a
    /// nonempty name, unique by its bytes, and a distinct raw value of the parent's width.
    /// The nesting limit is 64 descriptions on one parent chain, including the outer description.
    ///
    /// The parser accepts an opaque property filled with tag bytes without a NUL terminator.
    /// The writer uses terminated tags, as described in [`serialize`](Self::serialize).
    /// The properties are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete. EOF byte counts
    /// are relative to the start of `data`, including for nested descriptions. Returns
    /// [`FormatError::InvalidDatatypeMessageVersion`] for a version outside `1..=5`, except in a
    /// complex datatype's parent, or [`FormatError::InvalidDatatypeClass`] for an unsupported
    /// class, including references.
    /// Returns [`FormatError::InvalidDatatypeFlags`] or [`FormatError::InvalidOpaqueReservedBits`]
    /// if reserved atomic, compound, enumeration, or array class bits are set. Returns
    /// [`FormatError::InvalidDatatypeVersion`] for compound, enumeration, or array versions 1 or 2,
    /// complex versions 1 through 4, or a complex parent version outside `1..=5`.
    ///
    /// Returns [`FormatError::InvalidVariableLengthSize`] for an unsupported descriptor size,
    /// [`FormatError::DatatypeChildVersion`] if a parent has a higher version than its container,
    /// and [`FormatError::NestingDepthExceeded`] if the parent chain exceeds the nesting limit.
    /// Returns [`FormatError::ZeroSizedDatatype`] if an array dimension is zero,
    /// [`FormatError::LengthTooLarge`] if the computed array size exceeds the size field, or
    /// [`FormatError::InvalidDatatypeSize`] if the declared array size differs from the computed size.
    ///
    /// Returns [`FormatError::InvalidCompoundMemberCount`] for a compound without members,
    /// [`FormatError::DuplicateCompoundField`] for duplicate names, and
    /// [`FormatError::CompoundFieldOutOfBounds`] for a member outside its compound element.
    /// Returns [`FormatError::CompoundFieldOverlap`] for overlapping member ranges.
    /// Returns [`FormatError::UnsupportedDatatypeMemberEncoding`] for an invalid UTF-8 compound
    /// name, with its byte offset and the original [`core::str::Utf8Error`].
    ///
    /// Returns [`FormatError::EnumBaseSizeMismatch`] if the enumeration size differs from its
    /// parent's size.
    /// Returns [`FormatError::InvalidEnumerationMemberCount`] for an enumeration without members,
    /// [`FormatError::EmptyEnumMemberName`] for an empty name,
    /// [`FormatError::DuplicateEnumMemberName`] for duplicate names, and
    /// [`FormatError::DuplicateEnumMemberValue`] for duplicate raw values.
    /// Returns [`FormatError::AllocationFailed`] if a compound or enumeration member list
    /// reservation fails.
    /// Propagates checked range and platform-conversion errors when reading values.
    ///
    /// Returns [`FormatError::InvalidComplexFlags`] for reserved complex class bits,
    /// [`FormatError::UnsupportedComplexForm`] for heterogeneous or non-rectangular descriptions,
    /// [`FormatError::InvalidComplexParent`] for a non-floating parent, and
    /// [`FormatError::InvalidComplexSize`] if the declared size differs from twice the component
    /// size. Returns [`FormatError::InvalidComplexParentFlags`] for a floating-point parent's
    /// reserved class bits, reserved byte order, or VAX order before version 3.
    /// Propagates numeric layout, floating-point byte-order, string metadata, opaque-tag,
    /// variable-length metadata, component-size, and element-size errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::FormatError;
    /// use hdf5_pure_format::__private::{DatatypeMessage, DatatypeVersion, OffsetWidth};
    ///
    /// let bytes = [0x50, 0x08, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0, 0xff];
    /// let (message, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(message.version(), DatatypeVersion::Five);
    /// assert_eq!(consumed, 12);
    /// assert_eq!(message.serialize(OffsetWidth::Eight)?, bytes[..consumed]);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    pub fn parse(data: &[u8]) -> Result<(Self, usize), FormatError> {
        let mut cursor = ByteCursor::new(data);
        let decoded = Self::parse_at(&mut cursor, MessageDepth::ROOT)?;
        Ok((decoded.datatype, cursor.position()))
    }

    fn parse_at(
        cursor: &mut ByteCursor<'_>,
        depth: MessageDepth,
    ) -> Result<Decoded<Self>, FormatError> {
        let header = Header::parse(cursor, Context::Description)?;
        let mut offsets = DatatypeOffsets::default();
        let properties = match Layout::from_header(&header, Projection::Retained)? {
            Layout::Atomic(class) => {
                let properties = class
                    .decode(
                        cursor,
                        header.fields(),
                        Context::Description,
                        AtomicReservedFlagPolicy::Reject,
                    )?
                    .into_properties();
                return Ok(Decoded {
                    datatype: Self::with_properties(header.version, properties),
                    stored_size: super::positive_size(header.class.code(), header.size)?,
                    offsets,
                });
            }
            Layout::Compound(compound) => {
                let stored_size = super::positive_size(COMPOUND_CLASS, header.size)?;
                let count = compound.member_count();
                super::compound_member_count(count)?;
                let mut members: Vec<DatatypeMessageMember> = Vec::new();
                let mut layout = CompoundLayout::new(header.size);
                for _ in 0..count {
                    let Member {
                        name,
                        offset,
                        shape: _,
                    } = compound.member(cursor)?;
                    let Decoded {
                        datatype,
                        stored_size: field_size,
                        offsets: field_offsets,
                    } = Self::parse_at(cursor, depth.child()?)?;
                    let index = members.len();
                    layout.insert(offset, field_size, index, |member_index| {
                        if member_index == index {
                            Some(name.as_ref())
                        } else {
                            members.get(member_index).map(|member| member.name.as_str())
                        }
                    })?;
                    decode::push_member(
                        &mut members,
                        count,
                        DatatypeMessageMember::new(name.into_owned(), offset, datatype)?,
                    )?;
                    offsets.include_field(field_offsets);
                }
                drop(layout);
                let mut names = BTreeSet::new();
                for member in &members {
                    member.check_compound_name_and_version(header.version, &mut names)?;
                }
                return Ok(Decoded {
                    datatype: Self::with_properties(
                        header.version,
                        DatatypeProperties::Compound {
                            size: header.size,
                            members,
                        },
                    ),
                    stored_size,
                    offsets,
                });
            }
            Layout::Reference => return Err(FormatError::InvalidDatatypeClass(REFERENCE_CLASS)),
            Layout::Enumeration(enumeration) => {
                let Decoded {
                    datatype: base_type,
                    stored_size: base_size,
                    offsets: base_offsets,
                } = Self::parse_at(cursor, depth.child()?)?;
                let members = enumeration.members(cursor, header.size, base_size)?;
                super::compatible_child(header.version, ENUMERATION_CLASS, &base_type)?;
                super::check_enumeration_members(&members, base_size.get())?;
                return Ok(Decoded {
                    datatype: Self {
                        version: header.version,
                        properties: DatatypeProperties::Enumeration {
                            base_type: Box::new(base_type),
                            members,
                        },
                        enumeration_base_offsets: base_offsets,
                    },
                    stored_size: base_size,
                    offsets: base_offsets.in_enumeration(),
                });
            }
            Layout::VariableLength(metadata, width) => {
                let base = Self::parse_at(cursor, depth.child()?)?;
                offsets = DatatypeOffsets::descriptor(width, base.offsets);
                let base_type = Box::new(base.datatype);
                match metadata {
                    VariableLengthMetadata::Sequence => {
                        DatatypeProperties::VariableLengthSequence { base_type }
                    }
                    VariableLengthMetadata::String { padding, charset } => {
                        DatatypeProperties::VariableLengthString {
                            padding,
                            charset,
                            base_type,
                        }
                    }
                }
            }
            Layout::Array(array) => {
                let dimensions = array.dimensions(cursor)?;
                let Decoded {
                    datatype: base_type,
                    stored_size: base_size,
                    offsets: base_offsets,
                } = Self::parse_at(cursor, depth.child()?)?;
                let stored_size = decode::array_size(header.size, base_size, &dimensions)?;
                super::compatible_child(header.version, ARRAY_CLASS, &base_type)?;
                return Ok(Decoded {
                    datatype: Self::with_properties(
                        header.version,
                        DatatypeProperties::Array {
                            base_type: Box::new(base_type),
                            dimensions,
                        },
                    ),
                    stored_size,
                    offsets: base_offsets,
                });
            }
            Layout::Complex(complex) => {
                let (datatype, version) = complex.component(cursor, depth)?;
                DatatypeProperties::Complex {
                    component: Box::new(Self::with_properties(
                        version,
                        DatatypeProperties::FloatingPoint(datatype.component()),
                    )),
                }
            }
        };
        Ok(Decoded {
            datatype: Self::new(header.version, properties)?,
            stored_size: super::positive_size(header.class.code(), header.size)?,
            offsets,
        })
    }

    /// Serializes a datatype description with its retained message versions.
    ///
    /// Serializes the classes accepted by [`parse`](Self::parse). The writer derives each
    /// variable-length descriptor size from `offset_width`, including nested descriptions.
    /// The writer computes each array size from the dimensions and the base datatype's stored
    /// element size at that width.
    /// Atomic element sizes are independent of it. An enumeration's stored size must match
    /// the base type's size at that width. Descriptors in a decoded enumeration's base must
    /// use that width. The writer appends unpadded, NUL-terminated names, followed by the raw
    /// values in the same member order.
    /// Compound sizes and member offsets are retained. The writer checks member bounds and
    /// overlaps using the stored member sizes at that width and writes each UTF-8 name with a
    /// NUL terminator.
    /// Member offsets occupy the minimum number of bytes needed for the compound size.
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
    /// at `offset_width`, and [`FormatError::CompoundFieldOverlap`] if members overlap at that
    /// width. Returns [`FormatError::AllocationFailed`] if allocation for computed array or
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
    /// Variable-length parents retain their own versions:
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
    /// Arrays and complex descriptions retain their component versions:
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
    /// Compound members retain their offsets and individual versions:
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
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
    ///         members: vec![DatatypeMessageMember::new("x".into(), 3, integer)?],
    ///     },
    /// )?;
    /// let bytes = compound.serialize(OffsetWidth::Eight)?;
    /// assert_eq!(&bytes[..11], &[0x56, 1, 0, 0, 4, 0, 0, 0, b'x', 0, 3]);
    /// assert_eq!(bytes[11], 0x20);
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, compound);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// All enumeration names precede the values. The writer preserves non-UTF-8 name bytes:
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
        Ok(PreparedDatatype::prepare(Description::Preserved(self), offset_width)?.to_bytes())
    }
}

/// A datatype description checked for encoding at one file offset width.
///
/// [`Self::new`] borrows a semantic [`Datatype`] and determines its exact encoded size before
/// writing any bytes. Call [`Self::write_into`] for each destination slice.
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
    /// See [`datatype::serialize_datatype`] for the canonical encoding rules.
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

    fn prepare(
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
        candidate.emit(|position, bytes| {
            equal = equal && expected[position..position + bytes.len()] == *bytes;
        });
        Ok(equal)
    }

    /// Returns the exact body size in bytes, including all nested descriptions.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
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

    /// Writes the checked datatype into a body of exactly [`Self::encoded_size`] bytes.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is shorter than the encoded body.
    pub(crate) fn write_body(&self, bytes: &mut [u8]) {
        self.emit(|position, field| {
            bytes[position..position + field.len()].copy_from_slice(field);
        });
    }

    /// Passes each field and its byte position to `write_field`, with headers after their properties.
    fn emit(&self, write_field: impl FnMut(usize, &[u8])) {
        let Self {
            description,
            offset_width,
            computed_sizes,
            len,
        } = self;
        let mut output = Output {
            write_field,
            position: 0,
            descriptor_size: __private::variable_length_descriptor_size(*offset_width),
            computed_sizes: computed_sizes.iter(),
        };
        description.write(&mut output);
        debug_assert_eq!(output.position, *len);
        debug_assert!(output.computed_sizes.as_slice().is_empty());
    }
}

impl Encode for PreparedDatatype<'_> {
    type Error = bytes::UnexpectedEof;

    fn encoded_len(&self) -> usize {
        self.len
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let available = output.len();
        let body = output
            .get_mut(..self.len)
            .ok_or_else(|| bytes::UnexpectedEof::new(self.len, available))?;
        self.write_body(body);
        Ok(self.len)
    }
}

/// The source of a datatype description and its message version policy.
#[derive(Debug)]
enum Description<'a> {
    /// A semantic datatype that uses canonical versions.
    Canonical(&'a Datatype),
    /// The floating-point component of a semantic complex datatype.
    FloatingPoint(FloatingPointType),
    /// A checked description that retains its own message version.
    Preserved(&'a DatatypeMessage),
}

impl Description<'_> {
    fn prepare(
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

    fn write(&self, output: &mut Output<'_, impl FnMut(usize, &[u8])>) -> NodeHeader {
        let position = output.advance(DATATYPE_HEADER_SIZE);
        let version = self.version();
        let header = match self.body() {
            Body::Atomic(properties) => properties.write(output, version),
            Body::Reference { size, kind } => NodeHeader {
                class: REFERENCE_CLASS,
                version: DatatypeVersion::One,
                flags: [kind, 0, 0],
                size,
            },
            Body::Container(properties) => properties.write(output, version),
            #[expect(
                clippy::unreachable,
                reason = "the immutable source has a supported encoding"
            )]
            Body::Unsupported(_) => {
                // Preparation rejects unsupported descriptions before retaining the source.
                unreachable!("a prepared datatype has a supported description")
            }
        };
        (output.write_field)(position, &header.to_bytes());
        header
    }

    fn version(&self) -> Option<DatatypeVersion> {
        match self {
            Self::Preserved(message) => Some(message.version),
            _ => None,
        }
    }

    fn body(&self) -> Body<'_> {
        match self {
            Self::Canonical(datatype) => Body::canonical(datatype),
            Self::FloatingPoint(datatype) => {
                Body::Atomic(AtomicProperties::FloatingPoint(datatype))
            }
            Self::Preserved(message) => Body::preserved(message),
        }
    }
}

enum Body<'a> {
    Atomic(AtomicProperties<'a>),
    Reference { size: u32, kind: u8 },
    Container(ContainerProperties<'a>),
    Unsupported(&'static str),
}

impl<'a> Body<'a> {
    fn canonical(datatype: &'a Datatype) -> Self {
        let properties = match datatype {
            Datatype::FixedPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FixedPoint(datatype));
            }
            Datatype::FloatingPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FloatingPoint(datatype));
            }
            Datatype::Time(datatype) => return Self::Atomic(AtomicProperties::Time(datatype)),
            Datatype::String {
                size,
                padding,
                charset,
            } => {
                return Self::Atomic(AtomicProperties::String {
                    size: *size,
                    padding,
                    charset,
                });
            }
            Datatype::BitField(datatype) => {
                return Self::Atomic(AtomicProperties::BitField(datatype));
            }
            Datatype::Opaque { size, tag } => {
                return Self::Atomic(AtomicProperties::Opaque { size: *size, tag });
            }
            Datatype::Reference { size, ref_type } => {
                let kind = match ref_type {
                    ReferenceType::Object => OBJECT_REFERENCE_KIND,
                    ReferenceType::DatasetRegion => REGION_REFERENCE_KIND,
                    _ => {
                        return Self::Unsupported(
                            "a semantic reference kind has no legacy encoding",
                        );
                    }
                };
                return Self::Reference { size: *size, kind };
            }
            Datatype::Compound { size, members } => ContainerProperties::Compound {
                size: *size,
                members: CompoundMembers::Canonical(members),
            },
            Datatype::Enumeration(datatype) => ContainerProperties::Enumeration {
                size: datatype.size().get(),
                base_type: Description::Canonical(datatype.base_type()),
                members: datatype.members(),
                offsets: __private::enumeration_base_offsets(datatype),
            },
            Datatype::VariableLength(kind) => ContainerProperties::VariableLength {
                base_type: Description::Canonical(kind.base_type()),
                string: match kind {
                    VariableLengthType::Sequence { .. } => None,
                    VariableLengthType::String(string) => {
                        Some((string.padding(), string.charset()))
                    }
                },
            },
            Datatype::Array {
                base_type,
                dimensions,
            } => ContainerProperties::Array {
                base_type: Description::Canonical(base_type),
                dimensions,
            },
            Datatype::Complex(complex) => ContainerProperties::Complex {
                component: Description::FloatingPoint(complex.component()),
                size: Some(complex.size().get()),
            },
            _ => return Self::Unsupported("a semantic datatype has no canonical encoding"),
        };
        Self::Container(properties)
    }

    fn preserved(message: &'a DatatypeMessage) -> Self {
        let properties = match &message.properties {
            DatatypeProperties::FixedPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FixedPoint(datatype));
            }
            DatatypeProperties::FloatingPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FloatingPoint(datatype));
            }
            DatatypeProperties::Time(datatype) => {
                return Self::Atomic(AtomicProperties::Time(datatype));
            }
            DatatypeProperties::String {
                size,
                padding,
                charset,
            } => {
                return Self::Atomic(AtomicProperties::String {
                    size: *size,
                    padding,
                    charset,
                });
            }
            DatatypeProperties::BitField(datatype) => {
                return Self::Atomic(AtomicProperties::BitField(datatype));
            }
            DatatypeProperties::Opaque { size, tag } => {
                return Self::Atomic(AtomicProperties::Opaque { size: *size, tag });
            }
            DatatypeProperties::Compound { size, members } => ContainerProperties::Compound {
                size: *size,
                members: CompoundMembers::Preserved(members),
            },
            DatatypeProperties::Enumeration { base_type, members } => {
                ContainerProperties::Enumeration {
                    size: super::enumeration_value_width(members),
                    base_type: Description::Preserved(base_type),
                    members,
                    offsets: message.enumeration_base_offsets,
                }
            }
            DatatypeProperties::VariableLengthSequence { base_type } => {
                ContainerProperties::VariableLength {
                    base_type: Description::Preserved(base_type),
                    string: None,
                }
            }
            DatatypeProperties::VariableLengthString {
                padding,
                charset,
                base_type,
            } => ContainerProperties::VariableLength {
                base_type: Description::Preserved(base_type),
                string: Some((padding, charset)),
            },
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => ContainerProperties::Array {
                base_type: Description::Preserved(base_type),
                dimensions,
            },
            DatatypeProperties::Complex { component } => ContainerProperties::Complex {
                component: Description::Preserved(component),
                size: None,
            },
        };
        Self::Container(properties)
    }
}

enum ContainerProperties<'a> {
    Compound {
        size: u32,
        members: CompoundMembers<'a>,
    },
    Enumeration {
        size: u32,
        base_type: Description<'a>,
        members: &'a [EnumMember],
        offsets: DatatypeOffsets,
    },
    VariableLength {
        base_type: Description<'a>,
        string: Option<(&'a StringPadding, &'a CharacterSet)>,
    },
    Array {
        base_type: Description<'a>,
        dimensions: &'a [u32],
    },
    Complex {
        component: Description<'a>,
        size: Option<u32>,
    },
}

impl ContainerProperties<'_> {
    fn prepare(
        &self,
        preparation: &mut Preparation,
        child_depth: Option<MessageDepth>,
    ) -> Result<u32, FormatError> {
        match self {
            Self::Compound { size, members } => {
                check_member_count(members.len(), child_depth.is_some())?;
                members.prepare(preparation, *size, child_depth)?;
                Ok(*size)
            }
            Self::Enumeration {
                size,
                base_type,
                members,
                offsets,
            } => {
                check_member_count(members.len(), child_depth.is_some())?;
                let parent_size = base_type.prepare(preparation, child_depth)?;
                super::check_enum_base_layout(
                    *size,
                    parent_size,
                    *offsets,
                    preparation.offset_width,
                )?;
                for member in *members {
                    preparation.add(1)?;
                    preparation.add(member.name.as_bytes().len())?;
                    preparation.add(member.value.len())?;
                }
                Ok(parent_size)
            }
            Self::VariableLength { base_type, .. } => {
                base_type.prepare(preparation, child_depth)?;
                Ok(preparation.descriptor_size)
            }
            Self::Array {
                base_type,
                dimensions,
            } => {
                let rank: u8 = dimensions.len().narrow_or_else(|| {
                    if child_depth.is_some() {
                        FormatError::Internal("a checked array rank exceeds u8".into())
                    } else {
                        FormatError::ValueTooLargeForRepresentation {
                            value: dimensions.len().to_u64(),
                            target: "u8",
                        }
                    }
                })?;
                preparation.add(1 + usize::from(rank) * size_of::<u32>())?;
                let parent_size = base_type.prepare(preparation, child_depth)?;
                let size = __private::array_type_size(parent_size, dimensions)?;
                preparation.record_size(size)?;
                Ok(size)
            }
            Self::Complex { component, size } => {
                let component_size = component.prepare(preparation, child_depth)?;
                match size {
                    Some(size) => Ok(*size),
                    None => {
                        let size = component_size
                            .checked_mul(2)
                            .ok_or(FormatError::ComplexSizeOverflow { component_size })?;
                        preparation.record_size(size)?;
                        Ok(size)
                    }
                }
            }
        }
    }

    fn write(
        &self,
        output: &mut Output<'_, impl FnMut(usize, &[u8])>,
        preserved: Option<DatatypeVersion>,
    ) -> NodeHeader {
        let (class, minimum, child_version, flags, size) = match self {
            Self::Compound { size, members } => {
                let width = datatype::offset_bytes_for_size(*size);
                let mut version = DatatypeVersion::Three;
                for index in 0..members.len() {
                    let (name, offset, child) = members.member(index);
                    output.write(name.as_bytes());
                    output.write(&[0]);
                    output.write(&offset.to_le_bytes()[..width]);
                    version = version.max(child.write(output).version);
                }
                (
                    COMPOUND_CLASS,
                    DatatypeVersion::Three,
                    version,
                    member_flags(members.len()),
                    *size,
                )
            }
            Self::Enumeration {
                size,
                base_type,
                members,
                offsets: _,
            } => {
                let child = base_type.write(output);
                debug_assert_eq!(*size, child.size);
                for member in *members {
                    output.write(member.name.as_bytes());
                    output.write(&[0]);
                }
                for member in *members {
                    output.write(&member.value);
                }
                (
                    ENUMERATION_CLASS,
                    DatatypeVersion::Three,
                    child.version,
                    member_flags(members.len()),
                    child.size,
                )
            }
            Self::VariableLength { base_type, string } => {
                let child = base_type.write(output);
                let flags = match string {
                    Some((padding, charset)) => [
                        VARIABLE_LENGTH_STRING
                            | (padding_code(padding) << VARIABLE_LENGTH_PADDING_SHIFT),
                        charset_code(charset),
                        0,
                    ],
                    None => [VARIABLE_LENGTH_SEQUENCE, 0, 0],
                };
                (
                    VARIABLE_LENGTH_CLASS,
                    DatatypeVersion::One,
                    child.version,
                    flags,
                    output.descriptor_size,
                )
            }
            Self::Array {
                base_type,
                dimensions,
            } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the prepared dimension count fits the one-byte rank field"
                )]
                let rank = dimensions.len() as u8;
                output.write(&[rank]);
                for &dimension in *dimensions {
                    output.write(&dimension.to_le_bytes());
                }
                let child = base_type.write(output);
                (
                    ARRAY_CLASS,
                    DatatypeVersion::Three,
                    child.version,
                    [0; 3],
                    output.next_size(),
                )
            }
            Self::Complex { component, size } => {
                let child = component.write(output);
                (
                    COMPLEX_CLASS,
                    DatatypeVersion::Five,
                    child.version,
                    [COMPLEX_HOMOGENEOUS_BIT, 0, 0],
                    size.unwrap_or_else(|| output.next_size()),
                )
            }
        };
        // `H5O__dtype_encode_helper` requires the enclosing version to be at least each
        // child's version (H5Odtype.c, HDF5 2.2.0).
        NodeHeader {
            class,
            version: preserved.unwrap_or(minimum.max(child_version)),
            flags,
            size,
        }
    }
}

fn check_member_count(count: usize, preserved: bool) -> Result<(), FormatError> {
    super::member_count(count).map(|_| ()).map_err(|error| {
        if preserved {
            FormatError::Internal("a checked datatype member count exceeds u16".into())
        } else {
            error
        }
    })
}

enum CompoundMembers<'a> {
    Canonical(&'a [CompoundMember]),
    Preserved(&'a [DatatypeMessageMember]),
}

impl CompoundMembers<'_> {
    fn prepare(
        &self,
        preparation: &mut Preparation,
        size: u32,
        child_depth: Option<MessageDepth>,
    ) -> Result<(), FormatError> {
        let width = datatype::offset_bytes_for_size(size);
        let mut layout = CompoundLayout::new(size);
        for index in 0..self.len() {
            let (name, offset, child) = self.member(index);
            preparation.add_member_prefix(name, width)?;
            let field_size = match child.prepare(preparation, child_depth) {
                Ok(size) => size,
                Err(error) => {
                    // Check the canonical member's bounds before returning its preparation error.
                    if let Description::Canonical(datatype) = child {
                        let size = datatype::stored_type_size(datatype, preparation.offset_width)?;
                        layout.member_end(offset, size, || Ok(name))?;
                    }
                    return Err(error);
                }
            };
            if let Some(field_size) = NonZeroU32::new(field_size) {
                layout.insert(offset, field_size, index, |index| self.member_name(index))?;
            } else {
                layout.member_end(offset, field_size, || Ok(name))?;
            }
        }
        Ok(())
    }

    fn len(&self) -> usize {
        match self {
            Self::Canonical(members) => members.len(),
            Self::Preserved(members) => members.len(),
        }
    }

    fn member(&self, index: usize) -> (&str, u64, Description<'_>) {
        match self {
            Self::Canonical(members) => {
                let (name, offset, datatype) = __private::compound_member_parts(&members[index]);
                (name, offset, Description::Canonical(datatype))
            }
            Self::Preserved(members) => {
                let DatatypeMessageMember {
                    name,
                    byte_offset,
                    datatype,
                } = &members[index];
                (name, *byte_offset, Description::Preserved(datatype))
            }
        }
    }

    fn member_name(&self, index: usize) -> Option<&str> {
        match self {
            Self::Canonical(members) => members.get(index).map(|member| member.name.as_str()),
            Self::Preserved(members) => members.get(index).map(|member| member.name.as_str()),
        }
    }
}

/// The fields of the common header, including the selected version and stored element size.
pub(super) struct NodeHeader {
    pub(super) class: u8,
    pub(super) version: DatatypeVersion,
    /// The three class bit-field bytes.
    pub(super) flags: [u8; 3],
    /// The stored element size in bytes.
    pub(super) size: u32,
}

impl NodeHeader {
    fn to_bytes(&self) -> [u8; DATATYPE_HEADER_SIZE] {
        let &Self {
            class,
            version,
            flags: [first, second, third],
            size,
        } = self;
        let [low, middle_low, middle_high, high] = size.to_le_bytes();
        [
            class | (version.get() << DATATYPE_VERSION_SHIFT),
            first,
            second,
            third,
            low,
            middle_low,
            middle_high,
            high,
        ]
    }
}

/// The body length and stored sizes computed at the selected offset width.
struct Preparation {
    len: usize,
    offset_width: OffsetWidth,
    descriptor_size: u32,
    computed_sizes: Vec<u32>,
}

impl Preparation {
    /// Records the element size of an array or a complex description with a retained version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving space for the size fails.
    fn record_size(&mut self, size: u32) -> Result<(), FormatError> {
        self.computed_sizes
            .try_reserve(1)
            .map_err(|source| FormatError::AllocationFailed {
                values: self.computed_sizes.len() + 1,
                source,
            })?;
        self.computed_sizes.push(size);
        Ok(())
    }

    fn add_member_prefix(&mut self, name: &str, offset_bytes: usize) -> Result<(), FormatError> {
        self.add(name.len())?;
        self.add(1 + offset_bytes)
    }

    fn add(&mut self, len: usize) -> Result<(), FormatError> {
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

/// The field writer, current byte position, and sizes selected during preparation.
pub(super) struct Output<'a, WriteField> {
    write_field: WriteField,
    position: usize,
    descriptor_size: u32,
    computed_sizes: slice::Iter<'a, u32>,
}

impl<WriteField: FnMut(usize, &[u8])> Output<'_, WriteField> {
    /// Passes `bytes` to the field writer at the current position and advances by their length.
    pub(super) fn write(&mut self, bytes: &[u8]) {
        let position = self.advance(bytes.len());
        (self.write_field)(position, bytes);
    }

    fn advance(&mut self, len: usize) -> usize {
        let position = self.position;
        self.position += len;
        position
    }

    /// Returns the next prepared element size in child-before-parent order.
    ///
    /// # Panics
    ///
    /// Panics if the writer has consumed every prepared size.
    #[expect(
        clippy::expect_used,
        reason = "preparation records computed sizes in the same child-before-parent order"
    )]
    fn next_size(&mut self) -> u32 {
        *self.computed_sizes.next().expect("a prepared size exists")
    }
}

/// The number of enclosing datatype descriptions, from zero through 63.
#[derive(Clone, Copy)]
pub(in crate::datatype) struct MessageDepth(u8);

impl MessageDepth {
    /// Returns the depth of a child description within the checked nesting limit.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if the child would exceed 64 descriptions
    /// on one parent chain, including the outer description.
    pub(in crate::datatype) fn child(self) -> Result<Self, FormatError> {
        self.0
            .checked_add(1)
            .filter(|&depth| depth < MAX_MESSAGE_DEPTH)
            .map(Self)
            .ok_or(FormatError::NestingDepthExceeded)
    }

    /// The depth of the outer description.
    pub(in crate::datatype) const ROOT: Self = Self(0);
}

/// Encodes a checked member count in the low two class flag bytes.
///
/// The count field is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if `count` exceeds [`u16::MAX`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn member_flags(count: usize) -> [u8; 3] {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the prepared member count fits its two-byte field"
    )]
    let [low, high] = (count as u16).to_le_bytes();
    [low, high, 0]
}

/// Encodes the padding code for fixed- and variable-length strings.
///
/// Returns 0, 1, or 2 for [`StringPadding::NullTerminate`], [`StringPadding::NullPad`],
/// or [`StringPadding::SpacePad`], respectively. The fields are defined in "The Datatype
/// Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) fn padding_code(padding: &StringPadding) -> u8 {
    match padding {
        StringPadding::NullTerminate => NULL_TERMINATE,
        StringPadding::NullPad => NULL_PAD,
        StringPadding::SpacePad => SPACE_PAD,
    }
}

/// Encodes the character-set code for fixed- and variable-length strings.
///
/// Returns 0 for [`CharacterSet::Ascii`] and 1 for [`CharacterSet::Utf8`]. The fields
/// are defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) fn charset_code(charset: &CharacterSet) -> u8 {
    match charset {
        CharacterSet::Ascii => ASCII_CHARSET,
        CharacterSet::Utf8 => UTF8_CHARSET,
    }
}

/// Encodes bit 0 for fixed-point, time, and bitfield byte order.
///
/// Callers pass [`DatatypeByteOrder::LittleEndian`] or [`DatatypeByteOrder::BigEndian`]
/// from checked metadata. The fields are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) fn binary_order_bit(order: DatatypeByteOrder) -> u8 {
    if order == DatatypeByteOrder::BigEndian {
        BINARY_ORDER_BIT
    } else {
        0
    }
}

/// The mask for reserved fixed-point class bits 4 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const FIXED_RESERVED: u32 = 0xff_fff0;
/// The mask for reserved time class bits 1 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const TIME_RESERVED: u32 = 0xff_fffe;
/// The mask for reserved string class bits 8 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const STRING_RESERVED: u32 = 0xff_ff00;
/// The mask for reserved bitfield class bits 3 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const BITFIELD_RESERVED: u32 = 0xff_fff8;
/// Bit 0, the byte-order bit for fixed-point, time, and bitfield descriptions.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const BINARY_ORDER_BIT: u8 = 0x01;
/// Bit 3, the signed-integer flag for fixed-point descriptions.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const SIGNED_BIT: u8 = 0x08;
/// The mask for string padding bits 0 through 3.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const STRING_PADDING_MASK: u8 = 0x0f;
/// The position of the string character-set field at bit 4.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const STRING_CHARSET_SHIFT: u8 = 4;

/// The string padding code for NUL termination.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const NULL_TERMINATE: u8 = 0;
/// The string padding code for NUL padding.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const NULL_PAD: u8 = 1;
/// The string padding code for space padding.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const SPACE_PAD: u8 = 2;
/// The ASCII string character-set code.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const ASCII_CHARSET: u8 = 0;
/// The UTF-8 string character-set code.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const UTF8_CHARSET: u8 = 1;

/// The maximum number of descriptions on a parsing or serialization path.
const MAX_MESSAGE_DEPTH: u8 = 64;

/// The object-reference kind for datatype message versions 1 through 3.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const OBJECT_REFERENCE_KIND: u8 = 0;
/// The dataset-region reference kind for datatype message versions 1 through 3.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const REGION_REFERENCE_KIND: u8 = 1;
