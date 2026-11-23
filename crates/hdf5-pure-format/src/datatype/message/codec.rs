//! Datatype-message parsing and serialization with canonical or retained versions.
//!
//! The property layouts are defined in "The Datatype Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use core::str;

use hdf5_pure_core::__private;
use hdf5_pure_core::{
    CharacterSet, ComplexType, CompoundMember, Datatype, DatatypeByteOrder, EnumMember,
    FloatingPointType, FormatError, ReferenceType, StringPadding, VariableLengthType,
};

use super::atomic::{AtomicProperties, AtomicReservedFlagPolicy};
use super::compound_layout::CompoundLayout;
use super::{DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion};
use crate::bytes::{self, ByteCursor};
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::datatype::header::{Context, DatatypeClass, Header};
use crate::datatype::{
    self, ARRAY_CLASS, COMPLEX_CLASS, COMPLEX_FORM_MASK, COMPLEX_FORM_SHIFT,
    COMPLEX_HOMOGENEOUS_BIT, COMPLEX_RECTANGULAR_FORM, COMPLEX_RESERVED_MASK, COMPOUND_CLASS,
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
    /// address widths of 2, 4, or 8 bytes. The parser checks array element sizes against the
    /// recorded parent size and dimensions, and complex sizes against twice the component size.
    /// Compound member bounds and overlaps are checked using their declared source sizes.
    /// A compound must have at least one member, with unique UTF-8 names.
    /// Enumerations have a fixed-point parent and at least one member, with nonempty, unique
    /// UTF-8 names and distinct raw values of the parent's width. The enumeration and parent
    /// sizes must match. The nesting limit is 64 descriptions on one parent chain, including
    /// the outer description.
    ///
    /// The parser accepts an opaque property filled with tag bytes without a NUL terminator.
    /// The writer uses terminated tags, as described in [`serialize`](Self::serialize).
    /// The properties are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete. Returns
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
    /// Returns [`FormatError::InvalidDatatypeSize`] if an array size differs from its properties,
    /// and [`FormatError::LengthTooLarge`] if the array dimension product exceeds the size field.
    ///
    /// Returns [`FormatError::InvalidCompoundMemberCount`] for a compound without members,
    /// [`FormatError::DuplicateCompoundField`] for duplicate names, and
    /// [`FormatError::CompoundFieldOutOfBounds`] for a member outside its compound element.
    /// Returns [`FormatError::CompoundFieldOverlap`] for overlapping member ranges.
    /// Returns [`FormatError::UnsupportedDatatypeMemberEncoding`] if a member name is not
    /// UTF-8, retaining the name's byte offset and the original [`str::Utf8Error`].
    ///
    /// Returns [`FormatError::EnumBaseNotInteger`] for a non-fixed-point enumeration parent,
    /// and [`FormatError::EnumBaseSizeMismatch`] if its size differs from the enumeration size.
    /// Returns [`FormatError::InvalidEnumerationMemberCount`] for an enumeration without members,
    /// [`FormatError::EmptyEnumMemberName`] for an empty name,
    /// [`FormatError::DuplicateEnumMemberName`] for duplicate names, and
    /// [`FormatError::DuplicateEnumMemberValue`] for duplicate raw values.
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
        let Parsed {
            message,
            end,
            size: _,
        } = Self::parse_at(data, Context::Description, 0, 0)?;
        Ok((message, end))
    }

    /// Parses a description and its declared element size at `position`.
    ///
    /// The returned end position is relative to `data`. Complex-parent class checks and
    /// diagnostics are applied according to `context`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`parse`](Self::parse). The nesting limit includes `depth`
    /// enclosing descriptions.
    fn parse_at(
        data: &[u8],
        context: Context,
        position: usize,
        depth: u8,
    ) -> Result<Parsed, FormatError> {
        if depth >= MAX_MESSAGE_DEPTH {
            return Err(FormatError::NestingDepthExceeded);
        }
        let mut cursor = ByteCursor::with_position(data, position)?;
        let header = Header::parse(&mut cursor, context)?;
        let (properties, consumed) = match header.class {
            DatatypeClass::Atomic(class) => {
                let properties = class
                    .decode(
                        &mut cursor,
                        header.fields(),
                        context,
                        AtomicReservedFlagPolicy::Reject,
                    )?
                    .into_properties();
                return Ok(Parsed {
                    message: Self {
                        version: header.version,
                        properties,
                    },
                    end: cursor.position(),
                    size: header.size,
                });
            }
            DatatypeClass::Compound => {
                let count = header.modern_member_count()?;
                super::positive_size(COMPOUND_CLASS, header.size)?;
                let offset_bytes = datatype::offset_bytes_for_size(header.size);
                let mut members = Vec::new();
                let mut layout = CompoundLayout::new(header.size);
                for _ in 0..count {
                    let (name, end) = member_name(data, COMPOUND_CLASS, cursor.position())?;
                    cursor = ByteCursor::with_position(data, end)?;
                    let offset = cursor.uint(offset_bytes)?;
                    let Parsed {
                        message: datatype,
                        end,
                        size: field_size,
                    } = Self::parse_at(data, Context::Description, cursor.position(), depth + 1)?;
                    let index = members.len();
                    members.push(DatatypeMessageMember::new(name, offset, datatype)?);
                    layout.insert(&members, index, field_size)?;
                    cursor = ByteCursor::with_position(data, end)?;
                }
                (
                    DatatypeProperties::Compound {
                        size: header.size,
                        members,
                    },
                    cursor.position(),
                )
            }
            DatatypeClass::Reference => {
                return Err(FormatError::InvalidDatatypeClass(REFERENCE_CLASS));
            }
            DatatypeClass::Enumeration => {
                let count = header.modern_member_count()?;
                let Parsed {
                    message: base_type,
                    end,
                    size: base_size,
                } = Self::parse_at(data, Context::Description, cursor.position(), depth + 1)?;
                if !matches!(base_type.properties, DatatypeProperties::FixedPoint(_)) {
                    return Err(FormatError::EnumBaseNotInteger);
                }
                if header.size != base_size {
                    return Err(FormatError::EnumBaseSizeMismatch {
                        size: header.size,
                        base_size,
                    });
                }
                cursor = ByteCursor::with_position(data, end)?;
                let mut names = Vec::new();
                for _ in 0..count {
                    let (name, end) = member_name(data, ENUMERATION_CLASS, cursor.position())?;
                    names.push(name);
                    cursor = ByteCursor::with_position(data, end)?;
                }
                let value_size = base_size.to_usize()?;
                let members = names
                    .into_iter()
                    .map(|name| {
                        Ok(__private::enum_member(
                            name,
                            cursor.slice(value_size)?.to_vec(),
                        ))
                    })
                    .collect::<Result<Vec<_>, FormatError>>()?;
                (
                    DatatypeProperties::Enumeration {
                        base_type: Box::new(base_type),
                        members,
                    },
                    cursor.position(),
                )
            }
            DatatypeClass::VariableLength => {
                let metadata = datatype::parse_variable_length_metadata(header.flags)?;
                descriptor_width(header.size)?;
                let Parsed {
                    message: base_type,
                    end: consumed,
                    size: _,
                } = Self::parse_at(data, Context::Description, cursor.position(), depth + 1)?;
                let base_type = Box::new(base_type);
                let properties = match metadata {
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
                };
                (properties, consumed)
            }
            DatatypeClass::Array => {
                header
                    .version
                    .for_class(ARRAY_CLASS, DatatypeVersion::Three)?;
                header.reserved(ARRAY_RESERVED)?;
                let rank = cursor.u8()?;
                let dimensions = (0..rank)
                    .map(|_| cursor.u32())
                    .collect::<Result<Vec<_>, _>>()?;
                let Parsed {
                    message: base_type,
                    end,
                    size: base_size,
                } = Self::parse_at(data, Context::Description, cursor.position(), depth + 1)?;
                let expected = datatype::array_type_size(base_size, &dimensions)?;
                if header.size != expected {
                    return Err(FormatError::InvalidDatatypeSize {
                        class: ARRAY_CLASS,
                        size: header.size,
                        expected,
                    });
                }
                (
                    DatatypeProperties::Array {
                        base_type: Box::new(base_type),
                        dimensions,
                    },
                    end,
                )
            }
            DatatypeClass::Complex => {
                header
                    .version
                    .for_class(COMPLEX_CLASS, DatatypeVersion::Five)?;
                let flags = header.flag_bits();
                if flags & COMPLEX_RESERVED_MASK != 0 {
                    return Err(FormatError::InvalidComplexFlags(flags));
                }
                let homogeneous = header.flags[0] & COMPLEX_HOMOGENEOUS_BIT != 0;
                let form = (header.flags[0] & COMPLEX_FORM_MASK) >> COMPLEX_FORM_SHIFT;
                if !homogeneous || form != COMPLEX_RECTANGULAR_FORM {
                    return Err(FormatError::UnsupportedComplexForm { homogeneous, form });
                }
                let Parsed {
                    message: parent,
                    end,
                    size: _,
                } = Self::parse_at(data, Context::ComplexParent, cursor.position(), depth + 1)?;
                let DatatypeProperties::FloatingPoint(component) = parent.properties() else {
                    return Err(FormatError::Internal(
                        "a checked complex parent is not floating point".into(),
                    ));
                };
                let complex = ComplexType::new(*component)?;
                if header.size != complex.size().get() {
                    return Err(FormatError::InvalidComplexSize {
                        size: header.size,
                        component_size: component.size().get(),
                    });
                }
                (
                    DatatypeProperties::Complex {
                        component: Box::new(parent),
                    },
                    end,
                )
            }
        };
        Ok(Parsed {
            message: Self::new(header.version, properties)?,
            end: consumed,
            size: header.size,
        })
    }

    /// Serializes a datatype description with its retained message versions.
    ///
    /// Serializes the classes accepted by [`parse`](Self::parse). The writer derives each
    /// variable-length descriptor size from `offset_width`, including nested descriptions.
    /// Array sizes are derived from the encoded parent size and dimensions at that width.
    /// Atomic element sizes are independent of it. Enumeration sizes are derived from the
    /// encoded integer parent. The writer appends unpadded, NUL-terminated names, followed
    /// by the raw values in the same member order. Values retain their stored bytes.
    /// Compound sizes and member offsets are retained. The writer checks member bounds and
    /// overlaps using their encoded sizes at that width and writes each UTF-8 name with a NUL
    /// terminator.
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
    /// [`FormatError::CompoundFieldOutOfBounds`] if a member exceeds its compound element
    /// at `offset_width`, and [`FormatError::CompoundFieldOverlap`] if members overlap at that
    /// width. Returns [`FormatError::OffsetOverflow`] if the message body would exceed
    /// [`isize::MAX`] bytes.
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
    /// Enumeration parents retain their versions, and all names precede the values:
    ///
    /// ```
    /// use hdf5_pure_core::{__private, FixedPointType, FormatError};
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
    ///             __private::enum_member("low".into(), vec![1]),
    ///             __private::enum_member("high".into(), vec![2]),
    ///         ],
    ///     },
    /// )?;
    /// let bytes = enumeration.serialize(OffsetWidth::Eight)?;
    /// assert_eq!(&bytes[..8], &[0x58, 2, 0, 0, 1, 0, 0, 0]);
    /// assert_eq!(bytes[8], 0x20);
    /// assert_eq!(&bytes[20..], b"low\0high\0\x01\x02");
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
    len: usize,
}

impl<'a> PreparedDatatype<'a> {
    /// Prepares a datatype message with canonical versions and the file's offset width.
    ///
    /// Counts the encoded bytes without allocating an output buffer. Uses `offset_width` to size
    /// variable-length descriptors at every level. The writer uses at least version 3 for
    /// compounds, enumerations, and arrays, and at least version 1 for variable-length
    /// descriptions. The writer raises each container's version to the highest child version
    /// when necessary.
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
    /// its encoded field, or [`FormatError::CompoundFieldOutOfBounds`] if a member extends past
    /// the declared record size.
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
        let mut preparation = Preparation {
            len: 0,
            offset_width,
        };
        let depth = description.version().map(|_| MessageDepth::ROOT);
        description.prepare(&mut preparation, depth)?;
        Ok(Self {
            description,
            offset_width,
            len: preparation.len,
        })
    }

    /// Returns the exact body size in bytes, including all nested descriptions.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the entire body to the start of `output` and returns the number of bytes written.
    ///
    /// Leaves bytes after the body untouched.
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
        let Self {
            description,
            offset_width,
            len: _,
        } = self;
        description.write(&mut Output {
            bytes,
            offset_width: *offset_width,
        });
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

    fn write(&self, output: &mut Output<'_>) -> NodeHeader {
        let destination = output.take(DATATYPE_HEADER_SIZE);
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
        destination.copy_from_slice(&header.to_bytes());
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
            Self::Preserved(message) => Body::preserved(&message.properties),
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
            Datatype::Enumeration {
                size,
                base_type,
                members,
            } => ContainerProperties::Enumeration {
                declared_size: Some(*size),
                base_type: Description::Canonical(base_type),
                members,
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

    fn preserved(properties: &'a DatatypeProperties) -> Self {
        let properties = match properties {
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
                    declared_size: None,
                    base_type: Description::Preserved(base_type),
                    members,
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
        declared_size: Option<u32>,
        base_type: Description<'a>,
        members: &'a [EnumMember],
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
                declared_size,
                base_type,
                members,
            } => {
                check_member_count(members.len(), child_depth.is_some())?;
                let parent_size = base_type.prepare(preparation, child_depth)?;
                for member in *members {
                    let (name, value) = __private::enum_member_parts(member);
                    preparation.add(1)?;
                    preparation.add(name.len())?;
                    preparation.add(value.len())?;
                }
                Ok(declared_size.unwrap_or(parent_size))
            }
            Self::VariableLength { base_type, .. } => {
                base_type.prepare(preparation, child_depth)?;
                Ok(variable_length_size(preparation.offset_width))
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
                datatype::array_type_size(parent_size, dimensions)
            }
            Self::Complex { component, size } => {
                let component_size = component.prepare(preparation, child_depth)?;
                match size {
                    Some(size) => Ok(*size),
                    None => component_size
                        .checked_mul(2)
                        .ok_or(FormatError::ComplexSizeOverflow { component_size }),
                }
            }
        }
    }

    fn write(&self, output: &mut Output<'_>, preserved: Option<DatatypeVersion>) -> NodeHeader {
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
                base_type,
                members,
                declared_size,
            } => {
                let child = base_type.write(output);
                for member in *members {
                    let (name, _) = __private::enum_member_parts(member);
                    output.write(name.as_bytes());
                    output.write(&[0]);
                }
                for member in *members {
                    let (_, value) = __private::enum_member_parts(member);
                    output.write(value);
                }
                (
                    ENUMERATION_CLASS,
                    DatatypeVersion::Three,
                    child.version,
                    member_flags(members.len()),
                    declared_size.unwrap_or(child.size),
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
                    variable_length_size(output.offset_width),
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
                let mut elements = 1u32;
                for &dimension in *dimensions {
                    output.write(&dimension.to_le_bytes());
                    elements = elements.wrapping_mul(dimension);
                }
                let child = base_type.write(output);
                // Preparation establishes the product fits. A zero factor still gives an exact
                // zero even when an earlier dimension product wraps.
                let size = child.size.wrapping_mul(elements);
                (
                    ARRAY_CLASS,
                    DatatypeVersion::Three,
                    child.version,
                    [0; 3],
                    size,
                )
            }
            Self::Complex { component, size } => {
                let child = component.write(output);
                (
                    COMPLEX_CLASS,
                    DatatypeVersion::Five,
                    child.version,
                    [COMPLEX_HOMOGENEOUS_BIT, 0, 0],
                    size.unwrap_or_else(|| child.size * 2),
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

fn variable_length_size(offset_width: OffsetWidth) -> u32 {
    datatype::SEQUENCE_LENGTH_SIZE + u32::from(offset_width.get()) + datatype::HEAP_INDEX_SIZE
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
        match self {
            Self::Canonical(members) => {
                for member in *members {
                    Self::prepare_canonical_member(preparation, size, width, member)?;
                }
            }
            Self::Preserved(members) => {
                let mut layout = CompoundLayout::new(size);
                for (index, member) in members.iter().enumerate() {
                    let DatatypeMessageMember {
                        name,
                        byte_offset: _,
                        datatype,
                    } = member;
                    preparation.add_member_prefix(name, width)?;
                    let child_size =
                        Description::Preserved(datatype).prepare(preparation, child_depth)?;
                    layout.insert(members, index, child_size)?;
                }
            }
        }
        Ok(())
    }

    fn prepare_canonical_member(
        preparation: &mut Preparation,
        compound_size: u32,
        offset_bytes: usize,
        member: &CompoundMember,
    ) -> Result<(), FormatError> {
        let (name, offset, datatype) = __private::compound_member_parts(member);
        let width = preparation.offset_width;
        let child = Description::Canonical(datatype);
        if !matches!(datatype, Datatype::Array { .. }) {
            let field_size = datatype::stored_type_size(datatype, width)?;
            Self::check_member_range(compound_size, name, offset, field_size)?;
            preparation.add_member_prefix(name, offset_bytes)?;
            child.prepare(preparation, None)?;
            return Ok(());
        }

        preparation.add_member_prefix(name, offset_bytes)?;
        let prepared_size = child.prepare(preparation, None);
        // On a child error, check the stored array size and member bounds before returning
        // the child error.
        let field_size = match &prepared_size {
            Ok(size) => *size,
            Err(_) => datatype::stored_type_size(datatype, width)?,
        };
        Self::check_member_range(compound_size, name, offset, field_size)?;
        prepared_size?;
        Ok(())
    }

    fn check_member_range(
        compound_size: u32,
        name: &str,
        offset: u64,
        field_size: u32,
    ) -> Result<(), FormatError> {
        if offset
            .checked_add(u64::from(field_size))
            .is_none_or(|end| end > u64::from(compound_size))
        {
            return Err(FormatError::CompoundFieldOutOfBounds {
                name: name.into(),
                offset,
                field_size,
                compound_size,
            });
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

/// The encoded extent counted so far at the selected offset width.
struct Preparation {
    len: usize,
    offset_width: OffsetWidth,
}

impl Preparation {
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

/// The unwritten suffix of the body and the offset width used to size variable-length descriptors.
pub(super) struct Output<'a> {
    bytes: &'a mut [u8],
    offset_width: OffsetWidth,
}

impl<'a> Output<'a> {
    /// Takes the next `len` bytes and advances the destination.
    ///
    /// # Panics
    ///
    /// Panics if fewer than `len` bytes remain.
    pub(super) fn take(&mut self, len: usize) -> &'a mut [u8] {
        let (field, rest) = mem::take(&mut self.bytes).split_at_mut(len);
        self.bytes = rest;
        field
    }

    pub(super) fn write(&mut self, bytes: &[u8]) {
        self.take(bytes.len()).copy_from_slice(bytes);
    }
}

/// The number of enclosing descriptions on a retained-version path, less than 64.
#[derive(Clone, Copy)]
struct MessageDepth(u8);

impl MessageDepth {
    /// Returns the depth of a child description within the checked nesting limit.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if the child would exceed 64 descriptions
    /// on one parent chain, including the outer description.
    fn child(self) -> Result<Self, FormatError> {
        self.0
            .checked_add(1)
            .filter(|&depth| depth < MAX_MESSAGE_DEPTH)
            .map(Self)
            .ok_or(FormatError::NestingDepthExceeded)
    }

    /// The depth of the outer description.
    const ROOT: Self = Self(0);
}

impl Header {
    /// Checks the class flags against a mask of reserved bits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] if any masked bit is set.
    fn reserved(&self, mask: u32) -> Result<(), FormatError> {
        self.fields().reject_reserved_flags(self.class.code(), mask)
    }

    /// Parses the member count for the property layouts of versions 3 through 5.
    ///
    /// The layouts are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] for versions 1 or 2, and
    /// [`FormatError::InvalidDatatypeFlags`] if bits 16 through 23 are set.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    fn modern_member_count(&self) -> Result<u16, FormatError> {
        let Self {
            class,
            version,
            flags,
            size: _,
        } = self;
        version.for_class(class.code(), DatatypeVersion::Three)?;
        self.reserved(MEMBER_COUNT_RESERVED)?;
        Ok(u16::from_le_bytes([flags[0], flags[1]]))
    }
}

/// A checked description with its source end position and declared element size.
struct Parsed {
    /// The description, including retained versions at each node.
    message: DatatypeMessage,
    /// The first byte after this description within the input slice.
    end: usize,
    /// The element size from the source header, in bytes.
    size: u32,
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

/// Parses an unpadded UTF-8 name and the position after its NUL terminator.
///
/// The unpadded NUL-terminated name field is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec]. This representation requires UTF-8 text.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the terminator is missing, and
/// [`FormatError::UnsupportedDatatypeMemberEncoding`] if the name is not UTF-8.
/// The encoding error includes `class`, `position`, and the original [`str::Utf8Error`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn member_name(data: &[u8], class: u8, position: usize) -> Result<(String, usize), FormatError> {
    let rest = data.get(position..).ok_or(FormatError::UnexpectedEof {
        expected: position.saturating_add(1),
        available: data.len(),
    })?;
    let length = rest
        .iter()
        .position(|&byte| byte == 0)
        .ok_or(FormatError::UnexpectedEof {
            expected: data.len().saturating_add(1),
            available: data.len(),
        })?;
    let bytes = rest.get(..length).ok_or_else(|| {
        FormatError::Internal("a terminated member name exceeds its input".into())
    })?;
    let name =
        str::from_utf8(bytes).map_err(|source| FormatError::UnsupportedDatatypeMemberEncoding {
            class,
            offset: position,
            source,
        })?;
    let end = position
        .checked_add(length)
        .and_then(|end| end.checked_add(1))
        .ok_or(FormatError::UnexpectedEof {
            expected: usize::MAX,
            available: data.len(),
        })?;
    Ok((String::from(name), end))
}

/// Parses the heap-address width from a variable-length descriptor size.
///
/// # Errors
///
/// Returns [`FormatError::InvalidVariableLengthSize`] if `size` is not 10, 12, or 16 bytes.
fn descriptor_width(size: u32) -> Result<OffsetWidth, FormatError> {
    let width = size
        .checked_sub(datatype::SEQUENCE_LENGTH_SIZE + datatype::HEAP_INDEX_SIZE)
        .ok_or(FormatError::InvalidVariableLengthSize { size })?;
    let width =
        u8::try_from(width).map_err(|_err| FormatError::InvalidVariableLengthSize { size })?;
    OffsetWidth::try_from(width).map_err(|_err| FormatError::InvalidVariableLengthSize { size })
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

/// The mask for reserved array class bits 0 through 23.
///
/// Array descriptions at versions 3 through 5 have no class flags, as defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const ARRAY_RESERVED: u32 = 0xff_ffff;

/// The mask for reserved compound and enumeration class bits 16 through 23.
///
/// The fields are defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const MEMBER_COUNT_RESERVED: u32 = 0xff_0000;

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
