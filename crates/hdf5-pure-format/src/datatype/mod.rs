//! Parsing and serialization of HDF5 Datatype messages.

use alloc::boxed::Box;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::num::NonZeroUsize;

#[cfg(test)]
use byteorder::{ByteOrder, LittleEndian};

use hdf5_pure_core::__private;
pub(crate) use hdf5_pure_core::CharacterSet;
#[cfg(test)]
use hdf5_pure_core::ComplexType;
use hdf5_pure_core::CompoundMember;
pub(crate) use hdf5_pure_core::Datatype;
pub(crate) use hdf5_pure_core::ReferenceType;
pub(crate) use hdf5_pure_core::StringPadding;
use hdf5_pure_core::VariableLengthStringType;
use hdf5_pure_core::VariableLengthType;
#[cfg(test)]
use hdf5_pure_core::{
    BitFieldType, DatatypeByteOrder, FixedPointType, FloatingPointLayout, FloatingPointType,
    MantissaNormalization, TimeType,
};

use self::header::{Context, Header};
use self::message::atomic::{self, AtomicReservedFlagPolicy};
use self::message::decode::{self, Decoded, Layout, Member, Projection};
use self::message::{CompoundLayout, MessageDepth};
pub use self::slots::{EmbeddedReferenceSlots, embedded_reference_slots};
use crate::bytes::ByteCursor;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

pub(crate) mod byte_order;
mod header;
pub(crate) mod layout;
mod legacy_member;
pub(crate) mod message;
pub(crate) mod slots;

/// Returns the datatype's element size as a nonzero host-sized integer.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes
/// per element, or [`FormatError::ValueTooLargeForPlatform`] if its size exceeds
/// the target's `usize` range.
pub fn element_size_usize(datatype: &Datatype) -> Result<NonZeroUsize, FormatError> {
    usize::try_from(datatype.element_size()?.get())
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or(FormatError::ValueTooLargeForPlatform {
            value: u64::from(datatype.type_size()),
            target: "usize",
        })
}

/// Returns `true` if the datatype contains a variable-length string or sequence.
///
/// Examines array elements, enumeration bases, and compound members recursively.
pub fn datatype_holds_variable_length(datatype: &Datatype) -> bool {
    match datatype {
        Datatype::VariableLength(_) => true,
        Datatype::Array { base_type, .. } => datatype_holds_variable_length(base_type),
        Datatype::Enumeration(datatype) => datatype_holds_variable_length(datatype.base_type()),
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|member| datatype_holds_variable_length(&member.datatype)),
        _ => false,
    }
}

/// Returns the nonzero size in bytes of one value stored at `offset_width`.
///
/// Uses [`stored_type_size`] to account for variable-length references and array elements.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes, or the errors of
/// [`stored_type_size`] if an array size exceeds its encoded field.
pub fn stored_element_size(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<NonZeroU32, FormatError> {
    NonZeroU32::new(stored_type_size(datatype, offset_width)?)
        .map_or_else(|| datatype.element_size(), Ok)
}

/// Returns the size in bytes of one stored value, including zero-sized types.
///
/// Computes variable-length reference sizes at `offset_width` and array sizes from their base
/// types and dimensions. Uses [`Datatype::type_size`] for other types, including declared
/// compound record sizes.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte datatype size
/// field.
pub fn stored_type_size(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<u32, FormatError> {
    __private::stored_type_size(datatype, offset_width)
}

/// Returns the stored element size as a nonzero host-sized integer.
///
/// # Errors
///
/// Returns the errors of [`stored_element_size`], or [`FormatError::ValueTooLargeForPlatform`]
/// if the size exceeds the target's [`usize`] range.
pub fn stored_element_size_usize(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<NonZeroUsize, FormatError> {
    stored_element_size(datatype, offset_width)?.narrow()
}

/// Parses a datatype message body and returns the type and the number of bytes read.
///
/// Reads each complete eight-byte header before checking its version or class. Checks version
/// before class, except in a complex datatype's parent, where it checks class first.
/// Reads fixed-size atomic property blocks in full before checking class flags. Ignores
/// reserved class bits in fixed-point, time, string, and bitfield descriptions.
///
/// Accepts legacy object and dataset-region reference kinds at every supported message
/// version and preserves their declared element sizes.
///
/// Reads version 1 and 2 compounds and enumerations, and version 2 arrays. Decodes
/// compound and enumeration names with lossy UTF-8 conversion. Acquires the full
/// array dimension list and any version 2 permutation indices before reading the
/// base datatype. Checks the enumeration size against its base size before reading the
/// member names, then acquires the complete value list.
///
/// Checks compound member bounds and overlaps using the sizes stored in the member
/// descriptions. For a version 1 implicit array member, multiplies the base size by its extents.
/// Reserves member capacity after reading a member's nested datatype and checking its byte range.
/// Checks each array message's declared size against the stored base size multiplied by
/// the dimension extents.
/// Variable-length descriptors occupy 10, 12, or 16 bytes for heap address widths of 2, 4, or
/// 8 bytes, respectively.
///
/// The nesting limit is 64 descriptions along one path, including the outer description and
/// any complex datatype's floating-point parent.
///
/// Reads an opaque tag's complete recorded property and copies its logical bytes into
/// [`Datatype::Opaque`]. Reads a zero-length property as an absent tag and strips the first NUL
/// and its zero padding from a tagged property. Also accepts aligned properties filled entirely
/// by tag bytes. `H5O__dtype_encode_helper` writes that form for tags whose logical length is a
/// multiple of eight (H5Odtype.c, HDF5 2.2.0).
///
/// Parses floating-point message versions and byte-order bits together through
/// [`message::FloatingPointByteOrder::parse`], including nested descriptions in legacy containers.
/// Accepts little- and big-endian order at versions 1 through 5 and VAX order at versions
/// 3 through 5, subject to the numeric layout checks.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete. EOF byte counts
/// are relative to the start of `data`, including for nested descriptions. Returns
/// [`FormatError::InvalidDatatypeMessageVersion`] for a version outside `1..=5`, except in a
/// complex datatype's parent, or [`FormatError::InvalidDatatypeClass`] for an unsupported class.
/// Returns [`FormatError::NestingDepthExceeded`] if a nested description exceeds the limit.
/// Returns [`FormatError::InvalidDatatypeVersion`] for a version unsupported by an array,
/// complex datatype, or its parent. Returns
/// [`FormatError::InvalidReferenceType`] for an unknown reference kind,
/// [`FormatError::InvalidStringPadding`] for an unknown string padding code, or
/// [`FormatError::InvalidCharacterSet`] for an unknown character set code. Returns
/// [`FormatError::ZeroSizedDatatype`] if the parsed type has no element bytes.
/// Returns [`FormatError::LengthTooLarge`] if the computed array size exceeds the size field, or
/// [`FormatError::InvalidDatatypeSize`] if the declared array size differs from the computed size.
/// Returns [`FormatError::InvalidFixedPointRange`], [`FormatError::InvalidFloatingPointRange`], or
/// [`FormatError::InvalidNumericBitRange`] if a numeric bit range is empty or exceeds its element
/// size. Returns [`FormatError::InvalidFloatingPointField`] if a floating-point field is empty or
/// outside that range, [`FormatError::OverlappingFloatingPointFields`] if fields overlap, or
/// [`FormatError::InvalidMantissaNormalization`] if the normalization code is reserved.
///
/// For a floating-point description outside a complex parent, returns
/// [`FormatError::InvalidByteOrder`] for masked order bits `0x40`,
/// [`FormatError::InvalidDatatypeVersion`] for VAX order in versions 1 or 2, or
/// [`FormatError::InvalidDatatypeFlags`] for reserved class bits.
///
/// Returns [`FormatError::InvalidOpaqueReservedBits`] if an opaque datatype has nonzero reserved
/// bits, [`FormatError::InvalidOpaqueTagLength`] if its property length is not a multiple of
/// eight, or [`FormatError::InvalidOpaqueTagPadding`] if a byte after the first NUL is nonzero.
///
/// Returns [`FormatError::InvalidVariableLengthSubtype`] if a variable-length subtype is
/// neither sequence nor string, [`FormatError::InvalidVariableLengthReservedBits`] if its
/// class bit field has reserved bits set, or
/// [`FormatError::InvalidVariableLengthSequenceMetadata`] if a sequence has nonzero string
/// padding or character-set metadata. Returns [`FormatError::InvalidVariableLengthSize`] for an
/// unsupported descriptor size.
///
/// Returns [`FormatError::InvalidCompoundV1Rank`] if a version 1 compound member declares more
/// than four dimensions, [`FormatError::InvalidCompoundV1Extent`] if an active dimension is
/// zero, or [`FormatError::UnsupportedCompoundV1Permutation`] if its permutation field is nonzero.
/// Returns [`FormatError::CompoundArraySizeOverflow`] if a version 1 member's implicit array
/// size exceeds [`u32::MAX`]. Returns [`FormatError::CompoundFieldOutOfBounds`] if a member
/// extends past the compound's declared size or [`FormatError::CompoundFieldOverlap`] if member
/// byte ranges overlap.
///
/// Returns [`FormatError::EnumBaseSizeMismatch`] if the enumeration and its base datatype
/// declare different sizes, or [`FormatError::AllocationFailed`] if a compound or enumeration
/// member list reservation fails.
///
/// Returns [`FormatError::InvalidComplexFlags`] if a complex class bit field has reserved bits,
/// [`FormatError::UnsupportedComplexForm`] if the complex form is not homogeneous rectangular,
/// or [`FormatError::InvalidComplexParent`] if its parent is not floating point. Returns
/// [`FormatError::InvalidComplexParentFlags`] if the parent has reserved bits, the reserved
/// order pattern, or VAX order in versions 1 or 2. Returns
/// [`FormatError::ComplexSizeOverflow`] if twice its size exceeds [`u32::MAX`], or
/// [`FormatError::InvalidComplexSize`] if the declared complex size differs from that product.
pub fn parse_datatype(data: &[u8]) -> Result<(Datatype, usize), FormatError> {
    let mut cursor = ByteCursor::new(data);
    let Decoded { datatype, .. } = parse_datatype_at(&mut cursor, MessageDepth::ROOT)?;
    Ok((datatype, cursor.position()))
}

/// Parses the next datatype and retains its stored element size.
///
/// # Errors
///
/// Returns the errors of [`parse_datatype`].
fn parse_datatype_at(
    cursor: &mut ByteCursor<'_>,
    depth: MessageDepth,
) -> Result<Decoded<Datatype>, FormatError> {
    let header = Header::parse(cursor, Context::Description)?;
    let size = header.size;
    let mut offsets = __private::DatatypeOffsets::default();
    let datatype = match Layout::from_header(&header, Projection::Semantic)? {
        Layout::Atomic(class) => class
            .decode(
                cursor,
                header.fields(),
                Context::Description,
                AtomicReservedFlagPolicy::Ignore,
            )?
            .into_datatype(),
        Layout::Compound(compound) => {
            let count = compound.member_count();
            let mut members: Vec<CompoundMember> = Vec::new();
            let mut layout = CompoundLayout::new(size);
            for _ in 0..count {
                let Member {
                    name,
                    offset,
                    shape,
                } = compound.member(cursor)?;
                let Decoded {
                    datatype,
                    stored_size,
                    offsets: field_offsets,
                } = parse_nested_datatype(cursor, depth)?;
                let stored_size = match &shape {
                    Some(shape) => shape.stored_size(stored_size)?,
                    None => stored_size,
                };
                let index = members.len();
                layout.insert(offset, stored_size, index, |member_index| {
                    if member_index == index {
                        Some(name.as_ref())
                    } else {
                        members.get(member_index).map(|member| member.name.as_str())
                    }
                })?;
                let datatype = match shape {
                    Some(shape) => shape.into_datatype(datatype),
                    None => datatype,
                };
                decode::push_member(
                    &mut members,
                    count,
                    __private::compound_member(name.into_owned(), offset, datatype),
                )?;
                offsets.include_field(field_offsets);
            }
            Datatype::Compound { size, members }
        }
        Layout::Reference => atomic::decode_reference(&header)?,
        Layout::Enumeration(enumeration) => {
            let Decoded {
                datatype: base_type,
                stored_size: base_size,
                offsets: base_offsets,
            } = parse_nested_datatype(cursor, depth)?;
            let members = enumeration.members(cursor, size, base_size)?;
            offsets = base_offsets.in_enumeration();
            Datatype::Enumeration(__private::enumeration_from_stored_parts(
                base_type,
                base_size,
                base_offsets,
                members,
            )?)
        }
        Layout::VariableLength(metadata, width) => {
            let base = parse_nested_datatype(cursor, depth)?;
            offsets = __private::DatatypeOffsets::descriptor(width, base.offsets);
            let kind = match metadata {
                VariableLengthMetadata::Sequence => VariableLengthType::sequence(base.datatype),
                VariableLengthMetadata::String { padding, charset } => VariableLengthType::String(
                    VariableLengthStringType::with_base_type(padding, charset, base.datatype)?,
                ),
            };
            Datatype::VariableLength(kind)
        }
        Layout::Array(array) => {
            let dimensions = array.dimensions(cursor)?;
            let base = parse_nested_datatype(cursor, depth)?;
            let stored_size = decode::array_size(size, base.stored_size, &dimensions)?;
            return Ok(Decoded {
                datatype: Datatype::Array {
                    base_type: Box::new(base.datatype),
                    dimensions,
                },
                stored_size,
                offsets: base.offsets,
            });
        }
        Layout::Complex(complex) => {
            let (datatype, _) = complex.component(cursor, depth)?;
            Datatype::Complex(datatype)
        }
    };
    Ok(Decoded {
        datatype,
        stored_size: message::positive_size(header.class.code(), size)?,
        offsets,
    })
}

/// Parses the next description from the bytes remaining at the cursor.
///
/// Checks the nesting limit before reading the child. Uses byte offsets relative to the original
/// input when reading the child. Advances the outer cursor by the consumed length on success.
///
/// # Errors
///
/// Returns the errors of [`parse_datatype`].
fn parse_nested_datatype(
    cursor: &mut ByteCursor<'_>,
    parent_depth: MessageDepth,
) -> Result<Decoded<Datatype>, FormatError> {
    let depth = parent_depth.child()?;
    let mut nested = *cursor;
    let decoded = parse_datatype_at(&mut nested, depth)?;
    *cursor = nested;
    Ok(decoded)
}

/// Serializes a datatype into an HDF5 Datatype message body.
///
/// Writes fixed-point, time, string, bitfield, opaque, and little- or big-endian floating-point
/// descriptions at version 1, and VAX floating-point descriptions at version 3.
/// Uses version 1 as the minimum for variable-length descriptions and version 3 for compounds,
/// enumerations, and arrays. Writes object and dataset-region references at version 1, retaining
/// their supplied sizes.
/// Writes [`Datatype::Complex`] as class 11 in version 5, with one floating-point parent.
/// Writes each containing datatype at a message version at least as high as its children's.
/// Writes a nonempty opaque tag with a NUL terminator and zero padding to a multiple of eight
/// bytes, recording that complete property length in the class bit field. Writes an empty tag
/// with a zero-length property. Preserves non-ASCII tag bytes.
///
/// Sizes variable-length descriptors for an eight-byte offset width at every level. Checks
/// compound member ranges and computes array sizes from the stored element sizes at that width.
///
/// Canonical serialization accepts empty compound or enumeration member sets and zero
/// array extents. An enumeration's value width must match its base type's size at the
/// eight-byte offset width. Descriptors in a decoded enumeration's base must use eight-byte
/// offsets. Writes each enumeration value as supplied.
///
/// # Errors
///
/// Returns the errors of [`message::PreparedDatatype::new`].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{Datatype, FixedPointType, FormatError};
/// use hdf5_pure_format::__private::serialize_datatype;
///
/// let datatype = Datatype::FixedPoint(FixedPointType::u16());
/// assert_eq!(
///     serialize_datatype(&datatype)?,
///     [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0],
/// );
/// # Ok::<(), FormatError>(())
/// ```
pub fn serialize_datatype(datatype: &Datatype) -> Result<Vec<u8>, FormatError> {
    serialize_datatype_with_offset_width(datatype, OffsetWidth::Eight)
}

/// Serializes a datatype using the file's offset width.
///
/// Sizes variable-length descriptors at every level using `offset_width`. Selects member-offset
/// field widths from the declared compound size, and computes array sizes from their encoded base
/// types.
///
/// # Errors
///
/// Returns the errors of [`message::PreparedDatatype::new`].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{Datatype, FixedPointType, FormatError, VariableLengthType};
/// use hdf5_pure_format::__private::{
///     serialize_datatype_with_offset_width, DatatypeMessage, OffsetWidth,
/// };
///
/// let sequence = Datatype::VariableLength(VariableLengthType::sequence(
///     Datatype::FixedPoint(FixedPointType::u16()),
/// ));
/// let array = Datatype::Array { base_type: Box::new(sequence), dimensions: vec![2] };
/// let bytes = serialize_datatype_with_offset_width(&array, OffsetWidth::Two)?;
/// assert_eq!(&bytes[..8], &[0x3a, 0, 0, 0, 20, 0, 0, 0]);
/// assert_eq!(&bytes[13..21], &[0x19, 0, 0, 0, 10, 0, 0, 0]);
/// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
/// assert_eq!(consumed, bytes.len());
/// assert_eq!(parsed.to_datatype()?, array);
/// # Ok::<(), FormatError>(())
/// ```
pub fn serialize_datatype_with_offset_width(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<Vec<u8>, FormatError> {
    Ok(message::PreparedDatatype::new(datatype, offset_width)?.to_bytes())
}

/// An opaque tag with a checked aligned property length and borrowed logical bytes.
///
/// The parser accepts a full property without a terminator. The encoder reserves a terminator
/// for every nonempty tag.
struct OpaqueTagWire<'a> {
    encoded_len: u8,
    logical: &'a [u8],
}

impl<'a> OpaqueTagWire<'a> {
    /// Borrows the logical tag bytes from a complete aligned property.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOpaqueTagLength`] if the declared length is not a multiple
    /// of eight. Returns [`FormatError::UnexpectedEof`] if the property is incomplete, with
    /// byte counts relative to the cursor's original input. Returns
    /// [`FormatError::InvalidOpaqueTagPadding`] for a nonzero byte after the first NUL, with
    /// the byte's offset relative to the start of the property.
    fn parse(encoded_len: u8, cursor: &mut ByteCursor<'a>) -> Result<Self, FormatError> {
        if !encoded_len.is_multiple_of(OPAQUE_TAG_ALIGNMENT) {
            return Err(FormatError::InvalidOpaqueTagLength {
                length: encoded_len,
            });
        }
        let property = cursor.take(usize::from(encoded_len))?;
        // `H5O__dtype_encode_helper` fills aligned tags without a terminator
        // (H5Odtype.c, HDF5 2.2.0).
        let terminator = property.iter().position(|&byte| byte == 0);
        let logical = if let Some(terminator) = terminator {
            if let Some(offset) = property[terminator + 1..]
                .iter()
                .position(|&byte| byte != 0)
            {
                let offset = terminator + 1 + offset;
                return Err(FormatError::InvalidOpaqueTagPadding {
                    offset,
                    byte: property[offset],
                });
            }
            &property[..terminator]
        } else {
            property
        };
        Ok(Self {
            encoded_len,
            logical,
        })
    }

    fn encode(logical: &'a [u8]) -> Result<Self, FormatError> {
        if logical.len() > OPAQUE_TAG_MAX_LEN {
            return Err(FormatError::OpaqueTagTooLong {
                length: logical.len(),
            });
        }
        if let Some(offset) = logical.iter().position(|&byte| byte == 0) {
            return Err(FormatError::InvalidOpaqueTagByte {
                offset,
                byte: logical[offset],
            });
        }
        let encoded_len = u8::try_from(Self::property_size(logical.len())).map_err(|_error| {
            FormatError::OpaqueTagTooLong {
                length: logical.len(),
            }
        })?;
        Ok(Self {
            encoded_len,
            logical,
        })
    }

    fn property_size(tag_len: usize) -> usize {
        if tag_len == 0 {
            0
        } else {
            let alignment = usize::from(OPAQUE_TAG_ALIGNMENT);
            (tag_len + alignment) & !(alignment - 1)
        }
    }
}

/// The alignment of an opaque tag property, defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// `H5O__dtype_decode_helper` requires an aligned recorded length (H5Odtype.c, HDF5 2.2.0).
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const OPAQUE_TAG_ALIGNMENT: u8 = 8;
/// The maximum logical tag length with space for its terminator in an aligned one-byte length.
///
/// The length field and padding are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const OPAQUE_TAG_MAX_LEN: usize = 247;

/// The byte length of the common datatype message header.
///
/// The header for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_HEADER_SIZE: usize = 8;
/// The mask for the class code in the common datatype message header.
///
/// The class field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_CLASS_MASK: u8 = 0x0F;
/// The bit position of the version in the common datatype message header.
///
/// The version field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_SHIFT: u8 = 4;
/// The homogeneous flag in a native complex datatype's class bit field.
///
/// The flag for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_HOMOGENEOUS_BIT: u8 = 0x01;
/// The mask for the form in a native complex datatype's class bit field.
///
/// The field for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_FORM_MASK: u8 = 0x06;
/// The bit position of the form in a native complex datatype's class bit field.
///
/// The field for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_FORM_SHIFT: u8 = 1;
/// The rectangular form code of a native complex datatype.
///
/// The code for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_RECTANGULAR_FORM: u8 = 0;
/// The mask for reserved bits 3 through 23 of a complex class bit field.
///
/// The bits for datatype message version 5 are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_RESERVED_MASK: u32 = 0xFF_FFF8;
/// The mask for reserved bits 7 and 16 through 23 of a floating-point class bit field.
///
/// The bits for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_RESERVED_MASK: u32 = 0xFF_0080;

fn padding_bits(low: bool, high: bool) -> u8 {
    (if low { LOW_PADDING_BIT } else { 0 }) | (if high { HIGH_PADDING_BIT } else { 0 })
}

/// Bit 1 of the fixed-point, floating-point, and bitfield class bit fields, the low padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const LOW_PADDING_BIT: u8 = 0x02;
/// Bit 2 of the fixed-point, floating-point, and bitfield class bit fields, the high padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const HIGH_PADDING_BIT: u8 = 0x04;
/// Bit 3 of the floating-point class bit field, the internal padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_INTERNAL_PADDING_BIT: u8 = 0x08;

enum VariableLengthMetadata {
    Sequence,
    String {
        padding: StringPadding,
        charset: CharacterSet,
    },
}

fn parse_variable_length_metadata(flags: [u8; 3]) -> Result<VariableLengthMetadata, FormatError> {
    let [bf0, bf1, bf2] = flags;
    let subtype = bf0 & VARIABLE_LENGTH_FIELD_MASK;
    if !matches!(subtype, VARIABLE_LENGTH_SEQUENCE | VARIABLE_LENGTH_STRING) {
        return Err(FormatError::InvalidVariableLengthSubtype(subtype));
    }
    let bits = u32::from_le_bytes([bf0, bf1, bf2, 0]) & VARIABLE_LENGTH_RESERVED_MASK;
    if bits != 0 {
        return Err(FormatError::InvalidVariableLengthReservedBits(bits));
    }
    let padding = bf0 >> VARIABLE_LENGTH_PADDING_SHIFT;
    let charset = bf1 & VARIABLE_LENGTH_FIELD_MASK;
    if subtype == VARIABLE_LENGTH_STRING {
        return Ok(VariableLengthMetadata::String {
            padding: parse_string_padding(padding)?,
            charset: parse_charset(charset)?,
        });
    }
    if padding != 0 || charset != 0 {
        return Err(FormatError::InvalidVariableLengthSequenceMetadata { padding, charset });
    }
    Ok(VariableLengthMetadata::Sequence)
}

/// The mask for a four-bit variable-length subtype or character-set code.
///
/// The fields for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_FIELD_MASK: u8 = 0x0F;
/// The variable-length sequence subtype code.
///
/// The code for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_SEQUENCE: u8 = 0;
/// The variable-length string subtype code.
///
/// The code for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_STRING: u8 = 1;
/// The bit position of string padding in the variable-length class bit field.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_PADDING_SHIFT: u8 = 4;
/// The mask for reserved bits 12 through 23 of the variable-length class bit field.
///
/// The bits for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_RESERVED_MASK: u32 = 0xFF_F000;

fn parse_string_padding(val: u8) -> Result<StringPadding, FormatError> {
    match val {
        0 => Ok(StringPadding::NullTerminate),
        1 => Ok(StringPadding::NullPad),
        2 => Ok(StringPadding::SpacePad),
        _ => Err(FormatError::InvalidStringPadding(val)),
    }
}

/// Bit 0 of a floating-point class bit field, the low bit of its byte order.
const FLOAT_BYTE_ORDER_LOW_BIT: u8 = 0x01;

/// Bit 6 of a floating-point class bit field, the high bit of its byte order.
const FLOAT_BYTE_ORDER_HIGH_BIT: u8 = 0x40;

/// The bit position of mantissa normalization in the floating-point class bit field.
///
/// The field and its values are defined in "The Datatype Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_SHIFT: u8 = 4;
/// The mask for the two mantissa normalization bits after shifting.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_MASK: u8 = 0x03;
/// The encoded value for a mantissa stored without normalization.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_UNNORMALIZED: u8 = 0;
/// The encoded value for a stored leading mantissa bit that is set, except for zero.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_LEADING_ONE: u8 = 1;
/// The encoded value for an implied leading mantissa bit.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE: u8 = 2;
/// The reserved mantissa normalization value.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[cfg(test)]
const FLOAT_NORMALIZATION_RESERVED: u8 = 3;

fn parse_charset(val: u8) -> Result<CharacterSet, FormatError> {
    match val {
        0 => Ok(CharacterSet::Ascii),
        1 => Ok(CharacterSet::Utf8),
        _ => Err(FormatError::InvalidCharacterSet(val)),
    }
}

/// Returns the member offset width for a version 3 compound datatype of `compound_size` bytes.
fn offset_bytes_for_size(compound_size: u32) -> usize {
    // "The Datatype Message" in the format specification, version 4.0, defines the
    // version 3 member offset width as the minimum byte count that holds the datatype size.
    compound_size
        .to_le_bytes()
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(1, |index| index + 1)
}

/// Returns `true` if a datatype of this encoded class can hold an object
/// address, based only on the first byte of a datatype message.
///
/// The check is necessary for [`datatype_holds_object_address`] but not
/// sufficient. A compound of two integers passes it, so the caller must not
/// treat a `true` result as proof that an address is present. A walk over every
/// object in a file calls this to skip the common cases, such as fixed-point,
/// floating-point, string, and opaque datasets, without allocating a parsed
/// [`Datatype`] for each.
///
/// The admitted classes match the ones that `datatype_holds_object_address`
/// recurses through, plus the reference class. The test
/// `class_predicate_admits_every_reference_holding_type` enforces that match.
pub fn class_may_hold_object_address(class_and_version: u8) -> bool {
    matches!(
        class_and_version & 0x0F,
        COMPOUND_CLASS | REFERENCE_CLASS | ENUMERATION_CLASS | VARIABLE_LENGTH_CLASS | ARRAY_CLASS
    )
}

/// Datatype message class ids, as the low nibble of a datatype message's first
/// byte. Only the classes that can carry an object address downward are named.
// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, assigns these datatype class and version codes.
const FIXED_POINT_CLASS: u8 = 0;
const FLOATING_POINT_CLASS: u8 = 1;
const TIME_CLASS: u8 = 2;
const STRING_CLASS: u8 = 3;
const BIT_FIELD_CLASS: u8 = 4;
const OPAQUE_CLASS: u8 = 5;
const COMPOUND_CLASS: u8 = 6;
const REFERENCE_CLASS: u8 = 7;
const ENUMERATION_CLASS: u8 = 8;
const VARIABLE_LENGTH_CLASS: u8 = 9;
const ARRAY_CLASS: u8 = 10;
const COMPLEX_CLASS: u8 = 11;
const DATATYPE_VERSION_ONE: u8 = 1;
const DATATYPE_VERSION_TWO: u8 = 2;
const DATATYPE_VERSION_THREE: u8 = 3;
/// Datatype message version 4, defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_FOUR: u8 = 4;
const DATATYPE_VERSION_FIVE: u8 = 5;

/// Returns `true` if `dt` contains an object address anywhere in its structure.
///
/// An object address is an object reference or a dataset-region reference. This
/// function searches compound members, array entries, enumeration bases, and
/// variable-length contents.
///
/// [`embedded_reference_slots`] maps only the 8-byte form reachable through
/// compound members and array entries. An address outside that form cannot be
/// read or freed, so the caller must reject the datatype that carries it. The
/// caller's fall-through arm consults this function so the two cannot drift
/// apart.
///
/// A variable-length datatype counts only when its contents are an object
/// reference. A deletion frees object headers and dataset storage, and a global
/// heap collection survives, so a variable-length string keeps pointing at data
/// that is still present. A `H5T_VLEN` of `H5T_STD_REF_OBJ`, which the
/// reference library writes, keeps its addresses in the heap contents, where the
/// element bytes hold only a heap id.
pub fn datatype_holds_object_address(dt: &Datatype) -> bool {
    match dt {
        Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_object_address(&m.datatype)),
        Datatype::Array { base_type, .. } => datatype_holds_object_address(base_type),
        Datatype::Enumeration(datatype) => datatype_holds_object_address(datatype.base_type()),
        Datatype::VariableLength(kind) => datatype_holds_object_address(kind.base_type()),
        Datatype::Complex(_) => false,
        _ => false,
    }
}

/// Returns `true` if an element of `dt` holds a file address at any depth.
///
/// A variable-length element and a dataset region reference each hold the address of a global
/// heap collection and an index into it, and an object reference holds the address of an object
/// header. The search descends into compound members, array elements, and enumeration bases.
///
/// The function does not report which kind of address an element holds, or where.
/// [`datatype_holds_object_address`] checks for a reference alone, and returns `false` for a
/// variable-length string.
pub fn datatype_holds_file_address(dt: &Datatype) -> bool {
    match dt {
        Datatype::VariableLength(_) | Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_file_address(&m.datatype)),
        Datatype::Array { base_type, .. } => datatype_holds_file_address(base_type),
        Datatype::Enumeration(datatype) => datatype_holds_file_address(datatype.base_type()),
        Datatype::Complex(_) => false,
        _ => false,
    }
}

/// Build a datatype header (8 bytes) for testing.
#[cfg(test)]
fn build_dt_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
    buf[1] = bf[0];
    buf[2] = bf[1];
    buf[3] = bf[2];
    LittleEndian::write_u32(&mut buf[4..8], size);
    buf
}

/// The four-byte Size field in "The Datatype Message" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[cfg(test)]
const DATATYPE_SIZE_BYTES: u8 = 4;

#[cfg(test)]
mod tests;
