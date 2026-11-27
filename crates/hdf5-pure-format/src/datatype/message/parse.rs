//! Recursive parsing that retains the message version and properties of each datatype.

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::num::NonZeroU32;

use hdf5_pure_core::__private::DatatypeOffsets;
use hdf5_pure_core::{DatatypeMemberName, FormatError};

use super::atomic::AtomicReservedFlagPolicy;
use super::compound_layout::CompoundLayout;
use super::decode::{self, Layout, Member, Projection};
use super::representation::{
    self, CompoundV1Member, DatatypeMessage, DatatypeMessageMember, DatatypeProperties,
};
use super::version::DatatypeVersion;
use crate::bytes::ByteCursor;
use crate::codec::{self, DecodeFrame, DecodePrefix, Parsed};
use crate::datatype::header::{
    ARRAY_CLASS, COMPOUND_CLASS, Context, ENUMERATION_CLASS, Header, REFERENCE_CLASS,
};
use crate::datatype::legacy_member::CompoundV1Shape;
use crate::datatype::nesting::MessageDepth;
use crate::datatype::stored_size;
use crate::datatype::variable_length::VariableLengthMetadata;

impl DatatypeMessage {
    /// Parses one datatype message from the start of `data` and returns its length in bytes.
    ///
    /// Reads each complete eight-byte header before checking its version or class. Checks version
    /// before class, except in a complex datatype's parent, where it checks class first.
    /// Reads fixed-size atomic property blocks in full before checking class flags.
    /// Acquires all array dimensions before reading the base datatype. Checks the enumeration
    /// size against its base size, reads each complete name field including any padding, and
    /// then acquires the complete value list.
    ///
    /// Supports fixed-point, floating-point, time, string, bitfield, opaque, variable-length,
    /// enumeration, and compound descriptions at versions 1 through 5, arrays at versions
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
    /// Checks compound member bounds and overlaps using each nested datatype's stored size.
    /// For a version 1 inline array, multiplies that size by its active extents. Reads the
    /// complete dimension block before checking its fields and preserves all four extent slots.
    /// Reserves member capacity after reading a member's nested datatype and checking its
    /// byte range. A compound must have at least one member. Compound member names must
    /// be unique by byte sequence.
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
    /// [`FormatError::InvalidDatatypeVersion`] for array versions 1 or 2,
    /// complex versions 1 through 4, or a complex parent version outside `1..=5`.
    ///
    /// Returns [`FormatError::InvalidVariableLengthSize`] for an unsupported descriptor size,
    /// [`FormatError::DatatypeChildVersion`] if a nested description has a higher version than
    /// its container, and [`FormatError::NestingDepthExceeded`] if the parent chain exceeds the
    /// nesting limit.
    /// Returns [`FormatError::ZeroSizedDatatype`] if an array dimension is zero,
    /// [`FormatError::LengthTooLarge`] if the computed array size exceeds the size field, or
    /// [`FormatError::InvalidDatatypeSize`] if the declared array size differs from the computed size.
    ///
    /// Returns [`FormatError::InvalidCompoundMemberCount`] for a compound without members,
    /// [`FormatError::DuplicateCompoundField`] for duplicate names, and
    /// [`FormatError::CompoundFieldOutOfBounds`] for a member outside its compound element.
    /// Returns [`FormatError::CompoundFieldOverlap`] for overlapping member ranges.
    /// For a version 1 compound, returns [`FormatError::InvalidCompoundV1Rank`] if the rank
    /// exceeds four, [`FormatError::InvalidCompoundV1ReservedByte`] if a reserved byte in the
    /// dimension block is nonzero, [`FormatError::UnsupportedCompoundV1Permutation`] if the
    /// permutation is nonzero, [`FormatError::InvalidCompoundV1Extent`] if an active extent is
    /// zero, or [`FormatError::CompoundArraySizeOverflow`] if the member size exceeds [`u32::MAX`].
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
    /// let (message, consumed) = DatatypeMessage::parse_prefix(&bytes)?;
    /// assert_eq!(message.version(), DatatypeVersion::Five);
    /// assert_eq!(consumed, 12);
    /// assert_eq!(message.serialize(OffsetWidth::Eight)?, bytes[..consumed]);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    pub fn parse_prefix(data: &[u8]) -> Result<(Self, usize), FormatError> {
        Ok(Self::decode_prefix(data, ())?.into_parts())
    }

    /// Parses exactly one datatype description from `data`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::parse_prefix`] for an invalid description.
    /// Returns [`FormatError::TrailingMetadataBytes`] if bytes follow a valid description.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        Self::decode_frame(data, ())
    }

    fn parse_at(
        cursor: &mut ByteCursor<'_>,
        depth: MessageDepth,
    ) -> Result<DecodedMessage, FormatError> {
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
                return Ok(DecodedMessage {
                    datatype: Self::from_parsed(ParsedProperties::new(header.version, properties)),
                    stored_size: stored_size::positive_size(header.class.code(), header.size)?,
                    offsets,
                });
            }
            Layout::Compound(compound) => {
                let stored_size = stored_size::positive_size(COMPOUND_CLASS, header.size)?;
                let count = compound.member_count();
                representation::compound_member_count(count)?;
                let mut members = ParsedCompoundMembers::new(header.version);
                let mut layout = CompoundLayout::new(header.size);
                for _ in 0..count {
                    let Member {
                        name,
                        offset,
                        shape,
                    } = compound.member(cursor)?;
                    let DecodedMessage {
                        datatype,
                        stored_size: field_size,
                        offsets: field_offsets,
                    } = Self::parse_at(cursor, depth.child()?)?;
                    let field_size = match &shape {
                        Some(shape) => shape.stored_size(field_size)?,
                        None => field_size,
                    };
                    let index = members.len();
                    layout.insert(offset, field_size, index, |member_index| {
                        if member_index == index {
                            Some(DatatypeMemberName::from(name))
                        } else {
                            members
                                .get(member_index)
                                .map(|member| member.name().clone())
                        }
                    })?;
                    members.push(
                        DatatypeMessageMember::new(
                            DatatypeMemberName::from(name),
                            offset,
                            datatype,
                        ),
                        shape,
                        count,
                    )?;
                    offsets.include_field(field_offsets);
                }
                drop(layout);
                return Ok(DecodedMessage {
                    datatype: Self::from_parsed(ParsedProperties::new(
                        header.version,
                        members.into_properties(header.version, header.size)?,
                    )),
                    stored_size,
                    offsets,
                });
            }
            Layout::Reference => return Err(FormatError::InvalidDatatypeClass(REFERENCE_CLASS)),
            Layout::Enumeration(enumeration) => {
                let DecodedMessage {
                    datatype: base_type,
                    stored_size: base_size,
                    offsets: base_offsets,
                } = Self::parse_at(cursor, depth.child()?)?;
                let members = enumeration.members(cursor, header.size, base_size)?;
                header
                    .version
                    .check_child(ENUMERATION_CLASS, base_type.version())?;
                representation::check_enumeration_members(&members, base_size.get())?;
                return Ok(DecodedMessage {
                    datatype: Self::from_parsed(ParsedProperties {
                        version: header.version,
                        properties: DatatypeProperties::Enumeration {
                            base_type: Box::new(base_type),
                            members,
                        },
                        enumeration_base_offsets: base_offsets,
                    }),
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
                let DecodedMessage {
                    datatype: base_type,
                    stored_size: base_size,
                    offsets: base_offsets,
                } = Self::parse_at(cursor, depth.child()?)?;
                let stored_size = decode::array_size(header.size, base_size, &dimensions)?;
                header
                    .version
                    .check_child(ARRAY_CLASS, base_type.version())?;
                return Ok(DecodedMessage {
                    datatype: Self::from_parsed(ParsedProperties::new(
                        header.version,
                        DatatypeProperties::Array {
                            base_type: Box::new(base_type),
                            dimensions,
                        },
                    )),
                    stored_size,
                    offsets: base_offsets,
                });
            }
            Layout::Complex(complex) => {
                let (datatype, version) = complex.component(cursor, depth)?;
                DatatypeProperties::Complex {
                    component: Box::new(Self::from_parsed(ParsedProperties::new(
                        version,
                        DatatypeProperties::FloatingPoint(datatype.component()),
                    ))),
                }
            }
        };
        Ok(DecodedMessage {
            datatype: Self::new(header.version, properties)?,
            stored_size: stored_size::positive_size(header.class.code(), header.size)?,
            offsets,
        })
    }
}

impl DecodePrefix<'_> for DatatypeMessage {
    type Context = ();
    type Error = FormatError;

    fn decode_prefix(input: &[u8], (): ()) -> Result<Parsed<Self>, FormatError> {
        let mut cursor = ByteCursor::new(input);
        let decoded = Self::parse_at(&mut cursor, MessageDepth::ROOT)?;
        Ok(Parsed::from_cursor(decoded.datatype, &cursor))
    }
}

impl DecodeFrame<'_> for DatatypeMessage {
    type Context = ();
    type Error = FormatError;

    fn decode_frame(frame: &[u8], (): ()) -> Result<Self, FormatError> {
        Ok(codec::decode_exact(frame, ())?)
    }
}

/// A retained description and the stored layout used to decode its enclosing datatype.
struct DecodedMessage {
    datatype: DatatypeMessage,
    /// The size of one element in the file, in bytes.
    stored_size: NonZeroU32,
    /// The offset widths inferred from variable-length descriptors in the description.
    offsets: DatatypeOffsets,
}

/// A version and properties whose layout and relationships the parser has checked.
pub(super) struct ParsedProperties {
    version: DatatypeVersion,
    properties: DatatypeProperties,
    /// The descriptor widths recorded when parsing an enumeration's base.
    enumeration_base_offsets: DatatypeOffsets,
}

impl ParsedProperties {
    fn new(version: DatatypeVersion, properties: DatatypeProperties) -> Self {
        Self {
            version,
            properties,
            enumeration_base_offsets: DatatypeOffsets::default(),
        }
    }

    pub(super) fn into_parts(self) -> (DatatypeVersion, DatatypeProperties, DatatypeOffsets) {
        let Self {
            version,
            properties,
            enumeration_base_offsets,
        } = self;
        (version, properties, enumeration_base_offsets)
    }
}

/// The members of the compound being parsed, with dimension fields for version 1.
enum ParsedCompoundMembers {
    WithoutShape(Vec<DatatypeMessageMember>),
    WithShape(Vec<CompoundV1Member>),
}

impl ParsedCompoundMembers {
    fn new(version: DatatypeVersion) -> Self {
        if version == DatatypeVersion::One {
            Self::WithShape(Vec::new())
        } else {
            Self::WithoutShape(Vec::new())
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::WithoutShape(members) => members.len(),
            Self::WithShape(members) => members.len(),
        }
    }

    fn get(&self, index: usize) -> Option<&DatatypeMessageMember> {
        match self {
            Self::WithoutShape(members) => members.get(index),
            Self::WithShape(members) => members.get(index).map(CompoundV1Member::member),
        }
    }

    fn push(
        &mut self,
        member: DatatypeMessageMember,
        shape: Option<CompoundV1Shape>,
        count: usize,
    ) -> Result<(), FormatError> {
        match (self, shape) {
            (Self::WithoutShape(members), None) => decode::push_member(members, count, member),
            (Self::WithShape(members), Some(shape)) => {
                decode::push_member(members, count, CompoundV1Member::new(member, shape))
            }
            _ => Err(FormatError::Internal(
                "compound member shape differs from the selected layout".into(),
            )),
        }
    }

    fn into_properties(
        self,
        version: DatatypeVersion,
        size: u32,
    ) -> Result<DatatypeProperties, FormatError> {
        let mut names = BTreeSet::new();
        match self {
            Self::WithoutShape(members) => {
                for member in &members {
                    member.check_compound_name_and_version(version, &mut names)?;
                }
                Ok(DatatypeProperties::Compound { size, members })
            }
            Self::WithShape(members) => {
                for member in &members {
                    member
                        .member()
                        .check_compound_name_and_version(version, &mut names)?;
                }
                Ok(DatatypeProperties::CompoundV1 { size, members })
            }
        }
    }
}
