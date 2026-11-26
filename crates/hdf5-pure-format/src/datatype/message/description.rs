//! Datatype descriptions with checked message versions and recursive properties.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::num::NonZeroU16;
use core::num::NonZeroU32;

use hdf5_pure_core::__private;
use hdf5_pure_core::__private::DatatypeOffsets;
use hdf5_pure_core::{
    BitFieldType, CharacterSet, ComplexType, CompoundMember, Datatype, DatatypeByteOrder,
    DatatypeMemberName, EnumMember, FixedPointType, FloatingPointType, FormatError, StringPadding,
    TimeType, VariableLengthStringType, VariableLengthType,
};

use super::codec::ParsedProperties;
use super::compound_layout::CompoundLayout;
use crate::convert::Narrow;
use crate::datatype::header::{
    ARRAY_CLASS, BIT_FIELD_CLASS, COMPLEX_CLASS, COMPOUND_CLASS, DATATYPE_VERSION_FIVE,
    DATATYPE_VERSION_FOUR, DATATYPE_VERSION_ONE, DATATYPE_VERSION_THREE, DATATYPE_VERSION_TWO,
    ENUMERATION_CLASS, FIXED_POINT_CLASS, FLOATING_POINT_CLASS, OPAQUE_CLASS, STRING_CLASS,
    TIME_CLASS, VARIABLE_LENGTH_CLASS,
};
use crate::datatype::legacy_member::CompoundV1Shape;
use crate::datatype::opaque::OPAQUE_TAG_ALIGNMENT;
use crate::width::OffsetWidth;

/// A datatype message version from 1 through 5.
///
/// The version occupies the high four bits of the class-and-version byte, defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum DatatypeVersion {
    /// Datatype message version 1.
    One = DATATYPE_VERSION_ONE,
    /// Datatype message version 2.
    Two = DATATYPE_VERSION_TWO,
    /// Datatype message version 3.
    Three = DATATYPE_VERSION_THREE,
    /// Datatype message version 4.
    Four = DATATYPE_VERSION_FOUR,
    /// Datatype message version 5.
    Five = DATATYPE_VERSION_FIVE,
}

impl DatatypeVersion {
    /// Returns the version number stored in the message header.
    pub const fn get(self) -> u8 {
        self as u8
    }

    /// Returns this version if it meets the class layout's minimum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] if this version is below `minimum`.
    pub(super) fn for_class(self, class: u8, minimum: Self) -> Result<Self, FormatError> {
        if self < minimum {
            return Err(FormatError::InvalidDatatypeVersion {
                class,
                version: self.get(),
            });
        }
        Ok(self)
    }

    /// Checks the child's version against its containing message's version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DatatypeChildVersion`] if the child's version is higher.
    pub(super) fn check_child(self, class: u8, child: Self) -> Result<(), FormatError> {
        if child > self {
            return Err(FormatError::DatatypeChildVersion {
                class,
                version: self.get(),
                child_version: child.get(),
            });
        }
        Ok(())
    }
}

impl TryFrom<u8> for DatatypeVersion {
    type Error = FormatError;

    /// Converts a stored version number to a supported message version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeMessageVersion`] if `value` is outside `1..=5`.
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            DATATYPE_VERSION_ONE => Ok(Self::One),
            DATATYPE_VERSION_TWO => Ok(Self::Two),
            DATATYPE_VERSION_THREE => Ok(Self::Three),
            DATATYPE_VERSION_FOUR => Ok(Self::Four),
            DATATYPE_VERSION_FIVE => Ok(Self::Five),
            value => Err(FormatError::InvalidDatatypeMessageVersion(value)),
        }
    }
}

/// A floating-point byte order paired with a compatible datatype message version.
///
/// Little-endian and big-endian order support versions 1 through 5. VAX order requires
/// version 3 or later. The byte-order table in "The Datatype Message" of the [format
/// specification, version 4.0][spec] defines bits 6 and 0 together. Its generic reserved-bit
/// row also lists bit 6, which the byte-order table uses for VAX.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatingPointByteOrder {
    version: DatatypeVersion,
    order: DatatypeByteOrder,
}

impl FloatingPointByteOrder {
    /// Pairs `order` with a compatible message version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] for VAX order before version 3.
    pub fn new(version: DatatypeVersion, order: DatatypeByteOrder) -> Result<Self, FormatError> {
        if order == DatatypeByteOrder::Vax {
            version.for_class(FLOATING_POINT_CLASS, DatatypeVersion::Three)?;
        }
        Ok(Self { version, order })
    }

    /// Parses the byte order and reserved bits from a floating-point class bit field.
    ///
    /// Bits 6 and 0 select little-endian (`0x00`), big-endian (`0x01`), or VAX (`0x41`) order.
    /// The parser checks reserved bits 7 and 16 through 23. Padding, normalization, and sign
    /// location are separate properties.
    ///
    /// The parser rejects bit 6 in versions 1 and 2. `H5O__dtype_decode_helper` ignores that bit
    /// in those versions (`H5Odtype.c`, HDF5 2.0.0). `H5O__dtype_encode_helper` writes VAX as
    /// `0x41` with a minimum version of 3 in the same release.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] if a reserved class bit is set,
    /// [`FormatError::InvalidByteOrder`] for the reserved order pattern `0x40`, and
    /// [`FormatError::InvalidDatatypeVersion`] for VAX order before version 3.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeByteOrder, FormatError};
    /// use hdf5_pure_format::__private::{DatatypeVersion, FloatingPointByteOrder};
    ///
    /// let order = FloatingPointByteOrder::parse(DatatypeVersion::Three, [0x41, 31, 0])?;
    /// assert_eq!(order.order(), DatatypeByteOrder::Vax);
    /// assert_eq!(order.bits(), 0x41);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn parse(version: DatatypeVersion, flags: [u8; 3]) -> Result<Self, FormatError> {
        let bits = u32::from(flags[0]) | (u32::from(flags[1]) << 8) | (u32::from(flags[2]) << 16);
        if bits & FLOAT_RESERVED_MASK != 0 {
            return Err(FormatError::InvalidDatatypeFlags {
                class: FLOATING_POINT_CLASS,
                version: version.get(),
                flags: bits,
            });
        }
        let order = flags[0] & (FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT);
        let order = match order {
            0 => DatatypeByteOrder::LittleEndian,
            FLOAT_BYTE_ORDER_LOW_BIT => DatatypeByteOrder::BigEndian,
            FLOAT_BYTE_ORDER_HIGH_BIT => return Err(FormatError::InvalidByteOrder(order)),
            _ => DatatypeByteOrder::Vax,
        };
        Self::new(version, order)
    }

    /// Returns the message version associated with this byte order.
    pub const fn version(self) -> DatatypeVersion {
        self.version
    }

    /// Returns the decoded byte order.
    pub const fn order(self) -> DatatypeByteOrder {
        self.order
    }

    /// Returns bits 6 and 0 for this byte order, with all other bits clear.
    pub const fn bits(self) -> u8 {
        Self::bits_for(self.order)
    }

    pub(super) const fn bits_for(order: DatatypeByteOrder) -> u8 {
        match order {
            DatatypeByteOrder::LittleEndian => 0,
            DatatypeByteOrder::BigEndian => FLOAT_BYTE_ORDER_LOW_BIT,
            DatatypeByteOrder::Vax => FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT,
        }
    }
}

/// A datatype description with checked properties and a retained version at each node.
///
/// A containing description's version is at least its children's versions. Compound and
/// enumeration descriptions use the property layouts of versions 1 through 5. Array
/// descriptions use the layouts of versions 3 through 5.
/// Complex descriptions use version 5 and retain the floating-point component's version.
/// These layouts are defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec]. Reference descriptions and version 2 arrays are outside this
/// representation.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub struct DatatypeMessage {
    version: DatatypeVersion,
    properties: DatatypeProperties,
    /// The offset widths inferred from an enumeration's decoded base. [`Self::to_datatype`]
    /// records them in the semantic enumeration, and [`Self::stored_size`] checks
    /// them against the destination width.
    enumeration_base_offsets: DatatypeOffsets,
}

impl DatatypeMessage {
    /// Constructs a description with compatible versions and class properties.
    ///
    /// A compound must have at least one member, with unique names. Member ranges must be
    /// disjoint and fit within the declared size. For a version 1 inline array, the range
    /// covers all array elements. Construction measures variable-length descriptors at
    /// two-byte offsets and uses the retained value width for decoded enumerations.
    /// An enumeration must have at least one member, with nonempty, NUL-free, unique names
    /// and distinct raw values of the integer parent's width. Array dimensions and element
    /// sizes must fit their stored fields. Opaque tags may contain
    /// up to 248 bytes and must not contain NUL bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] if the class or floating-point order
    /// requires a higher version or if [`DatatypeProperties::CompoundV1`] is used with a
    /// version other than 1. Returns [`FormatError::DatatypeChildVersion`] if a child has a
    /// higher version than its container.
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] for a zero element size and
    /// [`FormatError::OpaqueTagTooLong`] or [`FormatError::InvalidOpaqueTagByte`] for an invalid
    /// opaque tag. Returns [`FormatError::ValueTooLargeForRepresentation`] if a member count or
    /// array rank exceeds its field.
    ///
    /// Returns [`FormatError::InvalidCompoundMemberCount`] for a compound without members,
    /// [`FormatError::DuplicateCompoundField`] for duplicate compound names, and
    /// [`FormatError::CompoundFieldOutOfBounds`] for a compound member outside its
    /// element. Returns [`FormatError::CompoundFieldOverlap`] for overlapping compound members
    /// and [`FormatError::CompoundArraySizeOverflow`] if an inline array's size exceeds
    /// [`u32::MAX`].
    /// Returns [`FormatError::EnumBaseNotInteger`] for a non-integer enumeration base and
    /// [`FormatError::EnumMemberValueSize`] for a value whose width differs from its base type.
    /// Returns [`FormatError::InvalidEnumerationMemberCount`] for an enumeration without members,
    /// [`FormatError::EmptyEnumMemberName`] for an empty name,
    /// [`FormatError::DuplicateEnumMemberName`] for duplicate names, and
    /// [`FormatError::DuplicateEnumMemberValue`] for duplicate raw values.
    /// Returns [`FormatError::LengthTooLarge`] if an array size exceeds its four-byte field.
    /// Returns [`FormatError::InvalidComplexParent`] if a complex component is not floating
    /// point, and [`FormatError::ComplexSizeOverflow`] if twice its size exceeds [`u32::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{DatatypeMessage, DatatypeProperties, DatatypeVersion};
    ///
    /// let integer = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FixedPoint(FixedPointType::i16()),
    /// )?;
    /// let sequence = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::VariableLengthSequence { base_type: Box::new(integer) },
    /// )?;
    /// assert_eq!(sequence.version(), DatatypeVersion::Five);
    /// let DatatypeProperties::VariableLengthSequence { base_type } = sequence.properties() else {
    ///     unreachable!();
    /// };
    /// assert_eq!(base_type.version(), DatatypeVersion::Two);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn new(
        version: DatatypeVersion,
        properties: DatatypeProperties,
    ) -> Result<Self, FormatError> {
        let class = properties.class();
        let minimum = match &properties {
            DatatypeProperties::CompoundV1 { .. } if version != DatatypeVersion::One => {
                return Err(FormatError::InvalidDatatypeVersion {
                    class,
                    version: version.get(),
                });
            }
            DatatypeProperties::Compound { .. } => DatatypeVersion::Two,
            DatatypeProperties::Array { .. } => DatatypeVersion::Three,
            DatatypeProperties::Complex { .. } => DatatypeVersion::Five,
            _ => DatatypeVersion::One,
        };
        version.for_class(class, minimum)?;
        match &properties {
            DatatypeProperties::FloatingPoint(datatype) => {
                FloatingPointByteOrder::new(version, datatype.byte_order())?;
            }
            DatatypeProperties::String { size, .. } => {
                positive_size(class, *size)?;
            }
            DatatypeProperties::Opaque { size, tag } => {
                positive_size(class, *size)?;
                if tag.len() > MAX_OPAQUE_TAG_LEN {
                    return Err(FormatError::OpaqueTagTooLong { length: tag.len() });
                }
                if let Some(offset) = tag.iter().position(|&byte| byte == 0) {
                    return Err(FormatError::InvalidOpaqueTagByte { offset, byte: 0 });
                }
            }
            DatatypeProperties::Compound { size, members } => {
                check_compound_members(version, *size, members.len(), |index| {
                    (&members[index], None)
                })?;
            }
            DatatypeProperties::CompoundV1 { size, members } => {
                check_compound_members(version, *size, members.len(), |index| {
                    let member = &members[index];
                    (&member.member, Some(&member.shape))
                })?;
            }
            DatatypeProperties::Enumeration { base_type, members } => {
                version.check_child(class, base_type.version())?;
                let DatatypeProperties::FixedPoint(base) = base_type.properties else {
                    return Err(FormatError::EnumBaseNotInteger);
                };
                check_enumeration_members(members, base.size().get())?;
            }
            DatatypeProperties::VariableLengthSequence { base_type }
            | DatatypeProperties::VariableLengthString { base_type, .. } => {
                version.check_child(class, base_type.version())?;
            }
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => {
                version.check_child(class, base_type.version())?;
                dimensions.len().narrow_or_else::<u8, _>(|| {
                    FormatError::ValueTooLargeForRepresentation {
                        value: dimensions.len().to_u64(),
                        target: "u8",
                    }
                })?;
                positive_size(
                    class,
                    __private::array_type_size(base_type.size_with_offsets(None)?, dimensions)?,
                )?;
            }
            DatatypeProperties::Complex { component } => {
                version.check_child(class, component.version())?;
                let DatatypeProperties::FloatingPoint(component) = component.properties else {
                    return Err(FormatError::InvalidComplexParent {
                        class: component.properties.class(),
                    });
                };
                ComplexType::new(component)?;
            }
            _ => {}
        }
        Ok(Self {
            version,
            properties,
            enumeration_base_offsets: DatatypeOffsets::default(),
        })
    }

    pub(super) fn from_parsed(parsed: ParsedProperties) -> Self {
        let (version, properties, enumeration_base_offsets) = parsed.into_parts();
        Self {
            version,
            properties,
            enumeration_base_offsets,
        }
    }

    /// Returns this description's message version.
    pub const fn version(&self) -> DatatypeVersion {
        self.version
    }

    /// Returns the class properties, including nested descriptions.
    pub const fn properties(&self) -> &DatatypeProperties {
        &self.properties
    }

    /// Returns one element's stored byte size at `offset_width`.
    ///
    /// Variable-length elements include a heap address of the selected width. Array sizes
    /// include their base type's stored size at that width. An enumeration keeps its encoded
    /// value width, which must match the base type's size at that width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte size
    /// field. Propagates errors from nested stored-size calculations and [`ComplexType::new`].
    /// Returns [`FormatError::EnumBaseSizeMismatch`] if the selected width changes the stored
    /// size of an enumeration's base type, or [`FormatError::EnumBaseOffsetWidthMismatch`] if
    /// a stored descriptor in the base uses a different offset width.
    /// Returns [`FormatError::Internal`] if a complex component is not floating point.
    pub fn stored_size(&self, offset_width: OffsetWidth) -> Result<u32, FormatError> {
        self.size_with_offsets(Some(offset_width))
    }

    pub(super) const fn enumeration_base_offsets(&self) -> DatatypeOffsets {
        self.enumeration_base_offsets
    }

    fn size_with_offsets(&self, destination: Option<OffsetWidth>) -> Result<u32, FormatError> {
        // Construction uses the minimum descriptor size and keeps retained enum widths.
        match &self.properties {
            DatatypeProperties::VariableLengthSequence { .. }
            | DatatypeProperties::VariableLengthString { .. } => Ok(
                __private::variable_length_descriptor_size(destination.unwrap_or(OffsetWidth::Two)),
            ),
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => __private::array_type_size(base_type.size_with_offsets(destination)?, dimensions),
            DatatypeProperties::FixedPoint(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::FloatingPoint(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::Time(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::BitField(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::String { size, .. }
            | DatatypeProperties::Opaque { size, .. }
            | DatatypeProperties::Compound { size, .. }
            | DatatypeProperties::CompoundV1 { size, .. } => Ok(*size),
            DatatypeProperties::Enumeration { base_type, members } => {
                let size = enumeration_value_width(members);
                if let Some(width) = destination {
                    check_enum_base_layout(
                        size,
                        base_type.stored_size(width)?,
                        self.enumeration_base_offsets,
                        width,
                    )?;
                }
                Ok(size)
            }
            DatatypeProperties::Complex { component } => {
                let DatatypeProperties::FloatingPoint(component) = component.properties else {
                    return Err(FormatError::Internal(
                        "a checked complex component is not floating point".into(),
                    ));
                };
                Ok(ComplexType::new(component)?.size().get())
            }
        }
    }

    /// Converts this description to semantic metadata without message versions.
    ///
    /// Represents each version 1 inline array member as a [`Datatype::Array`] with the
    /// active dimensions and the projected member type as its base.
    ///
    /// # Errors
    ///
    /// Returns errors from nested conversions, [`VariableLengthStringType::with_base_type`],
    /// and [`ComplexType::new`].
    /// Returns [`FormatError::Internal`] if a complex component is not floating point.
    pub fn to_datatype(&self) -> Result<Datatype, FormatError> {
        Ok(match &self.properties {
            DatatypeProperties::FixedPoint(datatype) => Datatype::FixedPoint(*datatype),
            DatatypeProperties::FloatingPoint(datatype) => Datatype::FloatingPoint(*datatype),
            DatatypeProperties::Time(datatype) => Datatype::Time(*datatype),
            DatatypeProperties::String {
                size,
                padding,
                charset,
            } => Datatype::String {
                size: *size,
                padding: padding.clone(),
                charset: charset.clone(),
            },
            DatatypeProperties::BitField(datatype) => Datatype::BitField(*datatype),
            DatatypeProperties::Opaque { size, tag } => Datatype::Opaque {
                size: *size,
                tag: tag.clone(),
            },
            DatatypeProperties::Compound { size, members } => Datatype::Compound {
                size: *size,
                members: members
                    .iter()
                    .map(|member| member.to_datatype(None))
                    .collect::<Result<_, FormatError>>()?,
            },
            DatatypeProperties::CompoundV1 { size, members } => Datatype::Compound {
                size: *size,
                members: members
                    .iter()
                    .map(|member| member.member.to_datatype(Some(member.shape)))
                    .collect::<Result<_, FormatError>>()?,
            },
            DatatypeProperties::Enumeration { base_type, members } => {
                let size = positive_size(ENUMERATION_CLASS, enumeration_value_width(members))?;
                Datatype::Enumeration(__private::enumeration_from_stored_parts(
                    base_type.to_datatype()?,
                    size,
                    self.enumeration_base_offsets,
                    members.clone(),
                )?)
            }
            DatatypeProperties::VariableLengthSequence { base_type } => {
                Datatype::VariableLength(VariableLengthType::sequence(base_type.to_datatype()?))
            }
            DatatypeProperties::VariableLengthString {
                padding,
                charset,
                base_type,
            } => Datatype::VariableLength(VariableLengthType::String(
                VariableLengthStringType::with_base_type(
                    padding.clone(),
                    charset.clone(),
                    base_type.to_datatype()?,
                )?,
            )),
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => Datatype::Array {
                base_type: Box::new(base_type.to_datatype()?),
                dimensions: dimensions.clone(),
            },
            DatatypeProperties::Complex { component } => {
                let DatatypeProperties::FloatingPoint(component) = component.properties else {
                    return Err(FormatError::Internal(
                        "a checked complex component is not floating point".into(),
                    ));
                };
                Datatype::Complex(ComplexType::new(component)?)
            }
        })
    }
}

/// Class properties supplied to [`DatatypeMessage::new`].
///
/// Each nested description has its own checked message version. The containing description's
/// version and property constraints are checked by [`DatatypeMessage::new`].
#[derive(Clone, Debug, PartialEq)]
pub enum DatatypeProperties {
    /// Fixed-point numeric layout and byte order.
    FixedPoint(FixedPointType),
    /// Floating-point numeric layout and byte order.
    FloatingPoint(FloatingPointType),
    /// Time precision and byte order.
    Time(TimeType),
    /// Fixed-length string metadata.
    String {
        /// The width of one string in bytes.
        size: u32,
        /// The treatment of unused bytes.
        padding: StringPadding,
        /// The encoding of the string bytes.
        charset: CharacterSet,
    },
    /// Bitfield precision, padding, and byte order.
    BitField(BitFieldType),
    /// Opaque data with a logical tag.
    Opaque {
        /// The width of one element in bytes.
        size: u32,
        /// The logical tag bytes, excluding terminator and padding.
        tag: Vec<u8>,
    },
    /// Compound members in versions 2 through 5.
    Compound {
        /// The width of one compound element in bytes.
        size: u32,
        /// The named members in declaration order.
        members: Vec<DatatypeMessageMember>,
    },
    /// Compound members with the version 1 dimension block.
    CompoundV1 {
        /// The width of one compound element in bytes.
        size: u32,
        /// The members in declaration order, with their stored dimension fields.
        members: Vec<CompoundV1Member>,
    },
    /// Enumeration members in stored order.
    Enumeration {
        /// The base description, including its message version.
        base_type: Box<DatatypeMessage>,
        /// The names and raw values of the members.
        members: Vec<EnumMember>,
    },
    /// A variable-length sequence.
    VariableLengthSequence {
        /// The description of one sequence element.
        base_type: Box<DatatypeMessage>,
    },
    /// A variable-length string with a retained parent description.
    VariableLengthString {
        /// The treatment of unused string bytes.
        padding: StringPadding,
        /// The encoding of the string bytes.
        charset: CharacterSet,
        /// The parent description stored with the string metadata.
        base_type: Box<DatatypeMessage>,
    },
    /// An array using the property layout of version 3 or later.
    Array {
        /// The description of one array element.
        base_type: Box<DatatypeMessage>,
        /// The extents in stored dimension order.
        dimensions: Vec<u32>,
    },
    /// A homogeneous rectangular complex description.
    Complex {
        /// The floating-point component description, including its message version.
        component: Box<DatatypeMessage>,
    },
}

pub(super) fn check_enumeration_members(
    members: &[EnumMember],
    size: u32,
) -> Result<(), FormatError> {
    enumeration_member_count(members.len())?;
    let mut names = BTreeSet::new();
    let mut values = BTreeMap::new();
    for (index, member) in members.iter().enumerate() {
        let (name, value) = __private::enum_member_parts(member);
        if name.as_bytes().is_empty() {
            return Err(FormatError::EmptyEnumMemberName { index });
        }
        if value.len().to_u64() != u64::from(size) {
            return Err(FormatError::EnumMemberValueSize(
                name.clone(),
                size,
                value.len(),
            ));
        }
        if !names.insert(name) {
            return Err(FormatError::DuplicateEnumMemberName(name.clone()));
        }
        if let Some(first) = values.insert(value, name) {
            return Err(FormatError::DuplicateEnumMemberValue {
                first: first.clone(),
                second: name.clone(),
            });
        }
    }
    Ok(())
}

#[expect(
    clippy::expect_used,
    reason = "the retained decoder and constructor check every value's u32 width"
)]
pub(super) fn enumeration_value_width(members: &[EnumMember]) -> u32 {
    // Retained enumerations have at least one member and each value has the checked u32 width.
    u32::try_from(members[0].value.len()).expect("the stored enumeration value fits in u32")
}

impl DatatypeProperties {
    fn class(&self) -> u8 {
        match self {
            Self::FixedPoint(_) => FIXED_POINT_CLASS,
            Self::FloatingPoint(_) => FLOATING_POINT_CLASS,
            Self::Time(_) => TIME_CLASS,
            Self::String { .. } => STRING_CLASS,
            Self::BitField(_) => BIT_FIELD_CLASS,
            Self::Opaque { .. } => OPAQUE_CLASS,
            Self::Compound { .. } | Self::CompoundV1 { .. } => COMPOUND_CLASS,
            Self::Enumeration { .. } => ENUMERATION_CLASS,
            Self::VariableLengthSequence { .. } | Self::VariableLengthString { .. } => {
                VARIABLE_LENGTH_CLASS
            }
            Self::Array { .. } => ARRAY_CLASS,
            Self::Complex { .. } => COMPLEX_CLASS,
        }
    }
}

/// A compound member's name, byte offset, and versioned datatype description.
#[derive(Clone, Debug, PartialEq)]
pub struct DatatypeMessageMember {
    name: DatatypeMemberName,
    byte_offset: u64,
    datatype: DatatypeMessage,
}

impl DatatypeMessageMember {
    /// Constructs a member with a NUL-free name.
    ///
    /// The member's range and message version are checked when constructing a containing
    /// description with [`DatatypeMessage::new`].
    pub const fn new(
        name: DatatypeMemberName,
        byte_offset: u64,
        datatype: DatatypeMessage,
    ) -> Self {
        Self {
            name,
            byte_offset,
            datatype,
        }
    }

    /// Returns the member name without its wire terminator.
    pub const fn name(&self) -> &DatatypeMemberName {
        &self.name
    }

    /// Returns the byte offset from the start of the compound element.
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Returns the member description, including its message version.
    pub const fn datatype(&self) -> &DatatypeMessage {
        &self.datatype
    }

    fn to_datatype(&self, shape: Option<CompoundV1Shape>) -> Result<CompoundMember, FormatError> {
        let datatype = self.datatype.to_datatype()?;
        let datatype = match shape {
            Some(shape) => shape.into_datatype(datatype),
            None => datatype,
        };
        Ok(CompoundMember::new(
            self.name.clone(),
            self.byte_offset,
            datatype,
        ))
    }

    /// Records the member name among the names in this compound and checks the member version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DuplicateCompoundField`] if `names` already contains the name,
    /// or [`FormatError::DatatypeChildVersion`] if the member's version exceeds `version`.
    pub(super) fn check_compound_name_and_version<'a>(
        &'a self,
        version: DatatypeVersion,
        names: &mut BTreeSet<&'a DatatypeMemberName>,
    ) -> Result<(), FormatError> {
        if !names.insert(&self.name) {
            return Err(FormatError::DuplicateCompoundField(self.name.clone()));
        }
        version.check_child(COMPOUND_CLASS, self.datatype.version())
    }
}

/// A version 1 compound member with its stored dimension fields.
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundV1Member {
    member: DatatypeMessageMember,
    shape: CompoundV1Shape,
}

impl CompoundV1Member {
    /// Pairs a member description with the dimensions stored before it.
    ///
    /// For an inline array, pass the array's base datatype as the member description.
    /// [`DatatypeMessage::new`] checks the full member range when constructing the compound.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{Datatype, FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     CompoundV1Member, CompoundV1Shape, DatatypeMessage, DatatypeMessageMember,
    ///     DatatypeProperties, DatatypeVersion,
    /// };
    ///
    /// let byte = DatatypeMessage::new(
    ///     DatatypeVersion::One,
    ///     DatatypeProperties::FixedPoint(FixedPointType::u8()),
    /// )?;
    /// let shape = CompoundV1Shape::new(1, [3, 0, 0, 0])?;
    /// let member = CompoundV1Member::new(DatatypeMessageMember::new(c"x".into(), 0, byte), shape);
    /// let compound = DatatypeMessage::new(
    ///     DatatypeVersion::One,
    ///     DatatypeProperties::CompoundV1 { size: 3, members: vec![member] },
    /// )?;
    /// let Datatype::Compound { members, .. } = compound.to_datatype()? else {
    ///     unreachable!();
    /// };
    /// assert_eq!(
    ///     members[0].datatype,
    ///     Datatype::Array {
    ///         base_type: Box::new(Datatype::FixedPoint(FixedPointType::u8())),
    ///         dimensions: vec![3],
    ///     },
    /// );
    /// # Ok::<(), FormatError>(())
    /// ```
    pub const fn new(member: DatatypeMessageMember, shape: CompoundV1Shape) -> Self {
        Self { member, shape }
    }

    /// Borrows the member name, byte offset, and stored datatype description.
    pub const fn member(&self) -> &DatatypeMessageMember {
        &self.member
    }

    /// Returns the rank and all four extent slots from the dimension block.
    pub const fn shape(&self) -> CompoundV1Shape {
        self.shape
    }
}

fn check_compound_members<'a>(
    version: DatatypeVersion,
    size: u32,
    count: usize,
    member_at: impl Fn(usize) -> (&'a DatatypeMessageMember, Option<&'a CompoundV1Shape>),
) -> Result<(), FormatError> {
    positive_size(COMPOUND_CLASS, size)?;
    compound_member_count(count)?;
    let mut names = BTreeSet::new();
    let mut layout = CompoundLayout::new(size);
    for index in 0..count {
        let (member, shape) = member_at(index);
        member.check_compound_name_and_version(version, &mut names)?;
        let datatype = &member.datatype;
        let field_size = positive_size(
            datatype.properties.class(),
            datatype.size_with_offsets(None)?,
        )?;
        let field_size = match shape {
            Some(shape) => shape.stored_size(field_size)?,
            None => field_size,
        };
        layout.insert(member.byte_offset, field_size, index, |index| {
            Some(member_at(index).0.name.clone())
        })?;
    }
    Ok(())
}

/// Converts a datatype element size to a positive byte count.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero.
pub(in crate::datatype) fn positive_size(class: u8, size: u32) -> Result<NonZeroU32, FormatError> {
    NonZeroU32::new(size).ok_or(FormatError::ZeroSizedDatatype { class })
}

/// Checks a declared enumeration size against its base datatype's stored size.
///
/// # Errors
///
/// Returns [`FormatError::EnumBaseSizeMismatch`] if the sizes differ.
pub(super) fn check_enum_base_size(size: u32, base_size: u32) -> Result<(), FormatError> {
    // `H5O__dtype_decode_helper` in `H5Odtype.c`, HDF5 1.14.6, checks that the sizes agree.
    if size != base_size {
        return Err(FormatError::EnumBaseSizeMismatch { size, base_size });
    }
    Ok(())
}

/// Checks the base's size at the destination and its stored descriptor widths.
///
/// # Errors
///
/// Returns [`FormatError::EnumBaseSizeMismatch`] if the sizes differ, then checks the
/// descriptor widths with [`DatatypeOffsets::check_destination`].
pub(super) fn check_enum_base_layout(
    size: u32,
    base_size: u32,
    offsets: DatatypeOffsets,
    destination: OffsetWidth,
) -> Result<(), FormatError> {
    check_enum_base_size(size, base_size)?;
    offsets.check_destination(destination)
}

/// Converts a compound member count to a positive two-byte count.
///
/// `H5O__dtype_decode_helper` in `H5Odtype.c`, HDF5 2.2.0, rejects zero compound members.
///
/// # Errors
///
/// Returns [`FormatError::InvalidCompoundMemberCount`] if `count` is zero, and
/// [`FormatError::ValueTooLargeForRepresentation`] if it exceeds [`u16::MAX`].
pub(super) fn compound_member_count(count: usize) -> Result<NonZeroU16, FormatError> {
    NonZeroU16::new(member_count(count)?).ok_or(FormatError::InvalidCompoundMemberCount { count })
}

/// Converts an enumeration member count to a positive two-byte count.
///
/// The count field is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec]. `H5T_is_sensible` in `H5T.c`, HDF5 2.2.0, returns false for an
/// enumeration without members.
///
/// # Errors
///
/// Returns [`FormatError::InvalidEnumerationMemberCount`] if `count` is zero, and
/// [`FormatError::ValueTooLargeForRepresentation`] if it exceeds [`u16::MAX`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn enumeration_member_count(count: usize) -> Result<NonZeroU16, FormatError> {
    NonZeroU16::new(member_count(count)?)
        .ok_or(FormatError::InvalidEnumerationMemberCount { count })
}

/// Converts a member count to the two-byte class field.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForRepresentation`] if `count` exceeds [`u16::MAX`].
pub(super) fn member_count(count: usize) -> Result<u16, FormatError> {
    count.narrow_or_else(|| FormatError::ValueTooLargeForRepresentation {
        value: count.to_u64(),
        target: "u16",
    })
}

/// The name field layout for compound and enumeration members.
///
/// libhdf5 omits name padding from version 3 onward (`H5O__dtype_decode_helper` and
/// `H5O__dtype_encode_helper` in `H5Odtype.c`, HDF5 2.2.0).
#[derive(Clone, Copy)]
pub(super) enum NameLayout {
    Padded,
    Unpadded,
}

impl NameLayout {
    pub(super) fn for_version(version: DatatypeVersion) -> Self {
        match version {
            DatatypeVersion::One | DatatypeVersion::Two => Self::Padded,
            DatatypeVersion::Three | DatatypeVersion::Four | DatatypeVersion::Five => {
                Self::Unpadded
            }
        }
    }

    /// Returns the field length, including the terminator and any padding.
    pub(super) fn field_len(self, name_len: usize) -> usize {
        name_len + self.suffix_len(name_len)
    }

    /// Returns the number of bytes for the terminator and any padding.
    pub(super) fn suffix_len(self, name_len: usize) -> usize {
        match self {
            Self::Padded => NAME_ALIGNMENT - name_len % NAME_ALIGNMENT,
            Self::Unpadded => 1,
        }
    }
}

// "The Datatype Message" in format specification version 4.0 defines this alignment in
// bytes for terminated compound and enumeration names in versions 1 and 2.
pub(super) const NAME_ALIGNMENT: usize = 8;

/// The largest aligned opaque tag property length representable in its one-byte field.
///
/// The length and alignment for datatype message versions 1 through 5 are defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const MAX_OPAQUE_TAG_LEN: usize = (u8::MAX - u8::MAX % OPAQUE_TAG_ALIGNMENT) as usize;

/// The mask for reserved bits 7 and 16 through 23 of a floating-point class bit field.
///
/// The bits for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_RESERVED_MASK: u32 = 0xFF_0080;

// The floating-point byte-order table in "The Datatype Message" of the format
// specification, version 4.0, defines these bits together. See `FloatingPointByteOrder`
// for how the table's use of bit 6 relates to the reserved-bit row.

/// Bit 0 of a floating-point class bit field, the low bit of its byte order.
pub(in crate::datatype) const FLOAT_BYTE_ORDER_LOW_BIT: u8 = 0x01;

/// Bit 6 of a floating-point class bit field, the high bit of its byte order.
pub(in crate::datatype) const FLOAT_BYTE_ORDER_HIGH_BIT: u8 = 0x40;
