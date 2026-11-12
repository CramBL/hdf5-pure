//! Datatype descriptions with checked message versions and recursive properties.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;
use core::num::NonZeroU16;
use core::num::NonZeroU32;

use hdf5_pure_core::__private;
use hdf5_pure_core::{
    BitFieldType, CharacterSet, ComplexType, Datatype, DatatypeByteOrder, EnumMember,
    FixedPointType, FloatingPointType, FormatError, StringPadding, TimeType,
    VariableLengthStringType, VariableLengthType,
};

use self::compound_layout::CompoundLayout;
use super::{
    ARRAY_CLASS, BIT_FIELD_CLASS, COMPLEX_CLASS, COMPOUND_CLASS, DATATYPE_VERSION_FIVE,
    DATATYPE_VERSION_FOUR, DATATYPE_VERSION_ONE, DATATYPE_VERSION_THREE, DATATYPE_VERSION_TWO,
    ENUMERATION_CLASS, FIXED_POINT_CLASS, FLOAT_BYTE_ORDER_HIGH_BIT, FLOAT_BYTE_ORDER_LOW_BIT,
    FLOAT_RESERVED_MASK, FLOATING_POINT_CLASS, OPAQUE_CLASS, STRING_CLASS, TIME_CLASS,
    VARIABLE_LENGTH_CLASS,
};
use crate::convert::Narrow;
use crate::width::OffsetWidth;

pub(super) use self::atomic::{AtomicMessage, AtomicProperties};

mod atomic;
mod codec;
mod compound_layout;

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
    fn for_class(self, class: u8, minimum: Self) -> Result<Self, FormatError> {
        if self < minimum {
            return Err(FormatError::InvalidDatatypeVersion {
                class,
                version: self.get(),
            });
        }
        Ok(self)
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
        match self.order {
            DatatypeByteOrder::LittleEndian => 0,
            DatatypeByteOrder::BigEndian => FLOAT_BYTE_ORDER_LOW_BIT,
            DatatypeByteOrder::Vax => FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT,
        }
    }
}

/// A datatype description with checked properties and a retained version at each node.
///
/// A containing description's version is at least its children's versions. Compound,
/// enumeration, and array descriptions use the property layouts of versions 3 through 5.
/// Complex descriptions use version 5 and retain the floating-point component's version.
/// These layouts are defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec]. Reference descriptions and legacy container layouts are outside this
/// representation.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub struct DatatypeMessage {
    version: DatatypeVersion,
    properties: DatatypeProperties,
}

impl DatatypeMessage {
    /// Constructs a description with compatible versions and class properties.
    ///
    /// A compound must have at least one member, with unique names. Member ranges must be
    /// disjoint and fit at the smallest supported offset width of two bytes.
    /// An enumeration must have at least one member, with nonempty, NUL-free, unique names
    /// and distinct raw values of the integer parent's width. Array dimensions and element
    /// sizes must fit their stored fields. Opaque tags may contain
    /// up to 248 bytes and must not contain NUL bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] if the class or floating-point order
    /// requires a higher version, and [`FormatError::DatatypeChildVersion`] if a child has a
    /// higher version than its container.
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] for a zero element size,
    /// [`FormatError::OpaqueTagTooLong`] or [`FormatError::InvalidOpaqueTagByte`] for an invalid
    /// opaque tag, and [`FormatError::InvalidDatatypeMemberName`] for an enumeration name with
    /// a NUL byte. Returns [`FormatError::ValueTooLargeForRepresentation`] if a member count
    /// or array rank exceeds its field.
    ///
    /// Returns [`FormatError::InvalidCompoundMemberCount`] for a compound without members,
    /// [`FormatError::DuplicateCompoundField`] for duplicate compound names, and
    /// [`FormatError::CompoundFieldOutOfBounds`] for a compound member outside its
    /// element. Returns [`FormatError::CompoundFieldOverlap`] for overlapping compound members,
    /// [`FormatError::EnumBaseNotInteger`] for a non-integer enumeration base, and
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
            DatatypeProperties::Compound { .. }
            | DatatypeProperties::Enumeration { .. }
            | DatatypeProperties::Array { .. } => DatatypeVersion::Three,
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
                positive_size(class, *size)?;
                compound_member_count(members.len())?;
                let mut names = BTreeSet::new();
                let mut layout = CompoundLayout::new(*size);
                for (
                    index,
                    DatatypeMessageMember {
                        name,
                        byte_offset: _,
                        datatype,
                    },
                ) in members.iter().enumerate()
                {
                    if !names.insert(name.as_str()) {
                        return Err(FormatError::DuplicateCompoundField(name.clone()));
                    }
                    compatible_child(version, class, datatype)?;
                    let field_size = datatype.stored_size(OffsetWidth::Two)?;
                    layout.insert(members, index, field_size)?;
                }
            }
            DatatypeProperties::Enumeration { base_type, members } => {
                compatible_child(version, class, base_type)?;
                if !matches!(base_type.properties, DatatypeProperties::FixedPoint(_)) {
                    return Err(FormatError::EnumBaseNotInteger);
                }
                enumeration_member_count(members.len())?;
                let size = base_type.stored_size(OffsetWidth::Two)?;
                let mut names = BTreeSet::new();
                let mut values = BTreeMap::new();
                for (index, member) in members.iter().enumerate() {
                    let (name, value) = __private::enum_member_parts(member);
                    if name.is_empty() {
                        return Err(FormatError::EmptyEnumMemberName { index });
                    }
                    member_name(class, name)?;
                    if value.len().to_u64() != u64::from(size) {
                        return Err(FormatError::EnumMemberValueSize(
                            name.into(),
                            size,
                            value.len(),
                        ));
                    }
                    if !names.insert(name) {
                        return Err(FormatError::DuplicateEnumMemberName(name.into()));
                    }
                    if let Some(first) = values.insert(value, name) {
                        return Err(FormatError::DuplicateEnumMemberValue {
                            first: first.into(),
                            second: name.into(),
                        });
                    }
                }
            }
            DatatypeProperties::VariableLengthSequence { base_type }
            | DatatypeProperties::VariableLengthString { base_type, .. } => {
                compatible_child(version, class, base_type)?;
            }
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => {
                compatible_child(version, class, base_type)?;
                dimensions.len().narrow_or_else::<u8, _>(|| {
                    FormatError::ValueTooLargeForRepresentation {
                        value: dimensions.len().to_u64(),
                        target: "u8",
                    }
                })?;
                positive_size(
                    class,
                    super::array_type_size(base_type.stored_size(OffsetWidth::Two)?, dimensions)?,
                )?;
            }
            DatatypeProperties::Complex { component } => {
                compatible_child(version, class, component)?;
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
        })
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
    /// include their base type's stored size at that width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte size
    /// field. Propagates errors from nested stored-size calculations and [`ComplexType::new`].
    /// Returns [`FormatError::Internal`] if a complex component is not floating point.
    pub fn stored_size(&self, offset_width: OffsetWidth) -> Result<u32, FormatError> {
        match &self.properties {
            DatatypeProperties::VariableLengthSequence { .. }
            | DatatypeProperties::VariableLengthString { .. } => Ok(super::SEQUENCE_LENGTH_SIZE
                + u32::from(offset_width.get())
                + super::HEAP_INDEX_SIZE),
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => super::array_type_size(base_type.stored_size(offset_width)?, dimensions),
            DatatypeProperties::FixedPoint(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::FloatingPoint(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::Time(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::BitField(datatype) => Ok(datatype.size().get()),
            DatatypeProperties::String { size, .. }
            | DatatypeProperties::Opaque { size, .. }
            | DatatypeProperties::Compound { size, .. } => Ok(*size),
            DatatypeProperties::Enumeration { base_type, .. } => {
                base_type.stored_size(offset_width)
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
                    .map(|member| {
                        Ok(__private::compound_member(
                            member.name.clone(),
                            member.byte_offset,
                            member.datatype.to_datatype()?,
                        ))
                    })
                    .collect::<Result<_, FormatError>>()?,
            },
            DatatypeProperties::Enumeration { base_type, members } => Datatype::Enumeration {
                size: base_type.stored_size(OffsetWidth::Eight)?,
                base_type: Box::new(base_type.to_datatype()?),
                members: members.clone(),
            },
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
    /// Compound members using the property layout of version 3 or later.
    Compound {
        /// The width of one compound element in bytes.
        size: u32,
        /// The named members in declaration order.
        members: Vec<DatatypeMessageMember>,
    },
    /// Enumeration members using the property layout of version 3 or later.
    Enumeration {
        /// The integer base description, including its message version.
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

impl DatatypeProperties {
    fn class(&self) -> u8 {
        match self {
            Self::FixedPoint(_) => FIXED_POINT_CLASS,
            Self::FloatingPoint(_) => FLOATING_POINT_CLASS,
            Self::Time(_) => TIME_CLASS,
            Self::String { .. } => STRING_CLASS,
            Self::BitField(_) => BIT_FIELD_CLASS,
            Self::Opaque { .. } => OPAQUE_CLASS,
            Self::Compound { .. } => COMPOUND_CLASS,
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
    name: String,
    byte_offset: u64,
    datatype: DatatypeMessage,
}

impl DatatypeMessageMember {
    /// Constructs a member with a NUL-free name.
    ///
    /// The member's range and message version are checked when constructing a containing
    /// description with [`DatatypeMessage::new`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeMemberName`] if `name` contains a NUL byte.
    pub fn new(
        name: String,
        byte_offset: u64,
        datatype: DatatypeMessage,
    ) -> Result<Self, FormatError> {
        member_name(COMPOUND_CLASS, &name)?;
        Ok(Self {
            name,
            byte_offset,
            datatype,
        })
    }

    /// Returns the member name without its wire terminator.
    pub fn name(&self) -> &str {
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
}

/// Checks the child's version against its containing message's version.
///
/// # Errors
///
/// Returns [`FormatError::DatatypeChildVersion`] if the child's version is higher.
fn compatible_child(
    version: DatatypeVersion,
    class: u8,
    child: &DatatypeMessage,
) -> Result<(), FormatError> {
    if child.version > version {
        return Err(FormatError::DatatypeChildVersion {
            class,
            version: version.get(),
            child_version: child.version.get(),
        });
    }
    Ok(())
}

/// Converts a datatype element size to a positive byte count.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero.
fn positive_size(class: u8, size: u32) -> Result<NonZeroU32, FormatError> {
    NonZeroU32::new(size).ok_or(FormatError::ZeroSizedDatatype { class })
}

/// Converts a compound member count to a positive two-byte count.
///
/// `H5O__dtype_decode_helper` in `H5Odtype.c`, HDF5 2.2.0, rejects zero compound members.
///
/// # Errors
///
/// Returns [`FormatError::InvalidCompoundMemberCount`] if `count` is zero, and
/// [`FormatError::ValueTooLargeForRepresentation`] if it exceeds [`u16::MAX`].
fn compound_member_count(count: usize) -> Result<NonZeroU16, FormatError> {
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
fn member_count(count: usize) -> Result<u16, FormatError> {
    count.narrow_or_else(|| FormatError::ValueTooLargeForRepresentation {
        value: count.to_u64(),
        target: "u16",
    })
}

/// Checks a member name for NUL bytes.
///
/// # Errors
///
/// Returns [`FormatError::InvalidDatatypeMemberName`] if `name` contains a NUL byte.
fn member_name(class: u8, name: &str) -> Result<(), FormatError> {
    if name.as_bytes().contains(&0) {
        return Err(FormatError::InvalidDatatypeMemberName {
            class,
            name: name.into(),
        });
    }
    Ok(())
}

/// The largest aligned opaque tag property length representable in its one-byte field.
///
/// The length and alignment for datatype message versions 1 through 5 are defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const MAX_OPAQUE_TAG_LEN: usize = 248;
