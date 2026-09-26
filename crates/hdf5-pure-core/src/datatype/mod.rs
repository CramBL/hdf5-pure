use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU32;

use self::byte_order::DatatypeByteOrder;
use self::layout::FixedPointLayout;
use self::layout::FloatingPointLayout;
use crate::FormatError;
use crate::display;
use crate::display::DISPLAY_MAX_MEMBERS;
use crate::display::Dims;
use crate::display::EscapedName;
use crate::display::QuotedBytes;

pub mod byte_order;
pub mod layout;

/// String padding type.
#[derive(Debug, Clone, PartialEq)]
pub enum StringPadding {
    /// Terminates a shortened string with a null byte.
    NullTerminate,
    /// Pads unused bytes with null bytes.
    NullPad,
    /// Pads unused bytes with spaces.
    SpacePad,
}

/// Character set encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum CharacterSet {
    /// Uses ASCII encoding.
    Ascii,
    /// Uses UTF-8 encoding.
    Utf8,
}

/// Reference type.
///
/// The format also defines attribute references in "The Datatype Message" of
/// the [format specification, version 4.0][spec], so this enum is non-exhaustive.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReferenceType {
    /// Refers to an object in the file.
    Object,
    /// Refers to a region of a dataset.
    DatasetRegion,
}

/// A member of a compound datatype.
///
/// The struct is non-exhaustive.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset within the compound.
    pub byte_offset: u64,
    /// Member datatype.
    pub datatype: Datatype,
}

/// A member of an enumeration datatype.
///
/// The struct is non-exhaustive.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EnumMember {
    /// Member name.
    pub name: String,
    /// Raw value bytes (length = base type size).
    pub value: Vec<u8>,
}

/// Represents the class-specific fields of an HDF5 Datatype message.
///
/// [`type_size`](Self::type_size) gives the element size in bytes. A
/// [`FixedPoint`](Self::FixedPoint) or [`FloatingPoint`](Self::FloatingPoint) type wider than eight
/// bytes parses, although the numeric conversion paths do not decode its values.
///
/// The [format specification, version 4.0][spec] defines a Complex class (11), which this enum
/// does not represent.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Datatype {
    /// Class 0: Fixed-point (integer) types.
    FixedPoint {
        /// Size of one element, in bytes.
        size: u32,
        /// Order of the element bytes.
        byte_order: DatatypeByteOrder,
        /// Sign and significant bit range.
        layout: FixedPointLayout,
    },
    /// Class 1: Floating-point types.
    FloatingPoint {
        /// Size of one element, in bytes.
        size: u32,
        /// Order of the element bytes.
        byte_order: DatatypeByteOrder,
        /// Bit fields and exponent bias.
        layout: FloatingPointLayout,
    },
    /// Class 2: Time type (rarely used).
    Time {
        /// Size of one element, in bytes.
        size: u32,
        /// Order of the element bytes.
        byte_order: DatatypeByteOrder,
        /// Number of significant bits in an element.
        bit_precision: u16,
    },
    /// Class 3: Fixed-length string.
    String {
        /// Width of one string, in bytes.
        size: u32,
        /// Treatment of unused bytes.
        padding: StringPadding,
        /// Encoding of the string bytes.
        charset: CharacterSet,
    },
    /// Class 4: Bit field.
    BitField {
        /// Size of one element, in bytes.
        size: u32,
        /// Order of the element bytes.
        byte_order: DatatypeByteOrder,
        /// Position of the first significant bit.
        bit_offset: u16,
        /// Number of significant bits in an element.
        bit_precision: u16,
    },
    /// Class 5: Opaque data.
    Opaque {
        /// Size of one element, in bytes.
        size: u32,
        /// Application-defined tag bytes.
        tag: Vec<u8>,
    },
    /// Class 6: Compound type.
    Compound {
        /// Total size of one element, in bytes.
        size: u32,
        /// Named fields within an element.
        members: Vec<CompoundMember>,
    },
    /// Class 7: Reference type.
    Reference {
        /// Size of one reference, in bytes.
        size: u32,
        /// Kind of object or region referenced.
        ref_type: ReferenceType,
    },
    /// Class 8: Enumeration type.
    Enumeration {
        /// Size of one value, in bytes.
        size: u32,
        /// Integer datatype of the stored values.
        base_type: Box<Datatype>,
        /// Names and stored values of the enumeration.
        members: Vec<EnumMember>,
    },
    /// Class 9: Variable-length type.
    VariableLength {
        /// Whether the variable-length values are strings.
        is_string: bool,
        /// String padding when `is_string` is true.
        padding: Option<StringPadding>,
        /// String encoding when `is_string` is true.
        charset: Option<CharacterSet>,
        /// Datatype of one member of the variable-length value.
        base_type: Box<Datatype>,
    },
    /// Class 10: Array type.
    Array {
        /// Datatype of one array element.
        base_type: Box<Datatype>,
        /// Extent of each array dimension.
        dimensions: Vec<u32>,
    },
}

impl Datatype {
    /// Return the size in bytes of one element of this type.
    pub fn type_size(&self) -> u32 {
        match self {
            Datatype::FixedPoint { size, .. } => *size,
            Datatype::FloatingPoint { size, .. } => *size,
            Datatype::Time { size, .. } => *size,
            Datatype::String { size, .. } => *size,
            Datatype::BitField { size, .. } => *size,
            Datatype::Opaque { size, .. } => *size,
            Datatype::Compound { size, .. } => *size,
            Datatype::Reference { size, .. } => *size,
            Datatype::Enumeration { size, .. } => *size,
            Datatype::VariableLength { .. } => 16, // typically pointer + length
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                let elem_count: u32 = dimensions
                    .iter()
                    .copied()
                    .fold(1u32, |a, b| a.saturating_mul(b));
                base_type.type_size().saturating_mul(elem_count)
            }
        }
    }

    /// The class code this type encodes as, the low nibble of a datatype message's first byte.
    fn class_code(&self) -> u8 {
        match self {
            Datatype::FixedPoint { .. } => 0,
            Datatype::FloatingPoint { .. } => 1,
            Datatype::Time { .. } => 2,
            Datatype::String { .. } => 3,
            Datatype::BitField { .. } => 4,
            Datatype::Opaque { .. } => 5,
            Datatype::Compound { .. } => 6,
            Datatype::Reference { .. } => 7,
            Datatype::Enumeration { .. } => 8,
            Datatype::VariableLength { .. } => 9,
            Datatype::Array { .. } => 10,
        }
    }

    /// Returns the size in bytes of one element.
    ///
    /// The size is returned as a [`NonZeroU32`], so callers can divide by it or
    /// divide it into a byte count without a separate zero check. Prefer this
    /// over [`type_size`](Self::type_size) whenever the result feeds a division.
    ///
    /// The zero-size check is here because [`type_size`](Self::type_size) is
    /// computed: an [`Array`](Self::Array) with a zero dimension reports a zero
    /// size even though its header field is non-zero, and the public variants let a
    /// caller build a zero-sized value directly.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if the element type occupies
    /// zero bytes.
    pub fn element_size(&self) -> Result<NonZeroU32, FormatError> {
        NonZeroU32::new(self.type_size()).ok_or(FormatError::ZeroSizedDatatype {
            class: self.class_code(),
        })
    }
}

impl fmt::Display for DatatypeByteOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::LittleEndian => "le",
            Self::BigEndian => "be",
            Self::Vax => "vax",
        })
    }
}

impl fmt::Display for StringPadding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::NullTerminate => "null-term",
            Self::NullPad => "null-pad",
            Self::SpacePad => "space-pad",
        })
    }
}

impl fmt::Display for CharacterSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Ascii => "ascii",
            Self::Utf8 => "utf8",
        })
    }
}

impl fmt::Display for ReferenceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Object => "object_ref",
            Self::DatasetRegion => "region_ref",
        })
    }
}

/// The width in bits of a `size`-byte type.
///
/// Widens first: `size` is an on-disk `u32`, so a crafted size near [`u32::MAX`]
/// would overflow a `u32` multiply.
fn bit_width(size: u32) -> u64 {
    u64::from(size) * 8
}

/// The bit span, written only when it is narrower than the whole type.
fn write_bit_span(
    f: &mut fmt::Formatter<'_>,
    size: u32,
    bit_offset: u16,
    bit_precision: u16,
) -> fmt::Result {
    if bit_offset != 0 || u64::from(bit_precision) != bit_width(size) {
        let end = u64::from(bit_offset) + u64::from(bit_precision);
        write!(f, "(bits {bit_offset}..{end})")?;
    }
    Ok(())
}

/// The byte order, written only when it is not little-endian.
fn write_byte_order(f: &mut fmt::Formatter<'_>, byte_order: &DatatypeByteOrder) -> fmt::Result {
    if *byte_order != DatatypeByteOrder::LittleEndian {
        write!(f, " {byte_order}")?;
    }
    Ok(())
}

impl fmt::Display for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FixedPoint {
                size,
                byte_order,
                layout:
                    FixedPointLayout {
                        signed,
                        bit_offset,
                        bit_precision,
                    },
            } => {
                let sign = if *signed { 'i' } else { 'u' };
                write!(f, "{sign}{}", bit_width(*size))?;
                write_bit_span(f, *size, *bit_offset, *bit_precision)?;
                write_byte_order(f, byte_order)
            }
            Self::FloatingPoint {
                size,
                byte_order,
                layout,
            } => {
                write!(f, "f{}", bit_width(*size))?;
                write_bit_span(f, *size, layout.bit_offset, layout.bit_precision)?;
                write_byte_order(f, byte_order)
            }
            Self::Time {
                size,
                byte_order,
                bit_precision,
            } => {
                write!(f, "time{}", bit_width(*size))?;
                write_bit_span(f, *size, 0, *bit_precision)?;
                write_byte_order(f, byte_order)
            }
            Self::String {
                size,
                padding,
                charset,
            } => write!(f, "string[{size}] {charset} {padding}"),
            Self::BitField {
                size,
                byte_order,
                bit_offset,
                bit_precision,
            } => {
                write!(f, "bitfield{}", bit_width(*size))?;
                write_bit_span(f, *size, *bit_offset, *bit_precision)?;
                write_byte_order(f, byte_order)
            }
            Self::Opaque { size, tag } => {
                write!(f, "opaque[{size}]")?;
                if !tag.is_empty() {
                    write!(f, " {}", QuotedBytes(tag))?;
                }
                Ok(())
            }
            Self::Compound { members, .. } => {
                f.write_str("compound{")?;
                for (i, member) in members.iter().take(DISPLAY_MAX_MEMBERS).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}: {}", EscapedName(&member.name), member.datatype)?;
                }
                display::write_elided(f, members.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                f.write_str("}")
            }
            Self::Reference { ref_type, .. } => write!(f, "{ref_type}"),
            Self::Enumeration {
                base_type, members, ..
            } => {
                write!(f, "enum<{base_type}>[")?;
                for (i, member) in members.iter().take(DISPLAY_MAX_MEMBERS).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", EscapedName(&member.name))?;
                }
                display::write_elided(f, members.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                f.write_str("]")
            }
            Self::VariableLength {
                is_string,
                charset,
                base_type,
                ..
            } => {
                if *is_string {
                    f.write_str("vlen_string")?;
                    if let Some(charset) = charset {
                        write!(f, " {charset}")?;
                    }
                    Ok(())
                } else {
                    write!(f, "vlen<{base_type}>")
                }
            }
            Self::Array {
                base_type,
                dimensions,
            } => write!(f, "array<{base_type}, {}>", Dims(dimensions)),
        }
    }
}
