use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU32;

use self::byte_order::DatatypeByteOrder;
use self::layout::{BitFieldType, FixedPointType, FloatingPointType, TimeType};
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
/// [`type_size`](Self::type_size) gives the element size in bytes. The fixed-point,
/// floating-point, time, and bit-field payloads contain checked sizes and bit ranges.
/// Metadata for a numeric type wider than eight bytes or with an unsupported floating-point
/// encoding is available even when numeric value conversions cannot decode its values.
///
/// "The Datatype Message" of the [format specification, version 4.0][spec] defines datatype
/// message versions 1 through 5, including a Complex class (11), which this enum does not
/// represent.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Datatype {
    /// Class 0: Fixed-point (integer) types.
    FixedPoint(FixedPointType),
    /// Class 1: Floating-point types.
    FloatingPoint(FloatingPointType),
    /// Class 2: Time type (rarely used).
    Time(TimeType),
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
    BitField(BitFieldType),
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
    VariableLength(VariableLengthType),
    /// Class 10: Array type.
    Array {
        /// Datatype of one array element.
        base_type: Box<Datatype>,
        /// Extent of each array dimension.
        dimensions: Vec<u32>,
    },
}

impl Datatype {
    /// Returns the size in bytes of one element of this type.
    pub fn type_size(&self) -> u32 {
        match self {
            Datatype::FixedPoint(datatype) => datatype.size().get(),
            Datatype::FloatingPoint(datatype) => datatype.size().get(),
            Datatype::Time(datatype) => datatype.size().get(),
            Datatype::String { size, .. } => *size,
            Datatype::BitField(datatype) => datatype.size().get(),
            Datatype::Opaque { size, .. } => *size,
            Datatype::Compound { size, .. } => *size,
            Datatype::Reference { size, .. } => *size,
            Datatype::Enumeration { size, .. } => *size,
            Datatype::VariableLength(_) => 16,
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
            Datatype::FixedPoint(..) => 0,
            Datatype::FloatingPoint(..) => 1,
            Datatype::Time(..) => 2,
            Datatype::String { .. } => 3,
            Datatype::BitField(..) => 4,
            Datatype::Opaque { .. } => 5,
            Datatype::Compound { .. } => 6,
            Datatype::Reference { .. } => 7,
            Datatype::Enumeration { .. } => 8,
            Datatype::VariableLength(_) => 9,
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
        match self {
            Self::FixedPoint(datatype) => Ok(datatype.size()),
            Self::FloatingPoint(datatype) => Ok(datatype.size()),
            Self::Time(datatype) => Ok(datatype.size()),
            Self::BitField(datatype) => Ok(datatype.size()),
            _ => NonZeroU32::new(self.type_size()).ok_or(FormatError::ZeroSizedDatatype {
                class: self.class_code(),
            }),
        }
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
            Self::FixedPoint(datatype) => {
                let layout = datatype.layout();
                let size = datatype.size().get();
                let sign = if layout.signed() { 'i' } else { 'u' };
                write!(f, "{sign}{}", bit_width(size))?;
                write_bit_span(f, size, layout.bit_offset(), layout.bit_precision())?;
                write_byte_order(f, &datatype.byte_order())
            }
            Self::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                let size = datatype.size().get();
                write!(f, "f{}", bit_width(size))?;
                write_bit_span(f, size, layout.bit_offset(), layout.bit_precision())?;
                write_byte_order(f, &datatype.byte_order())
            }
            Self::Time(datatype) => {
                let size = datatype.size().get();
                write!(f, "time{}", bit_width(size))?;
                write_bit_span(f, size, 0, datatype.bit_precision().get())?;
                write_byte_order(f, &datatype.byte_order())
            }
            Self::String {
                size,
                padding,
                charset,
            } => write!(f, "string[{size}] {charset} {padding}"),
            Self::BitField(datatype) => {
                let size = datatype.size().get();
                write!(f, "bitfield{}", bit_width(size))?;
                write_bit_span(
                    f,
                    size,
                    datatype.bit_offset(),
                    datatype.bit_precision().get(),
                )?;
                write_byte_order(f, &datatype.byte_order())
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
            Self::VariableLength(VariableLengthType::Sequence { element_type }) => {
                write!(f, "vlen<{element_type}>")
            }
            Self::VariableLength(VariableLengthType::String(string)) => {
                write!(f, "vlen_string {}", string.charset())
            }
            Self::Array {
                base_type,
                dimensions,
            } => write!(f, "array<{base_type}, {}>", Dims(dimensions)),
        }
    }
}

/// A variable-length sequence or string.
///
/// Strings have required padding and character-set metadata. Sequences hold their element
/// datatype.
///
/// The variable-length class for datatype message versions 1 through 5 is defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{CharacterSet, Datatype, StringPadding, VariableLengthType};
///
/// let element = Datatype::String {
///     size: 1,
///     padding: StringPadding::NullTerminate,
///     charset: CharacterSet::Ascii,
/// };
/// let sequence = VariableLengthType::sequence(element);
/// let VariableLengthType::Sequence { element_type } = sequence else {
///     panic!("expected a sequence");
/// };
/// assert_eq!(element_type.type_size(), 1);
///
/// let datatype = Datatype::VariableLength(VariableLengthType::string(
///     StringPadding::NullTerminate,
///     CharacterSet::Utf8,
/// ));
/// let Datatype::VariableLength(VariableLengthType::String(string)) = datatype else {
///     panic!("expected a variable-length string");
/// };
/// assert_eq!(string.charset(), &CharacterSet::Utf8);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub enum VariableLengthType {
    /// A sequence whose elements share one datatype.
    Sequence {
        /// The datatype of one element.
        element_type: Box<Datatype>,
    },
    /// A string with padding and character-set metadata.
    String(VariableLengthStringType),
}

impl VariableLengthType {
    /// Constructs a sequence with the supplied element datatype.
    pub fn sequence(element_type: Datatype) -> Self {
        Self::Sequence {
            element_type: Box::new(element_type),
        }
    }

    /// Constructs a string with an unsigned one-byte integer parent datatype.
    pub fn string(padding: StringPadding, charset: CharacterSet) -> Self {
        Self::String(VariableLengthStringType::new(padding, charset))
    }

    /// Returns the element datatype of a sequence or the parent datatype of a string.
    pub fn base_type(&self) -> &Datatype {
        match self {
            Self::Sequence { element_type } => element_type,
            Self::String(string) => string.base_type(),
        }
    }
}

/// A variable-length string's padding, character set, and parent datatype.
///
/// The parent datatype has a positive element size. [`Self::new`] uses an unsigned one-byte
/// integer parent, and [`Self::with_base_type`] retains a supplied parent datatype.
///
/// The variable-length class for datatype message versions 1 through 5 is defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub struct VariableLengthStringType {
    padding: StringPadding,
    charset: CharacterSet,
    base_type: Box<Datatype>,
}

impl VariableLengthStringType {
    /// Constructs a string with an unsigned one-byte integer parent datatype.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{CharacterSet, StringPadding, VariableLengthStringType};
    ///
    /// let string = VariableLengthStringType::new(StringPadding::SpacePad, CharacterSet::Ascii);
    /// assert_eq!(string.padding(), &StringPadding::SpacePad);
    /// assert_eq!(string.charset(), &CharacterSet::Ascii);
    /// assert_eq!(string.base_type().type_size(), 1);
    /// ```
    pub fn new(padding: StringPadding, charset: CharacterSet) -> Self {
        Self {
            padding,
            charset,
            base_type: Box::new(Datatype::FixedPoint(FixedPointType::u8())),
        }
    }

    /// Constructs a string with the supplied parent datatype.
    ///
    /// The parent datatype is retained as part of the string metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `base_type` occupies zero bytes per element.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{
    ///     CharacterSet, Datatype, FormatError, StringPadding, VariableLengthStringType,
    /// };
    ///
    /// let base = Datatype::String {
    ///     size: 1,
    ///     padding: StringPadding::NullPad,
    ///     charset: CharacterSet::Ascii,
    /// };
    /// let string = VariableLengthStringType::with_base_type(
    ///     StringPadding::SpacePad,
    ///     CharacterSet::Utf8,
    ///     base.clone(),
    /// )?;
    /// assert_eq!(string.base_type(), &base);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn with_base_type(
        padding: StringPadding,
        charset: CharacterSet,
        base_type: Datatype,
    ) -> Result<Self, FormatError> {
        // H5O__dtype_decode_helper in H5Odtype.c, libhdf5 1.14.6, rejects a zero-sized datatype.
        base_type.element_size()?;
        Ok(Self {
            padding,
            charset,
            base_type: Box::new(base_type),
        })
    }

    /// Returns the treatment of unused string bytes.
    pub fn padding(&self) -> &StringPadding {
        &self.padding
    }

    /// Returns the encoding of the string bytes.
    pub fn charset(&self) -> &CharacterSet {
        &self.charset
    }

    /// Returns the parent datatype stored with the string metadata.
    pub fn base_type(&self) -> &Datatype {
        &self.base_type
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::CharacterSet;
    use super::Datatype;
    use super::StringPadding;
    use super::VariableLengthStringType;
    use crate::FormatError;

    #[rstest]
    #[case::zero_width_string(Datatype::String {
        size: 0,
        padding: StringPadding::NullTerminate,
        charset: CharacterSet::Ascii,
    }, STRING_CLASS)]
    #[case::zero_width_opaque(Datatype::Opaque { size: 0, tag: Vec::new() }, OPAQUE_CLASS)]
    #[case::zero_element_array(Datatype::Array {
        base_type: Box::new(Datatype::String {
            size: 1,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        }),
        dimensions: vec![0],
    }, ARRAY_CLASS)]
    fn rejects_zero_sized_string_bases(#[case] base: Datatype, #[case] class: u8) {
        assert_eq!(
            VariableLengthStringType::with_base_type(
                StringPadding::NullTerminate,
                CharacterSet::Utf8,
                base,
            )
            .unwrap_err(),
            FormatError::ZeroSizedDatatype { class },
        );
    }

    const STRING_CLASS: u8 = 0x03;
    const OPAQUE_CLASS: u8 = 0x05;
    const ARRAY_CLASS: u8 = 0x0A;
}
