//! Atomic datatype property parsing and encoding.
//!
//! The atomic property layouts for datatype message versions 1 through 5 are defined in
//! "The Datatype Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use core::num::NonZeroU32;

use hdf5_pure_core::{
    BitFieldType, CharacterSet, Datatype, DatatypeByteOrder, FixedPointType, FloatingPointType,
    FormatError, MantissaNormalization, StringPadding, TimeType,
};

use super::encode::{NodeHeader, Output};
use super::{DatatypeProperties, DatatypeVersion, FloatingPointByteOrder};
use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::datatype::header::{AtomicClass, Context, HeaderFields};
use crate::datatype::header::{
    BIT_FIELD_CLASS, FIXED_POINT_CLASS, FLOATING_POINT_CLASS, OPAQUE_CLASS, STRING_CLASS,
    TIME_CLASS,
};
use crate::datatype::opaque::{OPAQUE_TAG_ALIGNMENT, OpaqueTagWire};
use crate::datatype::{stored_size, string};

impl AtomicClass {
    /// Decodes atomic properties from a cursor positioned after their header.
    ///
    /// Reads a complete fixed-size property block before checking class flags. Applies
    /// `flag_policy` to fixed-point, time, string, and bitfield descriptions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a property block is incomplete. Propagates
    /// class flag, element size, numeric layout, string metadata, and opaque tag errors.
    pub(in crate::datatype) fn decode<'input>(
        self,
        cursor: &mut ByteCursor<'input>,
        fields: HeaderFields,
        context: Context,
        flag_policy: AtomicReservedFlagPolicy,
    ) -> Result<DecodedAtomic<'input>, FormatError> {
        let [flags, middle, high] = fields.flags;
        Ok(match self {
            Self::FixedPoint => {
                let (offset, precision) = read_bit_range(cursor)?;
                flag_policy.check(self, fields, FIXED_RESERVED)?;
                let datatype = FixedPointType::new(
                    fields.size,
                    binary_order(flags),
                    flags & SIGNED_BIT != 0,
                    offset,
                    precision,
                )?
                .with_padding(flags & LOW_PADDING_BIT != 0, flags & HIGH_PADDING_BIT != 0);
                DecodedAtomic::FixedPoint(datatype)
            }
            Self::FloatingPoint => {
                DecodedAtomic::FloatingPoint(decode_floating_point(cursor, fields, context)?)
            }
            Self::Time => {
                let precision = u16::from_le_bytes(cursor.array::<TIME_PROPERTIES_SIZE>()?);
                flag_policy.check(self, fields, TIME_RESERVED)?;
                DecodedAtomic::Time(TimeType::new(fields.size, binary_order(flags), precision)?)
            }
            Self::String => {
                flag_policy.check(self, fields, STRING_RESERVED)?;
                let padding = string::parse_string_padding(flags & STRING_PADDING_MASK)?;
                let charset = string::parse_charset(flags >> STRING_CHARSET_SHIFT)?;
                DecodedAtomic::String {
                    size: stored_size::positive_size(self as u8, fields.size)?,
                    padding,
                    charset,
                }
            }
            Self::BitField => {
                let (offset, precision) = read_bit_range(cursor)?;
                flag_policy.check(self, fields, BITFIELD_RESERVED)?;
                let datatype =
                    BitFieldType::new(fields.size, binary_order(flags), offset, precision)?
                        .with_padding(flags & LOW_PADDING_BIT != 0, flags & HIGH_PADDING_BIT != 0);
                DecodedAtomic::BitField(datatype)
            }
            Self::Opaque => {
                let reserved = u16::from_le_bytes([middle, high]);
                if reserved != 0 {
                    return Err(FormatError::InvalidOpaqueReservedBits { bits: reserved });
                }
                let tag = OpaqueTagWire::parse(flags, cursor)?;
                DecodedAtomic::Opaque {
                    size: stored_size::positive_size(self as u8, fields.size)?,
                    tag,
                }
            }
        })
    }
}

/// How to handle reserved bits in fixed-point, time, string, and bitfield descriptions.
pub(in crate::datatype) enum AtomicReservedFlagPolicy {
    /// Ignores the reserved bits when decoding a semantic datatype.
    Ignore,
    /// Rejects set reserved bits when decoding a retained-version description.
    Reject,
}

impl AtomicReservedFlagPolicy {
    /// Applies the selected policy to reserved bits in the class bit field.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] under [`Self::Reject`] if any bit in
    /// `mask` is set.
    fn check(self, class: AtomicClass, fields: HeaderFields, mask: u32) -> Result<(), FormatError> {
        match self {
            Self::Ignore => Ok(()),
            Self::Reject => fields.reject_reserved_flags(class as u8, mask),
        }
    }
}

/// Checked properties shared by the semantic and retained-version parsers.
///
/// The decoder borrows an opaque tag's logical bytes until constructing an owned description.
pub(in crate::datatype) enum DecodedAtomic<'input> {
    FixedPoint(FixedPointType),
    FloatingPoint(FloatingPointType),
    Time(TimeType),
    String {
        size: NonZeroU32,
        padding: StringPadding,
        charset: CharacterSet,
    },
    BitField(BitFieldType),
    Opaque {
        size: NonZeroU32,
        tag: OpaqueTagWire<'input>,
    },
}

impl DecodedAtomic<'_> {
    /// Returns the semantic datatype, copying any opaque tag bytes.
    pub(in crate::datatype) fn into_datatype(self) -> Datatype {
        match self {
            Self::FixedPoint(datatype) => Datatype::FixedPoint(datatype),
            Self::FloatingPoint(datatype) => Datatype::FloatingPoint(datatype),
            Self::Time(datatype) => Datatype::Time(datatype),
            Self::String {
                size,
                padding,
                charset,
            } => Datatype::String {
                size: size.get(),
                padding,
                charset,
            },
            Self::BitField(datatype) => Datatype::BitField(datatype),
            Self::Opaque { size, tag } => Datatype::Opaque {
                size: size.get(),
                tag: tag.logical().to_vec(),
            },
        }
    }

    /// Returns the properties for a retained-version description, copying any opaque tag bytes.
    pub(super) fn into_properties(self) -> DatatypeProperties {
        match self {
            Self::FixedPoint(datatype) => DatatypeProperties::FixedPoint(datatype),
            Self::FloatingPoint(datatype) => DatatypeProperties::FloatingPoint(datatype),
            Self::Time(datatype) => DatatypeProperties::Time(datatype),
            Self::String {
                size,
                padding,
                charset,
            } => DatatypeProperties::String {
                size: size.get(),
                padding,
                charset,
            },
            Self::BitField(datatype) => DatatypeProperties::BitField(datatype),
            Self::Opaque { size, tag } => DatatypeProperties::Opaque {
                size: size.get(),
                tag: tag.logical().to_vec(),
            },
        }
    }
}

/// Reads the complete floating-point property block before checking the class bit field.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the property block is incomplete. Propagates errors
/// from [`FloatingPointByteOrder::parse`], mapping those errors through `context`, and errors
/// checking the numeric layout and mantissa normalization.
pub(in crate::datatype) fn decode_floating_point(
    cursor: &mut ByteCursor<'_>,
    fields: HeaderFields,
    context: Context,
) -> Result<FloatingPointType, FormatError> {
    let &[
        offset_low,
        offset_high,
        precision_low,
        precision_high,
        exponent_at,
        exponent_size,
        mantissa_at,
        mantissa_size,
        bias0,
        bias1,
        bias2,
        bias3,
    ] = cursor.take_array::<FLOATING_POINT_PROPERTIES_SIZE>()?;
    let HeaderFields {
        version,
        flags,
        size,
    } = fields;
    let order = FloatingPointByteOrder::parse(version, flags)
        .map_err(|error| context.map_floating_point_flags_error(fields, error))?;
    let [flags, sign_at, _] = flags;
    let normalization = match (flags >> FLOAT_NORMALIZATION_SHIFT) & FLOAT_NORMALIZATION_MASK {
        FLOAT_NORMALIZATION_UNNORMALIZED => MantissaNormalization::Unnormalized,
        FLOAT_NORMALIZATION_LEADING_ONE => MantissaNormalization::LeadingOne,
        FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE => MantissaNormalization::ImpliedLeadingOne,
        value => return Err(FormatError::InvalidMantissaNormalization(value)),
    };
    Ok(FloatingPointType::new(
        size,
        order.order(),
        normalization,
        sign_at,
        u16::from_le_bytes([offset_low, offset_high]),
        u16::from_le_bytes([precision_low, precision_high]),
        exponent_at,
        exponent_size,
        mantissa_at,
        mantissa_size,
        u32::from_le_bytes([bias0, bias1, bias2, bias3]),
    )?
    .with_padding(
        flags & LOW_PADDING_BIT != 0,
        flags & HIGH_PADDING_BIT != 0,
        flags & FLOAT_INTERNAL_PADDING_BIT != 0,
    ))
}

/// Reads the bit offset and precision as one four-byte block.
///
/// # Errors
///
/// Returns [`UnexpectedEof`] if the block is incomplete.
fn read_bit_range(cursor: &mut ByteCursor<'_>) -> Result<(u16, u16), UnexpectedEof> {
    let &[offset_low, offset_high, precision_low, precision_high] =
        cursor.take_array::<BIT_RANGE_SIZE>()?;
    Ok((
        u16::from_le_bytes([offset_low, offset_high]),
        u16::from_le_bytes([precision_low, precision_high]),
    ))
}

fn binary_order(flags: u8) -> DatatypeByteOrder {
    if flags & BINARY_ORDER_BIT == 0 {
        DatatypeByteOrder::LittleEndian
    } else {
        DatatypeByteOrder::BigEndian
    }
}

/// Borrowed atomic properties.
pub(in crate::datatype) enum AtomicProperties<'a> {
    /// Fixed-point layout and byte order.
    FixedPoint(&'a FixedPointType),
    /// Floating-point layout and byte order.
    FloatingPoint(&'a FloatingPointType),
    /// Time precision and byte order.
    Time(&'a TimeType),
    /// Fixed-length string metadata.
    String {
        /// The width of one string in bytes.
        size: u32,
        /// The treatment of unused bytes.
        padding: &'a StringPadding,
        /// The encoding of the string bytes.
        charset: &'a CharacterSet,
    },
    /// Bitfield precision, padding, and byte order.
    BitField(&'a BitFieldType),
    /// Opaque data with a logical tag.
    Opaque {
        /// The element size in bytes.
        size: u32,
        /// The tag bytes without terminator or padding.
        tag: &'a [u8],
    },
}

impl AtomicProperties<'_> {
    /// Returns the stored element size and encoded property length, in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if the type has no element bytes. Propagates
    /// errors from checking an opaque tag or a floating-point order at the supplied version.
    pub(super) fn prepare(
        &self,
        version: Option<DatatypeVersion>,
    ) -> Result<(u32, usize), FormatError> {
        let size = NonZeroU32::new(self.size()).ok_or_else(|| FormatError::ZeroSizedDatatype {
            class: self.class(),
        })?;
        let property_size = match self {
            Self::FixedPoint(_) | Self::BitField(_) => BIT_RANGE_SIZE,
            Self::FloatingPoint(datatype) => {
                if let Some(version) = version {
                    FloatingPointByteOrder::new(version, datatype.byte_order())?;
                }
                FLOATING_POINT_PROPERTIES_SIZE
            }
            Self::Time(_) => TIME_PROPERTIES_SIZE,
            Self::String { .. } => 0,
            Self::Opaque { tag, .. } => usize::from(OpaqueTagWire::encode(tag)?.encoded_len()),
        };
        Ok((size.get(), property_size))
    }

    /// Writes the checked property fields and returns the header that describes them.
    pub(super) fn write(
        &self,
        output: &mut Output<'_, impl FnMut(usize, &[u8])>,
        version: Option<DatatypeVersion>,
    ) -> NodeHeader {
        let flags = match self {
            Self::FixedPoint(datatype) => {
                let layout = datatype.layout();
                output.write(&bit_range(layout.bit_offset(), layout.bit_precision()));
                [
                    binary_order_bit(datatype.byte_order())
                        | padding_bits(layout.low_padding(), layout.high_padding())
                        | if layout.signed() { SIGNED_BIT } else { 0 },
                    0,
                    0,
                ]
            }
            Self::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                let [offset_low, offset_high, precision_low, precision_high] =
                    bit_range(layout.bit_offset(), layout.bit_precision());
                let [bias_low, bias_middle_low, bias_middle_high, bias_high] =
                    layout.exponent_bias().to_le_bytes();
                let properties: [u8; FLOATING_POINT_PROPERTIES_SIZE] = [
                    offset_low,
                    offset_high,
                    precision_low,
                    precision_high,
                    layout.exponent_location(),
                    layout.exponent_size(),
                    layout.mantissa_location(),
                    layout.mantissa_size(),
                    bias_low,
                    bias_middle_low,
                    bias_middle_high,
                    bias_high,
                ];
                output.write(&properties);
                let normalization = match layout.mantissa_normalization() {
                    MantissaNormalization::Unnormalized => FLOAT_NORMALIZATION_UNNORMALIZED,
                    MantissaNormalization::LeadingOne => FLOAT_NORMALIZATION_LEADING_ONE,
                    MantissaNormalization::ImpliedLeadingOne => {
                        FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE
                    }
                };
                [
                    FloatingPointByteOrder::bits_for(datatype.byte_order())
                        | padding_bits(layout.low_padding(), layout.high_padding())
                        | if layout.internal_padding() {
                            FLOAT_INTERNAL_PADDING_BIT
                        } else {
                            0
                        }
                        | (normalization << FLOAT_NORMALIZATION_SHIFT),
                    layout.sign_location(),
                    0,
                ]
            }
            Self::Time(datatype) => {
                let properties: [u8; TIME_PROPERTIES_SIZE] =
                    datatype.bit_precision().get().to_le_bytes();
                output.write(&properties);
                [binary_order_bit(datatype.byte_order()), 0, 0]
            }
            Self::String {
                padding,
                charset,
                size: _,
            } => [
                string::padding_code(padding)
                    | (string::charset_code(charset) << STRING_CHARSET_SHIFT),
                0,
                0,
            ],
            Self::BitField(datatype) => {
                output.write(&bit_range(
                    datatype.bit_offset(),
                    datatype.bit_precision().get(),
                ));
                [
                    binary_order_bit(datatype.byte_order())
                        | padding_bits(datatype.low_padding(), datatype.high_padding()),
                    0,
                    0,
                ]
            }
            Self::Opaque { size: _, tag } => {
                let property_size = OpaqueTagWire::property_size(tag.len());
                output.write(tag);
                output.write(&[0; OPAQUE_TAG_ALIGNMENT as usize][..property_size - tag.len()]);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the prepared padded tag length fits the one-byte property field"
                )]
                let property_size = property_size as u8;
                [property_size, 0, 0]
            }
        };
        NodeHeader {
            class: self.class(),
            version: version.unwrap_or_else(|| self.canonical_version()),
            flags,
            size: self.size(),
        }
    }

    fn canonical_version(&self) -> DatatypeVersion {
        match self {
            Self::FloatingPoint(datatype) if datatype.byte_order() == DatatypeByteOrder::Vax => {
                DatatypeVersion::Three
            }
            _ => DatatypeVersion::One,
        }
    }

    fn class(&self) -> u8 {
        match self {
            Self::FixedPoint(_) => FIXED_POINT_CLASS,
            Self::FloatingPoint(_) => FLOATING_POINT_CLASS,
            Self::Time(_) => TIME_CLASS,
            Self::String { .. } => STRING_CLASS,
            Self::BitField(_) => BIT_FIELD_CLASS,
            Self::Opaque { .. } => OPAQUE_CLASS,
        }
    }

    fn size(&self) -> u32 {
        match self {
            Self::FixedPoint(datatype) => datatype.size().get(),
            Self::FloatingPoint(datatype) => datatype.size().get(),
            Self::Time(datatype) => datatype.size().get(),
            Self::String { size, .. } | Self::Opaque { size, .. } => *size,
            Self::BitField(datatype) => datatype.size().get(),
        }
    }
}

fn bit_range(offset: u16, precision: u16) -> [u8; BIT_RANGE_SIZE] {
    let [offset_low, offset_high] = offset.to_le_bytes();
    let [precision_low, precision_high] = precision.to_le_bytes();
    [offset_low, offset_high, precision_low, precision_high]
}

/// Encodes the low and high padding values in bits 1 and 2 of the class bit field.
pub(in crate::datatype) fn padding_bits(low: bool, high: bool) -> u8 {
    (if low { LOW_PADDING_BIT } else { 0 }) | (if high { HIGH_PADDING_BIT } else { 0 })
}

/// Encodes bit 0 for fixed-point, time, and bitfield byte order.
///
/// Callers pass [`DatatypeByteOrder::LittleEndian`] or [`DatatypeByteOrder::BigEndian`]
/// from checked metadata. The fields are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn binary_order_bit(order: DatatypeByteOrder) -> u8 {
    if order == DatatypeByteOrder::BigEndian {
        BINARY_ORDER_BIT
    } else {
        0
    }
}

const BIT_RANGE_SIZE: usize = 2 * size_of::<u16>();
const FLOATING_POINT_PROPERTIES_SIZE: usize =
    BIT_RANGE_SIZE + size_of::<[u8; 4]>() + size_of::<u32>();
const TIME_PROPERTIES_SIZE: usize = size_of::<u16>();

/// Bit 1 of the fixed-point, floating-point, and bitfield class bit fields, the low padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const LOW_PADDING_BIT: u8 = 0x02;
/// Bit 2 of the fixed-point, floating-point, and bitfield class bit fields, the high padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const HIGH_PADDING_BIT: u8 = 0x04;
/// Bit 3 of the floating-point class bit field, the internal padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const FLOAT_INTERNAL_PADDING_BIT: u8 = 0x08;

/// The bit position of mantissa normalization in the floating-point class bit field.
///
/// The field and its values are defined in "The Datatype Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const FLOAT_NORMALIZATION_SHIFT: u8 = 4;
/// The mask for the two mantissa normalization bits after shifting.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const FLOAT_NORMALIZATION_MASK: u8 = 0x03;
/// The encoded value for a mantissa stored without normalization.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const FLOAT_NORMALIZATION_UNNORMALIZED: u8 = 0;
/// The encoded value for a stored leading mantissa bit that is set, except for zero.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const FLOAT_NORMALIZATION_LEADING_ONE: u8 = 1;
/// The encoded value for an implied leading mantissa bit.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(in crate::datatype) const FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE: u8 = 2;
/// The reserved mantissa normalization value.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[cfg(test)]
pub(in crate::datatype) const FLOAT_NORMALIZATION_RESERVED: u8 = 3;

/// The mask for reserved fixed-point class bits 4 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FIXED_RESERVED: u32 = 0xff_fff0;
/// The mask for reserved time class bits 1 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const TIME_RESERVED: u32 = 0xff_fffe;
/// The mask for reserved string class bits 8 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const STRING_RESERVED: u32 = 0xff_ff00;
/// The mask for reserved bitfield class bits 3 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const BITFIELD_RESERVED: u32 = 0xff_fff8;
/// Bit 0, the byte-order bit for fixed-point, time, and bitfield descriptions.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const BINARY_ORDER_BIT: u8 = 0x01;
/// Bit 3, the signed-integer flag for fixed-point descriptions.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const SIGNED_BIT: u8 = 0x08;
/// The mask for string padding bits 0 through 3.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const STRING_PADDING_MASK: u8 = 0x0f;
/// The position of the string character-set field at bit 4.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const STRING_CHARSET_SHIFT: u8 = 4;
