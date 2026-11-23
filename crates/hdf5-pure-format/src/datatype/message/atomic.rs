//! Borrowed atomic datatype descriptions and their encoding.
//!
//! The property layouts are defined in "The Datatype Message" of the [format
//! specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use core::num::NonZeroU32;

use hdf5_pure_core::{
    BitFieldType, CharacterSet, DatatypeByteOrder, FixedPointType, FloatingPointType, FormatError,
    MantissaNormalization, StringPadding, TimeType,
};

use super::codec::{self, NodeHeader, Output};
use super::{DatatypeVersion, FloatingPointByteOrder};
use crate::datatype::{
    self, BIT_FIELD_CLASS, FIXED_POINT_CLASS, FLOAT_INTERNAL_PADDING_BIT,
    FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE, FLOAT_NORMALIZATION_LEADING_ONE,
    FLOAT_NORMALIZATION_SHIFT, FLOAT_NORMALIZATION_UNNORMALIZED, FLOATING_POINT_CLASS,
    OPAQUE_CLASS, OpaqueTagWire, STRING_CLASS, TIME_CLASS,
};

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
            Self::Opaque { tag, .. } => usize::from(OpaqueTagWire::encode(tag)?.encoded_len),
        };
        Ok((size.get(), property_size))
    }

    /// Writes the checked property fields and returns the header that describes them.
    pub(super) fn write(
        &self,
        output: &mut Output<'_>,
        version: Option<DatatypeVersion>,
    ) -> NodeHeader {
        let flags = match self {
            Self::FixedPoint(datatype) => {
                let layout = datatype.layout();
                output.write(&bit_range(layout.bit_offset(), layout.bit_precision()));
                [
                    codec::binary_order_bit(datatype.byte_order())
                        | datatype::padding_bits(layout.low_padding(), layout.high_padding())
                        | if layout.signed() {
                            codec::SIGNED_BIT
                        } else {
                            0
                        },
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
                        | datatype::padding_bits(layout.low_padding(), layout.high_padding())
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
                [codec::binary_order_bit(datatype.byte_order()), 0, 0]
            }
            Self::String {
                padding,
                charset,
                size: _,
            } => [
                codec::padding_code(padding)
                    | (codec::charset_code(charset) << codec::STRING_CHARSET_SHIFT),
                0,
                0,
            ],
            Self::BitField(datatype) => {
                output.write(&bit_range(
                    datatype.bit_offset(),
                    datatype.bit_precision().get(),
                ));
                [
                    codec::binary_order_bit(datatype.byte_order())
                        | datatype::padding_bits(datatype.low_padding(), datatype.high_padding()),
                    0,
                    0,
                ]
            }
            Self::Opaque { size: _, tag } => {
                let property_size = OpaqueTagWire::property_size(tag.len());
                let (value, padding) = output.take(property_size).split_at_mut(tag.len());
                value.copy_from_slice(tag);
                padding.fill(0);
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

const BIT_RANGE_SIZE: usize = 2 * size_of::<u16>();
const FLOATING_POINT_PROPERTIES_SIZE: usize =
    BIT_RANGE_SIZE + size_of::<[u8; 4]>() + size_of::<u32>();
const TIME_PROPERTIES_SIZE: usize = size_of::<u16>();
