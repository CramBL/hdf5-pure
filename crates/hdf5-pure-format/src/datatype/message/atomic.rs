//! Borrowed atomic datatype descriptions and their encoding.
//!
//! The property layouts are defined in "The Datatype Message" of the [format
//! specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use alloc::vec::Vec;
use core::num::NonZeroU32;

use hdf5_pure_core::{
    BitFieldType, CharacterSet, DatatypeByteOrder, FixedPointType, FloatingPointType, FormatError,
    MantissaNormalization, StringPadding, TimeType,
};

use super::codec;
use super::{DatatypeVersion, FloatingPointByteOrder};
use crate::datatype::{
    self, BIT_FIELD_CLASS, DATATYPE_HEADER_SIZE, EncodedNode, FIXED_POINT_CLASS,
    FLOAT_INTERNAL_PADDING_BIT, FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE,
    FLOAT_NORMALIZATION_LEADING_ONE, FLOAT_NORMALIZATION_SHIFT, FLOAT_NORMALIZATION_UNNORMALIZED,
    FLOATING_POINT_CLASS, OPAQUE_CLASS, OpaqueTagWire, STRING_CLASS, TIME_CLASS,
};

/// A borrowed atomic description with a checked version, class flags, and positive size.
pub(in crate::datatype) struct AtomicMessage<'a> {
    /// The selected message version.
    version: DatatypeVersion,
    /// The three class-specific bytes derived from the checked properties.
    flags: [u8; 3],
    /// The element size in bytes.
    size: NonZeroU32,
    /// The borrowed properties used to encode the body.
    properties: AtomicProperties<'a>,
}

impl<'a> AtomicMessage<'a> {
    /// Constructs a description at version 3 for VAX or version 1 for other atomic types.
    ///
    /// # Errors
    ///
    /// Propagates size and opaque-tag errors from [`new`](Self::new).
    pub(in crate::datatype) fn canonical(
        properties: AtomicProperties<'a>,
    ) -> Result<Self, FormatError> {
        let version = match &properties {
            AtomicProperties::FloatingPoint(datatype)
                if datatype.byte_order() == DatatypeByteOrder::Vax =>
            {
                DatatypeVersion::Three
            }
            _ => DatatypeVersion::One,
        };
        Self::new(version, properties)
    }

    /// Constructs a description with checked size and class flags at `version`.
    ///
    /// The class fields are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec]. Floating-point version/order compatibility is checked through
    /// [`FloatingPointByteOrder::new`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] for a zero-sized string or opaque type,
    /// and [`FormatError::InvalidDatatypeVersion`] for VAX order before version 3.
    /// Returns [`FormatError::OpaqueTagTooLong`] for a tag longer than 247 bytes, and
    /// [`FormatError::InvalidOpaqueTagByte`] for a tag containing a NUL byte.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    pub(super) fn new(
        version: DatatypeVersion,
        properties: AtomicProperties<'a>,
    ) -> Result<Self, FormatError> {
        let size = NonZeroU32::new(properties.size()).ok_or(FormatError::ZeroSizedDatatype {
            class: properties.class(),
        })?;
        let flags = match &properties {
            AtomicProperties::FixedPoint(datatype) => [
                codec::binary_order_bit(datatype.byte_order())
                    | datatype::padding_bits(
                        datatype.layout().low_padding(),
                        datatype.layout().high_padding(),
                    )
                    | if datatype.layout().signed() {
                        codec::SIGNED_BIT
                    } else {
                        0
                    },
                0,
                0,
            ],
            AtomicProperties::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                let normalization = match layout.mantissa_normalization() {
                    MantissaNormalization::Unnormalized => FLOAT_NORMALIZATION_UNNORMALIZED,
                    MantissaNormalization::LeadingOne => FLOAT_NORMALIZATION_LEADING_ONE,
                    MantissaNormalization::ImpliedLeadingOne => {
                        FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE
                    }
                };
                [
                    FloatingPointByteOrder::new(version, datatype.byte_order())?.bits()
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
            AtomicProperties::Time(datatype) => {
                [codec::binary_order_bit(datatype.byte_order()), 0, 0]
            }
            AtomicProperties::String {
                size: _,
                padding,
                charset,
            } => [
                codec::padding_code(padding)
                    | (codec::charset_code(charset) << codec::STRING_CHARSET_SHIFT),
                0,
                0,
            ],
            AtomicProperties::BitField(datatype) => [
                codec::binary_order_bit(datatype.byte_order())
                    | datatype::padding_bits(datatype.low_padding(), datatype.high_padding()),
                0,
                0,
            ],
            AtomicProperties::Opaque { size: _, tag } => {
                [OpaqueTagWire::encode(tag)?.encoded_len, 0, 0]
            }
        };
        Ok(Self {
            version,
            flags,
            size,
            properties,
        })
    }

    /// Appends the complete message and returns its encoded header metadata.
    ///
    /// Writes borrowed properties directly into `output`. The property layouts are defined
    /// in "The Datatype Message" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    pub(in crate::datatype) fn encode_into(&self, output: &mut Vec<u8>) -> EncodedNode {
        let Self {
            version,
            flags,
            size,
            properties,
        } = self;
        let header = datatype::append_header(
            output,
            properties.class(),
            version.get(),
            *flags,
            size.get(),
        );
        match properties {
            AtomicProperties::FixedPoint(datatype) => {
                output.extend_from_slice(&datatype.layout().bit_offset().to_le_bytes());
                output.extend_from_slice(&datatype.layout().bit_precision().to_le_bytes());
            }
            AtomicProperties::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                output.extend_from_slice(&layout.bit_offset().to_le_bytes());
                output.extend_from_slice(&layout.bit_precision().to_le_bytes());
                output.extend_from_slice(&[
                    layout.exponent_location(),
                    layout.exponent_size(),
                    layout.mantissa_location(),
                    layout.mantissa_size(),
                ]);
                output.extend_from_slice(&layout.exponent_bias().to_le_bytes());
            }
            AtomicProperties::Time(datatype) => {
                output.extend_from_slice(&datatype.bit_precision().get().to_le_bytes());
            }
            AtomicProperties::String { .. } => {}
            AtomicProperties::BitField(datatype) => {
                output.extend_from_slice(&datatype.bit_offset().to_le_bytes());
                output.extend_from_slice(&datatype.bit_precision().get().to_le_bytes());
            }
            AtomicProperties::Opaque { size: _, tag } => {
                output.extend_from_slice(tag);
                output.resize(
                    header.offset + DATATYPE_HEADER_SIZE + usize::from(flags[0]),
                    0,
                );
            }
        }
        header.finish(output)
    }
}

/// Borrowed properties supplied when constructing an [`AtomicMessage`].
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
