//! Atomic datatype-message parsing and serialization with retained versions.

use alloc::vec::Vec;

use hdf5_pure_core::{
    BitFieldType, CharacterSet, DatatypeByteOrder, FixedPointType, FloatingPointType, FormatError,
    MantissaNormalization, StringPadding, TimeType,
};

use super::{DatatypeMessage, DatatypeProperties, DatatypeVersion, FloatingPointByteOrder};
use crate::bytes::Fields;
use crate::datatype::{
    self, BIT_FIELD_CLASS, DATATYPE_CLASS_MASK, DATATYPE_HEADER_SIZE, DATATYPE_VERSION_SHIFT,
    FIXED_POINT_CLASS, FLOAT_INTERNAL_PADDING_BIT, FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE,
    FLOAT_NORMALIZATION_LEADING_ONE, FLOAT_NORMALIZATION_MASK, FLOAT_NORMALIZATION_SHIFT,
    FLOAT_NORMALIZATION_UNNORMALIZED, FLOATING_POINT_CLASS, HIGH_PADDING_BIT, LOW_PADDING_BIT,
    OPAQUE_CLASS, OpaqueTagWire, STRING_CLASS, TIME_CLASS,
};
use crate::width::OffsetWidth;

impl DatatypeMessage {
    /// Parses an atomic datatype message and returns the number of bytes consumed.
    ///
    /// Supports fixed-point, floating-point, time, string, bitfield, and opaque descriptions
    /// at versions 1 through 5. VAX order requires version 3 or later. Floating-point order
    /// and reserved bits are checked through [`FloatingPointByteOrder::parse`].
    ///
    /// The parser accepts an opaque property filled with tag bytes without a NUL terminator.
    /// The writer uses terminated tags, as described in [`serialize`](Self::serialize).
    /// The properties are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete,
    /// [`FormatError::InvalidDatatypeMessageVersion`] if the version is outside `1..=5`, and
    /// [`FormatError::InvalidDatatypeClass`] for a class outside the supported atomic types.
    /// Returns [`FormatError::InvalidDatatypeFlags`] or [`FormatError::InvalidOpaqueReservedBits`]
    /// if reserved class bits are set. Propagates numeric layout, floating-point byte-order,
    /// string metadata, opaque-tag, and element-size errors.
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
        let mut fields = Fields::new(data, 0);
        let header = Header::parse(&mut fields)?;
        let (properties, consumed) = match header.class {
            FIXED_POINT_CLASS => {
                header.reserved(FIXED_RESERVED)?;
                let datatype = FixedPointType::new(
                    header.size,
                    binary_order(header.flags[0]),
                    header.flags[0] & SIGNED_BIT != 0,
                    fields.u16()?,
                    fields.u16()?,
                )?
                .with_padding(
                    header.flags[0] & LOW_PADDING_BIT != 0,
                    header.flags[0] & HIGH_PADDING_BIT != 0,
                );
                (DatatypeProperties::FixedPoint(datatype), fields.pos())
            }
            FLOATING_POINT_CLASS => {
                let order = FloatingPointByteOrder::parse(header.version, header.flags)?;
                let normalization = match (header.flags[0] >> FLOAT_NORMALIZATION_SHIFT)
                    & FLOAT_NORMALIZATION_MASK
                {
                    FLOAT_NORMALIZATION_UNNORMALIZED => MantissaNormalization::Unnormalized,
                    FLOAT_NORMALIZATION_LEADING_ONE => MantissaNormalization::LeadingOne,
                    FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE => {
                        MantissaNormalization::ImpliedLeadingOne
                    }
                    value => return Err(FormatError::InvalidMantissaNormalization(value)),
                };
                let datatype = FloatingPointType::new(
                    header.size,
                    order.order(),
                    normalization,
                    header.flags[1],
                    fields.u16()?,
                    fields.u16()?,
                    fields.u8()?,
                    fields.u8()?,
                    fields.u8()?,
                    fields.u8()?,
                    fields.u32()?,
                )?
                .with_padding(
                    header.flags[0] & LOW_PADDING_BIT != 0,
                    header.flags[0] & HIGH_PADDING_BIT != 0,
                    header.flags[0] & FLOAT_INTERNAL_PADDING_BIT != 0,
                );
                (DatatypeProperties::FloatingPoint(datatype), fields.pos())
            }
            TIME_CLASS => {
                header.reserved(TIME_RESERVED)?;
                (
                    DatatypeProperties::Time(TimeType::new(
                        header.size,
                        binary_order(header.flags[0]),
                        fields.u16()?,
                    )?),
                    fields.pos(),
                )
            }
            STRING_CLASS => {
                header.reserved(STRING_RESERVED)?;
                (
                    DatatypeProperties::String {
                        size: header.size,
                        padding: datatype::parse_string_padding(
                            header.flags[0] & STRING_PADDING_MASK,
                        )?,
                        charset: datatype::parse_charset(header.flags[0] >> STRING_CHARSET_SHIFT)?,
                    },
                    fields.pos(),
                )
            }
            BIT_FIELD_CLASS => {
                header.reserved(BITFIELD_RESERVED)?;
                let datatype = BitFieldType::new(
                    header.size,
                    binary_order(header.flags[0]),
                    fields.u16()?,
                    fields.u16()?,
                )?
                .with_padding(
                    header.flags[0] & LOW_PADDING_BIT != 0,
                    header.flags[0] & HIGH_PADDING_BIT != 0,
                );
                (DatatypeProperties::BitField(datatype), fields.pos())
            }
            OPAQUE_CLASS => {
                let reserved = u16::from_le_bytes([header.flags[1], header.flags[2]]);
                if reserved != 0 {
                    return Err(FormatError::InvalidOpaqueReservedBits { bits: reserved });
                }
                let tag = OpaqueTagWire::parse(header.flags[0], &data[fields.pos()..])?;
                (
                    DatatypeProperties::Opaque {
                        size: header.size,
                        tag: tag.logical.to_vec(),
                    },
                    fields.pos() + usize::from(tag.encoded_len),
                )
            }
            class => return Err(FormatError::InvalidDatatypeClass(class)),
        };
        Ok((Self::new(header.version, properties)?, consumed))
    }

    /// Serializes an atomic description with its retained message version.
    ///
    /// Atomic element sizes are independent of `offset_width`. The writer appends a NUL
    /// terminator to each nonempty opaque tag and pads its property to a multiple of eight
    /// bytes. The maximum writable logical tag length is 247 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeClass`] for a non-atomic description,
    /// [`FormatError::OpaqueTagTooLong`] for a tag longer than 247 bytes, and
    /// [`FormatError::InvalidOpaqueTagByte`] for a tag containing a NUL byte.
    /// Propagates errors from checking floating-point version/order compatibility and
    /// calculating the stored element size.
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
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, FormatError> {
        let flags = self.leaf_flags()?;
        let mut output = Vec::new();
        output.push(self.properties.class() | (self.version.get() << DATATYPE_VERSION_SHIFT));
        output.extend_from_slice(&flags);
        output.extend_from_slice(&self.stored_size(offset_width)?.to_le_bytes());
        match &self.properties {
            DatatypeProperties::FixedPoint(datatype) => {
                output.extend_from_slice(&datatype.layout().bit_offset().to_le_bytes());
                output.extend_from_slice(&datatype.layout().bit_precision().to_le_bytes());
            }
            DatatypeProperties::FloatingPoint(datatype) => {
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
            DatatypeProperties::Time(datatype) => {
                output.extend_from_slice(&datatype.bit_precision().get().to_le_bytes())
            }
            DatatypeProperties::BitField(datatype) => {
                output.extend_from_slice(&datatype.bit_offset().to_le_bytes());
                output.extend_from_slice(&datatype.bit_precision().get().to_le_bytes());
            }
            DatatypeProperties::Opaque { tag, .. } => {
                output.extend_from_slice(tag);
                output.resize(DATATYPE_HEADER_SIZE + usize::from(flags[0]), 0);
            }
            DatatypeProperties::String { .. } => {}
            properties => return Err(FormatError::InvalidDatatypeClass(properties.class())),
        }
        Ok(output)
    }

    /// Encodes the three class-specific flag bytes for an atomic description.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeClass`] for a non-atomic description.
    /// Propagates floating-point version/order and opaque-tag encoding errors.
    fn leaf_flags(&self) -> Result<[u8; 3], FormatError> {
        Ok(match &self.properties {
            DatatypeProperties::FixedPoint(datatype) => [
                binary_order_bit(datatype.byte_order())
                    | datatype::padding_bits(
                        datatype.layout().low_padding(),
                        datatype.layout().high_padding(),
                    )
                    | if datatype.layout().signed() {
                        SIGNED_BIT
                    } else {
                        0
                    },
                0,
                0,
            ],
            DatatypeProperties::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                let normalization = match layout.mantissa_normalization() {
                    MantissaNormalization::Unnormalized => FLOAT_NORMALIZATION_UNNORMALIZED,
                    MantissaNormalization::LeadingOne => FLOAT_NORMALIZATION_LEADING_ONE,
                    MantissaNormalization::ImpliedLeadingOne => {
                        FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE
                    }
                };
                [
                    FloatingPointByteOrder::new(self.version, datatype.byte_order())?.bits()
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
            DatatypeProperties::Time(datatype) => [binary_order_bit(datatype.byte_order()), 0, 0],
            DatatypeProperties::String {
                padding, charset, ..
            } => {
                let padding = match padding {
                    StringPadding::NullTerminate => NULL_TERMINATE,
                    StringPadding::NullPad => NULL_PAD,
                    StringPadding::SpacePad => SPACE_PAD,
                };
                let charset = match charset {
                    CharacterSet::Ascii => ASCII_CHARSET,
                    CharacterSet::Utf8 => UTF8_CHARSET,
                };
                [padding | (charset << STRING_CHARSET_SHIFT), 0, 0]
            }
            DatatypeProperties::BitField(datatype) => [
                binary_order_bit(datatype.byte_order())
                    | datatype::padding_bits(datatype.low_padding(), datatype.high_padding()),
                0,
                0,
            ],
            DatatypeProperties::Opaque { tag, .. } => {
                [OpaqueTagWire::encode(tag)?.encoded_len, 0, 0]
            }
            properties => return Err(FormatError::InvalidDatatypeClass(properties.class())),
        })
    }
}

/// The common datatype header with a checked message version.
struct Header {
    class: u8,
    version: DatatypeVersion,
    flags: [u8; 3],
    size: u32,
}

impl Header {
    /// Parses the common header and checks its message version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete, or
    /// [`FormatError::InvalidDatatypeMessageVersion`] if the version is outside `1..=5`.
    fn parse(fields: &mut Fields<'_>) -> Result<Self, FormatError> {
        let class_and_version = fields.u8()?;
        Ok(Self {
            class: class_and_version & DATATYPE_CLASS_MASK,
            version: DatatypeVersion::try_from(class_and_version >> DATATYPE_VERSION_SHIFT)?,
            flags: fields.array()?,
            size: fields.u32()?,
        })
    }

    /// Checks the class flags against a mask of reserved bits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] if any masked bit is set.
    fn reserved(&self, mask: u32) -> Result<(), FormatError> {
        let flags = u32::from(self.flags[0])
            | (u32::from(self.flags[1]) << 8)
            | (u32::from(self.flags[2]) << 16);
        if flags & mask != 0 {
            return Err(FormatError::InvalidDatatypeFlags {
                class: self.class,
                version: self.version.get(),
                flags,
            });
        }
        Ok(())
    }
}

fn binary_order(flags: u8) -> DatatypeByteOrder {
    if flags & BINARY_ORDER_BIT == 0 {
        DatatypeByteOrder::LittleEndian
    } else {
        DatatypeByteOrder::BigEndian
    }
}

fn binary_order_bit(order: DatatypeByteOrder) -> u8 {
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
