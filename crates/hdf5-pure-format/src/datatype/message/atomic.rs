//! Atomic datatype property parsing and encoding, and legacy reference parsing.
//!
//! The atomic property layouts for datatype message versions 1 through 5 are defined in
//! "The Datatype Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use core::num::NonZeroU32;

use hdf5_pure_core::{
    BitFieldType, CharacterSet, Datatype, DatatypeByteOrder, FixedPointType, FloatingPointType,
    FormatError, MantissaNormalization, ReferenceType, StringPadding, TimeType,
};

use super::codec::{self, NodeHeader, Output};
use super::{DatatypeProperties, DatatypeVersion, FloatingPointByteOrder};
use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::datatype::header::{AtomicClass, Context, Header, HeaderFields};
use crate::datatype::{
    self, BIT_FIELD_CLASS, FIXED_POINT_CLASS, FLOAT_INTERNAL_PADDING_BIT,
    FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE, FLOAT_NORMALIZATION_LEADING_ONE,
    FLOAT_NORMALIZATION_MASK, FLOAT_NORMALIZATION_SHIFT, FLOAT_NORMALIZATION_UNNORMALIZED,
    FLOATING_POINT_CLASS, HIGH_PADDING_BIT, LOW_PADDING_BIT, OPAQUE_CLASS, OPAQUE_TAG_ALIGNMENT,
    OpaqueTagWire, STRING_CLASS, TIME_CLASS,
};

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
                flag_policy.check(self, fields, codec::FIXED_RESERVED)?;
                let datatype = FixedPointType::new(
                    fields.size,
                    binary_order(flags),
                    flags & codec::SIGNED_BIT != 0,
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
                flag_policy.check(self, fields, codec::TIME_RESERVED)?;
                DecodedAtomic::Time(TimeType::new(fields.size, binary_order(flags), precision)?)
            }
            Self::String => {
                flag_policy.check(self, fields, codec::STRING_RESERVED)?;
                let padding = datatype::parse_string_padding(flags & codec::STRING_PADDING_MASK)?;
                let charset = datatype::parse_charset(flags >> codec::STRING_CHARSET_SHIFT)?;
                DecodedAtomic::String {
                    size: super::positive_size(self as u8, fields.size)?,
                    padding,
                    charset,
                }
            }
            Self::BitField => {
                let (offset, precision) = read_bit_range(cursor)?;
                flag_policy.check(self, fields, codec::BITFIELD_RESERVED)?;
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
                    size: super::positive_size(self as u8, fields.size)?,
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

/// Decodes a legacy object or dataset-region datatype from its common header.
///
/// Retains the declared element size.
///
/// # Errors
///
/// Returns [`FormatError::InvalidReferenceType`] for a kind other than legacy object or dataset
/// region, or [`FormatError::ZeroSizedDatatype`] for a zero element size.
pub(in crate::datatype) fn decode_reference(header: &Header) -> Result<Datatype, FormatError> {
    // `H5O__dtype_decode_helper` reads legacy kinds at every supported datatype message version
    // (H5Odtype.c, HDF5 2.2.0).
    let ref_type = match header.flags[0] & REFERENCE_KIND_MASK {
        codec::OBJECT_REFERENCE_KIND => ReferenceType::Object,
        codec::REGION_REFERENCE_KIND => ReferenceType::DatasetRegion,
        kind => return Err(FormatError::InvalidReferenceType(kind)),
    };
    Ok(Datatype::Reference {
        size: super::positive_size(header.class.code(), header.size)?.get(),
        ref_type,
    })
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
    if flags & codec::BINARY_ORDER_BIT == 0 {
        DatatypeByteOrder::LittleEndian
    } else {
        DatatypeByteOrder::BigEndian
    }
}

// "The Datatype Message" in format specification version 4.0 assigns bits 0 through 3
// of the reference class bit field to the reference kind.
const REFERENCE_KIND_MASK: u8 = 0x0f;

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

const BIT_RANGE_SIZE: usize = 2 * size_of::<u16>();
const FLOATING_POINT_PROPERTIES_SIZE: usize =
    BIT_RANGE_SIZE + size_of::<[u8; 4]>() + size_of::<u32>();
const TIME_PROPERTIES_SIZE: usize = size_of::<u16>();
