//! HDF5 Datatype message parsing (message type 0x0003).

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use byteorder::ByteOrder;
use byteorder::LittleEndian;

use hdf5_pure_core::__private;
pub(crate) use hdf5_pure_core::CharacterSet;
use hdf5_pure_core::ComplexType;
pub(crate) use hdf5_pure_core::Datatype;
use hdf5_pure_core::DatatypeByteOrder;

#[cfg(test)]
use hdf5_pure_core::FloatingPointLayout;
use hdf5_pure_core::MantissaNormalization;
pub(crate) use hdf5_pure_core::ReferenceType;
pub(crate) use hdf5_pure_core::StringPadding;
use hdf5_pure_core::VariableLengthStringType;
use hdf5_pure_core::VariableLengthType;
use hdf5_pure_core::{BitFieldType, FixedPointType, FloatingPointType, TimeType};

use crate::bytes;
use crate::error::FormatError;

pub(crate) mod byte_order;
pub(crate) mod layout;
mod legacy_member;

/// Returns the datatype's element size as a nonzero host-sized integer.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes
/// per element, or [`FormatError::ValueTooLargeForPlatform`] if its size exceeds
/// the target's `usize` range.
pub fn element_size_usize(datatype: &Datatype) -> Result<NonZeroUsize, FormatError> {
    usize::try_from(datatype.element_size()?.get())
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or(FormatError::ValueTooLargeForPlatform {
            value: u64::from(datatype.type_size()),
            target: "usize",
        })
}

/// Parses a datatype message body and returns the type and the number of bytes read.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the body ends within a field,
/// [`FormatError::InvalidDatatypeClass`] if the class is unsupported, or
/// [`FormatError::InvalidDatatypeVersion`] if a compound, array, complex datatype, or complex
/// parent has an unsupported version. Returns
/// [`FormatError::InvalidReferenceType`] for an unknown reference kind,
/// [`FormatError::InvalidStringPadding`] for an unknown string padding code, or
/// [`FormatError::InvalidCharacterSet`] for an unknown character set code. Returns
/// [`FormatError::ZeroSizedDatatype`] if the parsed type has no element bytes.
/// Returns [`FormatError::InvalidFixedPointRange`], [`FormatError::InvalidFloatingPointRange`], or
/// [`FormatError::InvalidNumericBitRange`] if a numeric bit range is empty or exceeds its element
/// size. Returns [`FormatError::InvalidFloatingPointField`] if a floating-point field is empty or
/// outside that range, [`FormatError::OverlappingFloatingPointFields`] if fields overlap, or
/// [`FormatError::InvalidMantissaNormalization`] if the normalization code is reserved.
///
/// Returns [`FormatError::InvalidVariableLengthSubtype`] if a variable-length subtype is
/// neither sequence nor string, [`FormatError::InvalidVariableLengthReservedBits`] if its
/// class bit field has reserved bits set, or
/// [`FormatError::InvalidVariableLengthSequenceMetadata`] if a sequence has nonzero string
/// padding or character-set metadata.
///
/// Returns [`FormatError::InvalidCompoundV1Rank`] if a version 1 compound member declares more
/// than four dimensions, [`FormatError::InvalidCompoundV1Extent`] if an active dimension is
/// zero, or [`FormatError::UnsupportedCompoundV1Permutation`] if its permutation field is nonzero.
/// Returns [`FormatError::CompoundArraySizeOverflow`] if a version 1 member's array size exceeds
/// [`u32::MAX`], or [`FormatError::CompoundFieldOutOfBounds`] if its byte range exceeds the
/// compound's declared size.
///
/// Returns [`FormatError::InvalidComplexFlags`] if a complex class bit field has reserved bits,
/// [`FormatError::UnsupportedComplexForm`] if the complex form is not homogeneous rectangular,
/// or [`FormatError::InvalidComplexParent`] if its parent is not floating point. Returns
/// [`FormatError::InvalidComplexParentFlags`] if the parent has reserved bits or byte order,
/// [`FormatError::ComplexSizeOverflow`] if twice its size exceeds [`u32::MAX`], or
/// [`FormatError::InvalidComplexSize`] if the declared complex size differs from that product.
pub fn parse_datatype(data: &[u8]) -> Result<(Datatype, usize), FormatError> {
    // Minimum header: 4 bytes (`class_and_version` + 3 bytes bit field) + 4 bytes size = 8
    bytes::ensure_len(data, 0, 8)?;

    let class_and_version = data[0];
    let class_id = class_and_version & 0x0F;
    let version = (class_and_version >> 4) & 0x0F;

    // 24-bit class bit field (little-endian)
    let bf0 = data[1];
    let bf1 = data[2];
    let bf2 = data[3];
    let _bit_field_24 = (bf0 as u32) | ((bf1 as u32) << 8) | ((bf2 as u32) << 16);

    let size = LittleEndian::read_u32(&data[4..8]);
    let mut pos = 8;

    let parsed = match class_id {
        FIXED_POINT_CLASS => {
            // Fixed-Point
            bytes::ensure_len(data, pos, 4)?;
            let byte_order = if bf0 & 0x01 == 0 {
                DatatypeByteOrder::LittleEndian
            } else {
                DatatypeByteOrder::BigEndian
            };
            let signed = (bf0 >> 3) & 0x01 == 1;
            let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
            let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
            pos += 4;
            Ok((
                Datatype::FixedPoint(
                    FixedPointType::new(size, byte_order, signed, bit_offset, bit_precision)?
                        .with_padding(bf0 & LOW_PADDING_BIT != 0, bf0 & HIGH_PADDING_BIT != 0),
                ),
                pos,
            ))
        }
        FLOATING_POINT_CLASS => {
            // Floating-Point
            bytes::ensure_len(data, pos, 12)?;
            let byte_order = parse_float_byte_order(bf0);
            let mantissa_normalization = match (bf0 >> FLOAT_NORMALIZATION_SHIFT)
                & FLOAT_NORMALIZATION_MASK
            {
                FLOAT_NORMALIZATION_UNNORMALIZED => MantissaNormalization::Unnormalized,
                FLOAT_NORMALIZATION_LEADING_ONE => MantissaNormalization::LeadingOne,
                FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE => MantissaNormalization::ImpliedLeadingOne,
                value => return Err(FormatError::InvalidMantissaNormalization(value)),
            };
            let sign_location = bf1;
            let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
            let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
            let exponent_location = data[pos + 4];
            let exponent_size = data[pos + 5];
            let mantissa_location = data[pos + 6];
            let mantissa_size = data[pos + 7];
            let exponent_bias = LittleEndian::read_u32(&data[pos + 8..pos + 12]);
            pos += 12;
            Ok((
                Datatype::FloatingPoint(
                    FloatingPointType::new(
                        size,
                        byte_order,
                        mantissa_normalization,
                        sign_location,
                        bit_offset,
                        bit_precision,
                        exponent_location,
                        exponent_size,
                        mantissa_location,
                        mantissa_size,
                        exponent_bias,
                    )?
                    .with_padding(
                        bf0 & LOW_PADDING_BIT != 0,
                        bf0 & HIGH_PADDING_BIT != 0,
                        bf0 & FLOAT_INTERNAL_PADDING_BIT != 0,
                    ),
                ),
                pos,
            ))
        }
        TIME_CLASS => {
            // Time
            bytes::ensure_len(data, pos, 2)?;
            let byte_order = if bf0 & 0x01 == 0 {
                DatatypeByteOrder::LittleEndian
            } else {
                DatatypeByteOrder::BigEndian
            };
            let bit_precision = LittleEndian::read_u16(&data[pos..pos + 2]);
            pos += 2;
            Ok((
                Datatype::Time(TimeType::new(size, byte_order, bit_precision)?),
                pos,
            ))
        }
        STRING_CLASS => {
            // String
            let padding_val = bf0 & 0x0F;
            let charset_val = (bf0 >> 4) & 0x0F;
            let padding = parse_string_padding(padding_val)?;
            let charset = parse_charset(charset_val)?;
            Ok((
                Datatype::String {
                    size,
                    padding,
                    charset,
                },
                pos,
            ))
        }
        BIT_FIELD_CLASS => {
            // Bit Field
            bytes::ensure_len(data, pos, 4)?;
            let byte_order = if bf0 & 0x01 == 0 {
                DatatypeByteOrder::LittleEndian
            } else {
                DatatypeByteOrder::BigEndian
            };
            let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
            let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
            pos += 4;
            Ok((
                Datatype::BitField(
                    BitFieldType::new(size, byte_order, bit_offset, bit_precision)?
                        .with_padding(bf0 & LOW_PADDING_BIT != 0, bf0 & HIGH_PADDING_BIT != 0),
                ),
                pos,
            ))
        }
        OPAQUE_CLASS => {
            // Opaque
            let tag_len = bf0 as usize;
            bytes::ensure_len(data, pos, tag_len)?;
            let tag = data[pos..pos + tag_len].to_vec();
            // Tags are padded to multiple of 8 bytes
            let padded = (tag_len + 7) & !7;
            let pos = 8 + padded; // from start of properties
            Ok((Datatype::Opaque { size, tag }, pos))
        }
        COMPOUND_CLASS => {
            // Compound
            let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
            let mut members = Vec::with_capacity(num_members as usize);

            // Versions 3 to 5 share one layout: version 4 added the revised
            // reference types and version 5 the complex class, and neither
            // changed a compound. HDF5 2.0 writes version 5 for every
            // datatype under its latest library bounds.
            if (DATATYPE_VERSION_THREE..=DATATYPE_VERSION_FIVE).contains(&version) {
                let ob = offset_bytes_for_size(size);
                for _ in 0..num_members {
                    let (name, name_len) = read_null_terminated_string(data, pos)?;
                    pos += name_len;
                    let byte_offset = read_uint(data, pos, ob)?;
                    pos += ob;
                    let (member_dt, consumed) = parse_datatype(&data[pos..])?;
                    pos += consumed;
                    members.push(__private::compound_member(name, byte_offset, member_dt));
                }
            } else if version == DATATYPE_VERSION_ONE || version == DATATYPE_VERSION_TWO {
                // v1 and v2: the member name is NUL-terminated and padded with
                // additional NULs to a multiple of 8 bytes, followed by a
                // 4-byte member byte offset.
                //
                // v1 then carries a fixed 28-byte dimension block before the member datatype message:
                //  dimensionality(1) + reserved(3) + dimension permutation(4) + reserved(4) + dimension sizes(16)
                // v2 drops that block.
                for _ in 0..num_members {
                    let (name, name_len) = read_null_terminated_string(data, pos)?;
                    let padded = (name_len + 7) & !7;
                    pos += padded;
                    bytes::ensure_len(data, pos, 4)?;
                    let byte_offset = LittleEndian::read_u32(&data[pos..pos + 4]) as u64;
                    pos += 4;
                    let shape = if version == DATATYPE_VERSION_ONE {
                        let (shape, consumed) = legacy_member::MemberShape::parse(data, pos)?;
                        pos += consumed;
                        Some(shape)
                    } else {
                        None
                    };
                    let (member_dt, consumed) = parse_datatype(&data[pos..])?;
                    pos += consumed;
                    let member_dt = match shape {
                        Some(shape) => shape.into_datatype(member_dt, &name, byte_offset, size)?,
                        None => member_dt,
                    };
                    members.push(__private::compound_member(name, byte_offset, member_dt));
                }
            } else {
                return Err(FormatError::InvalidDatatypeVersion {
                    class: class_id,
                    version,
                });
            }

            Ok((Datatype::Compound { size, members }, pos))
        }
        REFERENCE_CLASS => {
            // Reference
            let ref_type_val = bf0 & 0x0F;
            let ref_type = match ref_type_val {
                0 => ReferenceType::Object,
                1 => ReferenceType::DatasetRegion,
                _ => return Err(FormatError::InvalidReferenceType(ref_type_val)),
            };
            Ok((Datatype::Reference { size, ref_type }, pos))
        }
        ENUMERATION_CLASS => {
            // Enumeration
            let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
            // Parse base type
            let (base_type, base_consumed) = parse_datatype(&data[pos..])?;
            pos += base_consumed;
            let base_size = base_type.type_size();
            let mut members = Vec::with_capacity(num_members as usize);
            // Enum layout: `base_type`, then all names (null-terminated), then all values
            // v1/v2: names are padded to 8-byte boundaries
            // v3: names are just null-terminated
            let mut member_names = Vec::with_capacity(num_members as usize);
            for _ in 0..num_members {
                let (name, name_len) = read_null_terminated_string(data, pos)?;
                if version < DATATYPE_VERSION_THREE {
                    let padded = (name_len + 7) & !7;
                    pos += padded;
                } else {
                    pos += name_len;
                }
                member_names.push(name);
            }
            // Now values
            for name in &member_names {
                bytes::ensure_len(data, pos, base_size as usize)?;
                let value = data[pos..pos + base_size as usize].to_vec();
                pos += base_size as usize;
                members.push(__private::enum_member(name.clone(), value));
            }
            Ok((
                Datatype::Enumeration {
                    size,
                    base_type: Box::new(base_type),
                    members,
                },
                pos,
            ))
        }
        VARIABLE_LENGTH_CLASS => {
            let kind = parse_variable_length_metadata([bf0, bf1, bf2])?;
            let (base_type, consumed) = parse_datatype(&data[pos..])?;
            pos += consumed;
            let kind = match kind {
                VariableLengthMetadata::Sequence => VariableLengthType::sequence(base_type),
                VariableLengthMetadata::String { padding, charset } => VariableLengthType::String(
                    VariableLengthStringType::with_base_type(padding, charset, base_type)?,
                ),
            };
            Ok((Datatype::VariableLength(kind), pos))
        }

        ARRAY_CLASS => {
            // Array
            if version == DATATYPE_VERSION_TWO {
                bytes::ensure_len(data, pos, 4)?;
                let ndims = data[pos] as usize;
                pos += 4; // ndims(1) + reserved(3)
                bytes::ensure_len(data, pos, ndims * 4 + ndims * 4)?;
                let mut dimensions = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    dimensions.push(LittleEndian::read_u32(&data[pos..pos + 4]));
                    pos += 4;
                }
                // skip permutation indices
                pos += ndims * 4;
                let (base_type, consumed) = parse_datatype(&data[pos..])?;
                pos += consumed;
                Ok((
                    Datatype::Array {
                        base_type: Box::new(base_type),
                        dimensions,
                    },
                    pos,
                ))
            } else if (DATATYPE_VERSION_THREE..=DATATYPE_VERSION_FIVE).contains(&version) {
                bytes::ensure_len(data, pos, 1)?;
                let ndims = data[pos] as usize;
                pos += 1;
                bytes::ensure_len(data, pos, ndims * 4)?;
                let mut dimensions = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    dimensions.push(LittleEndian::read_u32(&data[pos..pos + 4]));
                    pos += 4;
                }
                let (base_type, consumed) = parse_datatype(&data[pos..])?;
                pos += consumed;
                Ok((
                    Datatype::Array {
                        base_type: Box::new(base_type),
                        dimensions,
                    },
                    pos,
                ))
            } else {
                Err(FormatError::InvalidDatatypeVersion {
                    class: class_id,
                    version,
                })
            }
        }
        COMPLEX_CLASS => parse_complex(data, version, [bf0, bf1, bf2], size),
        _ => Err(FormatError::InvalidDatatypeClass(class_id)),
    };

    // Array types compute their size from their base type and dimensions. A zero dimension
    // produces an element with zero bytes even when the header size is positive.
    let (datatype, consumed) = parsed?;
    datatype.element_size()?;
    Ok((datatype, consumed))
}

/// Serializes a datatype into an HDF5 Datatype message body.
///
/// Writes [`Datatype::Complex`] as class 11 in version 5, with one floating-point parent.
/// Writes each containing datatype at a message version at least as high as its children's.
pub fn serialize_datatype(datatype: &Datatype) -> Vec<u8> {
    match datatype {
        Datatype::FixedPoint(datatype) => {
            let layout = datatype.layout();
            let mut bf0 = padding_bits(layout.low_padding(), layout.high_padding());
            if datatype.byte_order() == DatatypeByteOrder::BigEndian {
                bf0 |= 0x01;
            }
            if layout.signed() {
                bf0 |= 0x08;
            }
            let mut buf = build_header(
                FIXED_POINT_CLASS,
                DATATYPE_VERSION_ONE,
                [bf0, 0, 0],
                datatype.size().get(),
            );
            buf.extend_from_slice(&layout.bit_offset().to_le_bytes());
            buf.extend_from_slice(&layout.bit_precision().to_le_bytes());
            buf
        }
        Datatype::FloatingPoint(datatype) => {
            let layout = datatype.layout();
            let normalization = match layout.mantissa_normalization() {
                MantissaNormalization::Unnormalized => FLOAT_NORMALIZATION_UNNORMALIZED,
                MantissaNormalization::LeadingOne => FLOAT_NORMALIZATION_LEADING_ONE,
                MantissaNormalization::ImpliedLeadingOne => FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE,
            };
            let mut bf0 = (normalization << FLOAT_NORMALIZATION_SHIFT)
                | padding_bits(layout.low_padding(), layout.high_padding());
            if layout.internal_padding() {
                bf0 |= FLOAT_INTERNAL_PADDING_BIT;
            }
            let version = match datatype.byte_order() {
                DatatypeByteOrder::LittleEndian => DATATYPE_VERSION_ONE,
                DatatypeByteOrder::BigEndian => {
                    bf0 |= FLOAT_BYTE_ORDER_LOW_BIT;
                    DATATYPE_VERSION_ONE
                }
                DatatypeByteOrder::Vax => {
                    bf0 |= FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT;
                    DATATYPE_VERSION_THREE
                }
            };
            let mut buf = build_header(
                FLOATING_POINT_CLASS,
                version,
                [bf0, layout.sign_location(), 0],
                datatype.size().get(),
            );
            buf.extend_from_slice(&layout.bit_offset().to_le_bytes());
            buf.extend_from_slice(&layout.bit_precision().to_le_bytes());
            buf.push(layout.exponent_location());
            buf.push(layout.exponent_size());
            buf.push(layout.mantissa_location());
            buf.push(layout.mantissa_size());
            buf.extend_from_slice(&layout.exponent_bias().to_le_bytes());
            buf
        }
        Datatype::Complex(complex) => {
            let mut buf = build_header(
                COMPLEX_CLASS,
                DATATYPE_VERSION_FIVE,
                [COMPLEX_HOMOGENEOUS_BIT, 0, 0],
                complex.size().get(),
            );
            append_datatype(&mut buf, &Datatype::FloatingPoint(complex.component()));
            buf
        }
        Datatype::String {
            size,
            padding,
            charset,
        } => {
            let pad_val = match padding {
                StringPadding::NullTerminate => 0,
                StringPadding::NullPad => 1,
                StringPadding::SpacePad => 2,
            };
            let cs_val = match charset {
                CharacterSet::Ascii => 0,
                CharacterSet::Utf8 => 1,
            };
            let bf0 = pad_val | (cs_val << 4);
            build_header(STRING_CLASS, DATATYPE_VERSION_ONE, [bf0, 0, 0], *size)
        }
        Datatype::VariableLength(kind) => {
            let (bf0, bf1) = match kind {
                VariableLengthType::Sequence { .. } => (VARIABLE_LENGTH_SEQUENCE, 0),
                VariableLengthType::String(string) => {
                    let padding = match string.padding() {
                        StringPadding::NullTerminate => 0,
                        StringPadding::NullPad => 1,
                        StringPadding::SpacePad => 2,
                    };
                    let charset = match string.charset() {
                        CharacterSet::Ascii => 0,
                        CharacterSet::Utf8 => 1,
                    };
                    (
                        VARIABLE_LENGTH_STRING | (padding << VARIABLE_LENGTH_PADDING_SHIFT),
                        charset,
                    )
                }
            };
            let mut buf = build_header(
                VARIABLE_LENGTH_CLASS,
                DATATYPE_VERSION_ONE,
                [bf0, bf1, 0],
                16,
            );
            append_datatype(&mut buf, kind.base_type());
            buf
        }
        Datatype::Compound { size, members } => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "compound member count is written into the 2-byte member-count field of the datatype message"
            )]
            let num = members.len() as u16;
            let bf0 = (num & 0xFF) as u8;
            let bf1 = ((num >> 8) & 0xFF) as u8;
            let mut buf =
                build_header(COMPOUND_CLASS, DATATYPE_VERSION_THREE, [bf0, bf1, 0], *size);
            let ob = offset_bytes_for_size(*size);
            for m in members {
                // Null-terminated name
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
                // Byte offset (variable-width)
                buf.extend_from_slice(&m.byte_offset.to_le_bytes()[..ob]);
                // Recursively serialize member datatype
                append_datatype(&mut buf, &m.datatype);
            }
            buf
        }
        Datatype::Enumeration {
            size,
            base_type,
            members,
        } => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "enumeration member count is written into the 2-byte member-count field of the datatype message"
            )]
            let num = members.len() as u16;
            let bf0 = (num & 0xFF) as u8;
            let bf1 = ((num >> 8) & 0xFF) as u8;
            let mut buf = build_header(
                ENUMERATION_CLASS,
                DATATYPE_VERSION_THREE,
                [bf0, bf1, 0],
                *size,
            );
            // Base type
            append_datatype(&mut buf, base_type);
            // All names (null-terminated)
            for m in members {
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
            }
            // All values
            for m in members {
                buf.extend_from_slice(&m.value);
            }
            buf
        }
        Datatype::Array {
            base_type,
            dimensions,
        } => {
            let mut buf = build_header(
                ARRAY_CLASS,
                DATATYPE_VERSION_THREE,
                [0, 0, 0],
                datatype.type_size(),
            );
            #[expect(
                clippy::cast_possible_truncation,
                reason = "array rank is written into the 1-byte dimensionality field; HDF5 caps array rank well below 255"
            )]
            buf.push(dimensions.len() as u8);
            for &d in dimensions {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            append_datatype(&mut buf, base_type);
            buf
        }
        Datatype::Reference { size, ref_type } => {
            let bf0 = match ref_type {
                ReferenceType::Object => 0,
                ReferenceType::DatasetRegion => 1,
                #[expect(
                    clippy::unreachable,
                    reason = "the version-locked core crate defines only the two matched reference kinds"
                )]
                _ => unreachable!("core reference kind lacks an HDF5 encoder"),
            };
            build_header(REFERENCE_CLASS, DATATYPE_VERSION_ONE, [bf0, 0, 0], *size)
        }
        Datatype::Time(datatype) => {
            let bf0 = u8::from(datatype.byte_order() == DatatypeByteOrder::BigEndian);
            let mut buf = build_header(
                TIME_CLASS,
                DATATYPE_VERSION_ONE,
                [bf0, 0, 0],
                datatype.size().get(),
            );
            buf.extend_from_slice(&datatype.bit_precision().get().to_le_bytes());
            buf
        }
        Datatype::BitField(datatype) => {
            let bf0 = padding_bits(datatype.low_padding(), datatype.high_padding())
                | u8::from(datatype.byte_order() == DatatypeByteOrder::BigEndian);
            let mut buf = build_header(
                BIT_FIELD_CLASS,
                DATATYPE_VERSION_ONE,
                [bf0, 0, 0],
                datatype.size().get(),
            );
            buf.extend_from_slice(&datatype.bit_offset().to_le_bytes());
            buf.extend_from_slice(&datatype.bit_precision().get().to_le_bytes());
            buf
        }
        Datatype::Opaque { size, tag } => {
            // `bf0` carries the ASCII tag length the tag is padded with zero
            // bytes to a multiple of 8, mirroring `parse`.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "opaque tag length is written into the 1-byte tag-length bit field (bf0)"
            )]
            let bf0 = tag.len() as u8;
            let mut buf = build_header(OPAQUE_CLASS, DATATYPE_VERSION_ONE, [bf0, 0, 0], *size);
            buf.extend_from_slice(tag);
            let padded = (tag.len() + 7) & !7;
            buf.resize(buf.len() + (padded - tag.len()), 0);
            buf
        }
        #[expect(
            clippy::unreachable,
            reason = "the version-locked core crate defines only the matched datatype variants"
        )]
        _ => unreachable!("core datatype variant lacks an HDF5 encoder"),
    }
}

// `H5O__dtype_decode_helper` checks that a container's datatype version is at least its
// highest child version (H5Odtype.c, HDF5 2.2.0).
fn append_datatype(buf: &mut Vec<u8>, datatype: &Datatype) {
    let encoded = serialize_datatype(datatype);
    let version = (buf[0] >> DATATYPE_VERSION_SHIFT).max(encoded[0] >> DATATYPE_VERSION_SHIFT);
    buf[0] = (buf[0] & DATATYPE_CLASS_MASK) | (version << DATATYPE_VERSION_SHIFT);
    buf.extend_from_slice(&encoded);
}

fn parse_complex(
    data: &[u8],
    version: u8,
    flags: [u8; 3],
    size: u32,
) -> Result<(Datatype, usize), FormatError> {
    if version != DATATYPE_VERSION_FIVE {
        return Err(FormatError::InvalidDatatypeVersion {
            class: COMPLEX_CLASS,
            version,
        });
    }
    let homogeneous = flags[0] & COMPLEX_HOMOGENEOUS_BIT != 0;
    let form = (flags[0] & COMPLEX_FORM_MASK) >> COMPLEX_FORM_SHIFT;
    let flags = u32::from(flags[0]) | (u32::from(flags[1]) << 8) | (u32::from(flags[2]) << 16);
    if flags & COMPLEX_RESERVED_MASK != 0 {
        return Err(FormatError::InvalidComplexFlags(flags));
    }
    if !homogeneous || form != COMPLEX_RECTANGULAR_FORM {
        return Err(FormatError::UnsupportedComplexForm { homogeneous, form });
    }
    bytes::ensure_len(data, DATATYPE_HEADER_SIZE, DATATYPE_HEADER_SIZE)?;
    let parent = &data[DATATYPE_HEADER_SIZE..];
    let class = parent[0] & DATATYPE_CLASS_MASK;
    if class != FLOATING_POINT_CLASS {
        return Err(FormatError::InvalidComplexParent { class });
    }
    let version = parent[0] >> DATATYPE_VERSION_SHIFT;
    if !(DATATYPE_VERSION_ONE..=DATATYPE_VERSION_FIVE).contains(&version) {
        return Err(FormatError::InvalidDatatypeVersion { class, version });
    }
    let flags = u32::from(parent[1]) | (u32::from(parent[2]) << 8) | (u32::from(parent[3]) << 16);
    let has_extended_order = flags & u32::from(FLOAT_BYTE_ORDER_HIGH_BIT) != 0;
    if flags & FLOAT_RESERVED_MASK != 0
        || (has_extended_order
            && (version < DATATYPE_VERSION_THREE
                || flags & u32::from(FLOAT_BYTE_ORDER_LOW_BIT) == 0))
    {
        return Err(FormatError::InvalidComplexParentFlags { version, flags });
    }
    let (parent, consumed) = parse_datatype(parent)?;
    bytes::ensure_len(data, DATATYPE_HEADER_SIZE, consumed)?;
    let Datatype::FloatingPoint(component) = parent else {
        return Err(FormatError::InvalidComplexParent { class });
    };
    let complex = ComplexType::new(component)?;
    if size != complex.size().get() {
        return Err(FormatError::InvalidComplexSize {
            size,
            component_size: component.size().get(),
        });
    }
    Ok((Datatype::Complex(complex), DATATYPE_HEADER_SIZE + consumed))
}

/// The byte length of the common datatype message header.
///
/// The header for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_HEADER_SIZE: usize = 8;
/// The mask for the class code in the common datatype message header.
///
/// The class field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_CLASS_MASK: u8 = 0x0F;
/// The bit position of the version in the common datatype message header.
///
/// The version field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_SHIFT: u8 = 4;
/// The homogeneous flag in a native complex datatype's class bit field.
///
/// The flag for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_HOMOGENEOUS_BIT: u8 = 0x01;
/// The mask for the form in a native complex datatype's class bit field.
///
/// The field for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_FORM_MASK: u8 = 0x06;
/// The bit position of the form in a native complex datatype's class bit field.
///
/// The field for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_FORM_SHIFT: u8 = 1;
/// The rectangular form code of a native complex datatype.
///
/// The code for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_RECTANGULAR_FORM: u8 = 0;
/// The mask for reserved bits 3 through 23 of a complex class bit field.
///
/// The bits for datatype message version 5 are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_RESERVED_MASK: u32 = 0xFF_FFF8;
/// The mask for reserved bits 7 and 16 through 23 of a floating-point class bit field.
///
/// The bits for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_RESERVED_MASK: u32 = 0xFF_0080;

fn padding_bits(low: bool, high: bool) -> u8 {
    (if low { LOW_PADDING_BIT } else { 0 }) | (if high { HIGH_PADDING_BIT } else { 0 })
}

/// Bit 1 of the fixed-point, floating-point, and bitfield class bit fields, the low padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const LOW_PADDING_BIT: u8 = 0x02;
/// Bit 2 of the fixed-point, floating-point, and bitfield class bit fields, the high padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const HIGH_PADDING_BIT: u8 = 0x04;
/// Bit 3 of the floating-point class bit field, the internal padding bit.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_INTERNAL_PADDING_BIT: u8 = 0x08;

fn build_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
    buf[1] = bf[0];
    buf[2] = bf[1];
    buf[3] = bf[2];
    buf[4..8].copy_from_slice(&size.to_le_bytes());
    buf
}

enum VariableLengthMetadata {
    Sequence,
    String {
        padding: StringPadding,
        charset: CharacterSet,
    },
}

fn parse_variable_length_metadata(flags: [u8; 3]) -> Result<VariableLengthMetadata, FormatError> {
    let [bf0, bf1, bf2] = flags;
    let subtype = bf0 & VARIABLE_LENGTH_FIELD_MASK;
    if !matches!(subtype, VARIABLE_LENGTH_SEQUENCE | VARIABLE_LENGTH_STRING) {
        return Err(FormatError::InvalidVariableLengthSubtype(subtype));
    }
    let bits = u32::from_le_bytes([bf0, bf1, bf2, 0]) & VARIABLE_LENGTH_RESERVED_MASK;
    if bits != 0 {
        return Err(FormatError::InvalidVariableLengthReservedBits(bits));
    }
    let padding = bf0 >> VARIABLE_LENGTH_PADDING_SHIFT;
    let charset = bf1 & VARIABLE_LENGTH_FIELD_MASK;
    if subtype == VARIABLE_LENGTH_STRING {
        return Ok(VariableLengthMetadata::String {
            padding: parse_string_padding(padding)?,
            charset: parse_charset(charset)?,
        });
    }
    if padding != 0 || charset != 0 {
        return Err(FormatError::InvalidVariableLengthSequenceMetadata { padding, charset });
    }
    Ok(VariableLengthMetadata::Sequence)
}

/// The mask for a four-bit variable-length subtype or character-set code.
///
/// The fields for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_FIELD_MASK: u8 = 0x0F;
/// The variable-length sequence subtype code.
///
/// The code for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_SEQUENCE: u8 = 0;
/// The variable-length string subtype code.
///
/// The code for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_STRING: u8 = 1;
/// The bit position of string padding in the variable-length class bit field.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_PADDING_SHIFT: u8 = 4;
/// The mask for reserved bits 12 through 23 of the variable-length class bit field.
///
/// The bits for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_RESERVED_MASK: u32 = 0xFF_F000;

fn parse_string_padding(val: u8) -> Result<StringPadding, FormatError> {
    match val {
        0 => Ok(StringPadding::NullTerminate),
        1 => Ok(StringPadding::NullPad),
        2 => Ok(StringPadding::SpacePad),
        _ => Err(FormatError::InvalidStringPadding(val)),
    }
}

/// Returns the byte order a floating-point datatype's class bit field encodes.
///
/// Bit 6 and bit 0 select it together: neither set is little-endian, bit 0 alone big-endian, and
/// both VAX. The reserved pattern with bit 6 alone also parses as VAX.
///
/// The bit field is defined in "The Datatype Message" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn parse_float_byte_order(bit_field: u8) -> DatatypeByteOrder {
    match (
        bit_field & FLOAT_BYTE_ORDER_HIGH_BIT != 0,
        bit_field & FLOAT_BYTE_ORDER_LOW_BIT != 0,
    ) {
        (false, false) => DatatypeByteOrder::LittleEndian,
        (false, true) => DatatypeByteOrder::BigEndian,
        (true, _) => DatatypeByteOrder::Vax,
    }
}

/// Bit 0 of a floating-point class bit field, the low bit of its byte order.
const FLOAT_BYTE_ORDER_LOW_BIT: u8 = 0x01;

/// Bit 6 of a floating-point class bit field, the high bit of its byte order.
const FLOAT_BYTE_ORDER_HIGH_BIT: u8 = 0x40;

/// The bit position of mantissa normalization in the floating-point class bit field.
///
/// The field and its values are defined in "The Datatype Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_SHIFT: u8 = 4;
/// The mask for the two mantissa normalization bits after shifting.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_MASK: u8 = 0x03;
/// The encoded value for a mantissa stored without normalization.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_UNNORMALIZED: u8 = 0;
/// The encoded value for a stored leading mantissa bit that is set, except for zero.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_LEADING_ONE: u8 = 1;
/// The encoded value for an implied leading mantissa bit.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE: u8 = 2;
/// The reserved mantissa normalization value.
///
/// Mantissa normalization is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[cfg(test)]
const FLOAT_NORMALIZATION_RESERVED: u8 = 3;

fn parse_charset(val: u8) -> Result<CharacterSet, FormatError> {
    match val {
        0 => Ok(CharacterSet::Ascii),
        1 => Ok(CharacterSet::Utf8),
        _ => Err(FormatError::InvalidCharacterSet(val)),
    }
}

/// Read a null-terminated string from `data` starting at `offset`.
/// Returns (`string`, `bytes_consumed` including the null terminator).
fn read_null_terminated_string(data: &[u8], offset: usize) -> Result<(String, usize), FormatError> {
    if offset >= data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: offset + 1,
            available: data.len(),
        });
    }
    let remaining = &data[offset..];
    let null_pos = remaining
        .iter()
        .position(|&b| b == 0)
        .ok_or(FormatError::UnexpectedEof {
            expected: offset + 1,
            available: data.len(),
        })?;
    let name = String::from_utf8_lossy(&remaining[..null_pos]).into_owned();
    Ok((name, null_pos + 1))
}

/// Returns the member offset width for a version 3 compound datatype of `compound_size` bytes.
fn offset_bytes_for_size(compound_size: u32) -> usize {
    // "The Datatype Message" in the format specification, version 4.0, defines the
    // version 3 member offset width as the minimum byte count that holds the datatype size.
    compound_size
        .to_le_bytes()
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(1, |index| index + 1)
}

/// Reads a little-endian unsigned integer of 1, 2, 3, 4, or 8 bytes.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the integer extends beyond `data` or `nbytes`
/// is not a supported width.
fn read_uint(data: &[u8], offset: usize, nbytes: usize) -> Result<u64, FormatError> {
    bytes::ensure_len(data, offset, nbytes)?;
    let slice = &data[offset..offset + nbytes];
    Ok(match nbytes {
        1 => slice[0] as u64,
        2 => LittleEndian::read_u16(slice) as u64,
        3 => LittleEndian::read_uint(slice, 3),
        4 => LittleEndian::read_u32(slice) as u64,
        8 => LittleEndian::read_u64(slice),
        _ => {
            return Err(FormatError::UnexpectedEof {
                expected: offset + nbytes,
                available: data.len(),
            });
        }
    })
}

/// Returns `true` if a datatype of this encoded class can hold an object
/// address, based only on the first byte of a datatype message.
///
/// The check is necessary for [`datatype_holds_object_address`] but not
/// sufficient. A compound of two integers passes it, so the caller must not
/// treat a `true` result as proof that an address is present. A walk over every
/// object in a file calls this to skip the common cases, such as fixed-point,
/// floating-point, string, and opaque datasets, without allocating a parsed
/// [`Datatype`] for each.
///
/// The admitted classes match the ones that `datatype_holds_object_address`
/// recurses through, plus the reference class. The test
/// `class_predicate_admits_every_reference_holding_type` enforces that match.
pub fn class_may_hold_object_address(class_and_version: u8) -> bool {
    matches!(
        class_and_version & 0x0F,
        COMPOUND_CLASS | REFERENCE_CLASS | ENUMERATION_CLASS | VARIABLE_LENGTH_CLASS | ARRAY_CLASS
    )
}

/// Datatype message class ids, as the low nibble of a datatype message's first
/// byte. Only the classes that can carry an object address downward are named.
// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, assigns these datatype class and version codes.
const FIXED_POINT_CLASS: u8 = 0;
const FLOATING_POINT_CLASS: u8 = 1;
const TIME_CLASS: u8 = 2;
const STRING_CLASS: u8 = 3;
const BIT_FIELD_CLASS: u8 = 4;
const OPAQUE_CLASS: u8 = 5;
const COMPOUND_CLASS: u8 = 6;
const REFERENCE_CLASS: u8 = 7;
const ENUMERATION_CLASS: u8 = 8;
const VARIABLE_LENGTH_CLASS: u8 = 9;
const ARRAY_CLASS: u8 = 10;
const COMPLEX_CLASS: u8 = 11;
const DATATYPE_VERSION_ONE: u8 = 1;
const DATATYPE_VERSION_TWO: u8 = 2;
const DATATYPE_VERSION_THREE: u8 = 3;
const DATATYPE_VERSION_FIVE: u8 = 5;

/// Returns `true` if `dt` contains an object address anywhere in its structure.
///
/// An object address is an object reference or a dataset-region reference. This
/// function searches compound members, array entries, enumeration bases, and
/// variable-length contents.
///
/// [`embedded_reference_slots`] maps only the 8-byte form reachable through
/// compound members and array entries. An address outside that form cannot be
/// read or freed, so the caller must reject the datatype that carries it. The
/// caller's fall-through arm consults this function so the two cannot drift
/// apart.
///
/// A variable-length datatype counts only when its contents are an object
/// reference. A deletion frees object headers and dataset storage, and a global
/// heap collection survives, so a variable-length string keeps pointing at data
/// that is still present. A `H5T_VLEN` of `H5T_STD_REF_OBJ`, which the
/// reference library writes, keeps its addresses in the heap contents, where the
/// element bytes hold only a heap id.
pub fn datatype_holds_object_address(dt: &Datatype) -> bool {
    match dt {
        Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_object_address(&m.datatype)),
        Datatype::Array { base_type, .. } | Datatype::Enumeration { base_type, .. } => {
            datatype_holds_object_address(base_type)
        }
        Datatype::VariableLength(kind) => datatype_holds_object_address(kind.base_type()),
        Datatype::Complex(_) => false,
        _ => false,
    }
}

/// Returns `true` if an element of `dt` holds a file address at any depth.
///
/// A variable-length element and a dataset region reference each hold the address of a global
/// heap collection and an index into it, and an object reference holds the address of an object
/// header. The search descends into compound members, array elements, and enumeration bases.
///
/// The function does not report which kind of address an element holds, or where.
/// [`datatype_holds_object_address`] checks for a reference alone, and returns `false` for a
/// variable-length string.
pub fn datatype_holds_file_address(dt: &Datatype) -> bool {
    match dt {
        Datatype::VariableLength(_) | Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_file_address(&m.datatype)),
        Datatype::Array { base_type, .. } | Datatype::Enumeration { base_type, .. } => {
            datatype_holds_file_address(base_type)
        }
        Datatype::Complex(_) => false,
        _ => false,
    }
}

/// Returns the byte offset of every 8-byte object reference `datatype` contains
/// in a compound member or array entry, in declaration order.
///
/// A datatype that is itself an object reference produces a slot at offset 0,
/// so a caller that handles that case separately must test for it first.
///
/// The counterpart of `embedded_vlen_slots`,
/// which locates the variable-length heap addresses a rewrite invalidates.
///
/// Returns `None` when a computed offset exceeds the datatype's declared element
/// size, or when the element contains an object reference whose location this
/// walker cannot report (see [`datatype_holds_object_address`]). The caller must
/// not treat `None` as an empty slot list. An empty list is safe to rewrite in
/// place, while `None` means an address may be present with no known location,
/// which the caller must reject.
pub fn embedded_reference_slots(datatype: &Datatype) -> Option<Vec<usize>> {
    /// Returns `false` when the datatype cannot be walked on this target, for the
    /// reasons [`embedded_vlen_slots`]' walker documents.
    fn collect(datatype: &Datatype, base: usize, capacity: usize, out: &mut Vec<usize>) -> bool {
        if out.len() > capacity {
            return true;
        }
        match datatype {
            Datatype::Reference {
                ref_type: ReferenceType::Object,
                size: 8,
            } => {
                out.push(base);
                true
            }
            Datatype::Compound { members, .. } => {
                for m in members {
                    let Some(at) = usize::try_from(m.byte_offset)
                        .ok()
                        .and_then(|off| base.checked_add(off))
                    else {
                        return false;
                    };
                    if !collect(&m.datatype, at, capacity, out) {
                        return false;
                    }
                }
                true
            }
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                // As in `embedded_vlen_slots`: probe once so that entries which can
                // never contribute do not drive a walk over huge declared
                // dimensions, and so every iteration below pushes at least one slot.
                // Walked once and translated per entry, for the reason
                // `embedded_vlen_slots` documents: re-walking is exponential in
                // nesting depth.
                let mut probe = Vec::new();
                if !collect(base_type, 0, capacity, &mut probe) {
                    return false;
                }
                if probe.is_empty() {
                    return true;
                }
                let count = dimensions
                    .iter()
                    .copied()
                    .fold(1u64, |a, b| a.saturating_mul(u64::from(b)));
                // As in `embedded_vlen_slots`: more entries than the element has
                // room for cannot fit, so reject without walking them.
                if count > capacity as u64 {
                    return false;
                }
                let entries = usize::try_from(count).unwrap_or(usize::MAX);
                let stride = base_type.type_size() as usize;
                for i in 0..entries {
                    let Some(at) = i.checked_mul(stride).and_then(|off| base.checked_add(off))
                    else {
                        return false;
                    };
                    for &slot in &probe {
                        let Some(off) = at.checked_add(slot) else {
                            return false;
                        };
                        out.push(off);
                        if out.len() > capacity {
                            return true;
                        }
                    }
                }
                true
            }
            Datatype::Complex(_) => true,
            // The walker maps none of these variants. A datatype here can still contain
            // an object reference, through a width other than 8, an enumeration over one,
            // or a variable-length sequence of them. Its addresses exist with no location
            // the walker can report, so the caller must not treat a `false` result as an
            // empty slot list. Consulting the predicate keeps the pair in sync as either
            // grows.
            _ => !datatype_holds_object_address(datatype),
        }
    }

    let element_size = datatype.type_size() as usize;
    let capacity = element_size / 8;
    let mut slots = Vec::new();
    if !collect(datatype, 0, capacity, &mut slots) {
        return None;
    }
    // `checked_add`: an offset near the top of the address space would otherwise
    // wrap here and read as "fits".
    if slots.len() > capacity
        || slots
            .iter()
            .any(|&s| s.checked_add(8).is_none_or(|end| end > element_size))
    {
        return None;
    }
    Some(slots)
}

/// Returns the byte offset within `raw` of each 8-byte object reference, paired
/// with the address stored there.
///
/// `slots` is the result of [`embedded_reference_slots`] for the datatype `raw`
/// holds elements of, and `element_size` is its `type_size`. This function is
/// the single implementation of the slot scan, shared by callers that screen
/// addresses against what a commit vacates and callers that rewrite them to a
/// new location. A second implementation could disagree about the element
/// stride, about a trailing partial element, or about which slots exist.
///
/// A trailing run shorter than one element is skipped. `chunks_exact` yields
/// whole elements only, matching what every reader of these bytes does with a
/// truncated tail.
pub fn stored_object_references<'a>(
    raw: &'a [u8],
    element_size: usize,
    slots: &'a [usize],
) -> impl Iterator<Item = (usize, u64)> + 'a {
    raw.chunks_exact(element_size.max(1))
        .enumerate()
        .flat_map(move |(i, element)| {
            slots.iter().map(move |&at| {
                let stored = u64::from_le_bytes(element[at..at + 8].try_into().expect(
                    "embedded_reference_slots keeps every slot 8 bytes inside the element",
                ));
                (i * element_size + at, stored)
            })
        })
}

/// Build a datatype header (8 bytes) for testing.
#[cfg(test)]
fn build_dt_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
    buf[1] = bf[0];
    buf[2] = bf[1];
    buf[3] = bf[2];
    LittleEndian::write_u32(&mut buf[4..8], size);
    buf
}

#[cfg(test)]
mod tests;
