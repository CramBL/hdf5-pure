use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use hdf5_pure_core::{
    BitFieldType, CharacterSet, Datatype, DatatypeByteOrder, FixedPointType, FloatingPointField,
    FloatingPointLayout, FloatingPointType, FormatError, MantissaNormalization, StringPadding,
    TimeType,
};
use rstest::rstest;

use super::super::header::{
    self, BIT_FIELD_CLASS, FIXED_POINT_CLASS, FLOATING_POINT_CLASS, TIME_CLASS,
};
use super::super::message::atomic::{
    self, FLOAT_INTERNAL_PADDING_BIT, FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE,
    FLOAT_NORMALIZATION_LEADING_ONE, FLOAT_NORMALIZATION_MASK, FLOAT_NORMALIZATION_RESERVED,
    FLOAT_NORMALIZATION_SHIFT, FLOAT_NORMALIZATION_UNNORMALIZED,
};
use super::super::message::version::DATATYPE_VERSION_ONE;
use super::super::semantic;
use super::super::semantic::DecodedDatatype;
use super::fixtures;

#[rstest]
#[case::fixed_point(FIXED_POINT_CLASS)]
#[case::bit_field(BIT_FIELD_CLASS)]
fn preserves_low_and_high_padding(
    #[case] class: u8,
    #[values(false, true)] low: bool,
    #[values(false, true)] high: bool,
    #[values(false, true)] big_endian: bool,
) {
    let mut bytes = fixtures::build_fixed_point(2, big_endian, false, 3, 9);
    bytes[0] = class | (DATATYPE_VERSION_ONE << 4);
    bytes[1] |= atomic::padding_bits(low, high);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(consumed, bytes.len());
    match &datatype {
        Datatype::FixedPoint(numeric) => {
            let layout = numeric.layout();
            assert_eq!((layout.low_padding(), layout.high_padding()), (low, high));
        }
        Datatype::BitField(numeric) => {
            assert_eq!((numeric.low_padding(), numeric.high_padding()), (low, high));
        }
        other => panic!("expected an integer or bit-field datatype, got {other:?}"),
    }
    assert_eq!(semantic::serialize_datatype(&datatype).unwrap(), bytes);
}

#[rstest]
fn preserves_floating_point_padding(
    #[values(false, true)] low: bool,
    #[values(false, true)] high: bool,
    #[values(false, true)] internal: bool,
) {
    let mut bytes = fixtures::build_float(4, 23, 8, 0, 23, 127);
    bytes[1] |= atomic::padding_bits(low, high);
    if internal {
        bytes[1] |= FLOAT_INTERNAL_PADDING_BIT;
    }
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(consumed, bytes.len());
    let Datatype::FloatingPoint(numeric) = &datatype else {
        panic!("expected a floating-point datatype, got {datatype:?}");
    };
    let layout = numeric.layout();
    assert_eq!(
        (
            layout.low_padding(),
            layout.high_padding(),
            layout.internal_padding()
        ),
        (low, high, internal)
    );
    assert_eq!(semantic::serialize_datatype(&datatype).unwrap(), bytes);
}

#[test]
fn test_fixed_point_u8() {
    let data = fixtures::build_fixed_point(1, false, false, 0, 8);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&data).unwrap();
    let dt = decoded.into_datatype();
    assert_eq!(consumed, 12);
    assert_eq!(
        dt,
        Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap()
        )
    );
}

#[test]
fn test_fixed_point_i16_le() {
    let data = fixtures::build_fixed_point(2, false, true, 0, 16);
    let dt = DecodedDatatype::from_record(&data).unwrap().into_datatype();
    assert_eq!(
        dt,
        Datatype::FixedPoint(
            FixedPointType::new(2, DatatypeByteOrder::LittleEndian, true, 0, 16).unwrap()
        )
    );
}

#[test]
fn test_fixed_point_u32_be() {
    let data = fixtures::build_fixed_point(4, true, false, 0, 32);
    let dt = DecodedDatatype::from_record(&data).unwrap().into_datatype();
    match &dt {
        Datatype::FixedPoint(numeric) => {
            assert_eq!(numeric.byte_order(), DatatypeByteOrder::BigEndian);
            assert!(!numeric.layout().signed());
            assert_eq!(numeric.size().get(), 4);
        }
        _ => panic!("expected FixedPoint"),
    }
}

#[test]
fn test_fixed_point_i64_le() {
    let data = fixtures::build_fixed_point(8, false, true, 0, 64);
    let dt = DecodedDatatype::from_record(&data).unwrap().into_datatype();
    assert_eq!(
        dt,
        Datatype::FixedPoint(
            FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap()
        )
    );
}

#[rstest]
#[case::unnormalized(MantissaNormalization::Unnormalized, FLOAT_NORMALIZATION_UNNORMALIZED)]
#[case::leading_one(MantissaNormalization::LeadingOne, FLOAT_NORMALIZATION_LEADING_ONE)]
#[case::implied_leading_one(
    MantissaNormalization::ImpliedLeadingOne,
    FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE
)]
fn preserves_floating_point_class_fields(
    #[case] normalization: MantissaNormalization,
    #[case] normalization_bits: u8,
    #[values(0, 30, 31, u8::MAX)] sign_location: u8,
    #[values(
        DatatypeByteOrder::LittleEndian,
        DatatypeByteOrder::BigEndian,
        DatatypeByteOrder::Vax
    )]
    byte_order: DatatypeByteOrder,
) {
    let (size, precision, exponent_size, mantissa_location, mantissa_size) = match sign_location {
        0 => (4, 32, 8, 1, 22),
        30 => (4, 32, 7, 0, 23),
        31 => (4, 32, 8, 0, 23),
        _ => (32, 256, 8, 0, 23),
    };
    let datatype = Datatype::FloatingPoint(
        FloatingPointType::new(
            size,
            byte_order,
            normalization,
            sign_location,
            0,
            precision,
            23,
            exponent_size,
            mantissa_location,
            mantissa_size,
            127,
        )
        .unwrap(),
    );
    let bytes = semantic::serialize_datatype(&datatype).unwrap();
    assert_eq!(
        (bytes[1] >> FLOAT_NORMALIZATION_SHIFT) & FLOAT_NORMALIZATION_MASK,
        normalization_bits
    );
    assert_eq!(bytes[2], sign_location);
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    assert_eq!(
        (decoded.into_datatype(), bytes_read),
        (datatype, bytes.len())
    );
}

#[test]
fn test_float_f32_le() {
    let data = fixtures::build_float(4, 23, 8, 0, 23, 127);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&data).unwrap();
    let dt = decoded.into_datatype();
    assert_eq!(consumed, 20);
    assert_eq!(
        dt,
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                4,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY32
            )
            .unwrap()
        )
    );
}

#[test]
fn test_float_f64_le() {
    let data = fixtures::build_float(8, 52, 11, 0, 52, 1023);
    let dt = DecodedDatatype::from_record(&data).unwrap().into_datatype();
    assert_eq!(
        dt,
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64
            )
            .unwrap()
        )
    );
}

#[test]
fn test_string_null_terminated_ascii() {
    let buf = header::build_datatype_header(3, DATATYPE_VERSION_ONE, [0x00, 0, 0], 10);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&buf).unwrap();
    let dt = decoded.into_datatype();
    assert_eq!(consumed, 8);
    assert_eq!(
        dt,
        Datatype::String {
            size: 10,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        }
    );
}

#[test]
fn test_string_space_padded_utf8() {
    let buf = header::build_datatype_header(3, DATATYPE_VERSION_ONE, [0x12, 0, 0], 32);
    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    assert_eq!(
        dt,
        Datatype::String {
            size: 32,
            padding: StringPadding::SpacePad,
            charset: CharacterSet::Utf8,
        }
    );
}

#[test]
fn parses_an_opaque_tag_without_padding() {
    let mut buf = header::build_datatype_header(5, DATATYPE_VERSION_ONE, [8, 0, 0], 64);
    buf.extend_from_slice(b"BLOB");
    buf.extend_from_slice(&[0, 0, 0, 0]);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&buf).unwrap();
    let dt = decoded.into_datatype();
    assert_eq!(consumed, 16);
    assert_eq!(
        dt,
        Datatype::Opaque {
            size: 64,
            tag: b"BLOB".to_vec(),
        }
    );
}

#[test]
fn parses_a_little_endian_bit_field() {
    let mut buf = header::build_datatype_header(4, DATATYPE_VERSION_ONE, [0, 0, 0], 2);
    let mut props = [0u8; 4];
    LittleEndian::write_u16(&mut props[0..2], 0);
    LittleEndian::write_u16(&mut props[2..4], 16);
    buf.extend_from_slice(&props);

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    assert_eq!(
        dt,
        Datatype::BitField(BitFieldType::new(2, DatatypeByteOrder::LittleEndian, 0, 16).unwrap())
    );
}

#[test]
fn parses_little_endian_time_precision() {
    let mut buf = header::build_datatype_header(2, DATATYPE_VERSION_ONE, [0, 0, 0], 8);
    let mut props = [0u8; 2];
    LittleEndian::write_u16(&mut props[0..2], 64);
    buf.extend_from_slice(&props);

    let (decoded, consumed) = DecodedDatatype::parse_prefix(&buf).unwrap();
    let dt = decoded.into_datatype();
    assert_eq!(consumed, 10);
    assert_eq!(
        dt,
        Datatype::Time(TimeType::new(8, DatatypeByteOrder::LittleEndian, 64).unwrap())
    );
}

#[test]
fn test_time_byte_order_roundtrips() {
    for (be, order) in [
        (0u8, DatatypeByteOrder::LittleEndian),
        (1u8, DatatypeByteOrder::BigEndian),
    ] {
        let mut buf = header::build_datatype_header(2, DATATYPE_VERSION_ONE, [be, 0, 0], 4);
        buf.extend_from_slice(&32u16.to_le_bytes());
        let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
        assert_eq!(dt, Datatype::Time(TimeType::new(4, order, 32).unwrap()));
        let reparsed = DecodedDatatype::from_record(&semantic::serialize_datatype(&dt).unwrap())
            .unwrap()
            .into_datatype();
        assert_eq!(reparsed, dt);
    }
}

#[test]
fn test_error_invalid_string_padding() {
    let buf = header::build_datatype_header(3, DATATYPE_VERSION_ONE, [0x03, 0, 0], 10);
    let err = DecodedDatatype::parse_prefix(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidStringPadding(3));
}

#[test]
fn test_error_invalid_charset() {
    let buf = header::build_datatype_header(3, DATATYPE_VERSION_ONE, [0x20, 0, 0], 10);
    let err = DecodedDatatype::parse_prefix(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidCharacterSet(2));
}

#[test]
fn serialize_parse_time_roundtrip() {
    let dt = Datatype::Time(TimeType::new(8, DatatypeByteOrder::LittleEndian, 64).unwrap());
    let bytes = semantic::serialize_datatype(&dt).unwrap();
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    let parsed = decoded.into_datatype();
    assert_eq!(parsed, dt);
    assert_eq!(consumed, bytes.len());
}

#[test]
fn serialize_parse_bitfield_roundtrip() {
    for byte_order in [
        DatatypeByteOrder::LittleEndian,
        DatatypeByteOrder::BigEndian,
    ] {
        let dt = Datatype::BitField(BitFieldType::new(4, byte_order, 3, 17).unwrap());
        let bytes = semantic::serialize_datatype(&dt).unwrap();
        let (decoded, consumed) = DecodedDatatype::parse_prefix(&bytes).unwrap();
        let parsed = decoded.into_datatype();
        assert_eq!(parsed, dt);
        assert_eq!(consumed, bytes.len());
    }
}

#[rstest]
#[case::zero_size(0, 0, 8, FormatError::ZeroSizedDatatype { class: FIXED_POINT_CLASS })]
#[case::zero_precision(1, 0, 0, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 0, bit_precision: 0 })]
#[case::offset_at_end(1, 8, 1, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 8, bit_precision: 1 })]
#[case::precision_past_end(1, 4, 5, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 4, bit_precision: 5 })]
#[case::maximum_bit_fields(8, u16::MAX, u16::MAX, FormatError::InvalidFixedPointRange { size: 8, bit_offset: u16::MAX, bit_precision: u16::MAX })]
fn rejects_invalid_fixed_point_metadata(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[case] expected: FormatError,
) {
    assert_eq!(
        DecodedDatatype::parse_prefix(&fixtures::build_fixed_point(
            size, false, false, offset, precision
        ))
        .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::zero_size(4, 0, FormatError::ZeroSizedDatatype { class: FLOATING_POINT_CLASS })]
#[case::shifted_full_precision(8, 1, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 1, bit_precision: 32 })]
#[case::zero_precision(10, 0, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 0, bit_precision: 0 })]
#[case::precision_past_size(10, 33, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 0, bit_precision: 33 })]
#[case::sign_at_end(2, 32, FormatError::InvalidFloatingPointField { field: FloatingPointField::Sign, location: 32, size: 1, bit_offset: 0, bit_precision: 32 })]
#[case::zero_exponent_size(13, 0, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 0, bit_offset: 0, bit_precision: 32 })]
#[case::exponent_past_end(13, 10, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 10, bit_offset: 0, bit_precision: 32 })]
#[case::zero_mantissa_size(15, 0, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 0, size: 0, bit_offset: 0, bit_precision: 32 })]
#[case::mantissa_past_end(14, 250, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 250, size: 23, bit_offset: 0, bit_precision: 32 })]
fn rejects_invalid_floating_point_metadata(
    #[case] index: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
) {
    let mut bytes = fixtures::build_float(4, 23, 8, 0, 23, 127);
    bytes[index] = value;
    assert_eq!(DecodedDatatype::parse_prefix(&bytes).unwrap_err(), expected);
}

#[test]
fn rejects_reserved_mantissa_normalization() {
    let mut bytes = fixtures::build_float(4, 23, 8, 0, 23, 127);
    bytes[1] = FLOAT_NORMALIZATION_RESERVED << FLOAT_NORMALIZATION_SHIFT;
    assert_eq!(
        DecodedDatatype::parse_prefix(&bytes).unwrap_err(),
        FormatError::InvalidMantissaNormalization(FLOAT_NORMALIZATION_RESERVED)
    );
}

#[rstest]
#[case::sign_exponent(
    2,
    23,
    FloatingPointField::Sign,
    23,
    1,
    FloatingPointField::Exponent,
    23,
    8
)]
#[case::sign_mantissa(
    2,
    0,
    FloatingPointField::Sign,
    0,
    1,
    FloatingPointField::Mantissa,
    0,
    23
)]
#[case::exponent_mantissa(
    15,
    24,
    FloatingPointField::Exponent,
    23,
    8,
    FloatingPointField::Mantissa,
    0,
    24
)]
fn rejects_overlapping_floating_point_metadata(
    #[case] index: usize,
    #[case] value: u8,
    #[case] first: FloatingPointField,
    #[case] first_location: u8,
    #[case] first_size: u8,
    #[case] second: FloatingPointField,
    #[case] second_location: u8,
    #[case] second_size: u8,
) {
    let mut bytes = fixtures::build_float(4, 23, 8, 0, 23, 127);
    bytes[index] = value;
    assert_eq!(
        DecodedDatatype::parse_prefix(&bytes).unwrap_err(),
        FormatError::OverlappingFloatingPointFields {
            first,
            first_location,
            first_size,
            second,
            second_location,
            second_size
        }
    );
}

#[rstest]
#[case::binary32(
    FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian),
    fixtures::build_float(4, 23, 8, 0, 23, 127)
)]
#[case::binary64(
    FloatingPointType::ieee754_binary64(DatatypeByteOrder::LittleEndian),
    fixtures::build_float(8, 52, 11, 0, 52, 1023)
)]
fn preserves_ieee_bytes(#[case] numeric: FloatingPointType, #[case] expected: Vec<u8>) {
    let datatype = Datatype::FloatingPoint(numeric);
    let encoded = semantic::serialize_datatype(&datatype).unwrap();
    assert_eq!(encoded, expected);
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&encoded).unwrap();
    assert_eq!(
        (decoded.into_datatype(), bytes_read),
        (datatype, encoded.len())
    );
}

#[rstest]
#[case::sixteen_bytes(16)]
#[case::u32_limit(u32::MAX)]
fn parses_wide_numeric_metadata(#[case] size: u32) {
    let fixed = Datatype::FixedPoint(
        FixedPointType::new(size, DatatypeByteOrder::BigEndian, true, 1, 127).unwrap(),
    );
    let encoded = semantic::serialize_datatype(&fixed).unwrap();
    assert_eq!(
        DecodedDatatype::from_record(&encoded)
            .unwrap()
            .into_datatype(),
        fixed
    );
    let floating = Datatype::FloatingPoint(
        FloatingPointType::from_layout(
            size,
            DatatypeByteOrder::Vax,
            FloatingPointLayout::IEEE754_BINARY64,
        )
        .unwrap(),
    );
    let encoded = semantic::serialize_datatype(&floating).unwrap();
    assert_eq!(
        DecodedDatatype::from_record(&encoded)
            .unwrap()
            .into_datatype(),
        floating
    );
}

#[rstest]
#[case::zero_size(0, 1, FormatError::ZeroSizedDatatype { class: TIME_CLASS })]
#[case::zero_precision(1, 0, FormatError::InvalidNumericBitRange { class: TIME_CLASS, size: 1, bit_offset: 0, bit_precision: 0 })]
#[case::precision_past_end(1, 9, FormatError::InvalidNumericBitRange { class: TIME_CLASS, size: 1, bit_offset: 0, bit_precision: 9 })]
fn rejects_invalid_time_metadata(
    #[case] size: u32,
    #[case] precision: u16,
    #[case] expected: FormatError,
) {
    let mut bytes =
        header::build_datatype_header(TIME_CLASS, DATATYPE_VERSION_ONE, [0, 0, 0], size);
    bytes.extend_from_slice(&precision.to_le_bytes());
    assert_eq!(DecodedDatatype::parse_prefix(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::zero_size(0, 0, 1, FormatError::ZeroSizedDatatype { class: BIT_FIELD_CLASS })]
#[case::zero_precision(1, 0, 0, FormatError::InvalidNumericBitRange { class: BIT_FIELD_CLASS, size: 1, bit_offset: 0, bit_precision: 0 })]
#[case::range_past_end(1, 4, 5, FormatError::InvalidNumericBitRange { class: BIT_FIELD_CLASS, size: 1, bit_offset: 4, bit_precision: 5 })]
fn rejects_invalid_bit_field_metadata(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[case] expected: FormatError,
) {
    let mut bytes =
        header::build_datatype_header(BIT_FIELD_CLASS, DATATYPE_VERSION_ONE, [0, 0, 0], size);
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.extend_from_slice(&precision.to_le_bytes());
    assert_eq!(DecodedDatatype::parse_prefix(&bytes).unwrap_err(), expected);
}
