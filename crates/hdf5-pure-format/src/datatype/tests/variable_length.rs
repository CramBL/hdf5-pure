use hdf5_pure_core::{
    CharacterSet, Datatype, DatatypeByteOrder, FixedPointType, FormatError, StringPadding,
    VariableLengthStringType, VariableLengthType,
};
use rstest::rstest;

use super::super::header::{self, OPAQUE_CLASS, STRING_CLASS, VARIABLE_LENGTH_CLASS};
use super::super::message::version::DATATYPE_VERSION_ONE;
use super::super::semantic;
use super::super::semantic::DecodedDatatype;
use super::super::variable_length::{
    VARIABLE_LENGTH_PADDING_SHIFT, VARIABLE_LENGTH_SEQUENCE, VARIABLE_LENGTH_STRING,
};
use super::fixtures;

#[rstest]
#[case::reserved_subtype_2(2)]
#[case::reserved_subtype_3(3)]
#[case::reserved_subtype_4(4)]
#[case::reserved_subtype_5(5)]
#[case::reserved_subtype_6(6)]
#[case::reserved_subtype_7(7)]
#[case::reserved_subtype_8(8)]
#[case::reserved_subtype_9(9)]
#[case::reserved_subtype_10(10)]
#[case::reserved_subtype_11(11)]
#[case::reserved_subtype_12(12)]
#[case::reserved_subtype_13(13)]
#[case::reserved_subtype_14(14)]
#[case::reserved_subtype_15(15)]
fn rejects_reserved_variable_length_subtypes(#[case] subtype: u8) {
    let mut bytes = header::build_datatype_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [subtype, 0, 0],
        16,
    );
    bytes.extend_from_slice(&fixtures::build_fixed_point(1, false, false, 0, 8));
    assert_eq!(
        DecodedDatatype::parse_prefix(&bytes).unwrap_err(),
        FormatError::InvalidVariableLengthSubtype(subtype)
    );
}

#[rstest]
#[case::reserved_bit_12(0x00_1000)]
#[case::reserved_bit_13(0x00_2000)]
#[case::reserved_bit_14(0x00_4000)]
#[case::reserved_bit_15(0x00_8000)]
#[case::reserved_bit_16(0x01_0000)]
#[case::reserved_bit_17(0x02_0000)]
#[case::reserved_bit_18(0x04_0000)]
#[case::reserved_bit_19(0x08_0000)]
#[case::reserved_bit_20(0x10_0000)]
#[case::reserved_bit_21(0x20_0000)]
#[case::reserved_bit_22(0x40_0000)]
#[case::reserved_bit_23(0x80_0000)]
fn rejects_variable_length_reserved_bits(
    #[case] bits: u32,
    #[values(VARIABLE_LENGTH_SEQUENCE, VARIABLE_LENGTH_STRING)] subtype: u8,
) {
    let [bf0, bf1, bf2, _] = bits.to_le_bytes();
    let mut bytes = header::build_datatype_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [bf0 | subtype, bf1, bf2],
        16,
    );
    bytes.extend_from_slice(&fixtures::build_fixed_point(1, false, false, 0, 8));
    assert_eq!(
        DecodedDatatype::parse_prefix(&bytes).unwrap_err(),
        FormatError::InvalidVariableLengthReservedBits(bits)
    );
}

#[rstest]
#[case::null_pad(1, 0)]
#[case::space_pad(2, 0)]
#[case::reserved_padding(15, 0)]
#[case::utf8_charset(0, 1)]
#[case::reserved_charset(0, 15)]
#[case::null_pad_and_utf8(1, 1)]
fn rejects_string_metadata_on_variable_length_sequences(#[case] padding: u8, #[case] charset: u8) {
    let mut bytes = header::build_datatype_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [padding << VARIABLE_LENGTH_PADDING_SHIFT, charset, 0],
        16,
    );
    bytes.extend_from_slice(&fixtures::build_fixed_point(1, false, false, 0, 8));
    assert_eq!(
        DecodedDatatype::parse_prefix(&bytes).unwrap_err(),
        FormatError::InvalidVariableLengthSequenceMetadata { padding, charset }
    );
}

#[rstest]
#[case::padding_three([0x31, 0, 0], FormatError::InvalidStringPadding(3))]
#[case::padding_fifteen([0xF1, 0, 0], FormatError::InvalidStringPadding(15))]
#[case::charset_two([1, 2, 0], FormatError::InvalidCharacterSet(2))]
#[case::charset_fifteen([1, 15, 0], FormatError::InvalidCharacterSet(15))]
fn rejects_invalid_variable_length_string_metadata(
    #[case] flags: [u8; 3],
    #[case] expected: FormatError,
) {
    let mut bytes =
        header::build_datatype_header(VARIABLE_LENGTH_CLASS, DATATYPE_VERSION_ONE, flags, 16);
    bytes.extend_from_slice(&fixtures::build_fixed_point(1, false, false, 0, 8));
    assert_eq!(DecodedDatatype::parse_prefix(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::string_base(STRING_CLASS)]
#[case::opaque_base(OPAQUE_CLASS)]
fn rejects_zero_sized_variable_length_string_bases(#[case] class: u8) {
    let mut bytes = header::build_datatype_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [VARIABLE_LENGTH_STRING, 0, 0],
        16,
    );
    bytes.extend_from_slice(&header::build_datatype_header(
        class,
        DATATYPE_VERSION_ONE,
        [0, 0, 0],
        0,
    ));
    assert_eq!(
        DecodedDatatype::parse_prefix(&bytes).unwrap_err(),
        FormatError::ZeroSizedDatatype { class }
    );
}

#[rstest]
#[case::null_terminate(0, StringPadding::NullTerminate)]
#[case::null_pad(1, StringPadding::NullPad)]
#[case::space_pad(2, StringPadding::SpacePad)]
fn variable_length_strings_preserve_metadata_and_base_byte_order(
    #[case] padding_code: u8,
    #[case] padding: StringPadding,
    #[values(false, true)] utf8: bool,
    #[values(false, true)] big_endian: bool,
) {
    let mut bytes = header::build_datatype_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [
            VARIABLE_LENGTH_STRING | (padding_code << VARIABLE_LENGTH_PADDING_SHIFT),
            u8::from(utf8),
            0,
        ],
        16,
    );
    bytes.extend_from_slice(&fixtures::build_fixed_point(1, big_endian, false, 0, 8));
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(consumed, bytes.len());
    assert_eq!(
        datatype,
        Datatype::VariableLength(VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                padding,
                if utf8 {
                    CharacterSet::Utf8
                } else {
                    CharacterSet::Ascii
                },
                Datatype::FixedPoint(
                    FixedPointType::new(
                        1,
                        if big_endian {
                            DatatypeByteOrder::BigEndian
                        } else {
                            DatatypeByteOrder::LittleEndian
                        },
                        false,
                        0,
                        8
                    )
                    .unwrap()
                )
            )
            .unwrap()
        ))
    );
    assert_eq!(semantic::serialize_datatype(&datatype).unwrap(), bytes);
}

#[test]
fn nested_variable_length_sequences_roundtrip() {
    let datatype = Datatype::VariableLength(VariableLengthType::sequence(
        Datatype::VariableLength(VariableLengthType::sequence(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        ))),
    ));
    let mut expected = header::build_datatype_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [VARIABLE_LENGTH_SEQUENCE, 0, 0],
        16,
    );
    expected.extend_from_slice(&header::build_datatype_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [VARIABLE_LENGTH_SEQUENCE, 0, 0],
        16,
    ));
    expected.extend_from_slice(&fixtures::build_fixed_point(4, false, true, 0, 32));
    assert_eq!(semantic::serialize_datatype(&datatype).unwrap(), expected);
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&expected).unwrap();
    assert_eq!(
        (decoded.into_datatype(), bytes_read),
        (datatype, expected.len())
    );
}

#[test]
fn test_variable_length_string_utf8() {
    let mut buf = header::build_datatype_header(9, DATATYPE_VERSION_ONE, [0x01, 0x01, 0], 16);
    buf.extend_from_slice(&fixtures::build_fixed_point(1, false, false, 0, 8));

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    match dt {
        Datatype::VariableLength(VariableLengthType::String(string)) => {
            assert_eq!(string.padding(), &StringPadding::NullTerminate);
            assert_eq!(string.charset(), &CharacterSet::Utf8);
            assert_eq!(string.base_type().type_size(), 1);
        }
        _ => panic!("expected VariableLength"),
    }
}

#[test]
fn test_variable_length_sequence_f32() {
    let mut buf = header::build_datatype_header(9, DATATYPE_VERSION_ONE, [0x00, 0x00, 0], 16);
    buf.extend_from_slice(&fixtures::build_float(4, 23, 8, 0, 23, 127));

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    match dt {
        Datatype::VariableLength(VariableLengthType::Sequence { element_type }) => {
            assert_eq!(element_type.type_size(), 4);
        }
        _ => panic!("expected VariableLength"),
    }
}
