use crate::attribute::AttributeMessage;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::{Error, FormatError};
use crate::shared_message::DatatypeLocation;
use crate::source::BytesSource;
use crate::type_builders;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

use super::*;

#[rstest]
#[case::signed_scalar(type_builders::make_i16_type(), vec![], 2, 0)]
#[case::unsigned_scalar(type_builders::make_u16_type(), vec![], 2, 0)]
#[case::binary32_scalar(type_builders::make_f32_type(), vec![], 4, 0)]
#[case::binary64_scalar(type_builders::make_f64_type(), vec![], 8, 0)]
#[case::signed_element(type_builders::make_i16_type(), vec![0], 0, 1)]
#[case::unsigned_element(type_builders::make_u16_type(), vec![0], 0, 1)]
#[case::binary32_element(type_builders::make_f32_type(), vec![0], 0, 1)]
#[case::binary64_element(type_builders::make_f64_type(), vec![0], 0, 1)]
#[case::fixed_string(Datatype::String { size: 2, padding: StringPadding::NullPad, charset: CharacterSet::Utf8 }, vec![0], 0, 1)]
fn supported_decoding_errors_keep_their_original_fields(
    #[case] datatype: Datatype,
    #[case] raw: Vec<u8>,
    #[case] expected: usize,
    #[case] actual: usize,
) {
    let attr = attribute(datatype, raw, DataspaceType::Scalar);
    let error =
        decode_attr_value(&attr, &BytesSource::new(Vec::new()), widths(), addresses()).unwrap_err();
    assert_eq!(error, FormatError::DataSizeMismatch { expected, actual });
}

#[test]
fn a_truncated_variable_length_descriptor_reports_its_required_extent() {
    let attr = attribute(
        Datatype::VariableLength(VariableLengthType::String(VariableLengthStringType::new(
            StringPadding::NullTerminate,
            CharacterSet::Utf8,
        ))),
        vec![0; 15],
        DataspaceType::Scalar,
    );
    let error =
        decode_attr_value(&attr, &BytesSource::new(Vec::new()), widths(), addresses()).unwrap_err();
    assert_eq!(
        error,
        FormatError::UnexpectedEof {
            expected: 16,
            available: 15
        }
    );
}

#[rstest]
#[case::signed(type_builders::make_i16_type(), AttrValue::I16Array(vec![]))]
#[case::unsigned(type_builders::make_u16_type(), AttrValue::U16Array(vec![]))]
#[case::binary32(type_builders::make_f32_type(), AttrValue::F32Array(vec![]))]
#[case::binary64(type_builders::make_f64_type(), AttrValue::F64Array(vec![]))]
#[case::variable_length_utf8(type_builders::make_vlen_string_type(CharacterSet::Utf8), AttrValue::VarLenStringArray(vec![]))]
#[case::variable_length_ascii(type_builders::make_vlen_string_type(CharacterSet::Ascii), AttrValue::VarLenAsciiStringArray(vec![]))]
#[case::fixed_string(Datatype::String { size: 2, padding: StringPadding::NullPad, charset: CharacterSet::Utf8 }, AttrValue::string_array_sized(vec![], 2).unwrap())]
fn empty_arrays_remain_decoded_values(
    #[case] datatype: Datatype,
    #[case] value: AttrValue,
    #[values(DataspaceType::Simple, DataspaceType::Null)] space: DataspaceType,
) {
    let attr = attribute(datatype, vec![], space);
    assert_eq!(
        decode_attr_value(&attr, &BytesSource::new(Vec::new()), widths(), addresses()).unwrap(),
        DecodedAttrValue::Value(value)
    );
}

#[rstest]
#[case::opaque(Datatype::Opaque { size: 1, tag: vec![] })]
#[case::wide_integer(Datatype::FixedPoint(FixedPointType::new(16, DatatypeByteOrder::LittleEndian, false, 0, 128).unwrap()))]
#[case::wide_float(Datatype::FloatingPoint(FloatingPointType::from_layout(16, DatatypeByteOrder::LittleEndian, FloatingPointLayout::IEEE754_BINARY64).unwrap()))]
fn unsupported_representations_are_distinct_from_decoding_errors(#[case] datatype: Datatype) {
    let attr = attribute(datatype, vec![0; 16], DataspaceType::Scalar);
    assert_eq!(
        decode_attr_value(&attr, &BytesSource::new(Vec::new()), widths(), addresses()).unwrap(),
        DecodedAttrValue::Unsupported
    );
}

#[rstest]
#[case::numeric(type_builders::make_i16_type(), vec![], FormatError::DataSizeMismatch { expected: 2, actual: 0 })]
#[case::fixed_string(Datatype::String { size: 2, padding: StringPadding::NullPad, charset: CharacterSet::Utf8 }, vec![0], FormatError::DataSizeMismatch { expected: 0, actual: 1 })]
#[case::variable_length(type_builders::make_vlen_string_type(CharacterSet::Utf8), vec![0; 15], FormatError::UnexpectedEof { expected: 16, available: 15 })]
fn attribute_map_decoding_returns_the_original_error_with_context(
    #[case] datatype: Datatype,
    #[case] raw: Vec<u8>,
    #[case] expected: FormatError,
    #[values(false, true)] valid_first: bool,
) {
    let valid = attribute(
        type_builders::make_i16_type(),
        vec![7, 0],
        DataspaceType::Scalar,
    );
    let mut invalid = attribute(datatype, raw, DataspaceType::Scalar);
    invalid.name = "invalid".into();
    let attrs = if valid_first {
        vec![valid, invalid]
    } else {
        vec![invalid, valid]
    };
    let error =
        attrs_to_map(&attrs, &BytesSource::new(Vec::new()), widths(), addresses()).unwrap_err();
    let Error::AttributeDecode { name, source } = error else {
        panic!("expected AttributeDecode, got {error:?}");
    };
    assert_eq!(name, "invalid");
    assert_eq!(source, expected);
}

#[rstest]
#[case::signed_scalar(vec![128], true, AttrValue::I64(128))]
#[case::signed_array(vec![1, 128], false, AttrValue::I64Array(vec![1, 128]))]
fn signed_integer_narrowing_keeps_the_wider_fallback(
    #[case] values: Vec<i64>,
    #[case] scalar: bool,
    #[case] expected: AttrValue,
) {
    assert_eq!(signed_attr_value(values, scalar, 1).unwrap(), expected);
}

#[rstest]
#[case::unsigned_scalar(vec![256], true, AttrValue::U64(256))]
#[case::unsigned_array(vec![1, 256], false, AttrValue::U64Array(vec![1, 256]))]
fn unsigned_integer_narrowing_keeps_the_wider_fallback(
    #[case] values: Vec<u64>,
    #[case] scalar: bool,
    #[case] expected: AttrValue,
) {
    assert_eq!(unsigned_attr_value(values, scalar, 1).unwrap(), expected);
}

fn attribute(datatype: Datatype, raw_data: Vec<u8>, space_type: DataspaceType) -> AttributeMessage {
    let simple = space_type == DataspaceType::Simple;
    AttributeMessage {
        name: "value".into(),
        datatype,
        dataspace: Dataspace {
            space_type,
            rank: u8::from(simple),
            dimensions: if simple { vec![0] } else { vec![] },
            max_dimensions: None,
        },
        raw_data,
        datatype_location: DatatypeLocation::Inline,
    }
}

fn widths() -> FormatWidths {
    FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    }
}

fn addresses() -> AddressContext {
    AddressContext::from_effective_base(BaseAddress::ZERO)
}
