use alloc::boxed::Box;
use alloc::vec;

use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, FloatingPointLayout, FloatingPointType, FormatError,
};

use super::super::header;
use super::super::message::version::DATATYPE_VERSION_THREE;
use super::super::semantic;
use super::super::semantic::DecodedDatatype;
use super::fixtures;

#[test]
fn a_version_5_array_parses_like_version_3() {
    let array = |version: u8| {
        let mut buf = header::build_datatype_header(10, version, [0, 0, 0], 48);
        buf.push(2);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&fixtures::build_fixed_point(4, false, true, 0, 32));
        buf
    };
    let (decoded, expected_len) = DecodedDatatype::parse_prefix(&array(3)).unwrap();
    let expected = decoded.into_datatype();
    let (decoded, len) = DecodedDatatype::parse_prefix(&array(5)).unwrap();
    let parsed = decoded.into_datatype();
    assert_eq!((parsed, len), (expected, expected_len));
}

#[test]
fn parses_array_dimensions_and_base_type() {
    let mut buf = header::build_datatype_header(10, DATATYPE_VERSION_THREE, [0, 0, 0], 48);
    buf.push(2); // Rank
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&fixtures::build_fixed_point(4, false, true, 0, 32));

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    match dt {
        Datatype::Array {
            base_type,
            dimensions,
        } => {
            assert_eq!(dimensions, vec![3, 4]);
            match *base_type {
                Datatype::FixedPoint(numeric)
                    if numeric.size().get() == 4 && numeric.layout().signed() => {}
                other => panic!("expected i32, got {other:?}"),
            }
        }
        _ => panic!("expected Array"),
    }
}

#[test]
fn rejects_a_zero_dimension_with_a_nonzero_declared_size() {
    let mut buf = header::build_datatype_header(10, DATATYPE_VERSION_THREE, [0, 0, 0], 48);
    buf.push(2); // Rank
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&fixtures::build_fixed_point(4, false, true, 0, 32));

    assert_eq!(
        DecodedDatatype::parse_prefix(&buf).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 10 }
    );
}

#[test]
fn serialize_parse_array_roundtrip() {
    let dt = Datatype::Array {
        base_type: Box::new(Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64,
            )
            .unwrap(),
        )),
        dimensions: vec![3],
    };
    let bytes = semantic::serialize_datatype(&dt).unwrap();
    let parsed = DecodedDatatype::from_record(&bytes)
        .unwrap()
        .into_datatype();
    assert_eq!(parsed, dt);
}
