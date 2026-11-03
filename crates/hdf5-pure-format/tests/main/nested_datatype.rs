use hdf5_pure_core::__private;
use hdf5_pure_core::Datatype;
use hdf5_pure_core::DatatypeByteOrder;
use hdf5_pure_core::FixedPointType;
use hdf5_pure_core::FormatError;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::dataspace;
use test_util::datatype;
use test_util::datatype::Class;

#[rstest]
#[case::empty(0)]
#[case::maximum(247)]
fn inline_nested_datatype_preserves_attribute_bytes(
    #[case] tag_len: usize,
    #[values(2, 3)] version: u8,
) {
    let attribute = attribute(Datatype::Array {
        base_type: Box::new(Datatype::Opaque {
            size: 1,
            tag: vec![b'x'; tag_len],
        }),
        dimensions: vec![2],
    });
    let mut expected_type = datatype::header(Class::ARRAY, 3, [0; 3], 2);
    expected_type.extend_from_slice(&[1, 2, 0, 0, 0]);
    let property_len = if tag_len == 0 { 0 } else { 248 };
    expected_type.extend(datatype::header(Class::OPAQUE, 1, [property_len, 0, 0], 1));
    expected_type.extend_from_slice(&vec![b'x'; tag_len]);
    expected_type.resize(21 + usize::from(property_len), 0);
    let mut expected = vec![version, 0, 2, 0];
    expected.extend_from_slice(&u16::try_from(expected_type.len()).unwrap().to_le_bytes());
    expected.extend_from_slice(&[4, 0]);
    if version == 3 {
        expected.push(0);
    }
    expected.extend_from_slice(b"a\0");
    expected.extend(expected_type);
    expected.extend(dataspace::scalar());
    expected.extend_from_slice(&[7, 9]);
    let encoded = serialize(&attribute, version).unwrap();
    assert_eq!(encoded, expected);
    let parsed = format::AttributeMessage::parse(&encoded, 8).unwrap();
    assert_eq!(parsed, attribute);
}

#[rstest]
#[case::too_long(vec![b'x'; 248], FormatError::OpaqueTagTooLong { length: 248 })]
#[case::nul(vec![b'x', 0], FormatError::InvalidOpaqueTagByte { offset: 1, byte: 0 })]
fn inline_nested_datatype_preserves_attribute_errors(
    #[case] tag: Vec<u8>,
    #[case] expected: FormatError,
    #[values(2, 3)] version: u8,
) {
    let attribute = attribute(Datatype::Array {
        base_type: Box::new(Datatype::Opaque { size: 1, tag }),
        dimensions: vec![2],
    });
    assert_eq!(serialize(&attribute, version).unwrap_err(), expected);
}

#[rstest]
#[case::fits(1_500)]
#[case::exceeds(3_000)]
fn nested_datatype_retains_attribute_field_length_checks(#[case] count: usize) {
    let datatype = Datatype::Compound {
        size: 16_000,
        members: (0..count)
            .map(|index| {
                __private::compound_member(
                    format!("f{index:04}"),
                    u64::try_from(index * 4).unwrap(),
                    Datatype::Array {
                        base_type: Box::new(
                            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32)
                                .map(Datatype::FixedPoint)
                                .unwrap(),
                        ),
                        dimensions: vec![1],
                    },
                )
            })
            .collect(),
    };
    let mut attribute = attribute(datatype);
    attribute.dataspace.space_type = format::DataspaceType::Null;
    attribute.raw_data.clear();
    let length = 8 + count * (6 + 2 + 13 + 12);
    let result = serialize(&attribute, 2);
    if length <= usize::from(u16::MAX) {
        let encoded = result.unwrap();
        assert_eq!(
            usize::from(u16::from_le_bytes([encoded[4], encoded[5]])),
            length
        );
    } else {
        assert_eq!(
            result.unwrap_err(),
            FormatError::AttributeFieldTooLong {
                name: "a".into(),
                field: "datatype",
                size: length,
                limit: usize::from(u16::MAX),
            }
        );
    }
}

fn serialize(attribute: &format::AttributeMessage, version: u8) -> Result<Vec<u8>, FormatError> {
    if version == 3 {
        attribute.serialize_v3(format::OffsetWidth::Eight, 8)
    } else {
        attribute.serialize(format::OffsetWidth::Eight, 8)
    }
}

fn attribute(datatype: Datatype) -> format::AttributeMessage {
    format::AttributeMessage {
        name: "a".into(),
        datatype,
        dataspace: format::Dataspace {
            rank: 0,
            space_type: format::DataspaceType::Scalar,
            dimensions: Vec::new(),
            max_dimensions: None,
        },
        raw_data: vec![7, 9],
        datatype_location: format::DatatypeLocation::Inline,
    }
}
