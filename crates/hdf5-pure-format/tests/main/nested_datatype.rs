use hdf5_pure_core::CompoundMember;
use hdf5_pure_core::Datatype;
use hdf5_pure_core::DatatypeByteOrder;
use hdf5_pure_core::DatatypeMemberName;
use hdf5_pure_core::FixedPointType;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::ReferenceType;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::DecodedDatatype;
use rstest::rstest;
use test_util::dataspace;
use test_util::datatype;
use test_util::datatype::Class;
use test_util::datatype::containers::{self, CompoundField, MemberOffset};

#[test]
fn object_reference_slots_scan_complete_elements_in_order() {
    let datatype = Datatype::Compound {
        size: 10,
        members: vec![CompoundMember::new(
            DatatypeMemberName::from(c"target"),
            1,
            Datatype::Reference {
                size: 8,
                ref_type: ReferenceType::Object,
            },
        )],
    };
    let first = 0x1234_5678_9abc_def0_u64;
    let second = 0x9876_5432_10fe_dcba_u64;
    let mut bytes = Vec::new();
    for address in [first, second] {
        bytes.push(0xff);
        bytes.extend_from_slice(&address.to_le_bytes());
        bytes.push(0xff);
    }
    bytes.extend_from_slice(&[0xff; 9]);

    let slots = format::embedded_reference_slots(&datatype).unwrap();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots.element_size(), 10);
    assert_eq!(
        slots
            .stored_references(&bytes)
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        [(1, first), (11, second)]
    );
}

#[test]
fn nested_reference_slots_preserve_member_and_array_order() {
    let reference = datatype::object_reference(8);
    let pair = containers::compound(
        3,
        [2, 0, 0],
        24,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"last",
                offset: 16,
                datatype: &reference,
            },
            CompoundField {
                name: b"first",
                offset: 0,
                datatype: &reference,
            },
        ],
    );
    let inner = datatype::array(3, [0; 3], 48, 1, &[2], &pair);
    let outer = datatype::array(3, [0; 3], 96, 1, &[2], &inner);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&outer).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(consumed, outer.len());

    let slots = format::embedded_reference_slots(&datatype).unwrap();
    assert_eq!(slots.element_size(), 96);
    assert_eq!(slots.len(), 8);

    let mut raw = vec![0xf0; 2 * 96 + 95];
    let positions = [
        16, 0, 40, 24, 64, 48, 88, 72, 112, 96, 136, 120, 160, 144, 184, 168,
    ];
    let expected: Vec<(usize, u64)> = positions.into_iter().zip(1..=16).collect();
    for &(offset, value) in &expected {
        raw[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    assert_eq!(
        slots
            .stored_references(&raw)
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::nested_member_outside_its_record(4)]
#[case::offset_overflow(u64::MAX)]
fn reference_slot_must_fit_its_own_record(#[case] offset: u64) {
    let datatype = Datatype::Compound {
        size: 32,
        members: vec![CompoundMember::new(
            DatatypeMemberName::from(c"inner"),
            0,
            Datatype::Compound {
                size: 8,
                members: vec![CompoundMember::new(
                    DatatypeMemberName::from(c"r"),
                    offset,
                    Datatype::Reference {
                        size: 8,
                        ref_type: ReferenceType::Object,
                    },
                )],
            },
        )],
    };
    assert!(format::embedded_reference_slots(&datatype).is_none());
}

#[test]
fn reference_array_size_must_fit_the_datatype_size_field() {
    let datatype = Datatype::Array {
        base_type: Box::new(Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        }),
        dimensions: vec![u32::MAX / 8 + 1],
    };
    assert!(format::embedded_reference_slots(&datatype).is_none());
}

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
                CompoundMember::new(
                    DatatypeMemberName::new(format!("f{index:04}")).unwrap(),
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
    attribute.dataspace = format::Dataspace::null();
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
        dataspace: format::Dataspace::scalar(),
        raw_data: vec![7, 9],
        datatype_location: format::DatatypeLocation::Inline,
    }
}
