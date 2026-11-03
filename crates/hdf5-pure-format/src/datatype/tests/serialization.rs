use super::*;

#[rstest]
#[case::fixed_unsigned(build_fixed_point(1, false, false, 0, 8))]
#[case::fixed_signed_big_endian(build_fixed_point(4, true, true, 3, 25))]
#[case::float_binary32(build_float(4, 23, 8, 0, 23, 127))]
#[case::float_binary64_big_endian(big_endian_float())]
#[case::float_vax(vax_float())]
#[case::time_little_endian(time_bytes(0))]
#[case::time_big_endian(time_bytes(1))]
#[case::string_ascii(build_dt_header(STRING_CLASS, DATATYPE_VERSION_ONE, [0, 0, 0], 7))]
#[case::string_utf8_null_pad(build_dt_header(STRING_CLASS, DATATYPE_VERSION_ONE, [0x11, 0, 0], 7))]
#[case::string_utf8_space_pad(build_dt_header(STRING_CLASS, DATATYPE_VERSION_ONE, [0x12, 0, 0], 7))]
#[case::bitfield(bitfield_bytes())]
#[case::opaque_empty(build_dt_header(OPAQUE_CLASS, DATATYPE_VERSION_ONE, [0, 0, 0], 4))]
#[case::opaque_non_ascii(opaque_bytes())]
#[case::compound_empty(build_dt_header(COMPOUND_CLASS, DATATYPE_VERSION_THREE, [0, 0, 0], 4))]
#[case::object_reference(build_dt_header(REFERENCE_CLASS, DATATYPE_VERSION_ONE, [0, 0, 0], 8))]
#[case::region_reference(build_dt_header(REFERENCE_CLASS, DATATYPE_VERSION_ONE, [1, 0, 0], 12))]
#[case::enumeration_empty(enum_bytes(0))]
#[case::enumeration_three_hundred(enum_bytes(300))]
#[case::variable_sequence(sequence_bytes(
    build_fixed_point(1, false, false, 0, 8),
    DATATYPE_VERSION_ONE
))]
#[case::variable_string(variable_string_bytes())]
#[case::array(array_bytes(build_fixed_point(1, false, false, 0, 8), &[2, 3], 6, DATATYPE_VERSION_THREE))]
#[case::complex(complex_bytes())]
#[case::mixed_containers(mixed_bytes())]
#[case::vax_sequence(sequence_bytes(vax_float(), DATATYPE_VERSION_THREE))]
#[case::vax_array_sequence(sequence_bytes(array_bytes(vax_float(), &[3], 12, DATATYPE_VERSION_THREE), DATATYPE_VERSION_THREE))]
fn canonical_messages_preserve_bytes_and_extent(#[case] expected: Vec<u8>) {
    let (datatype, consumed) = parse_datatype(&expected).unwrap();
    assert_eq!(consumed, expected.len());
    assert_eq!(serialize_datatype(&datatype).unwrap(), expected);
    let mut following = expected.clone();
    following.extend_from_slice(b"next message");
    assert_eq!(
        parse_datatype(&following).unwrap(),
        (datatype, expected.len())
    );
}

#[rstest]
#[case::one_byte_limit(255, 1)]
#[case::two_byte_start(256, 2)]
#[case::two_byte_limit(65_535, 2)]
#[case::three_byte_start(65_536, 3)]
fn canonical_compound_array_offsets(#[case] size: u32, #[case] width: usize) {
    let datatype = Datatype::Compound {
        size,
        members: vec![__private::compound_member(
            "last".into(),
            u64::from(size - 2),
            Datatype::Array {
                base_type: Box::new(byte_member_type()),
                dimensions: vec![2],
            },
        )],
    };
    let mut expected = build_dt_header(COMPOUND_CLASS, DATATYPE_VERSION_THREE, [1, 0, 0], size);
    expected.extend_from_slice(b"last\0");
    expected.extend_from_slice(&(size - 2).to_le_bytes()[..width]);
    expected.extend_from_slice(&array_bytes(
        build_fixed_point(1, false, false, 0, 8),
        &[2],
        2,
        DATATYPE_VERSION_THREE,
    ));
    assert_eq!(serialize_datatype(&datatype).unwrap(), expected);
    assert_eq!(
        parse_datatype(&expected).unwrap(),
        (datatype, expected.len())
    );
}

#[rstest]
#[case::ordinary(vec![2, 3], 6)]
#[case::empty_rank(vec![], 1)]
#[case::zero_extent(vec![0, 4], 0)]
#[case::saturated(vec![u32::MAX, 2], u32::MAX)]
#[case::zero_after_saturation(vec![u32::MAX, 2, 0], 0)]
fn canonical_array_sizes_preserve_constructed_models(
    #[case] dimensions: Vec<u32>,
    #[case] size: u32,
) {
    let datatype = Datatype::Array {
        base_type: Box::new(Datatype::Array {
            base_type: Box::new(byte_member_type()),
            dimensions: dimensions.clone(),
        }),
        dimensions: vec![3],
    };
    let expected_size = size.saturating_mul(3);
    let expected = array_bytes(
        array_bytes(
            build_fixed_point(1, false, false, 0, 8),
            &dimensions,
            size,
            DATATYPE_VERSION_THREE,
        ),
        &[3],
        expected_size,
        DATATYPE_VERSION_THREE,
    );
    assert_eq!(datatype.type_size(), expected_size);
    assert_eq!(serialize_datatype(&datatype).unwrap(), expected);
}

#[test]
fn canonical_enum_values_preserve_the_supplied_bytes() {
    let datatype = Datatype::Enumeration {
        size: 1,
        base_type: Box::new(byte_member_type()),
        members: vec![
            __private::enum_member("wide".into(), vec![1, 2, 3]),
            __private::enum_member("empty".into(), vec![]),
        ],
    };
    let mut expected = build_dt_header(ENUMERATION_CLASS, DATATYPE_VERSION_THREE, [2, 0, 0], 1);
    expected.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));
    expected.extend_from_slice(b"wide\0empty\0\x01\x02\x03");
    assert_eq!(serialize_datatype(&datatype).unwrap(), expected);
}

#[rstest]
#[case::vax(vax_float(), DATATYPE_VERSION_THREE, 4)]
#[case::complex(complex_bytes(), DATATYPE_VERSION_FIVE, 8)]
fn constructed_enum_bases_propagate_versions(
    #[case] base: Vec<u8>,
    #[case] version: u8,
    #[case] size: u32,
) {
    let datatype = Datatype::Enumeration {
        size,
        base_type: Box::new(parse_datatype(&base).unwrap().0),
        members: Vec::new(),
    };
    let mut expected = build_dt_header(ENUMERATION_CLASS, version, [0, 0, 0], size);
    expected.extend_from_slice(&base);
    assert_eq!(serialize_datatype(&datatype).unwrap(), expected);
    assert_eq!(
        parse_datatype(&expected).unwrap(),
        (datatype, expected.len())
    );
}

#[rstest]
#[case::vax(
    sequence_bytes(vax_float(), DATATYPE_VERSION_THREE),
    DATATYPE_VERSION_THREE
)]
#[case::complex(mixed_bytes(), DATATYPE_VERSION_FIVE)]
fn appending_after_a_prefix_preserves_earlier_headers(
    #[case] message: Vec<u8>,
    #[case] version: u8,
) {
    let prefix = [0xa5, 0x5a, 0xff];
    let mut output = prefix.to_vec();
    let first = Datatype::Opaque {
        size: 4,
        tag: b"first".to_vec(),
    };
    let first_node = serialize_datatype_into(&mut output, &first).unwrap();
    assert_eq!(first_node.version, DATATYPE_VERSION_ONE);
    assert_eq!(first_node.type_size, 4);
    let mut expected = prefix.to_vec();
    expected.extend_from_slice(&build_dt_header(
        OPAQUE_CLASS,
        DATATYPE_VERSION_ONE,
        [8, 0, 0],
        4,
    ));
    expected.extend_from_slice(b"first\0\0\0");
    assert_eq!(output, expected);

    let datatype = parse_datatype(&message).unwrap().0;
    let node = serialize_datatype_into(&mut output, &datatype).unwrap();
    assert_eq!(node.version, version);
    assert_eq!(node.type_size, datatype.type_size());
    expected.extend_from_slice(&message);
    assert_eq!(output, expected);

    let final_node = serialize_datatype_into(&mut output, &byte_member_type()).unwrap();
    assert_eq!(final_node.version, DATATYPE_VERSION_ONE);
    assert_eq!(final_node.type_size, 1);
    expected.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));
    assert_eq!(output, expected);
}

#[rstest]
#[case::absent(vec![], Ok(0))]
#[case::one(vec![b'x'; 1], Ok(8))]
#[case::seven(vec![b'x'; 7], Ok(8))]
#[case::eight(vec![b'x'; 8], Ok(16))]
#[case::maximum(vec![b'x'; 247], Ok(248))]
#[case::unterminated_property_limit(vec![b'x'; 248], Err(FormatError::OpaqueTagTooLong { length: 248 }))]
#[case::field_limit(vec![b'x'; 255], Err(FormatError::OpaqueTagTooLong { length: 255 }))]
#[case::field_overflow(vec![b'x'; 256], Err(FormatError::OpaqueTagTooLong { length: 256 }))]
#[case::first_nul(vec![0], Err(FormatError::InvalidOpaqueTagByte { offset: 0, byte: 0 }))]
#[case::interior_nul(b"abc\0def".to_vec(), Err(FormatError::InvalidOpaqueTagByte { offset: 3, byte: 0 }))]
#[case::last_nul([vec![b'x'; 246], vec![0]].concat(), Err(FormatError::InvalidOpaqueTagByte { offset: 246, byte: 0 }))]
#[case::length_before_nul(vec![0; 248], Err(FormatError::OpaqueTagTooLong { length: 248 }))]
fn nested_opaque_tags_keep_their_encoding_boundaries(
    #[case] tag: Vec<u8>,
    #[case] property: Result<u8, FormatError>,
) {
    let datatype = Datatype::Compound {
        size: 32,
        members: vec![
            __private::compound_member("first".into(), 0, byte_member_type()),
            __private::compound_member(
                "nested".into(),
                8,
                Datatype::VariableLength(VariableLengthType::sequence(Datatype::Array {
                    base_type: Box::new(Datatype::Enumeration {
                        size: 4,
                        base_type: Box::new(Datatype::Opaque {
                            size: 4,
                            tag: tag.clone(),
                        }),
                        members: Vec::new(),
                    }),
                    dimensions: vec![2],
                })),
            ),
        ],
    };
    let encoded = serialize_datatype(&datatype);
    match property {
        Ok(property) => {
            let mut expected =
                build_dt_header(COMPOUND_CLASS, DATATYPE_VERSION_THREE, [2, 0, 0], 32);
            expected.extend_from_slice(b"first\0\0");
            expected.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));
            expected.extend_from_slice(b"nested\0\x08");
            let mut opaque =
                build_dt_header(OPAQUE_CLASS, DATATYPE_VERSION_ONE, [property, 0, 0], 4);
            opaque.extend_from_slice(&tag);
            opaque.resize(DATATYPE_HEADER_SIZE + usize::from(property), 0);
            let mut enumeration =
                build_dt_header(ENUMERATION_CLASS, DATATYPE_VERSION_THREE, [0, 0, 0], 4);
            enumeration.extend_from_slice(&opaque);
            expected.extend_from_slice(&sequence_bytes(
                array_bytes(enumeration, &[2], 8, DATATYPE_VERSION_THREE),
                DATATYPE_VERSION_THREE,
            ));
            assert_eq!(encoded.unwrap(), expected);
            assert_eq!(
                parse_datatype(&expected).unwrap(),
                (datatype, expected.len())
            );
        }
        Err(error) => assert_eq!(encoded.unwrap_err(), error),
    }
}

fn big_endian_float() -> Vec<u8> {
    let mut bytes = build_float(8, 52, 11, 0, 52, 1023);
    bytes[1] |= FLOAT_BYTE_ORDER_LOW_BIT;
    bytes
}

fn vax_float() -> Vec<u8> {
    let mut bytes = build_float(4, 23, 8, 0, 23, 127);
    bytes[0] = (DATATYPE_VERSION_THREE << 4) | FLOATING_POINT_CLASS;
    bytes[1] |= FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT;
    bytes
}

fn time_bytes(order: u8) -> Vec<u8> {
    let mut bytes = build_dt_header(TIME_CLASS, DATATYPE_VERSION_ONE, [order, 0, 0], 8);
    bytes.extend_from_slice(&64u16.to_le_bytes());
    bytes
}

fn bitfield_bytes() -> Vec<u8> {
    let mut bytes = build_fixed_point(4, true, false, 3, 25);
    bytes[0] = (DATATYPE_VERSION_ONE << 4) | BIT_FIELD_CLASS;
    bytes[1] |= LOW_PADDING_BIT | HIGH_PADDING_BIT;
    bytes
}

fn opaque_bytes() -> Vec<u8> {
    let mut bytes = build_dt_header(OPAQUE_CLASS, DATATYPE_VERSION_ONE, [8, 0, 0], 4);
    bytes.extend_from_slice(&[0xff, 0xfe, 0, 0, 0, 0, 0, 0]);
    bytes
}

fn enum_bytes(count: u16) -> Vec<u8> {
    let [low, high] = count.to_le_bytes();
    let mut bytes = build_dt_header(ENUMERATION_CLASS, DATATYPE_VERSION_THREE, [low, high, 0], 2);
    bytes.extend_from_slice(&build_fixed_point(2, false, false, 0, 16));
    for value in 0..count {
        bytes.extend_from_slice(format!("M{value}\0").as_bytes());
    }
    for value in 0..count {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn variable_string_bytes() -> Vec<u8> {
    let mut bytes = build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [0x21, 1, 0],
        16,
    );
    bytes.extend_from_slice(&build_fixed_point(1, true, false, 0, 8));
    bytes
}

fn sequence_bytes(child: Vec<u8>, version: u8) -> Vec<u8> {
    let mut bytes = build_dt_header(VARIABLE_LENGTH_CLASS, version, [0, 0, 0], 16);
    bytes.extend_from_slice(&child);
    bytes
}

fn array_bytes(child: Vec<u8>, dimensions: &[u32], size: u32, version: u8) -> Vec<u8> {
    let mut bytes = build_dt_header(ARRAY_CLASS, version, [0, 0, 0], size);
    bytes.push(u8::try_from(dimensions.len()).unwrap());
    for dimension in dimensions {
        bytes.extend_from_slice(&dimension.to_le_bytes());
    }
    bytes.extend_from_slice(&child);
    bytes
}

fn complex_bytes() -> Vec<u8> {
    let mut bytes = build_dt_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    bytes.extend_from_slice(&build_float(4, 23, 8, 0, 23, 127));
    bytes
}

fn mixed_bytes() -> Vec<u8> {
    let mut bytes = build_dt_header(COMPOUND_CLASS, DATATYPE_VERSION_FIVE, [3, 0, 0], 44);
    bytes.extend_from_slice(b"items\0\0");
    bytes.extend_from_slice(&array_bytes(
        sequence_bytes(enum_bytes(3), DATATYPE_VERSION_THREE),
        &[2],
        32,
        DATATYPE_VERSION_THREE,
    ));
    bytes.extend_from_slice(b"tag\0\x20");
    bytes.extend_from_slice(&opaque_bytes());
    bytes.extend_from_slice(b"late\0\x24");
    bytes.extend_from_slice(&complex_bytes());
    bytes
}
