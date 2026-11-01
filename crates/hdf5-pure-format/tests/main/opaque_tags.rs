use hdf5_pure_core::CharacterSet;
use hdf5_pure_core::Datatype;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::StringPadding;
use hdf5_pure_core::VariableLengthType;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::Class;
use test_util::opaque::{ArrayVersion, CompoundLayout};

#[rstest]
#[case::absent(0, 0)]
#[case::one(1, 8)]
#[case::blob(4, 8)]
#[case::seven(7, 8)]
#[case::eight(8, 16)]
#[case::nine(9, 16)]
#[case::fifteen(15, 16)]
#[case::sixteen(16, 24)]
#[case::maximum(247, 248)]
fn opaque_tags_use_terminated_aligned_properties(#[case] length: usize, #[case] wire_length: u8) {
    let tag = if length == 4 {
        b"BLOB".to_vec()
    } else {
        vec![b'x'; length]
    };
    let datatype = Datatype::Opaque {
        size: 4,
        tag: tag.clone(),
    };
    let encoded = format::serialize_datatype(&datatype).unwrap();
    let mut expected = datatype::header(Class::OPAQUE, 1, [wire_length, 0, 0], 4);
    expected.extend_from_slice(&tag);
    expected.extend_from_slice(&vec![0; usize::from(wire_length) - length]);
    assert_eq!(encoded, expected);
    assert_eq!(
        format::parse_datatype(&expected).unwrap(),
        (datatype, 8 + usize::from(wire_length)),
    );
}

#[rstest]
#[case::encoded_empty(&[0; 8], &[])]
#[case::full_native_block(b"12345678", b"12345678")]
#[case::native_non_ascii(&[0xff, 0xfe, 0, 0, 0, 0, 0, 0], &[0xff, 0xfe])]
fn opaque_tags_preserve_native_logical_bytes(#[case] property: &[u8], #[case] logical: &[u8]) {
    let mut bytes = datatype::header(Class::OPAQUE, 1, [8, 0, 0], 4);
    bytes.extend_from_slice(property);
    let datatype = Datatype::Opaque {
        size: 4,
        tag: logical.to_vec(),
    };
    assert_eq!(
        format::parse_datatype(&bytes).unwrap(),
        (datatype.clone(), 16)
    );
    let canonical = format::serialize_datatype(&datatype).unwrap();
    assert_eq!(
        format::parse_datatype(&canonical).unwrap(),
        (datatype, canonical.len()),
    );
    if !logical.is_empty() {
        assert_eq!(&canonical[8..8 + logical.len()], logical);
        assert_eq!(canonical[8 + logical.len()], 0);
    } else {
        assert_eq!(canonical, datatype::header(Class::OPAQUE, 1, [0; 3], 4));
    }
}

#[test]
fn native_full_property_is_preserved_when_canonical_encoding_cannot_fit() {
    let mut bytes = datatype::header(Class::OPAQUE, 1, [248, 0, 0], 4);
    bytes.extend_from_slice(&[b'x'; 248]);
    let (datatype, consumed) = format::parse_datatype(&bytes).unwrap();
    assert_eq!(consumed, 256);
    assert_eq!(
        datatype,
        Datatype::Opaque {
            size: 4,
            tag: vec![b'x'; 248]
        }
    );
    assert_eq!(
        format::serialize_datatype(&datatype).unwrap_err(),
        FormatError::OpaqueTagTooLong { length: 248 },
    );
}

#[rstest]
#[case::historical_blob([4, 0, 0], b"BLOB\0\0\0\0", FormatError::InvalidOpaqueTagLength { length: 4 })]
#[case::unaligned_larger([9, 0, 0], &[0; 16], FormatError::InvalidOpaqueTagLength { length: 9 })]
#[case::reserved_low([0, 1, 0], &[], FormatError::InvalidOpaqueReservedBits { bits: 1 })]
#[case::reserved_high([0, 0, 1], &[], FormatError::InvalidOpaqueReservedBits { bits: 256 })]
#[case::nonzero_padding([8, 0, 0], b"BLOB\0\0x\0", FormatError::InvalidOpaqueTagPadding { offset: 6, byte: b'x' })]
#[case::nonzero_after_empty([8, 0, 0], b"\0x\0\0\0\0\0\0", FormatError::InvalidOpaqueTagPadding { offset: 1, byte: b'x' })]
fn opaque_tags_reject_malformed_properties(
    #[case] fields: [u8; 3],
    #[case] property: &[u8],
    #[case] expected: FormatError,
) {
    let mut bytes = datatype::header(Class::OPAQUE, 1, fields, 4);
    bytes.extend_from_slice(property);
    assert_eq!(format::parse_datatype(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::missing_property(0)]
#[case::within_tag(3)]
#[case::missing_terminator(4)]
#[case::missing_padding(5)]
#[case::last_padding_byte(7)]
fn opaque_tags_require_the_entire_declared_property(#[case] available: usize) {
    let mut bytes = datatype::header(Class::OPAQUE, 1, [8, 0, 0], 4);
    bytes.extend_from_slice(&b"BLOB\0\0\0\0"[..available]);
    assert_eq!(
        format::parse_datatype(&bytes).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 8,
            available
        },
    );
}

#[rstest]
#[case::too_long(vec![b'x'; 248], FormatError::OpaqueTagTooLong { length: 248 })]
#[case::truncated_old_field(vec![b'x'; 256], FormatError::OpaqueTagTooLong { length: 256 })]
#[case::leading_nul(b"\0BLOB".to_vec(), FormatError::InvalidOpaqueTagByte { offset: 0, byte: 0 })]
#[case::embedded_nul(b"BL\0OB".to_vec(), FormatError::InvalidOpaqueTagByte { offset: 2, byte: 0 })]
#[case::trailing_nul(b"BLOB\0".to_vec(), FormatError::InvalidOpaqueTagByte { offset: 4, byte: 0 })]
fn opaque_tags_reject_unrepresentable_caller_input(
    #[case] tag: Vec<u8>,
    #[case] expected: FormatError,
) {
    let datatype = Datatype::Opaque { size: 4, tag };
    assert_eq!(format::serialize_datatype(&datatype).unwrap_err(), expected);
}

#[rstest]
#[case::compound_v1(CompoundLayout::VersionOneScalar, 5)]
#[case::compound_v1_array(CompoundLayout::VersionOneArray, 9)]
#[case::compound_v2(CompoundLayout::VersionTwo, 5)]
#[case::compound_v3_one_byte_limit(CompoundLayout::VersionThreeOneByteLimit, 255)]
#[case::compound_v3_two_byte_start(CompoundLayout::VersionThreeTwoByteStart, 256)]
#[case::compound_v3_three_byte_start(CompoundLayout::VersionThreeThreeByteStart, 65_536)]
#[case::compound_v4(CompoundLayout::VersionFour, 5)]
#[case::compound_v5(CompoundLayout::VersionFive, 5)]
fn opaque_members_leave_adjacent_compound_members_at_their_declared_offsets(
    #[case] layout: CompoundLayout,
    #[case] size: u32,
) {
    let bytes = layout.build();
    let (parsed, consumed) = format::parse_datatype(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let Datatype::Compound {
        size: actual_size,
        members,
    } = &parsed
    else {
        panic!("expected compound datatype, got {parsed:?}");
    };
    assert_eq!(*actual_size, size);
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].name, "blob");
    assert_eq!(members[0].byte_offset, 0);
    let opaque = Datatype::Opaque {
        size: 4,
        tag: b"BLOB".to_vec(),
    };
    let expected_first = if layout.is_array() {
        Datatype::Array {
            base_type: Box::new(opaque),
            dimensions: vec![2],
        }
    } else {
        opaque
    };
    assert_eq!(members[0].datatype, expected_first);
    assert_eq!(members[1].name, "next");
    assert_eq!(members[1].byte_offset, u64::from(size - 1));
    assert_eq!(
        members[1].datatype,
        Datatype::String {
            size: 1,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii
        }
    );
    let encoded = format::serialize_datatype(&parsed).unwrap();
    assert_eq!(
        format::parse_datatype(&encoded).unwrap(),
        (parsed, encoded.len())
    );
}

#[rstest]
#[case::version_two(ArrayVersion::Two)]
#[case::version_three(ArrayVersion::Three)]
#[case::version_four(ArrayVersion::Four)]
#[case::version_five(ArrayVersion::Five)]
fn opaque_array_parsing_stops_before_the_next_field(#[case] version: ArrayVersion) {
    let mut bytes = version.build();
    let expected_consumed = bytes.len();
    bytes.extend_from_slice(b"next field");
    let expected = Datatype::Array {
        base_type: Box::new(Datatype::Opaque {
            size: 4,
            tag: b"BLOB".to_vec(),
        }),
        dimensions: vec![2],
    };
    assert_eq!(
        format::parse_datatype(&bytes).unwrap(),
        (expected.clone(), expected_consumed)
    );
    let encoded = format::serialize_datatype(&expected).unwrap();
    assert_eq!(
        format::parse_datatype(&encoded).unwrap(),
        (expected, encoded.len())
    );
}

#[rstest]
#[case::array(|base| Datatype::Array { base_type: Box::new(base), dimensions: vec![2] })]
#[case::compound(|base| Datatype::Compound { size: 4, members: vec![hdf5_pure_core::__private::compound_member("blob".into(), 0, base)] })]
#[case::sequence(|base| Datatype::VariableLength(VariableLengthType::sequence(base)))]
fn nested_opaque_encoding_propagates_the_original_error(#[case] nest: fn(Datatype) -> Datatype) {
    let datatype = nest(Datatype::Opaque {
        size: 4,
        tag: vec![b'x'; 248],
    });
    assert_eq!(
        format::serialize_datatype(&datatype).unwrap_err(),
        FormatError::OpaqueTagTooLong { length: 248 }
    );
}

#[rstest]
#[case::version_two(false)]
#[case::version_three(true)]
fn inline_opaque_attributes_propagate_encoding_errors(#[case] version_three: bool) {
    let attribute = format::AttributeMessage {
        name: "blob".into(),
        datatype: Datatype::Opaque {
            size: 4,
            tag: vec![b'x'; 248],
        },
        dataspace: format::Dataspace {
            rank: 0,
            space_type: format::DataspaceType::Scalar,
            dimensions: Vec::new(),
            max_dimensions: None,
        },
        raw_data: vec![1, 2, 3, 4],
        datatype_location: format::DatatypeLocation::Inline,
    };
    let error = if version_three {
        attribute.serialize_v3(format::OffsetWidth::Eight, 8)
    } else {
        attribute.serialize(format::OffsetWidth::Eight, 8)
    }
    .unwrap_err();
    assert_eq!(error, FormatError::OpaqueTagTooLong { length: 248 });
}
