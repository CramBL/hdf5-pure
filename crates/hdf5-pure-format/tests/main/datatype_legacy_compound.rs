use hdf5_pure_core::{
    CompoundMember, Datatype, DatatypeMemberName, FixedPointType, FormatError, VariableLengthType,
};
use hdf5_pure_format::__private::DecodedDatatype;
use hdf5_pure_format::__private::{
    CompoundV1Member, CompoundV1Shape, DatatypeMessage, DatatypeMessageMember, DatatypeProperties,
    DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype::containers::{self, CompoundField, MemberOffset};
use test_util::datatype::leaf::{self, Version};
use test_util::datatype::{Class, VariableLengthSequence};
use test_util::legacy_compound::{self, SequenceMemberShape};

#[rstest]
#[case::rank_one(1, [2, 0, u32::MAX, 3], &[2])]
#[case::rank_two(2, [2, 3, u32::MAX, 0], &[2, 3])]
#[case::rank_three(3, [2, 3, 4, u32::MAX], &[2, 3, 4])]
#[case::rank_four(4, [2, 3, 4, 5], &[2, 3, 4, 5])]
fn retained_version_one_compound_preserves_all_inline_array_extent_slots(
    #[case] rank: u8,
    #[case] extents: [u32; 4],
    #[case] dimensions: &[u32],
) {
    let byte = leaf::unsigned_byte(Version::One);
    let size = MEMBER_OFFSET + dimensions.iter().product::<u32>();
    let mut bytes = legacy_compound::raw_member(size, MEMBER_OFFSET, rank, 0, extents, &byte);
    let consumed = bytes.len();
    bytes.extend_from_slice(&[0xff, 0xfe]);

    let (message, length) = DatatypeMessage::parse_prefix(&bytes).unwrap();
    assert_eq!(length, consumed);
    assert_eq!(message.version(), DatatypeVersion::One);
    let DatatypeProperties::CompoundV1 {
        size: actual_size,
        members,
    } = message.properties()
    else {
        panic!("expected a version 1 compound");
    };
    assert_eq!(*actual_size, size);
    let [member] = members.as_slice() else {
        panic!("expected one member");
    };
    assert_eq!(member.shape().rank(), rank);
    assert_eq!(member.shape().extents(), extents);
    assert_eq!(member.member().name().as_bytes(), b"arr");
    assert_eq!(member.member().byte_offset(), u64::from(MEMBER_OFFSET));
    assert_eq!(
        member.member().datatype(),
        &byte_description(DatatypeVersion::One)
    );

    let properties = DatatypeProperties::CompoundV1 {
        size,
        members: vec![version_one_member(
            c"arr".into(),
            MEMBER_OFFSET,
            rank,
            extents,
        )],
    };
    assert_eq!(
        DatatypeMessage::new(DatatypeVersion::One, properties).unwrap(),
        message
    );
    let field = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(FixedPointType::u8())),
        dimensions: dimensions.to_vec(),
    };
    let expected = Datatype::Compound {
        size,
        members: vec![CompoundMember::new(
            c"arr".into(),
            u64::from(MEMBER_OFFSET),
            field,
        )],
    };
    assert_eq!(message.to_datatype().unwrap(), expected);
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    assert_eq!((decoded.into_datatype(), bytes_read), (expected, consumed));
}

#[test]
fn retained_version_one_scalar_preserves_inactive_extents() {
    let extents = [3, 0, u32::MAX, 7];
    let size = MEMBER_OFFSET + 1;
    let byte = leaf::unsigned_byte(Version::One);
    let bytes = legacy_compound::raw_member(size, MEMBER_OFFSET, 0, 0, extents, &byte);
    let (message, consumed) = DatatypeMessage::parse_prefix(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let properties = DatatypeProperties::CompoundV1 {
        size,
        members: vec![version_one_member(c"arr".into(), MEMBER_OFFSET, 0, extents)],
    };
    let expected = DatatypeMessage::new(DatatypeVersion::One, properties).unwrap();
    assert_eq!(message, expected);
    let DatatypeProperties::CompoundV1 { members, .. } = message.properties() else {
        panic!("expected a version 1 compound");
    };
    assert_eq!(members[0].shape().rank(), 0);
    assert_eq!(members[0].shape().extents(), extents);
    let projected = Datatype::Compound {
        size,
        members: vec![CompoundMember::new(
            c"arr".into(),
            u64::from(MEMBER_OFFSET),
            Datatype::FixedPoint(FixedPointType::u8()),
        )],
    };
    assert_eq!(message.to_datatype().unwrap(), projected);
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    assert_eq!((decoded.into_datatype(), bytes_read), (projected, consumed));
}

#[test]
fn retained_version_two_compound_preserves_name_bytes_and_nested_versions() {
    let byte = leaf::unsigned_byte(Version::Two);
    let nested = legacy_compound::array_member();
    let size = MEMBER_OFFSET + 1;
    let mut bytes = containers::compound(
        Version::Two.get(),
        [2, 0, 0],
        size,
        MemberOffset::Four,
        &[
            CompoundField {
                name: b"abcdef\xff",
                offset: u64::from(MEMBER_OFFSET),
                datatype: &byte,
            },
            CompoundField {
                name: b"abcdefgh",
                offset: 0,
                datatype: &nested,
            },
        ],
    );
    let length = bytes.len();
    bytes.extend_from_slice(&[0xff, 0xfe]);
    let (message, consumed) = DatatypeMessage::parse_prefix(&bytes).unwrap();
    let expected = DatatypeMessage::new(
        DatatypeVersion::Two,
        DatatypeProperties::Compound {
            size,
            members: vec![
                DatatypeMessageMember::new(
                    c"abcdef\xff".into(),
                    u64::from(MEMBER_OFFSET),
                    byte_description(DatatypeVersion::Two),
                ),
                DatatypeMessageMember::new(
                    c"abcdefgh".into(),
                    0,
                    DatatypeMessage::parse(&nested).unwrap(),
                ),
            ],
        },
    )
    .unwrap();
    assert_eq!((message, consumed), (expected.clone(), length));
    let projected = Datatype::Compound {
        size,
        members: vec![
            CompoundMember::new(
                c"abcdef\xff".into(),
                u64::from(MEMBER_OFFSET),
                Datatype::FixedPoint(FixedPointType::u8()),
            ),
            CompoundMember::new(
                c"abcdefgh".into(),
                0,
                Datatype::Compound {
                    size: 16,
                    members: vec![CompoundMember::new(
                        c"arr".into(),
                        0,
                        Datatype::Array {
                            base_type: Box::new(Datatype::FixedPoint(FixedPointType::i32())),
                            dimensions: vec![4],
                        },
                    )],
                },
            ),
        ],
    };
    assert_eq!(expected.to_datatype().unwrap(), projected);
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    assert_eq!((decoded.into_datatype(), bytes_read), (projected, length));
}

#[rstest]
#[case::rank_above_four(5, 0, [1; 4], FormatError::InvalidCompoundV1Rank(5))]
#[case::zero_active_extent(4, 0, [1, 1, 0, 1], FormatError::InvalidCompoundV1Extent { dimension: 2 })]
#[case::nonzero_permutation(1, 0x12_3456, [1; 4], FormatError::UnsupportedCompoundV1Permutation(0x12_3456))]
#[case::product_overflow(2, 0, [u32::MAX, 2, 0, 0], FormatError::CompoundArraySizeOverflow { size: u32::MAX, extent: 2 })]
fn datatype_parsers_reject_invalid_inline_array_fields(
    #[case] rank: u8,
    #[case] permutation: u32,
    #[case] extents: [u32; 4],
    #[case] expected: FormatError,
) {
    let byte = leaf::unsigned_byte(Version::One);
    let bytes = legacy_compound::raw_member(u32::MAX, 0, rank, permutation, extents, &byte);
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
    assert_eq!(DecodedDatatype::parse_prefix(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::first_reserved_field(DIMENSION_BLOCK_OFFSET + 1, 0x10)]
#[case::second_reserved_field(DIMENSION_BLOCK_OFFSET + 11, 0xff)]
fn retained_version_one_compound_rejects_nonzero_reserved_bytes(
    #[case] offset: usize,
    #[case] byte: u8,
) {
    let mut bytes = legacy_compound::array_member();
    let (expected, expected_len) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    bytes[offset] = byte;
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidCompoundV1ReservedByte { offset, byte }
    );
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&bytes).unwrap();
    assert_eq!(
        (decoded.datatype(), bytes_read),
        (expected.datatype(), expected_len)
    );
}

#[rstest]
#[case::reserved_byte_before_permutation(
    DIMENSION_BLOCK_OFFSET + 1,
    FormatError::InvalidCompoundV1ReservedByte { offset: DIMENSION_BLOCK_OFFSET + 1, byte: 0x10 },
)]
#[case::reserved_byte_after_permutation(
    DIMENSION_BLOCK_OFFSET + 11,
    FormatError::UnsupportedCompoundV1Permutation(1),
)]
fn retained_version_one_compound_checks_reserved_and_permutation_fields_in_order(
    #[case] reserved_offset: usize,
    #[case] expected: FormatError,
) {
    let child = leaf::unsigned_byte(Version::One);
    let mut bytes = legacy_compound::raw_member(1, 0, 0, 1, [0; 4], &child);
    bytes[reserved_offset] = 0x10;
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::v1(Version::One)]
#[case::v2(Version::Two)]
fn retained_legacy_compound_rejects_reserved_class_bits(#[case] version: Version) {
    let child = leaf::unsigned_byte(Version::One);
    let bytes = containers::compound(
        version.get(),
        [1, 0, 0x10],
        1,
        MemberOffset::Four,
        &[CompoundField {
            name: b"a",
            offset: 0,
            datatype: &child,
        }],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class: Class::COMPOUND.0,
            version: version.get(),
            flags: 0x10_0001
        }
    );
}

#[rstest]
#[case::name_padding(15, 16, 1)]
#[case::offset(19, 20, 1)]
#[case::dimension_block(47, 48, 5)]
#[case::child_header(55, 56, 1)]
fn compound_v1_parser_acquires_complete_fields_before_checking_their_values(
    #[case] available: usize,
    #[case] expected: usize,
    #[case] rank: u8,
) {
    let byte = leaf::unsigned_byte(Version::One);
    let bytes = legacy_compound::raw_member(4, 0, rank, 0, [4, 0, 0, 0], &byte);
    assert_eq!(
        DatatypeMessage::parse(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        },
    );
    assert_eq!(
        DecodedDatatype::parse_prefix(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        },
    );
}

#[rstest]
#[case::v1(legacy_compound::array_member())]
#[case::v2(containers::legacy_compound_v2())]
fn datatype_parsers_report_nested_legacy_compound_eof_from_the_outer_description(
    #[case] mut child: Vec<u8>,
) {
    child.truncate(child.len() - 1);
    let body = VariableLengthSequence::new(Version::Five, 8, &child).build();
    let expected = FormatError::UnexpectedEof {
        expected: body.len() + 1,
        available: body.len(),
    };
    assert_eq!(DecodedDatatype::parse_prefix(&body).unwrap_err(), expected);
    assert_eq!(DatatypeMessage::parse(&body).unwrap_err(), expected);
}

#[rstest]
#[case::v1(Version::One, Version::Two)]
#[case::v2(Version::Two, Version::Three)]
fn retained_legacy_compound_rejects_a_newer_child(#[case] outer: Version, #[case] child: Version) {
    let base = leaf::unsigned_byte(child);
    let bytes = containers::compound(
        outer.get(),
        [1, 0, 0],
        1,
        MemberOffset::Four,
        &[CompoundField {
            name: b"a",
            offset: 0,
            datatype: &base,
        }],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::DatatypeChildVersion {
            class: Class::COMPOUND.0,
            version: outer.get(),
            child_version: child.get(),
        }
    );
}

#[rstest]
#[case::two_byte_offsets(2, 10)]
#[case::four_byte_offsets(4, 12)]
#[case::eight_byte_offsets(8, 16)]
fn retained_inline_array_uses_the_declared_source_descriptor_size(
    #[case] width: usize,
    #[case] descriptor_size: u32,
) {
    let offset = 4;
    let size = offset + 2 * descriptor_size;
    let bytes = legacy_compound::sequence_member(width, offset, SequenceMemberShape::ImplicitArray);
    let (message, consumed) = DatatypeMessage::parse_prefix(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(
        message.to_datatype().unwrap(),
        Datatype::Compound {
            size,
            members: vec![CompoundMember::new(
                c"arr".into(),
                u64::from(offset),
                Datatype::Array {
                    base_type: Box::new(Datatype::VariableLength(VariableLengthType::sequence(
                        Datatype::FixedPoint(FixedPointType::u8())
                    ))),
                    dimensions: vec![2],
                }
            )],
        }
    );
    let too_small = size - 1;
    let mut bytes = bytes;
    bytes[4..8].copy_from_slice(&too_small.to_le_bytes());
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: c"arr".into(),
            offset: u64::from(offset),
            field_size: 2 * descriptor_size,
            compound_size: too_small,
        }
    );
}

#[rstest]
#[case::rank(5, [1; 4], FormatError::InvalidCompoundV1Rank(5))]
#[case::extent(3, [1, 0, 1, 0], FormatError::InvalidCompoundV1Extent { dimension: 1 })]
fn version_one_shape_construction_rejects_invalid_rank_and_extent(
    #[case] rank: u8,
    #[case] extents: [u32; 4],
    #[case] expected: FormatError,
) {
    assert_eq!(CompoundV1Shape::new(rank, extents).unwrap_err(), expected);
}

#[test]
fn version_two_construction_rejects_version_one_compound_properties() {
    let properties = DatatypeProperties::CompoundV1 {
        size: 1,
        members: vec![version_one_member(c"a".into(), 0, 0, [0; 4])],
    };
    assert_eq!(
        DatatypeMessage::new(DatatypeVersion::Two, properties).unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: Class::COMPOUND.0,
            version: Version::Two.get(),
        },
    );
}

#[rstest]
#[case::too_small(1, [2, 1, 0, 0], FormatError::CompoundFieldOutOfBounds {
    name: c"a".into(), offset: 0, field_size: 2, compound_size: 1,
})]
#[case::product_overflow(u32::MAX, [u32::MAX, 2, 0, 0], FormatError::CompoundArraySizeOverflow {
    size: u32::MAX, extent: 2,
})]
fn version_one_compound_construction_checks_inline_array_footprints(
    #[case] size: u32,
    #[case] extents: [u32; 4],
    #[case] expected: FormatError,
) {
    let members = vec![version_one_member(c"a".into(), 0, 2, extents)];
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::One,
            DatatypeProperties::CompoundV1 { size, members }
        )
        .unwrap_err(),
        expected
    );
}

#[test]
fn retained_version_two_compound_preserves_the_largest_member_count() {
    let byte = leaf::unsigned_byte(Version::One);
    let names: Vec<_> = (0..u16::MAX).map(|index| format!("m{index}")).collect();
    let fields: Vec<_> = (0..u16::MAX)
        .zip(&names)
        .map(|(index, name)| CompoundField {
            name: name.as_bytes(),
            offset: u64::from(index),
            datatype: &byte,
        })
        .collect();
    let [low, high] = u16::MAX.to_le_bytes();
    let bytes = containers::compound(
        Version::Two.get(),
        [low, high, 0],
        u32::from(u16::MAX),
        MemberOffset::Four,
        &fields,
    );
    let (message, consumed) = DatatypeMessage::parse_prefix(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let DatatypeProperties::Compound { size, members } = message.properties() else {
        panic!("expected a version 2 compound");
    };
    assert_eq!(*size, u32::from(u16::MAX));
    assert_eq!(members.len(), usize::from(u16::MAX));
    let last = members.last().unwrap();
    assert_eq!(last.byte_offset(), u64::from(u16::MAX - 1));
    assert_eq!(last.name().as_bytes(), names.last().unwrap().as_bytes());
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
}

#[test]
fn version_one_compound_construction_rejects_an_unrepresentable_member_count() {
    let count = usize::from(u16::MAX) + 1;
    let members = vec![version_one_member(c"".into(), 0, 0, [0; 4]); count];
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::One,
            DatatypeProperties::CompoundV1 { size: 1, members }
        )
        .unwrap_err(),
        FormatError::ValueTooLargeForRepresentation {
            value: u64::from(u16::MAX) + 1,
            target: "u16"
        },
    );
}

#[rstest]
#[case::scalar(0, [3, 0, u32::MAX, 7])]
#[case::rank_one(1, [2, 0, u32::MAX, 3])]
#[case::rank_two(2, [2, 3, u32::MAX, 0])]
#[case::rank_three(3, [2, 3, 4, u32::MAX])]
#[case::rank_four(4, [2, 3, 4, 5])]
fn retained_version_one_compound_encodes_all_extent_slots(
    #[case] rank: u8,
    #[case] extents: [u32; 4],
) {
    let size = MEMBER_OFFSET + extents[..usize::from(rank)].iter().product::<u32>();
    let message = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::CompoundV1 {
            size,
            members: vec![version_one_member(
                c"arr".into(),
                MEMBER_OFFSET,
                rank,
                extents,
            )],
        },
    )
    .unwrap();
    let child = leaf::unsigned_byte(Version::One);
    let expected = legacy_compound::raw_member(size, MEMBER_OFFSET, rank, 0, extents, &child);
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), expected,);
}

#[test]
fn retained_version_two_compound_encodes_padded_names_and_nested_versions() {
    let child = leaf::unsigned_byte(Version::Two);
    let nested = legacy_compound::array_member();
    let size = MEMBER_OFFSET + 1;
    let message = DatatypeMessage::new(
        DatatypeVersion::Two,
        DatatypeProperties::Compound {
            size,
            members: vec![
                DatatypeMessageMember::new(
                    c"abcdef\xff".into(),
                    u64::from(MEMBER_OFFSET),
                    byte_description(DatatypeVersion::Two),
                ),
                DatatypeMessageMember::new(
                    c"abcdefgh".into(),
                    0,
                    DatatypeMessage::parse(&nested).unwrap(),
                ),
            ],
        },
    )
    .unwrap();
    let expected = containers::compound(
        Version::Two.get(),
        [2, 0, 0],
        size,
        MemberOffset::Four,
        &[
            CompoundField {
                name: b"abcdef\xff",
                offset: u64::from(MEMBER_OFFSET),
                datatype: &child,
            },
            CompoundField {
                name: b"abcdefgh",
                offset: 0,
                datatype: &nested,
            },
        ],
    );
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), expected);
}

#[rstest]
#[case::v1(legacy_compound::array_member())]
#[case::v2(containers::legacy_compound_v2())]
fn retained_legacy_compound_encoding_preserves_the_output_suffix(#[case] bytes: Vec<u8>) {
    let (message, consumed) = DatatypeMessage::parse_prefix(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let prepared = message.prepare(OffsetWidth::Eight).unwrap();
    assert_eq!(prepared.encoded_size(), bytes.len());
    let suffix = [0xff, 0xfe];
    let mut output = vec![0xa5; bytes.len()];
    output.extend_from_slice(&suffix);
    let written = prepared.write_into(&mut output).unwrap();
    assert_eq!(written, bytes.len());
    assert_eq!(&output[..written], bytes);
    assert_eq!(&output[written..], suffix);
}

#[rstest]
#[case::v1(legacy_compound::array_member())]
#[case::v2(containers::legacy_compound_v2())]
fn retained_legacy_compound_encoding_leaves_short_output_unchanged(#[case] bytes: Vec<u8>) {
    let message = DatatypeMessage::parse(&bytes).unwrap();
    let prepared = message.prepare(OffsetWidth::Eight).unwrap();
    let mut output = vec![0xa5; bytes.len() - 1];
    assert_eq!(
        prepared.write_into(&mut output).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: bytes.len(),
            available: output.len(),
        },
    );
    assert_eq!(output, vec![0xa5; bytes.len() - 1]);
}

#[rstest]
#[case::two_byte_offsets(OffsetWidth::Two, 2)]
#[case::four_byte_offsets(OffsetWidth::Four, 4)]
#[case::eight_byte_offsets(OffsetWidth::Eight, 8)]
fn retained_inline_array_encoding_uses_the_destination_descriptor_size(
    #[case] width: OffsetWidth,
    #[case] fixture_width: usize,
) {
    let offset = 4;
    let size = 36;
    let shape = [2, 0, u32::MAX, 1];
    let base = leaf::unsigned_byte(Version::One);
    let source_child = VariableLengthSequence::new(Version::One, 8, &base).build();
    let source = legacy_compound::raw_member(size, offset, 1, 0, shape, &source_child);
    let message = DatatypeMessage::parse(&source).unwrap();
    let child = VariableLengthSequence::new(Version::One, fixture_width, &base).build();
    let expected = legacy_compound::raw_member(size, offset, 1, 0, shape, &child);
    assert_eq!(message.serialize(width).unwrap(), expected);
}

#[rstest]
#[case::four_byte_offsets(OffsetWidth::Four, 24)]
#[case::eight_byte_offsets(OffsetWidth::Eight, 32)]
fn retained_inline_array_encoding_checks_bounds_at_selected_width(
    #[case] width: OffsetWidth,
    #[case] field_size: u32,
) {
    let offset = 4;
    let source = legacy_compound::sequence_member(2, offset, SequenceMemberShape::ImplicitArray);
    let message = DatatypeMessage::parse(&source).unwrap();
    assert_eq!(
        message.prepare(width).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: c"arr".into(),
            offset: u64::from(offset),
            field_size,
            compound_size: 24,
        },
    );
}

#[test]
fn retained_inline_array_encoding_checks_overlap_at_selected_width() {
    let base = leaf::unsigned_byte(Version::One);
    let sequence = VariableLengthSequence::new(Version::One, 2, &base).build();
    let description = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::CompoundV1 {
            size: 40,
            members: vec![
                CompoundV1Member::new(
                    DatatypeMessageMember::new(
                        c"arr".into(),
                        0,
                        DatatypeMessage::parse(&sequence).unwrap(),
                    ),
                    CompoundV1Shape::new(1, [2, 0, 0, 0]).unwrap(),
                ),
                version_one_member(c"next".into(), 20, 0, [0; 4]),
            ],
        },
    )
    .unwrap();
    assert_eq!(
        description.prepare(OffsetWidth::Four).unwrap_err(),
        FormatError::CompoundFieldOverlap {
            first: c"arr".into(),
            second: c"next".into(),
        },
    );
}

#[test]
fn retained_inline_array_encoding_rejects_footprint_overflow_at_selected_width() {
    let base = leaf::unsigned_byte(Version::One);
    let child = VariableLengthSequence::new(Version::One, 2, &base);
    let extent = u32::MAX / child.descriptor_size();
    let child = child.build();
    let source = legacy_compound::raw_member(u32::MAX, 0, 1, 0, [extent, 0, 0, 0], &child);
    let message = DatatypeMessage::parse(&source).unwrap();
    assert_eq!(
        message.prepare(OffsetWidth::Eight).unwrap_err(),
        FormatError::CompoundArraySizeOverflow { size: 16, extent },
    );
}

fn version_one_member(
    name: DatatypeMemberName,
    offset: u32,
    rank: u8,
    extents: [u32; 4],
) -> CompoundV1Member {
    CompoundV1Member::new(
        DatatypeMessageMember::new(
            name,
            u64::from(offset),
            byte_description(DatatypeVersion::One),
        ),
        CompoundV1Shape::new(rank, extents).unwrap(),
    )
}

fn byte_description(version: DatatypeVersion) -> DatatypeMessage {
    DatatypeMessage::new(
        version,
        DatatypeProperties::FixedPoint(FixedPointType::u8()),
    )
    .unwrap()
}

const MEMBER_OFFSET: u32 = 0x0102_0304;
const DIMENSION_BLOCK_OFFSET: usize = 20;
