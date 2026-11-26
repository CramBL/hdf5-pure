use hdf5_pure_core::CompoundMember;
use hdf5_pure_core::DatatypeMemberName;
use hdf5_pure_core::{Datatype, FixedPointType, FormatError, VariableLengthType};
use hdf5_pure_format::__private::{
    self as format, DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
    OffsetWidth,
};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::containers::{self, CompoundField, MemberOffset};
use test_util::datatype::leaf::{self, Leaf, OpaqueTag, Order, VaxVersion, Version};
use test_util::legacy_compound::{self, SequenceMemberShape};

#[rstest]
#[case::integer(Leaf::Integer)]
#[case::little_float(Leaf::Float(Order::Little))]
#[case::big_float(Leaf::Float(Order::Big))]
#[case::time(Leaf::Time)]
#[case::string(Leaf::string())]
#[case::bitfield(Leaf::Bitfield)]
#[case::opaque(Leaf::Opaque(OpaqueTag::Aligned))]
fn datatype_compound_codecs_preserve_compound_properties(
    #[case] leaf: Leaf,
    #[values(DatatypeVersion::Three, DatatypeVersion::Four, DatatypeVersion::Five)]
    version: DatatypeVersion,
) {
    let child = leaf.bytes(Version::One);
    let bytes = containers::compound(
        version.get(),
        [2, 0, 0],
        32,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"first",
                offset: 0,
                datatype: &child,
            },
            CompoundField {
                name: "第二".as_bytes(),
                offset: 16,
                datatype: &child,
            },
        ],
    );
    let child = DatatypeMessage::parse(&child).unwrap().0;
    let expected = DatatypeMessage::new(
        version,
        DatatypeProperties::Compound {
            size: 32,
            members: vec![
                DatatypeMessageMember::new(DatatypeMemberName::from(c"first"), 0, child.clone()),
                DatatypeMessageMember::new(DatatypeMemberName::from(c"第二"), 16, child),
            ],
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, OffsetWidth::Eight);
}

#[rstest]
#[case::one(255, 251, MemberOffset::One)]
#[case::two(256, 252, MemberOffset::Two)]
#[case::three(65_536, 65_532, MemberOffset::Three)]
#[case::four(16_777_216, 16_777_212, MemberOffset::Four)]
#[case::maximum(u32::MAX, u64::from(u32::MAX) - 4, MemberOffset::Four)]
fn datatype_compound_codecs_preserve_member_offset_widths(
    #[case] size: u32,
    #[case] offset: u64,
    #[case] width: MemberOffset,
) {
    let child = Leaf::Integer.bytes(Version::One);
    let bytes = containers::compound(
        3,
        [1, 0, 0],
        size,
        width,
        &[CompoundField {
            name: b"x",
            offset,
            datatype: &child,
        }],
    );
    let expected = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Compound {
            size,
            members: vec![DatatypeMessageMember::new(
                DatatypeMemberName::from(c"x"),
                offset,
                DatatypeMessage::parse(&child).unwrap().0,
            )],
        },
    )
    .unwrap();
    assert_eq!(
        bytes[10..10 + width.get()],
        offset.to_le_bytes()[..width.get()]
    );
    assert_encoding(&bytes, expected, OffsetWidth::Eight);
}

#[test]
fn datatype_compound_parser_uses_source_member_extent() {
    let sequence = datatype::variable_length(3, [0; 3], 16, &Leaf::Integer.bytes(Version::One));
    let bytes = containers::compound(
        3,
        [1, 0, 0],
        10,
        MemberOffset::One,
        &[CompoundField {
            name: b"\xff",
            offset: 0,
            datatype: &sequence,
        }],
    );
    let expected = FormatError::CompoundFieldOutOfBounds {
        name: DatatypeMemberName::from(c"\xff"),
        offset: 0,
        field_size: 16,
        compound_size: 10,
    };
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
    assert_eq!(format::parse_datatype(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::scalar(SequenceMemberShape::Scalar, 1)]
#[case::implicit_array(SequenceMemberShape::ImplicitArray, 2)]
#[case::explicit_array(SequenceMemberShape::ExplicitArray, 2)]
fn compound_v1_sequence_member_fits_at_its_stored_size(
    #[case] shape: SequenceMemberShape,
    #[case] count: u32,
    #[values((2, 10), (4, 12), (8, 16))] width_and_descriptor_size: (usize, u32),
) {
    let (width, descriptor_size) = width_and_descriptor_size;
    let bytes = legacy_compound::sequence_member(width, SEQUENCE_MEMBER_OFFSET, shape);
    let sequence = Datatype::VariableLength(VariableLengthType::sequence(Datatype::FixedPoint(
        FixedPointType::u8(),
    )));
    let member = match shape {
        SequenceMemberShape::Scalar => sequence,
        SequenceMemberShape::ExplicitArray | SequenceMemberShape::ImplicitArray => {
            Datatype::Array {
                base_type: Box::new(sequence),
                dimensions: vec![2],
            }
        }
    };
    assert_eq!(
        format::parse_datatype(&bytes).unwrap(),
        (
            Datatype::Compound {
                size: SEQUENCE_MEMBER_OFFSET + count * descriptor_size,
                members: vec![CompoundMember::new(
                    DatatypeMemberName::from(c"arr"),
                    u64::from(SEQUENCE_MEMBER_OFFSET),
                    member,
                )],
            },
            bytes.len(),
        ),
    );
}

#[rstest]
#[case::scalar(SequenceMemberShape::Scalar, 1)]
#[case::implicit_array(SequenceMemberShape::ImplicitArray, 2)]
#[case::explicit_array(SequenceMemberShape::ExplicitArray, 2)]
fn compound_v1_sequence_member_rejects_a_one_byte_overrun(
    #[case] shape: SequenceMemberShape,
    #[case] count: u32,
    #[values((2, 10), (4, 12), (8, 16))] width_and_descriptor_size: (usize, u32),
) {
    let (width, descriptor_size) = width_and_descriptor_size;
    let field_size = count * descriptor_size;
    let compound_size = SEQUENCE_MEMBER_OFFSET + field_size - 1;
    let mut bytes = legacy_compound::sequence_member(width, SEQUENCE_MEMBER_OFFSET, shape);
    bytes[4..8].copy_from_slice(&compound_size.to_le_bytes());
    assert_eq!(
        format::parse_datatype(&bytes).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: DatatypeMemberName::from(c"arr"),
            offset: u64::from(SEQUENCE_MEMBER_OFFSET),
            field_size,
            compound_size,
        }
    );
}

#[rstest]
fn compound_v1_implicit_array_rejects_an_overlapping_member(
    #[values((2, 10), (4, 12), (8, 16))] width_and_descriptor_size: (usize, u32),
) {
    const DATATYPE_HEADER_SIZE: usize = 8;

    let (width, descriptor_size) = width_and_descriptor_size;
    let mut bytes = legacy_compound::sequence_member(
        width,
        SEQUENCE_MEMBER_OFFSET,
        SequenceMemberShape::ImplicitArray,
    );
    let mut scalar = legacy_compound::raw_member(
        SEQUENCE_MEMBER_OFFSET + 2 * descriptor_size,
        SEQUENCE_MEMBER_OFFSET + descriptor_size + 1,
        0,
        0,
        [0; 4],
        &leaf::unsigned_byte(Version::One),
    );
    let scalar = &mut scalar[DATATYPE_HEADER_SIZE..];
    scalar[..3].copy_from_slice(b"tag");
    bytes[1..3].copy_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(scalar);

    let expected = FormatError::CompoundFieldOverlap {
        first: DatatypeMemberName::from(c"arr"),
        second: DatatypeMemberName::from(c"tag"),
    };
    assert_eq!(format::parse_datatype(&bytes).unwrap_err(), expected);
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

const SEQUENCE_MEMBER_OFFSET: u32 = 4;

#[test]
fn datatype_compound_encoder_uses_selected_member_extent() {
    let sequence = datatype::variable_length(3, [0; 3], 10, &Leaf::Integer.bytes(Version::One));
    let bytes = containers::compound(
        3,
        [1, 0, 0],
        10,
        MemberOffset::One,
        &[CompoundField {
            name: b"\xff",
            offset: 0,
            datatype: &sequence,
        }],
    );
    let message = DatatypeMessage::parse(&bytes).unwrap().0;
    assert_eq!(
        message.serialize(OffsetWidth::Eight).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: DatatypeMemberName::from(c"\xff"),
            offset: 0,
            field_size: 16,
            compound_size: 10
        }
    );
}

#[rstest]
#[case::reserved(0x40, Version::Five, FormatError::InvalidByteOrder(0x40))]
#[case::old_vax(0x41, Version::Two, FormatError::InvalidDatatypeVersion { class: 1, version: 2 })]
fn datatype_compound_parser_preserves_float_order_errors(
    #[case] order: u8,
    #[case] version: Version,
    #[case] expected: FormatError,
) {
    let mut parent = Leaf::Float(Order::Little).bytes(version);
    parent[1] |= order;
    let bytes = containers::compound(
        5,
        [1, 0, 0],
        8,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &parent,
        }],
    );
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
fn datatype_compound_codecs_preserve_vax_members(
    #[values(VaxVersion::Three, VaxVersion::Four, VaxVersion::Five)] version: VaxVersion,
) {
    let parent = leaf::vax_bytes(version);
    let bytes = containers::compound(
        5,
        [1, 0, 0],
        8,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &parent,
        }],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes)
            .unwrap()
            .0
            .serialize(OffsetWidth::Eight)
            .unwrap(),
        bytes
    );
}

#[rstest]
fn datatype_compound_codecs_preserve_recursive_mixtures(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = Leaf::Integer.bytes(Version::One);
    let sequence = datatype::variable_length(4, [0; 3], 10, &parent);
    let array = datatype::array(4, [0; 3], 20, 1, &[2], &sequence);
    let complex = datatype::complex(5, [1, 0, 0], 16, &datatype::f64_le());
    let inner = containers::compound(
        5,
        [2, 0, 0],
        64,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"array\xff",
                offset: 0,
                datatype: &array,
            },
            CompoundField {
                name: b"complex\xfe",
                offset: 32,
                datatype: &complex,
            },
        ],
    );
    let outer = containers::compound(
        5,
        [1, 0, 0],
        64,
        MemberOffset::One,
        &[CompoundField {
            name: b"inner\xff",
            offset: 0,
            datatype: &inner,
        }],
    );
    let array = datatype::array(5, [0; 3], 128, 1, &[2], &outer);
    let bytes = datatype::variable_length(Version::Five.get(), [0; 3], 10, &array);
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(
        message.to_datatype().unwrap(),
        format::parse_datatype(&bytes).unwrap().0
    );
    let descriptor = 8 + u32::from(width.get());
    let sequence = datatype::variable_length(4, [0; 3], descriptor, &parent);
    let array = datatype::array(4, [0; 3], descriptor * 2, 1, &[2], &sequence);
    let inner = containers::compound(
        5,
        [2, 0, 0],
        64,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"array\xff",
                offset: 0,
                datatype: &array,
            },
            CompoundField {
                name: b"complex\xfe",
                offset: 32,
                datatype: &complex,
            },
        ],
    );
    let outer = containers::compound(
        5,
        [1, 0, 0],
        64,
        MemberOffset::One,
        &[CompoundField {
            name: b"inner\xff",
            offset: 0,
            datatype: &inner,
        }],
    );
    let array = datatype::array(5, [0; 3], 128, 1, &[2], &outer);
    assert_eq!(
        message.serialize(width).unwrap(),
        datatype::variable_length(Version::Five.get(), [0; 3], descriptor, &array)
    );
}

#[test]
fn datatype_compound_codecs_preserve_high_member_count_byte() {
    let count = 257u16;
    let parent = Leaf::Integer.bytes(Version::One);
    let names = (0..count)
        .map(|index| format!("member{index}"))
        .collect::<Vec<_>>();
    let fields = names
        .iter()
        .enumerate()
        .map(|(index, name)| CompoundField {
            name: name.as_bytes(),
            offset: u64::try_from(index).unwrap() * 4,
            datatype: &parent,
        })
        .collect::<Vec<_>>();
    let [low, high] = count.to_le_bytes();
    let bytes = containers::compound(3, [low, high, 0], 1028, MemberOffset::Two, &fields);
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let DatatypeProperties::Compound { members, .. } = message.properties() else {
        panic!("{message:?}")
    };
    assert_eq!(members.len(), usize::from(count));
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
}

#[rstest]
#[case::positive_size(4, FormatError::InvalidCompoundMemberCount { count: 0 })]
#[case::zero_size(0, FormatError::ZeroSizedDatatype { class: 6 })]
fn datatype_compound_codecs_reject_empty_member_sets(
    #[case] size: u32,
    #[case] expected: FormatError,
) {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Compound {
                size,
                members: vec![]
            }
        )
        .unwrap_err(),
        expected
    );
    let bytes = containers::compound(3, [0; 3], size, MemberOffset::One, &[]);
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[test]
fn datatype_compound_codecs_reject_duplicate_member_names() {
    let parent = Leaf::Integer.bytes(Version::One);
    let child = DatatypeMessage::parse(&parent).unwrap().0;
    let members = vec![
        DatatypeMessageMember::new(DatatypeMemberName::from(c"\xff"), 0, child.clone()),
        DatatypeMessageMember::new(DatatypeMemberName::from(c"\xff"), 4, child),
    ];
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Compound { size: 8, members }
        )
        .unwrap_err(),
        FormatError::DuplicateCompoundField(DatatypeMemberName::from(c"\xff"))
    );
    let bytes = containers::compound(
        3,
        [2, 0, 0],
        8,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"\xff",
                offset: 0,
                datatype: &parent,
            },
            CompoundField {
                name: b"\xff",
                offset: 4,
                datatype: &parent,
            },
        ],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::DuplicateCompoundField(DatatypeMemberName::from(c"\xff"))
    );
}

#[rstest]
#[case::one(Version::One)]
#[case::two(Version::Two)]
fn datatype_compound_parser_rejects_empty_legacy_layouts(#[case] version: Version) {
    assert_eq!(
        DatatypeMessage::parse(&datatype::header(
            datatype::Class::COMPOUND,
            version.get(),
            [0; 3],
            4
        ))
        .unwrap_err(),
        FormatError::InvalidCompoundMemberCount { count: 0 }
    );
}

#[rstest]
#[case::low([1, 0, 1], 0x01_0001)]
#[case::high([0, 0, 0x80], 0x80_0000)]
fn datatype_compound_parser_rejects_reserved_flags(#[case] flags: [u8; 3], #[case] bits: u32) {
    assert_eq!(
        DatatypeMessage::parse(&datatype::header(datatype::Class::COMPOUND, 3, flags, 4))
            .unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class: 6,
            version: 3,
            flags: bits
        }
    );
}

#[test]
fn datatype_compound_parser_rejects_newer_children() {
    let parent = Leaf::Integer.bytes(Version::Five);
    let bytes = containers::compound(
        3,
        [1, 0, 0],
        4,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &parent,
        }],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::DatatypeChildVersion {
            class: 6,
            version: 3,
            child_version: 5
        }
    );
}

#[rstest]
fn datatype_compound_codecs_preserve_member_name_bytes(
    #[values(DatatypeVersion::Three, DatatypeVersion::Four, DatatypeVersion::Five)]
    version: DatatypeVersion,
) {
    let all: Vec<u8> = (1..=u8::MAX).collect();
    let names = [
        b"first".as_slice(),
        "第二".as_bytes(),
        b"\xff",
        b"\xfe",
        "\u{fffd}".as_bytes(),
        &all,
        b"",
    ];
    let integer = leaf::unsigned_byte(Version::One);
    let wide_integer = Leaf::Integer.bytes(Version::Two);
    let fields: Vec<_> = names
        .into_iter()
        .enumerate()
        .map(|(index, name)| CompoundField {
            name,
            offset: u64::try_from(index).unwrap(),
            datatype: if index == names.len() - 1 {
                &wide_integer
            } else {
                &integer
            },
        })
        .collect();
    let size = 10;
    let count = u8::try_from(fields.len()).unwrap();
    let bytes = containers::compound(
        version.get(),
        [count, 0, 0],
        size,
        MemberOffset::One,
        &fields,
    );
    let expected = Datatype::Compound {
        size,
        members: fields
            .iter()
            .map(|field| {
                CompoundMember::new(
                    DatatypeMemberName::new(field.name).unwrap(),
                    field.offset,
                    format::parse_datatype(field.datatype).unwrap().0,
                )
            })
            .collect(),
    };
    let mut input = bytes.clone();
    input.push(0xff);
    let (message, consumed) = DatatypeMessage::parse(&input).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.version(), version);
    let DatatypeProperties::Compound { members, .. } = message.properties() else {
        panic!("expected compound properties");
    };
    assert_eq!(members.len(), fields.len());
    for (member, field) in members.iter().zip(fields) {
        assert_eq!(member.name().as_bytes(), field.name);
        assert_eq!(member.byte_offset(), field.offset);
        assert_eq!(
            member.datatype().serialize(OffsetWidth::Eight).unwrap(),
            field.datatype
        );
    }
    assert_eq!(message.to_datatype().unwrap(), expected);
    assert_eq!(
        format::parse_datatype(&input).unwrap(),
        (expected, bytes.len())
    );
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
}

#[test]
fn prepared_compound_write_preserves_output_outside_its_extent() {
    let child = leaf::unsigned_byte(Version::One);
    let bytes = containers::compound(
        Version::Three.get(),
        [1, 0, 0],
        1,
        MemberOffset::One,
        &[CompoundField {
            name: b"a\xff",
            offset: 0,
            datatype: &child,
        }],
    );
    let (message, _) = DatatypeMessage::parse(&bytes).unwrap();
    let prepared = message.prepare(OffsetWidth::Eight).unwrap();
    let expected = bytes.len();
    let available = expected - 1;
    let mut output = vec![0xaa; expected + 1];
    assert_eq!(prepared.encoded_size(), expected);
    assert_eq!(
        prepared.write_into(&mut output[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
    assert_eq!(output, vec![0xaa; expected + 1]);
    assert_eq!(prepared.write_into(&mut output).unwrap(), expected);
    assert_eq!(&output[..expected], bytes);
    assert_eq!(output[expected], 0xaa);
}

#[test]
fn compound_member_text_access_returns_the_original_utf8_error() {
    let name = DatatypeMemberName::from(c"x\xff");
    let expected = core::str::from_utf8(name.as_bytes()).unwrap_err();
    assert_eq!(name.as_str(), Err(expected));
    assert_eq!(name.as_bytes(), b"x\xff");
}

#[rstest]
fn canonical_encoding_preserves_legacy_compound_member_name_bytes(
    #[values(Version::One, Version::Two)] version: Version,
) {
    let child = leaf::unsigned_byte(Version::One);
    let names = [b"\xff234567".as_slice(), b"\xfe2345678", b"\xff23456789"];
    let fields: Vec<_> = names
        .into_iter()
        .zip([5, 0, 2])
        .map(|(name, offset)| CompoundField {
            name,
            offset,
            datatype: &child,
        })
        .collect();
    let flags = [u8::try_from(fields.len()).unwrap(), 0, 0];
    let size = 6;
    let legacy = containers::compound(version.get(), flags, size, MemberOffset::Four, &fields);
    let canonical = containers::compound(
        Version::Three.get(),
        flags,
        size,
        MemberOffset::One,
        &fields,
    );
    let expected = Datatype::Compound {
        size,
        members: fields
            .iter()
            .map(|field| {
                CompoundMember::new(
                    DatatypeMemberName::new(field.name).unwrap(),
                    field.offset,
                    Datatype::FixedPoint(FixedPointType::u8()),
                )
            })
            .collect(),
    };
    let (datatype, consumed) = format::parse_datatype(&legacy).unwrap();
    assert_eq!(consumed, legacy.len());
    assert_eq!(datatype, expected);
    assert_eq!(
        format::serialize_datatype_with_offset_width(&datatype, OffsetWidth::Eight).unwrap(),
        canonical
    );
}

#[rstest]
#[case::missing_terminator(16, 9)]
#[case::missing_padding(17, 24)]
#[case::truncated_offset(27, 28)]
fn semantic_compound_parser_reports_incomplete_legacy_name_and_offset_fields(
    #[case] available: usize,
    #[case] expected: usize,
    #[values(Version::One, Version::Two)] version: Version,
) {
    let child = leaf::unsigned_byte(Version::One);
    let bytes = containers::compound(
        version.get(),
        [1, 0, 0],
        1,
        MemberOffset::Four,
        &[CompoundField {
            name: b"\xff2345678",
            offset: 0,
            datatype: &child,
        }],
    );
    assert_eq!(
        format::parse_datatype(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[rstest]
#[case::name(8, 9)]
#[case::unterminated_name(9, 10)]
#[case::offset(10, 11)]
#[case::child(22, 23)]
fn datatype_compound_parser_reports_incomplete_fields(
    #[case] available: usize,
    #[case] expected: usize,
) {
    let parent = Leaf::Integer.bytes(Version::One);
    let bytes = containers::compound(
        3,
        [1, 0, 0],
        4,
        MemberOffset::One,
        &[CompoundField {
            name: b"\xff",
            offset: 0,
            datatype: &parent,
        }],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[test]
fn datatype_compound_parser_rejects_zero_size() {
    assert_eq!(
        DatatypeMessage::parse(&datatype::header(datatype::Class::COMPOUND, 3, [0; 3], 0))
            .unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 6 }
    );
}

#[rstest]
#[case::little(Leaf::Float(Order::Little))]
#[case::big(Leaf::Float(Order::Big))]
fn datatype_compound_codecs_preserve_float_member_versions(
    #[case] leaf: Leaf,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
) {
    let parent = leaf.bytes(version);
    let bytes = containers::compound(
        5,
        [1, 0, 0],
        8,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &parent,
        }],
    );
    let expected = DatatypeMessage::new(
        DatatypeVersion::Five,
        DatatypeProperties::Compound {
            size: 8,
            members: vec![DatatypeMessageMember::new(
                DatatypeMemberName::from(c"x"),
                0,
                DatatypeMessage::parse(&parent).unwrap().0,
            )],
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, OffsetWidth::Eight);
}

#[test]
fn datatype_compound_codecs_bound_recursion() {
    let mut bytes = leaf::unsigned_byte(Version::One);
    let mut expected = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::FixedPoint(FixedPointType::u8()),
    )
    .unwrap();
    for _ in 0..63 {
        bytes = containers::compound(
            Version::Three.get(),
            [1, 0, 0],
            1,
            MemberOffset::One,
            &[CompoundField {
                name: b"x",
                offset: 0,
                datatype: &bytes,
            }],
        );
        expected = DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Compound {
                size: 1,
                members: vec![DatatypeMessageMember::new(
                    DatatypeMemberName::from(c"x"),
                    0,
                    expected,
                )],
            },
        )
        .unwrap();
    }
    assert_eq!(
        format::parse_datatype(&bytes).unwrap(),
        (expected.to_datatype().unwrap(), bytes.len())
    );
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message, expected);
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
    let too_deep = containers::compound(
        Version::Three.get(),
        [1, 0, 0],
        1,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &bytes,
        }],
    );
    assert_eq!(
        DatatypeMessage::parse(&too_deep).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
    assert_eq!(
        format::parse_datatype(&too_deep).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
    let too_deep = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Compound {
            size: 1,
            members: vec![DatatypeMessageMember::new(
                DatatypeMemberName::from(c"x"),
                0,
                message,
            )],
        },
    )
    .unwrap();
    assert_eq!(
        too_deep.serialize(OffsetWidth::Eight).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
}

#[test]
fn datatype_parsers_accept_a_v1_compound_in_a_v3_compound() {
    let child = legacy_compound::array_member();
    let bytes = containers::compound(
        Version::Three.get(),
        [1, 0, 0],
        20,
        MemberOffset::One,
        &[CompoundField {
            name: b"inner",
            offset: 4,
            datatype: &child,
        }],
    );
    let expected = Datatype::Compound {
        size: 20,
        members: vec![CompoundMember::new(
            DatatypeMemberName::from(c"inner"),
            4,
            Datatype::Compound {
                size: 16,
                members: vec![CompoundMember::new(
                    DatatypeMemberName::from(c"arr"),
                    0,
                    Datatype::Array {
                        base_type: Box::new(Datatype::FixedPoint(FixedPointType::i32())),
                        dimensions: vec![4],
                    },
                )],
            },
        )],
    };
    assert_eq!(
        format::parse_datatype(&bytes).unwrap(),
        (expected.clone(), bytes.len())
    );
    let (retained, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let DatatypeProperties::Compound { members, .. } = retained.properties() else {
        panic!("expected a compound datatype");
    };
    assert_eq!(members[0].datatype().version(), DatatypeVersion::One);
    assert_eq!(retained.to_datatype().unwrap(), expected);
    assert_eq!(
        retained.prepare(OffsetWidth::Eight).unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: datatype::Class::COMPOUND.0,
            version: Version::One.get(),
        }
    );
}

fn assert_encoding(bytes: &[u8], expected: DatatypeMessage, width: OffsetWidth) {
    let mut input = bytes.to_vec();
    input.push(0xff);
    let (message, consumed) = DatatypeMessage::parse(&input).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message, expected);
    assert_eq!(message.serialize(width).unwrap(), bytes);
}
