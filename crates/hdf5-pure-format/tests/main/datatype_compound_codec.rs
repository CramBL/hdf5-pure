use hdf5_pure_core::FormatError;
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::containers::{self, CompoundField, MemberOffset};
use test_util::datatype::leaf::{self, Leaf, OpaqueTag, Order, VaxVersion, Version};

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
                DatatypeMessageMember::new("first".into(), 0, child.clone()).unwrap(),
                DatatypeMessageMember::new("第二".into(), 16, child).unwrap(),
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
            members: vec![
                DatatypeMessageMember::new(
                    "x".into(),
                    offset,
                    DatatypeMessage::parse(&child).unwrap().0,
                )
                .unwrap(),
            ],
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
            name: b"x",
            offset: 0,
            datatype: &sequence,
        }],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: "x".into(),
            offset: 0,
            field_size: 16,
            compound_size: 10
        }
    );
}

#[test]
fn datatype_compound_encoder_uses_selected_member_extent() {
    let sequence = datatype::variable_length(3, [0; 3], 10, &Leaf::Integer.bytes(Version::One));
    let bytes = containers::compound(
        3,
        [1, 0, 0],
        10,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &sequence,
        }],
    );
    let message = DatatypeMessage::parse(&bytes).unwrap().0;
    assert_eq!(
        message.serialize(OffsetWidth::Eight).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: "x".into(),
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
                name: b"array",
                offset: 0,
                datatype: &array,
            },
            CompoundField {
                name: b"complex",
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
            name: b"inner",
            offset: 0,
            datatype: &inner,
        }],
    );
    let bytes = datatype::array(5, [0; 3], 128, 1, &[2], &outer);
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
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
                name: b"array",
                offset: 0,
                datatype: &array,
            },
            CompoundField {
                name: b"complex",
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
            name: b"inner",
            offset: 0,
            datatype: &inner,
        }],
    );
    assert_eq!(
        message.serialize(width).unwrap(),
        datatype::array(5, [0; 3], 128, 1, &[2], &outer)
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
        DatatypeMessageMember::new("x".into(), 0, child.clone()).unwrap(),
        DatatypeMessageMember::new("x".into(), 4, child).unwrap(),
    ];
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Compound { size: 8, members }
        )
        .unwrap_err(),
        FormatError::DuplicateCompoundField("x".into())
    );
    let bytes = containers::compound(
        3,
        [2, 0, 0],
        8,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"x",
                offset: 0,
                datatype: &parent,
            },
            CompoundField {
                name: b"x",
                offset: 4,
                datatype: &parent,
            },
        ],
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::DuplicateCompoundField("x".into())
    );
}

#[rstest]
#[case::one(1)]
#[case::two(2)]
fn datatype_compound_parser_rejects_legacy_layouts(#[case] version: u8) {
    assert_eq!(
        DatatypeMessage::parse(&datatype::header(
            datatype::Class::COMPOUND,
            version,
            [0; 3],
            4
        ))
        .unwrap_err(),
        FormatError::InvalidDatatypeVersion { class: 6, version }
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

#[test]
fn datatype_compound_parser_rejects_non_utf8_member_names() {
    let parent = Leaf::Integer.bytes(Version::One);
    let bytes = containers::compound(
        3,
        [1, 0, 0],
        4,
        MemberOffset::One,
        &[CompoundField {
            name: &[0xff],
            offset: 0,
            datatype: &parent,
        }],
    );
    let error = DatatypeMessage::parse(&bytes).unwrap_err();
    let FormatError::UnsupportedDatatypeMemberEncoding {
        class,
        offset,
        source,
    } = &error
    else {
        panic!("{error:?}")
    };
    assert_eq!(*class, 6);
    assert_eq!(*offset, 8);
    assert_eq!(source.valid_up_to(), 0);
    assert_eq!(source.error_len(), Some(1));
    #[cfg(feature = "std")]
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<core::str::Utf8Error>()
            .unwrap(),
        source
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
            name: b"x",
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
            members: vec![
                DatatypeMessageMember::new(
                    "x".into(),
                    0,
                    DatatypeMessage::parse(&parent).unwrap().0,
                )
                .unwrap(),
            ],
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, OffsetWidth::Eight);
}

#[test]
fn datatype_compound_codecs_bound_recursion() {
    let bytes = (0..63).fold(Leaf::Integer.bytes(Version::One), |parent, _| {
        containers::compound(
            3,
            [1, 0, 0],
            4,
            MemberOffset::One,
            &[CompoundField {
                name: b"x",
                offset: 0,
                datatype: &parent,
            }],
        )
    });
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
    let too_deep = containers::compound(
        3,
        [1, 0, 0],
        4,
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
    let too_deep = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Compound {
            size: 4,
            members: vec![DatatypeMessageMember::new("x".into(), 0, message).unwrap()],
        },
    )
    .unwrap();
    assert_eq!(
        too_deep.serialize(OffsetWidth::Eight).unwrap_err(),
        FormatError::NestingDepthExceeded
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
