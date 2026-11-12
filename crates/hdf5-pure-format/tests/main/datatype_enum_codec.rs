use hdf5_pure_core::__private;
use hdf5_pure_core::{EnumMember, FormatError};
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::containers::{self, CompoundField, EnumValue, MemberOffset};
use test_util::datatype::leaf::{Leaf, Order, Version};

#[rstest]
#[case::three_one(DatatypeVersion::Three, Version::One)]
#[case::three_two(DatatypeVersion::Three, Version::Two)]
#[case::three_three(DatatypeVersion::Three, Version::Three)]
#[case::four_one(DatatypeVersion::Four, Version::One)]
#[case::four_two(DatatypeVersion::Four, Version::Two)]
#[case::four_three(DatatypeVersion::Four, Version::Three)]
#[case::four_four(DatatypeVersion::Four, Version::Four)]
#[case::five_one(DatatypeVersion::Five, Version::One)]
#[case::five_two(DatatypeVersion::Five, Version::Two)]
#[case::five_three(DatatypeVersion::Five, Version::Three)]
#[case::five_four(DatatypeVersion::Five, Version::Four)]
#[case::five_five(DatatypeVersion::Five, Version::Five)]
fn datatype_enum_codecs_preserve_versions_and_integer_properties(
    #[case] version: DatatypeVersion,
    #[case] child_version: Version,
    #[values(Leaf::Integer, Leaf::LittleInteger)] leaf: Leaf,
) {
    let child = leaf.bytes(child_version);
    let members = [
        EnumValue {
            name: b"first",
            value: &[0xf8, 0xff, 0xff, 0xff],
        },
        EnumValue {
            name: "第二".as_bytes(),
            value: &[0x38, 0, 0, 0],
        },
    ];
    let bytes = containers::enumeration(version.get(), [2, 0, 0], 4, &child, &members);
    let expected = checked(version, &child, &members).unwrap();
    assert_encoding(&bytes, expected);
}

#[test]
fn datatype_enum_codecs_preserve_wide_values() {
    let mut child = datatype::header(datatype::Class::FIXED_POINT, 1, [8, 0, 0], 16);
    child.extend([0, 0, 128, 0]);
    let members = [EnumValue {
        name: b"wide",
        value: &[0xa5; 16],
    }];
    let bytes = containers::enumeration(5, [1, 0, 0], 16, &child, &members);
    assert_encoding(
        &bytes,
        checked(DatatypeVersion::Five, &child, &members).unwrap(),
    );
}

#[rstest]
#[case::high_byte(257)]
#[case::maximum(u16::MAX)]
fn datatype_enum_codecs_preserve_member_counts(#[case] count: u16) {
    let child = Leaf::LittleInteger.bytes(Version::One);
    let names = (0..count)
        .map(|index| format!("member{index}"))
        .collect::<Vec<_>>();
    let values = (0..count)
        .map(|index| (u32::from(index) << 3).to_le_bytes())
        .collect::<Vec<_>>();
    let members = names
        .iter()
        .zip(&values)
        .map(|(name, value)| EnumValue {
            name: name.as_bytes(),
            value,
        })
        .collect::<Vec<_>>();
    let [low, high] = count.to_le_bytes();
    let bytes = containers::enumeration(3, [low, high, 0], 4, &child, &members);
    assert_encoding(
        &bytes,
        checked(DatatypeVersion::Three, &child, &members).unwrap(),
    );
}

#[rstest]
fn datatype_enum_codecs_preserve_nested_mixtures(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let child = Leaf::Integer.bytes(Version::Two);
    let enumeration = containers::enumeration(
        3,
        [1, 0, 0],
        4,
        &child,
        &[EnumValue {
            name: b"x",
            value: &[1, 2, 3, 4],
        }],
    );
    let array = datatype::array(4, [0; 3], 8, 1, &[2], &enumeration);
    let descriptor = 8 + u32::from(width.get());
    let sequence = datatype::variable_length(4, [0; 3], descriptor, &array);
    let compound = containers::compound(
        5,
        [1, 0, 0],
        descriptor,
        MemberOffset::One,
        &[CompoundField {
            name: b"values",
            offset: 0,
            datatype: &sequence,
        }],
    );
    let bytes = datatype::array(5, [0; 3], descriptor * 2, 1, &[2], &compound);
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.serialize(width).unwrap(), bytes);
    let DatatypeProperties::Array {
        base_type,
        dimensions,
    } = message.properties()
    else {
        panic!("{message:?}")
    };
    assert_eq!(dimensions, &[2]);
    assert_eq!(base_type.version(), DatatypeVersion::Five);
    let DatatypeProperties::Compound { size, members } = base_type.properties() else {
        panic!("{base_type:?}")
    };
    assert_eq!(*size, descriptor);
    let sequence = members[0].datatype();
    assert_eq!(sequence.version(), DatatypeVersion::Four);
    let DatatypeProperties::VariableLengthSequence { base_type } = sequence.properties() else {
        panic!("{sequence:?}")
    };
    assert_eq!(base_type.version(), DatatypeVersion::Four);
    let DatatypeProperties::Array {
        base_type,
        dimensions,
    } = base_type.properties()
    else {
        panic!("{base_type:?}")
    };
    assert_eq!(dimensions, &[2]);
    assert_eq!(base_type.version(), DatatypeVersion::Three);
    let DatatypeProperties::Enumeration { base_type, members } = base_type.properties() else {
        panic!("{base_type:?}")
    };
    assert_eq!(base_type.version(), DatatypeVersion::Two);
    assert_eq!(
        members,
        &[__private::enum_member("x".into(), vec![1, 2, 3, 4])]
    );
}

#[rstest]
#[case::empty(vec![], FormatError::InvalidEnumerationMemberCount { count: 0 })]
#[case::empty_name(vec![("", vec![0; 4])], FormatError::EmptyEnumMemberName { index: 0 })]
#[case::duplicate_name(vec![("x", vec![0; 4]), ("x", vec![1; 4])], FormatError::DuplicateEnumMemberName("x".into()))]
#[case::duplicate_value(vec![("x", vec![0; 4]), ("y", vec![0; 4])], FormatError::DuplicateEnumMemberValue { first: "x".into(), second: "y".into() })]
fn datatype_enum_codecs_reject_invalid_members(
    #[case] members: Vec<(&str, Vec<u8>)>,
    #[case] expected: FormatError,
) {
    let child = Leaf::Integer.bytes(Version::One);
    let fields = members
        .iter()
        .map(|(name, value)| EnumValue {
            name: name.as_bytes(),
            value,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        checked(DatatypeVersion::Three, &child, &fields).unwrap_err(),
        expected
    );
    let [low, high] = u16::try_from(fields.len()).unwrap().to_le_bytes();
    let bytes = containers::enumeration(3, [low, high, 0], 4, &child, &fields);
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::one(1)]
#[case::two(2)]
fn datatype_enum_parser_rejects_legacy_layouts(#[case] version: u8) {
    assert_eq!(
        DatatypeMessage::parse(&datatype::header(
            datatype::Class::ENUMERATED,
            version,
            [0; 3],
            4
        ))
        .unwrap_err(),
        FormatError::InvalidDatatypeVersion { class: 8, version }
    );
}

#[rstest]
#[case::low([1, 0, 1], 0x01_0001)]
#[case::high([0, 0, 0x80], 0x80_0000)]
fn datatype_enum_parser_rejects_reserved_flags(#[case] flags: [u8; 3], #[case] bits: u32) {
    assert_eq!(
        DatatypeMessage::parse(&datatype::header(datatype::Class::ENUMERATED, 3, flags, 4))
            .unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class: 8,
            version: 3,
            flags: bits
        }
    );
}

#[test]
fn datatype_enum_parser_rejects_newer_bases() {
    let child = Leaf::Integer.bytes(Version::Five);
    let members = [EnumValue {
        name: b"x",
        value: &[0; 4],
    }];
    let expected = FormatError::DatatypeChildVersion {
        class: 8,
        version: 3,
        child_version: 5,
    };
    assert_eq!(
        checked(DatatypeVersion::Three, &child, &members).unwrap_err(),
        expected
    );
    assert_eq!(
        DatatypeMessage::parse(&containers::enumeration(3, [1, 0, 0], 4, &child, &members))
            .unwrap_err(),
        expected
    );
}

#[test]
fn datatype_enum_parser_rejects_size_mismatch() {
    let child = Leaf::Integer.bytes(Version::One);
    assert_eq!(
        DatatypeMessage::parse(&containers::enumeration(3, [0; 3], 8, &child, &[])).unwrap_err(),
        FormatError::EnumBaseSizeMismatch {
            size: 8,
            base_size: 4
        }
    );
}

#[rstest]
#[case::float(Leaf::Float(Order::Little))]
#[case::bitfield(Leaf::Bitfield)]
#[case::string(Leaf::string())]
fn datatype_enum_codecs_reject_noninteger_bases(#[case] leaf: Leaf) {
    let child = leaf.bytes(Version::One);
    assert_eq!(
        checked(DatatypeVersion::Three, &child, &[]).unwrap_err(),
        FormatError::EnumBaseNotInteger
    );
    assert_eq!(
        DatatypeMessage::parse(&containers::enumeration(3, [0; 3], 4, &child, &[])).unwrap_err(),
        FormatError::EnumBaseNotInteger
    );
}

#[rstest]
#[case::reserved(0x40, Version::Five, FormatError::InvalidByteOrder(0x40))]
#[case::old_vax(0x41, Version::Two, FormatError::InvalidDatatypeVersion { class: 1, version: 2 })]
fn datatype_enum_parser_preserves_float_base_errors(
    #[case] order: u8,
    #[case] version: Version,
    #[case] expected: FormatError,
) {
    let mut child = Leaf::Float(Order::Little).bytes(version);
    child[1] |= order;
    assert_eq!(
        DatatypeMessage::parse(&containers::enumeration(5, [0; 3], 8, &child, &[])).unwrap_err(),
        expected
    );
}

#[test]
fn datatype_enum_parser_reports_non_utf8_names() {
    let child = Leaf::Integer.bytes(Version::One);
    let bytes = containers::enumeration(
        3,
        [1, 0, 0],
        4,
        &child,
        &[EnumValue {
            name: &[0xff],
            value: &[0; 4],
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
    assert_eq!(*class, 8);
    assert_eq!(*offset, 20);
    assert_eq!(source.valid_up_to(), 0);
    assert_eq!(source.error_len(), Some(1));
}

#[rstest]
#[case::parent(8, 9)]
#[case::name(20, 21)]
#[case::unterminated_name(21, 22)]
#[case::value(25, 26)]
fn datatype_enum_parser_reports_incomplete_fields(
    #[case] available: usize,
    #[case] expected: usize,
) {
    let child = Leaf::Integer.bytes(Version::One);
    let bytes = containers::enumeration(
        3,
        [1, 0, 0],
        4,
        &child,
        &[EnumValue {
            name: b"x",
            value: &[0; 4],
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
fn datatype_enum_parser_checks_wide_value_bounds_before_copying() {
    let mut child = Leaf::Integer.bytes(Version::One);
    child[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    let bytes = containers::enumeration(
        3,
        [1, 0, 0],
        u32::MAX,
        &child,
        &[EnumValue {
            name: b"x",
            value: &[],
        }],
    );
    let end = u64::from(u32::MAX) + 22;
    let expected = if let Ok(expected) = usize::try_from(end) {
        FormatError::UnexpectedEof {
            expected,
            available: bytes.len(),
        }
    } else {
        FormatError::ValueTooLargeForPlatform {
            value: end,
            target: "usize",
        }
    };
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

fn checked(
    version: DatatypeVersion,
    child: &[u8],
    members: &[EnumValue<'_>],
) -> Result<DatatypeMessage, FormatError> {
    let members: Vec<EnumMember> = members
        .iter()
        .map(|member| {
            let EnumValue { name, value } = member;
            __private::enum_member(core::str::from_utf8(name).unwrap().into(), value.to_vec())
        })
        .collect();
    DatatypeMessage::new(
        version,
        DatatypeProperties::Enumeration {
            base_type: Box::new(DatatypeMessage::parse(child).unwrap().0),
            members,
        },
    )
}

fn assert_encoding(bytes: &[u8], expected: DatatypeMessage) {
    let mut input = bytes.to_vec();
    input.push(0xff);
    let (message, consumed) = DatatypeMessage::parse(&input).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message, expected);
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
}

#[test]
fn datatype_enum_construction_rejects_unencodable_member_counts() {
    let child = DatatypeMessage::parse(&Leaf::Integer.bytes(Version::One))
        .unwrap()
        .0;
    let count = usize::from(u16::MAX) + 1;
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Enumeration {
                base_type: Box::new(child),
                members: vec![__private::enum_member("x".into(), vec![0; 4]); count],
            }
        )
        .unwrap_err(),
        FormatError::ValueTooLargeForRepresentation {
            value: u64::try_from(count).unwrap(),
            target: "u16"
        }
    );
}
