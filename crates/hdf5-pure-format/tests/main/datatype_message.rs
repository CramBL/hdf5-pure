use hdf5_pure_core::__private;
use hdf5_pure_core::{
    BitFieldType, CharacterSet, ComplexType, Datatype, DatatypeByteOrder, DatatypeMemberName,
    EnumMember, EnumerationType, FixedPointType, FloatingPointType, FormatError, StringPadding,
    TimeType, VariableLengthType,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
    FloatingPointByteOrder, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype::containers::{self, CompoundField, EnumValue, MemberOffset};
use test_util::datatype::float_context::Context;
use test_util::datatype::leaf::{Leaf, OpaqueTag, Version};
use test_util::datatype::{self, Class, VariableLengthSequence};

#[rstest]
#[case::one(1, DatatypeVersion::One)]
#[case::two(2, DatatypeVersion::Two)]
#[case::three(3, DatatypeVersion::Three)]
#[case::four(4, DatatypeVersion::Four)]
#[case::five(5, DatatypeVersion::Five)]
fn datatype_message_versions_are_bounded(#[case] raw: u8, #[case] expected: DatatypeVersion) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    assert_eq!(version, expected);
    assert_eq!(version.get(), raw);
}

#[rstest]
#[case::zero(0)]
#[case::six(6)]
#[case::maximum(255)]
fn invalid_message_versions_return_the_original_value(#[case] raw: u8) {
    assert_eq!(
        DatatypeVersion::try_from(raw).unwrap_err(),
        FormatError::InvalidDatatypeMessageVersion(raw),
    );
}

#[rstest]
#[case::unsupported_version(Class::FIXED_POINT, 15)]
#[case::unknown_class(Class(13), Version::One.get())]
fn incomplete_datatype_headers_precede_version_and_class_errors(
    #[case] class: Class,
    #[case] version: u8,
    #[values(0, 1, 4, 7)] available: usize,
) {
    let header = datatype::header(class, version, [0; 3], 4);
    assert_decode_error(
        &header[..available],
        FormatError::UnexpectedEof {
            expected: header.len(),
            available,
        },
    );
}

#[rstest]
#[case::valid_version(Version::One.get(), FormatError::InvalidDatatypeClass(13))]
#[case::unsupported_version(15, FormatError::InvalidDatatypeMessageVersion(15))]
fn datatype_header_checks_version_before_unknown_class(
    #[case] version: u8,
    #[case] error: FormatError,
) {
    assert_decode_error(&datatype::header(Class(13), version, [0; 3], 4), error);
}

#[rstest]
#[case::zero(0)]
#[case::six(6)]
#[case::fifteen(15)]
fn complete_complex_headers_reject_unsupported_message_versions(#[case] version: u8) {
    let bytes = datatype::header(Class::COMPLEX, version, [0; 3], 16);
    assert_decode_error(&bytes, FormatError::InvalidDatatypeMessageVersion(version));
}

#[rstest]
#[case::unsupported_version(Class::FLOATING_POINT, 15)]
#[case::wrong_class(Class::FIXED_POINT, Version::One.get())]
fn incomplete_complex_parent_headers_precede_version_and_class_errors(
    #[case] class: Class,
    #[case] version: u8,
    #[values(1, 7)] available: usize,
) {
    let parent = datatype::header(class, version, [0; 3], 8);
    let bytes = Context::Complex.bytes(&parent[..available]);
    assert_decode_error(
        &bytes,
        FormatError::UnexpectedEof {
            expected: bytes.len() - available + parent.len(),
            available: bytes.len(),
        },
    );
}

#[rstest]
#[case::unsupported_parent_version(Class::FLOATING_POINT, 15, FormatError::InvalidDatatypeVersion { class: Class::FLOATING_POINT.0, version: 15 })]
#[case::wrong_parent_class(Class::FIXED_POINT, Version::One.get(), FormatError::InvalidComplexParent { class: Class::FIXED_POINT.0 })]
#[case::wrong_parent_class_and_version(Class::FIXED_POINT, 15, FormatError::InvalidComplexParent { class: Class::FIXED_POINT.0 })]
fn complex_parent_header_checks_class_before_version(
    #[case] class: Class,
    #[case] version: u8,
    #[case] error: FormatError,
) {
    assert_decode_error(
        &Context::Complex.bytes(&datatype::header(class, version, [0; 3], 8)),
        error,
    );
}

fn assert_decode_error(bytes: &[u8], expected: FormatError) {
    assert_eq!(format::parse_datatype(bytes).unwrap_err(), expected);
    assert_eq!(DatatypeMessage::parse(bytes).unwrap_err(), expected);
}

#[rstest]
#[case::integer(
    DatatypeProperties::FixedPoint(FixedPointType::i16()),
    Datatype::FixedPoint(FixedPointType::i16())
)]
#[case::little_endian_float(
    DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary32(
        DatatypeByteOrder::LittleEndian
    )),
    Datatype::FloatingPoint(FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian))
)]
#[case::big_endian_float(
    DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary32(
        DatatypeByteOrder::BigEndian
    )),
    Datatype::FloatingPoint(FloatingPointType::ieee754_binary32(DatatypeByteOrder::BigEndian))
)]
#[case::time(DatatypeProperties::Time(TimeType::new(4, DatatypeByteOrder::BigEndian, 24).unwrap()), Datatype::Time(TimeType::new(4, DatatypeByteOrder::BigEndian, 24).unwrap()))]
#[case::string(DatatypeProperties::String { size: 7, padding: StringPadding::SpacePad, charset: CharacterSet::Utf8 }, Datatype::String { size: 7, padding: StringPadding::SpacePad, charset: CharacterSet::Utf8 })]
#[case::bitfield(DatatypeProperties::BitField(BitFieldType::new(4, DatatypeByteOrder::BigEndian, 3, 17).unwrap()), Datatype::BitField(BitFieldType::new(4, DatatypeByteOrder::BigEndian, 3, 17).unwrap()))]
#[case::opaque(DatatypeProperties::Opaque { size: 3, tag: vec![0xff, b't'] }, Datatype::Opaque { size: 3, tag: vec![0xff, b't'] })]
fn leaf_messages_retain_versions_and_properties(
    #[values(1, 2, 3, 4, 5)] raw: u8,
    #[case] properties: DatatypeProperties,
    #[case] expected: Datatype,
) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    let message = DatatypeMessage::new(version, properties.clone()).unwrap();
    assert_eq!(message.version(), version);
    assert_eq!(message.properties(), &properties);
    assert_eq!(message.to_datatype().unwrap(), expected);
    assert_eq!(
        message.stored_size(OffsetWidth::Four).unwrap(),
        expected.type_size()
    );
}

#[rstest]
#[case::little(0x00, DatatypeByteOrder::LittleEndian)]
#[case::big(0x01, DatatypeByteOrder::BigEndian)]
fn float_order_reads_only_its_two_bits(
    #[values(1, 2, 3, 4, 5)] raw: u8,
    #[case] bits: u8,
    #[case] expected: DatatypeByteOrder,
) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    let order = FloatingPointByteOrder::parse(version, [bits | 0x2e, 31, 0]).unwrap();
    assert_eq!(
        order,
        FloatingPointByteOrder::new(version, expected).unwrap()
    );
    assert_eq!(order.version(), version);
    assert_eq!(order.order(), expected);
    assert_eq!(order.bits(), bits);
}

#[rstest]
fn reserved_float_order_is_rejected_in_every_version(#[values(1, 2, 3, 4, 5)] raw: u8) {
    assert_eq!(
        FloatingPointByteOrder::parse(DatatypeVersion::try_from(raw).unwrap(), [0x6e, 31, 0])
            .unwrap_err(),
        FormatError::InvalidByteOrder(0x40),
    );
}

#[rstest]
#[case::reserved_low_flag([0x80, 0, 0], 0x80)]
#[case::reserved_third_byte([0, 0, 1], 0x01_0000)]
fn float_order_rejects_reserved_class_flags(#[case] flags: [u8; 3], #[case] expected: u32) {
    assert_eq!(
        FloatingPointByteOrder::parse(DatatypeVersion::Five, flags).unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class: 1,
            version: 5,
            flags: expected
        },
    );
}

#[rstest]
fn vax_metadata_is_available_from_version_three(#[values(3, 4, 5)] raw: u8) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    let order = FloatingPointByteOrder::parse(version, [0x41, 0, 0]).unwrap();
    assert_eq!(order.order(), DatatypeByteOrder::Vax);
    assert_eq!(order.bits(), 0x41);
    let numeric = FloatingPointType::ieee754_binary64(DatatypeByteOrder::Vax);
    let message =
        DatatypeMessage::new(version, DatatypeProperties::FloatingPoint(numeric)).unwrap();
    assert_eq!(message.version(), version);
    assert_eq!(
        message.to_datatype().unwrap(),
        Datatype::FloatingPoint(numeric)
    );
}

#[rstest]
fn early_vax_metadata_is_rejected_before_construction(#[values(1, 2)] raw: u8) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    let expected = FormatError::InvalidDatatypeVersion {
        class: 1,
        version: raw,
    };
    assert_eq!(
        FloatingPointByteOrder::parse(version, [0x41, 0, 0]).unwrap_err(),
        expected
    );
    assert_eq!(
        DatatypeMessage::new(
            version,
            DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary32(
                DatatypeByteOrder::Vax
            ),)
        )
        .unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::compound(Container::Compound)]
#[case::enumeration(Container::Enumeration)]
#[case::array(Container::Array)]
fn modern_containers_retain_legal_version_mixtures(
    #[values(3, 4, 5)] raw: u8,
    #[case] class: Container,
) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    let properties = container(class, integer(DatatypeVersion::One));
    let message = DatatypeMessage::new(version, properties.clone()).unwrap();
    assert_eq!(message.version(), version);
    assert_eq!(message.properties(), &properties);
    assert_eq!(
        message.to_datatype().unwrap().type_size(),
        if class == Container::Array { 4 } else { 2 }
    );
}

#[rstest]
#[case::compound(Container::Compound)]
#[case::enumeration(Container::Enumeration)]
#[case::sequence(Container::Sequence)]
#[case::string(Container::String)]
#[case::array(Container::Array)]
fn container_versions_below_child_versions_are_rejected(#[case] class: Container) {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            container(class, integer(DatatypeVersion::Five))
        )
        .unwrap_err(),
        FormatError::DatatypeChildVersion {
            class: class.class(),
            version: 3,
            child_version: 5
        },
    );
}

#[rstest]
#[case::compound(Container::Compound)]
#[case::array(Container::Array)]
fn legacy_container_versions_are_outside_the_message_representation(
    #[values(1, 2)] raw: u8,
    #[case] class: Container,
) {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::try_from(raw).unwrap(),
            container(class, integer(DatatypeVersion::One))
        )
        .unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: class.class(),
            version: raw
        },
    );
}

#[rstest]
#[case::sequence(Container::Sequence)]
#[case::string(Container::String)]
fn variable_length_nodes_retain_both_versions(
    #[values(1, 2, 3, 4, 5)] raw: u8,
    #[case] class: Container,
) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    let properties = container(class, integer(version));
    let message = DatatypeMessage::new(version, properties.clone()).unwrap();
    assert_eq!(message.version(), version);
    assert_eq!(message.properties(), &properties);
    assert_eq!(message.stored_size(OffsetWidth::Two).unwrap(), 10);
    assert_eq!(message.stored_size(OffsetWidth::Four).unwrap(), 12);
    assert_eq!(message.stored_size(OffsetWidth::Eight).unwrap(), 16);
    let Datatype::VariableLength(kind) = message.to_datatype().unwrap() else {
        panic!("expected variable-length metadata");
    };
    assert_eq!(
        kind.base_type(),
        &Datatype::FixedPoint(FixedPointType::i16())
    );
}

#[rstest]
fn complex_nodes_retain_the_component_version(#[values(1, 2, 3, 4, 5)] raw: u8) {
    let version = DatatypeVersion::try_from(raw).unwrap();
    let component = DatatypeMessage::new(
        version,
        DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary32(
            DatatypeByteOrder::BigEndian,
        )),
    )
    .unwrap();
    let properties = DatatypeProperties::Complex {
        component: Box::new(component),
    };
    let message = DatatypeMessage::new(DatatypeVersion::Five, properties.clone()).unwrap();
    assert_eq!(message.properties(), &properties);
    assert_eq!(message.version(), DatatypeVersion::Five);
    assert_eq!(message.to_datatype().unwrap().type_size(), 8);
}

#[rstest]
#[case::one(DatatypeVersion::One)]
#[case::four(DatatypeVersion::Four)]
fn complex_outer_versions_below_five_are_rejected(#[case] version: DatatypeVersion) {
    assert_eq!(
        DatatypeMessage::new(
            version,
            DatatypeProperties::Complex {
                component: Box::new(integer(DatatypeVersion::One))
            }
        )
        .unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: 11,
            version: version.get()
        },
    );
}

#[test]
fn complex_components_must_be_floating_point() {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Five,
            DatatypeProperties::Complex {
                component: Box::new(integer(DatatypeVersion::One))
            }
        )
        .unwrap_err(),
        FormatError::InvalidComplexParent { class: 0 },
    );
}

#[rstest]
#[case::two_byte_short(integer(DatatypeVersion::One), "value", vec![1], 2, 1)]
#[case::four_byte_short(DatatypeMessage::parse(&Leaf::Integer.bytes(Version::One)).unwrap().0, "x", vec![0; 3], 4, 3)]
#[case::four_byte_long(DatatypeMessage::parse(&Leaf::Integer.bytes(Version::One)).unwrap().0, "x", vec![0; 5], 4, 5)]
fn enum_values_have_the_integer_base_width(
    #[case] base_type: DatatypeMessage,
    #[case] name: &str,
    #[case] value: Vec<u8>,
    #[case] expected: u32,
    #[case] actual: usize,
) {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Enumeration {
                base_type: Box::new(base_type),
                members: vec![EnumMember::new(
                    DatatypeMemberName::new(name).unwrap(),
                    value
                )],
            }
        )
        .unwrap_err(),
        FormatError::EnumMemberValueSize(DatatypeMemberName::new(name).unwrap(), expected, actual),
    );
}

#[test]
fn compound_member_names_reject_embedded_nuls() {
    assert_eq!(
        DatatypeMessageMember::new("bad\0name".into(), 0, integer(DatatypeVersion::One))
            .unwrap_err(),
        FormatError::InvalidDatatypeMemberName {
            class: 6,
            name: "bad\0name".into()
        },
    );
}

#[test]
fn compound_member_ranges_must_fit_at_a_supported_offset_width() {
    let member =
        DatatypeMessageMember::new("field".into(), 1, integer(DatatypeVersion::One)).unwrap();
    assert_eq!(member.name(), "field");
    assert_eq!(member.byte_offset(), 1);
    assert_eq!(member.datatype(), &integer(DatatypeVersion::One));
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Compound {
                size: 2,
                members: vec![member]
            }
        )
        .unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: "field".into(),
            offset: 1,
            field_size: 2,
            compound_size: 2
        },
    );
}

#[rstest]
#[case::maximum(248)]
#[case::unaligned(247)]
fn opaque_messages_retain_encodable_raw_tags(#[case] length: usize) {
    let properties = DatatypeProperties::Opaque {
        size: 1,
        tag: vec![0xff; length],
    };
    let message = DatatypeMessage::new(DatatypeVersion::Five, properties.clone()).unwrap();
    assert_eq!(message.properties(), &properties);
    assert_eq!(
        message.to_datatype().unwrap(),
        Datatype::Opaque {
            size: 1,
            tag: vec![0xff; length]
        }
    );
}

#[rstest]
#[case::too_long(vec![1; 249], FormatError::OpaqueTagTooLong { length: 249 })]
#[case::nul(vec![1, 0, 3], FormatError::InvalidOpaqueTagByte { offset: 1, byte: 0 })]
fn opaque_messages_reject_unencodable_tags(#[case] tag: Vec<u8>, #[case] expected: FormatError) {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::One,
            DatatypeProperties::Opaque { size: 1, tag }
        )
        .unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::string(DatatypeProperties::String { size: 0, padding: StringPadding::NullPad, charset: CharacterSet::Ascii }, 3)]
#[case::opaque(DatatypeProperties::Opaque { size: 0, tag: vec![] }, 5)]
#[case::compound(DatatypeProperties::Compound { size: 0, members: vec![] }, 6)]
#[case::zero_array(DatatypeProperties::Array { base_type: Box::new(integer(DatatypeVersion::One)), dimensions: vec![0] }, 10)]
fn checked_messages_require_nonzero_element_sizes(
    #[case] properties: DatatypeProperties,
    #[case] class: u8,
) {
    assert_eq!(
        DatatypeMessage::new(DatatypeVersion::Five, properties).unwrap_err(),
        FormatError::ZeroSizedDatatype { class }
    );
}

#[test]
fn enumeration_bases_must_be_integer_messages() {
    let base_type = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary32(
            DatatypeByteOrder::LittleEndian,
        )),
    )
    .unwrap();
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Enumeration {
                base_type: Box::new(base_type),
                members: vec![]
            }
        )
        .unwrap_err(),
        FormatError::EnumBaseNotInteger,
    );
}

#[test]
fn array_dimension_counts_must_fit_the_rank_field() {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Array {
                base_type: Box::new(integer(DatatypeVersion::One)),
                dimensions: vec![1; 256],
            }
        )
        .unwrap_err(),
        FormatError::ValueTooLargeForRepresentation {
            value: 256,
            target: "u8"
        },
    );
}

#[test]
fn array_products_must_fit_the_size_field() {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Array {
                base_type: Box::new(integer(DatatypeVersion::One)),
                dimensions: vec![u32::MAX],
            }
        )
        .unwrap_err(),
        FormatError::LengthTooLarge {
            length: 2 * u64::from(u32::MAX),
            length_size: 4
        },
    );
}

#[test]
fn nested_messages_retain_each_containing_and_child_version() {
    let sequence = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(integer(DatatypeVersion::Two)),
        },
    )
    .unwrap();
    let array = DatatypeMessage::new(
        DatatypeVersion::Four,
        DatatypeProperties::Array {
            base_type: Box::new(sequence),
            dimensions: vec![2],
        },
    )
    .unwrap();
    let member = DatatypeMessageMember::new("values".into(), 0, array).unwrap();
    let message = DatatypeMessage::new(
        DatatypeVersion::Five,
        DatatypeProperties::Compound {
            size: 32,
            members: vec![member],
        },
    )
    .unwrap();
    assert_eq!(message.version(), DatatypeVersion::Five);
    let DatatypeProperties::Compound { size, members } = message.properties() else {
        panic!("expected compound metadata");
    };
    assert_eq!(*size, 32);
    assert_eq!(members.len(), 1);
    let array = members[0].datatype();
    assert_eq!(array.version(), DatatypeVersion::Four);
    let DatatypeProperties::Array {
        base_type: sequence,
        dimensions,
    } = array.properties()
    else {
        panic!("expected array metadata");
    };
    assert_eq!(dimensions, &[2]);
    assert_eq!(sequence.version(), DatatypeVersion::Three);
    let DatatypeProperties::VariableLengthSequence { base_type } = sequence.properties() else {
        panic!("expected sequence metadata");
    };
    assert_eq!(base_type, &Box::new(integer(DatatypeVersion::Two)));
    assert_eq!(message.stored_size(OffsetWidth::Eight).unwrap(), 32);
    assert_eq!(
        message.to_datatype().unwrap(),
        Datatype::Compound {
            size: 32,
            members: vec![__private::compound_member(
                "values".into(),
                0,
                Datatype::Array {
                    base_type: Box::new(Datatype::VariableLength(VariableLengthType::sequence(
                        Datatype::FixedPoint(FixedPointType::i16())
                    ))),
                    dimensions: vec![2],
                }
            )],
        }
    );
}

fn integer(version: DatatypeVersion) -> DatatypeMessage {
    DatatypeMessage::new(
        version,
        DatatypeProperties::FixedPoint(FixedPointType::i16()),
    )
    .unwrap()
}

fn container(class: Container, child: DatatypeMessage) -> DatatypeProperties {
    match class {
        Container::Compound => DatatypeProperties::Compound {
            size: 2,
            members: vec![DatatypeMessageMember::new("field".into(), 0, child).unwrap()],
        },
        Container::Enumeration => DatatypeProperties::Enumeration {
            base_type: Box::new(child),
            members: vec![EnumMember::new(
                DatatypeMemberName::new("value").unwrap(),
                vec![1, 2],
            )],
        },
        Container::Sequence => DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(child),
        },
        Container::Array => DatatypeProperties::Array {
            base_type: Box::new(child),
            dimensions: vec![2],
        },
        Container::String => DatatypeProperties::VariableLengthString {
            padding: StringPadding::NullPad,
            charset: CharacterSet::Utf8,
            base_type: Box::new(child),
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Container {
    Array,
    Compound,
    Enumeration,
    Sequence,
    String,
}

impl Container {
    fn class(self) -> u8 {
        match self {
            Self::Array => 10,
            Self::Compound => 6,
            Self::Enumeration => 8,
            Self::Sequence | Self::String => 9,
        }
    }
}

#[test]
fn nested_containers_preserve_datatype_values_and_wire_versions() {
    let integer = Leaf::Integer.bytes(Version::One);
    let enumeration =
        containers::enumeration(Version::Three.get(), [2, 0, 0], 4, &integer, &enum_values());
    let array = datatype::array(Version::Four.get(), [0; 3], 24, 2, &[2, 3], &enumeration);
    let sequence = VariableLengthSequence::new(Version::Two, 8, &integer).build();
    let complex = datatype::complex(
        Version::Five.get(),
        HOMOGENEOUS_COMPLEX,
        16,
        &datatype::f64_le(),
    );
    let expected = Datatype::Compound {
        size: 56,
        members: vec![
            __private::compound_member(
                "array".into(),
                0,
                Datatype::Array {
                    base_type: Box::new(enumeration_datatype()),
                    dimensions: vec![2, 3],
                },
            ),
            __private::compound_member(
                "sequence".into(),
                24,
                Datatype::VariableLength(VariableLengthType::sequence(integer_datatype())),
            ),
            __private::compound_member(
                "complex".into(),
                40,
                Datatype::Complex(
                    ComplexType::new(FloatingPointType::ieee754_binary64(
                        DatatypeByteOrder::LittleEndian,
                    ))
                    .unwrap(),
                ),
            ),
        ],
    };
    let bytes = containers::compound(
        Version::Five.get(),
        [3, 0, 0],
        56,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"array",
                offset: 0,
                datatype: &array,
            },
            CompoundField {
                name: b"sequence",
                offset: 24,
                datatype: &sequence,
            },
            CompoundField {
                name: b"complex",
                offset: 40,
                datatype: &complex,
            },
        ],
    );
    let mut input = bytes.clone();
    input.extend_from_slice(&[0xff; 8]);
    assert_eq!(
        format::parse_datatype(&input).unwrap(),
        (expected.clone(), bytes.len()),
    );
    let (message, consumed) = DatatypeMessage::parse(&input).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.to_datatype().unwrap(), expected);
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
}

#[rstest]
#[case::compound_v1(test_util::legacy_compound::array_member(), Class::COMPOUND, Version::One,
    Datatype::Compound {
        size: 16,
        members: vec![__private::compound_member(
            "arr".into(),
            0,
            Datatype::Array {
                base_type: Box::new(Datatype::FixedPoint(FixedPointType::i32())),
                dimensions: vec![4],
            },
        )],
    })]
#[case::compound_v2(containers::legacy_compound_v2(), Class::COMPOUND, Version::Two,
    Datatype::Compound {
        size: 8,
        members: vec![__private::compound_member("offset".into(), 4, integer_datatype())],
    })]
#[case::array_v2(containers::legacy_array(), Class::ARRAY, Version::Two,
    Datatype::Array {
        base_type: Box::new(integer_datatype()),
        dimensions: vec![2, 3],
    })]
fn semantic_parser_accepts_and_retained_parser_rejects_legacy_containers(
    #[case] mut bytes: Vec<u8>,
    #[case] class: Class,
    #[case] version: Version,
    #[case] expected: Datatype,
) {
    let consumed = bytes.len();
    bytes.extend_from_slice(&[0xff; 4]);
    assert_eq!(
        format::parse_datatype(&bytes).unwrap(),
        (expected, consumed)
    );
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: class.0,
            version: version.get()
        },
    );
}

#[test]
fn incomplete_enumeration_values_report_the_full_value_list_extent() {
    let parent = Leaf::Integer.bytes(Version::One);
    let mut bytes =
        containers::enumeration(Version::Three.get(), [2, 0, 0], 4, &parent, &enum_values());
    let expected = bytes.len();
    bytes.truncate(expected - 6);
    assert_decode_error(
        &bytes,
        FormatError::UnexpectedEof {
            expected,
            available: bytes.len(),
        },
    );
}

#[test]
fn incomplete_array_dimensions_report_the_full_dimension_list_extent() {
    let parent = Leaf::Integer.bytes(Version::One);
    let mut bytes = datatype::array(Version::Three.get(), [0; 3], 120, 3, &[2, 3, 5], &parent);
    let expected = bytes.len() - parent.len();
    bytes.truncate(expected - 5);
    assert_decode_error(
        &bytes,
        FormatError::UnexpectedEof {
            expected,
            available: bytes.len(),
        },
    );
}

#[test]
fn retained_array_flag_error_precedes_dimension_list_eof() {
    let version = Version::Three;
    let parent = Leaf::Integer.bytes(Version::One);
    let mut bytes = datatype::array(version.get(), [0; 3], 24, 2, &[2, 3], &parent);
    let expected = bytes.len() - parent.len();
    bytes.truncate(expected - 1);
    bytes[1] = 0x80;
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class: Class::ARRAY.0,
            version: version.get(),
            flags: 0x80
        },
    );
    assert_eq!(
        format::parse_datatype(&bytes).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available: bytes.len()
        },
    );
}

#[test]
fn datatype_parsers_report_member_eof_from_the_compound_start() {
    let child = Leaf::Integer.bytes(Version::One);
    let mut bytes = containers::compound(
        Version::Three.get(),
        [1, 0, 0],
        4,
        MemberOffset::One,
        &[CompoundField {
            name: b"value",
            offset: 0,
            datatype: &child,
        }],
    );
    let expected = bytes.len();
    bytes.truncate(expected - 1);
    assert_decode_error(
        &bytes,
        FormatError::UnexpectedEof {
            expected,
            available: bytes.len(),
        },
    );
}

#[rstest]
#[case::complex(Context::Complex.bytes(&datatype::f64_le()), 28)]
#[case::opaque(Leaf::Opaque(OpaqueTag::NonAscii).bytes(Version::One), 16)]
#[case::array_dimensions(datatype::array(Version::Three.get(), [0; 3], 24, 2, &[2, 3], &[]), 17)]
#[case::enumeration_values(containers::enumeration(
    Version::Three.get(), [2, 0, 0], 4, &Leaf::Integer.bytes(Version::One), &enum_values()
), 41)]
fn datatype_parsers_report_nested_eof_from_the_supplied_subslice(
    #[case] child: Vec<u8>,
    #[case] child_extent: usize,
) {
    assert_eq!(child.len(), child_extent);
    let inner = VariableLengthSequence::new(Version::Five, 8, &child[..child_extent - 1]).build();
    let body = VariableLengthSequence::new(Version::Five, 8, &inner).build();
    let mut enclosing = vec![0xa5; 7];
    let start = enclosing.len();
    enclosing.extend_from_slice(&body);
    enclosing.extend_from_slice(&[0x5a; 3]);
    assert_decode_error(
        &enclosing[start..start + body.len()],
        FormatError::UnexpectedEof {
            expected: body.len() + 1,
            available: body.len(),
        },
    );
}

#[rstest]
#[case::compound_v1(
    test_util::legacy_compound::array_member(),
    Class::COMPOUND,
    Version::One
)]
#[case::compound_v2(containers::legacy_compound_v2(), Class::COMPOUND, Version::Two)]
#[case::array_v2(containers::legacy_array(), Class::ARRAY, Version::Two)]
fn semantic_parser_reports_nested_legacy_eof_from_the_outer_description(
    #[case] mut child: Vec<u8>,
    #[case] class: Class,
    #[case] version: Version,
) {
    child.truncate(child.len() - 1);
    let body = VariableLengthSequence::new(Version::Five, 8, &child).build();
    assert_eq!(
        format::parse_datatype(&body).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: body.len() + 1,
            available: body.len(),
        },
    );
    assert_eq!(
        DatatypeMessage::parse(&body).unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: class.0,
            version: version.get(),
        },
    );
}

fn enum_values() -> [EnumValue<'static>; 2] {
    [
        EnumValue {
            name: b"first",
            value: &[0xf0, 0, 0, 0x0f],
        },
        EnumValue {
            name: b"second",
            value: &[0xf0, 0, 0, 0x17],
        },
    ]
}

fn enumeration_datatype() -> Datatype {
    let Datatype::FixedPoint(base) = integer_datatype() else {
        panic!("expected fixed-point base");
    };
    let members = enum_values()
        .into_iter()
        .map(|EnumValue { name, value }| {
            EnumMember::new(DatatypeMemberName::new(name).unwrap(), value.to_vec())
        })
        .collect();
    Datatype::Enumeration(EnumerationType::new(base, members).unwrap())
}

fn integer_datatype() -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::BigEndian, true, 3, 25)
            .unwrap()
            .with_padding(true, true),
    )
}

// "The Datatype Message" in format specification version 4.0 uses bit 0 of the version 5
// complex class bit field for the homogeneous flag.
const HOMOGENEOUS_COMPLEX: [u8; 3] = [1, 0, 0];
