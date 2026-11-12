use hdf5_pure_core::__private;
use hdf5_pure_core::{
    BitFieldType, CharacterSet, Datatype, DatatypeByteOrder, FixedPointType, FloatingPointType,
    FormatError, StringPadding, TimeType, VariableLengthType,
};
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
    FloatingPointByteOrder, OffsetWidth,
};
use rstest::rstest;

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
#[case::enumeration(Container::Enumeration)]
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

#[test]
fn enum_values_have_the_integer_base_width() {
    let member = __private::enum_member("value".into(), vec![1]);
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Enumeration {
                base_type: Box::new(integer(DatatypeVersion::One)),
                members: vec![member],
            }
        )
        .unwrap_err(),
        FormatError::EnumMemberValueSize("value".into(), 2, 1),
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
fn enumeration_names_reject_embedded_nuls() {
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Enumeration {
                base_type: Box::new(integer(DatatypeVersion::One)),
                members: vec![__private::enum_member("bad\0name".into(), vec![0; 2])],
            }
        )
        .unwrap_err(),
        FormatError::InvalidDatatypeMemberName {
            class: 8,
            name: "bad\0name".into()
        },
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
            members: vec![__private::enum_member("value".into(), vec![1, 2])],
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
