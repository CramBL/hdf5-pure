use hdf5_pure_core::{
    BitFieldType, CharacterSet, DatatypeByteOrder, FixedPointType, FloatingPointType, FormatError,
    MantissaNormalization, StringPadding, TimeType,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype::leaf::{
    self, Charset, FullOpaqueProperty, Leaf, OpaqueTag, Order, Padding, VaxVersion, Version,
};
use test_util::datatype::{self, Class};

#[rstest]
#[case::integer(Leaf::Integer)]
#[case::little_unsigned_integer(Leaf::LittleInteger)]
#[case::little_float(Leaf::Float(Order::Little))]
#[case::big_float(Leaf::Float(Order::Big))]
#[case::unnormalized_float(Leaf::UnnormalizedFloat(Order::Little))]
#[case::implied_float(Leaf::ImpliedFloat(Order::Big))]
#[case::low_padded_float(Leaf::LowPaddedFloat(Order::Little))]
#[case::high_padded_float(Leaf::HighPaddedFloat(Order::Little))]
#[case::internally_padded_float(Leaf::InternallyPaddedFloat(Order::Little))]
#[case::time(Leaf::Time)]
#[case::little_time(Leaf::LittleTime)]
#[case::string(Leaf::string())]
#[case::ascii_null_terminate(Leaf::String { padding: Padding::NullTerminate, charset: Charset::Ascii })]
#[case::ascii_null_pad(Leaf::String { padding: Padding::NullPad, charset: Charset::Ascii })]
#[case::ascii_space_pad(Leaf::String { padding: Padding::SpacePad, charset: Charset::Ascii })]
#[case::utf8_null_terminate(Leaf::String { padding: Padding::NullTerminate, charset: Charset::Utf8 })]
#[case::utf8_null_pad(Leaf::String { padding: Padding::NullPad, charset: Charset::Utf8 })]
#[case::bitfield(Leaf::Bitfield)]
#[case::little_bitfield(Leaf::LittleBitfield)]
#[case::opaque_empty(Leaf::Opaque(OpaqueTag::Empty))]
#[case::opaque_raw(Leaf::Opaque(OpaqueTag::NonAscii))]
#[case::opaque_aligned(Leaf::Opaque(OpaqueTag::Aligned))]
#[case::opaque_maximum(Leaf::Opaque(OpaqueTag::Maximum))]
fn datatype_leaf_codecs_preserve_versions_and_properties(
    #[case] leaf: Leaf,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
) {
    assert_leaf(&leaf, version, offsets);
}

#[rstest]
fn datatype_leaf_codecs_preserve_vax_metadata(
    #[values(VaxVersion::Three, VaxVersion::Four, VaxVersion::Five)] version: VaxVersion,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
) {
    assert_leaf_bytes(
        leaf::vax_bytes(version),
        floating_properties(
            DatatypeByteOrder::Vax,
            MantissaNormalization::LeadingOne,
            [true; 3],
        ),
        offsets,
    );
}

#[rstest]
#[case::zero(0)]
#[case::six(6)]
#[case::fifteen(15)]
fn datatype_leaf_parser_rejects_unsupported_versions(#[case] version: u8) {
    let mut bytes = Leaf::Integer.bytes(Version::One);
    bytes[0] = version << 4;
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeMessageVersion(version)
    );
}

#[rstest]
fn datatype_leaf_parser_rejects_reserved_float_order(
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
) {
    let mut bytes = Leaf::Float(Order::Little).bytes(version);
    bytes[1] |= 0x40;
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidByteOrder(0x40)
    );
}

#[rstest]
fn datatype_leaf_parser_rejects_vax_before_version_three(
    #[values(Version::One, Version::Two)] version: Version,
) {
    let mut bytes = leaf::vax_bytes(VaxVersion::Three);
    bytes[0] = (version.get() << 4) | Class::FLOATING_POINT.0;
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: 1,
            version: version.get()
        }
    );
}

#[rstest]
#[case::fixed(Leaf::Integer, [0x10, 0, 0], 0)]
#[case::float(Leaf::Float(Order::Little), [0x80, 0, 0], 1)]
#[case::float_third_byte(Leaf::Float(Order::Little), [0, 0, 1], 1)]
#[case::time(Leaf::Time, [2, 0, 0], 2)]
#[case::string(Leaf::string(), [0, 1, 0], 3)]
#[case::bitfield(Leaf::Bitfield, [8, 0, 0], 4)]
fn datatype_leaf_parser_rejects_reserved_flags(
    #[case] leaf: Leaf,
    #[case] flags: [u8; 3],
    #[case] class: u8,
) {
    let mut bytes = leaf.bytes(Version::Five);
    bytes[1..4].copy_from_slice(&flags);
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class,
            version: 5,
            flags: u32::from_le_bytes([flags[0], flags[1], flags[2], 0]),
        }
    );
}

#[rstest]
#[case::normalization(Leaf::Float(Order::Little), [0x30, 63, 0], FormatError::InvalidMantissaNormalization(3))]
#[case::padding(Leaf::string(), [3, 0, 0], FormatError::InvalidStringPadding(3))]
#[case::charset(Leaf::string(), [0x20, 0, 0], FormatError::InvalidCharacterSet(2))]
#[case::opaque_flags(Leaf::Opaque(OpaqueTag::Empty), [0, 1, 0], FormatError::InvalidOpaqueReservedBits { bits: 1 })]
#[case::opaque_length(Leaf::Opaque(OpaqueTag::NonAscii), [1, 0, 0], FormatError::InvalidOpaqueTagLength { length: 1 })]
fn datatype_leaf_parser_preserves_property_errors(
    #[case] leaf: Leaf,
    #[case] flags: [u8; 3],
    #[case] expected: FormatError,
) {
    let mut bytes = leaf.bytes(Version::Five);
    bytes[1..4].copy_from_slice(&flags);
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::class(0, 1)]
#[case::flags(1, 4)]
#[case::flags_end(3, 4)]
#[case::size(4, 8)]
#[case::size_end(7, 8)]
#[case::offset(8, 10)]
#[case::offset_end(9, 10)]
#[case::precision(10, 12)]
#[case::precision_end(11, 12)]
#[case::exponent_location(12, 13)]
#[case::exponent_size(13, 14)]
#[case::mantissa_location(14, 15)]
#[case::mantissa_size(15, 16)]
#[case::bias(16, 20)]
#[case::bias_end(19, 20)]
fn datatype_leaf_parser_reports_truncated_fields(
    #[case] available: usize,
    #[case] expected: usize,
) {
    let bytes = Leaf::Float(Order::Little).bytes(Version::Three);
    assert_eq!(
        DatatypeMessage::parse(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[rstest]
#[case::integer(Leaf::Integer, 0)]
#[case::float(Leaf::Float(Order::Little), 1)]
#[case::time(Leaf::Time, 2)]
#[case::string(Leaf::string(), 3)]
#[case::bitfield(Leaf::Bitfield, 4)]
#[case::opaque(Leaf::Opaque(OpaqueTag::Empty), 5)]
fn datatype_leaf_parser_rejects_zero_element_sizes(#[case] leaf: Leaf, #[case] class: u8) {
    let mut bytes = leaf.bytes(Version::Three);
    bytes[4..8].fill(0);
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::ZeroSizedDatatype { class }
    );
}

#[test]
fn datatype_leaf_parser_preserves_numeric_layout_errors() {
    let mut bytes = Leaf::Float(Order::Little).bytes(Version::Three);
    bytes[10..12].fill(0);
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidFloatingPointRange {
            size: 8,
            bit_offset: 3,
            bit_precision: 0
        }
    );
}

#[test]
fn datatype_leaf_parser_rejects_nonzero_opaque_padding() {
    let mut bytes = Leaf::Opaque(OpaqueTag::NonAscii).bytes(Version::One);
    bytes[11] = 1;
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidOpaqueTagPadding { offset: 3, byte: 1 }
    );
}

#[test]
fn datatype_leaf_parser_rejects_reference_descriptions() {
    let bytes = datatype::header(Class::REFERENCE, 1, [0; 3], 8);
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeClass(7)
    );
    assert_eq!(
        hdf5_pure_format::__private::parse_datatype(&bytes)
            .unwrap()
            .1,
        8
    );
}

#[rstest]
#[case::aligned(FullOpaqueProperty::Eight)]
#[case::maximum(FullOpaqueProperty::Maximum)]
fn datatype_leaf_parser_retains_versions_in_full_opaque_properties(
    #[case] property: FullOpaqueProperty,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
) {
    let bytes = property.bytes(version);
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.version().get(), version.get());
    assert_eq!(
        message.properties(),
        &DatatypeProperties::Opaque {
            size: 3,
            tag: vec![b't'; usize::from(property.length())]
        }
    );
    match property {
        FullOpaqueProperty::Eight => {
            assert_eq!(
                message.serialize(OffsetWidth::Eight).unwrap(),
                Leaf::Opaque(OpaqueTag::Aligned).bytes(version)
            );
        }
        FullOpaqueProperty::Maximum => {
            assert_eq!(
                message.serialize(OffsetWidth::Eight).unwrap_err(),
                FormatError::OpaqueTagTooLong { length: 248 }
            );
        }
    }
}

fn assert_leaf(leaf: &Leaf, version: Version, offsets: OffsetWidth) {
    assert_leaf_bytes(leaf.bytes(version), properties(leaf), offsets);
}

fn assert_leaf_bytes(expected: Vec<u8>, properties: DatatypeProperties, offsets: OffsetWidth) {
    let message = DatatypeMessage::new(
        DatatypeVersion::try_from(expected[0] >> 4).unwrap(),
        properties.clone(),
    )
    .unwrap();
    assert_eq!(message.serialize(offsets).unwrap(), expected);
    let mut input = expected.clone();
    input.extend_from_slice(b"next");
    let (parsed, consumed) = DatatypeMessage::parse(&input).unwrap();
    assert_eq!(consumed, expected.len());
    assert_eq!(parsed, message);
    assert_eq!(parsed.properties(), &properties);
    assert_eq!(parsed.serialize(offsets).unwrap(), expected);
    assert_eq!(
        format::parse_datatype(&input).unwrap(),
        (message.to_datatype().unwrap(), expected.len())
    );
}

fn properties(leaf: &Leaf) -> DatatypeProperties {
    match leaf {
        Leaf::Integer => DatatypeProperties::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::BigEndian, true, 3, 25)
                .unwrap()
                .with_padding(true, true),
        ),
        Leaf::LittleInteger => DatatypeProperties::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, false, 3, 25)
                .unwrap()
                .with_padding(true, true),
        ),
        Leaf::Float(order) => floating_properties(
            byte_order(*order),
            MantissaNormalization::LeadingOne,
            [true; 3],
        ),
        Leaf::UnnormalizedFloat(order) => floating_properties(
            byte_order(*order),
            MantissaNormalization::Unnormalized,
            [false; 3],
        ),
        Leaf::ImpliedFloat(order) => floating_properties(
            byte_order(*order),
            MantissaNormalization::ImpliedLeadingOne,
            [false; 3],
        ),
        Leaf::LowPaddedFloat(order) => floating_properties(
            byte_order(*order),
            MantissaNormalization::LeadingOne,
            [true, false, false],
        ),
        Leaf::HighPaddedFloat(order) => floating_properties(
            byte_order(*order),
            MantissaNormalization::LeadingOne,
            [false, true, false],
        ),
        Leaf::InternallyPaddedFloat(order) => floating_properties(
            byte_order(*order),
            MantissaNormalization::LeadingOne,
            [false, false, true],
        ),
        Leaf::Time => {
            DatatypeProperties::Time(TimeType::new(4, DatatypeByteOrder::BigEndian, 24).unwrap())
        }
        Leaf::LittleTime => {
            DatatypeProperties::Time(TimeType::new(4, DatatypeByteOrder::LittleEndian, 24).unwrap())
        }
        Leaf::String { padding, charset } => DatatypeProperties::String {
            size: 7,
            padding: match padding {
                Padding::NullTerminate => StringPadding::NullTerminate,
                Padding::NullPad => StringPadding::NullPad,
                Padding::SpacePad => StringPadding::SpacePad,
            },
            charset: match charset {
                Charset::Ascii => CharacterSet::Ascii,
                Charset::Utf8 => CharacterSet::Utf8,
            },
        },
        Leaf::Bitfield => DatatypeProperties::BitField(
            BitFieldType::new(4, DatatypeByteOrder::BigEndian, 3, 17)
                .unwrap()
                .with_padding(true, true),
        ),
        Leaf::LittleBitfield => DatatypeProperties::BitField(
            BitFieldType::new(4, DatatypeByteOrder::LittleEndian, 3, 17)
                .unwrap()
                .with_padding(true, true),
        ),
        Leaf::Opaque(tag) => DatatypeProperties::Opaque {
            size: 3,
            tag: tag.bytes(),
        },
    }
}

fn floating_properties(
    order: DatatypeByteOrder,
    normalization: MantissaNormalization,
    padding: [bool; 3],
) -> DatatypeProperties {
    DatatypeProperties::FloatingPoint(
        FloatingPointType::new(8, order, normalization, 63, 3, 61, 52, 11, 3, 49, 1023)
            .unwrap()
            .with_padding(padding[0], padding[1], padding[2]),
    )
}

fn byte_order(order: Order) -> DatatypeByteOrder {
    match order {
        Order::Little => DatatypeByteOrder::LittleEndian,
        Order::Big => DatatypeByteOrder::BigEndian,
    }
}
