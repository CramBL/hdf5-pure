use hdf5_pure_core::{CharacterSet, FormatError, StringPadding};
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::leaf::{
    self, Charset, Leaf, OpaqueTag, Order, Padding, VaxVersion, Version,
};

#[rstest]
#[case::integer(Leaf::Integer)]
#[case::float_le(Leaf::Float(Order::Little))]
#[case::float_be(Leaf::Float(Order::Big))]
#[case::time(Leaf::Time)]
#[case::string(Leaf::string())]
#[case::bitfield(Leaf::Bitfield)]
#[case::opaque(Leaf::Opaque(OpaqueTag::Aligned))]
#[case::opaque_maximum(Leaf::Opaque(OpaqueTag::Maximum))]
fn datatype_variable_length_codecs_preserve_parent_properties(
    #[case] leaf: Leaf,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = leaf.bytes(version);
    let bytes = datatype::variable_length(version.get(), [0; 3], descriptor_size(width), &parent);
    let expected = DatatypeMessage::new(
        DatatypeVersion::try_from(version.get()).unwrap(),
        DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(DatatypeMessage::parse(&parent).unwrap().0),
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, width);
}

#[rstest]
#[case::ascii_terminated(Padding::NullTerminate, Charset::Ascii)]
#[case::ascii_null_padded(Padding::NullPad, Charset::Ascii)]
#[case::ascii_space_padded(Padding::SpacePad, Charset::Ascii)]
#[case::utf8_terminated(Padding::NullTerminate, Charset::Utf8)]
#[case::utf8_null_padded(Padding::NullPad, Charset::Utf8)]
#[case::utf8_space_padded(Padding::SpacePad, Charset::Utf8)]
fn datatype_variable_length_codecs_preserve_string_metadata(
    #[case] padding: Padding,
    #[case] charset: Charset,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = Leaf::LittleInteger.bytes(Version::One);
    let bytes = datatype::variable_length(
        version.get(),
        [1 | ((padding as u8) << 4), charset as u8, 0],
        descriptor_size(width),
        &parent,
    );
    let expected = DatatypeMessage::new(
        DatatypeVersion::try_from(version.get()).unwrap(),
        DatatypeProperties::VariableLengthString {
            padding: match padding {
                Padding::NullTerminate => StringPadding::NullTerminate,
                Padding::NullPad => StringPadding::NullPad,
                Padding::SpacePad => StringPadding::SpacePad,
            },
            charset: match charset {
                Charset::Ascii => CharacterSet::Ascii,
                Charset::Utf8 => CharacterSet::Utf8,
            },
            base_type: Box::new(DatatypeMessage::parse(&parent).unwrap().0),
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, width);
}

#[rstest]
#[case::three(VaxVersion::Three)]
#[case::four(VaxVersion::Four)]
#[case::five(VaxVersion::Five)]
fn datatype_variable_length_codecs_preserve_vax_parent(#[case] version: VaxVersion) {
    let parent = leaf::vax_bytes(version);
    let bytes = datatype::variable_length(5, [0; 3], 16, &parent);
    let expected = DatatypeMessage::new(
        DatatypeVersion::Five,
        DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(DatatypeMessage::parse(&parent).unwrap().0),
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, OffsetWidth::Eight);
}

#[rstest]
#[case::v1(Version::One, Version::One, Version::One)]
#[case::v2(Version::Two, Version::One, Version::One)]
#[case::v3(Version::Three, Version::Two, Version::One)]
#[case::v4(Version::Four, Version::Three, Version::Two)]
#[case::v5(Version::Five, Version::Five, Version::Five)]
fn datatype_variable_length_codecs_preserve_nested_versions_and_change_widths(
    #[case] outer: Version,
    #[case] inner: Version,
    #[case] leaf: Version,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = Leaf::Integer.bytes(leaf);
    let nested = datatype::variable_length(inner.get(), [0; 3], 10, &parent);
    let bytes = datatype::variable_length(outer.get(), [0; 3], 10, &nested);
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.version().get(), outer.get());
    let DatatypeProperties::VariableLengthSequence { base_type } = message.properties() else {
        panic!("{message:?}")
    };
    assert_eq!(base_type.version().get(), inner.get());
    let DatatypeProperties::VariableLengthSequence { base_type } = base_type.properties() else {
        panic!("{base_type:?}")
    };
    assert_eq!(base_type.version().get(), leaf.get());
    let nested = datatype::variable_length(inner.get(), [0; 3], descriptor_size(width), &parent);
    assert_eq!(
        message.serialize(width).unwrap(),
        datatype::variable_length(outer.get(), [0; 3], descriptor_size(width), &nested)
    );
}

#[rstest]
#[case::sequence(false)]
#[case::string(true)]
fn datatype_variable_length_codecs_preserve_nested_string_parents(#[case] string: bool) {
    let parent = Leaf::Opaque(OpaqueTag::Aligned).bytes(Version::One);
    let inner = datatype::variable_length(3, [0x21, 1, 0], 12, &parent);
    let flags = if string { [0x11, 0, 0] } else { [0; 3] };
    let bytes = datatype::variable_length(5, flags, 12, &inner);
    let inner = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::VariableLengthString {
            padding: StringPadding::SpacePad,
            charset: CharacterSet::Utf8,
            base_type: Box::new(DatatypeMessage::parse(&parent).unwrap().0),
        },
    )
    .unwrap();
    let properties = if string {
        DatatypeProperties::VariableLengthString {
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
            base_type: Box::new(inner),
        }
    } else {
        DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(inner),
        }
    };
    assert_encoding(
        &bytes,
        DatatypeMessage::new(DatatypeVersion::Five, properties).unwrap(),
        OffsetWidth::Four,
    );
}

#[rstest]
#[case::subtype([2, 0, 0], FormatError::InvalidVariableLengthSubtype(2))]
#[case::reserved_low([0, 0x10, 0], FormatError::InvalidVariableLengthReservedBits(0x1000))]
#[case::reserved_high([0, 0, 0x80], FormatError::InvalidVariableLengthReservedBits(0x80_0000))]
#[case::sequence_padding([0x10, 0, 0], FormatError::InvalidVariableLengthSequenceMetadata { padding: 1, charset: 0 })]
#[case::sequence_charset([0, 1, 0], FormatError::InvalidVariableLengthSequenceMetadata { padding: 0, charset: 1 })]
#[case::string_padding([0x31, 0, 0], FormatError::InvalidStringPadding(3))]
#[case::string_charset([1, 2, 0], FormatError::InvalidCharacterSet(2))]
fn datatype_variable_length_parser_rejects_invalid_flags(
    #[case] flags: [u8; 3],
    #[case] expected: FormatError,
) {
    let bytes = datatype::variable_length(5, flags, 16, &Leaf::Integer.bytes(Version::One));
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::zero(0)]
#[case::short(9)]
#[case::between_widths(11)]
#[case::oversized(17)]
#[case::maximum(u32::MAX)]
fn datatype_variable_length_parser_rejects_invalid_descriptor_size(#[case] size: u32) {
    let bytes = datatype::variable_length(5, [0; 3], size, &Leaf::Integer.bytes(Version::One));
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidVariableLengthSize { size }
    );
}

#[rstest]
#[case::v1_v2(1, Version::Two)]
#[case::v2_v3(2, Version::Three)]
#[case::v3_v4(3, Version::Four)]
#[case::v4_v5(4, Version::Five)]
fn datatype_variable_length_parser_rejects_newer_parents(
    #[case] version: u8,
    #[case] child: Version,
) {
    let bytes = datatype::variable_length(version, [0; 3], 16, &Leaf::Integer.bytes(child));
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::DatatypeChildVersion {
            class: 9,
            version,
            child_version: child.get()
        }
    );
}

#[rstest]
#[case::reserved_order_v1(0x40, Version::One, FormatError::InvalidByteOrder(0x40))]
#[case::reserved_order_v2(0x40, Version::Two, FormatError::InvalidByteOrder(0x40))]
#[case::reserved_order_v3(0x40, Version::Three, FormatError::InvalidByteOrder(0x40))]
#[case::reserved_order_v4(0x40, Version::Four, FormatError::InvalidByteOrder(0x40))]
#[case::reserved_order_v5(0x40, Version::Five, FormatError::InvalidByteOrder(0x40))]
#[case::old_vax_v1(0x41, Version::One, FormatError::InvalidDatatypeVersion { class: 1, version: 1 })]
#[case::old_vax_v2(0x41, Version::Two, FormatError::InvalidDatatypeVersion { class: 1, version: 2 })]
fn datatype_variable_length_parser_rejects_invalid_float_parent(
    #[case] order: u8,
    #[case] version: Version,
    #[case] expected: FormatError,
) {
    let mut parent = Leaf::Float(Order::Little).bytes(version);
    parent[1] |= order;
    let bytes = datatype::variable_length(5, [0; 3], 16, &parent);
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::empty(0, 1)]
#[case::outer_header(7, 8)]
#[case::missing_parent(8, 9)]
#[case::inner_header(15, 16)]
#[case::missing_leaf(16, 17)]
#[case::leaf_precision(27, 28)]
fn datatype_variable_length_parser_reports_incomplete_nested_fields(
    #[case] available: usize,
    #[case] expected: usize,
) {
    let nested = datatype::variable_length(3, [0; 3], 16, &Leaf::Integer.bytes(Version::One));
    let bytes = datatype::variable_length(5, [0; 3], 16, &nested);
    assert_eq!(
        DatatypeMessage::parse(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[test]
fn datatype_variable_length_codecs_bound_recursive_descriptions() {
    let leaf = Leaf::Integer.bytes(Version::One);
    let bytes = (0..63).fold(leaf, |parent, _| {
        datatype::variable_length(1, [0; 3], 16, &parent)
    });
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
    let too_deep = datatype::variable_length(1, [0; 3], 16, &bytes);
    assert_eq!(
        DatatypeMessage::parse(&too_deep).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
    let too_deep = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(message),
        },
    )
    .unwrap();
    assert_eq!(
        too_deep.serialize(OffsetWidth::Eight).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
}

fn descriptor_size(width: OffsetWidth) -> u32 {
    8 + u32::from(width.get())
}

fn assert_encoding(bytes: &[u8], expected: DatatypeMessage, width: OffsetWidth) {
    let mut input = bytes.to_vec();
    input.push(0xff);
    let (message, consumed) = DatatypeMessage::parse(&input).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message, expected);
    assert_eq!(message.serialize(width).unwrap(), bytes);
}
