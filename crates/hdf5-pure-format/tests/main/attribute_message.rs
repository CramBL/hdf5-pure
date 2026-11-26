use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::CompoundMember;
use hdf5_pure_core::DatatypeMemberName;
use hdf5_pure_core::{
    CharacterSet, FormatError, LengthWidth, MaxExtent, OffsetWidth, StoredAddress, StringPadding,
    VariableLengthType,
};
use hdf5_pure_format::__private::{
    AttributeMessage, Dataspace, DataspaceType, DatatypeLocation, FormatWidths, MessageFlags,
    MessageRecordLayout, PreparedAttributeBody,
};
use hdf5_pure_format::{Datatype, MessageType};
use rstest::rstest;
use test_util::attribute::{UnpaddedAttribute, UnpaddedVersion};
use test_util::datatype::containers::MemberOffset;
use test_util::datatype::leaf::Version;
use test_util::object_header::v2::{self, HeaderFlags};
use test_util::object_header::{Message, MessageFlags as FixtureFlags, MessageType as FixtureType};
use test_util::widths::Widths;
use test_util::{dataspace, datatype};

#[rstest]
#[case::ascii_v2("array", AttributeMessage::prepare, UnpaddedVersion::Two)]
#[case::ascii_v3("array", AttributeMessage::prepare_v3, UnpaddedVersion::Three)]
#[case::utf8_v3("échelle", AttributeMessage::prepare, UnpaddedVersion::Three)]
fn prepared_inline_attribute_matches_independent_bytes(
    #[case] name: &str,
    #[case] prepare: for<'a> fn(
        &'a AttributeMessage,
        FormatWidths,
    ) -> Result<PreparedAttributeBody<'a>, FormatError>,
    #[case] version: UnpaddedVersion,
    #[values((OffsetWidth::Two, LengthWidth::Four), (OffsetWidth::Four, LengthWidth::Eight), (OffsetWidth::Eight, LengthWidth::Two))]
    width_pair: (OffsetWidth, LengthWidth),
) {
    let (offsets, lengths) = width_pair;
    let widths = FormatWidths { offsets, lengths };
    let wire_widths = Widths::new(usize::from(offsets.get()), usize::from(lengths.get()));
    let parent = datatype::fixed_string(1);
    let sequence = datatype::VariableLengthSequence::new(Version::One, wire_widths.offset, &parent);
    let data = vec![0; usize::try_from(sequence.descriptor_size()).unwrap() * 2];
    let message = AttributeMessage {
        name: name.into(),
        datatype: Datatype::VariableLength(VariableLengthType::sequence(one_byte_string())),
        dataspace: Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![2],
            max_dimensions: Some(vec![MaxExtent::Fixed(0x1234)]),
        },
        raw_data: data,
        datatype_location: DatatypeLocation::Inline,
    };
    let datatype = sequence.build();
    let dataspace = dataspace::VersionTwo::simple(wire_widths, &[2], Some(&[0x1234])).build();
    let expected =
        UnpaddedAttribute::inline(version, name, &datatype, &dataspace, &message.raw_data).build();

    let prepared = prepare(&message, widths).unwrap();
    assert_bounded_encoding(&prepared, &expected);
    assert_eq!(
        AttributeMessage::parse_with_widths(&expected, widths).unwrap(),
        message
    );
}

#[rstest]
#[case::zero(OffsetWidth::Two, 0)]
#[case::two_bytes(OffsetWidth::Two, 0xfffe)]
#[case::four_bytes(OffsetWidth::Four, 0xffff_fffe)]
#[case::eight_bytes(OffsetWidth::Eight, u64::MAX - 1)]
fn prepared_committed_attribute_matches_independent_bytes(
    #[case] offsets: OffsetWidth,
    #[case] address: u64,
) {
    let message = committed_attribute(address);
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Four,
    };
    let expected = UnpaddedAttribute::committed(
        UnpaddedVersion::Three,
        &message.name,
        address,
        Widths::new(usize::from(offsets.get()), 4),
        &dataspace::VersionTwo::scalar().build(),
        &message.raw_data,
    )
    .build();
    assert_bounded_encoding(&message.prepare_v3(widths).unwrap(), &expected);
}

#[rstest]
#[case::plain(HeaderFlags::default())]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER)]
fn prepared_attribute_composes_as_an_inline_record(#[case] header_flags: HeaderFlags) {
    let address = 0x1234;
    let message = committed_attribute(address);
    let widths = FormatWidths {
        offsets: OffsetWidth::Two,
        lengths: LengthWidth::Eight,
    };
    let expected_body = UnpaddedAttribute::committed(
        UnpaddedVersion::Two,
        &message.name,
        address,
        Widths::new(2, 8),
        &dataspace::VersionTwo::scalar().build(),
        &message.raw_data,
    )
    .build();
    let creation_index = 0x3456;
    let expected = v2::message_record(
        &Message::new(FixtureType::ATTRIBUTE, &expected_body)
            .with_flags(FixtureFlags::CONSTANT)
            .with_creation_order(creation_index),
        header_flags,
    );
    let record = MessageRecordLayout::from_header_flags(header_flags.0)
        .prepare_record(
            MessageFlags::CONSTANT,
            creation_index,
            message.prepare(widths).unwrap(),
        )
        .unwrap();
    let mut output = vec![0xa5; expected.len() + 1];
    assert_eq!(record.encoded_size(), expected.len());
    assert_eq!(record.write_into(&mut output).unwrap(), expected.len());
    assert_eq!(&output[..expected.len()], expected);
    assert_eq!(output[expected.len()], 0xa5);
}

#[rstest]
#[case::two_byte_overflow(OffsetWidth::Two, 0x1_0000, FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 })]
#[case::four_byte_overflow(OffsetWidth::Four, 0x1_0000_0000, FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 })]
#[case::two_byte_sentinel(OffsetWidth::Two, 0xffff, FormatError::DefinedAddressIsUndefined { address: 0xffff, offset_size: 2 })]
#[case::four_byte_sentinel(OffsetWidth::Four, 0xffff_ffff, FormatError::DefinedAddressIsUndefined { address: 0xffff_ffff, offset_size: 4 })]
#[case::eight_byte_sentinel(OffsetWidth::Eight, u64::MAX, FormatError::DefinedAddressIsUndefined { address: u64::MAX, offset_size: 8 })]
fn attribute_preparation_rejects_invalid_committed_addresses(
    #[case] offsets: OffsetWidth,
    #[case] address: u64,
    #[case] expected: FormatError,
) {
    let message = committed_attribute(address);
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Eight,
    };
    assert_eq!(message.prepare(widths).unwrap_err(), expected);
}

#[rstest]
#[case::valid_datatype(one_byte_string(), FormatError::InvalidLengthSize(1))]
#[case::invalid_datatype(
    Datatype::Opaque { size: 1, tag: vec![0] },
    FormatError::InvalidOpaqueTagByte { offset: 0, byte: 0 }
)]
fn attribute_checks_datatype_before_raw_length_width(
    #[case] datatype: Datatype,
    #[case] expected: FormatError,
) {
    let mut message = inline_attribute();
    message.datatype = datatype;
    assert_eq!(
        message.serialize(OffsetWidth::Two, 1).unwrap_err(),
        expected
    );
}

#[test]
fn attribute_dataspace_error_precedes_an_overlong_name() {
    let mut message = inline_attribute();
    message.name = "x".repeat(usize::from(u16::MAX));
    message.dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![0x1_0000],
        max_dimensions: None,
    };
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Two,
    };
    assert_eq!(
        message.prepare(widths).unwrap_err(),
        FormatError::LengthTooLarge {
            length: 0x1_0000,
            length_size: 2
        },
    );
}

#[test]
fn attribute_name_size_error_precedes_the_datatype_size_error() {
    let mut message = inline_attribute();
    let long_name = "x".repeat(usize::from(u16::MAX));
    message.datatype = Datatype::Compound {
        size: 1,
        members: vec![CompoundMember::new(
            DatatypeMemberName::new(&long_name).unwrap(),
            0,
            one_byte_string(),
        )],
    };
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    let FormatError::AttributeFieldTooLong {
        name,
        field,
        size,
        limit,
    } = message.prepare(widths).unwrap_err()
    else {
        panic!("expected AttributeFieldTooLong")
    };
    assert_eq!(name, message.name);
    assert_eq!(field, "datatype");
    assert_eq!(
        size,
        DATATYPE_HEADER_SIZE
            + long_name.len()
            + size_of::<u8>()
            + MemberOffset::One.get()
            + datatype::fixed_string(1).len(),
    );
    assert_eq!(limit, usize::from(u16::MAX));

    message.name = long_name;
    assert_eq!(
        message.prepare(widths).unwrap_err(),
        FormatError::AttributeFieldTooLong {
            name: message.name.clone(),
            field: "name",
            size: message.name.len() + 1,
            limit: usize::from(u16::MAX),
        }
    );
}

#[test]
fn attribute_body_length_is_independent_of_the_compact_record_limit() {
    let count = usize::from(u16::MAX) + 1;
    let mut message = inline_attribute();
    let dimensions = [u64::try_from(count).unwrap()];
    message.dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: dimensions.to_vec(),
        max_dimensions: None,
    };
    message.raw_data = vec![7; count];
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    let prepared = message.prepare(widths).unwrap();
    let dataspace = dataspace::VersionTwo::simple(Widths::EIGHT, &dimensions, None).build();
    let expected_len = UnpaddedAttribute::inline(
        UnpaddedVersion::Two,
        &message.name,
        &datatype::fixed_string(1),
        &dataspace,
        &message.raw_data,
    )
    .build()
    .len();
    assert_eq!(prepared.encoded_size(), expected_len);
    let Err(error) = MessageRecordLayout::PLAIN.prepare_record(MessageFlags::NONE, 0, prepared)
    else {
        panic!("expected the body to exceed the record size limit");
    };
    assert_eq!(
        error,
        FormatError::ObjectHeaderMessageTooLarge {
            message_type: MessageType::ATTRIBUTE.to_u16(),
            size: expected_len,
        },
    );
}

fn assert_bounded_encoding(prepared: &PreparedAttributeBody<'_>, expected: &[u8]) {
    let len = expected.len();
    assert_eq!(prepared.encoded_size(), len);
    assert_eq!(prepared.to_bytes(), expected);
    #[repr(C, align(8))]
    struct Aligned([u8; 128]);
    let mut storage = Aligned([0xa5; 128]);
    let output = &mut storage.0[..len + 3];
    for available in [0, len - 1] {
        assert_eq!(
            prepared.write_into(&mut output[1..1 + available]),
            Err(FormatError::UnexpectedEof {
                expected: len,
                available
            }),
        );
        assert_eq!(output, &[0xa5; 128][..len + 3]);
    }
    assert_eq!(prepared.write_into(&mut output[1..]).unwrap(), len);
    assert_eq!(output[0], 0xa5);
    assert_eq!(&output[1..1 + len], expected);
    assert_eq!(output[1 + len..], [0xa5; 2]);
}

fn inline_attribute() -> AttributeMessage {
    AttributeMessage {
        name: "a".into(),
        datatype: one_byte_string(),
        dataspace: Dataspace {
            space_type: DataspaceType::Scalar,
            rank: 0,
            dimensions: Vec::new(),
            max_dimensions: None,
        },
        raw_data: vec![7],
        datatype_location: DatatypeLocation::Inline,
    }
}

fn committed_attribute(address: u64) -> AttributeMessage {
    AttributeMessage {
        datatype_location: DatatypeLocation::Committed(StoredAddress::new(address)),
        ..inline_attribute()
    }
}

fn one_byte_string() -> Datatype {
    Datatype::String {
        size: 1,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Ascii,
    }
}

// "The Datatype Message" in version 4.0 defines the header before the class-specific properties.
const DATATYPE_HEADER_SIZE: usize = size_of::<[u8; 4]>() + size_of::<u32>();
