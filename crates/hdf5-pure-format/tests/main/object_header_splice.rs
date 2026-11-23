use hdf5_pure_format::__private::{
    AttributeInfoMessage, GroupInfo, MessageRecordLayout, RecordSplice,
};
use hdf5_pure_format::{FormatError, MessageType, OffsetWidth};
use rstest::rstest;
use test_util::attribute_info::{AttributeInfo, CreationOrderIndex, Width};
use test_util::object_header::v2::{self, HeaderFlags};
use test_util::object_header::{Message, MessageFlags as FixtureFlags, MessageType as FixtureType};

#[rstest]
#[case::plain(HeaderFlags::default())]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER)]
fn splice_writer_preserves_record_metadata_and_surrounding_bytes_for_a_typed_body(
    #[case] flags: HeaderFlags,
) {
    let original_body =
        AttributeInfo::new(Width::Two, None, None, None, CreationOrderIndex::Absent).build();
    let updated_body = AttributeInfo::new(
        Width::Two,
        Some(13),
        None,
        None,
        CreationOrderIndex::Undefined,
    )
    .build();
    let source = records(&original_body, flags);
    let layout = MessageRecordLayout::from_header_flags(flags.0);
    let splice = RecordSplice::first(layout, &source, MessageType::ATTRIBUTE_INFO)
        .unwrap()
        .unwrap();
    let mut output = vec![0xa5; 5];

    splice
        .replace_typed_body_into(
            &mut output,
            AttributeInfoMessage::compact(OffsetWidth::Two, Some(13), true),
        )
        .unwrap();

    let mut expected = vec![0xa5; 5];
    expected.extend_from_slice(&records(&updated_body, flags));
    assert_eq!(output, expected);
}

#[rstest]
#[case::type_mismatch_with_shared_flag(
    FixtureType::ATTRIBUTE_INFO,
    FixtureFlags::SHARED,
    FormatError::ObjectHeaderMessageTypeMismatch {
        expected: FixtureType::GROUP_INFO.0,
        actual: FixtureType::ATTRIBUTE_INFO.0,
    },
)]
#[case::shared_reference(
    FixtureType::GROUP_INFO,
    FixtureFlags::SHARED,
    FormatError::SharedFlagForInlineMessage {
        message_type: FixtureType::GROUP_INFO.0,
        flags: FixtureFlags::SHARED.0,
    },
)]
fn splice_writer_preserves_the_destination_when_typed_preparation_fails(
    #[case] fixture_type: FixtureType,
    #[case] flags: FixtureFlags,
    #[case] expected: FormatError,
) {
    let layout = MessageRecordLayout::PLAIN;
    let source = v2::message_record(
        &Message::new(fixture_type, &[0x37; 4]).with_flags(flags),
        HeaderFlags::default(),
    );
    let splice = RecordSplice::first(layout, &source, MessageType::from_u16(fixture_type.0))
        .unwrap()
        .unwrap();
    let original = [0xa5, 0x5a, 0xaa];
    let mut output = original.to_vec();

    assert_eq!(
        splice.replace_typed_body_into(&mut output, GroupInfo::DEFAULT),
        Err(expected)
    );
    assert_eq!(output, original);
}

#[test]
fn splice_writer_preserves_shared_record_metadata_during_raw_replacement() {
    let flags = HeaderFlags::TRACKS_CREATION_ORDER;
    let layout = MessageRecordLayout::from_header_flags(flags.0);
    let fixture_type = FixtureType(0xff);
    let record = |body| {
        v2::message_record(
            &Message::new(fixture_type, body)
                .with_flags(FixtureFlags::SHARED | FixtureFlags::MARK_IF_UNKNOWN)
                .with_creation_order(19),
            flags,
        )
    };
    let source = record(b"before");
    let splice = RecordSplice::first(layout, &source, MessageType::from_u16(fixture_type.0))
        .unwrap()
        .unwrap();

    assert_eq!(splice.replace_body(b"after").unwrap(), record(b"after"));
}

fn records(body: &[u8], flags: HeaderFlags) -> Vec<u8> {
    let mut region = v2::message_records(
        &[
            Message::nil(3),
            Message::new(FixtureType::ATTRIBUTE_INFO, body)
                .with_flags(FixtureFlags::CONSTANT)
                .with_creation_order(7),
            Message::new(FixtureType::ATTRIBUTE_INFO, &[0x56; 5]),
        ],
        flags,
    );
    region.extend_from_slice(&[0x78; 3]);
    region
}
