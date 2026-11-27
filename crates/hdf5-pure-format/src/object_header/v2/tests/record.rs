use rstest::rstest;
use test_util::object_header::v2 as v2_bytes;

use super::*;

#[rstest]
#[case::a_record_that_ends_the_region(
    MessageRecordLayout::PLAIN,
    &[0x01, 2, 0, 0x01, 7, 8],
    Ok(Some(MessageRecord {
        msg_type: MessageType::DATASPACE,
        flags: MessageFlags::CONSTANT,
        creation_index: None,
        body: &[7, 8],
        body_range: 4..6,
    }))
)]
#[case::a_tracked_record(
    TRACKED,
    &[0x01, 1, 0, 0, 0x34, 0x12, 7],
    Ok(Some(MessageRecord {
        msg_type: MessageType::DATASPACE,
        flags: MessageFlags::NONE,
        creation_index: Some(0x1234),
        body: &[7],
        body_range: 6..7,
    }))
)]
#[case::a_tail_shorter_than_a_record_prefix(MessageRecordLayout::PLAIN, &[0x01, 2, 0], Ok(None))]
#[case::a_tail_shorter_than_a_tracked_prefix(TRACKED, &[0x01, 0, 0, 0, 0x34], Ok(None))]
#[case::a_body_past_the_region(
    MessageRecordLayout::PLAIN,
    &[0x01, 3, 0, 0, 7, 8],
    Err(FormatError::UnexpectedEof { expected: 7, available: 6 })
)]
fn a_record_is_read_within_its_region(
    #[case] layout: MessageRecordLayout,
    #[case] region: &[u8],
    #[case] expected: Result<Option<MessageRecord<'_>>, FormatError>,
) {
    assert_eq!(layout.next_message(region, 0), expected);
}

#[rstest]
#[case::a_plain_record(MessageRecordLayout::PLAIN, &[0x01, 0, 0, 0][..], None)]
#[case::a_tracked_record(TRACKED, &[0x01, 0, 0, 0, 0x34, 0x12][..], Some(0x1234))]
#[case::a_tracked_record_cut_short(TRACKED, &[0x01, 0, 0, 0, 0x34][..], None)]
fn a_creation_index_is_read_where_the_layout_has_one(
    #[case] layout: MessageRecordLayout,
    #[case] region: &[u8],
    #[case] expected: Option<u16>,
) {
    assert_eq!(layout.creation_index(region, 0), expected);
}

#[rstest]
#[case::plain(MessageRecordLayout::PLAIN, &[0x0C, 1, 0, 0x02, 0xAA][..])]
#[case::tracked(TRACKED, &[0x0C, 1, 0, 0x02, 0x02, 0x01, 0xAA][..])]
fn a_record_stores_a_creation_index_only_where_the_header_tracks_one(
    #[case] layout: MessageRecordLayout,
    #[case] expected: &[u8],
) {
    assert_eq!(
        layout
            .prepare_raw_record(
                MessageType::ATTRIBUTE,
                MessageFlags::SHARED,
                0x0102,
                &[0xaa]
            )
            .unwrap()
            .to_bytes()
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::plain(MessageRecordLayout::PLAIN, &[0x0A, 1, 0, 0, 0xAA][..])]
#[case::tracked(TRACKED, &[0x0A, 1, 0, 0, 0, 0, 0xAA][..])]
fn group_info_encodes_the_supplied_zero_index_only_when_tracked(
    #[case] layout: MessageRecordLayout,
    #[case] expected: &[u8],
) {
    let record = layout
        .prepare_raw_record(MessageType::GROUP_INFO, MessageFlags::NONE, 0, &[0xaa])
        .unwrap();
    assert_eq!(record.to_bytes().unwrap(), expected);
}

const TRACKED: MessageRecordLayout = MessageRecordLayout::from_header_flags(
    v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0
        | v2_bytes::HeaderFlags::INDEXES_CREATION_ORDER.0,
);
