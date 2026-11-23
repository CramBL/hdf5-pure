use hdf5_pure_format::__private::{
    FilterDescription, GroupEstimates, GroupInfo, MessageBody, MessageFlags, MessageRecord,
    MessageRecordLayout, PreparedFilterPipeline,
};
use hdf5_pure_format::{FormatError, MessageType};
use rstest::rstest;
use test_util::group_info::{Estimates, GroupInfo as GroupInfoFixture};
use test_util::object_header::v2::{self, HeaderFlags};
use test_util::object_header::{Message, MessageFlags as FixtureFlags, MessageType as FixtureType};

#[rstest]
#[case::plain(HeaderFlags::default(), None)]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER, Some(CREATION_INDEX))]
fn writer_encodes_a_typed_record_with_the_selected_prefix(
    #[case] header_flags: HeaderFlags,
    #[case] creation_index: Option<u16>,
) {
    let body = group_info_body();
    let info = GroupInfo::new(
        None,
        Some(GroupEstimates {
            entries: ENTRIES,
            name_length: NAME_LENGTH,
        }),
    );
    let layout = MessageRecordLayout::from_header_flags(header_flags.0);
    let expected = v2::message_record(
        &Message::new(FixtureType::GROUP_INFO, &body)
            .with_flags(FixtureFlags::CONSTANT)
            .with_creation_order(CREATION_INDEX),
        header_flags,
    );

    let encoded = typed_record(layout, &info);
    assert_eq!(encoded, expected);
    assert_eq!(
        layout.next_message(&encoded, 0).unwrap().unwrap(),
        MessageRecord {
            msg_type: MessageType::GROUP_INFO,
            flags: MessageFlags::CONSTANT,
            creation_index,
            body: &body,
            body_range: (expected.len() - body.len())..expected.len(),
        }
    );
}

#[rstest]
#[case::plain(HeaderFlags::default(), None)]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER, Some(CREATION_INDEX))]
fn writer_appends_an_opaque_shared_record_with_its_flags(
    #[case] header_flags: HeaderFlags,
    #[case] creation_index: Option<u16>,
) {
    let body = test_util::datatype::committed_reference(0x1248);
    let layout = MessageRecordLayout::from_header_flags(header_flags.0);
    let flags = MessageFlags::SHARED | MessageFlags::CONSTANT;
    let record = layout
        .prepare_raw_record(MessageType::DATATYPE, flags, CREATION_INDEX, &body)
        .unwrap();
    let expected = v2::message_record(
        &Message::new(FixtureType::DATATYPE, &body)
            .with_flags(FixtureFlags::SHARED | FixtureFlags::CONSTANT)
            .with_creation_order(CREATION_INDEX),
        header_flags,
    );
    let mut output = vec![0xa5];
    let start = output.len();

    assert_eq!(record.encoded_size(), expected.len());
    record.append_to(&mut output).unwrap();
    assert_eq!(output[0], 0xa5);
    assert_eq!(&output[start..], expected);
    assert_eq!(
        layout.next_message(&output, start).unwrap().unwrap(),
        MessageRecord {
            msg_type: MessageType::DATATYPE,
            flags,
            creation_index,
            body: &body,
            body_range: (output.len() - body.len())..output.len(),
        }
    );
}

#[rstest]
#[case::one_byte_type_limit(MAX_RECORD_TYPE, 0)]
#[case::two_byte_size_limit(MAX_RECORD_TYPE, MAX_RECORD_BODY_SIZE)]
fn writer_preserves_an_unknown_type_and_boundary_body_size(
    #[case] type_code: u16,
    #[case] body_size: usize,
) {
    let body = vec![0x7b; body_size];
    let expected = v2::message_record(
        &Message::new(FixtureType(type_code), &body).with_flags(FixtureFlags::WAS_UNKNOWN),
        HeaderFlags::default(),
    );
    let record = MessageRecordLayout::PLAIN
        .prepare_raw_record(
            MessageType::from_u16(type_code),
            MessageFlags::WAS_UNKNOWN,
            0,
            &body,
        )
        .unwrap();
    let mut bytes = vec![0; expected.len()];
    assert_eq!(record.write_into(&mut bytes).unwrap(), expected.len());
    assert_eq!(bytes, expected);
}

#[rstest]
#[case::type_first(
    MAX_RECORD_TYPE + 1,
    MAX_RECORD_BODY_SIZE + 1,
    FormatError::ObjectHeaderMessageTypeTooLarge(MAX_RECORD_TYPE + 1),
)]
#[case::body_too_large(
    FixtureType::NIL.0,
    MAX_RECORD_BODY_SIZE + 1,
    FormatError::ObjectHeaderMessageTooLarge {
        message_type: FixtureType::NIL.0,
        size: MAX_RECORD_BODY_SIZE + 1,
    },
)]
fn raw_preparation_checks_the_type_and_body_size(
    #[case] type_code: u16,
    #[case] size: usize,
    #[case] expected: FormatError,
) {
    let bytes = vec![0; size];
    assert_eq!(
        MessageRecordLayout::PLAIN
            .prepare_raw_record(
                MessageType::from_u16(type_code),
                MessageFlags::NONE,
                0,
                &bytes
            )
            .unwrap_err(),
        expected,
    );
}

#[test]
fn typed_preparation_checks_the_body_size() {
    let filters = [FilterDescription {
        filter_id: CUSTOM_FILTER_ID,
        name: None,
        flags: 0,
        client_data: vec![0; MAX_FILTER_CLIENT_COUNT],
    }];
    let pipeline =
        PreparedFilterPipeline::new(test_util::filter_pipeline::Version::Two.byte(), &filters)
            .unwrap();
    assert_eq!(
        MessageRecordLayout::PLAIN
            .prepare_record(MessageFlags::NONE, 0, pipeline)
            .unwrap_err(),
        FormatError::ObjectHeaderMessageTooLarge {
            message_type: FixtureType::FILTER_PIPELINE.0,
            size: 262_150,
        }
    );
}

#[test]
fn typed_preparation_rejects_shared_flags() {
    let flags = MessageFlags::SHARED | MessageFlags::CONSTANT;
    assert_eq!(
        MessageRecordLayout::PLAIN
            .prepare_record(flags, 0, GroupInfo::DEFAULT)
            .unwrap_err(),
        FormatError::SharedFlagForInlineMessage {
            message_type: FixtureType::GROUP_INFO.0,
            flags: (FixtureFlags::SHARED | FixtureFlags::CONSTANT).0,
        }
    );
}

#[rstest]
#[case::plain(HeaderFlags::default())]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER)]
fn writer_preserves_short_destinations_and_guards(#[case] header_flags: HeaderFlags) {
    let body = group_info_body();
    let info = GroupInfo::new(
        None,
        Some(GroupEstimates {
            entries: ENTRIES,
            name_length: NAME_LENGTH,
        }),
    );
    let layout = MessageRecordLayout::from_header_flags(header_flags.0);
    let expected = v2::message_record(&Message::new(FixtureType::GROUP_INFO, &body), header_flags);
    let record = layout.prepare_record(MessageFlags::NONE, 0, info).unwrap();
    let required = expected.len();
    assert_eq!(record.encoded_size(), required);
    let original = vec![0xa5; required + 3];
    let mut output = original.clone();

    for available in 0..required {
        assert_eq!(
            record.write_into(&mut output[1..1 + available]),
            Err(FormatError::UnexpectedEof {
                expected: required,
                available
            })
        );
        assert_eq!(output, original);
    }
    assert_eq!(record.write_into(&mut output[1..]).unwrap(), required);
    assert_eq!(output[0], original[0]);
    assert_eq!(&output[1..1 + required], expected);
    assert_eq!(output[1 + required..], original[1 + required..]);
}

fn typed_record<B: MessageBody>(layout: MessageRecordLayout, body: &B) -> Vec<u8> {
    let record = layout
        .prepare_record(MessageFlags::CONSTANT, CREATION_INDEX, body)
        .unwrap();
    let mut bytes = vec![0; record.encoded_size()];
    assert_eq!(record.write_into(&mut bytes).unwrap(), bytes.len());
    bytes
}

fn group_info_body() -> Vec<u8> {
    GroupInfoFixture {
        phase_change: None,
        estimates: Some(Estimates {
            entries: ENTRIES,
            name_length: NAME_LENGTH,
        }),
    }
    .build()
}

const CREATION_INDEX: u16 = 0x1234;
const ENTRIES: u16 = 7;
const NAME_LENGTH: u16 = 19;
const CUSTOM_FILTER_ID: u16 = 300;
const MAX_RECORD_TYPE: u16 = u8::MAX as u16;
const MAX_RECORD_BODY_SIZE: usize = u16::MAX as usize;
const MAX_FILTER_CLIENT_COUNT: usize = u16::MAX as usize;
