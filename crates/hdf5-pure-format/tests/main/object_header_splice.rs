use hdf5_pure_format::__private::{MessageFlags, MessageRecordLayout, RecordSplice};
use hdf5_pure_format::{FormatError, MessageType};
use rstest::rstest;
use test_util::object_header::v2::{self, HeaderFlags};
use test_util::object_header::{self, Message};

#[rstest]
#[case::plain(HeaderFlags(0))]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER)]
fn a_body_writer_receives_only_the_replacement_body(#[case] flags: HeaderFlags) {
    let source = records(&[0x12; 4], flags);
    let layout = MessageRecordLayout::from_header_flags(flags.0);
    let splice = RecordSplice::first(layout, &source, MessageType::DATA_LAYOUT)
        .unwrap()
        .unwrap();
    let mut output = vec![0xa5; 5];

    splice
        .replace_body_with(&mut output, 9, |body| {
            assert_eq!(body.len(), 9);
            body.fill(0x34);
            Ok(())
        })
        .unwrap();

    let mut expected = vec![0xa5; 5];
    expected.extend_from_slice(&records(&[0x34; 9], flags));
    assert_eq!(output, expected);
    let head_len = v2::message_record(&Message::nil(3), flags).len();
    let record = layout
        .next_message(&output[5..], head_len)
        .unwrap()
        .unwrap();
    assert_eq!(record.msg_type, MessageType::DATA_LAYOUT);
    assert_eq!(record.flags, MessageFlags::CONSTANT);
    assert_eq!(
        record.creation_index,
        flags.tracks_creation_order().then_some(7)
    );
}

#[test]
fn an_oversized_body_is_rejected_before_the_writer_runs() {
    let flags = HeaderFlags::TRACKS_CREATION_ORDER;
    let source = records(&[0x12; 4], flags);
    let layout = MessageRecordLayout::from_header_flags(flags.0);
    let splice = RecordSplice::first(layout, &source, MessageType::DATA_LAYOUT)
        .unwrap()
        .unwrap();
    let original = [0xa5, 0x5a, 0xaa];
    let mut output = original.to_vec();
    let size = usize::from(u16::MAX) + 1;

    let error = splice
        .replace_body_with(&mut output, size, |_| panic!("oversized body writer ran"))
        .unwrap_err();

    assert_eq!(
        error,
        FormatError::ObjectHeaderMessageTooLarge {
            message_type: MessageType::DATA_LAYOUT.to_u16(),
            size,
        }
    );
    assert_eq!(output, original);
}

#[rstest]
#[case::plain(HeaderFlags(0))]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER)]
fn a_failed_body_write_leaves_the_destination_unchanged(#[case] flags: HeaderFlags) {
    let source = records(&[0x12; 4], flags);
    let layout = MessageRecordLayout::from_header_flags(flags.0);
    let splice = RecordSplice::first(layout, &source, MessageType::DATA_LAYOUT)
        .unwrap()
        .unwrap();
    let original = [0xa5, 0x5a, 0xaa];
    let mut output = original.to_vec();
    let expected = FormatError::UnexpectedEof {
        expected: 12,
        available: 9,
    };

    let error = splice
        .replace_body_with(&mut output, 9, |body| {
            body[..3].copy_from_slice(&[0x34; 3]);
            Err(FormatError::UnexpectedEof {
                expected: 12,
                available: body.len(),
            })
        })
        .unwrap_err();

    assert_eq!(error, expected);
    assert_eq!(output, original);
}

fn records(body: &[u8], flags: HeaderFlags) -> Vec<u8> {
    let mut region = v2::message_records(
        &[
            Message::nil(3),
            Message::new(object_header::MessageType::DATA_LAYOUT, body)
                .with_flags(object_header::MessageFlags::CONSTANT)
                .with_creation_order(7),
            Message::new(object_header::MessageType::DATA_LAYOUT, &[0x56; 5]),
        ],
        flags,
    );
    region.extend_from_slice(&[0x78; 3]);
    region
}
