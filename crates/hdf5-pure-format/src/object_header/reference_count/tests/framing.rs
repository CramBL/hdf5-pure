use rstest::rstest;

use crate::object_header::{ObjectReferenceCountIssue, ObjectReferenceCountMessage};

#[rstest]
#[case::zero(0, [0, 0, 0, 0, 0])]
#[case::one(1, [0, 1, 0, 0, 0])]
#[case::ordinary(0x1234_5678, [0, 0x78, 0x56, 0x34, 0x12])]
#[case::maximum(u32::MAX, [0, 0xFF, 0xFF, 0xFF, 0xFF])]
fn count_message_framing_matches_known_bytes(#[case] count: u32, #[case] bytes: [u8; 5]) {
    assert_eq!(ObjectReferenceCountMessage::new(count).encode(), bytes);
    assert_eq!(
        ObjectReferenceCountMessage::parse(&bytes).unwrap().count(),
        count
    );
    let mut padded = bytes.to_vec();
    padded.extend_from_slice(&[0xA7, 0x59, 0x31]);
    assert_eq!(
        ObjectReferenceCountMessage::parse(&padded).unwrap().count(),
        count
    );
}

#[rstest]
#[case::empty(&[], ObjectReferenceCountIssue::BodyLength(0))]
#[case::short(&[0, 7, 0, 0], ObjectReferenceCountIssue::BodyLength(4))]
#[case::version(&[1, 7, 0, 0, 0], ObjectReferenceCountIssue::MessageVersion(1))]
fn count_message_framing_reports_unsupported_bodies(
    #[case] bytes: &[u8],
    #[case] expected: ObjectReferenceCountIssue,
) {
    assert_eq!(
        ObjectReferenceCountMessage::parse(bytes).unwrap_err(),
        expected
    );
}
