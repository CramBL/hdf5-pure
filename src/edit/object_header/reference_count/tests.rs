use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::ObjectReferenceCountIssue;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::checksum;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2;

use crate::error::FormatError;

use super::HeaderCountError;
use super::OhRegion;

#[rstest]
#[case::unchanged(0x1234_5678)]
#[case::one(1)]
#[case::maximum(u32::MAX)]
#[case::zero(0)]
fn explicit_count_projection_preserves_record_geometry_and_unrelated_bytes(
    #[case] count: u32,
    #[values(false, true)] tracked: bool,
    #[values(0, 2)] gap: usize,
) {
    let flags = flags(tracked);
    let initial = records(0x1234_5678);
    let mut region = region(&initial, tracked, gap);
    let props = region.props();
    let capacity = region.bytes.capacity();
    let projection = region.reference_count_mut().unwrap();
    assert_eq!(projection.count(), 0x1234_5678);
    projection.set(count).unwrap();
    let mut expected = v2::message_records(&records(count), flags);
    expected.extend(core::iter::repeat_n(0xA5, gap));
    assert_eq!(&region[..], expected);
    assert_eq!(region.props(), props);
    assert_eq!(region.bytes.capacity(), capacity);
    let encoded = props.encode_header(&region);
    let checksum_at = encoded.len() - checksum::CHECKSUM;
    assert_eq!(
        &encoded[checksum_at..],
        checksum::lookup3(&encoded[..checksum_at]).to_le_bytes()
    );
}

#[rstest]
#[case::implicit_one(1)]
#[case::explicit(2)]
#[case::zero(0)]
fn absent_counts_are_inserted_before_a_trailing_gap_only_when_needed(
    #[case] count: u32,
    #[values(false, true)] tracked: bool,
    #[values(false, true)] occupied: bool,
) {
    let mut records = if occupied {
        vec![Message::new(MessageType::UNKNOWN, &[0xD5; 7])]
    } else {
        vec![]
    };
    let mut region = region(&records, tracked, 2);
    let props = region.props();
    let projection = region.reference_count_mut().unwrap();
    assert_eq!(projection.count(), 1);
    projection.set(count).unwrap();
    if count != 1 {
        records.push(Message::reference_count(count));
    }
    let mut expected = v2::message_records(&records, flags(tracked));
    expected.extend_from_slice(&[0xA5; 2]);
    assert_eq!(&region[..], expected);
    assert_eq!(region.props(), props);
}

#[rstest]
#[case::shared(
    vec![Message::reference_count(7).with_flags(MessageFlags::SHARED)],
    HeaderCountError::Unsupported(ObjectReferenceCountIssue::Shared),
)]
#[case::version(
    vec![Message::new(MessageType::OBJECT_REFERENCE_COUNT, &[1, 7, 0, 0, 0])],
    HeaderCountError::Unsupported(ObjectReferenceCountIssue::MessageVersion(1)),
)]
#[case::short(
    vec![Message::new(MessageType::OBJECT_REFERENCE_COUNT, &[0, 7])],
    HeaderCountError::Unsupported(ObjectReferenceCountIssue::BodyLength(2)),
)]
#[case::repeated(
    vec![Message::reference_count(7), Message::reference_count(7)],
    HeaderCountError::Ambiguous,
)]
fn unusable_count_projections_leave_the_owned_region_unchanged(
    #[case] records: Vec<Message>,
    #[case] expected: HeaderCountError,
    #[values(false, true)] tracked: bool,
) {
    let mut region = region(&records, tracked, 2);
    let original = region.clone();
    let Err(error) = region.reference_count_mut() else {
        panic!("expected an unusable count projection")
    };
    assert_eq!(error, expected);
    assert_eq!(&region[..], &original[..]);
    assert_eq!(region.props(), original.props());
}

#[rstest]
fn a_count_projection_requires_complete_local_framing(#[values(false, true)] tracked: bool) {
    let mut region = region(&[Message::reference_count(7)], tracked, 0);
    let mut truncated = v2::message_record(
        &Message::new(MessageType::UNKNOWN, &[0; 20]),
        flags(tracked),
    );
    let expected = region.len() + truncated.len();
    truncated.truncate(truncated.len() - 17);
    region.bytes.extend_from_slice(&truncated);
    let original = region.clone();
    let Err(error) = region.reference_count_mut() else {
        panic!("expected incomplete local framing")
    };
    assert_eq!(
        error,
        HeaderCountError::Framing(FormatError::UnexpectedEof {
            expected,
            available: original.len(),
        })
    );
    assert_eq!(&region[..], &original[..]);
    assert_eq!(region.props(), original.props());
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
fn existing_count_updates_allocate_no_replacement_buffers(#[values(false, true)] tracked: bool) {
    let mut existing = region(&records(7), tracked, 2);
    let mut absent = region(&[], tracked, 2);
    let (control, allocated) = allocation::measure(&format!("insert_count_{tracked}"), || {
        absent.reference_count_mut().and_then(|count| count.set(3))
    });
    control.unwrap();
    assert!(allocated.blocks >= 1 && allocated.bytes >= 9, "{allocated}");
    let (result, measured) = allocation::measure(&format!("update_count_{tracked}"), || {
        existing
            .reference_count_mut()
            .and_then(|count| count.set(3))
    });
    result.unwrap();
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(existing.reference_count_mut().unwrap().count(), 3);
    assert_eq!(absent.reference_count_mut().unwrap().count(), 3);
}

fn records(count: u32) -> Vec<Message> {
    let mut count = Message::reference_count(count)
        .with_flags(MessageFlags::FORBID_SHARING)
        .with_creation_order(17);
    count.data.extend_from_slice(&[0xA7, 0x59, 0x31]);
    vec![
        Message::new(MessageType::UNKNOWN, &[0xD5; 7]).with_creation_order(11),
        count,
        Message::nil(9),
    ]
}

fn region(records: &[Message], tracked: bool, gap: usize) -> OhRegion {
    let mut bytes = v2::message_records(records, flags(tracked));
    bytes.extend(core::iter::repeat_n(0xA5, gap));
    let header = v2::Header::new()
        .flags(flags(tracked))
        .timestamps(v2::Timestamps {
            access: 11,
            modification: 13,
            change: 17,
            birth: 19,
        })
        .phase_change(9, 4)
        .messages(records.iter().cloned())
        .build();
    OhRegion::new(
        crate::file_writer::WIDTHS,
        bytes,
        ObjectHeaderPrefix::parse(&header).unwrap().prefix,
    )
}

fn flags(tracked: bool) -> v2::HeaderFlags {
    if tracked {
        v2::HeaderFlags::TRACKS_CREATION_ORDER | v2::HeaderFlags::INDEXES_CREATION_ORDER
    } else {
        v2::HeaderFlags::default()
    }
}
