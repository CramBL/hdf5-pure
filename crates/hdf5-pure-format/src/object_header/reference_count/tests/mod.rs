use hdf5_pure_core::__private::FileRegion;
use rstest::rstest;
use test_util::image::Image;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags as WireFlags;
use test_util::object_header::MessageType as WireType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::widths::Widths;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::object_header::MessageFilter;
use crate::width::FormatWidths;

use super::*;

mod update;

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

#[rstest]
#[case::zero(0)]
#[case::ordinary(0x1234_5678)]
#[case::maximum(u32::MAX)]
fn version_1_count_has_a_checked_prefix_location(
    #[case] count: u32,
    #[values(0, 8_589_934_592)] bias: u64,
) {
    let root = v1::Header::new()
        .reference_count(count)
        .message(Message::nil(16))
        .build();
    let ObjectReferenceCount::Version1(field) = observe(&root, None, bias, 8).unwrap() else {
        panic!("expected a version 1 count");
    };
    assert_field(field, count, bias + 516, bias + 512, 40, None);
    assert_eq!(&root[4..8], &count.to_le_bytes());
}

#[rstest]
#[case::plain(v2::HeaderFlags::default(), 7, None)]
#[case::tracked(v2::HeaderFlags::TRACKS_CREATION_ORDER, 7, None)]
#[case::prefix_fields(v2::HeaderFlags::STORES_TIMES | v2::HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE, 27, None)]
#[case::continuation(v2::HeaderFlags::default(), 4, Some(512))]
#[case::tracked_continuation(v2::HeaderFlags::TRACKS_CREATION_ORDER, 4, Some(512))]
fn version_2_count_retains_its_record_flags_and_checksum_chunk(
    #[case] flags: v2::HeaderFlags,
    #[case] prefix: u64,
    #[case] child: Option<u64>,
    #[values(2, 4, 8)] width: u8,
    #[values(0, 8_589_934_592)] bias: u64,
) {
    let body = [0, 0x78, 0x56, 0x34, 0x12, 0xA5, 0x5A];
    let message = Message::new(WireType::OBJECT_REFERENCE_COUNT, &body)
        .with_flags(WireFlags::CONSTANT)
        .with_creation_order(19);
    let continuation =
        child.map(|_| v2::continuation_chunk(core::slice::from_ref(&message), flags));
    let root = match &continuation {
        Some(bytes) => v2::Header::new()
            .flags(flags)
            .continuation(
                512,
                bytes,
                Widths {
                    offset: usize::from(width),
                    length: usize::from(width),
                },
            )
            .build(),
        None => v2::Header::new().flags(flags).message(message).build(),
    };
    let ObjectReferenceCount::Version2(field) =
        observe(&root, continuation.as_deref(), bias, width).unwrap()
    else {
        panic!("expected a version 2 count");
    };
    let chunk_start = bias + 512 + child.unwrap_or(0);
    let record_prefix = if flags.0 & v2::HeaderFlags::TRACKS_CREATION_ORDER.0 == 0 {
        4
    } else {
        6
    };
    let length = prefix + record_prefix + 7 + 4;
    assert_field(
        field,
        0x1234_5678,
        chunk_start + prefix + record_prefix + 1,
        chunk_start,
        length,
        Some(MessageFlags::CONSTANT),
    );
    assert_eq!(field.chunk().end().get() - field.range().end().get(), 6);
}

#[rstest]
#[case::absent(&[], WireFlags::NONE, ObjectReferenceCount::ImplicitOne)]
#[case::empty(&[], WireFlags::NONE, ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::BodyLength(0)))]
#[case::short(&[0, 1, 2, 3], WireFlags::NONE, ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::BodyLength(4)))]
#[case::version(&[1, 1, 2, 3, 4], WireFlags::NONE, ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::MessageVersion(1)))]
#[case::shared(&[0, 1, 2, 3, 4], WireFlags::SHARED, ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::Shared))]
fn unsupported_count_bodies_preserve_accepted_header_reads(
    #[case] body: &[u8],
    #[case] flags: WireFlags,
    #[case] expected: ObjectReferenceCount,
) {
    let header = v2::Header::new();
    let root = if expected == ObjectReferenceCount::ImplicitOne {
        header.message(Message::nil(16)).build()
    } else {
        header
            .message(Message::new(WireType::OBJECT_REFERENCE_COUNT, body).with_flags(flags))
            .build()
    };
    assert_eq!(observe(&root, None, 0, 8).unwrap(), expected);
}

#[rstest]
#[case::same_chunk(false)]
#[case::continuation(true)]
fn repeated_explicit_counts_remain_ambiguous(#[case] continuation: bool) {
    let count = Message::new(WireType::OBJECT_REFERENCE_COUNT, &[0, 2, 0, 0, 0]);
    let child = continuation
        .then(|| v2::continuation_chunk(core::slice::from_ref(&count), v2::HeaderFlags::default()));
    let header = v2::Header::new().message(count.clone());
    let root = match &child {
        Some(bytes) => header.continuation(512, bytes, Widths::EIGHT).build(),
        None => header.message(count).build(),
    };
    assert_eq!(
        observe(&root, child.as_deref(), 0, 8).unwrap(),
        ObjectReferenceCount::Ambiguous
    );
}

#[test]
fn a_version_1_prefix_with_an_explicit_count_is_ambiguous() {
    let root = v1::Header::new()
        .reference_count(3)
        .message(Message::new(WireType::OBJECT_REFERENCE_COUNT, &[0; 8]))
        .build();
    assert_eq!(
        observe(&root, None, 0, 8).unwrap(),
        ObjectReferenceCount::Ambiguous
    );
}

#[test]
fn a_later_parse_failure_does_not_produce_a_count_observation() {
    let root = v2::Header::new()
        .message(Message::new(
            WireType::OBJECT_REFERENCE_COUNT,
            &[0, 2, 0, 0, 0],
        ))
        .message(Message::new(WireType::UNKNOWN, &[]).with_flags(WireFlags::FAIL_IF_UNKNOWN_ALWAYS))
        .build();
    assert_eq!(
        observe(&root, None, 0, 8).unwrap_err(),
        FormatError::UnsupportedMessage(WireType::UNKNOWN.0)
    );
}

#[rstest]
#[case::missing(
    0,
    None,
    Ok(ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::InitialChunk))
)]
#[case::wrong_start(0, Some((1, 16)), Ok(ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::InitialChunk)))]
#[case::short_prefix(0, Some((0, 7)), Ok(ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::InitialChunk)))]
#[case::overflow(u64::MAX - 3, Some((u64::MAX - 3, 3)), Err(FormatError::OffsetOverflow { offset: u64::MAX - 3, length: 4 }))]
fn prefix_locations_require_the_initial_containing_chunk(
    #[case] address: u64,
    #[case] initial: Option<(u64, u64)>,
    #[case] expected: Result<ObjectReferenceCount, FormatError>,
) {
    let bytes = v1::Header::new().message(Message::nil(16)).build();
    let header = ObjectHeader::parse(&bytes, AccessMode::ReadWrite, 0, 8, 8).unwrap();
    let observer = ObjectReferenceCountObserver::new(PhysicalAddress::new(address));
    if let Some((start, length)) = initial {
        let start = PhysicalAddress::new(start);
        observer
            .observe(ObjectHeaderChunk::new(
                FileRange::new(start, length).unwrap(),
                FileRegion::new(FileRange::new(start, 0).unwrap(), &[]).unwrap(),
                None,
            ))
            .unwrap();
    }
    assert_eq!(observer.finish(&header), expected);
}

#[rstest]
#[case::missing_prefix(1, ObjectReferenceCountIssue::MissingPrefixCount)]
#[case::unsupported_version(3, ObjectReferenceCountIssue::HeaderVersion(3))]
fn incomplete_header_facts_do_not_supply_a_count(
    #[case] version: u8,
    #[case] expected: ObjectReferenceCountIssue,
) {
    let bytes = v1::Header::new().message(Message::nil(16)).build();
    let mut header = ObjectHeader::parse(&bytes, AccessMode::ReadWrite, 0, 8, 8).unwrap();
    header.version = version;
    header.reference_count = None;
    let observer = ObjectReferenceCountObserver::new(PhysicalAddress::new(0));
    observer
        .observe(ObjectHeaderChunk::new(
            FileRange::new(PhysicalAddress::new(0), 16).unwrap(),
            FileRegion::new(FileRange::new(PhysicalAddress::new(16), 0).unwrap(), &[]).unwrap(),
            None,
        ))
        .unwrap();
    assert_eq!(
        observer.finish(&header).unwrap(),
        ObjectReferenceCount::Unsupported(expected)
    );
}

fn observe(
    root: &[u8],
    child: Option<&[u8]>,
    bias: u64,
    width: u8,
) -> Result<ObjectReferenceCount, FormatError> {
    let mut image = Image::starting_with(&[]);
    image.place(512, root);
    if let Some(child) = child {
        image.place(1_024, child);
    }
    let bytes = image.build();
    let observer = ObjectReferenceCountObserver::new(PhysicalAddress::new(bias + 512));
    let mut error = None;
    let header = ObjectHeader::parse_from_source_with_chunk_observer(
        bytes.as_slice(),
        AccessMode::ReadWrite,
        PhysicalAddress::new(512),
        FormatWidths::from_sizes(width, width).unwrap(),
        BaseAddress::new(512),
        MessageFilter::Only(&mut |kind, _| {
            observer.observe_message_type(kind);
            false
        }),
        &mut |chunk| {
            let shift = |range: FileRange| {
                FileRange::new(range.start().checked_add(bias).unwrap(), range.length()).unwrap()
            };
            let chunk = ObjectHeaderChunk::new(
                shift(chunk.allocation()),
                FileRegion::new(shift(chunk.records().range()), chunk.records().bytes()).unwrap(),
                chunk.v2_layout(),
            );
            if error.is_none() {
                error = observer.observe(chunk).err();
            }
        },
    )?;
    if let Some(error) = error {
        return Err(error);
    }
    observer.finish(&header)
}

fn assert_field(
    field: ObjectReferenceCountField,
    count: u32,
    start: u64,
    chunk_start: u64,
    chunk_length: u64,
    flags: Option<MessageFlags>,
) {
    assert_eq!(field.value(), count);
    assert_eq!(
        field.range(),
        FileRange::new(PhysicalAddress::new(start), 4).unwrap()
    );
    assert_eq!(
        field.chunk(),
        FileRange::new(PhysicalAddress::new(chunk_start), chunk_length).unwrap()
    );
    assert!(field.chunk().contains(field.range()));
    assert_eq!(field.flags(), flags);
}
