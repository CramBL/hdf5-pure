use std::ops::Range;

use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::{BaseAddress, PhysicalAddress};
use hdf5_pure_format::__private::{AccessMode, FormatWidths, MessageFilter, ObjectHeader};
use hdf5_pure_format::{FormatError, MessageType};
use rstest::rstest;
use test_util::checksum;
use test_util::group_info::GroupInfo;
use test_util::image::Image;
use test_util::object_header::{
    Message, MessageFlags as FixtureFlags, MessageType as FixtureType, v1, v2,
};
use test_util::widths::Widths;

#[rstest]
fn version_1_readers_report_an_incomplete_root_extent_before_interpreting_records(
    #[values(Backend::Buffered, Backend::Source, Backend::ObservedSource)] backend: Backend,
) {
    let message = Message::new(FixtureType::UNKNOWN, &[0x42; v1::MESSAGE_ALIGNMENT])
        .with_flags(FixtureFlags::FAIL_IF_UNKNOWN_ALWAYS);
    let declared = v1::RECORD_PREFIX + 2 * v1::MESSAGE_ALIGNMENT;
    let bytes = v1::Header::new()
        .message(message)
        .declared_data_size(u32::try_from(declared).unwrap())
        .build();
    let expected = v1::PREFIX + declared;
    let available = bytes.len();
    let error = match backend {
        Backend::Buffered => FormatError::UnexpectedEof {
            expected,
            available,
        },
        Backend::Source | Backend::ObservedSource => FormatError::UnexpectedSourceEof {
            expected: u64::try_from(expected).unwrap(),
            available: u64::try_from(available).unwrap(),
        },
    };

    assert_eq!(
        backend.parse(&bytes, MessageFilter::All).unwrap_err(),
        error
    );
}

#[rstest]
#[case::unaligned_without_body(1, 0, FormatError::InvalidObjectHeaderMessageSize(1))]
#[case::unaligned_with_body(1, 1, FormatError::InvalidObjectHeaderMessageSize(1))]
#[case::aligned_without_body(8, 0, FormatError::UnexpectedEof {
    expected: 16,
    available: 8,
})]
fn version_1_readers_check_record_alignment_before_acquiring_the_body(
    #[case] declared: u16,
    #[case] present: usize,
    #[case] expected: FormatError,
    #[values(Chunk::Root, Chunk::Continuation)] chunk: Chunk,
    #[values(Backend::Buffered, Backend::Source, Backend::ObservedSource)] backend: Backend,
) {
    let image = chunk.v1_image(&v1_record(declared, present));

    let mut discard = |_: MessageType, _: &[u8]| false;
    for filter in [MessageFilter::All, MessageFilter::Only(&mut discard)] {
        assert_eq!(backend.parse(&image, filter).unwrap_err(), expected);
    }
}

#[rstest]
fn a_version_1_root_reader_ignores_an_incomplete_record_prefix(
    #[values(1, 4, 7)] available: usize,
    #[values(Backend::Buffered, Backend::Source, Backend::ObservedSource)] backend: Backend,
) {
    let record = v1_record(1, 0);
    let image = Chunk::Root.v1_image(&record[..available]);

    let header = backend.parse(&image, MessageFilter::All).unwrap();
    assert_eq!(header.version, 1);
    assert_eq!(header.reference_count, Some(1));
    assert_eq!(header.messages.len(), 0);
}

#[rstest]
fn a_version_1_continuation_reader_reports_an_incomplete_record_prefix(
    #[values(1, 4, 7)] available: usize,
    #[values(Backend::Buffered, Backend::Source, Backend::ObservedSource)] backend: Backend,
) {
    let record = v1_record(1, 0);
    let image = Chunk::Continuation.v1_image(&record[..available]);

    assert_eq!(
        backend.parse(&image, MessageFilter::All).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: v1::RECORD_PREFIX,
            available,
        }
    );
}

fn v1_record(declared: u16, present: usize) -> Vec<u8> {
    let mut record = v1::message_record(&Message::new(
        FixtureType::DATATYPE,
        &[0x43; v1::MESSAGE_ALIGNMENT],
    ));
    record[size_of::<u16>()..2 * size_of::<u16>()].copy_from_slice(&declared.to_le_bytes());
    record.truncate(v1::RECORD_PREFIX + present);
    record
}

#[rstest]
#[case::plain(v2::HeaderFlags::default(), &[0x03, 2, 0, 0, 0x43])]
#[case::tracked(v2::HeaderFlags::TRACKS_CREATION_ORDER, &[0x03, 2, 0, 0, 2, 0, 0x43])]
fn an_object_header_record_body_overrunning_its_chunk_is_rejected(
    #[case] flags: v2::HeaderFlags,
    #[case] record: &[u8],
    #[values(Chunk::Root, Chunk::Continuation)] chunk: Chunk,
    #[values(Backend::Buffered, Backend::Source, Backend::ObservedSource)] backend: Backend,
) {
    let (image, extent) = chunk.image(flags, record);
    let available = extent.len() - checksum::CHECKSUM;
    let expected = FormatError::UnexpectedEof {
        expected: available + 1,
        available,
    };

    assert_eq!(
        backend.parse(&image, MessageFilter::All).unwrap_err(),
        expected
    );

    let mut discarded = Vec::new();
    let filtered = backend.parse(
        &image,
        MessageFilter::Only(&mut |kind, body| {
            discarded.push((kind, body.to_vec()));
            false
        }),
    );
    assert_eq!(filtered.unwrap_err(), expected);
    assert_eq!(discarded, [(MessageType::GROUP_INFO, vec![0, 0])]);
}

#[rstest]
#[case::plain(v2::HeaderFlags::default(), &[0x03, 2, 0], None)]
#[case::tracked(v2::HeaderFlags::TRACKS_CREATION_ORDER, &[0x03, 2, 0, 0, 2], Some(1))]
fn an_object_header_chunk_can_end_with_a_gap_shorter_than_a_record_prefix(
    #[case] flags: v2::HeaderFlags,
    #[case] gap: &[u8],
    #[case] creation_index: Option<u16>,
    #[values(Chunk::Root, Chunk::Continuation)] chunk: Chunk,
    #[values(Backend::Buffered, Backend::Source, Backend::ObservedSource)] backend: Backend,
) {
    let (image, _) = chunk.image(flags, gap);

    let header = backend.parse(&image, MessageFilter::All).unwrap();

    assert_eq!(header.messages.len(), 1);
    let message = &header.messages[0];
    assert_eq!(message.msg_type, MessageType::GROUP_INFO);
    assert_eq!(message.data, [0, 0]);
    assert_eq!(message.creation_order, creation_index);
}

#[rstest]
fn an_object_header_chunk_checksum_mismatch_precedes_framing_when_verification_is_enabled(
    #[values(Chunk::Root, Chunk::Continuation)] chunk: Chunk,
    #[values(Backend::Buffered, Backend::Source)] backend: Backend,
) {
    let (mut image, extent) = chunk.image(v2::HeaderFlags::default(), &[0x03, 2, 0, 0, 0x43]);
    let checksum_at = extent.end - checksum::CHECKSUM;
    let computed = checksum::lookup3(&image[extent.start..checksum_at]);
    let stored = !computed;
    image[checksum_at..extent.end].copy_from_slice(&stored.to_le_bytes());
    let available = checksum_at - extent.start;

    assert_eq!(
        backend.parse(&image, MessageFilter::All).unwrap_err(),
        if cfg!(feature = "checksum") {
            FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            }
        } else {
            FormatError::UnexpectedEof {
                expected: available + 1,
                available,
            }
        },
    );
}

#[rstest]
fn an_object_header_chunk_missing_checksum_bytes_reports_source_eof(
    #[values(Chunk::Root, Chunk::Continuation)] chunk: Chunk,
    #[values(Backend::Buffered, Backend::Source)] backend: Backend,
) {
    let (mut image, extent) = chunk.image(v2::HeaderFlags::default(), &[0x03, 2, 0, 0, 0x43]);
    image.truncate(extent.end - 1);

    assert_eq!(
        backend.parse(&image, MessageFilter::All).unwrap_err(),
        FormatError::UnexpectedSourceEof {
            expected: u64::try_from(extent.end).unwrap(),
            available: u64::try_from(image.len()).unwrap(),
        },
    );
}

#[rstest]
fn an_earlier_record_interpretation_error_precedes_a_later_body_overrun(
    #[values(Chunk::Root, Chunk::Continuation)] chunk: Chunk,
    #[values(Backend::Buffered, Backend::Source, Backend::ObservedSource)] backend: Backend,
) {
    let unknown = Message::new(FixtureType::UNKNOWN, &[0x42])
        .with_flags(FixtureFlags::FAIL_IF_UNKNOWN_ALWAYS);
    let (image, _) =
        chunk.image_with_message(v2::HeaderFlags::default(), unknown, &[0x03, 2, 0, 0, 0x43]);

    let mut discard = |_: MessageType, _: &[u8]| false;
    for filter in [MessageFilter::All, MessageFilter::Only(&mut discard)] {
        assert_eq!(
            backend.parse(&image, filter).unwrap_err(),
            FormatError::UnsupportedMessage(FixtureType::UNKNOWN.0),
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Chunk {
    Continuation,
    Root,
}

impl Chunk {
    fn v1_image(self, record: &[u8]) -> Vec<u8> {
        match self {
            Self::Root => {
                let mut root = v1::Header::new()
                    .message(Message::nil(0))
                    .declared_data_size(u32::try_from(record.len()).unwrap())
                    .build();
                root.truncate(v1::PREFIX);
                root.extend_from_slice(record);
                root
            }
            Self::Continuation => {
                let start = 64;
                let root = v1::Header::new()
                    .continuation(start, record, Widths::EIGHT)
                    .build();
                let mut image = Image::starting_with(&root);
                image.place(start, record);
                image.build()
            }
        }
    }

    fn image(self, flags: v2::HeaderFlags, tail: &[u8]) -> (Vec<u8>, Range<usize>) {
        let group_info = GroupInfo {
            phase_change: None,
            estimates: None,
        };
        let message =
            Message::new(FixtureType::GROUP_INFO, &group_info.build()).with_creation_order(1);
        self.image_with_message(flags, message, tail)
    }

    fn image_with_message(
        self,
        flags: v2::HeaderFlags,
        message: Message,
        tail: &[u8],
    ) -> (Vec<u8>, Range<usize>) {
        match self {
            Self::Continuation => {
                let chunk = v2::continuation_chunk_with_record_tail(&[message], flags, tail);
                let start = 64;
                let root = v2::Header::new()
                    .flags(flags)
                    .continuation(start, &chunk, Widths::EIGHT)
                    .build();
                let mut image = Image::starting_with(&root);
                image.place(start, &chunk);
                (image.build(), start..start + chunk.len())
            }
            Self::Root => {
                let chunk = v2::Header::new()
                    .flags(flags)
                    .message(message)
                    .build_with_record_tail(tail);
                let end = chunk.len();
                (chunk, 0..end)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Backend {
    Buffered,
    ObservedSource,
    Source,
}

impl Backend {
    fn parse(self, image: &[u8], filter: MessageFilter<'_>) -> Result<ObjectHeader, FormatError> {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let access = AccessMode::ReadOnly;
        let base = BaseAddress::ZERO;
        match self {
            Self::Buffered => {
                ObjectHeader::parse_filtered_with_widths(image, access, 0, widths, base, filter)
            }
            Self::ObservedSource => ObjectHeader::parse_from_source_with_chunk_observer(
                image,
                access,
                PhysicalAddress::new(0),
                widths,
                base,
                filter,
                &mut |_| {},
            ),
            Self::Source => ObjectHeader::parse_from_source_filtered_with_widths(
                image, access, 0, widths, base, filter,
            ),
        }
    }
}
