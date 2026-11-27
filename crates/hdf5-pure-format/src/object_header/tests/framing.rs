use alloc::vec::Vec;

use rstest::rstest;
use test_util::checksum::CHECKSUM;
use test_util::object_header::v1 as v1_bytes;
use test_util::object_header::v2 as v2_bytes;
use test_util::object_header::{Message, MessageType as RecordType};

use super::support;
use super::support::{Driver, LENGTH_SIZE, OFFSET_SIZE};
use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;
use crate::object_header::{MessageFilter, ObjectHeader};

#[test]
fn parse_v1_zero_messages() {
    let header = support::parse(&v1_bytes::Header::new().build()).unwrap();

    assert_eq!(header.version, 1);
    assert!(header.messages.is_empty());
    assert_eq!(header.reference_count, Some(1));
    assert_eq!(header.flags, 0);
}

#[test]
fn parse_v1_two_messages() {
    const DATASPACE: [u8; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
    const DATA_LAYOUT: [u8; 8] = [5, 6, 0, 0, 0, 0, 0, 0];

    let data = v1_bytes::Header::new()
        .message(Message::new(RecordType::DATASPACE, &DATASPACE))
        .message(Message::new(RecordType::DATA_LAYOUT, &DATA_LAYOUT))
        .build();
    let header = support::parse(&data).unwrap();

    assert_eq!(header.messages.len(), 2);
    support::assert_message(&header.messages[0], MessageType::DATASPACE, &DATASPACE);
    support::assert_message(&header.messages[1], MessageType::DATA_LAYOUT, &DATA_LAYOUT);
}

#[rstest]
#[case::missing_padding(2, 12)]
#[case::partial_padding(0, 15)]
#[case::complete_padding(255, 16)]
fn invalid_version_1_header_versions_are_rejected_before_padding(
    #[case] version: u8,
    #[case] length: usize,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let mut data = v1_bytes::Header::new().build();
    data[0] = version;
    data.truncate(length);
    assert_eq!(
        driver.parse(&data, MessageFilter::All).unwrap_err(),
        FormatError::InvalidObjectHeaderVersion(version)
    );
}

#[rstest]
#[case::incomplete_prefix(11, 12)]
#[case::missing_padding(12, 16)]
#[case::partial_padding(15, 16)]
fn truncated_version_1_header_prefixes_preserve_source_diagnostics(
    #[case] length: usize,
    #[case] expected: usize,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let mut data = v1_bytes::Header::new().build();
    data.truncate(length);
    let error = match driver {
        Driver::Buffered => FormatError::UnexpectedEof {
            expected,
            available: length,
        },
        Driver::Source => FormatError::UnexpectedSourceEof {
            expected: expected as u64,
            available: length as u64,
        },
    };
    assert_eq!(driver.parse(&data, MessageFilter::All).unwrap_err(), error);
}

#[test]
fn version_1_header_padding_rejects_an_overflowing_source_address() {
    let data = v1_bytes::Header::new().build();
    let address = u64::MAX - 12;
    let source = ShiftedSource {
        bytes: &data[..12],
        address,
    };
    let error = ObjectHeader::parse_from_source(
        &source,
        AccessMode::ReadOnly,
        address,
        OFFSET_SIZE,
        LENGTH_SIZE,
        BaseAddress::ZERO,
    )
    .unwrap_err();
    assert_eq!(
        error,
        FormatError::OffsetOverflow {
            offset: address,
            length: 16
        }
    );
}

#[test]
fn parse_v2_no_timestamps_one_message() {
    let data = v2_bytes::Header::new()
        .message(Message::new(RecordType::DATASPACE, &[10, 20]))
        .build();
    let header = support::parse(&data).unwrap();

    assert_eq!(header.version, 2);
    assert_eq!(header.flags, 0);
    assert_eq!(header.messages.len(), 1);
    support::assert_message(&header.messages[0], MessageType::DATASPACE, &[10, 20]);
    assert!(header.access_time.is_none());
}

#[test]
fn parse_v2_with_timestamps() {
    let data = v2_bytes::Header::new()
        .timestamps(v2_bytes::Timestamps {
            access: 100,
            modification: 200,
            change: 300,
            birth: 400,
        })
        .message(Message::new(RecordType::DATASPACE, &[1]))
        .build();
    let header = support::parse(&data).unwrap();

    assert_eq!(header.access_time, Some(100));
    assert_eq!(header.modification_time, Some(200));
    assert_eq!(header.change_time, Some(300));
    assert_eq!(header.birth_time, Some(400));
    assert_eq!(header.messages.len(), 1);
    assert!(header.messages[0].creation_order.is_none());
}

#[test]
fn parse_v2_creation_order() {
    let data = v2_bytes::Header::new()
        .flags(v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER)
        .timestamps(v2_bytes::Timestamps::default())
        .message(Message::new(RecordType::DATATYPE, &[9]))
        .message(Message::new(RecordType::FILL_VALUE, &[8]))
        .build();
    let header = support::parse(&data).unwrap();

    assert_eq!(header.messages.len(), 2);
    assert!(header.messages[0].creation_order.is_some());
    assert!(header.messages[1].creation_order.is_some());
    assert_eq!(header.access_time, Some(0));
}

#[test]
fn parse_v2_verifies_checksums_only_with_the_checksum_feature() {
    let mut data = v2_bytes::Header::new()
        .message(Message::new(RecordType::DATASPACE, &[1, 2, 3]))
        .build();

    let header = support::parse(&data).unwrap();
    let [message] = header.messages.as_slice() else {
        panic!("expected one message, got {:?}", header.messages);
    };
    assert_eq!(message.msg_type, MessageType::DATASPACE);
    assert_eq!(message.size, 3);
    assert_eq!(message.flags, MessageFlags::NONE);
    assert_eq!(message.creation_order, None);
    assert_eq!(message.data, [1, 2, 3]);

    *data.last_mut().unwrap() ^= 0xFF;
    if cfg!(feature = "checksum") {
        let (body, stored) = data.split_last_chunk::<CHECKSUM>().unwrap();
        assert_eq!(
            support::parse(&data).unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: u32::from_le_bytes(*stored),
                computed: test_util::checksum::lookup3(body),
            }
        );
    } else {
        support::assert_same_header(&support::parse(&data).unwrap(), &header);
    }
}

#[rstest]
#[case::one_byte(0x00)]
#[case::two_bytes(0x01)]
#[case::four_bytes(0x02)]
fn parse_v2_chunk_size_widths(#[case] flags: u8) {
    let data = v2_bytes::Header::new()
        .flags(v2_bytes::HeaderFlags(flags))
        .message(Message::new(RecordType::DATASPACE, &[1]))
        .build();

    assert_eq!(support::parse(&data).unwrap().messages.len(), 1);
}

#[rstest]
#[case::version_1(&[1, 0], 4)]
#[case::version_2(&[b'O', b'H', b'D', b'R', 2], 6)]
fn truncated_headers_are_rejected(#[case] data: &[u8], #[case] expected: usize) {
    assert_eq!(
        support::parse(data).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available: data.len(),
        }
    );
}

#[test]
fn version_1_chunk0_message_with_unaligned_size_is_rejected() {
    let data = v1_bytes::Header::new()
        .message(Message::new(RecordType::DATATYPE, &[0xAB; 7]))
        .build();

    support::assert_invalid_v1_message_size_all_parse_paths(&data, 7);
}

struct ShiftedSource<'a> {
    bytes: &'a [u8],
    address: u64,
}

impl MetadataSource for ShiftedSource<'_> {
    type Bytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
        self.bytes
            .read_metadata(offset.checked_sub(self.address).unwrap(), len)
    }

    fn len(&self) -> u64 {
        self.address.checked_add(self.bytes.len() as u64).unwrap()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        buf.copy_from_slice(self.read_metadata(offset, buf.len())?);
        Ok(())
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.read_metadata(offset, len)?.to_vec())
    }
}
