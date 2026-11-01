use rstest::rstest;
use test_util::file_space_info;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::superblock::v2 as superblock_bytes;
use test_util::widths::Widths;

use crate::Error;
use crate::edit::OpeningExtension;
use crate::edit::context::FileContext;
use crate::source::BytesSource;
use crate::superblock;

#[derive(Clone, Copy)]
enum AllocationError {
    LegacyHeader,
    TruncatedRecord,
}

#[rstest]
#[case::legacy_header(AllocationError::LegacyHeader)]
#[case::truncated_later_record(AllocationError::TruncatedRecord)]
fn readable_file_space_info_preserves_extension_allocation_errors(#[case] case: AllocationError) {
    let mut body = file_space_info::version_0(
        Widths::EIGHT,
        file_space_info::ALL_PERSIST,
        7,
        &[u64::MAX; file_space_info::VERSION_0_MANAGERS],
    );
    let header = match case {
        AllocationError::LegacyHeader => {
            body.resize(
                body.len().div_ceil(v1::MESSAGE_ALIGNMENT) * v1::MESSAGE_ALIGNMENT,
                0,
            );
            v1::Header::new()
                .message(Message::new(MessageType::FILE_SPACE_INFO, &body))
                .build()
        }
        AllocationError::TruncatedRecord => {
            let mut header = v2::Header::new()
                .message(Message::new(MessageType::FILE_SPACE_INFO, &body))
                .message(Message::new(MessageType::UNKNOWN, &[0xA5]))
                .build();
            let body_end = header.len() - test_util::checksum::CHECKSUM;
            header[body_end - 4..body_end - 2].copy_from_slice(&2u16.to_le_bytes());
            let len = header.len();
            test_util::checksum::restamp(&mut header, 0, len);
            header
        }
    };
    let extension = 128;
    let mut bytes = superblock_bytes::Superblock::new(Widths::EIGHT)
        .extension_address(extension)
        .eof_address(extension + u64::try_from(header.len()).unwrap())
        .build();
    bytes.resize(usize::try_from(extension).unwrap(), 0);
    bytes.extend(header);
    let context = FileContext::new(superblock::read(&bytes[..]).unwrap()).unwrap();
    let source = BytesSource::new(bytes);
    let facts = OpeningExtension::read(&source, &context).unwrap();
    let (address, info) = facts.info.unwrap();
    assert_eq!(address, extension);
    assert!(info.persist);
    assert_eq!(info.threshold, 7);
    let error = facts.allocations.unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}")
    };
    assert_eq!(
        reason,
        match case {
            AllocationError::LegacyHeader => "an object does not use a version 2 object header",
            AllocationError::TruncatedRecord => "malformed object header message",
        }
    );
}
