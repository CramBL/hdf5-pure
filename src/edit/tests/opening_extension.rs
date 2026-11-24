use test_util::file_space_info;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::superblock::v2 as superblock_bytes;
use test_util::widths::Widths;

use crate::Error;
use crate::FormatError;
use crate::edit::OpeningExtension;
use crate::edit::context::FileContext;
use crate::source::BytesSource;
use crate::superblock;

#[test]
fn opening_extension_returns_file_space_info_and_allocation_error_from_version_1_header() {
    let mut body = persistent_file_space_info();
    body.resize(
        body.len().div_ceil(v1::MESSAGE_ALIGNMENT) * v1::MESSAGE_ALIGNMENT,
        0,
    );
    let header = v1::Header::new()
        .message(Message::new(MessageType::FILE_SPACE_INFO, &body))
        .build();
    let facts = read_extension(header).unwrap();
    let (address, info) = facts.info.unwrap();
    assert_eq!(address, EXTENSION_ADDRESS);
    assert!(info.persist);
    assert_eq!(info.threshold, 7);
    let error = facts.allocations.unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}")
    };
    assert_eq!(reason, "an object does not use a version 2 object header");
}

#[test]
fn opening_extension_reports_truncated_record_after_file_space_info() {
    let mut record = v2::message_record(
        &Message::new(MessageType::UNKNOWN, &[0xA5]),
        v2::HeaderFlags::default(),
    );
    v2::set_record_body_size(&mut record, 0, 2);
    let header = v2::Header::new()
        .message(Message::new(
            MessageType::FILE_SPACE_INFO,
            &persistent_file_space_info(),
        ))
        .build_with_record_tail(&record);
    let Err(error) = read_extension(header) else {
        panic!("expected a framing error for the truncated record")
    };
    assert_eq!(
        error,
        FormatError::UnexpectedEof {
            expected: 75,
            available: 74,
        }
    );
}

fn persistent_file_space_info() -> Vec<u8> {
    file_space_info::version_0(
        Widths::EIGHT,
        file_space_info::ALL_PERSIST,
        7,
        &[u64::MAX; file_space_info::VERSION_0_MANAGERS],
    )
}

fn read_extension(header: Vec<u8>) -> Result<OpeningExtension, FormatError> {
    let mut bytes = superblock_bytes::Superblock::new(Widths::EIGHT)
        .extension_address(EXTENSION_ADDRESS)
        .eof_address(EXTENSION_ADDRESS + u64::try_from(header.len()).unwrap())
        .build();
    bytes.resize(usize::try_from(EXTENSION_ADDRESS).unwrap(), 0);
    bytes.extend(header);
    let context = FileContext::new(superblock::read(&bytes[..]).unwrap()).unwrap();
    let source = BytesSource::new(bytes);
    OpeningExtension::read(&source, &context)
}

const EXTENSION_ADDRESS: u64 = 128;
