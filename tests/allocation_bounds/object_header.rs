use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_format::__private::{AccessMode, FormatWidths, MessageFlags, ObjectHeader};
use hdf5_pure_format::{FormatError, MessageType};
use rstest::rstest;
use test_util::allocation;
use test_util::datatype;
use test_util::object_header::{Message, MessageType as FixtureType, v1};

#[rstest]
#[case::missing(0)]
#[case::short(v1::RECORD_PREFIX + v1::MESSAGE_ALIGNMENT)]
fn version_1_readers_report_root_chunk_eof_without_allocating(#[case] body_len: usize) {
    const DECLARED_SIZE: u32 = 524_280;
    let mut type_bytes = datatype::f64_le();
    type_bytes.resize(type_bytes.len().next_multiple_of(v1::MESSAGE_ALIGNMENT), 0);
    let message = Message::new(FixtureType::DATATYPE, &type_bytes);
    let valid = v1::Header::new()
        .message(message.clone())
        .reference_count(7)
        .build();
    let mut truncated = v1::Header::new()
        .message(message)
        .declared_message_count(u16::MAX)
        .declared_data_size(DECLARED_SIZE)
        .build();
    truncated.truncate(v1::PREFIX + body_len);
    let expected = v1::PREFIX + usize::try_from(DECLARED_SIZE).unwrap();
    let available = truncated.len();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let buffered =
        |bytes: &[u8]| ObjectHeader::parse_with_widths(bytes, AccessMode::ReadOnly, 0, widths);
    let source = |bytes: &[u8]| {
        ObjectHeader::parse_from_source_with_widths(
            bytes,
            AccessMode::ReadOnly,
            0,
            widths,
            BaseAddress::ZERO,
        )
    };

    let (parsed, control) =
        allocation::measure(&format!("v1_root_control_{body_len}"), || source(&valid));
    let header = parsed.unwrap();
    assert!(control.blocks > 0, "{control}");
    assert!(control.peak_bytes >= type_bytes.len() as u64, "{control}");
    assert_eq!(header.version, 1);
    assert_eq!(header.reference_count, Some(7));
    assert_eq!(header.messages.len(), 1);
    let message = &header.messages[0];
    assert_eq!(message.msg_type, MessageType::DATATYPE);
    assert_eq!(message.size, type_bytes.len());
    assert_eq!(message.flags, MessageFlags::new(0));
    assert_eq!(message.creation_order, None);
    assert_eq!(message.data, type_bytes);

    let (parsed, measured) = allocation::measure(&format!("v1_root_buffer_{body_len}"), || {
        buffered(&truncated)
    });
    assert_eq!(
        parsed.unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available,
        }
    );
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.peak_bytes, 0, "{measured}");

    let (parsed, measured) =
        allocation::measure(&format!("v1_root_source_{body_len}"), || source(&truncated));
    assert_eq!(
        parsed.unwrap_err(),
        FormatError::UnexpectedSourceEof {
            expected: expected as u64,
            available: available as u64,
        }
    );
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.peak_bytes, 0, "{measured}");
}
