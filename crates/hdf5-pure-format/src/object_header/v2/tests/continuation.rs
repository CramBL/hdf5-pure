use rstest::rstest;

use super::*;
use crate::address::StoredAddressExt;

#[test]
fn a_continuation_message_parses_to_its_address_and_length() {
    let mut body = 0x800u32.to_le_bytes().to_vec();
    body.extend_from_slice(&0x40u64.to_le_bytes());
    assert_eq!(
        ObjectHeaderContinuation::parse(&body, 4, 8),
        Ok(ObjectHeaderContinuation {
            address: StoredAddress::new(0x800),
            length: 0x40,
        })
    );
}

#[rstest]
#[case::a_block_of_messages(b"OCHK\x01\x00\x00\x00cksm", Ok(4..8))]
#[case::an_empty_block(b"OCHKcksm", Ok(4..4))]
#[case::a_block_shorter_than_its_signature_and_checksum(
    b"OCHKcks",
    Err(FormatError::UnexpectedEof { expected: 8, available: 7 })
)]
#[case::another_signature(b"OHDRcksm", Err(FormatError::InvalidObjectHeaderSignature))]
fn a_continuation_block_holds_its_messages_between_signature_and_checksum(
    #[case] block: &[u8],
    #[case] expected: Result<Range<usize>, FormatError>,
) {
    assert_eq!(continuation_block_messages(block), expected);
}
