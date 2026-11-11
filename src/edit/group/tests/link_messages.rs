use hdf5_pure_format::__private::HeaderMessage;
use rstest::rstest;

use crate::edit::group;
use crate::error::{Error, FormatError};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::width::OffsetWidth;

#[rstest]
#[case::one(b"\x01\x00\x01x\x34\x12")]
#[case::two(b"\x01\x01\x01\x00x\x34\x12")]
#[case::four(b"\x01\x02\x01\x00\x00\x00x\x34\x12")]
#[case::eight(b"\x01\x03\x01\x00\x00\x00\x00\x00\x00\x00x\x34\x12")]
fn version_1_padding_preserves_the_original_link_extent(#[case] body: &[u8]) {
    let mut data = body.to_vec();
    data.resize(data.len().next_multiple_of(8), 0xcc);
    let message = HeaderMessage {
        msg_type: MessageType::LINK,
        size: data.len(),
        flags: MessageFlags::NONE,
        creation_order: None,
        data,
    };
    assert_eq!(
        group::unpadded_body(&message, OffsetWidth::Two).unwrap(),
        body
    );
}

#[rstest]
#[case::user_65(b"\x01\x08\x41\x01u\x00\x00", None)]
#[case::user_255_padded(b"\x01\x08\xff\x01u\x00\x00\xcc", None)]
#[case::missing_length(b"\x01\x08\x41\x01u", Some(FormatError::UnexpectedEof { expected: 7, available: 5 }))]
#[case::partial_length(b"\x01\x08\xff\x01u\x00", Some(FormatError::UnexpectedEof { expected: 7, available: 6 }))]
#[case::truncated_value(b"\x01\x08\xff\x01u\x04\x00\x00\xff", Some(FormatError::UnexpectedEof { expected: 11, available: 9 }))]
#[case::reserved_2(b"\x01\x08\x02\x01u", Some(FormatError::InvalidLinkType(2)))]
#[case::reserved_63(b"\x01\x08\x3f\x01u", Some(FormatError::InvalidLinkType(63)))]
fn version_1_link_inspection_decodes_before_projecting(
    #[case] body: &[u8],
    #[case] expected: Option<FormatError>,
) {
    let message = HeaderMessage {
        msg_type: MessageType::LINK,
        size: body.len(),
        flags: MessageFlags::NONE,
        creation_order: None,
        data: body.to_vec(),
    };
    let error = group::unpadded_body(&message, OffsetWidth::Two).unwrap_err();
    match expected {
        Some(expected) => {
            let Error::Format(actual) = error else {
                panic!("expected a format error, got {error:?}");
            };
            assert_eq!(actual, expected);
        }
        None => assert!(
            matches!(
                error,
                Error::EditUnsupported("a group's user-defined links cannot be reproduced")
            ),
            "{error:?}"
        ),
    }
}
