use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;

use super::support;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link::storage::{LinkStorageError, OwnedLink, OwnedLinkTarget};
use crate::link::{LinkMessage, LinkTarget};
use crate::width::OffsetWidth;

#[rstest]
#[case::hard(
    b"\x01\x00\x01l\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    OwnedLinkTarget::Hard { object_header_address: StoredAddress::new(512) },
    12,
)]
#[case::soft(
    b"\x01\x08\x01\x01l\x04\x00../x\x00\x00\x00\x00\x00\x00",
    OwnedLinkTarget::soft(b"../x".to_vec()),
    11,
)]
fn a_padded_parse_reads_the_link_before_the_padding(
    #[case] bytes: &[u8],
    #[case] target: OwnedLinkTarget,
    #[case] len: usize,
) {
    let expected = OwnedLink::new(b"l".to_vec(), CharacterSet::Ascii, target, None).unwrap();
    assert_eq!(
        OwnedLink::parse_padded(bytes, OffsetWidth::Eight),
        Ok((expected, len))
    );
    assert_eq!(
        OwnedLink::parse(bytes, OffsetWidth::Eight),
        Err(LinkStorageError::TrailingBytes)
    );
}

#[rstest]
#[case::hard(
    b"\x01\x00\x01l\x00\x02\x00\x00\x00\x00\x00\x00",
    OwnedLinkTarget::Hard { object_header_address: StoredAddress::new(512) },
    LinkTarget::Hard { object_header_address: StoredAddress::new(512) },
)]
#[case::soft(
    b"\x01\x08\x01\x01l\x04\x00../x",
    OwnedLinkTarget::soft(b"../x".to_vec()),
    LinkTarget::soft("../x".into()),
)]
#[case::external(
    b"\x01\x08\x40\x01l\x09\x00\x00f.h5\x00/x\x00",
    OwnedLinkTarget::external(b"f.h5".to_vec(), b"/x".to_vec()),
    LinkTarget::external("f.h5".into(), "/x".into()),
)]
fn link_targets_preserve_independent_message_bytes(
    #[case] bytes: &[u8],
    #[case] owned_target: OwnedLinkTarget,
    #[case] target: LinkTarget,
) {
    let owned = OwnedLink::new(b"l".to_vec(), CharacterSet::Ascii, owned_target, None).unwrap();
    let message = LinkMessage {
        name: "l".into(),
        link_target: target,
        creation_order: None,
        charset: CharacterSet::Ascii,
    };
    assert_eq!(OwnedLink::parse(bytes, OffsetWidth::Eight).unwrap(), owned);
    assert_eq!(LinkMessage::parse(bytes, 8).unwrap(), message);
    assert_eq!(owned.encoded_len(OffsetWidth::Eight).unwrap(), bytes.len());
    assert_eq!(owned.clone().serialize(OffsetWidth::Eight).unwrap(), bytes);
    assert_eq!(
        message.clone().serialize(OffsetWidth::Eight).unwrap(),
        bytes
    );
}

#[rstest]
#[case::hard(OwnedLinkTarget::Hard { object_header_address: StoredAddress::new(512) })]
#[case::soft(OwnedLinkTarget::soft(b"../target".to_vec()))]
#[case::external(OwnedLinkTarget::external(b"other.h5".to_vec(), b"/target".to_vec()))]
#[case::raw_soft(OwnedLinkTarget::soft(vec![0xfe]))]
#[case::raw_external(OwnedLinkTarget::external(vec![0xfe], vec![b'/', 0xff]))]
fn owned_links_round_trip_raw_names_targets_and_creation_indexes(
    #[case] target: OwnedLinkTarget,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    #[values(CharacterSet::Ascii, CharacterSet::Utf8)] charset: CharacterSet,
    #[values(None, Some(19))] creation_order: Option<u64>,
) {
    let link = OwnedLink::new(
        vec![0xff, b'x'],
        charset.clone(),
        target.clone(),
        creation_order,
    )
    .unwrap();
    let encoded = link.serialize(width).unwrap();
    assert_eq!(encoded.len(), link.encoded_len(width).unwrap());
    let parsed = OwnedLink::parse(&encoded, width).unwrap();
    assert_eq!(parsed, link);
    assert_eq!(parsed.name(), &[0xff, b'x']);
    assert_eq!(parsed.charset(), charset);
    assert_eq!(parsed.target(), &target);
    assert_eq!(parsed.creation_order(), creation_order);
}

#[rstest]
#[case::hard(LinkTarget::Hard { object_header_address: StoredAddress::new(512) })]
#[case::soft(LinkTarget::soft("../软".into()))]
#[case::external(LinkTarget::external("文件.h5".into(), "/target".into()))]
fn owned_links_preserve_messages_written_by_the_existing_serializer(#[case] target: LinkTarget) {
    let original = LinkMessage {
        name: "链".into(),
        link_target: target,
        creation_order: Some(7),
        charset: CharacterSet::Utf8,
    };
    let bytes = original.serialize(OffsetWidth::Eight).unwrap();
    let link = OwnedLink::parse(&bytes, OffsetWidth::Eight).unwrap();
    assert_eq!(link.serialize(OffsetWidth::Eight).unwrap(), bytes);
    assert_eq!(
        LinkMessage::parse_with_offset_width(&bytes, OffsetWidth::Eight).unwrap(),
        original
    );
}

#[rstest]
#[case::one_byte(255)]
#[case::two_bytes(256)]
#[case::two_bytes_max(65_535)]
#[case::four_bytes(65_536)]
fn name_length_field_boundaries_preserve_the_full_name(#[case] length: usize) {
    let link = support::hard_link(vec![b'x'; length], None);
    let bytes = link.serialize(OffsetWidth::Eight).unwrap();
    assert_eq!(bytes.len(), link.encoded_len(OffsetWidth::Eight).unwrap());
    assert_eq!(OwnedLink::parse(&bytes, OffsetWidth::Eight).unwrap(), link);
}

#[rstest]
#[case::empty(vec![])]
#[case::null(vec![0])]
#[case::slash(b"a/b".to_vec())]
#[case::current(b".".to_vec())]
fn invalid_component_names_are_rejected(#[case] name: Vec<u8>) {
    assert_eq!(
        OwnedLink::new(
            name.clone(),
            CharacterSet::Ascii,
            OwnedLinkTarget::soft(b"target".to_vec()),
            None
        )
        .unwrap_err(),
        LinkStorageError::InvalidName(name)
    );
}

#[rstest]
#[case::empty(vec![], LinkStorageError::InvalidTarget)]
#[case::null(vec![b'a', 0], LinkStorageError::InvalidTarget)]
#[case::long(vec![b'x'; 65_536], LinkStorageError::TargetLength(65_536))]
fn invalid_target_paths_are_rejected(
    #[case] target_path: Vec<u8>,
    #[case] error: LinkStorageError,
) {
    assert_eq!(
        OwnedLink::new(
            b"a".to_vec(),
            CharacterSet::Ascii,
            OwnedLinkTarget::soft(target_path),
            None
        )
        .unwrap_err(),
        error
    );
}

#[rstest]
#[case::soft(OwnedLinkTarget::soft(vec![b'x'; 65_535]))]
#[case::external(OwnedLinkTarget::external(vec![b'x'; 65_531], vec![b'/']))]
fn the_maximum_target_value_length_round_trips(#[case] target: OwnedLinkTarget) {
    let link = OwnedLink::new(b"a".to_vec(), CharacterSet::Ascii, target, None).unwrap();
    let bytes = link.serialize(OffsetWidth::Eight).unwrap();
    assert_eq!(OwnedLink::parse(&bytes, OffsetWidth::Eight).unwrap(), link);
}

#[rstest]
#[case::empty_filename(vec![], vec![b'/'], LinkStorageError::InvalidTarget)]
#[case::empty_path(vec![b'f'], vec![], LinkStorageError::InvalidTarget)]
#[case::embedded_null(vec![b'f', 0], vec![b'/'], LinkStorageError::InvalidTarget)]
#[case::too_long(vec![b'f'; 65_532], vec![b'/'], LinkStorageError::TargetLength(65_536))]
fn external_target_framing_is_bounded(
    #[case] filename: Vec<u8>,
    #[case] object_path: Vec<u8>,
    #[case] error: LinkStorageError,
) {
    assert_eq!(
        OwnedLink::new(
            b"a".to_vec(),
            CharacterSet::Ascii,
            OwnedLinkTarget::external(filename, object_path),
            None
        )
        .unwrap_err(),
        error
    );
}

#[rstest]
#[case::version(vec![2, 0, 1, b'a', 0, 0], LinkStorageError::Format(FormatError::InvalidLinkVersion(2)))]
#[case::reserved_flags(vec![1, 0x20, 1, b'a', 0, 0], LinkStorageError::Flags(0x20))]
#[case::charset(vec![1, 0x10, 2, 1, b'a', 0, 0], LinkStorageError::Format(FormatError::InvalidCharacterSet(2)))]
#[case::missing_user_value_length(vec![1, 0x08, 65, 1, b'a'], LinkStorageError::Format(FormatError::UnexpectedEof { expected: 7, available: 5 }))]
#[case::undefined(vec![1, 0, 1, b'a', 0xff, 0xff], LinkStorageError::UndefinedAddress)]
#[case::trailing(vec![1, 0, 1, b'a', 0, 0, 1], LinkStorageError::TrailingBytes)]
#[case::external_version(vec![1, 0x08, 0x40, 1, b'a', 5, 0, 0x10, b'f', 0, b'/', 0], LinkStorageError::ExternalVersionFlags(0x10))]
#[case::external_flags(vec![1, 0x08, 0x40, 1, b'a', 5, 0, 1, b'f', 0, b'/', 0], LinkStorageError::ExternalVersionFlags(1))]
#[case::external_separator(vec![1, 0x08, 0x40, 1, b'a', 4, 0, 0, b'f', b'/', 0], LinkStorageError::ExternalValue)]
#[case::external_terminator(vec![1, 0x08, 0x40, 1, b'a', 4, 0, 0, b'f', 0, b'/'], LinkStorageError::ExternalValue)]
fn unsupported_or_unreproducible_messages_return_explicit_errors(
    #[case] bytes: Vec<u8>,
    #[case] error: LinkStorageError,
) {
    assert_eq!(
        OwnedLink::parse(&bytes, OffsetWidth::Two).unwrap_err(),
        error
    );
}

#[rstest]
#[case::version(0, 2)]
#[case::flags(1, 2)]
#[case::name_length(2, 3)]
#[case::name(3, 4)]
#[case::target(4, 12)]
#[case::partial_target(11, 12)]
fn truncated_messages_report_the_required_field_extent(
    #[case] length: usize,
    #[case] expected: usize,
) {
    let bytes = support::hard_link(b"a".to_vec(), None)
        .serialize(OffsetWidth::Eight)
        .unwrap();
    assert_eq!(
        OwnedLink::parse(&bytes[..length], OffsetWidth::Eight).unwrap_err(),
        LinkStorageError::Format(FormatError::UnexpectedEof {
            expected,
            available: length
        })
    );
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn undefined_hard_targets_are_classified_at_the_encoded_width(#[case] width: OffsetWidth) {
    let mut bytes = vec![1, 0, 1, b'a'];
    bytes.extend_from_slice(&vec![0xff; usize::from(width.get())]);
    assert_eq!(
        OwnedLink::parse(&bytes, width).unwrap_err(),
        LinkStorageError::UndefinedAddress
    );
}

#[test]
fn hard_target_replacement_preserves_other_link_metadata_and_rejects_narrow_sentinels() {
    let mut link = support::hard_link(vec![0xff], Some(13));
    link.set_hard_target(StoredAddress::new(0xffff), OffsetWidth::Eight)
        .unwrap();
    let saved = link.clone();
    assert_eq!(
        link.set_hard_target(StoredAddress::new(0xffff), OffsetWidth::Two)
            .unwrap_err(),
        LinkStorageError::UndefinedAddress
    );
    assert_eq!(link, saved);
    assert_eq!(
        link.serialize(OffsetWidth::Two).unwrap_err(),
        LinkStorageError::UndefinedAddress
    );
    assert_eq!(link.name(), &[0xff]);
    assert_eq!(link.creation_order(), Some(13));
    assert_eq!(link.charset(), CharacterSet::Ascii);
    assert_eq!(
        link.target(),
        &OwnedLinkTarget::Hard {
            object_header_address: StoredAddress::new(0xffff)
        }
    );
    link.set_hard_target(StoredAddress::new(4096), OffsetWidth::Two)
        .unwrap();
    assert_eq!(
        OwnedLink::parse(&link.serialize(OffsetWidth::Two).unwrap(), OffsetWidth::Two).unwrap(),
        link
    );
}

#[test]
fn target_replacement_and_sizing_reject_oversized_addresses() {
    let mut link = support::hard_link(b"a".to_vec(), None);
    let saved = link.clone();
    let error = LinkStorageError::Format(FormatError::AddressTooLarge {
        address: 0x1_0000,
        offset_size: 2,
    });
    assert_eq!(
        link.set_hard_target(StoredAddress::new(0x1_0000), OffsetWidth::Two)
            .unwrap_err(),
        error
    );
    assert_eq!(link, saved);
    link.set_hard_target(StoredAddress::new(0x1_0000), OffsetWidth::Eight)
        .unwrap();
    assert_eq!(link.encoded_len(OffsetWidth::Two).unwrap_err(), error);
}

#[test]
fn a_symbolic_link_cannot_be_retargeted_as_a_hard_link() {
    let mut link = OwnedLink::new(
        b"a".to_vec(),
        CharacterSet::Ascii,
        OwnedLinkTarget::soft(b"target".to_vec()),
        None,
    )
    .unwrap();
    let original = link.clone();
    assert_eq!(
        link.set_hard_target(StoredAddress::new(512), OffsetWidth::Eight)
            .unwrap_err(),
        LinkStorageError::NotHardLink
    );
    assert_eq!(link, original);
}
