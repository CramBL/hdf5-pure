use rstest::rstest;
use test_util::link_message;
use test_util::widths::Widths;

use super::*;
use crate::address::StoredAddressExt;

#[rstest]
#[case::hard(LinkTarget::Hard { object_header_address: StoredAddress::new(42) })]
#[case::soft(LinkTarget::soft("target".into()))]
#[case::external(LinkTarget::external("file.h5".into(), "/target".into()))]
fn link_names_borrow_prefix_bytes_before_the_target(#[case] link_target: LinkTarget) {
    let data = LinkMessage {
        name: "link".into(),
        link_target,
        creation_order: Some(7),
        charset: CharacterSet::Utf8,
    }
    .serialize(OffsetWidth::Eight)
    .unwrap();
    let prefix = LinkPrefix::parse(&data).unwrap();
    let name = LinkMessage::name_bytes(&data[..prefix.target_start()]).unwrap();
    assert_eq!(name, b"link");
    assert_eq!(name.as_ptr(), prefix.name.as_ptr());
}

#[rstest]
#[case::version(vec![2, 0], FormatError::InvalidLinkVersion(2))]
#[case::truncated(vec![1, 0, 3, b'x'], FormatError::UnexpectedEof { expected: 6, available: 4 })]
#[case::charset(vec![1, 0x10, 2, 1], FormatError::InvalidCharacterSet(2))]
fn borrowed_link_names_reject_malformed_prefixes(
    #[case] data: Vec<u8>,
    #[case] expected: FormatError,
) {
    assert_eq!(LinkMessage::name_bytes(&data).unwrap_err(), expected);
}

#[test]
fn borrowed_link_names_preserve_invalid_utf8() {
    let mut data = build_hard_link("a", 42, 8, None, None, 1);
    data[3] = 0xff;
    assert_eq!(LinkMessage::name_bytes(&data).unwrap(), &[0xff]);
}

fn build_hard_link(
    name: &str,
    addr: u64,
    offset_size: usize,
    creation_order: Option<u64>,
    charset: Option<u8>,
    name_size_width: usize,
) -> Vec<u8> {
    let mut link = link_message::HardLink::new(name, addr).name_size_width(name_size_width);
    if let Some(creation_order) = creation_order {
        link = link.creation_order(creation_order);
    }
    if let Some(charset) = charset {
        link = link.character_set(link_message::CharacterSet(charset));
    }
    link.build(Widths::new(offset_size, offset_size))
}

#[test]
fn hard_link_ascii_no_creation_order() {
    let data = build_hard_link("mydata", 0x1000, 8, None, None, 1);
    let msg = LinkMessage::parse(&data, 8).unwrap();
    assert_eq!(msg.name, "mydata");
    assert_eq!(
        msg.link_target,
        LinkTarget::Hard {
            object_header_address: StoredAddress::new(0x1000)
        }
    );
    assert_eq!(msg.creation_order, None);
    assert_eq!(msg.charset, CharacterSet::Ascii);
}

#[test]
fn hard_link_utf8_with_creation_order() {
    let data = build_hard_link("données", 0x2000, 8, Some(42), Some(1), 1);
    let msg = LinkMessage::parse(&data, 8).unwrap();
    assert_eq!(msg.name, "données");
    assert_eq!(
        msg.link_target,
        LinkTarget::Hard {
            object_header_address: StoredAddress::new(0x2000)
        }
    );
    assert_eq!(msg.creation_order, Some(42));
    assert_eq!(msg.charset, CharacterSet::Utf8);
}

#[test]
fn soft_link() {
    let target = "/group1/dataset";
    let mut data = Vec::new();
    data.push(1); // version
    data.push(0x08); // flags: bit 3 = link type present, name size = 1 byte (bits 0-1 = 0)
    data.push(1); // link type = soft
    data.push(4); // name length = 4
    data.extend_from_slice(b"link");
    data.extend_from_slice(&(target.len() as u16).to_le_bytes());
    data.extend_from_slice(target.as_bytes());

    let msg = LinkMessage::parse(&data, 8).unwrap();
    assert_eq!(msg.name, "link");
    assert_eq!(msg.link_target, LinkTarget::soft(target.to_string()));
}

#[test]
fn a_soft_link_target_with_invalid_utf8_is_rejected() {
    let mut data = LinkMessage {
        name: String::from("link"),
        link_target: LinkTarget::soft(String::from("x")),
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
    .serialize(OffsetWidth::Eight)
    .unwrap();
    *data.last_mut().unwrap() = 0xff;
    let error = LinkMessage::target_if_named(&data, OffsetWidth::Eight, "link").unwrap_err();
    let FormatError::InvalidSoftLinkEncoding { source } = error else {
        panic!("expected InvalidSoftLinkEncoding, got {error:?}");
    };
    assert_eq!(source.valid_up_to(), 0);
    assert_eq!(source.error_len(), Some(1));
    assert_eq!(
        LinkMessage::target_if_named(&data, OffsetWidth::Eight, "other").unwrap(),
        None
    );
}

#[test]
fn name_length_2bytes() {
    let data = build_hard_link("test", 0x500, 8, None, None, 2);
    let msg = LinkMessage::parse(&data, 8).unwrap();
    assert_eq!(msg.name, "test");
}

#[test]
fn name_length_4bytes() {
    let data = build_hard_link("abcd", 0x600, 8, None, None, 4);
    let msg = LinkMessage::parse(&data, 8).unwrap();
    assert_eq!(msg.name, "abcd");
}

#[test]
fn a_named_hard_link_answers_with_its_address() {
    let data = build_hard_link("mydata", 0x1000, 8, None, None, 1);
    assert_eq!(
        LinkMessage::hard_link_address_if_named(&data, 8, "mydata").unwrap(),
        Some(StoredAddress::new(0x1000))
    );
    assert_eq!(
        LinkMessage::hard_link_address_if_named(&data, 8, "other").unwrap(),
        None
    );
}

/// A soft link refers to a path, not an object header, so a lookup passes over it
/// however it is named.
#[test]
fn a_soft_link_of_the_wanted_name_is_not_an_address() {
    let target = "/group1/dataset";
    let mut data = Vec::new();
    data.push(1); // version
    data.push(0x08); // flags: link type present, 1-byte name length
    data.push(1); // link type = soft
    data.push(4); // name length
    data.extend_from_slice(b"link");
    data.extend_from_slice(&(target.len() as u16).to_le_bytes());
    data.extend_from_slice(target.as_bytes());

    assert_eq!(
        LinkMessage::hard_link_address_if_named(&data, 8, "link").unwrap(),
        None
    );
}

/// A name that is not valid UTF-8 reaches a caller as the replacement
/// characters [`LinkMessage::parse`] decodes it to, and that spelling is the
/// only one such a link can be requested by, so it is the one that matches.
#[test]
fn a_name_that_is_not_utf8_matches_the_spelling_a_reader_gets() {
    let mut data = Vec::new();
    data.push(1); // version
    data.push(0x00); // flags: hard link, 1-byte name length
    data.push(2); // name length
    data.extend_from_slice(&[0xFF, 0xFE]); // not UTF-8
    data.extend_from_slice(&0x2000u64.to_le_bytes());

    let decoded = LinkMessage::parse(&data, 8).unwrap().name;
    assert_eq!(
        LinkMessage::hard_link_address_if_named(&data, 8, &decoded).unwrap(),
        Some(StoredAddress::new(0x2000)),
        "a link found by listing must be findable by the name the listing gave"
    );
    assert!(link_is_named(&data, &decoded));
}

#[test]
fn hard_address_lookup_reports_a_user_defined_kind_only_for_a_matching_name() {
    let mut data = Vec::new();
    data.push(1); // version
    data.push(0x08); // flags: link type present, 1-byte name length
    data.push(99); // user-defined link kind
    data.push(1); // name length
    data.push(b'x');

    assert_eq!(
        LinkMessage::hard_link_address_if_named(&data, 8, "x").unwrap_err(),
        FormatError::InvalidLinkType(99)
    );
    assert_eq!(
        LinkMessage::hard_link_address_if_named(&data, 8, "y").unwrap(),
        None
    );
    assert!(!link_is_named(&data, "y"));
}

#[test]
fn an_unreadable_link_is_kept_by_the_filter() {
    assert!(link_is_named(&[2, 0, 0, 0], "anything"), "bad version");
    assert!(link_is_named(&[], "anything"), "empty body");
}

#[test]
fn invalid_version() {
    let data = vec![2, 0, 0, 0]; // version 2
    let err = LinkMessage::parse(&data, 8).unwrap_err();
    assert_eq!(err, FormatError::InvalidLinkVersion(2));
}

#[test]
fn a_soft_link_target_longer_than_its_length_field_is_rejected() {
    let error = LinkMessage {
        name: "link".into(),
        link_target: LinkTarget::soft("a".repeat(usize::from(u16::MAX) + 1)),
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
    .serialize(OffsetWidth::Eight)
    .unwrap_err();
    assert_eq!(
        error,
        FormatError::LengthTooLarge {
            length: 65_536,
            length_size: 2
        }
    );
}

#[rstest]
#[case::a_hard_link(build_hard_link("a", 0x1234, 8, None, None, 1), Ok(Some(4..12)))]
#[case::a_soft_link(soft_link_body(), Ok(None))]
#[case::a_hard_link_cut_short(
    build_hard_link("a", 0x1234, 8, None, None, 1)[..10].to_vec(),
    Err(FormatError::UnexpectedEof { expected: 12, available: 10 })
)]
fn a_hard_link_address_is_located_past_the_name(
    #[case] data: Vec<u8>,
    #[case] expected: Result<Option<Range<usize>>, FormatError>,
) {
    assert_eq!(
        LinkMessage::hard_link_address_range(&data, OffsetWidth::Eight),
        expected
    );
}

fn soft_link_body() -> Vec<u8> {
    LinkMessage {
        name: "link".into(),
        link_target: LinkTarget::soft("/group1".into()),
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
    .serialize(OffsetWidth::Eight)
    .unwrap()
}
