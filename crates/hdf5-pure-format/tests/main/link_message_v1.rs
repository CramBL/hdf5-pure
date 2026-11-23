use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{CharacterSet, FormatError, OffsetWidth, StoredAddress};
use hdf5_pure_format::__private::{
    LinkMessage, LinkMessageV1, LinkMessageView, LinkName, LinkProjectionError, LinkStorageError,
    LinkTarget, LinkTargetV1, LinkTargetView, LinkWireError, UintWidth, UserLinkKind,
    UserLinkValue,
};
use rstest::rstest;
use test_util::link_message::{HardLink, SymbolicLink, UserDefinedLink};
use test_util::widths::Widths;

#[rstest]
#[case::two(OffsetWidth::Two, Widths::new(2, 8), 0x1234)]
#[case::four(OffsetWidth::Four, Widths::FOUR, 0x1234_5678)]
#[case::eight(OffsetWidth::Eight, Widths::EIGHT, 0x1234_5678_9abc_def0)]
fn borrowed_hard_link_preserves_raw_fields_and_the_target_range(
    #[case] width: OffsetWidth,
    #[case] fixture_widths: Widths,
    #[case] address: u64,
) {
    let mut bytes = HardLink::from_bytes(b"r\xff", address)
        .name_size_width(2)
        .creation_order(23)
        .character_set(test_util::link_message::CharacterSet::ASCII)
        .build(fixture_widths);
    let end = bytes.len();
    bytes.extend_from_slice(&[0xab, 0xcd]);

    let view = LinkMessageView::parse_prefix(&bytes, width).unwrap();
    assert_eq!(view.name_bytes(), b"r\xff");
    assert!(core::ptr::eq(view.name_bytes(), &bytes[13..15]));
    assert_eq!(view.name_width(), UintWidth::Two);
    assert_eq!(view.creation_order(), Some(23));
    assert_eq!(view.charset(), Some(CharacterSet::Ascii));
    assert_eq!(view.consumed(), end);
    assert_eq!(view.hard_link_address_range(), Some(15..end));
    assert_eq!(
        view.target(),
        LinkTargetView::Hard {
            object_header_address: StoredAddress::new(address),
            explicit_type: false,
        }
    );
    assert_eq!(
        view.to_text().unwrap(),
        LinkMessage {
            name: "r\u{fffd}".into(),
            link_target: LinkTarget::Hard {
                object_header_address: StoredAddress::new(address),
            },
            creation_order: Some(23),
            charset: CharacterSet::Ascii,
        }
    );
}

#[rstest]
#[case::two(OffsetWidth::Two, Widths::new(2, 8), 9)]
#[case::four(OffsetWidth::Four, Widths::FOUR, 11)]
#[case::eight(OffsetWidth::Eight, Widths::EIGHT, 15)]
fn hard_link_decoders_report_a_truncated_address_at_each_width(
    #[case] width: OffsetWidth,
    #[case] fixture_widths: Widths,
    #[case] expected: usize,
) {
    let mut bytes = HardLink::new("name", 42).build(fixture_widths);
    bytes.pop();
    let error = FormatError::UnexpectedEof {
        expected,
        available: expected - 1,
    };

    assert_eq!(
        LinkMessageView::parse_prefix(&bytes, width).unwrap_err(),
        error
    );
    assert_eq!(
        LinkMessage::parse_with_offset_width(&bytes, width).unwrap_err(),
        error
    );
    assert_eq!(
        LinkMessageV1::parse(&bytes, width).unwrap_err(),
        LinkWireError::Storage(LinkStorageError::Format(error))
    );
}

#[test]
fn a_borrowed_soft_link_preserves_bytes_until_text_projection() {
    let bytes = SymbolicLink::soft_bytes("x", b"\xff/target").build();
    let view = LinkMessageView::parse_prefix(&bytes, OffsetWidth::Two).unwrap();
    assert_eq!(view.name_bytes(), b"x");
    assert!(core::ptr::eq(view.name_bytes(), &bytes[4..5]));
    let LinkTargetView::Soft { target_path } = view.target() else {
        panic!("expected a soft link, got {:?}", view.target());
    };
    assert_eq!(target_path, b"\xff/target");
    assert!(core::ptr::eq(target_path, &bytes[7..]));
    let utf8_error = String::from_utf8(b"\xff/target".to_vec())
        .unwrap_err()
        .utf8_error();
    assert_eq!(
        view.to_text().unwrap_err(),
        LinkProjectionError::Format(FormatError::InvalidSoftLinkEncoding { source: utf8_error })
    );
    assert_eq!(
        LinkMessage::parse_with_offset_width(&bytes, OffsetWidth::Two).unwrap_err(),
        FormatError::InvalidSoftLinkEncoding { source: utf8_error }
    );
}

#[test]
fn a_borrowed_external_link_retains_both_paths_without_text_conversion() {
    let bytes = SymbolicLink::external_bytes("x", b"\xff.h5", b"/\xfe").build();
    let view = LinkMessageView::parse_prefix(&bytes, OffsetWidth::Two).unwrap();
    assert_eq!(view.name_bytes(), b"x");
    assert!(core::ptr::eq(view.name_bytes(), &bytes[4..5]));
    let LinkTargetView::External {
        filename,
        object_path,
    } = view.target()
    else {
        panic!("expected an external link, got {:?}", view.target());
    };
    assert_eq!(filename, b"\xff.h5");
    assert_eq!(object_path, b"/\xfe");
    assert!(core::ptr::eq(filename, &bytes[8..12]));
    assert!(core::ptr::eq(object_path, &bytes[13..15]));
    assert_eq!(view.hard_link_address_range(), None);
    assert_eq!(view.consumed(), bytes.len());
    assert_eq!(
        view.to_text().unwrap(),
        LinkMessage {
            name: "x".into(),
            link_target: LinkTarget::external("\u{fffd}.h5".into(), "/\u{fffd}".into()),
            creation_order: None,
            charset: CharacterSet::Ascii,
        },
    );
}

#[test]
fn a_borrowed_user_link_retains_opaque_bytes_and_rejects_text_conversion() {
    let bytes = UserDefinedLink::new(b"x", 65, b"\x00\xff").unwrap().build();
    let view = LinkMessageView::parse_prefix(&bytes, OffsetWidth::Two).unwrap();
    let LinkTargetView::UserDefined { kind, value } = view.target() else {
        panic!("expected a user-defined link, got {:?}", view.target());
    };
    assert_eq!(kind.get(), 65);
    assert_eq!(value, b"\x00\xff");
    assert!(core::ptr::eq(value, &bytes[7..]));
    assert_eq!(
        view.to_text().unwrap_err(),
        LinkProjectionError::UnsupportedKind(kind)
    );
    assert_eq!(
        LinkMessage::parse_with_offset_width(&bytes, OffsetWidth::Two).unwrap_err(),
        FormatError::InvalidLinkType(65)
    );
}

#[rstest]
#[case::empty_external(
    b"\x01\x08\x40\x01x\x00\x00",
    FormatError::InvalidExternalLinkValue,
    LinkStorageError::ExternalValue
)]
#[case::external_version_flags(
    b"\x01\x08\x40\x01x\x01\x00\x10",
    FormatError::InvalidExternalLinkVersionFlags(16),
    LinkStorageError::ExternalVersionFlags(16)
)]
#[case::soft_empty(
    b"\x01\x08\x01\x01x\x00\x00",
    FormatError::InvalidLinkTarget,
    LinkStorageError::InvalidTarget
)]
#[case::undefined_hard(b"\x01\x00\x01x\xff\xff", FormatError::DefinedAddressIsUndefined { address: 0xffff, offset_size: 2 }, LinkStorageError::UndefinedAddress)]
#[case::invalid_name(b"\x01\x00\x01.\xff\xff", FormatError::InvalidLinkNameBytes(b".".to_vec()), LinkStorageError::InvalidName(b".".to_vec()))]
#[case::reserved_flags(
    b"\x01\x3f",
    FormatError::InvalidLinkFlags(0x3f),
    LinkStorageError::Flags(0x3f)
)]
#[case::reserved_kind(
    b"\x01\x08\x3f\x01x",
    FormatError::InvalidLinkType(63),
    LinkStorageError::Format(FormatError::InvalidLinkType(63))
)]
fn full_link_decoders_reject_the_same_malformed_fields(
    #[case] bytes: &[u8],
    #[case] format_error: FormatError,
    #[case] storage_error: LinkStorageError,
) {
    assert_eq!(
        LinkMessageView::parse_prefix(bytes, OffsetWidth::Two).unwrap_err(),
        format_error
    );
    assert_eq!(
        LinkMessage::parse_with_offset_width(bytes, OffsetWidth::Two).unwrap_err(),
        format_error
    );
    assert_eq!(
        LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkWireError::Storage(storage_error)
    );
}

#[rstest]
#[case::version_without_flags(b"\x02", FormatError::UnexpectedEof { expected: 2, available: 1 })]
#[case::version_before_flags(b"\x02\xff", FormatError::InvalidLinkVersion(2))]
#[case::charset_before_name(b"\x01\x10\x02\x01", FormatError::InvalidCharacterSet(2))]
#[case::truncated_name_length_before_invalid_charset(b"\x01\x11\x02\x01", FormatError::UnexpectedEof { expected: 5, available: 4 })]
#[case::truncated_hard_with_invalid_name(b"\x01\x00\x01.\x12", FormatError::UnexpectedEof { expected: 6, available: 5 })]
#[case::truncated_external_with_invalid_flags(b"\x01\x08\x40\x01x\x04\x00\x10", FormatError::UnexpectedEof { expected: 11, available: 8 })]
fn link_decoder_acquisition_and_interpretation_follow_the_selected_fields(
    #[case] bytes: &[u8],
    #[case] error: FormatError,
) {
    assert_eq!(
        LinkMessage::parse_with_offset_width(bytes, OffsetWidth::Two).unwrap_err(),
        error
    );
    assert_eq!(
        LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkWireError::Storage(LinkStorageError::Format(error))
    );
}

#[test]
fn link_prefix_inspection_and_other_names_leave_the_target_unread() {
    let bytes = b"\x01\x08\x40\x01x\x00\x00";
    assert!(core::ptr::eq(
        LinkMessage::name_bytes(&bytes[..5]).unwrap(),
        &bytes[4..5]
    ));
    assert_eq!(
        LinkMessage::target_if_named(bytes, OffsetWidth::Two, "other").unwrap(),
        None
    );
    assert_eq!(
        LinkMessage::target_if_named(bytes, OffsetWidth::Two, "x").unwrap_err(),
        FormatError::InvalidExternalLinkValue
    );
    assert_eq!(
        LinkMessage::hard_link_address_if_named(bytes, 2, "x").unwrap(),
        None
    );
    assert_eq!(
        LinkMessage::discover_hard_link(bytes, OffsetWidth::Two).unwrap(),
        None
    );
}

#[test]
fn hard_link_discovery_preserves_an_invalid_name_for_the_caller() {
    let mut bytes = HardLink::new("x", 42).build(Widths::EIGHT);
    bytes[3] = b'.';
    let (name, address) = LinkMessage::discover_hard_link(&bytes, OffsetWidth::Eight)
        .unwrap()
        .unwrap();
    assert_eq!(name, b".");
    assert!(core::ptr::eq(name, &bytes[3..4]));
    assert_eq!(address, StoredAddress::new(42));
    assert_eq!(
        LinkMessageView::parse_prefix(&bytes, OffsetWidth::Eight).unwrap_err(),
        FormatError::InvalidLinkNameBytes(b".".to_vec()),
    );
}

#[rstest]
#[case::hard_one(b"\x01\x00\x01x\x34\x12", UintWidth::One, hard(false))]
#[case::hard_two(b"\x01\x01\x01\x00x\x34\x12", UintWidth::Two, hard(false))]
#[case::hard_four(b"\x01\x02\x01\x00\x00\x00x\x34\x12", UintWidth::Four, hard(false))]
#[case::hard_eight(
    b"\x01\x03\x01\x00\x00\x00\x00\x00\x00\x00x\x34\x12",
    UintWidth::Eight,
    hard(false)
)]
#[case::soft_one(b"\x01\x08\x01\x01x\x01\x00\xff", UintWidth::One, soft())]
#[case::soft_two(b"\x01\x09\x01\x01\x00x\x01\x00\xff", UintWidth::Two, soft())]
#[case::soft_four(b"\x01\x0a\x01\x01\x00\x00\x00x\x01\x00\xff", UintWidth::Four, soft())]
#[case::soft_eight(
    b"\x01\x0b\x01\x01\x00\x00\x00\x00\x00\x00\x00x\x01\x00\xff",
    UintWidth::Eight,
    soft()
)]
#[case::external_one(
    b"\x01\x08\x40\x01x\x05\x00\x00\xff\x00/\x00",
    UintWidth::One,
    external()
)]
#[case::external_two(
    b"\x01\x09\x40\x01\x00x\x05\x00\x00\xff\x00/\x00",
    UintWidth::Two,
    external()
)]
#[case::external_four(
    b"\x01\x0a\x40\x01\x00\x00\x00x\x05\x00\x00\xff\x00/\x00",
    UintWidth::Four,
    external()
)]
#[case::external_eight(
    b"\x01\x0b\x40\x01\x00\x00\x00\x00\x00\x00\x00x\x05\x00\x00\xff\x00/\x00",
    UintWidth::Eight,
    external()
)]
#[case::user_one(
    b"\x01\x08\x41\x01x\x02\x00\x00\xff",
    UintWidth::One,
    user(65, b"\x00\xff")
)]
#[case::user_two(
    b"\x01\x09\x41\x01\x00x\x02\x00\x00\xff",
    UintWidth::Two,
    user(65, b"\x00\xff")
)]
#[case::user_four(
    b"\x01\x0a\x41\x01\x00\x00\x00x\x02\x00\x00\xff",
    UintWidth::Four,
    user(65, b"\x00\xff")
)]
#[case::user_eight(
    b"\x01\x0b\x41\x01\x00\x00\x00\x00\x00\x00\x00x\x02\x00\x00\xff",
    UintWidth::Eight,
    user(65, b"\x00\xff")
)]
#[case::last_user_empty(b"\x01\x08\xff\x01x\x00\x00", UintWidth::One, user(255, b""))]
fn wire_links_preserve_name_width_and_target(
    #[case] bytes: &[u8],
    #[case] width: UintWidth,
    #[case] target: LinkTargetV1,
) {
    let link = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    assert_eq!(link.name().as_bytes(), b"x");
    assert_eq!(link.name().width(), width);
    assert_eq!(link.target(), &target);
    assert_eq!(link.charset(), None);
    assert_eq!(link.creation_order(), None);
    assert_eq!(link.encoded_len(OffsetWidth::Two).unwrap(), bytes.len());
    assert_eq!(link.serialize(OffsetWidth::Two).unwrap(), bytes);
    let constructed = LinkMessageV1::new(
        LinkName::with_width(b"x".to_vec(), width).unwrap(),
        target,
        None,
        None,
    )
    .unwrap();
    assert_eq!(constructed, link);
    assert_eq!(constructed.serialize(OffsetWidth::Two).unwrap(), bytes);
}

#[rstest]
#[case::explicit_ascii(b"\x01\x18\x00\x00\x01\xff\x34\x12", None, Some(CharacterSet::Ascii))]
#[case::declared_utf8(b"\x01\x10\x01\x01\xff\x34\x12", None, Some(CharacterSet::Utf8))]
#[case::creation_order(
    b"\x01\x1c\x00\xff\xff\xff\xff\xff\xff\xff\xff\x01\x01\xff\x34\x12",
    Some(u64::MAX),
    Some(CharacterSet::Utf8)
)]
fn wire_links_preserve_raw_names_and_optional_fields(
    #[case] bytes: &[u8],
    #[case] creation_order: Option<u64>,
    #[case] charset: Option<CharacterSet>,
) {
    let link = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    assert_eq!(link.name().as_bytes(), b"\xff");
    assert_eq!(link.charset(), charset);
    assert_eq!(link.creation_order(), creation_order);
    assert_eq!(link.serialize(OffsetWidth::Two).unwrap(), bytes);
}

#[rstest]
#[case::empty(b"\x01\x08\x41\x00\x00\x00", b"")]
#[case::dot(b"\x01\x08\x41\x01.\x00\x00", b".")]
#[case::slash(b"\x01\x08\x41\x03a/b\x00\x00", b"a/b")]
#[case::nul(b"\x01\x08\x41\x03a\x00b\x00\x00", b"a\x00b")]
fn user_link_parsing_rejects_invalid_name_components(#[case] bytes: &[u8], #[case] name: &[u8]) {
    assert_eq!(
        LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkWireError::Storage(LinkStorageError::InvalidName(name.to_vec()))
    );
}

#[rstest]
#[case::reserved_two(b"\x01\x08\x02\x01x", FormatError::InvalidLinkType(2))]
#[case::reserved_sixty_three(b"\x01\x08\x3f\x01x", FormatError::InvalidLinkType(63))]
#[case::missing_length(b"\x01\x08\x41\x01x", FormatError::UnexpectedEof { expected: 7, available: 5 })]
#[case::partial_length(b"\x01\x08\xff\x01x\x00", FormatError::UnexpectedEof { expected: 7, available: 6 })]
#[case::truncated_value(b"\x01\x08\xff\x01x\x02\x00\x00", FormatError::UnexpectedEof { expected: 9, available: 8 })]
#[case::charset(b"\x01\x10\x02\x01x\x34\x12", FormatError::InvalidCharacterSet(2))]
#[case::version(b"\x02\x00", FormatError::InvalidLinkVersion(2))]
fn wire_links_report_decoding_errors(#[case] bytes: &[u8], #[case] error: FormatError) {
    assert_eq!(
        LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkWireError::Storage(LinkStorageError::Format(error))
    );
}

#[rstest]
#[case::reserved_flags(b"\x01\x20\x01x\x34\x12", LinkStorageError::Flags(0x20))]
#[case::undefined_address(b"\x01\x00\x01x\xff\xff", LinkStorageError::UndefinedAddress)]
#[case::empty_soft(b"\x01\x08\x01\x01x\x00\x00", LinkStorageError::InvalidTarget)]
#[case::nul_soft(b"\x01\x08\x01\x01x\x01\x00\x00", LinkStorageError::InvalidTarget)]
#[case::external_flags(
    b"\x01\x08\x40\x01x\x01\x00\x01",
    LinkStorageError::ExternalVersionFlags(1)
)]
#[case::external_framing(b"\x01\x08\x40\x01x\x03\x00\x00ab", LinkStorageError::ExternalValue)]
#[case::external_empty_path(
    b"\x01\x08\x40\x01x\x04\x00\x00a\x00\x00",
    LinkStorageError::InvalidTarget
)]
fn wire_links_reject_malformed_fields(#[case] bytes: &[u8], #[case] error: LinkStorageError) {
    assert_eq!(
        LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkWireError::Storage(error)
    );
}

#[rstest]
#[case::hard(b"\x01\x01\x01\x00x\x34\x12\x00\x00", 7)]
#[case::user(b"\x01\x08\xff\x01x\x00\x00\x00\x00", 7)]
fn prefix_parsing_reports_extent_without_object_header_padding(
    #[case] bytes: &[u8],
    #[case] len: usize,
) {
    let (link, consumed) = LinkMessageV1::parse_prefix(bytes, OffsetWidth::Two).unwrap();
    assert_eq!(consumed, len);
    assert_eq!(link.serialize(OffsetWidth::Two).unwrap(), &bytes[..len]);
    assert_eq!(
        LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkWireError::Storage(LinkStorageError::TrailingBytes)
    );
}

#[rstest]
#[case::external_kind(64)]
#[case::hard_kind(0)]
#[case::reserved(63)]
fn user_link_kinds_reject_other_classes(#[case] kind: u8) {
    assert_eq!(
        UserLinkKind::new(kind).unwrap_err(),
        FormatError::InvalidLinkType(kind)
    );
}

#[rstest]
#[case::empty(0)]
#[case::maximum(65_535)]
fn user_values_fit_their_own_length_field(#[case] len: usize) {
    let value = UserLinkValue::new(vec![0xff; len]).unwrap();
    assert_eq!(value.as_bytes(), vec![0xff; len]);
    let link = LinkMessageV1::new(
        LinkName::new(b"x".to_vec()).unwrap(),
        LinkTargetV1::UserDefined {
            kind: UserLinkKind::new(255).unwrap(),
            value,
        },
        None,
        None,
    )
    .unwrap();
    let bytes = link.serialize(OffsetWidth::Two).unwrap();
    assert_eq!(bytes.len(), 7 + len);
    assert_eq!(&bytes[..5], b"\x01\x08\xff\x01x");
    assert_eq!(&bytes[5..7], u16::try_from(len).unwrap().to_le_bytes());
    assert_eq!(&bytes[7..], vec![0xff; len]);
    assert_eq!(
        LinkMessageV1::parse(&bytes, OffsetWidth::Two).unwrap(),
        link
    );
}

#[test]
fn user_values_reject_lengths_above_the_field_limit() {
    assert_eq!(
        UserLinkValue::new(vec![0; 65_536]).unwrap_err(),
        LinkStorageError::TargetLength(65_536)
    );
}

#[rstest]
#[case::one_byte_boundary(255, UintWidth::One)]
#[case::two_byte_boundary(65_535, UintWidth::Two)]
fn explicit_name_widths_accept_their_maximum(#[case] len: usize, #[case] width: UintWidth) {
    let name = LinkName::with_width(vec![b'x'; len], width).unwrap();
    assert_eq!(name.as_bytes(), vec![b'x'; len]);
    assert_eq!(name.width(), width);
}

#[rstest]
#[case::one_byte_overflow(256, UintWidth::One)]
#[case::two_byte_overflow(65_536, UintWidth::Two)]
fn explicit_name_widths_reject_lengths_that_do_not_fit(
    #[case] len: usize,
    #[case] width: UintWidth,
) {
    assert_eq!(
        LinkName::with_width(vec![b'x'; len], width).unwrap_err(),
        LinkWireError::NameWidth { length: len, width }
    );
}

#[test]
fn hard_target_width_errors_are_preserved_by_length_and_encoding() {
    let link = LinkMessageV1::new(
        LinkName::new(b"x".to_vec()).unwrap(),
        LinkTargetV1::Hard {
            object_header_address: StoredAddress::new(65_536),
            explicit_type: false,
        },
        None,
        None,
    )
    .unwrap();
    let expected = LinkWireError::Storage(LinkStorageError::Format(FormatError::AddressTooLarge {
        address: 65_536,
        offset_size: 2,
    }));
    assert_eq!(link.encoded_len(OffsetWidth::Two).unwrap_err(), expected);
    assert_eq!(link.serialize(OffsetWidth::Two).unwrap_err(), expected);
}

#[rstest]
#[case::hard(
    b"\x01\x1b\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\x34\x12",
    hard(true)
)]
#[case::soft(
    b"\x01\x1b\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\x01\x00\xff",
    soft()
)]
#[case::external(
    b"\x01\x1b\x40\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\x05\x00\x00\xff\x00/\x00",
    external()
)]
#[case::user(
    b"\x01\x1b\x41\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\x02\x00\x00\xff",
    user(65, b"\x00\xff")
)]
fn wider_fields_preserve_explicit_ascii_and_binary_names(
    #[case] bytes: &[u8],
    #[case] target: LinkTargetV1,
) {
    let link = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    assert_eq!(link.target(), &target);
    assert_eq!(link.name().width(), UintWidth::Eight);
    assert_eq!(link.name().as_bytes(), b"\xff");
    assert_eq!(link.charset(), Some(CharacterSet::Ascii));
    assert_eq!(link.serialize(OffsetWidth::Two).unwrap(), bytes);
}

#[rstest]
#[case::two(b"\x01\x00\x01x\x34\x12", OffsetWidth::Two)]
#[case::four(b"\x01\x00\x01x\x34\x12\x00\x00", OffsetWidth::Four)]
#[case::eight(b"\x01\x00\x01x\x34\x12\x00\x00\x00\x00\x00\x00", OffsetWidth::Eight)]
fn hard_targets_use_the_supplied_address_width(#[case] bytes: &[u8], #[case] width: OffsetWidth) {
    let link = LinkMessageV1::parse(bytes, width).unwrap();
    assert_eq!(link.target(), &hard(false));
    assert_eq!(link.serialize(width).unwrap(), bytes);
}

fn hard(explicit_type: bool) -> LinkTargetV1 {
    LinkTargetV1::Hard {
        object_header_address: StoredAddress::new(0x1234),
        explicit_type,
    }
}

fn soft() -> LinkTargetV1 {
    LinkTargetV1::Soft {
        target_path: vec![0xff],
    }
}

fn external() -> LinkTargetV1 {
    LinkTargetV1::External {
        filename: vec![0xff],
        object_path: b"/".to_vec(),
    }
}

fn user(kind: u8, value: &[u8]) -> LinkTargetV1 {
    LinkTargetV1::UserDefined {
        kind: UserLinkKind::new(kind).unwrap(),
        value: UserLinkValue::new(value.to_vec()).unwrap(),
    }
}
