use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{CharacterSet, FormatError, OffsetWidth, StoredAddress};
use hdf5_pure_format::__private::{
    LinkMessageV1, LinkName, LinkStorageError, LinkTargetV1, LinkWireError, UintWidth,
    UserLinkKind, UserLinkValue,
};
use rstest::rstest;

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
