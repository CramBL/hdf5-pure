use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{CharacterSet, FormatError, OffsetWidth, StoredAddress};
use hdf5_pure_format::__private::{
    LinkMessage, LinkMessageV1, LinkProjectionError, LinkStorageError, LinkTarget, OwnedLink,
    OwnedLinkTarget, UserLinkKind,
};
use rstest::rstest;

#[rstest]
#[case::hard_one(b"\x01\x00\x01x\x34\x12", hard())]
#[case::hard_two(b"\x01\x01\x01\x00x\x34\x12", hard())]
#[case::hard_four(b"\x01\x02\x01\x00\x00\x00x\x34\x12", hard())]
#[case::hard_eight(b"\x01\x03\x01\x00\x00\x00\x00\x00\x00\x00x\x34\x12", hard())]
#[case::soft_one(b"\x01\x08\x01\x01x\x01\x00\xff", OwnedLinkTarget::soft(vec![0xff]))]
#[case::soft_two(b"\x01\x09\x01\x01\x00x\x01\x00\xff", OwnedLinkTarget::soft(vec![0xff]))]
#[case::soft_four(b"\x01\x0a\x01\x01\x00\x00\x00x\x01\x00\xff", OwnedLinkTarget::soft(vec![0xff]))]
#[case::soft_eight(b"\x01\x0b\x01\x01\x00\x00\x00\x00\x00\x00\x00x\x01\x00\xff", OwnedLinkTarget::soft(vec![0xff]))]
#[case::external_one(b"\x01\x08\x40\x01x\x05\x00\x00\xff\x00/\x00", external())]
#[case::external_two(b"\x01\x09\x40\x01\x00x\x05\x00\x00\xff\x00/\x00", external())]
#[case::external_four(b"\x01\x0a\x40\x01\x00\x00\x00x\x05\x00\x00\xff\x00/\x00", external())]
#[case::external_eight(
    b"\x01\x0b\x40\x01\x00\x00\x00\x00\x00\x00\x00x\x05\x00\x00\xff\x00/\x00",
    external()
)]
fn owned_projection_preserves_selected_name_widths_and_targets(
    #[case] bytes: &[u8],
    #[case] target: OwnedLinkTarget,
) {
    let wire = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    let owned = OwnedLink::try_from(wire).unwrap();
    assert_eq!(owned.name(), b"x");
    assert_eq!(owned.charset(), CharacterSet::Ascii);
    assert_eq!(owned.creation_order(), None);
    assert_eq!(owned.target(), &target);
    assert_eq!(owned.encoded_len(OffsetWidth::Two).unwrap(), bytes.len());
    assert_eq!(owned.serialize(OffsetWidth::Two).unwrap(), bytes);
    assert_eq!(OwnedLink::parse(bytes, OffsetWidth::Two).unwrap(), owned);
}

#[rstest]
#[case::hard_ascii(
    b"\x01\x1b\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\x34\x12",
    hard(),
    CharacterSet::Ascii,
    None
)]
#[case::soft_utf8_order(
    b"\x01\x1d\x01\x09\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\xff\x01\x00\xfe",
    OwnedLinkTarget::soft(vec![0xfe]), CharacterSet::Utf8, Some(9)
)]
#[case::external_ascii_order(
    b"\x01\x1e\x40\x09\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xff\x05\x00\x00\xff\x00/\x00",
    external(), CharacterSet::Ascii, Some(9)
)]
fn owned_projection_preserves_raw_names_and_optional_fields(
    #[case] bytes: &[u8],
    #[case] target: OwnedLinkTarget,
    #[case] charset: CharacterSet,
    #[case] creation_order: Option<u64>,
) {
    let wire = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    let owned = OwnedLink::try_from(wire).unwrap();
    assert_eq!(owned.name(), b"\xff");
    assert_eq!(owned.charset(), charset);
    assert_eq!(owned.creation_order(), creation_order);
    assert_eq!(owned.target(), &target);
    assert_eq!(owned.encoded_len(OffsetWidth::Two).unwrap(), bytes.len());
    assert_eq!(owned.serialize(OffsetWidth::Two).unwrap(), bytes);
}

#[rstest]
#[case::first_empty(b"\x01\x08\x41\x01u\x00\x00", 65)]
#[case::first_binary(b"\x01\x08\x41\x01u\x02\x00\x00\xff", 65)]
#[case::last_empty(b"\x01\x08\xff\x01u\x00\x00", 255)]
#[case::last_binary(b"\x01\x08\xff\x01u\x02\x00\x00\xff", 255)]
fn supported_models_report_typed_user_kind_projection_errors(
    #[case] bytes: &[u8],
    #[case] kind: u8,
) {
    let wire = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    let expected = LinkProjectionError::UnsupportedKind(UserLinkKind::new(kind).unwrap());
    assert_eq!(OwnedLink::try_from(wire.clone()).unwrap_err(), expected);
    assert_eq!(LinkMessage::try_from(wire).unwrap_err(), expected);
    assert_eq!(
        OwnedLink::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkStorageError::Format(FormatError::InvalidLinkType(kind))
    );
}

#[rstest]
#[case::missing_length(b"\x01\x08\x41\x01u", 7, 5)]
#[case::partial_length(b"\x01\x08\xff\x01u\x00", 7, 6)]
#[case::truncated_value(b"\x01\x08\xff\x01u\x02\x00\x00", 9, 8)]
fn owned_compatibility_parsing_decodes_user_framing_before_rejecting_kind(
    #[case] bytes: &[u8],
    #[case] expected: usize,
    #[case] available: usize,
) {
    assert_eq!(
        OwnedLink::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkStorageError::Format(FormatError::UnexpectedEof {
            expected,
            available
        })
    );
}

#[rstest]
#[case::hard(b"\x01\x01\x01\x00x\x34\x12\x00\x00", 7)]
#[case::soft(b"\x01\x09\x01\x01\x00x\x01\x00y\x00\x00", 9)]
fn owned_prefix_parsing_preserves_original_consumed_extent(
    #[case] bytes: &[u8],
    #[case] len: usize,
) {
    let (owned, consumed) = OwnedLink::parse_padded(bytes, OffsetWidth::Two).unwrap();
    assert_eq!(consumed, len);
    assert_eq!(owned.encoded_len(OffsetWidth::Two).unwrap(), len);
    assert_eq!(owned.serialize(OffsetWidth::Two).unwrap(), bytes[..len]);
    assert_eq!(
        OwnedLink::parse(bytes, OffsetWidth::Two).unwrap_err(),
        LinkStorageError::TrailingBytes
    );
}

#[rstest]
#[case::hard(
    b"\x01\x19\x00\x00\x01\x00\xff\x34\x12",
    LinkTarget::Hard { object_header_address: StoredAddress::new(0x1234) },
    CharacterSet::Ascii
)]
#[case::soft(b"\x01\x19\x01\x00\x01\x00\xff\x01\x00x", LinkTarget::soft("x".into()), CharacterSet::Ascii)]
#[case::external(
    b"\x01\x19\x40\x01\x01\x00\xff\x05\x00\x00\xfe\x00/\x00",
    LinkTarget::external("\u{fffd}".into(), "/".into()), CharacterSet::Utf8
)]
fn text_projection_preserves_compatibility_conversion(
    #[case] bytes: &[u8],
    #[case] link_target: LinkTarget,
    #[case] charset: CharacterSet,
) {
    let wire = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    let text = LinkMessage::try_from(wire).unwrap();
    assert_eq!(
        text,
        LinkMessage {
            name: "\u{fffd}".into(),
            link_target,
            creation_order: None,
            charset,
        }
    );
    assert_eq!(text, LinkMessage::parse(bytes, 2).unwrap());
}

#[test]
fn text_projection_retains_invalid_soft_utf8_error() {
    let bytes = b"\x01\x08\x01\x01x\x01\x00\xff";
    let wire = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    let error = LinkMessage::try_from(wire).unwrap_err();
    let LinkProjectionError::Format(FormatError::InvalidSoftLinkEncoding { source }) = &error
    else {
        panic!("expected InvalidSoftLinkEncoding, got {error:?}");
    };
    assert_eq!(source.valid_up_to(), 0);
    assert_eq!(source.error_len(), Some(1));
    assert_eq!(
        error,
        LinkProjectionError::Format(LinkMessage::parse(bytes, 2).unwrap_err())
    );
}

#[test]
fn text_projection_serializes_with_canonical_framing() {
    let wire =
        LinkMessageV1::parse(b"\x01\x19\x00\x00\x01\x00x\x34\x12", OffsetWidth::Two).unwrap();
    let text = LinkMessage::try_from(wire).unwrap();
    assert_eq!(
        text.serialize(OffsetWidth::Two).unwrap(),
        b"\x01\x00\x01x\x34\x12"
    );
}

#[test]
fn replacing_a_hard_target_preserves_framing() {
    let bytes = b"\x01\x19\x00\x00\x01\x00x\x34\x12";
    let mut owned = OwnedLink::parse(bytes, OffsetWidth::Two).unwrap();
    owned
        .set_hard_target(StoredAddress::new(0x2345), OffsetWidth::Two)
        .unwrap();
    assert_eq!(
        owned.serialize(OffsetWidth::Two).unwrap(),
        b"\x01\x19\x00\x00\x01\x00x\x45\x23"
    );
}

fn hard() -> OwnedLinkTarget {
    OwnedLinkTarget::Hard {
        object_header_address: StoredAddress::new(0x1234),
    }
}

fn external() -> OwnedLinkTarget {
    OwnedLinkTarget::external(vec![0xff], b"/".to_vec())
}
