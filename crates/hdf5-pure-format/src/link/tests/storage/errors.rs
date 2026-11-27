use alloc::string::ToString;
use core::error::Error;

use rstest::rstest;

use crate::link::storage::LinkStorageError;

#[rstest]
#[case::creation_order_counter(
    LinkStorageError::CreationOrderCounter { index: 23, next: 17 },
    "link creation index 23 must be below counter 17"
)]
#[case::creation_order_counter_range(
    LinkStorageError::CreationOrderCounterRange { next: 23, maximum: 17 },
    "link creation order counter 23 exceeds maximum 17"
)]
#[case::creation_order_exhausted(
    LinkStorageError::CreationOrderExhausted,
    "link creation order counter is exhausted"
)]
#[case::creation_order_presence(
    LinkStorageError::CreationOrderPresence,
    "link creation index presence disagrees with tracking policy"
)]
#[case::duplicate_creation_order(
    LinkStorageError::DuplicateCreationOrder(23),
    "duplicate link creation index 23"
)]
#[case::duplicate_name(
    LinkStorageError::DuplicateName(b"a\n\xff".to_vec()),
    r#"duplicate link name b"a\n\xff""#
)]
#[case::external_value(
    LinkStorageError::ExternalValue,
    "external link value lacks its version, separator, or terminator"
)]
#[case::external_version_flags(
    LinkStorageError::ExternalVersionFlags(0x23),
    "unsupported external link version or flags: 0x23"
)]
#[case::flags(LinkStorageError::Flags(0x80), "reserved link flag bits are set: 0x80")]
#[case::indexing_without_tracking(
    LinkStorageError::IndexingWithoutTracking,
    "link creation order indexing requires tracking"
)]
#[case::invalid_name(
    LinkStorageError::InvalidName(b"\0\"\xff".to_vec()),
    r#"invalid link name b"\x00\"\xff""#
)]
#[case::invalid_target(
    LinkStorageError::InvalidTarget,
    "symbolic link target is empty or contains a null byte"
)]
#[case::length_overflow(
    LinkStorageError::LengthOverflow,
    "encoded link length exceeds the addressable buffer size"
)]
#[case::not_hard_link(
    LinkStorageError::NotHardLink,
    "replacing a hard link target requires a hard link"
)]
#[case::storage_addresses(
    LinkStorageError::StorageAddresses,
    "link storage addresses disagree with each other or with the indexing policy"
)]
#[case::target_length(
    LinkStorageError::TargetLength(usize::from(u16::MAX) + 1),
    "link target length 65536 exceeds its two-byte field"
)]
#[case::trailing_bytes(
    LinkStorageError::TrailingBytes,
    "trailing bytes after link message fields"
)]
#[case::undefined_address(
    LinkStorageError::UndefinedAddress,
    "required link address is undefined"
)]
fn link_storage_errors_display_context_without_a_source(
    #[case] error: LinkStorageError,
    #[case] display: &str,
) {
    assert_eq!(error.to_string(), display);
    assert!(error.source().is_none());
}
