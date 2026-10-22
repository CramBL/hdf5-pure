//! The types a caller reaches by value through the public API can also be
//! named.
//!
//! Every module in this crate is `pub(crate)`, so a type is public API only if
//! `lib.rs` re-exports it — but reading a `pub` field and calling an inherent
//! method need no name, so a type can be reachable and unwritable at once.
//! `File::superblock` hands back a [`Superblock`] and with it a
//! [`BaseAddress`], and `Error::MissingMessage` carries a [`MessageType`].
//!
//! This file is a separate crate, exactly like a consumer, which is what makes
//! it the gate for two things. A re-export dropped from `lib.rs` stops it
//! compiling. So does a *retyped* field of `Superblock`, which nothing else
//! catches: `cargo-semver-checks` ships no lint for one (`struct_pub_field_missing`
//! covers removal and rename), and it could not see these types anyway, since
//! its lints match items by importable path. That combination is how
//! `Superblock::base_address` changed from `u64` to `BaseAddress` in 0.40.0
//! with no finding and no changelog line, leaving callers a value they could
//! compare and print but not use as a number.

use hdf5_pure::LengthWidth;
use hdf5_pure::OffsetWidth;
use hdf5_pure::StoredAddress;
use hdf5_pure::SuperblockStatusFlag;
use hdf5_pure::SuperblockStatusFlags;
use hdf5_pure::SuperblockVersion;
use hdf5_pure::{BaseAddress, Error, File, FileBuilder, LibVer, MessageType, Superblock};

use test_util::temp;

/// A consumer's own helper over the superblock: both types in one signature,
/// which is the thing that could not be written before.
fn image_start(sb: &Superblock) -> BaseAddress {
    sb.base_address
}

/// What a consumer would use the superblock for — deciding which format a file
/// is in, since HDF5 1.8 cannot open a version 3 superblock at all — written as
/// a function over the type rather than inline at the call site.
fn format_of(sb: &Superblock) -> LibVer {
    LibVer::from_superblock_version(sb.version)
}

/// Every public field, bound to the type it is declared with.
///
/// A field that changes type still compiles everywhere inside the crate and is
/// reported by no semver lint; it breaks only at a caller that named the old
/// type, which is what this function is. Nothing here needs to run — the
/// compile is the assertion — and it is called below only to keep `dead_code`
/// quiet under `clippy --all-targets -D warnings`.
fn field_types(sb: &Superblock) {
    let _: fn(StoredAddress) -> u64 = StoredAddress::get;
    let _: fn(hdf5_pure_core::StoredAddress) -> u64 = StoredAddress::get;
    let _: fn(OffsetWidth) -> StoredAddress = StoredAddress::undefined;
    let _: fn(StoredAddress, OffsetWidth) -> bool = StoredAddress::is_undefined;
    let _: fn(hdf5_pure_core::OffsetWidth) -> StoredAddress = StoredAddress::undefined;
    let _: fn(StoredAddress, hdf5_pure_core::OffsetWidth) -> bool = StoredAddress::is_undefined;
    let _: SuperblockVersion = sb.version;
    let _: hdf5_pure_core::SuperblockVersion = sb.version;
    let _: OffsetWidth = sb.offset_size;
    let _: LengthWidth = sb.length_size;
    let _: BaseAddress = sb.base_address;
    let _: u64 = sb.eof_address;
    let _: StoredAddress = sb.root_group_address;
    let _: Option<u16> = sb.group_leaf_node_k;
    let _: Option<u16> = sb.group_internal_node_k;
    let _: Option<u16> = sb.indexed_storage_internal_node_k;
    let _: Option<StoredAddress> = sb.free_space_address;
    let _: Option<StoredAddress> = sb.driver_info_address;
    let _: SuperblockStatusFlags = sb.consistency_flags;
    let _: hdf5_pure_core::SuperblockStatusFlags = sb.consistency_flags;
    let _: fn(SuperblockStatusFlags, SuperblockStatusFlag) -> bool =
        SuperblockStatusFlags::contains;
    let _: Option<StoredAddress> = sb.superblock_extension_address;
    let _: Option<u32> = sb.checksum;
}

#[test]
fn the_superblock_and_its_base_address_can_both_be_named() {
    let path = temp::temp_path("hdf5_pure_api_surface_superblock.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("alpha").with_f64_data(&[1.0]);
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    let file = File::open(&path).unwrap();
    let sb: &Superblock = file.superblock();
    field_types(sb);
    let base: BaseAddress = image_start(sb);
    assert_eq!(base.get(), 512, "the base is the userblock size");
    assert_eq!(
        sb.version.get(),
        3,
        "the default writer emits a version 3 superblock"
    );
    assert_eq!(format_of(sb), LibVer::V110);

    // Storable, not merely readable: a consumer can keep one past the file it
    // came from. Only the compiler can assert this — the binding names the type
    // and the read happens after the borrow it came from is gone.
    let kept: Superblock = sb.clone();
    drop(file);
    field_types(&kept);

    // The other half of the same question: bounds that select the 1.8 format
    // are visible through the same accessor.
    let old = temp::temp_path("hdf5_pure_api_surface_libver18.h5");
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("alpha").with_f64_data(&[1.0]);
    std::fs::write(&old, b.finish().unwrap()).unwrap();
    let file = File::open(&old).unwrap();
    assert_eq!(format_of(file.superblock()), LibVer::V18);
    assert_eq!(image_start(file.superblock()).get(), 0, "no userblock");
}

/// What a consumer does with the error: bind the message type out of the
/// variant and answer a question about it. Naming the type is what lets this be
/// a function at all rather than an arm that can only format its binding.
fn missing_message_id(e: &Error) -> Option<u16> {
    match e {
        Error::MissingMessage(t) => Some(t.to_u16()),
        _ => None,
    }
}

#[test]
fn a_header_message_type_can_be_named_where_an_error_hands_one_over() {
    let named = MessageType::from_u16(0x0008);
    assert_eq!(named, MessageType::DATA_LAYOUT);
    assert_eq!(
        missing_message_id(&Error::MissingMessage(named)),
        Some(0x0008)
    );

    const UNKNOWN_TYPE: u16 = 0x7FFF;

    let unknown = MessageType::from_u16(UNKNOWN_TYPE);
    assert_eq!(unknown.unknown_id(), Some(UNKNOWN_TYPE));
    assert_eq!(
        missing_message_id(&Error::MissingMessage(unknown)),
        Some(UNKNOWN_TYPE)
    );

    assert_eq!(missing_message_id(&Error::SwmrUnsupported), None);
}

#[rstest::rstest]
#[case(SuperblockVersion::Zero, 0, LibVer::Earliest)]
#[case(SuperblockVersion::One, 1, LibVer::Earliest)]
#[case(SuperblockVersion::Two, 2, LibVer::V18)]
#[case(SuperblockVersion::Three, 3, LibVer::V110)]
fn a_public_superblock_version_has_a_wire_value_and_a_library_bound(
    #[case] version: SuperblockVersion,
    #[case] byte: u8,
    #[case] bound: LibVer,
) {
    assert_eq!(SuperblockVersion::try_from(byte).unwrap(), version);
    assert_eq!(version.get(), byte);
    assert_eq!(LibVer::from_superblock_version(version), bound);
}

#[rstest::rstest]
#[case(SuperblockStatusFlags::new(), 0x00, false, false)]
#[case(SuperblockStatusFlags::write_marked(), 0x01, true, false)]
#[case(SuperblockStatusFlags::swmr_write_marked(), 0x05, true, true)]
fn status_flags_have_named_constructors_and_typed_queries(
    #[case] flags: SuperblockStatusFlags,
    #[case] bits: u8,
    #[case] write: bool,
    #[case] swmr: bool,
) {
    assert_eq!(flags.bits(), u32::from(bits));
    assert_eq!(u8::try_from(flags).unwrap(), bits);
    assert_eq!(flags.contains(SuperblockStatusFlag::WriteAccess), write);
    assert_eq!(flags.contains(SuperblockStatusFlag::SwmrWriteAccess), swmr);
    assert!(!flags.is_inconsistent());
    assert_eq!(flags.unknown_bits(), 0);
}

#[test]
fn filter_metadata_types_are_public() {
    let _: fn(&hdf5_pure::Dataset) -> Result<Vec<hdf5_pure::FilterId>, Error> =
        hdf5_pure::Dataset::filters;
    let _: fn(&hdf5_pure::Dataset) -> Result<Vec<hdf5_pure::Filter>, Error> =
        hdf5_pure::Dataset::filter_pipeline;
    let _: fn(&hdf5_pure::ScaleOffsetParameters) -> hdf5_pure::ScaleOffset =
        hdf5_pure::ScaleOffsetParameters::mode;
    let id: hdf5_pure::CustomFilterId = 40000_u16.try_into().unwrap();
    let id: hdf5_pure::FilterId = hdf5_pure::FilterId::Custom(id);
    assert_eq!(id.get(), 40000);
    let _: hdf5_pure::FilterParameters =
        hdf5_pure::FilterParameters::Deflate(hdf5_pure::DeflateLevel::Six);
}
