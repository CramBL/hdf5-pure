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
//!
//! `scripts/check-api-surface.sh` audits the whole surface for the same class.

use hdf5_pure::{BaseAddress, Error, File, FileBuilder, LibVer, MessageType, Superblock};

use temp::temp_path;
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
    let _: u8 = sb.version;
    let _: u8 = sb.offset_size;
    let _: u8 = sb.length_size;
    let _: BaseAddress = sb.base_address;
    let _: u64 = sb.eof_address;
    let _: u64 = sb.root_group_address;
    let _: Option<u16> = sb.group_leaf_node_k;
    let _: Option<u16> = sb.group_internal_node_k;
    let _: Option<u16> = sb.indexed_storage_internal_node_k;
    let _: Option<u64> = sb.free_space_address;
    let _: Option<u64> = sb.driver_info_address;
    let _: u32 = sb.consistency_flags;
    let _: Option<u64> = sb.superblock_extension_address;
    let _: Option<u32> = sb.checksum;
}

#[test]
fn the_superblock_and_its_base_address_can_both_be_named() {
    let path = temp_path("hdf5_pure_api_surface_superblock.h5");
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
        sb.version, 3,
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
    let old = temp_path("hdf5_pure_api_surface_libver18.h5");
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
    let named: MessageType = MessageType::from_u16(0x0008);
    assert_eq!(named, MessageType::DataLayout);
    assert_eq!(
        missing_message_id(&Error::MissingMessage(named)),
        Some(0x0008)
    );

    // An identifier this crate does not recognize keeps its number rather than
    // being discarded, which is what makes the enum safe to grow.
    let unknown = MessageType::from_u16(0x7FFF);
    assert_eq!(unknown, MessageType::Unknown(0x7FFF));
    assert_eq!(
        missing_message_id(&Error::MissingMessage(unknown)),
        Some(0x7FFF)
    );
    assert_eq!(missing_message_id(&Error::SwmrUnsupported), None);
}
