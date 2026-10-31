use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::soft_link;
use test_util_hdf5::soft_link::GroupFormat;
use test_util_hdf5::soft_link::NativeLink;

use crate::CharacterSet;
use crate::Error;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::group_inspection;
use crate::file_writer::COMPACT_LINK_INFO;

#[rstest]
fn both_readers_convert_the_links_libhdf5_writes_in_a_symbol_table_group(
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("symbol_table_links.h5");
    GroupFormat::SymbolTable.generate(&path);
    let native = soft_link::native_links(&file::libhdf5_open(&path), "links");
    assert!(
        native
            .iter()
            .any(|(_, link)| matches!(link, NativeLink::Soft { .. }))
    );
    let expected: Vec<OwnedLink> = native
        .into_iter()
        .map(|(name, link)| {
            let target = match link {
                NativeLink::Hard { address } => OwnedLinkTarget::Hard {
                    object_header_address: StoredAddress::new(address),
                },
                NativeLink::Soft { target } => OwnedLinkTarget::soft(target.into_bytes()),
            };
            OwnedLink::new(name.into_bytes(), CharacterSet::Ascii, target, None).unwrap()
        })
        .collect();
    let converted = group_inspection::inspect_group_links(&path, "links", source_only).unwrap();
    assert_eq!(converted.links.links(), expected);
    assert_eq!(converted.links.info(), COMPACT_LINK_INFO);
    dir.close().unwrap();
}

#[rstest]
fn a_link_longer_than_a_link_message_is_rejected(#[values(false, true)] source_only: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long_link.h5");
    {
        let file = file::libhdf5_create_earliest(&path);
        file.new_dataset::<i32>()
            .shape([1])
            .create("d")
            .unwrap()
            .write_raw(&[1])
            .unwrap();
        file.link_hard("d", &"x".repeat(usize::from(u16::MAX) + 1))
            .unwrap();
        file.close().unwrap();
    }
    let Err(error) = group_inspection::inspect_group_links(&path, "/", source_only) else {
        panic!("expected the conversion to reject the link");
    };
    assert!(
        matches!(
            error,
            Error::EditUnsupported("a version 1 group's link is too large to convert in place")
        ),
        "{error:?}"
    );
    dir.close().unwrap();
}
