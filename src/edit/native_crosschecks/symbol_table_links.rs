use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::soft_link;
use test_util_hdf5::soft_link::GroupFormat;
use test_util_hdf5::soft_link::NativeLink;

use crate::CharacterSet;
use crate::Error;
use crate::File;
use crate::MessageType;
use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::group_inspection;
use crate::file_writer::COMPACT_LINK_INFO;
use crate::object_header::ObjectHeader;

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
                NativeLink::External {
                    filename,
                    object_path,
                } => OwnedLinkTarget::external(filename.into_bytes(), object_path.into_bytes()),
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

#[rstest]
fn oversized_new_links_leave_legacy_groups_available_for_retry(
    #[values("/", "/parent")] group_path: &str,
    #[values(false, true)] source_only: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy-limit.h5");
    let native = file::libhdf5_create_earliest(&path);
    native
        .new_dataset::<i32>()
        .shape([3])
        .create("original")
        .unwrap()
        .write_raw(&[7, 11, 13])
        .unwrap();
    native
        .create_group("parent")
        .unwrap()
        .new_dataset::<i32>()
        .shape([2])
        .create("nested")
        .unwrap()
        .write_raw(&[17, 19])
        .unwrap();
    native.close().unwrap();
    let before = std::fs::read(&path).unwrap();
    let reader = File::open(&path).unwrap();
    let address = reader.group(group_path).unwrap().header_address().unwrap();
    let header = ObjectHeader::parse_with_widths(
        &before,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        reader.widths(),
    )
    .unwrap();
    assert!(
        header
            .messages
            .iter()
            .any(|message| message.msg_type == MessageType::SYMBOL_TABLE)
    );
    drop(reader);
    let session = group_inspection::open_group_inspection(&path, source_only).unwrap();
    let group = session.group(group_path).unwrap();
    group.create_group("retained").unwrap();
    let name = "x".repeat(65_532);
    group.create_group(&name).unwrap();
    let error = session.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a new hard link message exceeds the object-header message size limit"
            )
        ),
        "{error:?}"
    );
    assert_eq!(session.group_inspection_snapshot().unwrap(), before);
    assert_eq!(session.page_publication_snapshot().unwrap(), before);
    group.delete(&name).unwrap();
    session.commit().unwrap();
    drop(group);
    drop(session);
    let native = file::libhdf5_open(&path);
    let expected = if group_path == "/" {
        vec!["original", "parent", "retained"]
    } else {
        vec!["nested", "retained"]
    };
    let mut names = native.group(group_path).unwrap().member_names().unwrap();
    names.sort();
    assert_eq!(names, expected);
    assert_eq!(
        native
            .dataset("original")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [7, 11, 13]
    );
    assert_eq!(
        native
            .dataset("parent/nested")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [17, 19]
    );
    native.close().unwrap();
    directory.close().unwrap();
}
