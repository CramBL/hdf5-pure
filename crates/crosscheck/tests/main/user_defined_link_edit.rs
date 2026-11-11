#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::GroupStorageType;
use hdf5_pure::{Error, File, MessageType};
use hdf5_pure_format::__private::{AccessMode, LinkInfoMessage, ObjectHeader};
use rstest::rstest;
use test_util_hdf5::{file, user_defined_link};

#[rstest]
#[case::version_1_compact_empty(file::libhdf5_create_earliest, 1, false, b"")]
#[case::version_1_compact_binary(file::libhdf5_create_earliest, 1, false, b"\x00\xff\x80\x00")]
#[case::version_2_compact_empty(file::libhdf5_create_v18, 2, false, b"")]
#[case::version_2_compact_binary(file::libhdf5_create_v18, 2, false, b"\x00\xff\x80\x00")]
#[case::version_2_dense_empty(file::libhdf5_create_v18, 2, true, b"")]
#[case::version_2_dense_binary(file::libhdf5_create_v18, 2, true, b"\x00\xff\x80\x00")]
fn root_dataset_creation_rejects_native_user_links_as_unsupported(
    #[case] create: fn(&Path) -> hdf5::File,
    #[case] version: u8,
    #[case] dense: bool,
    #[case] value: &[u8],
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user_link_edit.h5");
    let native = create(&path);
    if dense {
        for index in 0..12 {
            native.create_group(&format!("g{index}")).unwrap();
        }
    }
    user_defined_link::create(&native, c"u", value);
    let info = native.group("/").unwrap().info().unwrap();
    assert_eq!(info.nlinks, if dense { 13 } else { 1 });
    assert_eq!(
        info.storage_type,
        if dense {
            GroupStorageType::Dense
        } else {
            GroupStorageType::Compact
        }
    );
    native.close().unwrap();
    let original = std::fs::read(&path).unwrap();
    let session = File::open_rw(&path).unwrap();
    let superblock = session.superblock();
    let header = ObjectHeader::parse(
        &original,
        AccessMode::ReadOnly,
        usize::try_from(superblock.root_group_address.get()).unwrap(),
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    assert_eq!(header.version, version);
    let info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    let info = LinkInfoMessage::parse(&info.data, superblock.offset_size.get()).unwrap();
    assert_eq!(info.fractal_heap_address.is_some(), dense);
    session
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[1]);
        })
        .unwrap();
    let error = session.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("a group's user-defined links cannot be reproduced")
        ),
        "{error:?}"
    );
    drop(session);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    dir.close().unwrap();
}
