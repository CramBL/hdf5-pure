#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5_pure::{File, MessageType};
use hdf5_pure_format::__private::{AccessMode, LinkMessageV1, LinkTargetV1, ObjectHeader};
use rstest::rstest;
use test_util_hdf5::{file, user_defined_link};

#[rstest]
#[case::version_1_empty(file::libhdf5_create_earliest, b"")]
#[case::version_1_binary(file::libhdf5_create_earliest, b"\x00\xff\x80\x00")]
#[case::version_2_empty(file::libhdf5_create_v18, b"")]
#[case::version_2_binary(file::libhdf5_create_v18, b"\x00\xff\x80\x00")]
fn wire_codec_preserves_libhdf5_user_defined_link_messages(
    #[case] create: fn(&Path) -> hdf5::File,
    #[case] value: &[u8],
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user_link.h5");
    let file = create(&path);
    user_defined_link::create(&file, c"u", value);
    file.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let pure = File::open(&path).unwrap();
    let superblock = pure.superblock();
    let width = superblock.offset_size;
    let header = ObjectHeader::parse(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(superblock.root_group_address.get()).unwrap(),
        width.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    drop(pure);
    let messages = header
        .messages
        .iter()
        .filter(|message| message.msg_type == MessageType::LINK)
        .collect::<Vec<_>>();
    assert_eq!(messages.len(), 1);
    let (wire, consumed) = LinkMessageV1::parse_prefix(&messages[0].data, width).unwrap();
    assert_eq!(wire.name().as_bytes(), b"u");
    assert_eq!(wire.creation_order(), None);
    assert_eq!(wire.charset(), None);
    let LinkTargetV1::UserDefined {
        kind,
        value: parsed_value,
    } = wire.target()
    else {
        panic!("expected a user-defined link, got {:?}", wire.target());
    };
    assert_eq!(kind.get(), 255);
    assert_eq!(parsed_value.as_bytes(), value);
    assert_eq!(wire.encoded_len(width).unwrap(), consumed);
    assert_eq!(wire.serialize(width).unwrap(), messages[0].data[..consumed]);
}
