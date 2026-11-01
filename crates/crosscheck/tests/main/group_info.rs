#![cfg(feature = "hdf5")]

use hdf5::file::LibraryVersion;
use hdf5::plist::group_create::LinkPhaseChange;
use hdf5_pure::{File, MessageType};
use hdf5_pure_format::__private::{AccessMode, GroupInfo, GroupPhaseChange, ObjectHeader};
use rstest::rstest;
use test_util_hdf5::storage;

#[rstest]
#[case::defaults(8, 6)]
#[case::thresholds(12, 8)]
#[case::one_way(12, 0)]
#[case::always_dense(0, 0)]
fn encoded_group_info_agrees_with_native_group_messages(
    #[case] max_compact: u16,
    #[case] min_dense: u16,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("group-info.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .create(&path)
        .unwrap();
    let expected = LinkPhaseChange {
        max_compact: u32::from(max_compact),
        min_dense: u32::from(min_dense),
    };
    let group = native
        .create_group_builder()
        .with_gcpl(|properties| properties.link_phase_change(expected))
        .create("policy")
        .unwrap();
    assert_eq!(
        group.gcpl().unwrap().get_link_phase_change().unwrap(),
        expected
    );
    drop(group);
    native.close().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let address = storage::root_object_address(&file, &bytes, "policy");
    let superblock = file.superblock();
    let header = ObjectHeader::parse(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(address.get()).unwrap(),
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    let message = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::GROUP_INFO)
        .unwrap();
    let info = GroupInfo::parse(&message.data).unwrap();
    assert_eq!(
        info.phase_change(),
        GroupPhaseChange::new(max_compact, min_dense).unwrap()
    );
    assert_eq!(info.estimates(), None);
    assert_eq!(info.serialize(), message.data);
    file.close().unwrap();
    directory.close().unwrap();
}
