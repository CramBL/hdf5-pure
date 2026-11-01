#![cfg(feature = "hdf5")]

use hdf5::file::LibraryVersion;
use hdf5::plist::group_create::LinkPhaseChange;
use hdf5_pure::{File, MessageType};
use hdf5_pure_format::__private::{AccessMode, GroupInfo, GroupPhaseChange, ObjectHeader};
use rstest::rstest;
use test_util_hdf5::reader::ReadBackend;
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

#[cfg(feature = "__hdf5-1.10")]
#[rstest]
#[case::defaults(None)]
#[case::thresholds(Some((12, 8)))]
#[case::one_way(Some((12, 0)))]
#[case::always_dense(Some((0, 0)))]
fn inspected_link_policy_agrees_with_native_creation_properties(
    #[case] thresholds: Option<(u32, u32)>,
    #[values("/", "nested")] group_path: &str,
    #[values(0, 13)] count: usize,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-policy.h5");
    test_util_hdf5::dense_links::write_edit_fixture(&path, group_path, count, thresholds);
    let native = hdf5::File::open(&path).unwrap();
    let group = native.group(group_path).unwrap();
    let expected = group.gcpl().unwrap().get_link_phase_change().unwrap();
    if count > 0 {
        assert_eq!(
            group.info().unwrap().storage_type,
            hdf5::GroupStorageType::Dense
        );
    }
    drop(group);
    native.close().unwrap();
    let file = backend.open(&path).unwrap();
    let policy = file.group(group_path).unwrap().link_phase_change().unwrap();
    assert_eq!(
        (
            u32::from(policy.max_compact()),
            u32::from(policy.min_dense())
        ),
        (expected.max_compact, expected.min_dense)
    );
    file.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::root("/")]
#[case::nested("legacy")]
fn legacy_group_policy_matches_native_defaults(
    #[case] group_path: &str,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy-policy.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18)
        })
        .create(&path)
        .unwrap();
    let child = native.create_group("legacy").unwrap();
    drop(child);
    let group = native.group(group_path).unwrap();
    assert_eq!(
        group.info().unwrap().storage_type,
        hdf5::GroupStorageType::SymbolTable
    );
    let expected = group.gcpl().unwrap().get_link_phase_change().unwrap();
    drop(group);
    native.close().unwrap();
    let file = backend.open(&path).unwrap();
    let policy = file.group(group_path).unwrap().link_phase_change().unwrap();
    assert_eq!(
        (
            u32::from(policy.max_compact()),
            u32::from(policy.min_dense())
        ),
        (expected.max_compact, expected.min_dense)
    );
    assert_eq!(policy, hdf5_pure::LinkPhaseChange::default());
    file.close().unwrap();
    directory.close().unwrap();
}

#[cfg(feature = "__hdf5-1.10")]
#[test]
fn policy_inspection_resolves_a_replaced_group_again() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("replaced-policy.h5");
    test_util_hdf5::dense_links::write_edit_fixture(&path, "group", 0, Some((12, 8)));
    let file = File::open_rw(&path).unwrap();
    let group = file.group("group").unwrap();
    let before = group.link_phase_change().unwrap();
    assert_eq!((before.max_compact(), before.min_dense()), (12, 8));
    file.root().delete("group").unwrap();
    let replacement = file.root().create_group("group").unwrap();
    drop(replacement);
    file.commit().unwrap();
    assert_eq!(
        group.link_phase_change().unwrap(),
        hdf5_pure::LinkPhaseChange::default()
    );
    drop(group);
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let group = native.group("group").unwrap();
    assert_eq!(
        group.gcpl().unwrap().get_link_phase_change().unwrap(),
        LinkPhaseChange::default()
    );
    drop(group);
    native.close().unwrap();
    directory.close().unwrap();
}
