#![cfg(feature = "__hdf5-1.10")]

use hdf5::File as CFile;
use hdf5::GroupStorageType;
use rstest::rstest;
use test_util_hdf5::dense_links::GroupOrder;

#[rstest]
#[case::untracked(GroupOrder::Untracked)]
#[case::tracked(GroupOrder::Tracked)]
#[case::indexed(GroupOrder::Indexed)]
fn generated_groups_have_expected_storage_policy_and_links(#[case] order: GroupOrder) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.h5");
    test_util_hdf5::dense_links::write_inspection_fixture(&path, order);
    let file = CFile::open(&path).unwrap();
    assert_eq!(file.info().unwrap().storage_type, GroupStorageType::Dense);
    test_util_hdf5::dense_links::assert_inspection_group(
        &file,
        &format!("compact_{}", order.label()),
        6,
        GroupStorageType::Compact,
        order.creation_order(),
    );
    test_util_hdf5::dense_links::assert_inspection_group(
        &file,
        &format!("dense_{}", order.label()),
        2000,
        GroupStorageType::Dense,
        order.creation_order(),
    );
    assert_eq!(
        file.group("filtered").unwrap().info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    file.close().unwrap();
    dir.close().unwrap();
}
