//! Subtree reclamation in files with native huge-link storage.

use test_util::temp;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use crate::edit;

#[test]
fn deleting_a_native_dense_link_group_reclaims_its_whole_subtree() {
    let path = temp::temp_path("native_huge_links.h5");
    Fixture::HugeLinks12.write(&path);
    let order = if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    Fixture::HugeLinks12.assert_native(&path, order);
    edit::test_util::assert_huge_subtree_reclaimed(&path);
    let file = test_util_hdf5::file::libhdf5_open(&path);
    assert!(file.member_names().unwrap().is_empty());
    file.close().unwrap();
}
