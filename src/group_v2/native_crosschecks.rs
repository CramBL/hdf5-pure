//! Dense-link index parsing in files written by libhdf5.

use test_util::temp;
use test_util_hdf5::native_fixtures;

#[test]
fn a_native_dense_link_walk_parses_its_huge_object_index_once() {
    let path = temp::temp_path("native_huge_links.h5");
    native_fixtures::huge_links_12(&path);
    super::huge_link_tests::assert_dense_walk(std::fs::read(&path).unwrap());
}
