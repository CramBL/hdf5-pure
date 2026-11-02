use crate::group_v2::test_util as group_test_util;

#[test]
fn a_dense_link_walk_parses_its_huge_object_index_once() {
    group_test_util::assert_dense_walk(test_util::modern::huge_links_12());
}
