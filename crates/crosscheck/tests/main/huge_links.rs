//! Native reads of independently constructed huge-link storage.

#![cfg(feature = "__hdf5-1.10")]

use test_util::temp;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

#[test]
fn libhdf5_reads_independent_huge_links() {
    let path = temp::temp_path("independent_huge_links.h5");
    std::fs::write(&path, test_util::modern::huge_links_12()).unwrap();
    Fixture::HugeLinks12.assert_native(&path, ByteOrder::LittleEndian);
}
