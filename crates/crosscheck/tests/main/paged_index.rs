//! Native content and free-space observations of an independent paged index.

#![cfg(feature = "__hdf5-1.10")]

use test_util::temp;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

#[test]
fn libhdf5_reads_independent_paged_index() {
    let path = temp::temp_path("independent_paged_index.h5");
    std::fs::write(&path, test_util::paged_index::paged_index()).unwrap();
    Fixture::PagedIndex.assert_native(&path, ByteOrder::LittleEndian);
    file_space_oracle::assert_persisted_sections(&path);
}
