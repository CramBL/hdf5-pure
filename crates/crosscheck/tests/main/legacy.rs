#![cfg(feature = "hdf5")]
//! Native conformance for independent symbol-table files and files written by the reference library.

use hdf5::datatype::ByteOrder;
use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::legacy::Fixture;
use test_util::temp;
use test_util_hdf5::legacy;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::nested_groups(Fixture::NestedGroups)]
#[case::simple_dataset(Fixture::SimpleDataset)]
#[case::two_groups(Fixture::TwoGroups)]
fn libhdf5_symbol_tables_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("native_legacy.h5");
    legacy::write_libhdf5(&path, fixture);
    let (pure_order, native_order) = if cfg!(target_endian = "big") {
        (DatatypeByteOrder::BigEndian, ByteOrder::BigEndian)
    } else {
        (DatatypeByteOrder::LittleEndian, ByteOrder::LittleEndian)
    };
    legacy::assert_libhdf5_contents(&path, fixture, native_order);
    let bytes = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    legacy::assert_contents(&file, fixture, pure_order);
    legacy::assert_symbol_tables(&file, &bytes, fixture);
    drop(file);
}

#[rstest]
#[case::nested_groups(Fixture::NestedGroups)]
#[case::simple_dataset(Fixture::SimpleDataset)]
#[case::two_groups(Fixture::TwoGroups)]
fn libhdf5_reads_independent_symbol_tables(#[case] fixture: Fixture) {
    let path = temp::temp_path("independent_legacy.h5");
    std::fs::write(&path, fixture.build()).unwrap();
    legacy::assert_libhdf5_contents(&path, fixture, ByteOrder::LittleEndian);
}
