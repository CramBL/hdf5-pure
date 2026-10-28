//! Native conformance for dense dataset and root attributes.

#![cfg(feature = "hdf5")]
use hdf5::datatype::ByteOrder;
use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::dense_attributes::Fixture;
use test_util::temp;
use test_util_hdf5::dense_attributes;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::dataset(Fixture::Dataset)]
#[case::root(Fixture::Root)]
fn libhdf5_dense_attribute_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("native_dense_attributes.h5");
    dense_attributes::write_libhdf5(&path, fixture);
    let (pure_order, native_order) = if cfg!(target_endian = "big") {
        (DatatypeByteOrder::BigEndian, ByteOrder::BigEndian)
    } else {
        (DatatypeByteOrder::LittleEndian, ByteOrder::LittleEndian)
    };
    dense_attributes::assert_libhdf5_contents(&path, fixture, native_order);
    let bytes = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    dense_attributes::assert_contents(&file, fixture, pure_order);
    dense_attributes::assert_storage(&file, &bytes, fixture, pure_order);
    drop(file);
}

#[rstest]
#[case::dataset(Fixture::Dataset)]
#[case::root(Fixture::Root)]
fn libhdf5_reads_independent_dense_attribute_files(#[case] fixture: Fixture) {
    let path = temp::temp_path("independent_dense_attributes.h5");
    std::fs::write(&path, fixture.build()).unwrap();
    dense_attributes::assert_libhdf5_contents(&path, fixture, ByteOrder::LittleEndian);
}
