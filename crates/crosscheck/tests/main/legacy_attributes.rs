//! Native conformance for independent compact attributes and fresh reference files.

#![cfg(feature = "hdf5")]

use hdf5::datatype::ByteOrder;
use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::legacy_attributes::Fixture;
use test_util::temp;
use test_util_hdf5::legacy_attributes;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::compact_attributes(Fixture::CompactAttributes)]
#[case::mixed_attributes(Fixture::MixedAttributes)]
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
    let path = temp::temp_path("native_legacy_attributes.h5");
    legacy_attributes::write_libhdf5(&path, fixture);
    let (pure_order, native_order) = if cfg!(target_endian = "big") {
        (DatatypeByteOrder::BigEndian, ByteOrder::BigEndian)
    } else {
        (DatatypeByteOrder::LittleEndian, ByteOrder::LittleEndian)
    };
    legacy_attributes::assert_libhdf5_contents(&path, fixture, native_order);
    let bytes = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    legacy_attributes::assert_contents(&file, fixture, pure_order);
    legacy_attributes::assert_storage(&file, &bytes, fixture);
    drop(file);
}

#[rstest]
#[case::compact_attributes(Fixture::CompactAttributes)]
#[case::mixed_attributes(Fixture::MixedAttributes)]
fn libhdf5_reads_independent_symbol_tables(#[case] fixture: Fixture) {
    let path = temp::temp_path("independent_legacy_attributes.h5");
    std::fs::write(&path, fixture.build()).unwrap();
    legacy_attributes::assert_libhdf5_contents(&path, fixture, ByteOrder::LittleEndian);
}
