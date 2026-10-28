//! Native reads and header-preservation checks for generic-large layouts.

#![cfg(feature = "__hdf5-1.10")]

use hdf5_pure::AttrValue;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::generic_large;
use test_util_hdf5::generic_large::HeaderImage;
use test_util_hdf5::generic_large::State;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::reader::ReadBackend;

#[test]
fn libhdf5_reads_independent_generic_large() {
    let path = temp::temp_path("independent_generic_large.h5");
    std::fs::write(&path, test_util::generic_large::generic_large()).unwrap();
    Fixture::GenericLarge.assert_native(&path, ByteOrder::LittleEndian);
    file_space_oracle::assert_persisted_sections(&path);
}

#[rstest]
fn native_generic_large_preserves_contents_and_unmodified_headers(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("native_generic_large.h5");
    Fixture::GenericLarge.write(&path);
    let order = if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    Fixture::GenericLarge.assert_native(&path, order);
    let image = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    generic_large::assert_storage(&file, &image);
    Fixture::GenericLarge.assert_contents(&file, order);
    drop(file);
    let header = HeaderImage::read(&path);
    let file = File::open_rw(&path).unwrap();
    file.root().set_attr("probe", AttrValue::I32(9)).unwrap();
    file.commit().unwrap();
    drop(file);
    assert_eq!(HeaderImage::read(&path), header);
    let file = backend.open(&path).unwrap();
    let pure_order = if cfg!(target_endian = "big") {
        DatatypeByteOrder::BigEndian
    } else {
        DatatypeByteOrder::LittleEndian
    };
    generic_large::assert_contents(&file, pure_order, State::RootAttribute);
    drop(file);
    generic_large::assert_native_after_probe(&path, order);
    file_space_oracle::assert_persisted_sections(&path);
}
