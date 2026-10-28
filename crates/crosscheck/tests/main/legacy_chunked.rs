//! Native conformance for independent and repacked compressed chunks.

#![cfg(feature = "hdf5")]

use hdf5::datatype::ByteOrder;
use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::legacy_chunked;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn libhdf5_legacy_chunked_deflate_has_complete_contents(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("native_legacy_chunked.h5");
    legacy_chunked::write_libhdf5(&path);
    let (pure_order, native_order) = if cfg!(target_endian = "big") {
        (DatatypeByteOrder::BigEndian, ByteOrder::BigEndian)
    } else {
        (DatatypeByteOrder::LittleEndian, ByteOrder::LittleEndian)
    };
    legacy_chunked::assert_libhdf5_contents(&path, native_order);
    let bytes = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    legacy_chunked::assert_contents(&file, pure_order);
    legacy_chunked::assert_source_storage(&file, &bytes);
    drop(file);
}

#[test]
fn libhdf5_reads_independent_legacy_chunked_deflate() {
    let path = temp::temp_path("independent_legacy_chunked.h5");
    std::fs::write(&path, test_util::legacy_chunked::chunked_deflate()).unwrap();
    legacy_chunked::assert_libhdf5_contents(&path, ByteOrder::LittleEndian);
}

test_util_hdf5::hdf5_1_10_tests! {
use hdf5_pure::RepackOptions;

#[test]
fn libhdf5_reads_repacked_legacy_chunked_deflate() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("destination.h5");
    std::fs::write(&source, test_util::legacy_chunked::chunked_deflate()).unwrap();
    hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
    legacy_chunked::assert_libhdf5_contents(&destination, ByteOrder::LittleEndian);
}
}
