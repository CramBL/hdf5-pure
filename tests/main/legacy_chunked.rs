//! Independent compressed chunks across reader backends and repacking.

use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::RepackOptions;
use rstest::rstest;
use test_util_hdf5::legacy_chunked;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn independent_legacy_chunked_deflate_preserves_contents_and_upgrades(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("destination.h5");
    let bytes = test_util::legacy_chunked::chunked_deflate();
    std::fs::write(&source, &bytes).unwrap();
    let file = backend.open(&source).unwrap();
    legacy_chunked::assert_contents(&file, DatatypeByteOrder::LittleEndian);
    legacy_chunked::assert_source_storage(&file, &bytes);
    drop(file);
    hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
    let file = backend.open(&destination).unwrap();
    legacy_chunked::assert_repacked(&file, DatatypeByteOrder::LittleEndian);
    drop(file);
}
