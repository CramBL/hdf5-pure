use hdf5_pure::AttrValue;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::generic_large;
use test_util_hdf5::generic_large::HeaderImage;
use test_util_hdf5::generic_large::State;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn independent_generic_large_preserves_contents_and_unmodified_headers(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("generic_large.h5");
    let image = test_util::generic_large::generic_large();
    std::fs::write(&path, &image).unwrap();
    let file = backend.open(&path).unwrap();
    generic_large::assert_contents(&file, DatatypeByteOrder::LittleEndian, State::Original);
    generic_large::assert_storage(&file, &image);
    generic_large::assert_independent_spans(&file, &image);
    drop(file);
    let header = HeaderImage::read(&path);
    let file = File::open_rw(&path).unwrap();
    file.root().set_attr("probe", AttrValue::I32(9)).unwrap();
    file.commit().unwrap();
    drop(file);
    assert_eq!(HeaderImage::read(&path), header);
    let file = backend.open(&path).unwrap();
    generic_large::assert_contents(&file, DatatypeByteOrder::LittleEndian, State::RootAttribute);
    drop(file);
    let file = File::open_rw(&path).unwrap();
    file.root().delete("d").unwrap();
    file.commit().unwrap();
    drop(file);
    let file = backend.open(&path).unwrap();
    generic_large::assert_contents(&file, DatatypeByteOrder::LittleEndian, State::Deleted);
    drop(file);
}
