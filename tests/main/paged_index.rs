use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util_hdf5::paged_index;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn independent_paged_index_has_complete_contents(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let image = test_util::paged_index::paged_index();
    let file = backend.open_image(image.clone()).unwrap();
    paged_index::assert_contents(&file, DatatypeByteOrder::LittleEndian);
    paged_index::assert_storage(&file, &image);
    file.close().unwrap();
}
