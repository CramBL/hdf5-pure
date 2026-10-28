use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::huge_links;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn independent_huge_links_have_complete_contents(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("huge_links.h5");
    let bytes = test_util::modern::huge_links_12();
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    huge_links::assert_contents(&file, DatatypeByteOrder::LittleEndian);
    huge_links::assert_storage(&file, &bytes);
    drop(file);
}
