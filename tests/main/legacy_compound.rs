use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util_hdf5::legacy_compound;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn compound_v1_array_members_retain_all_values(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let file = backend
        .open_image(test_util::legacy_compound::file())
        .unwrap();
    legacy_compound::assert_contents(&file, DatatypeByteOrder::LittleEndian);
    file.close().unwrap();
}
