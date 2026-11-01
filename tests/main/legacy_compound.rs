use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::temp;
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
    let path = temp::temp_path("compound_v1_array.h5");
    std::fs::write(&path, test_util::legacy_compound::file()).unwrap();
    let file = backend.open(&path).unwrap();
    legacy_compound::assert_contents(&file, DatatypeByteOrder::LittleEndian);
    drop(file);
}
