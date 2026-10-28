//! Complete observations of independent compact and dense groups across reader backends.

use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::SuperblockVersion;
use rstest::rstest;
use test_util::modern::Fixture;
use test_util::temp;
use test_util_hdf5::modern;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::groups(Fixture::Groups)]
#[case::many_links(Fixture::ManyLinks)]
fn independent_link_message_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("modern.h5");
    let bytes = fixture.build();
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    assert_eq!(file.superblock().version(), SuperblockVersion::Three);
    modern::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
    modern::assert_representation(&file, &bytes, fixture);
    drop(file);
}
