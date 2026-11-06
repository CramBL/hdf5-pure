//! Complete observations of independent legacy files across reader backends.

use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::legacy::Fixture;
use test_util_hdf5::legacy;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::nested_groups(Fixture::NestedGroups)]
#[case::simple_dataset(Fixture::SimpleDataset)]
#[case::two_groups(Fixture::TwoGroups)]
fn independent_symbol_table_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let bytes = fixture.build();
    let file = backend.open_image(bytes.clone()).unwrap();
    legacy::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
    legacy::assert_symbol_tables(&file, &bytes, fixture);
    file.close().unwrap();
}
