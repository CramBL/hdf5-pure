//! Complete observations of independent legacy files across reader backends.

use hdf5_pure::{DatatypeByteOrder, Error, FormatError};
use rstest::rstest;
use test_util::legacy::Fixture;
use test_util::object_header::v1;
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

#[rstest]
fn group_reads_report_an_incomplete_version_1_root_extent(
    #[values(ReadBackend::Bytes, ReadBackend::Source)] backend: ReadBackend,
) {
    let mut bytes = Fixture::SimpleDataset.build();
    let chunks = v1::root_group_chunks(&bytes);
    let chunk = &chunks[0];
    let declared = chunk.range.len() + v1::MESSAGE_ALIGNMENT;
    chunk
        .length_field
        .set(&mut bytes, u64::try_from(declared).unwrap());
    let expected = chunk.range.start + declared;
    let available = bytes.len();
    let error = match backend {
        ReadBackend::Buffered | ReadBackend::Bytes => FormatError::UnexpectedEof {
            expected,
            available,
        },
        ReadBackend::Source | ReadBackend::Streaming => FormatError::UnexpectedSourceEof {
            expected: u64::try_from(expected).unwrap(),
            available: u64::try_from(available).unwrap(),
        },
    };

    let file = backend.open_image(bytes).unwrap();
    let result = file.root().groups().unwrap_err();
    let Error::Format(actual) = result else {
        panic!("expected a format error, got {result:?}");
    };
    assert_eq!(actual, error);
    file.close().unwrap();
}
