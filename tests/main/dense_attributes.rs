//! Complete observations of independent dense attributes across reader backends.

use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::dense_attributes::Fixture;
use test_util::temp;
use test_util_hdf5::dense_attributes;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::dataset(Fixture::Dataset)]
#[case::root(Fixture::Root)]
fn independent_dense_attribute_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("dense_attributes.h5");
    let bytes = match fixture {
        Fixture::Dataset => test_util::dense_attributes::dense_attrs(),
        Fixture::Root => test_util::dense_attributes::dense_attrs_root(),
    };
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    dense_attributes::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
    dense_attributes::assert_storage(&file, &bytes, fixture, DatatypeByteOrder::LittleEndian);
    drop(file);
}
