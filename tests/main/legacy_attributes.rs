//! Complete observations of independent legacy attributes and strings across reader backends.

use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::legacy_attributes::Fixture;
use test_util::temp;
use test_util_hdf5::legacy_attributes;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::compact_attributes(Fixture::CompactAttributes)]
#[case::mixed_attributes(Fixture::MixedAttributes)]
#[case::vl_strings(Fixture::VlenStrings)]
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
    let path = temp::temp_path("legacy_attributes.h5");
    let bytes = fixture.build();
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    legacy_attributes::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
    legacy_attributes::assert_storage(&file, &bytes, fixture);
    drop(file);
}
