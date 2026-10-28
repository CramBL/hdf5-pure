//! Byte preservation of native superblock-extension records.

use rstest::rstest;
use test_util::temp;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use crate::edit;
use crate::source::BytesSource;

#[rstest]
#[case::paged_index(Fixture::PagedIndex)]
#[case::generic_large(Fixture::GenericLarge)]
fn a_native_extension_keeps_its_unmodified_record_bytes(#[case] fixture: Fixture) {
    let path = temp::temp_path("native_extension.h5");
    fixture.write(&path);
    let order = if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    fixture.assert_native(&path, order);
    let source = BytesSource::new(std::fs::read(&path).unwrap());
    edit::test_util::assert_extension_records(&source);
}
