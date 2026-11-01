//! Byte preservation of native superblock-extension records.

use rstest::rstest;
use test_util::temp;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use crate::edit;
use crate::edit::OpeningExtension;
use crate::edit::context::FileContext;
use crate::edit::object_header;
use crate::source::BytesSource;
use crate::superblock;

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

#[rstest]
#[case::paged_index(Fixture::PagedIndex)]
#[case::generic_large(Fixture::GenericLarge)]
fn native_extension_opening_facts_retain_complete_allocations(#[case] fixture: Fixture) {
    let path = temp::temp_path("native_extension_opening.h5");
    fixture.write(&path);
    let order = if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    fixture.assert_native(&path, order);
    let source = BytesSource::new(std::fs::read(&path).unwrap());
    let context = FileContext::new(superblock::read(&source).unwrap()).unwrap();
    let facts = OpeningExtension::read(&source, &context).unwrap();
    let (address, info) = facts.info.unwrap();
    assert!(info.persist);
    assert_eq!(info.threshold, 1);
    assert_eq!(
        info.page_size.get(),
        match fixture {
            Fixture::GenericLarge => 512,
            Fixture::PagedIndex => 4_096,
            _ => panic!("fixture has no recorded page size: {fixture:?}"),
        }
    );
    let expected = object_header::read_oh_chunks(&source, address, context.addresses()).unwrap();
    let allocations = facts.allocations.unwrap();
    assert_eq!(
        allocations,
        expected.iter().map(|chunk| chunk.span).collect::<Vec<_>>()
    );
    assert!(!allocations.is_empty());
    assert!(allocations.iter().all(|span| span.length() > 0));
}
