use std::error::Error as StdError;

use hdf5_pure::AddressRange;
use hdf5_pure::EditStorageConflict;
use hdf5_pure::Error;
use hdf5_pure::PhysicalAddress;
use rstest::rstest;

#[rstest]
#[case::bounds(
    EditStorageConflict::Bounds { range: range(4_294_967_600, 16), end: PhysicalAddress::new(4_294_967_608) },
    "range 4294967600..4294967616 extends past 4294967608"
)]
#[case::overlap(
    EditStorageConflict::Overlap { first: range(4_294_967_600, 16), second: range(4_294_967_608, 16) },
    "ranges 4294967600..4294967616 and 4294967608..4294967624 overlap"
)]
#[case::truncation(
    EditStorageConflict::Truncation { proposed: PhysicalAddress::new(4_294_967_600), required: PhysicalAddress::new(4_294_967_616) },
    "proposed end 4294967600 precedes required end 4294967616"
)]
fn edit_storage_errors_preserve_geometry_and_source(
    #[case] conflict: hdf5_pure_core::EditStorageConflict,
    #[case] diagnostic: &str,
) {
    let error = Error::from(conflict);
    let Error::EditStorageConflict(actual) = &error else {
        panic!("unexpected storage conflict: {error:?}");
    };
    assert_eq!(*actual, conflict);
    assert_eq!(actual.to_string(), diagnostic);
    assert_eq!(
        error.to_string(),
        format!("edit storage conflict: {diagnostic}")
    );
    let source = error
        .source()
        .unwrap()
        .downcast_ref::<EditStorageConflict>()
        .unwrap();
    assert!(std::ptr::eq(source, actual));
    assert_eq!(source.source().map(ToString::to_string), None);
}

fn range(start: u64, length: u64) -> AddressRange<PhysicalAddress> {
    AddressRange::new(PhysicalAddress::new(start), length).unwrap()
}
