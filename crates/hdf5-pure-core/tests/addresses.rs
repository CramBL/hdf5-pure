use hdf5_pure_core::{FormatError, OffsetWidth, PhysicalAddress, StoredAddress};

#[test]
fn public_addresses_keep_their_coordinate_types() {
    let stored: StoredAddress = StoredAddress::undefined(OffsetWidth::Eight);
    assert_eq!(stored.get(), u64::MAX);
    let physical: PhysicalAddress = PhysicalAddress::new(512).checked_add(32).unwrap();
    assert_eq!(physical.get(), 544);
    assert_eq!(
        PhysicalAddress::new(u64::MAX).checked_add(1).unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX,
            length: 1
        }
    );
}

#[rstest::rstest]
#[case::zero(0, 0, Ok(0), Some(0))]
#[case::ordinary(512, 96, Ok(608), Some(416))]
#[case::maximum_zero(u64::MAX, 0, Ok(u64::MAX), Some(u64::MAX))]
#[case::maximum_addition(u64::MAX - 1, 1, Ok(u64::MAX), Some(u64::MAX - 2))]
#[case::maximum_subtraction(u64::MAX, u64::MAX, Err(FormatError::OffsetOverflow { offset: u64::MAX, length: u64::MAX }), Some(0))]
#[case::overflow(u64::MAX, 1, Err(FormatError::OffsetOverflow { offset: u64::MAX, length: 1 }), Some(u64::MAX - 1))]
#[case::underflow(0, 1, Ok(1), None)]
#[case::underflow_many(96, 512, Ok(608), None)]
fn physical_address_displacements_are_checked(
    #[case] address: u64,
    #[case] length: u64,
    #[case] advanced: Result<u64, FormatError>,
    #[case] retreated: Option<u64>,
) {
    let address = PhysicalAddress::new(address);
    assert_eq!(
        address.checked_add(length).map(PhysicalAddress::get),
        advanced
    );
    assert_eq!(
        address.checked_sub(length).map(PhysicalAddress::get),
        retreated
    );
}

#[rstest::rstest]
#[case::zero(0, 0, Ok(0), Some(0))]
#[case::ordinary(512, 96, Ok(608), Some(416))]
#[case::maximum_zero(u64::MAX, 0, Ok(u64::MAX), Some(u64::MAX))]
#[case::maximum_addition(u64::MAX - 1, 1, Ok(u64::MAX), Some(u64::MAX - 2))]
#[case::maximum_subtraction(u64::MAX, u64::MAX, Err(FormatError::OffsetOverflow { offset: u64::MAX, length: u64::MAX }), Some(0))]
#[case::overflow(u64::MAX, 1, Err(FormatError::OffsetOverflow { offset: u64::MAX, length: 1 }), Some(u64::MAX - 1))]
#[case::underflow(0, 1, Ok(1), None)]
#[case::underflow_many(96, 512, Ok(608), None)]
fn stored_address_displacements_are_checked(
    #[case] address: u64,
    #[case] length: u64,
    #[case] advanced: Result<u64, FormatError>,
    #[case] retreated: Option<u64>,
) {
    let address = stored(address);
    assert_eq!(
        address.checked_add(length).map(StoredAddress::get),
        advanced
    );
    assert_eq!(
        address.checked_sub(length).map(StoredAddress::get),
        retreated
    );
}

#[rstest::rstest]
#[case::zero(0, 0, Some(0))]
#[case::equal(512, 512, Some(0))]
#[case::forward(608, 512, Some(96))]
#[case::reversed(512, 608, None)]
#[case::maximum(u64::MAX, 0, Some(u64::MAX))]
#[case::maximum_equal(u64::MAX, u64::MAX, Some(0))]
#[case::maximum_reversed(0, u64::MAX, None)]
fn physical_address_distances_require_ordered_endpoints(
    #[case] end: u64,
    #[case] start: u64,
    #[case] expected: Option<u64>,
) {
    assert_eq!(
        PhysicalAddress::new(end).checked_distance_from(PhysicalAddress::new(start)),
        expected
    );
    assert_eq!(
        PhysicalAddress::new(end).cmp(&PhysicalAddress::new(start)),
        end.cmp(&start)
    );
}

#[rstest::rstest]
#[case::zero(0, 0, Some(0))]
#[case::equal(512, 512, Some(0))]
#[case::forward(608, 512, Some(96))]
#[case::reversed(512, 608, None)]
#[case::maximum(u64::MAX, 0, Some(u64::MAX))]
#[case::maximum_equal(u64::MAX, u64::MAX, Some(0))]
#[case::maximum_reversed(0, u64::MAX, None)]
fn stored_address_distances_require_ordered_endpoints(
    #[case] end: u64,
    #[case] start: u64,
    #[case] expected: Option<u64>,
) {
    assert_eq!(stored(end).checked_distance_from(stored(start)), expected);
    assert_eq!(stored(end).cmp(&stored(start)), end.cmp(&start));
}

fn stored(value: u64) -> StoredAddress {
    use hdf5_pure_core::__private::StoredAddressExt;

    StoredAddress::new(value)
}
