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
