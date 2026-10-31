use hdf5_pure::{
    File, FileBuilder, FormatError, OffsetWidth, PhysicalAddress, StoredAddress, UserblockSize,
};
use rstest::rstest;

#[rstest]
#[case::zero(0, 0)]
#[case::userblock(512, 512)]
#[case::forward(512, 1024)]
#[case::backward(512, 0)]
#[case::inserted_userblock(0, 512)]
fn file_address_conversions_use_the_opened_signature(
    #[case] encoded_base: usize,
    #[case] signature: usize,
) {
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(u64::try_from(encoded_base).unwrap()).unwrap());
    builder.create_dataset("data").with_i32_data(&[17, 29]);
    let bytes = builder.finish().unwrap();
    let mut relocated = vec![0; signature];
    relocated.extend_from_slice(&bytes[encoded_base..]);
    let file = File::from_bytes(relocated.clone()).unwrap();
    assert_eq!(
        file.superblock().base_address.get(),
        u64::try_from(encoded_base).unwrap()
    );
    let metadata = test_util_hdf5::storage::file_metadata_extents(&file, &relocated);
    test_util_hdf5::storage::assert_unavailable(&file, &metadata);
    let stored = file.superblock().root_group_address;
    let expected = PhysicalAddress::new(u64::try_from(signature).unwrap() + stored.get());
    assert_eq!(file.resolve_address(stored).unwrap(), expected);
    assert_eq!(file.relative_address(expected).unwrap(), stored);
    let origin = file
        .relative_address(PhysicalAddress::new(u64::try_from(signature).unwrap()))
        .unwrap();
    assert_eq!(origin.get(), 0);
    assert_eq!(
        file.resolve_address(origin).unwrap(),
        PhysicalAddress::new(u64::try_from(signature).unwrap())
    );
    if signature > 0 {
        let address = u64::try_from(signature - 1).unwrap();
        assert_eq!(
            file.relative_address(PhysicalAddress::new(address))
                .unwrap_err(),
            FormatError::AddressBelowBase {
                address,
                base: u64::try_from(signature).unwrap()
            }
        );
        let stored = origin.checked_add(u64::MAX - 1).unwrap();
        assert_eq!(
            file.resolve_address(stored).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX - 1,
                length: u64::try_from(signature).unwrap()
            }
        );
    }
    let last = PhysicalAddress::new(u64::MAX - 1);
    let stored_last = file.relative_address(last).unwrap();
    assert_eq!(file.resolve_address(stored_last).unwrap(), last);
    let undefined = StoredAddress::undefined(file.superblock().offset_size);
    assert_eq!(
        file.resolve_address(undefined).unwrap_err(),
        FormatError::DefinedAddressIsUndefined {
            address: undefined.get(),
            offset_size: file.superblock().offset_size.get(),
        }
    );
    let narrow_marker = StoredAddress::undefined(OffsetWidth::Two);
    assert_eq!(
        file.resolve_address(narrow_marker).unwrap(),
        PhysicalAddress::new(u64::try_from(signature).unwrap() + 0xffff)
    );
    if signature == 0 {
        assert_eq!(
            file.relative_address(PhysicalAddress::new(u64::MAX))
                .unwrap_err(),
            FormatError::DefinedAddressIsUndefined {
                address: u64::MAX,
                offset_size: file.superblock().offset_size.get(),
            }
        );
    }
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [17, 29]);
}
