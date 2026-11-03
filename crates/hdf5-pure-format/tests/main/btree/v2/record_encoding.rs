use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::BTreeV2Record;
use hdf5_pure_format::__private::HugeObjectRecord;
use rstest::rstest;

#[rstest]
#[case::address_two(OffsetWidth::Two, LengthWidth::Eight, 65_536, 17, 3, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::address_four(OffsetWidth::Four, LengthWidth::Eight, 0x1_0000_0000, 17, 3, FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 })]
#[case::undefined_two(OffsetWidth::Two, LengthWidth::Eight, 65_535, 17, 3, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::undefined_four(OffsetWidth::Four, LengthWidth::Eight, 0xFFFF_FFFF, 17, 3, FormatError::DefinedAddressIsUndefined { address: 0xFFFF_FFFF, offset_size: 4 })]
#[case::undefined_eight(OffsetWidth::Eight, LengthWidth::Eight, u64::MAX, 17, 3, FormatError::DefinedAddressIsUndefined { address: u64::MAX, offset_size: 8 })]
#[case::length_two(OffsetWidth::Eight, LengthWidth::Two, 1024, 65_536, 3, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
#[case::length_four(OffsetWidth::Eight, LengthWidth::Four, 1024, 0x1_0000_0000, 3, FormatError::LengthTooLarge { length: 0x1_0000_0000, length_size: 4 })]
#[case::id_two(OffsetWidth::Eight, LengthWidth::Two, 1024, 17, 65_536, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
#[case::id_four(OffsetWidth::Eight, LengthWidth::Four, 1024, 17, 0x1_0000_0000, FormatError::LengthTooLarge { length: 0x1_0000_0000, length_size: 4 })]
fn huge_record_rejection_preserves_the_destination(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] address: u64,
    #[case] length: u64,
    #[case] id: u64,
    #[case] expected: FormatError,
) {
    let record = HugeObjectRecord {
        address: StoredAddress::new(address),
        length,
        id,
    };
    let mut bytes = vec![0xAB, 0xCD];
    assert_eq!(
        record.encode(&mut bytes, offsets, lengths).unwrap_err(),
        expected
    );
    assert_eq!(bytes, [0xAB, 0xCD]);
}

#[rstest]
#[case::address_limit(true)]
#[case::length_and_id_limit(false)]
fn huge_records_preserve_each_representable_field(
    #[case] address_limit: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
) {
    let maximum_length = u64::MAX >> (64 - 8 * lengths.get());
    let record = HugeObjectRecord {
        address: StoredAddress::new(if address_limit {
            StoredAddress::undefined(offsets).get() - 1
        } else {
            0
        }),
        length: if address_limit { 1 } else { maximum_length },
        id: maximum_length,
    };
    let mut bytes = vec![0xAB, 0xCD];
    record.encode(&mut bytes, offsets, lengths).unwrap();
    let mut expected = vec![0xAB, 0xCD];
    expected.extend_from_slice(&record.address.get().to_le_bytes()[..usize::from(offsets.get())]);
    expected.extend_from_slice(&record.length.to_le_bytes()[..usize::from(lengths.get())]);
    expected.extend_from_slice(&record.id.to_le_bytes()[..usize::from(lengths.get())]);
    assert_eq!(bytes, expected);
    assert_eq!(
        BTreeV2Record {
            data: bytes[2..].to_vec()
        }
        .huge_object(offsets, lengths)
        .unwrap(),
        record
    );
}
