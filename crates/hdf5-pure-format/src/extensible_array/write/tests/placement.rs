use alloc::vec::Vec;

use hdf5_pure_core::__private::{BaseAddressExt, StoredAddressExt};
use hdf5_pure_core::BaseAddress;
use rstest::rstest;

use super::*;
use crate::extensible_array::write::test_support;

#[rstest]
#[case::ascending(8, &[1_000, 2_000, 3_000], 1_u64 << 48)]
#[case::reverse(8, &[3_000, 2_000, 1_000], 0)]
#[case::adjacent(2, &[0, 66, 220], 512)]
#[case::exclusive_end(2, &[0, 1_000, 65_535 - 48], 0)]
fn placement_preserves_independent_complete_ranges(
    #[case] width: u8,
    #[case] starts: &[u64],
    #[case] base: u64,
) {
    let planned = test_support::plan(&[32, 4, 4, 16, 10], &[4], 5, width);
    let addresses: Vec<_> = starts.iter().copied().map(StoredAddress::new).collect();
    let placed = planned.place(&addresses).unwrap();
    assert!(core::ptr::eq(placed.plan(), &planned));
    let actual: Vec<_> = placed
        .blocks()
        .map(|(address, length)| (address.get(), length))
        .collect();
    assert_eq!(
        actual,
        starts
            .iter()
            .copied()
            .zip(planned.block_lengths())
            .collect::<Vec<_>>()
    );
    let base = BaseAddress::new(base);
    for (stored, length) in placed.blocks() {
        let physical = base.resolve(stored).unwrap();
        assert_eq!(physical.get(), base.get() + stored.get());
        assert_eq!(base.relative(physical).unwrap(), stored);
        assert_eq!(
            base.resolve(StoredAddress::new(stored.get() + length))
                .unwrap()
                .get(),
            physical.get() + length
        );
    }
}

#[rstest]
#[case::undefined(2, &[65_535, 100, 1_000], 65_535, true)]
#[case::wide_start(2, &[65_536, 100, 1_000], 65_536, false)]
#[case::wide_end(2, &[65_530, 100, 1_000], 65_596, false)]
fn placement_rejects_unencodable_addresses(
    #[case] width: u8,
    #[case] starts: &[u64],
    #[case] expected: u64,
    #[case] undefined: bool,
) {
    let planned = test_support::plan(&[32, 4, 4, 16, 10], &[4], 5, width);
    let addresses: Vec<_> = starts.iter().copied().map(StoredAddress::new).collect();
    let error = planned.place(&addresses).unwrap_err();
    if undefined {
        assert!(
            matches!(error, FormatError::DefinedAddressIsUndefined { address, offset_size } if address == expected && offset_size == width),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(error, FormatError::AddressTooLarge { address, offset_size } if address == expected && offset_size == width),
            "{error:?}"
        );
    }
}

#[test]
fn placement_rejects_count_overlap_and_overflow() {
    let planned = test_support::plan(&[32, 4, 4, 16, 10], &[4], 5, 8);
    let error = planned.place(&[]).unwrap_err();
    assert!(
        matches!(
            error,
            FormatError::ExtensibleArrayBlockCount {
                expected: 3,
                actual: 0
            }
        ),
        "{error:?}"
    );
    let error = planned
        .place(&[
            StoredAddress::new(1_000),
            StoredAddress::new(0),
            StoredAddress::new(1_001),
        ])
        .unwrap_err();
    assert!(
        matches!(
            error,
            FormatError::ExtensibleArrayBlockOverlap {
                first: 0,
                second: 2
            }
        ),
        "{error:?}"
    );
    let error = planned
        .place(&[
            StoredAddress::new(u64::MAX - 1),
            StoredAddress::new(0),
            StoredAddress::new(1_000),
        ])
        .unwrap_err();
    assert!(
        matches!(error, FormatError::OffsetOverflow { offset, length: 72 } if offset == u64::MAX - 1),
        "{error:?}"
    );
}
