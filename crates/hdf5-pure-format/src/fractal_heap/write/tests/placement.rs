use rstest::rstest;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapGeometry;
use crate::fractal_heap::write::ManagedHeapPlan;
use crate::width::OffsetWidth;

#[rstest]
#[case::too_wide_owner(OffsetWidth::Two, 0, 0x1_0000)]
#[case::undefined_owner(OffsetWidth::Four, 0, 0xffff_ffff)]
#[case::region_extent(OffsetWidth::Two, 0xff00, 0)]
#[case::overflowing_extent(OffsetWidth::Eight, u64::MAX - 10, 0)]
fn invalid_relocation_addresses_are_rejected(
    #[case] width: OffsetWidth,
    #[case] base: u64,
    #[case] owner: u64,
) {
    let plan = ManagedHeapPlan::attributes(&[1], width).unwrap();
    let error = plan
        .serialize(&[&[0]], StoredAddress::new(base), StoredAddress::new(owner))
        .unwrap_err();
    match (width, base, owner) {
        (OffsetWidth::Two, 0, _) => assert!(
            matches!(
                error,
                FormatError::AddressTooLarge {
                    address: 0x1_0000,
                    offset_size: 2
                }
            ),
            "{error:?}"
        ),
        (OffsetWidth::Four, _, _) => assert!(
            matches!(
                error,
                FormatError::DefinedAddressIsUndefined {
                    address: 0xffff_ffff,
                    offset_size: 4
                }
            ),
            "{error:?}"
        ),
        (OffsetWidth::Two, _, _) => assert!(
            matches!(
                error,
                FormatError::AddressTooLarge {
                    address: 66_304,
                    offset_size: 2
                }
            ),
            "{error:?}"
        ),
        _ => assert!(
            matches!(error, FormatError::OffsetOverflow { offset, length: 1024 } if offset == u64::MAX - 10),
            "{error:?}"
        ),
    }
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn undefined_roots_are_rejected_at_the_encoding_width(#[case] width: OffsetWidth) {
    let plan = ManagedHeapPlan::attributes(&[1], width).unwrap();
    let address = StoredAddress::undefined(width);
    let error = plan.root_address(address).unwrap_err();
    assert!(
        matches!(error, FormatError::DefinedAddressIsUndefined { address: value, offset_size } if value == address.get() && offset_size == width.get()),
        "{error:?}"
    );
}

#[test]
fn root_address_addition_is_checked_for_nested_indirect_blocks() {
    let geometry = FractalHeapGeometry::new(4, 64, 128, 16).unwrap();
    let plan = ManagedHeapPlan::new(&[100; 100], OffsetWidth::Eight, geometry).unwrap();
    let root_offset = plan.indirects.last().unwrap().region_offset;
    assert!(root_offset > 1);
    let error = plan
        .root_address(StoredAddress::new(u64::MAX - 1))
        .unwrap_err();
    assert!(
        matches!(error, FormatError::OffsetOverflow { offset, length } if offset == u64::MAX - 1 && length == root_offset),
        "{error:?}"
    );
}
