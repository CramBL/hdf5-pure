use alloc::vec;

use rstest::rstest;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_message::AttributeMessage;
use crate::dense_attribute_write::{DenseAttrCreationOrder, PreparedDenseAttrs, test_support};
use crate::error::FormatError;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[rstest]
#[case::undefined(true)]
#[case::end(false)]
fn dense_attribute_plan_rejects_invalid_extent(
    #[case] undefined: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(false, true)] huge: bool,
) {
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Two,
    };
    let attrs = [if huge {
        test_support::canonical_attribute("huge", 33_000)
    } else {
        test_support::metadata_attribute("a", 1)
    }];
    let plan =
        PreparedDenseAttrs::with_widths(widths, &attrs, DenseAttrCreationOrder::Untracked, [])
            .unwrap()
            .finalize(
                |reference| panic!("unexpected reference {reference:?}"),
                |attribute| panic!("unexpected payload {attribute}"),
            )
            .unwrap();
    let maximum = StoredAddress::undefined(offsets).get();
    let base = if undefined {
        maximum
    } else {
        maximum - plan.blob_len() + 1
    };
    let expected = if undefined {
        FormatError::DefinedAddressIsUndefined {
            address: base,
            offset_size: offsets.get(),
        }
    } else if offsets == OffsetWidth::Eight {
        FormatError::OffsetOverflow {
            offset: base,
            length: plan.blob_len(),
        }
    } else {
        FormatError::AddressTooLarge {
            address: maximum + 1,
            offset_size: offsets.get(),
        }
    };
    assert_eq!(
        plan.attribute_info_for_heap(StoredAddress::new(base))
            .unwrap_err(),
        expected
    );
    assert_eq!(
        plan.build(StoredAddress::new(base)).err().unwrap(),
        expected
    );
}

#[rstest]
#[case::empty(vec![])]
#[case::managed(vec![test_support::metadata_attribute("a", 1)])]
#[case::huge(vec![test_support::canonical_attribute("huge", 70_000)])]
fn dense_attribute_plan_fills_a_valid_extent_at_the_address_limit(
    #[case] attrs: Vec<AttributeMessage>,
    #[values(OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
) {
    assert_valid_extent(
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        &attrs,
    );
}

#[rstest]
#[case::empty(vec![])]
#[case::managed(vec![test_support::metadata_attribute("a", 1)])]
#[case::huge(vec![test_support::canonical_attribute("huge", 33_000)])]
fn dense_attribute_plan_fills_a_two_byte_extent_at_the_address_limit(
    #[case] attrs: Vec<AttributeMessage>,
) {
    assert_valid_extent(
        FormatWidths {
            offsets: OffsetWidth::Two,
            lengths: LengthWidth::Two,
        },
        &attrs,
    );
}

fn assert_valid_extent(widths: FormatWidths, attrs: &[AttributeMessage]) {
    let plan =
        PreparedDenseAttrs::with_widths(widths, attrs, DenseAttrCreationOrder::Untracked, [])
            .unwrap()
            .finalize(
                |reference| panic!("unexpected reference {reference:?}"),
                |attribute| panic!("unexpected payload {attribute}"),
            )
            .unwrap();
    let maximum = StoredAddress::undefined(widths.offsets).get();
    let base = StoredAddress::new(maximum - plan.blob_len());
    let built = plan.build(base).unwrap();
    assert_eq!(built.blob.len() as u64, plan.blob_len());
    assert_eq!(
        built.attribute_info,
        plan.attribute_info_for_heap(base).unwrap()
    );
    let info = AttributeInfoMessage::parse_with_offset_width(
        &built.attribute_info.to_bytes(),
        widths.offsets,
    )
    .unwrap();
    assert_eq!(info.fractal_heap_address(), Some(base));
    assert!(info.btree_name_index_address().unwrap().get() < maximum);
}
