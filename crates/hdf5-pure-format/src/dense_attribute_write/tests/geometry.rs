use rstest::rstest;

use crate::__private as format_boundary;
use crate::btree::v2::HugeObjectRecord;
use crate::btree::v2::write::BTREE_V2_NODE_SIZE;
use crate::dense_attribute_write::plan::{DENSE_ATTR_LENGTH_WIDTH, DENSE_ATTR_OFFSET_WIDTH};
use crate::dense_attribute_write::{
    DenseAttrCreationOrder, PreparedDenseAttrs, plan, test_support,
};
use crate::error::FormatError;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[test]
fn a_fixed_node_size_keeps_the_derived_count_width_at_one_byte() {
    for record_size in [
        super::DENSE_ATTR_BTREE_RECORD,
        HugeObjectRecord::size(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH),
    ] {
        let (info, depth) = format_boundary::BTreeV2NodeInfo::for_record_count(
            BTREE_V2_NODE_SIZE,
            record_size,
            format_boundary::OffsetWidth::Eight,
            0,
        )
        .expect("the emitted geometry is plannable");
        assert_eq!(depth, 0);
        let capacity = info.max_nrec(0);
        assert!(
            capacity >= 20,
            "a {record_size}-byte record should leave room for 20 per node, got {capacity}"
        );
        assert_eq!(
            plan::encoded_size_width(capacity),
            1,
            "a {record_size}-byte record's leaf capacity must stay inside one byte"
        );
    }
}

#[rstest]
fn dense_attribute_huge_totals_fit_the_selected_length_width(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
) {
    let attrs = [
        test_support::canonical_attribute("a", 33_000),
        test_support::canonical_attribute("b", 33_000),
    ];
    let total = attrs
        .iter()
        .map(|attribute| attribute.serialize_v3(offsets, 2).unwrap().len() as u64)
        .sum::<u64>();
    assert!(total > u64::from(u16::MAX));
    let error = PreparedDenseAttrs::with_widths(
        FormatWidths {
            offsets,
            lengths: LengthWidth::Two,
        },
        &attrs,
        DenseAttrCreationOrder::Untracked,
        [],
    )
    .err()
    .unwrap();
    assert_eq!(
        error,
        FormatError::LengthTooLarge {
            length: total,
            length_size: 2
        }
    );
}
