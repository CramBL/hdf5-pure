use alloc::vec;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::btree::v2::write::{BTREE_V2_NODE_SIZE, BTreeV2Plan, BTreeV2WriteDialect};
use crate::btree::v2::{self as btree_v2, BTreeV2Header};
use crate::error::FormatError;
use crate::link::dense::NAME_RECORD_LEN;
use crate::width::{LengthWidth, OffsetWidth};

const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
const LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;

#[test]
fn internal_child_record_counts_use_the_leaf_capacity_width() {
    let plan = BTreeV2Plan::new_with_undefined_empty_root(
        BTreeV2WriteDialect::Libhdf5Global,
        btree_v2::BTREE_V2_LINK_NAME,
        100_000,
        NAME_RECORD_LEN,
        4096,
        OFFSET_WIDTH,
    )
    .unwrap();
    assert!(plan.depth >= 2);
    assert!(plan.info.max_nrec(0) > u64::from(u8::MAX));
    assert!(plan.info.max_nrec(1) <= u64::from(u8::MAX));
    assert_eq!(plan.info.max_nrec_size(), 2);
    let records = vec![0; 100_000 * usize::from(NAME_RECORD_LEN)];
    let nodes_address =
        StoredAddress::new(btree_v2::btree_v2_header_size(OFFSET_WIDTH, LENGTH_WIDTH) as u64);
    let image = plan
        .serialize(&records, nodes_address, OFFSET_WIDTH, LENGTH_WIDTH)
        .unwrap();
    let bytes = [image.header, image.nodes].concat();
    let header = BTreeV2Header::parse(&bytes, 0, 8, 8).unwrap();
    let layout = btree_v2::BTreeV2Layout::new(&header, OFFSET_WIDTH).unwrap();
    let mut pending = vec![layout.root().unwrap()];
    let mut count = 0;
    while let Some(node) = pending.pop() {
        let view = node
            .parse(&bytes[usize::try_from(node.address().get()).unwrap()..])
            .unwrap();
        count += view.record_count();
        for index in 0..=view.record_count() {
            pending.extend(view.child(index).unwrap());
        }
    }
    assert_eq!(count, 100_000);
}

#[test]
fn no_node_exceeds_its_capacity_or_sits_empty() {
    for count in [1usize, 30, 569, 570, 10_260, 40_000] {
        let plan =
            BTreeV2Plan::new(8, count, 17, BTREE_V2_NODE_SIZE, OFFSET_WIDTH).expect("plannable");
        for node in &plan.nodes {
            assert!(
                !node.records.is_empty(),
                "empty node at depth {} for {count} records",
                node.depth
            );
            assert!(
                node.records.len() as u64 <= plan.info.max_nrec(node.depth),
                "node at depth {} holds {} records, capacity {}",
                node.depth,
                node.records.len(),
                plan.info.max_nrec(node.depth)
            );
            assert_eq!(
                node.children.len(),
                if node.depth == 0 {
                    0
                } else {
                    node.records.len() + 1
                },
                "a node's children must interleave with its records"
            );
        }
    }
}

#[test]
fn a_shape_needing_an_empty_node_has_no_plan() {
    // A 256-byte node holds one 200-byte record at every depth, so the counts are 1, 3, 7, 15,
    // and so on.
    assert!(BTreeV2Plan::new(8, 1_000, 200, 256, OFFSET_WIDTH).is_none());
    assert!(BTreeV2Plan::new(8, 4, 200, 256, OFFSET_WIDTH).is_none());
    assert!(BTreeV2Plan::new(8, 7, 200, 256, OFFSET_WIDTH).is_some());
}

#[test]
fn a_leaf_that_holds_more_than_u16_max_records_has_no_plan() {
    assert!(BTreeV2Plan::new(8, 100_000, 1, 1_000_000, OFFSET_WIDTH).is_none());
}

#[test]
fn record_preparation_rejects_an_overflowing_byte_length() {
    let mut plan = BTreeV2Plan::new(
        btree_v2::BTREE_V2_ATTRIBUTE_NAME,
        1,
        17,
        BTREE_V2_NODE_SIZE,
        OFFSET_WIDTH,
    )
    .unwrap();
    plan.total_records = u64::MAX / 17 + 1;
    let error = plan.prepare_records(&[]).err().unwrap();
    assert_eq!(
        error,
        FormatError::Internal("planned record buffer length exceeds u64::MAX".into())
    );
}
