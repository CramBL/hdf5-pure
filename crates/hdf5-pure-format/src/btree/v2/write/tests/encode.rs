use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::btree::v2::write::{BTREE_V2_NODE_SIZE, BTreeV2Plan, BTreeV2WriteDialect};
use crate::btree::v2::{self as btree_v2, BTreeV2Header};
use crate::error::FormatError;
use crate::link::dense::NAME_RECORD_LEN;
use crate::width::{LengthWidth, OffsetWidth};

const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
const LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;

#[test]
fn empty_tree_constructors_use_distinct_root_representations() {
    let legacy = BTreeV2Plan::new(
        btree_v2::BTREE_V2_LINK_NAME,
        0,
        NAME_RECORD_LEN,
        BTREE_V2_NODE_SIZE,
        OFFSET_WIDTH,
    )
    .unwrap();
    let canonical = BTreeV2Plan::new_with_undefined_empty_root(
        BTreeV2WriteDialect::Libhdf5Global,
        btree_v2::BTREE_V2_LINK_NAME,
        0,
        NAME_RECORD_LEN,
        BTREE_V2_NODE_SIZE,
        OFFSET_WIDTH,
    )
    .unwrap();
    let legacy = legacy
        .serialize(&[], StoredAddress::new(1024), OFFSET_WIDTH, LENGTH_WIDTH)
        .unwrap();
    let canonical = canonical
        .serialize(
            &[],
            StoredAddress::undefined(OFFSET_WIDTH),
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        )
        .unwrap();
    let legacy_header = BTreeV2Header::parse(&legacy.header, 0, 8, 8).unwrap();
    let canonical_header = BTreeV2Header::parse(&canonical.header, 0, 8, 8).unwrap();
    assert_eq!(legacy_header.root_node_address, StoredAddress::new(1024));
    assert_eq!(legacy.nodes.len(), BTREE_V2_NODE_SIZE as usize);
    assert_eq!(
        canonical_header.root_node_address,
        StoredAddress::undefined(OFFSET_WIDTH)
    );
    assert_eq!(canonical.nodes, Vec::<u8>::new());
    assert_eq!(canonical_header.depth, 0);
    assert_eq!(canonical_header.num_records_in_root, 0);
    assert_eq!(canonical_header.total_records, 0);
}

#[test]
fn nodes_are_all_one_node_size_long() {
    let plan = BTreeV2Plan::new(8, 5_000, 17, BTREE_V2_NODE_SIZE, OFFSET_WIDTH).expect("plannable");
    let image = plan
        .serialize(
            &vec![0; 5_000 * 17],
            StoredAddress::new(4_096),
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        )
        .unwrap();
    assert_eq!(image.nodes.len() as u64, plan.nodes_size());
    assert_eq!(image.nodes.len() % BTREE_V2_NODE_SIZE as usize, 0);
}

#[test]
fn serialization_rejects_a_record_buffer_of_another_length() {
    let plan = BTreeV2Plan::new(8, 2, 17, BTREE_V2_NODE_SIZE, OFFSET_WIDTH).expect("plannable");
    assert_eq!(
        plan.serialize(&[0; 35], StoredAddress::new(0), OFFSET_WIDTH, LENGTH_WIDTH)
            .map(|_| ())
            .unwrap_err(),
        FormatError::Internal("record buffer length differs from the planned records".into())
    );
}

#[test]
fn a_planned_tree_parses_back_to_its_header_and_records() {
    let records: Vec<u8> = (0..34).collect();
    let nodes_address = StoredAddress::new(crate::btree::v2::btree_v2_header_size(
        OFFSET_WIDTH,
        LENGTH_WIDTH,
    ) as u64);
    let plan = BTreeV2Plan::new(8, 2, 17, BTREE_V2_NODE_SIZE, OFFSET_WIDTH).unwrap();
    let image = plan
        .serialize(&records, nodes_address, OFFSET_WIDTH, LENGTH_WIDTH)
        .unwrap();

    assert_eq!(
        BTreeV2Header::parse(&image.header, 0, 8, 8),
        Ok(BTreeV2Header {
            tree_type: 8,
            node_size: BTREE_V2_NODE_SIZE,
            record_size: 17,
            depth: 0,
            root_node_address: nodes_address,
            num_records_in_root: 2,
            total_records: 2,
        })
    );
    assert_eq!(image.nodes.len(), BTREE_V2_NODE_SIZE as usize);
    let header = BTreeV2Header::parse(&image.header, 0, 8, 8).unwrap();
    let layout = btree_v2::BTreeV2Layout::new(&header, OFFSET_WIDTH).unwrap();
    let view = layout.root().unwrap().parse(&image.nodes).unwrap();
    assert_eq!(view.record_count(), 2);
    assert_eq!(view.record(0).unwrap(), &records[..17]);
    assert_eq!(view.record(1).unwrap(), &records[17..]);
}
#[rstest]
#[case::origin(0)]
#[case::maximum_end(u64::MAX - 512)]
fn serialization_preserves_representable_node_ranges(#[case] start: u64) {
    let plan = BTreeV2Plan::new(8, 2, 17, 512, OFFSET_WIDTH).unwrap();
    let image = plan
        .serialize(
            &[0; 34],
            StoredAddress::new(start),
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        )
        .unwrap();
    let header = BTreeV2Header::parse(&image.header, 0, 8, 8).unwrap();
    assert_eq!(header.root_node_address, StoredAddress::new(start));
    assert_eq!(header.total_records, 2);
    assert_eq!(image.nodes.len(), 512);
}

#[test]
fn serialization_rejects_an_overflowing_node_range() {
    let plan = BTreeV2Plan::new(8, 2, 17, 512, OFFSET_WIDTH).unwrap();
    let start = u64::MAX - 511;
    let error = plan
        .serialize(
            &[0; 34],
            StoredAddress::new(start),
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        )
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::OffsetOverflow {
            offset: start,
            length: 512
        }
    );
}
