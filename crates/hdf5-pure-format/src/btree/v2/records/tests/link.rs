use alloc::vec;

use rstest::rstest;
use test_util::btree::v2 as btree_v2;
use test_util::widths::Widths;

use super::*;
use crate::btree::v2::BTREE_V2_ATTRIBUTE_NAME;
use crate::btree::v2::BTreeV2Header;
use crate::btree::v2::BTreeV2Record;
use crate::error::FormatError;
use crate::width::OffsetWidth;

#[rstest]
#[case::short_hash(3, 7)]
#[case::short_id(10, 7)]
#[case::empty_id(11, 0)]
fn link_name_indexes_reject_invalid_record_widths(
    #[case] record_size: u16,
    #[case] heap_id_length: u16,
) {
    let bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, record_size, 256, 1).build(Widths::EIGHT);
    let header = BTreeV2Header::parse(&bytes, 0, 8, 8).unwrap();
    let Err(error) = BTreeV2LinkNameIndex::new(&header, OffsetWidth::Eight, heap_id_length) else {
        panic!("expected invalid name-index record width");
    };
    assert_eq!(error, FormatError::InvalidBTreeV2Signature);
}

#[rstest]
#[case::exact(11)]
#[case::extended(15)]
fn link_name_records_borrow_the_heap_id(#[case] record_size: u16) {
    let bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, record_size, 256, 1).build(Widths::EIGHT);
    let header = BTreeV2Header::parse(&bytes, 0, 8, 8).unwrap();
    let index = BTreeV2LinkNameIndex::new(&header, OffsetWidth::Eight, 7).unwrap();
    let record = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0];
    let (hash, id) = index.record(&record[..usize::from(record_size)]).unwrap();
    assert_eq!(hash, 0x0403_0201);
    assert_eq!(id, &[5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(id.as_ptr(), record[4..].as_ptr());
    assert_eq!(
        index.record(&record[..10]).unwrap_err(),
        FormatError::InvalidBTreeV2Signature
    );
}

#[rstest]
#[case::unknown_client(0xFF, 7, FormatError::InvalidBTreeNodeType(0xFF))]
#[case::attribute_client(
    BTREE_V2_ATTRIBUTE_NAME,
    7,
    FormatError::InvalidBTreeNodeType(BTREE_V2_ATTRIBUTE_NAME)
)]
#[case::zero_heap_id(BTREE_V2_LINK_NAME, 0, FormatError::InvalidBTreeV2Signature)]
#[case::unrepresentable_name_record(
    BTREE_V2_LINK_NAME,
    u16::MAX,
    FormatError::InvalidBTreeV2Signature
)]
#[case::unrepresentable_creation_record(
    BTREE_V2_LINK_CREATION_ORDER,
    u16::MAX,
    FormatError::InvalidBTreeV2Signature
)]
fn link_record_contexts_reject_invalid_encodings(
    #[case] tree_type: u8,
    #[case] heap_id_length: u16,
    #[case] expected: FormatError,
) {
    assert_eq!(
        BTreeV2LinkRecordContext::new(tree_type, heap_id_length).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::short_hash(BTREE_V2_LINK_NAME, 3)]
#[case::short_name_id(BTREE_V2_LINK_NAME, 10)]
#[case::short_creation_order(BTREE_V2_LINK_CREATION_ORDER, 7)]
#[case::short_creation_id(BTREE_V2_LINK_CREATION_ORDER, 14)]
fn link_record_views_require_complete_keys_and_ids(#[case] tree_type: u8, #[case] length: usize) {
    let context = BTreeV2LinkRecordContext::new(tree_type, 7).unwrap();
    assert_eq!(
        context.record(&vec![0; length]).unwrap_err(),
        FormatError::InvalidBTreeV2Signature
    );
}

#[rstest]
#[case::name(BTREE_V2_LINK_NAME, NAME_HASH_LEN)]
#[case::creation_order(BTREE_V2_LINK_CREATION_ORDER, u16::try_from(LINK_CREATION_ORDER_LEN).unwrap())]
fn link_record_contexts_accept_the_largest_representable_record(
    #[case] tree_type: u8,
    #[case] prefix: u16,
) {
    let length = u16::MAX - prefix;
    let context = BTreeV2LinkRecordContext::new(tree_type, length).unwrap();
    context.require_record_size(u16::MAX).unwrap();
    let bytes = vec![0; usize::from(u16::MAX)];
    let record = context.record(&bytes).unwrap();
    assert_eq!(record.heap_id().len(), usize::from(length));
    assert_eq!(
        record.heap_id().as_ptr(),
        bytes[usize::from(prefix)..].as_ptr()
    );
    assert_eq!(
        BTreeV2LinkRecordContext::new(tree_type, length + 1).unwrap_err(),
        FormatError::InvalidBTreeV2Signature
    );
}

#[rstest]
#[case::name(BTREE_V2_LINK_NAME, usize::from(NAME_HASH_LEN))]
#[case::creation_order(BTREE_V2_LINK_CREATION_ORDER, LINK_CREATION_ORDER_LEN)]
fn link_record_views_borrow_ids_and_allow_trailing_fields(
    #[case] tree_type: u8,
    #[case] prefix: usize,
) {
    let context = BTreeV2LinkRecordContext::new(tree_type, 7).unwrap();
    let mut bytes = vec![0; prefix];
    let id = [0, 1, 2, 3, 4, 5, 6];
    bytes.extend_from_slice(&id);
    bytes.push(99);
    let record = context.record(&bytes).unwrap();
    assert_eq!(record.heap_id(), id);
    assert_eq!(record.heap_id().as_ptr(), bytes[prefix..].as_ptr());
}

#[rstest]
#[case::a_name_index(BTREE_V2_LINK_NAME, &[0xA, 0xB, 0xC, 0xD, 1, 2, 3, 4, 5, 6, 7], Some(&[1, 2, 3, 4, 5, 6, 7][..]))]
#[case::a_creation_order_index(6, &[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7], Some(&[1, 2, 3, 4, 5, 6, 7][..]))]
#[case::a_record_shorter_than_its_heap_id(BTREE_V2_LINK_NAME, &[0xA, 0xB, 0xC, 0xD, 1, 2], None)]
fn a_link_record_yields_the_heap_id_past_its_key(
    #[case] tree_type: u8,
    #[case] data: &[u8],
    #[case] heap_id: Option<&[u8]>,
) {
    let record = BTreeV2Record {
        data: data.to_vec(),
    };
    match heap_id {
        Some(expected) => assert_eq!(record.link_heap_id(tree_type, 7).unwrap(), expected),
        None => assert_eq!(
            record.link_heap_id(tree_type, 7).unwrap_err(),
            FormatError::InvalidBTreeV2Signature
        ),
    }
}
