use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;
use test_util::btree::v2 as btree_v2;
use test_util::widths::Widths;

use super::*;
use crate::btree::v2::BTREE_V2_LINK_NAME;
use crate::btree::v2::BTreeV2Record;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::FractalHeapIdKind;
use crate::fractal_heap::FractalHeapIdLayout;
use crate::message_flags::MessageFlags;
use crate::shared_message::FHEAP_ID_LEN;
use crate::width::{LengthWidth, OffsetWidth};

// "Fractal Heap" of the format specification, version 4.0, defines tiny heap ID type 2
// in bits 4 and 5. Zero length bits encode a one-byte object.
const TINY_ID_TAG: u8 = 0x20;

fn attribute_id_layout() -> FractalHeapIdLayout {
    let mut header = FractalHeapHeader::parse(
        &test_util::fractal_heap::Header::new(256).build(Widths::EIGHT),
        0,
        8,
        8,
    )
    .unwrap();
    header.heap_id_length = 8;
    FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
}

#[rstest]
#[case::name(BTREE_V2_ATTRIBUTE_NAME)]
#[case::creation_order(BTREE_V2_ATTRIBUTE_CREATION_ORDER)]
fn dense_attribute_targets_keep_shared_ids_out_of_the_private_parser(#[case] tree_type: u8) {
    let id = [0xFF; FHEAP_ID_LEN];
    let mut bytes = btree_v2::attribute_name_record(&id, MessageFlags::SHARED.get(), 42, 9);
    if tree_type == BTREE_V2_ATTRIBUTE_CREATION_ORDER {
        bytes.truncate(13);
    }
    let record = DenseAttributeRecord::parse(&bytes, tree_type, 8).unwrap();
    assert_eq!(record.flags(), MessageFlags::SHARED);
    assert_eq!(record.creation_order(), 42);
    let DenseAttributeTarget::Shared(target) = record.target(&attribute_id_layout()).unwrap()
    else {
        panic!("expected a shared target")
    };
    assert_eq!(target.bytes(), &id);
}

#[rstest]
#[case::name(BTREE_V2_ATTRIBUTE_NAME)]
#[case::creation_order(BTREE_V2_ATTRIBUTE_CREATION_ORDER)]
fn dense_attribute_private_targets_contain_parsed_ids(#[case] tree_type: u8) {
    let id = [TINY_ID_TAG, b'a', 0, 0, 0, 0, 0, 0];
    let mut bytes = btree_v2::attribute_name_record(&id, 0, 42, 9);
    if tree_type == BTREE_V2_ATTRIBUTE_CREATION_ORDER {
        bytes.truncate(13);
    }
    let record = DenseAttributeRecord::parse(&bytes, tree_type, 8).unwrap();
    let DenseAttributeTarget::Private(target) = record.target(&attribute_id_layout()).unwrap()
    else {
        panic!("expected a private target")
    };
    assert_eq!(target.id().kind(), FractalHeapIdKind::Tiny { bytes: b"a" });
}

#[rstest]
#[case::short_id(BTREE_V2_ATTRIBUTE_NAME, 7, 17)]
#[case::missing_flags(BTREE_V2_ATTRIBUTE_NAME, 8, 17)]
#[case::short_creation_order(BTREE_V2_ATTRIBUTE_CREATION_ORDER, 12, 13)]
#[case::short_name_hash(BTREE_V2_ATTRIBUTE_NAME, 16, 17)]
fn dense_attribute_record_fields_must_be_complete(
    #[case] tree_type: u8,
    #[case] len: usize,
    #[case] required: usize,
) {
    let bytes = vec![0; len];
    assert_eq!(
        DenseAttributeRecord::parse(&bytes, tree_type, 8).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: required,
            available: len
        }
    );
}

#[rstest]
#[case::short(7)]
#[case::long(9)]
fn dense_shared_targets_require_eight_byte_ids(#[case] len: u16) {
    let bytes = btree_v2::attribute_name_record(
        &vec![0; usize::from(len)],
        MessageFlags::SHARED.get(),
        0,
        0,
    );
    let record = DenseAttributeRecord::parse(&bytes, BTREE_V2_ATTRIBUTE_NAME, len).unwrap();
    assert_eq!(
        record.target(&attribute_id_layout()).unwrap_err(),
        FormatError::InvalidFractalHeapIdLength {
            expected: FHEAP_ID_LEN,
            actual: usize::from(len)
        }
    );
}

#[test]
fn dense_private_targets_reject_an_unknown_heap_id_type() {
    let bytes = btree_v2::attribute_name_record(&[0x30, 0, 0, 0, 0, 0, 0, 0], 0, 0, 0);
    let record = DenseAttributeRecord::parse(&bytes, BTREE_V2_ATTRIBUTE_NAME, 8).unwrap();
    assert_eq!(
        record.target(&attribute_id_layout()).unwrap_err(),
        FormatError::InvalidHeapIdType(3)
    );
}

#[test]
fn dense_attribute_records_reject_other_tree_types() {
    assert_eq!(
        DenseAttributeRecord::parse(&[0; 17], BTREE_V2_LINK_NAME, 8).unwrap_err(),
        FormatError::InvalidBTreeNodeType(BTREE_V2_LINK_NAME)
    );
}

#[test]
fn dense_private_targets_reject_ids_of_a_different_length_from_the_heap() {
    let bytes = btree_v2::attribute_name_record(&[0; 7], 0, 0, 0);
    let record = DenseAttributeRecord::parse(&bytes, BTREE_V2_ATTRIBUTE_NAME, 7).unwrap();
    assert_eq!(
        record.target(&attribute_id_layout()).unwrap_err(),
        FormatError::InvalidFractalHeapIdLength {
            expected: 8,
            actual: 7
        }
    );
}

#[test]
fn an_attribute_name_record_holds_its_heap_id_flags_creation_order_and_hash() {
    let mut data = Vec::new();
    AttributeRecord {
        heap_id: &[1, 2, 3, 4, 5, 6, 7, 8],
        flags: MessageFlags::CONSTANT,
        creation_order: 0x0A0B,
    }
    .encode_name_record(&mut data, 0x1122_3344);

    assert_eq!(
        data,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 0x01, 0x0B, 0x0A, 0, 0, 0x44, 0x33, 0x22, 0x11
        ]
    );
    assert_eq!(
        data.len(),
        usize::from(AttributeRecord::name_record_size(8))
    );
    let record = BTreeV2Record { data };
    assert_eq!(
        record.attribute_heap_id(8),
        Some(&[1, 2, 3, 4, 5, 6, 7, 8][..])
    );
    assert_eq!(record.attribute_creation_order(8), Some(0x0A0B));
}

#[test]
fn an_attribute_creation_order_record_omits_the_hash() {
    let mut data = Vec::new();
    AttributeRecord {
        heap_id: &[9; 8],
        flags: MessageFlags::NONE,
        creation_order: 7,
    }
    .encode_creation_order_record(&mut data);

    assert_eq!(data, [9, 9, 9, 9, 9, 9, 9, 9, 0, 7, 0, 0, 0]);
    assert_eq!(
        data.len(),
        usize::from(AttributeRecord::creation_order_record_size(8))
    );
}

#[rstest]
#[case::without_a_creation_order(vec![1, 2, 3, 4, 5, 6, 7, 8, 0], Some(&[1, 2, 3, 4, 5, 6, 7, 8][..]), None)]
#[case::shorter_than_a_heap_id(vec![1, 2, 3], None, None)]
fn a_short_attribute_record_yields_the_fields_it_holds(
    #[case] data: Vec<u8>,
    #[case] heap_id: Option<&[u8]>,
    #[case] creation_order: Option<u32>,
) {
    let record = BTreeV2Record { data };
    assert_eq!(record.attribute_heap_id(8), heap_id);
    assert_eq!(record.attribute_creation_order(8), creation_order);
}
