use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU64;

use test_util::bytes as test_bytes;

use super::*;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2::BTREE_V2_ATTRIBUTE_NAME;
use crate::btree::v2::BTreeV2Record;
use crate::data_layout::LayoutVersion;
use crate::error::FormatError;
use crate::width::OffsetWidth;

#[test]
fn type_10_chunk_records_decode_scaled_offsets() {
    let context = BTreeV2ChunkRecordContext::new(
        BTREE_V2_CHUNK,
        OffsetWidth::Four,
        3,
        NonZeroU64::new(80).unwrap(),
        LayoutVersion::Four,
    )
    .unwrap();
    assert_eq!(context.record_size(), 4 + 3 * 8);

    let mut data = Vec::new();
    test_bytes::push_address(&mut data, Some(0x1020_3040), 4);
    test_bytes::push_uint(&mut data, 2, 8);
    test_bytes::push_uint(&mut data, 7, 8);
    test_bytes::push_uint(&mut data, 11, 8);
    let record = context.decode(&BTreeV2Record { data }).unwrap();

    assert_eq!(record.address, StoredAddress::new(0x1020_3040));
    assert_eq!(record.stored_size, 80);
    assert_eq!(record.filter_mask, 0);
    assert_eq!(record.scaled_offsets, vec![2, 7, 11]);
}

#[test]
fn type_10_chunk_records_support_8_byte_offsets_and_rank_1() {
    let context = BTreeV2ChunkRecordContext::new(
        BTREE_V2_CHUNK,
        OffsetWidth::Eight,
        1,
        NonZeroU64::new(32).unwrap(),
        LayoutVersion::Four,
    )
    .unwrap();
    assert_eq!(context.record_size(), 16);

    let mut data = Vec::new();
    test_bytes::push_address(&mut data, Some(0x1020_3040_5060_7080), 8);
    test_bytes::push_uint(&mut data, 9, 8);
    let record = context.decode(&BTreeV2Record { data }).unwrap();

    assert_eq!(record.address, StoredAddress::new(0x1020_3040_5060_7080));
    assert_eq!(record.scaled_offsets, vec![9]);
}

#[test]
fn type_11_chunk_records_decode_size_mask_and_scaled_offsets() {
    // An 80-byte logical chunk uses a 2-byte stored-size field in a version 4 layout.
    let context = BTreeV2ChunkRecordContext::new(
        BTREE_V2_FILTERED_CHUNK,
        OffsetWidth::Eight,
        2,
        NonZeroU64::new(80).unwrap(),
        LayoutVersion::Four,
    )
    .unwrap();
    assert_eq!(context.record_size(), 8 + 2 + 4 + 2 * 8);

    let mut data = Vec::new();
    test_bytes::push_address(&mut data, Some(0x1234), 8);
    test_bytes::push_uint(&mut data, 57, 2);
    data.extend_from_slice(&0x0000_0005u32.to_le_bytes());
    test_bytes::push_uint(&mut data, 3, 8);
    test_bytes::push_uint(&mut data, 4, 8);
    let record = context.decode(&BTreeV2Record { data }).unwrap();

    assert_eq!(record.address, StoredAddress::new(0x1234));
    assert_eq!(record.stored_size, 57);
    assert_eq!(record.filter_mask, 5);
    assert_eq!(record.scaled_offsets, vec![3, 4]);
}

#[test]
fn chunk_record_context_rejects_declared_record_size_mismatches() {
    let context = BTreeV2ChunkRecordContext::new(
        BTREE_V2_CHUNK,
        OffsetWidth::Eight,
        2,
        NonZeroU64::new(64).unwrap(),
        LayoutVersion::Four,
    )
    .unwrap();
    let expected = context.record_size();

    assert_eq!(
        context
            .require_record_size(u16::try_from(expected - 1).unwrap())
            .unwrap_err(),
        FormatError::DataSizeMismatch {
            expected,
            actual: expected - 1,
        }
    );
    assert_eq!(
        context
            .require_record_size(u16::try_from(expected + 1).unwrap())
            .unwrap_err(),
        FormatError::DataSizeMismatch {
            expected,
            actual: expected + 1,
        }
    );
}

#[test]
fn filtered_chunk_records_reject_zero_size_and_undefined_address() {
    let context = BTreeV2ChunkRecordContext::new(
        BTREE_V2_FILTERED_CHUNK,
        OffsetWidth::Four,
        1,
        NonZeroU64::new(32).unwrap(),
        LayoutVersion::Four,
    )
    .unwrap();

    let mut zero_size = Vec::new();
    test_bytes::push_address(&mut zero_size, Some(0x100), 4);
    test_bytes::push_uint(&mut zero_size, 0, 2);
    zero_size.extend_from_slice(&0u32.to_le_bytes());
    test_bytes::push_uint(&mut zero_size, 0, 8);
    assert_eq!(
        context
            .decode(&BTreeV2Record { data: zero_size })
            .unwrap_err(),
        FormatError::InvalidChunkGeometry(
            "filtered version 2 B-tree chunk record has zero stored size"
        )
    );

    let mut undefined = Vec::new();
    test_bytes::push_undefined_address(&mut undefined, 4);
    test_bytes::push_uint(&mut undefined, 1, 2);
    undefined.extend_from_slice(&0u32.to_le_bytes());
    test_bytes::push_uint(&mut undefined, 0, 8);
    assert_eq!(
        context
            .decode(&BTreeV2Record { data: undefined })
            .unwrap_err(),
        FormatError::ChunkedReadError(
            "version 2 B-tree chunk record has an undefined chunk address".into()
        )
    );
}

#[test]
fn chunk_record_context_rejects_non_chunk_tree_types() {
    assert_eq!(
        BTreeV2ChunkRecordContext::new(
            BTREE_V2_ATTRIBUTE_NAME,
            OffsetWidth::Eight,
            1,
            NonZeroU64::new(32).unwrap(),
            LayoutVersion::Four,
        )
        .unwrap_err(),
        FormatError::InvalidBTreeNodeType(BTREE_V2_ATTRIBUTE_NAME)
    );
}
