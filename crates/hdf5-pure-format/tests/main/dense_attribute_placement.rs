use std::borrow::Cow;

use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::{
    AttributeMessage, Dataspace, DataspaceType, DatatypeLocation, DenseAttrBlockKind,
    DenseAttrCreationOrder, DenseAttrFixup, DenseAttrPlan, PreparedDenseAttrs,
};
use hdf5_pure_format::{CharacterSet, Datatype, StringPadding};
use rstest::rstest;
use test_util::{btree_v2, fractal_heap};

#[rstest]
#[case::empty(vec![])]
#[case::managed(vec![8; 80])]
#[case::huge(vec![70_000])]
#[case::huge_index(vec![70_000; 40])]
#[case::mixed(vec![8, 70_000, 16, 80_000])]
fn contiguous_dense_attribute_placement_preserves_bytes(
    #[case] sizes: Vec<u32>,
    #[values(None, Some(false), Some(true))] creation: Option<bool>,
) {
    let prepared = prepared(&sizes, creation);
    let lengths: Vec<_> = prepared.block_lengths().collect();
    assert_eq!(lengths.iter().sum::<u64>(), prepared.blob_len());
    let base = StoredAddress::new(0x8000);
    let mut cursor = base.get();
    let addresses: Vec<_> = lengths
        .iter()
        .map(|length| {
            let address = StoredAddress::new(cursor);
            cursor += length;
            address
        })
        .collect();
    let info = prepared.attribute_info(&addresses).unwrap();
    let plan = finalize(prepared);
    assert_eq!(plan.block_lengths().collect::<Vec<_>>(), lengths);
    let placed = plan.place(&addresses).unwrap();
    assert_eq!(placed.attribute_info(), &info);
    let mut bytes = Vec::new();
    let mut emitted = Vec::new();
    for ((address, length), block) in placed.blocks().zip(placed.serialize_blocks().unwrap()) {
        let block = block.unwrap();
        let (stored, data) = (block.address, block.bytes);
        assert_eq!((stored, data.len() as u64), (address, length));
        emitted.push(stored);
        bytes.extend_from_slice(&data);
    }
    assert_eq!(emitted, addresses);
    let contiguous = plan.build(base).unwrap();
    assert_eq!(bytes, contiguous.blob);
    assert_eq!(
        info.serialize(hdf5_pure_core::OffsetWidth::Eight),
        contiguous.attr_info_message
    );
}

#[rstest]
#[case::empty(vec![], 0)]
#[case::managed(vec![512; 80], 0)]
#[case::huge(vec![70_000, 80_000], 2)]
#[case::mixed(vec![8, 70_000, 16, 80_000], 2)]
fn dense_attribute_geometry_classifies_and_borrows_huge_payloads(
    #[case] sizes: Vec<u32>,
    #[case] huge_count: usize,
) {
    let prepared = prepared(&sizes, Some(true));
    let kinds: Vec<_> = prepared.block_kinds().collect();
    assert_eq!(kinds.len(), prepared.block_lengths().len());
    let plan = finalize(prepared);
    assert_eq!(plan.block_kinds().collect::<Vec<_>>(), kinds);
    let mut cursor = 512u64;
    let mut addresses = vec![StoredAddress::ZERO; plan.block_lengths().len()];
    for (index, length) in plan
        .block_lengths()
        .enumerate()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        cursor = cursor.div_ceil(512) * 512;
        addresses[index] = StoredAddress::new(cursor);
        cursor += length + 512;
    }
    let placed = plan.place(&addresses).unwrap();
    assert_eq!(
        placed.attribute_info().fractal_heap_address,
        Some(addresses[0])
    );
    let blocks = placed
        .serialize_blocks()
        .unwrap()
        .map(|block| {
            let block = block.unwrap();
            (block.address, block.bytes)
        })
        .collect::<Vec<_>>();
    assert_eq!(blocks.len(), addresses.len());
    for (kind, (_, bytes)) in kinds.iter().zip(&blocks) {
        match kind {
            DenseAttrBlockKind::HugeObject => assert!(matches!(bytes, Cow::Borrowed(_))),
            DenseAttrBlockKind::Metadata => {
                assert!(matches!(bytes, Cow::Owned(_)));
                assert!(
                    [
                        fractal_heap::SIGNATURE,
                        fractal_heap::INDIRECT_BLOCK_SIGNATURE,
                        fractal_heap::DIRECT_BLOCK_SIGNATURE,
                        btree_v2::SIGNATURE,
                        btree_v2::LEAF_SIGNATURE,
                        btree_v2::INTERNAL_SIGNATURE,
                    ]
                    .contains(&bytes.first_chunk::<4>().unwrap())
                );
            }
        }
    }
    assert_eq!(
        blocks
            .iter()
            .map(|(_, bytes)| usize::from(matches!(bytes, Cow::Borrowed(_))))
            .sum::<usize>(),
        huge_count
    );
    assert_eq!(
        blocks.iter().map(|block| block.0).collect::<Vec<_>>(),
        addresses
    );
    assert_eq!(
        blocks
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .collect::<Vec<_>>(),
        plan.block_lengths().collect::<Vec<_>>()
    );
    let huge = blocks
        .iter()
        .filter_map(|(_, bytes)| match bytes {
            Cow::Borrowed(bytes) => Some(bytes.len()),
            Cow::Owned(_) => None,
        })
        .collect::<Vec<_>>();
    let expected = sizes
        .iter()
        .enumerate()
        .filter(|(_, size)| **size >= 70_000)
        .map(|(index, size)| {
            attribute(index, *size)
                .serialize_v3(hdf5_pure_core::OffsetWidth::Eight, 8)
                .unwrap()
                .len()
        })
        .collect::<Vec<_>>();
    assert_eq!(huge, expected);
}

#[rstest]
#[case::missing(vec![], FormatError::DenseAttributeBlockCount { expected: 10, actual: 0 })]
#[case::extra(vec![0; 11], FormatError::DenseAttributeBlockCount { expected: 10, actual: 11 })]
#[case::heap_header(vec![0, 1, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 150_000], FormatError::DenseAttributeBlockOverlap { first: 0, second: 1 })]
#[case::name_header(vec![0, 10_000, 10_001, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 150_000], FormatError::DenseAttributeBlockOverlap { first: 1, second: 2 })]
#[case::creation_header(vec![0, 10_000, 20_000, 30_000, 30_001, 50_000, 60_000, 70_000, 80_000, 150_000], FormatError::DenseAttributeBlockOverlap { first: 3, second: 4 })]
#[case::huge_header(vec![0, 10_000, 20_000, 30_000, 40_000, 50_000, 50_001, 70_000, 80_000, 150_000], FormatError::DenseAttributeBlockOverlap { first: 5, second: 6 })]
#[case::huge_payload(vec![0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 70_001, 150_000], FormatError::DenseAttributeBlockOverlap { first: 7, second: 8 })]
#[case::distant_payload(vec![200_001, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 200_000], FormatError::DenseAttributeBlockOverlap { first: 9, second: 0 })]
fn dense_attribute_placement_rejects_incomplete_or_overlapping_ranges(
    #[case] addresses: Vec<u64>,
    #[case] expected: FormatError,
) {
    let pending = prepared(&[8, 70_000, 80_000], Some(true));
    let addresses: Vec<_> = addresses.into_iter().map(StoredAddress::new).collect();
    assert_eq!(pending.attribute_info(&addresses).unwrap_err(), expected);
    let plan = finalize(pending);
    assert_eq!(plan.place(&addresses).err().unwrap(), expected);
}

#[rstest]
#[case::undefined(u64::MAX, true)]
#[case::overflow(u64::MAX - 1, false)]
fn dense_attribute_placement_rejects_invalid_extents(
    #[case] address: u64,
    #[case] undefined: bool,
    #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)] component: usize,
) {
    let pending = prepared(&[8, 70_000, 80_000], Some(true));
    let plan = finalize(pending.clone());
    let mut addresses = (0..plan.block_lengths().len())
        .map(|index| StoredAddress::new(index as u64 * 100_000))
        .collect::<Vec<_>>();
    let length = plan.block_lengths().nth(component).unwrap();
    addresses[component] = StoredAddress::new(address);
    let expected = if undefined {
        FormatError::DefinedAddressIsUndefined {
            address,
            offset_size: 8,
        }
    } else {
        FormatError::OffsetOverflow {
            offset: address,
            length,
        }
    };
    assert_eq!(pending.attribute_info(&addresses).unwrap_err(), expected);
    assert_eq!(plan.place(&addresses).err().unwrap(), expected);
}

#[rstest]
#[case::heap_header(0)]
#[case::managed(1)]
#[case::name_header(2)]
#[case::name_node(3)]
#[case::creation_header(4)]
#[case::creation_node(5)]
#[case::huge_header(6)]
#[case::huge_node(7)]
#[case::huge_payload(8)]
fn dense_attribute_placement_accepts_the_maximum_exclusive_end(#[case] component: usize) {
    let plan = finalize(prepared(&[8, 70_000], Some(true)));
    let mut addresses = (0..plan.block_lengths().len())
        .map(|index| StoredAddress::new(index as u64 * 100_000))
        .collect::<Vec<_>>();
    let length = plan.block_lengths().nth(component).unwrap();
    addresses[component] = StoredAddress::new(u64::MAX - length);
    let placed = plan.place(&addresses).unwrap();
    let blocks = placed
        .serialize_blocks()
        .unwrap()
        .map(|block| {
            let block = block.unwrap();
            (block.address, block.bytes)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        blocks[component].0.get() + blocks[component].1.len() as u64,
        u64::MAX
    );
}

#[rstest]
#[case::managed(8)]
#[case::huge(70_000)]
fn finalized_payloads_use_the_prepared_placement(#[case] size: u32) {
    let prepared = PreparedDenseAttrs::new(
        &[attribute(0, size)],
        DenseAttrCreationOrder::Untracked,
        [DenseAttrFixup::Payload { attribute: 0 }],
    )
    .unwrap();
    let addresses = prepared
        .block_lengths()
        .enumerate()
        .map(|(index, _)| StoredAddress::new(index as u64 * 100_000))
        .collect::<Vec<_>>();
    let metadata = prepared.attribute_info(&addresses).unwrap();
    let payload = vec![b'z'; size as usize];
    let plan = prepared
        .finalize(
            |reference| panic!("unexpected reference {reference:?}"),
            |_| Ok(&payload),
        )
        .unwrap();
    let placement = plan.place(&addresses).unwrap();
    assert_eq!(placement.attribute_info(), &metadata);
    let expected = AttributeMessage {
        raw_data: payload,
        ..attribute(0, size)
    }
    .serialize_v3(hdf5_pure_core::OffsetWidth::Eight, 8)
    .unwrap();
    let blocks = placement
        .serialize_blocks()
        .unwrap()
        .map(|block| {
            let block = block.unwrap();
            (block.address, block.bytes)
        })
        .collect::<Vec<_>>();
    assert!(blocks.iter().any(|(_, bytes)| {
        bytes
            .windows(expected.len())
            .any(|window| window == expected)
    }));
}

fn prepared(sizes: &[u32], creation: Option<bool>) -> PreparedDenseAttrs {
    PreparedDenseAttrs::new(
        &sizes
            .iter()
            .enumerate()
            .map(|(index, size)| attribute(index, *size))
            .collect::<Vec<_>>(),
        match creation {
            Some(indexed) => DenseAttrCreationOrder::Tracked {
                indices: (0..sizes.len()).map(|index| index as u16 * 2 + 1).collect(),
                max: sizes.len() as u16 * 2 + 7,
                indexed,
            },
            None => DenseAttrCreationOrder::Untracked,
        },
        [],
    )
    .unwrap()
}

fn finalize(prepared: PreparedDenseAttrs) -> DenseAttrPlan {
    prepared
        .finalize(
            |reference| panic!("unexpected reference {reference:?}"),
            |attribute| panic!("unexpected payload {attribute}"),
        )
        .unwrap()
}

fn attribute(index: usize, size: u32) -> AttributeMessage {
    AttributeMessage {
        name: format!("attribute_{index}"),
        datatype: Datatype::String {
            size,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        },
        dataspace: Dataspace {
            space_type: DataspaceType::Scalar,
            rank: 0,
            dimensions: vec![],
            max_dimensions: None,
        },
        raw_data: vec![b'y'; size as usize],
        datatype_location: DatatypeLocation::Inline,
    }
}
