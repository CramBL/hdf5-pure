use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::BTREE_V2_LINK_NAME;
use hdf5_pure_format::__private::BTREE_V2_NODE_SIZE;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::BTreeV2Plan;
use hdf5_pure_format::__private::BTreeV2WriteDialect;
use rstest::rstest;

#[rstest]
#[case::empty(0)]
#[case::leaf(2)]
#[case::internal(120)]
#[case::depth_two(5000)]
fn independently_placed_nodes_preserve_records_and_checksums(
    #[case] count: usize,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let plan = BTreeV2Plan::new_with_undefined_empty_root(
        BTreeV2WriteDialect::Libhdf5Global,
        BTREE_V2_LINK_NAME,
        count,
        11,
        BTREE_V2_NODE_SIZE,
        width,
    )
    .unwrap();
    let records: Vec<u8> = (0..count)
        .flat_map(|index| {
            let mut record = [0; 11];
            record[..8].copy_from_slice(&(index as u64).to_le_bytes());
            record
        })
        .collect();
    let base = if width == OffsetWidth::Eight {
        1u64 << 40
    } else {
        1024
    };
    let addresses: Vec<_> = (0..plan.block_lengths().len())
        .rev()
        .map(|index| StoredAddress::new(base + index as u64 * 549))
        .collect();
    let placement = plan.place(&addresses).unwrap();
    assert_eq!(
        placement.blocks().map(|block| block.0).collect::<Vec<_>>(),
        addresses
    );
    assert_eq!(
        placement.blocks().map(|block| block.1).sum::<u64>(),
        plan.nodes_size()
    );
    let encoded: Vec<_> = placement
        .serialize_nodes(&records)
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let header = placement.serialize_header(LengthWidth::Eight).unwrap();
    let header = BTreeV2Header::parse(&header, 0, width.get(), 8).unwrap();
    assert_eq!(header.total_records, count as u64);
    assert_eq!(
        header.root_node_address,
        addresses
            .last()
            .copied()
            .unwrap_or_else(|| StoredAddress::undefined(width))
    );
    let layout = BTreeV2Layout::new(&header, width).unwrap();
    let mut pending: Vec<_> = layout.root().into_iter().collect();
    let mut found = Vec::new();
    let mut visited = Vec::new();
    while let Some(node) = pending.pop() {
        let (_, bytes) = encoded
            .iter()
            .find(|block| block.0 == node.address())
            .unwrap();
        assert_eq!(bytes.len(), 512);
        let view = node.parse(bytes).unwrap();
        visited.push(node.address());
        for index in 0..view.record_count() {
            found.push(view.record(index).unwrap().to_vec());
        }
        for index in 0..=view.record_count() {
            pending.extend(view.child(index).unwrap());
        }
    }
    visited.sort_unstable_by_key(|address| address.get());
    let mut expected = addresses;
    expected.sort_unstable_by_key(|address| address.get());
    assert_eq!(visited, expected);
    found.sort_unstable_by_key(|record| u64::from_le_bytes(record[..8].try_into().unwrap()));
    assert_eq!(found.concat(), records);
}

#[rstest]
#[case::empty(0)]
#[case::leaf(2)]
#[case::internal(120)]
#[case::depth_two(5000)]
fn contiguous_placement_preserves_serialization(#[case] count: usize) {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, count, 11, 512, OffsetWidth::Eight).unwrap();
    let addresses: Vec<_> = (0..plan.block_lengths().len())
        .map(|index| StoredAddress::new(1024 + index as u64 * 512))
        .collect();
    let placement = plan.place(&addresses).unwrap();
    let records = vec![37; count * 11];
    let legacy = plan
        .serialize(
            &records,
            StoredAddress::new(1024),
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .unwrap();
    assert_eq!(
        placement.serialize_header(LengthWidth::Eight).unwrap(),
        legacy.header
    );
    let nodes: Vec<_> = placement
        .serialize_nodes(&records)
        .unwrap()
        .flat_map(|block| block.unwrap().1)
        .collect();
    assert_eq!(nodes, legacy.nodes);
}

#[rstest]
#[case::missing(vec![], FormatError::BTreeV2NodeCount { expected: 1, actual: 0 })]
#[case::extra(vec![0, 1024], FormatError::BTreeV2NodeCount { expected: 1, actual: 2 })]
#[case::undefined(vec![65_535], FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::address_width(vec![65_536], FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::extent_width(vec![65_024], FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
fn placement_rejects_invalid_node_addresses(
    #[case] addresses: Vec<u64>,
    #[case] expected: FormatError,
) {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 2, 11, 512, OffsetWidth::Two).unwrap();
    let addresses: Vec<_> = addresses.into_iter().map(StoredAddress::new).collect();
    assert_eq!(plan.place(&addresses).err().unwrap(), expected);
}

#[rstest]
#[case::duplicate([0, 0, 1024], 0, 1)]
#[case::partial_overlap([0, 30, 1024], 0, 1)]
#[case::reverse_overlap([1024, 0, 1000], 2, 0)]
fn placement_rejects_overlapping_nodes(
    #[case] addresses: [u64; 3],
    #[case] first: usize,
    #[case] second: usize,
) {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 50, 11, 512, OffsetWidth::Eight).unwrap();
    let addresses = addresses.map(StoredAddress::new);
    assert_eq!(
        plan.place(&addresses).err().unwrap(),
        FormatError::BTreeV2NodeOverlap { first, second }
    );
}

#[test]
fn placement_rejects_overflowing_node_ends() {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 2, 11, 512, OffsetWidth::Eight).unwrap();
    assert_eq!(
        plan.place(&[StoredAddress::new(u64::MAX - 1)])
            .err()
            .unwrap(),
        FormatError::OffsetOverflow {
            offset: u64::MAX - 1,
            length: 512
        }
    );
}

#[rstest]
#[case::missing(vec![])]
#[case::extra(vec![0; 23])]
fn node_serialization_rejects_another_record_length(#[case] records: Vec<u8>) {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 2, 11, 512, OffsetWidth::Eight).unwrap();
    let placement = plan.place(&[StoredAddress::new(0)]).unwrap();
    assert_eq!(
        placement.serialize_nodes(&records).err().unwrap(),
        FormatError::Internal("record buffer length differs from the planned records".into())
    );
}

#[test]
fn header_serialization_rejects_an_unrepresentable_total() {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 65_536, 11, 512, OffsetWidth::Eight).unwrap();
    let addresses: Vec<_> = (0..plan.block_lengths().len())
        .map(|index| StoredAddress::new(index as u64 * 512))
        .collect();
    let placement = plan.place(&addresses).unwrap();
    assert_eq!(
        placement.serialize_header(LengthWidth::Two).unwrap_err(),
        FormatError::LengthTooLarge {
            length: 65_536,
            length_size: 2
        }
    );
}

#[test]
fn placement_accepts_the_maximum_exclusive_end() {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 2, 11, 512, OffsetWidth::Two).unwrap();
    let placement = plan.place(&[StoredAddress::new(65_023)]).unwrap();
    assert_eq!(
        placement.blocks().collect::<Vec<_>>(),
        [(StoredAddress::new(65_023), 512)]
    );
    assert_eq!(
        placement
            .serialize_nodes(&[0; 22])
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .1
            .len(),
        512
    );
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
fn contiguous_serialization_rejects_another_address_width(#[case] width: OffsetWidth) {
    let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 2, 11, 512, OffsetWidth::Eight).unwrap();
    let error = plan
        .serialize(
            &[0; 22],
            StoredAddress::new(1024),
            width,
            LengthWidth::Eight,
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        FormatError::Internal("B-tree serialization address width differs from the plan".into())
    );
}
