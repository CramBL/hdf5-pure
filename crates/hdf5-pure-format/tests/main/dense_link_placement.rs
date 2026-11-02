use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::DenseLinkPlan;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use hdf5_pure_format::CharacterSet;
use rstest::rstest;

#[rstest]
#[case::empty(0)]
#[case::leaf(2)]
#[case::internal(200)]
fn contiguous_child_placement_preserves_the_complete_image(
    #[case] count: usize,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    #[values(false, true)] indexed: bool,
) {
    let plan = plan(count, width, indexed);
    let base = StoredAddress::new(1024);
    let mut cursor = base.get();
    let addresses: Vec<_> = plan
        .block_lengths()
        .map(|length| {
            let address = StoredAddress::new(cursor);
            cursor += length;
            address
        })
        .collect();
    let placement = plan.place(&addresses).unwrap();
    assert_eq!(placement.link_info(), &plan.link_info(base).unwrap());
    let mut bytes = Vec::new();
    let mut actual = Vec::new();
    for ((address, length), encoded) in placement
        .blocks()
        .zip(placement.serialize_blocks().unwrap())
    {
        let (stored, block) = encoded.unwrap();
        assert_eq!(address, stored);
        assert_eq!(length, block.len() as u64);
        actual.push(stored);
        bytes.extend(block);
    }
    assert_eq!(actual, addresses);
    assert_eq!(bytes, plan.serialize(base).unwrap().bytes);
}

#[rstest]
#[case::missing(vec![], FormatError::DenseLinkBlockCount { expected: 6, actual: 0 })]
#[case::extra(vec![0; 7], FormatError::DenseLinkBlockCount { expected: 6, actual: 7 })]
#[case::heap_header_overlap(vec![0, 1, 1024, 2048, 3072, 4096], FormatError::DenseLinkBlockOverlap { first: 0, second: 1 })]
#[case::heap_name_overlap(vec![0, 1024, 1025, 2048, 3072, 4096], FormatError::DenseLinkBlockOverlap { first: 1, second: 2 })]
#[case::name_header_node_overlap(vec![0, 1024, 2048, 2049, 3072, 4096], FormatError::DenseLinkBlockOverlap { first: 2, second: 3 })]
#[case::name_creation_overlap(vec![0, 1024, 2048, 3072, 3073, 4096], FormatError::DenseLinkBlockOverlap { first: 3, second: 4 })]
#[case::creation_header_node_overlap(vec![0, 1024, 2048, 3072, 4096, 4097], FormatError::DenseLinkBlockOverlap { first: 4, second: 5 })]
#[case::distant_overlap(vec![4096, 1024, 2048, 3072, 0, 4097], FormatError::DenseLinkBlockOverlap { first: 0, second: 5 })]
fn placement_rejects_incomplete_or_overlapping_component_ranges(
    #[case] addresses: Vec<u64>,
    #[case] expected: FormatError,
) {
    let plan = plan(2, OffsetWidth::Eight, true);
    let addresses: Vec<_> = addresses.into_iter().map(StoredAddress::new).collect();
    assert_eq!(plan.place(&addresses).err().unwrap(), expected);
}

#[rstest]
#[case::undefined(65_535, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::address_width(65_536, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::extent_width(65_024, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
fn placement_rejects_invalid_complete_component_extents(
    #[case] invalid: u64,
    #[case] expected: FormatError,
    #[values(1, 3, 5)] block: usize,
) {
    let plan = plan(2, OffsetWidth::Two, true);
    let mut addresses = [0, 1024, 2048, 3072, 4096, 5120].map(StoredAddress::new);
    addresses[block] = StoredAddress::new(invalid);
    assert_eq!(plan.place(&addresses).err().unwrap(), expected);
}

#[test]
fn placement_accepts_the_maximum_exclusive_end() {
    let plan = plan(2, OffsetWidth::Two, true);
    let addresses = [0, 1024, 2048, 3072, 4096, 65_023].map(StoredAddress::new);
    let placement = plan.place(&addresses).unwrap();
    let blocks: Vec<_> = placement
        .serialize_blocks()
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(blocks.len(), addresses.len());
    assert_eq!(blocks.last().unwrap().0, StoredAddress::new(65_023));
    assert_eq!(blocks.last().unwrap().1.len(), 512);
}

#[test]
fn placement_rejects_an_overflowing_component_end() {
    let plan = plan(2, OffsetWidth::Eight, false);
    let addresses = [0, u64::MAX - 1, 2048, 3072].map(StoredAddress::new);
    assert_eq!(
        plan.place(&addresses).err().unwrap(),
        FormatError::OffsetOverflow {
            offset: u64::MAX - 1,
            length: 512,
        }
    );
}

fn plan(count: usize, width: OffsetWidth, indexed: bool) -> DenseLinkPlan {
    let info = LinkInfoMessage {
        max_creation_order: indexed.then_some(count as u64 + 7),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: indexed.then_some(None),
    };
    let links = (0..count)
        .map(|index| {
            OwnedLink::new(
                format!("link_{index}").into_bytes(),
                CharacterSet::Ascii,
                OwnedLinkTarget::soft(b"/payload".to_vec()),
                indexed.then_some(index as u64),
            )
            .unwrap()
        })
        .collect();
    DenseLinkPlan::new(
        GroupLinkStorage::new(&info, links, width).unwrap(),
        LengthWidth::Eight,
    )
    .unwrap()
}
