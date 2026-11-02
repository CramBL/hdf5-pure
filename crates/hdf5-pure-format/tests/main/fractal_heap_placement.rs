use hdf5_pure_core::__private::{FileRange, FileRegion, StoredAddressExt};
use hdf5_pure_core::{FormatError, PhysicalAddress, StoredAddress};
use hdf5_pure_format::__private::{
    FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED, FractalHeapBlock, FractalHeapBlockLayout,
    FractalHeapGeometry, FractalHeapHeader, LengthWidth, ManagedHeapPlan, OffsetWidth,
};
use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

#[rstest]
#[case::empty(vec![], 512, 65_536, 32)]
#[case::direct(vec![491], 512, 65_536, 32)]
#[case::indirect(vec![491; 12], 512, 65_536, 32)]
#[case::skipped_slots(vec![2000; 3], 512, 65_536, 33)]
#[case::nested_indirect(vec![100; 100], 64, 128, 16)]
fn independently_placed_blocks_preserve_objects_and_checksums(
    #[case] sizes: Vec<u64>,
    #[case] start: u64,
    #[case] maximum: u64,
    #[case] bits: u16,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let plan = ManagedHeapPlan::new(
        &sizes,
        width,
        FractalHeapGeometry::new(4, start, maximum, bits).unwrap(),
    )
    .unwrap();
    let objects: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(index, &size)| vec![u8::try_from(index).unwrap(); usize::try_from(size).unwrap()])
        .collect();
    let borrowed: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
    let base = if width == OffsetWidth::Eight {
        1u64 << 40
    } else {
        1024
    };
    let mut cursor = base;
    let mut addresses: Vec<_> = plan
        .block_lengths()
        .rev()
        .map(|length| {
            let address = StoredAddress::new(cursor);
            cursor += length + 37;
            address
        })
        .collect();
    addresses.reverse();
    let placement = plan.place(&addresses).unwrap();
    let geometry: Vec<_> = placement.blocks().collect();
    assert_eq!(
        geometry.iter().map(|entry| entry.1).sum::<u64>(),
        plan.region_size()
    );
    let owner = StoredAddress::new(128);
    let mut header =
        FractalHeapHeader::parse(&fractal_heap::Header::new(0).build(Widths::EIGHT), 0, 8, 8)
            .unwrap();
    header.flags = FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
    header.table_width = 4;
    header.starting_block_size = start;
    header.max_direct_block_size = maximum;
    header.max_heap_size = bits;
    header.root_block_address = placement.root_address();
    header.current_rows_in_root_indirect_block = plan.root_rows();
    let layout = FractalHeapBlockLayout::new(&header, width, LengthWidth::Eight, owner).unwrap();
    let encoded: Vec<_> = placement
        .serialize_blocks(&borrowed, owner)
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(encoded.len(), geometry.len());
    for ((address, data), expected) in encoded.iter().zip(&geometry) {
        assert_eq!((*address, data.len() as u64), *expected);
    }
    let mut pending = vec![layout.root().unwrap().unwrap()];
    let mut visited = Vec::new();
    let mut found = Vec::new();
    while let Some(block) = pending.pop() {
        let address = match block {
            FractalHeapBlock::Direct(block) => block.address(),
            FractalHeapBlock::Indirect(block) => block.address(),
        };
        let data = &encoded.iter().find(|entry| entry.0 == address).unwrap().1;
        let image = FileRegion::new(
            FileRange::new(PhysicalAddress::new(address.get()), data.len() as u64).unwrap(),
            data,
        )
        .unwrap();
        visited.push(address);
        match block {
            FractalHeapBlock::Direct(block) => {
                let reference = block.at(PhysicalAddress::new(address.get())).unwrap();
                reference.parse_checked(image).unwrap();
                for (index, object) in objects.iter().enumerate() {
                    let offset = plan.heap_offset(index);
                    if offset < block.heap_offset()
                        || offset >= block.heap_offset() + block.decoded_size()
                    {
                        continue;
                    }
                    let object_ref = reference
                        .parse_object(image, offset, object.len() as u64)
                        .unwrap();
                    assert_eq!(object_ref.parse(image).unwrap(), object);
                    found.push(index);
                }
            }
            FractalHeapBlock::Indirect(block) => {
                let reference = block.at(PhysicalAddress::new(address.get())).unwrap();
                let view = reference.parse(image).unwrap();
                pending.extend(view.children().map(Result::unwrap));
            }
        }
    }
    visited.sort_unstable_by_key(|address| address.get());
    addresses.sort_unstable_by_key(|address| address.get());
    assert_eq!(visited, addresses);
    found.sort_unstable();
    assert_eq!(found, (0..sizes.len()).collect::<Vec<_>>());
}

#[rstest]
#[case::direct(vec![1])]
#[case::indirect(vec![491; 12])]
fn contiguous_placement_matches_region_serialization(#[case] sizes: Vec<u64>) {
    let plan = ManagedHeapPlan::group_links(&sizes, OffsetWidth::Eight).unwrap();
    let owner = StoredAddress::new(128);
    let base = StoredAddress::new(512);
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
    assert_eq!(placement.root_address(), plan.root_address(base).unwrap());
    let objects: Vec<Vec<u8>> = sizes
        .iter()
        .map(|&size| vec![7; usize::try_from(size).unwrap()])
        .collect();
    let borrowed: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
    let emitted: Vec<u8> = placement
        .serialize_blocks(&borrowed, owner)
        .unwrap()
        .flat_map(|block| block.unwrap().1)
        .collect();
    assert_eq!(emitted, plan.serialize(&borrowed, base, owner).unwrap());
}

#[rstest]
#[case::missing(vec![], FormatError::FractalHeapBlockCount { expected: 1, actual: 0 })]
#[case::extra(vec![0, 1024], FormatError::FractalHeapBlockCount { expected: 1, actual: 2 })]
#[case::undefined(vec![65_535], FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::address_width(vec![65_536], FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::extent_width(vec![65_024], FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
fn placement_rejects_unrepresentable_blocks(
    #[case] addresses: Vec<u64>,
    #[case] expected: FormatError,
) {
    let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Two).unwrap();
    let addresses: Vec<_> = addresses.into_iter().map(StoredAddress::new).collect();
    assert_eq!(plan.place(&addresses).err().unwrap(), expected);
}

#[rstest]
#[case::duplicate([0, 0, 1024], 0, 1)]
#[case::partial_overlap([0, 30, 1024], 0, 1)]
#[case::reverse_overlap([1024, 0, 1000], 2, 0)]
fn placement_rejects_overlapping_complete_blocks(
    #[case] addresses: [u64; 3],
    #[case] first: usize,
    #[case] second: usize,
) {
    let plan = ManagedHeapPlan::group_links(&[491, 1], OffsetWidth::Eight).unwrap();
    let addresses = addresses.map(StoredAddress::new);
    assert_eq!(
        plan.place(&addresses).err().unwrap(),
        FormatError::FractalHeapBlockOverlap { first, second }
    );
}

#[test]
fn placement_rejects_overflowing_block_ends() {
    let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Eight).unwrap();
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
#[case::count(vec![], 128, FormatError::Internal("managed heap emission received another object count".into()))]
#[case::size(vec![vec![1, 2]], 128, FormatError::DataSizeMismatch { expected: 1, actual: 2 })]
#[case::owner_width(vec![vec![1]], 65_536, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::undefined_owner(vec![vec![1]], 65_535, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
fn block_serialization_rejects_invalid_inputs_before_emission(
    #[case] objects: Vec<Vec<u8>>,
    #[case] owner: u64,
    #[case] expected: FormatError,
) {
    let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Two).unwrap();
    let placement = plan.place(&[StoredAddress::new(0)]).unwrap();
    let borrowed: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
    assert_eq!(
        placement
            .serialize_blocks(&borrowed, StoredAddress::new(owner))
            .err()
            .unwrap(),
        expected
    );
}

#[test]
fn placement_accepts_an_exclusive_end_at_the_undefined_address() {
    let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Two).unwrap();
    let placement = plan.place(&[StoredAddress::new(65_023)]).unwrap();
    assert_eq!(
        placement.blocks().collect::<Vec<_>>(),
        [(StoredAddress::new(65_023), 512)]
    );
    let data = placement
        .serialize_blocks(&[&[37]], StoredAddress::new(128))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(data.0, StoredAddress::new(65_023));
    assert_eq!(data.1.len(), 512);
}
