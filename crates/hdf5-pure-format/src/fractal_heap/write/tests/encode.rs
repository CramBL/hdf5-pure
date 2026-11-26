use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::error::FormatError;
use crate::fractal_heap::block::{CHECKSUM_LEN, FractalHeapBlock, FractalHeapBlockLayout};
use crate::fractal_heap::write::ManagedHeapPlan;
use crate::fractal_heap::{
    FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED, FractalHeapGeometry, FractalHeapHeader,
    FractalHeapIdKind, FractalHeapIdLayout,
};
use crate::width::{LengthWidth, OffsetWidth};

fn header(plan: &ManagedHeapPlan, region_address: StoredAddress) -> FractalHeapHeader {
    let mut header =
        FractalHeapHeader::parse(&fractal_heap::Header::new(0).build(Widths::EIGHT), 0, 8, 8)
            .unwrap();
    header.flags = FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
    header.btree_huge_objects_address = StoredAddress::undefined(plan.offset_width);
    header.managed_block_free_space_manager_address = StoredAddress::undefined(plan.offset_width);
    header.heap_id_length = 12;
    header.table_width = u16::try_from(plan.geometry.table_width()).unwrap();
    header.starting_block_size = plan.geometry.starting_block_size();
    header.max_direct_block_size = plan.geometry.max_direct_block_size();
    header.max_heap_size = plan.geometry.max_heap_size();
    header.max_managed_object_size =
        u32::try_from(header.max_direct_block_size - plan.direct_header_size).unwrap();
    header.root_block_address = plan.root_address(region_address).unwrap();
    header.current_rows_in_root_indirect_block = plan.root_rows();
    header.managed_space = plan.managed_space();
    header.allocated_managed_space = plan.allocated_space();
    header.free_space_in_managed_blocks = plan.free_space();
    header.direct_block_allocation_iterator_offset = plan.allocation_iterator();
    header.managed_objects_count = plan.sizes.len() as u64;
    header
}

#[rstest]
#[case::empty_attributes(1024, 65_536, 40, vec![], OffsetWidth::Eight)]
#[case::attributes_direct(1024, 65_536, 40, vec![1002], OffsetWidth::Eight)]
#[case::attributes_indirect(1024, 65_536, 40, vec![1002, 1], OffsetWidth::Eight)]
#[case::group_direct(512, 65_536, 32, vec![491], OffsetWidth::Eight)]
#[case::group_indirect(512, 65_536, 32, vec![491, 1], OffsetWidth::Eight)]
#[case::skip_small_blocks(1024, 65_536, 40, vec![65_514; 3], OffsetWidth::Eight)]
#[case::nested_attributes(1024, 65_536, 40, vec![65_514; 40], OffsetWidth::Eight)]
#[case::deep_small_geometry(64, 128, 16, vec![100; 100], OffsetWidth::Eight)]
#[case::non_byte_aligned_heap_offsets(512, 65_536, 33, vec![491; 8], OffsetWidth::Eight)]
#[case::two_byte_addresses(512, 65_536, 32, vec![100; 50], OffsetWidth::Two)]
#[case::four_byte_addresses(1024, 65_536, 40, vec![65_518; 40], OffsetWidth::Four)]
fn planned_blocks_and_objects_conform_to_the_format_parser(
    #[case] start: u64,
    #[case] max_direct: u64,
    #[case] bits: u16,
    #[case] sizes: Vec<u64>,
    #[values(512, 0x2000)] base: u64,
    #[case] width: OffsetWidth,
) {
    let geometry = FractalHeapGeometry::new(4, start, max_direct, bits).unwrap();
    let plan = ManagedHeapPlan::new(&sizes, width, geometry).unwrap();
    let objects: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(index, &size)| vec![u8::try_from(index).unwrap(); usize::try_from(size).unwrap()])
        .collect();
    let borrowed: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
    let region_address = StoredAddress::new(base);
    let owner = StoredAddress::new(128);
    let emitted = plan.serialize(&borrowed, region_address, owner).unwrap();
    assert_eq!(emitted.len() as u64, plan.region_size());
    let header = header(&plan, region_address);
    let header_bytes = header.serialize(width, LengthWidth::Eight).unwrap();
    let parsed_header = FractalHeapHeader::parse(&header_bytes, 0, width.get(), 8).unwrap();
    assert_eq!(parsed_header, header);
    let layout =
        FractalHeapBlockLayout::new(&parsed_header, width, LengthWidth::Eight, owner).unwrap();
    let image = FileRegion::new(
        FileRange::new(PhysicalAddress::new(base), emitted.len() as u64).unwrap(),
        &emitted,
    )
    .unwrap();
    let ids = FractalHeapIdLayout::new(&parsed_header, width, LengthWidth::Eight);
    let mut blocks = vec![layout.root().unwrap().unwrap()];
    let mut object_indices = Vec::new();
    let mut direct_count = 0;
    let mut indirect_count = 0;
    let mut allocated = 0;
    while let Some(block) = blocks.pop() {
        match block {
            FractalHeapBlock::Direct(block) => {
                direct_count += 1;
                allocated += block.decoded_size();
                let reference = block
                    .at(PhysicalAddress::new(block.address().get()))
                    .unwrap();
                reference.parse_checked(image).unwrap();
                for (index, object) in objects.iter().enumerate() {
                    let offset = plan.heap_offset(index);
                    if offset < block.heap_offset()
                        || offset >= block.heap_offset() + block.decoded_size()
                    {
                        continue;
                    }
                    let id = parsed_header
                        .encode_managed_id(offset, object.len() as u64)
                        .unwrap();
                    assert_eq!(
                        ids.parse(&id).unwrap().kind(),
                        FractalHeapIdKind::Managed {
                            heap_offset: offset,
                            object_length: object.len() as u64
                        }
                    );
                    let object_ref = reference
                        .parse_object(image, offset, object.len() as u64)
                        .unwrap();
                    assert_eq!(object_ref.parse(image).unwrap(), object);
                    object_indices.push(index);
                }
            }
            FractalHeapBlock::Indirect(block) => {
                indirect_count += 1;
                let view = block
                    .at(PhysicalAddress::new(block.address().get()))
                    .unwrap()
                    .parse(image)
                    .unwrap();
                let children: Vec<_> = view.children().map(Result::unwrap).collect();
                assert!(!children.is_empty());
                blocks.extend(children);
            }
        }
    }
    object_indices.sort_unstable();
    assert_eq!(object_indices, (0..sizes.len()).collect::<Vec<_>>());
    assert_eq!(direct_count, plan.directs.len());
    assert_eq!(indirect_count, plan.indirects.len());
    assert_eq!(allocated, plan.allocated_space());
    assert_eq!(
        plan.allocated_space()
            + plan
                .indirects
                .iter()
                .map(|block| plan.indirect_block_size(block.nrows).unwrap())
                .sum::<u64>(),
        plan.region_size()
    );
    let used: u64 = sizes.iter().sum();
    let mut cursor = 0;
    let mut capacity = 0;
    while cursor < plan.managed_space() {
        let (offset, size) = plan.locate(cursor).unwrap();
        assert_eq!(offset, cursor);
        capacity += size - plan.direct_header_size;
        cursor += size;
    }
    assert_eq!(cursor, plan.managed_space());
    assert_eq!(plan.free_space(), capacity - used);
    if plan.root_rows() == 0 {
        assert_eq!(plan.free_space(), start - plan.direct_header_size - used);
        assert_eq!(plan.allocation_iterator(), 0);
    } else {
        let last = plan.directs.last().unwrap();
        assert_eq!(plan.allocation_iterator(), last.heap_offset + last.size);
    }
}

#[test]
fn serializing_objects_of_other_sizes_is_rejected() {
    let plan = ManagedHeapPlan::attributes(&[5], OffsetWidth::Eight).unwrap();
    let error = plan
        .serialize(&[b"hello!"], StoredAddress::new(0), StoredAddress::new(0))
        .unwrap_err();
    assert!(
        matches!(
            error,
            FormatError::DataSizeMismatch {
                expected: 5,
                actual: 6
            }
        ),
        "{error:?}"
    );
}

#[test]
fn serializing_another_object_count_reports_an_internal_error() {
    let plan = ManagedHeapPlan::attributes(&[5], OffsetWidth::Eight).unwrap();
    let error = plan
        .serialize(&[], StoredAddress::new(0), StoredAddress::new(0))
        .unwrap_err();
    assert!(
        matches!(error, FormatError::Internal(ref detail) if detail == "managed heap emission received another object count"),
        "{error:?}"
    );
}

#[test]
fn block_addresses_above_32_bits_use_local_buffers() {
    let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Eight).unwrap();
    let base = StoredAddress::new(1u64 << 40);
    let owner = StoredAddress::new(1u64 << 39);
    let data = plan.serialize(&[&[37]], base, owner).unwrap();
    let header = header(&plan, base);
    let layout =
        FractalHeapBlockLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight, owner)
            .unwrap();
    let FractalHeapBlock::Direct(block) = layout.root().unwrap().unwrap() else {
        panic!("expected a direct root")
    };
    let reference = block.at(PhysicalAddress::new(base.get())).unwrap();
    let image = FileRegion::new(reference.range(), &data).unwrap();
    reference.parse_checked(image).unwrap();
    let object = reference
        .parse_object(image, plan.heap_offset(0), 1)
        .unwrap();
    assert_eq!(object.parse(image).unwrap(), &[37]);
}

#[test]
fn direct_blocks_include_a_checksum_over_the_zeroed_checksum_field() {
    let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Eight).unwrap();
    let base = StoredAddress::new(512);
    let owner = StoredAddress::new(128);
    let mut data = plan.serialize(&[&[37]], base, owner).unwrap();
    let header = header(&plan, base);
    let layout =
        FractalHeapBlockLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight, owner)
            .unwrap();
    let FractalHeapBlock::Direct(block) = layout.root().unwrap().unwrap() else {
        panic!("expected a direct root")
    };
    let reference = block.at(PhysicalAddress::new(base.get())).unwrap();
    let checksum_at = usize::try_from(plan.direct_header_size).unwrap() - CHECKSUM_LEN;
    let stored = u32::from_le_bytes(
        data[checksum_at..checksum_at + CHECKSUM_LEN]
            .try_into()
            .unwrap(),
    );
    let last = data.last_mut().unwrap();
    *last ^= 1;
    let image = FileRegion::new(reference.range(), &data).unwrap();
    let error = reference.parse_checked(image).unwrap_err();
    data[checksum_at..checksum_at + CHECKSUM_LEN].fill(0);
    let computed = crate::checksum::jenkins_lookup3(&data);
    assert!(
        matches!(error, FormatError::ChecksumMismatch { expected, computed: actual } if expected == stored && actual == computed),
        "{error:?}"
    );
    assert_ne!(stored, computed);
}
