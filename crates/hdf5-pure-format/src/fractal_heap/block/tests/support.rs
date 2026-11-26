use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::super::SIGNATURE_LEN;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::checksum;
use crate::filter_pipeline::{FilterDescription, FilterPipeline, VERSION_TWO};
use crate::fractal_heap::block::{
    CHECKSUM_LEN, DIRECT_SIGNATURE, FractalHeapBlock, FractalHeapBlockLayout,
    FractalHeapDirectBlock, FractalHeapIndirectBlock, INDIRECT_SIGNATURE, VERSION,
};
use crate::fractal_heap::{
    FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED, FractalHeapFiltering, FractalHeapHeader,
};
use crate::width::{LengthWidth, OffsetWidth};

pub(super) fn heap(
    offsets: OffsetWidth,
    rows: u16,
    checksummed: bool,
    filtered: bool,
) -> FractalHeapBlockLayout {
    let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
    let mut header =
        FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    header.current_rows_in_root_indirect_block = rows;
    header.flags = if checksummed {
        FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED
    } else {
        0
    };
    if filtered {
        header.filtering = FractalHeapFiltering::Filtered {
            root_direct_block_size: 47,
            root_filter_mask: 1,
            pipeline: FilterPipeline::new(
                VERSION_TWO,
                vec![FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 1,
                    client_data: vec![6],
                }],
            )
            .unwrap(),
        };
    }
    FractalHeapBlockLayout::new(&header, offsets, LengthWidth::Eight, StoredAddress::new(96))
        .unwrap()
}

fn push_number(bytes: &mut Vec<u8>, number: u64, width: usize) {
    bytes.extend_from_slice(&number.to_le_bytes()[..width]);
}

pub(super) fn prefix(heap: &FractalHeapBlockLayout, signature: &[u8; 4], offset: u64) -> Vec<u8> {
    let mut bytes = signature.to_vec();
    bytes.push(VERSION);
    push_number(
        &mut bytes,
        heap.owner.get(),
        usize::from(heap.layout.offset_width().get()),
    );
    push_number(&mut bytes, offset, heap.layout.block_offset_size());
    bytes
}

pub(super) fn direct_bytes(heap: &FractalHeapBlockLayout) -> Vec<u8> {
    let mut bytes = prefix(heap, &DIRECT_SIGNATURE, 0);
    let checksum_at = bytes.len();
    bytes.resize(128, 0);
    let data_at = checksum_at + if heap.checksummed { CHECKSUM_LEN } else { 0 };
    bytes[data_at..].fill(0x5A);
    if heap.checksummed {
        let computed = checksum::jenkins_lookup3(&bytes);
        bytes[checksum_at..checksum_at + CHECKSUM_LEN].copy_from_slice(&computed.to_le_bytes());
    }
    bytes
}

pub(super) fn indirect_bytes(
    heap: &FractalHeapBlockLayout,
    entries: &[(usize, usize, u64)],
) -> Vec<u8> {
    let mut bytes = prefix(heap, &INDIRECT_SIGNATURE, 0);
    let width = heap.layout.offset_width();
    let undefined = StoredAddress::undefined(width).get();
    for row in 0..usize::from(heap.root_rows) {
        for column in 0..heap.layout.table_width() {
            let address = entries
                .iter()
                .find(|&&(r, c, _)| r == row && c == column)
                .map_or(undefined, |&(_, _, addr)| addr);
            push_number(&mut bytes, address, usize::from(width.get()));
            if row < heap.layout.direct_rows() && heap.filtering.is_filtered() {
                push_number(
                    &mut bytes,
                    if address == undefined { 0 } else { 47 },
                    usize::from(heap.layout.length_width().get()),
                );
                bytes.extend_from_slice(&1u32.to_le_bytes());
            }
        }
    }
    let stored = checksum::jenkins_lookup3(&bytes);
    bytes.extend_from_slice(&stored.to_le_bytes());
    assert_eq!(
        bytes.len() as u64,
        heap.layout.indirect_block_size(heap.root_rows).unwrap()
    );
    bytes
}

pub(super) fn region(at: u64, bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(at), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
}

pub(super) fn direct_root(heap: &FractalHeapBlockLayout) -> FractalHeapDirectBlock<'_> {
    let Some(FractalHeapBlock::Direct(block)) = heap.root().unwrap() else {
        panic!("expected direct root");
    };
    block
}

pub(super) fn indirect_root(heap: &FractalHeapBlockLayout) -> FractalHeapIndirectBlock<'_> {
    let Some(FractalHeapBlock::Indirect(block)) = heap.root().unwrap() else {
        panic!("expected indirect root");
    };
    block
}
// The fixture heaps store eight-byte offsets, so the block offset follows an eight-byte owner.
pub(super) const OWNER_AT: usize = SIGNATURE_LEN + 1;
pub(super) const BLOCK_OFFSET_AT: usize = OWNER_AT + 8;

pub(super) const FILTER_DEFLATE: u16 = 1;
