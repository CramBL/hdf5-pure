use test_util::fractal_heap;
use test_util::widths::Widths;

use super::*;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::test_support;

#[test]
fn the_indirect_block_walk_of_a_filtered_heap_is_unsupported() {
    let mut header = test_support::dtable_header(512, 65536, 4);
    header.filtering = test_support::filtered_heap();
    let layout = FractalHeapLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    let mut block = b"FHIB".to_vec();
    block.resize(256, 0);
    assert_eq!(
        layout.find_child_for_offset(&block, 2, 0, 0),
        Err(FormatError::UnsupportedFilteredHeapObject)
    );

    // In an unfiltered heap, the function returns `None` for an offset past the block's space.
    header.filtering = FractalHeapFiltering::Unfiltered;
    let layout = FractalHeapLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    assert_eq!(
        layout.find_child_for_offset(&block, 2, 0, u64::MAX),
        Ok(None)
    );
}

fn dtable_layout(header: &FractalHeapHeader) -> FractalHeapLayout {
    FractalHeapLayout::new(header, OffsetWidth::Eight, LengthWidth::Eight).unwrap()
}

#[test]
fn layout_derives_direct_rows_from_the_doubling_table() {
    assert_eq!(
        dtable_layout(&test_support::dtable_header(512, 65536, 4)).direct_rows(),
        9
    );
    assert_eq!(
        dtable_layout(&test_support::dtable_header(4096, 65536, 4)).direct_rows(),
        6
    );
    let smallest = FractalHeapHeader {
        max_heap_size: 12,
        ..test_support::dtable_header(512, 512, 4)
    };
    assert_eq!(dtable_layout(&smallest).direct_rows(), 2);
}

#[test]
fn layout_derives_child_indirect_rows_from_the_slot_size() {
    let layout = dtable_layout(&test_support::dtable_header(512, 65536, 4));
    assert_eq!(layout.block_size_for_row(9), Ok(131072));
    assert_eq!(layout.child_indirect_rows(9), Ok(7));
}

#[test]
fn an_indirect_block_locates_the_child_that_holds_an_offset() {
    let header = FractalHeapHeader::parse(
        &fractal_heap::Header::new(0x100).build(Widths::EIGHT),
        0,
        8,
        8,
    )
    .unwrap();
    let mut block = fractal_heap::INDIRECT_BLOCK_SIGNATURE.to_vec();
    block.push(0);
    block.extend_from_slice(&0u64.to_le_bytes());
    block.extend_from_slice(&[0, 0]);
    for address in [0x1000u64, u64::MAX, 0x2000, u64::MAX] {
        block.extend_from_slice(&address.to_le_bytes());
    }

    let layout = FractalHeapLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    assert_eq!(layout.indirect_block_entries_len(1), Ok(block.len() as u64));
    assert_eq!(
        layout.find_child_for_offset(&block, 1, 0, 300),
        Ok(Some(FractalHeapChild::Direct {
            addr: StoredAddress::new(0x2000),
            block_size: 128,
            heap_offset: 256,
        }))
    );
    assert_eq!(layout.find_child_for_offset(&block, 1, 0, 200), Ok(None));
}
