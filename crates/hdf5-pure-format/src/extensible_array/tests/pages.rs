use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use rstest::rstest;

use super::support::{FIRST_PAGED_SLOT, Fixture, WIDTHS};
use crate::chunk_record::{self, ChunkRecord, IndexSlots};
use crate::extensible_array::ExtensibleArrayPageRef;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[rstest]
fn a_paged_array_reads_back(
    #[values(
        FormatWidths { offsets: OffsetWidth::Four, lengths: LengthWidth::Four },
        FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight }
    )]
    widths: FormatWidths,
    #[values(false, true)] has_filters: bool,
) {
    let n = 140_000;
    let fixture = Fixture::at(0, widths, 8, n, has_filters);
    assert!(!fixture.paged_data_blocks().is_empty());
    assert_eq!(
        fixture.records(8).unwrap(),
        chunk_record::slotted(chunk_record::chunks(8, n, has_filters))
    );
}

#[test]
fn the_pages_of_a_paged_data_block_follow_its_prefix() {
    let fixture = Fixture::new(140_000, false);
    let block = fixture.paged_data_blocks()[0];
    let prefix = block.parse_range();
    assert_eq!(prefix.length(), 4 + 2 + 8 + 4 + 4);
    assert_eq!(
        block.slots().unwrap(),
        FIRST_PAGED_SLOT..FIRST_PAGED_SLOT + 2_048
    );
    let placed = |page: ExtensibleArrayPageRef| {
        let range = page.parse_range();
        (range.start().get(), range.length(), page.slots().unwrap())
    };
    let unparsed: Vec<(u64, u64, Range<u64>)> = block
        .pages()
        .unwrap()
        .unwrap()
        .pages()
        .map(|page| placed(page.unwrap()))
        .collect();
    let pages: Vec<(u64, u64, Range<u64>)> = fixture.pages(block).into_iter().map(placed).collect();
    let start = prefix.end().get();
    assert_eq!(fixture.data_blocks()[0].pages().unwrap(), None);
    let expected = vec![
        (
            start,
            1_024 * 8 + 4,
            FIRST_PAGED_SLOT..FIRST_PAGED_SLOT + 1_024,
        ),
        (
            start + 1_024 * 8 + 4,
            1_024 * 8 + 4,
            FIRST_PAGED_SLOT + 1_024..FIRST_PAGED_SLOT + 2_048,
        ),
    ];
    assert_eq!(unparsed, expected);
    assert_eq!(pages, expected);
    assert_eq!(pages[1].0 + pages[1].1, block.range().end().get());
}

#[rstest]
#[case::first(0b0111_0000, &[1, 2, 3])]
#[case::second(0b1011_0000, &[0, 2, 3])]
#[case::every_page(0b0000_0000, &[])]
fn a_cleared_page_bit_skips_its_page(
    #[case] bitmap: u8,
    #[case] pages: &[u64],
    #[values(false, true)] has_filters: bool,
) {
    let n = FIRST_PAGED_SLOT + 4_096;
    let mut fixture = Fixture::new(n, has_filters);
    let super_block = fixture.super_blocks()[9].parse_range();
    let at = fixture.local(super_block.start()) + 4 + 2 + 8 + 4;
    fixture.bytes[at] = bitmap;
    fixture.restamp(super_block);
    let expected: Vec<(u64, ChunkRecord)> =
        chunk_record::slotted(chunk_record::chunks(8, n, has_filters))
            .into_iter()
            .filter(|&(slot, _)| {
                slot < FIRST_PAGED_SLOT || pages.contains(&((slot - FIRST_PAGED_SLOT) / 1_024))
            })
            .collect();
    assert_eq!(fixture.records(8).unwrap(), expected);
}

#[test]
fn an_unallocated_block_and_an_uninitialized_page_have_no_reference() {
    let records = chunk_record::chunks(8, 2, false);
    let occupied = [FIRST_PAGED_SLOT + 1_030, FIRST_PAGED_SLOT + 3 * 2_048 + 10];
    let slots = IndexSlots::new(&records, &occupied, 140_000).unwrap();
    let fixture = Fixture::with_slots(0, WIDTHS, 8, &slots, false);
    assert_eq!(fixture.super_blocks().len(), 1);
    let blocks = fixture.data_blocks();
    assert_eq!(
        blocks
            .iter()
            .map(|block| block.slots().unwrap())
            .collect::<Vec<_>>(),
        vec![
            FIRST_PAGED_SLOT..FIRST_PAGED_SLOT + 2_048,
            FIRST_PAGED_SLOT + 3 * 2_048..FIRST_PAGED_SLOT + 4 * 2_048,
        ]
    );
    let pages: Vec<Range<u64>> = blocks
        .iter()
        .flat_map(|&block| fixture.pages(block))
        .map(|page| page.slots().unwrap())
        .collect();
    assert_eq!(
        pages,
        vec![
            FIRST_PAGED_SLOT + 1_024..FIRST_PAGED_SLOT + 2_048,
            FIRST_PAGED_SLOT + 3 * 2_048..FIRST_PAGED_SLOT + 3 * 2_048 + 1_024,
        ]
    );
}

#[rstest]
#[case::inside_the_index_block(2, 0, 0, 0)]
#[case::inside_the_first_data_block(5, 1, 0, 0)]
#[case::inside_the_first_super_block(300, 6 + 1, 1, 0)]
#[case::inside_the_first_page(FIRST_PAGED_SLOT + 10, 6 + 184 + 1, 10, 1)]
fn a_block_or_page_past_the_max_index_set_holds_no_set_slots(
    #[case] max_index_set: u64,
    #[case] data_blocks: usize,
    #[case] super_blocks: usize,
    #[case] pages: usize,
) {
    let mut fixture = Fixture::new(FIRST_PAGED_SLOT + 4_096, false);
    let header = fixture.header().unwrap().range();
    let at = fixture.local(header.start()) + 12 + 4 * 8;
    fixture.bytes[at..at + 8].copy_from_slice(&max_index_set.to_le_bytes());
    fixture.restamp(header);
    let all_super_blocks = fixture.super_blocks();
    assert_eq!(all_super_blocks.len(), 10);
    assert_eq!(
        all_super_blocks
            .iter()
            .filter(|block| block.holds_set_slots())
            .count(),
        super_blocks
    );
    let all_data_blocks = fixture.data_blocks();
    assert_eq!(all_data_blocks.len(), 6 + 184 + 2);
    assert_eq!(
        all_data_blocks
            .iter()
            .filter(|block| block.holds_set_slots())
            .count(),
        data_blocks
    );
    assert_eq!(
        fixture
            .paged_data_blocks()
            .into_iter()
            .map(|block| fixture.pages(block).len())
            .sum::<usize>(),
        pages
    );
}
