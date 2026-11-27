use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;

use super::support::Fixture;
use crate::chunk_record;
use crate::chunk_record::ChunkRecord;

#[test]
fn the_pages_of_a_paged_data_block_are_contiguous_with_an_unpadded_last_page() {
    let fixture = Fixture::new(3_000, false);
    let block = fixture.data_block();
    let prefix = block.parse_range().unwrap();
    let pages: Vec<(u64, u64, u64)> = fixture
        .pages()
        .into_iter()
        .map(|page| {
            let range = page.parse_range();
            (
                range.start().get(),
                range.length(),
                page.slots().unwrap().start,
            )
        })
        .collect();
    let start = prefix.end().get();
    assert_eq!(prefix.length(), 4 + 2 + 8 + 1 + 4);
    assert_eq!(
        pages,
        vec![
            (start, 1_024 * 8 + 4, 0),
            (start + 1_024 * 8 + 4, 1_024 * 8 + 4, 1_024),
            (start + 2 * (1_024 * 8 + 4), 952 * 8 + 4, 2_048),
        ]
    );
    assert_eq!(pages[2].0 + pages[2].1, block.range().end().get());
}

#[rstest]
#[case::second(0b1011_0000, &[0, 2, 3])]
#[case::first_and_last(0b0110_0000, &[1, 2])]
#[case::all_but_the_partial_last(0b0001_0000, &[3])]
#[case::every_page(0b0000_0000, &[])]
fn a_cleared_page_bit_skips_its_page(
    #[case] bitmap: u8,
    #[case] pages: &[u64],
    #[values(false, true)] has_filters: bool,
) {
    let mut fixture = Fixture::new(4_000, has_filters);
    let at = fixture.data_block_at() + 4 + 2 + 8;
    fixture.bytes[at] = bitmap;
    fixture.restamp_prefix();
    let expected: Vec<(u64, ChunkRecord)> =
        chunk_record::slotted(chunk_record::chunks(8, 4_000, has_filters))
            .into_iter()
            .filter(|(slot, _)| pages.contains(&(slot / 1_024)))
            .collect();
    assert_eq!(fixture.records(8).unwrap(), expected);
}

#[test]
fn a_page_parses_from_a_region_holding_only_that_page() {
    let fixture = Fixture::new(3_000, true);
    let page = fixture.pages()[1];
    let read: Vec<(u64, ChunkRecord)> = page
        .parse(fixture.image().subregion(page.parse_range()).unwrap())
        .unwrap()
        .records(8)
        .collect::<Result<_, _>>()
        .unwrap();
    let expected: Vec<(u64, ChunkRecord)> =
        chunk_record::slotted(chunk_record::chunks(8, 3_000, true))
            .into_iter()
            .skip(1_024)
            .take(1_024)
            .collect();
    assert_eq!(read, expected);
}
