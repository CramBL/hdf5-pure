use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRegion;
use rstest::rstest;

use super::support::Fixture;
use super::support::WIDTHS;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::IndexSlots;
use crate::fixed_array::FixedArrayDataBlock;
use crate::fixed_array::FixedArrayDataBlockRef;
use crate::fixed_array::FixedArrayHeaderLocator;
use crate::fixed_array::FixedArrayPageBitmap;
use crate::fixed_array::FixedArrayPlan;

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

#[test]
fn page_bitmap_equality_includes_the_owning_header() {
    let prepare = |header_address| {
        let header_address = StoredAddress::new(header_address);
        let locator = FixedArrayHeaderLocator::new(WIDTHS, BaseAddress::ZERO, header_address);
        let [(_, header), (_, block)] = FixedArrayPlan::new(3, 8, WIDTHS, false)
            .unwrap()
            .with_page_bits(1)
            .unwrap()
            .place(header_address, StoredAddress::new(256))
            .unwrap()
            .serialize_blocks(&IndexSlots::new(&[], &[], 3).unwrap())
            .unwrap();
        let header = locator
            .parse(FileRegion::new(locator.parse_range().unwrap(), &header).unwrap())
            .unwrap();
        (header.data_block().unwrap(), block)
    };
    let (first, first_bytes) = prepare(32);
    let (second, second_bytes) = prepare(64);
    assert_eq!(first.range(), second.range());
    assert_eq!(bitmap(&first, &first_bytes), bitmap(&first, &first_bytes));
    assert_ne!(bitmap(&first, &first_bytes), bitmap(&second, &second_bytes));
}

fn bitmap<'a>(block: &FixedArrayDataBlockRef, bytes: &'a [u8]) -> FixedArrayPageBitmap<'a> {
    let region = FileRegion::new(block.range(), bytes).unwrap();
    let FixedArrayDataBlock::PageBitmap(bitmap) = block.parse(region).unwrap() else {
        panic!("expected a paged block");
    };
    bitmap
}
