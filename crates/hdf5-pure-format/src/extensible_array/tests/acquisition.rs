use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use rstest::rstest;

use super::support::{self, ARRAY_AT, FIRST_PAGED_SLOT, Fixture, WIDTHS};
use crate::chunk_record::{self, ChunkRecord, IndexSlots};
use crate::extensible_array::{
    self, ExtensibleArrayDataBlock, ExtensibleArrayDataBlockRef, ExtensibleArraySuperBlockRef,
    SlotOccupancy,
};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[test]
fn read_inline_only() {
    let os: u8 = 8;
    let ls: u8 = 8;
    let osv = os as usize;
    let num_chunks = 2usize;
    let chunk_byte_size = 20u64 * 8; // 20 elements × 8 bytes

    let mut file_data = vec![0u8; 0x3000];

    // EAHD at offset 0x100
    let header_offset = 0x100usize;
    let index_block_offset = 0x200usize;

    // Build EAHD
    file_data[header_offset..header_offset + 4].copy_from_slice(b"EAHD");
    file_data[header_offset + 4] = 0; // version
    file_data[header_offset + 5] = 0; // `client_id` = non-filtered
    file_data[header_offset + 6] = osv as u8; // `element_size`
    file_data[header_offset + 7] = 10; // `max_nelmts_bits`
    file_data[header_offset + 8] = num_chunks as u8; // `idx_blk_elmts` (all inline)
    file_data[header_offset + 9] = 4; // `min_dblk_nelmts`
    file_data[header_offset + 10] = 2; // `super_blk_min_data_ptrs`
    file_data[header_offset + 11] = 8; // log2 of the maximum elements per data block page
    // 6 stats fields (each 8 bytes), `max_idx_set` at stat[4]
    file_data[header_offset + 44..header_offset + 52]
        .copy_from_slice(&(num_chunks as u64).to_le_bytes());
    file_data[header_offset + 60..header_offset + 68]
        .copy_from_slice(&(index_block_offset as u64).to_le_bytes());
    test_util::checksum::restamp(&mut file_data, header_offset, support::eahd_len(os, ls));

    // Build EAIB at `index_block_offset`
    file_data[index_block_offset..index_block_offset + 4].copy_from_slice(b"EAIB");
    file_data[index_block_offset + 4] = 0; // version
    file_data[index_block_offset + 5] = 0; // `client_id`
    file_data[index_block_offset + 6..index_block_offset + 14]
        .copy_from_slice(&(header_offset as u64).to_le_bytes());

    // Inline elements
    let elem_start = index_block_offset + 6 + osv;
    let base_addr = 0x1000u64;
    for i in 0..num_chunks {
        let addr = base_addr + i as u64 * chunk_byte_size;
        let p = elem_start + i * osv;
        file_data[p..p + osv].copy_from_slice(&addr.to_le_bytes());
    }

    // The index block is sized by the header's geometry whatever the array
    // holds: prefix, two inline slots, then a pointer per direct data block
    // (min_dblk_nelmts=4 and super_blk_min_data_ptrs=2 give two) and one per
    // super block (nine levels less the two taken as direct, so seven).
    // The pointers stay zero, and the read stops at the inline slots.
    test_util::checksum::restamp(
        &mut file_data,
        index_block_offset,
        (6 + osv) + 2 * osv + 2 * osv + 7 * osv + 4,
    );

    assert_eq!(
        support::records_at(&file_data, header_offset, chunk_byte_size),
        support::back_to_back(0..2, base_addr, chunk_byte_size)
    );
}

#[test]
fn read_inline_plus_data_blocks() {
    let os: u8 = 8;
    let ls: u8 = 8;
    let osv = os as usize;
    let chunk_byte_size = 10u64 * 8; // 10 elements × 8 bytes
    let idx_blk_elmts = 2u8;
    let min_dblk_nelmts = 2u8;
    let sblk_min = 2u8;
    let total_chunks = 4usize; // 2 inline and 2 in the first data block

    let mut file_data = vec![0u8; 0x5000];
    let header_offset = 0x100usize;
    let index_block_offset = 0x200usize;
    let data_block_offset = 0x300usize;

    // EAHD
    file_data[header_offset..header_offset + 4].copy_from_slice(b"EAHD");
    file_data[header_offset + 4] = 0;
    file_data[header_offset + 5] = 0; // `client_id`
    file_data[header_offset + 6] = osv as u8; // `element_size`
    file_data[header_offset + 7] = 10;
    file_data[header_offset + 8] = idx_blk_elmts;
    file_data[header_offset + 9] = min_dblk_nelmts;
    file_data[header_offset + 10] = sblk_min;
    file_data[header_offset + 11] = 8;
    // 6 stats fields (each 8 bytes), `max_idx_set` at stat[4] (offset 12 + 4*8 = 44)
    file_data[header_offset + 44..header_offset + 52]
        .copy_from_slice(&(total_chunks as u64).to_le_bytes());
    // The index block address at offset 12 + 6*8 = 60
    file_data[header_offset + 60..header_offset + 68]
        .copy_from_slice(&(index_block_offset as u64).to_le_bytes());
    test_util::checksum::restamp(&mut file_data, header_offset, support::eahd_len(os, ls));

    // EAIB
    file_data[index_block_offset..index_block_offset + 4].copy_from_slice(b"EAIB");
    file_data[index_block_offset + 4] = 0;
    file_data[index_block_offset + 5] = 0;
    file_data[index_block_offset + 6..index_block_offset + 14]
        .copy_from_slice(&(header_offset as u64).to_le_bytes());

    let mut pos = index_block_offset + 6 + osv;

    // Inline elements (2 chunks)
    let base_addr = 0x1000u64;
    for i in 0..idx_blk_elmts as usize {
        let addr = base_addr + i as u64 * chunk_byte_size;
        file_data[pos..pos + osv].copy_from_slice(&addr.to_le_bytes());
        pos += osv;
    }

    // For minimum two data block pointers, the direct blocks hold 2 and 4 elements.
    let n_direct_dblks = 2;
    file_data[pos..pos + osv].copy_from_slice(&(data_block_offset as u64).to_le_bytes());
    pos += osv;
    for _ in 1..n_direct_dblks {
        file_data[pos..pos + osv].copy_from_slice(&u64::MAX.to_le_bytes());
        pos += osv;
    }

    // EADB at `data_block_offset` (`min_dblk_nelmts` elements)
    file_data[data_block_offset..data_block_offset + 4].copy_from_slice(b"EADB");
    file_data[data_block_offset + 4] = 0;
    file_data[data_block_offset + 5] = 0;
    file_data[data_block_offset + 6..data_block_offset + 14]
        .copy_from_slice(&(header_offset as u64).to_le_bytes());
    // The block offset takes ceil(10 / 8) = 2 bytes, and is 0 for the first data block.
    let blk_off_size = (10usize).div_ceil(8); // max_nelmts_bits=10
    let mut dbpos = data_block_offset + 6 + osv + blk_off_size;
    for i in 0..min_dblk_nelmts as usize {
        let addr = base_addr + (idx_blk_elmts as u64 + i as u64) * chunk_byte_size;
        file_data[dbpos..dbpos + osv].copy_from_slice(&addr.to_le_bytes());
        dbpos += osv;
    }

    // Prefix, two inline slots, the two direct pointers written above, and
    // one pointer per remaining super block (ten levels less the two taken
    // as direct).
    test_util::checksum::restamp(
        &mut file_data,
        index_block_offset,
        (6 + osv) + 2 * osv + n_direct_dblks * osv + 8 * osv + 4,
    );
    // Prefix, block offset, `min_dblk_nelmts` element slots, checksum.
    test_util::checksum::restamp(
        &mut file_data,
        data_block_offset,
        (6 + osv) + blk_off_size + min_dblk_nelmts as usize * osv + 4,
    );

    assert_eq!(
        support::records_at(&file_data, header_offset, chunk_byte_size),
        support::back_to_back(0..4, base_addr, chunk_byte_size)
    );
}

#[rstest]
fn an_array_reads_back_at_every_width(
    #[values(
        FormatWidths { offsets: OffsetWidth::Two, lengths: LengthWidth::Two },
        FormatWidths { offsets: OffsetWidth::Four, lengths: LengthWidth::Four },
        FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight }
    )]
    widths: FormatWidths,
    #[values(0, 3, 4, 20, 244, 2_000)] n: u64,
    #[values(false, true)] has_filters: bool,
    #[values(8, 300, 100_000, 1 << 32)] chunk_bytes: u64,
) {
    let fixture = Fixture::at(0, widths, chunk_bytes, n, has_filters);
    assert_eq!(
        fixture.records(chunk_bytes).unwrap(),
        chunk_record::slotted(chunk_record::chunks(chunk_bytes, n, has_filters))
    );
}

#[rstest]
fn the_block_ranges_tile_the_array(#[values(4, 244, 2_000, 140_000)] n: u64) {
    let fixture = Fixture::new(n, false);
    let stats = extensible_array::extensible_array_layout(
        SlotOccupancy::Dense(n),
        n,
        8,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        false,
    )
    .unwrap()
    .stats;
    let header = fixture.header().unwrap();
    let mut ranges = vec![header.range(), header.index_block().unwrap().parse_range()];
    ranges.extend(
        fixture
            .super_blocks()
            .iter()
            .map(ExtensibleArraySuperBlockRef::parse_range),
    );
    ranges.extend(
        fixture
            .data_blocks()
            .iter()
            .map(ExtensibleArrayDataBlockRef::range),
    );
    assert_eq!(
        ranges.len() as u64,
        2 + stats.nsuper_blks + stats.ndata_blks
    );
    ranges.sort_by_key(|range| range.start());
    assert_eq!(ranges[0].start().get(), ARRAY_AT);
    for pair in ranges.windows(2) {
        assert_eq!(pair[0].end(), pair[1].start());
    }
    assert_eq!(
        ranges.last().unwrap().end().get(),
        fixture.bytes.len() as u64
    );
}

#[rstest]
#[case::inline_elements(&[1, 3], 4)]
#[case::direct_data_blocks(&[0, 30, 200], 244)]
#[case::super_blocks(&[2, 1_500, 1_999], 2_000)]
#[case::pages(&[FIRST_PAGED_SLOT + 1_030, FIRST_PAGED_SLOT + 3 * 2_048 + 10], 140_000)]
fn an_array_with_unallocated_blocks_reads_its_chunks(
    #[case] occupied: &[u64],
    #[case] len: u64,
    #[values(false, true)] has_filters: bool,
) {
    let records = chunk_record::chunks(8, occupied.len() as u64, has_filters);
    let slots = IndexSlots::new(&records, occupied, len).unwrap();
    let fixture = Fixture::with_slots(0, WIDTHS, 8, &slots, has_filters);
    assert_eq!(
        fixture.records(8).unwrap(),
        occupied.iter().copied().zip(records).collect::<Vec<_>>()
    );
}

#[rstest]
#[case::none(0)]
#[case::inside_the_index_block(2)]
#[case::at_the_first_data_block(4)]
#[case::inside_a_data_block(5)]
#[case::inside_a_super_block(1_000)]
#[case::inside_a_page(FIRST_PAGED_SLOT + 1_500)]
#[case::past_the_last_chunk(200_000)]
#[case::full_width(u64::MAX)]
fn the_records_stop_at_the_max_index_set(#[case] max_index_set: u64) {
    let n = FIRST_PAGED_SLOT + 2_048;
    let mut fixture = Fixture::new(n, true);
    let header = fixture.header().unwrap().range();
    let at = fixture.local(header.start()) + 12 + 4 * 8;
    fixture.bytes[at..at + 8].copy_from_slice(&max_index_set.to_le_bytes());
    fixture.restamp(header);
    let expected: Vec<(u64, ChunkRecord)> = chunk_record::slotted(chunk_record::chunks(8, n, true))
        .into_iter()
        .take(usize::try_from(max_index_set).unwrap_or(usize::MAX))
        .collect();
    assert_eq!(fixture.records(8).unwrap(), expected);
}

#[test]
fn a_slot_parses_from_the_regions_on_its_path_alone() {
    let n = 140_000;
    let slot = FIRST_PAGED_SLOT + 2 * 2_048 + 1_100;
    let fixture = Fixture::new(n, true);
    let only = |range: FileRange| fixture.read(range).unwrap();
    let locator = fixture.locator();
    let header = locator.parse(only(locator.parse_range().unwrap())).unwrap();
    let index_block = header.index_block().unwrap();
    let index_block = index_block.parse(only(index_block.parse_range())).unwrap();
    let super_block = index_block
        .super_blocks()
        .map(Result::unwrap)
        .find(|block| block.slots().unwrap().contains(&slot))
        .unwrap();
    let super_block = super_block.parse(only(super_block.parse_range())).unwrap();
    let data_block = super_block
        .data_blocks()
        .map(Result::unwrap)
        .find(|block| block.slots().unwrap().contains(&slot))
        .unwrap();
    let ExtensibleArrayDataBlock::Pages(pages) =
        data_block.parse(only(data_block.parse_range())).unwrap()
    else {
        panic!("the data block {data_block:?} is not paged");
    };
    let page = pages
        .pages()
        .map(Result::unwrap)
        .find(|page| page.slots().unwrap().contains(&slot))
        .unwrap();
    let records: Vec<(u64, ChunkRecord)> = page
        .parse(only(page.parse_range()))
        .unwrap()
        .records(8)
        .collect::<Result<_, _>>()
        .unwrap();
    let expected: Vec<(u64, ChunkRecord)> = chunk_record::slotted(chunk_record::chunks(8, n, true))
        .into_iter()
        .skip(usize::try_from(FIRST_PAGED_SLOT + 2 * 2_048 + 1_024).unwrap())
        .take(1_024)
        .collect();
    assert_eq!(records, expected);
}

#[test]
fn an_undefined_index_block_address_leaves_the_array_without_an_index_block() {
    let mut fixture = Fixture::new(5, false);
    let header = fixture.header().unwrap().range();
    let at = fixture.local(header.start()) + 12 + 6 * 8;
    fixture.bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    fixture.restamp(header);
    let array = fixture.header().unwrap();
    assert_eq!(array.index_block(), None);
    assert!(
        array
            .header()
            .index_block_address()
            .is_undefined(WIDTHS.offsets)
    );
    assert_eq!(array.header().stored_statistics().max_idx_set, 5);
    assert_eq!(array.range().length(), 12 + 6 * 8 + 8 + 4);
    assert_eq!(fixture.records(8).unwrap(), Vec::new());
}

#[rstest]
#[case::plain(0)]
#[case::userblock(512)]
#[case::above_u32(1 << 40)]
fn an_array_reads_at_its_physical_position(#[case] base: u64, #[values(2_000, 140_000)] n: u64) {
    let fixture = Fixture::at(base, WIDTHS, 8, n, true);
    assert_eq!(
        fixture.header().unwrap().range().start().get(),
        base + ARRAY_AT
    );
    assert_eq!(
        fixture.records(8).unwrap(),
        chunk_record::slotted(chunk_record::chunks(8, n, true))
    );
}
