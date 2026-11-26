use hdf5_pure_core::__private::StoredAddressExt;
use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use crate::checksum;
use crate::free_space_manager::FreeSpaceManagerHeader;
use crate::free_space_manager::FreeSpaceManagerHeaderRef;
use crate::width::FormatWidths;

use super::*;

#[test]
fn parses_c_written_single_and_first_row_sections_with_reconstructed_ghost_counts() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = FractalHeapHeader::parse_bytes(
        &fractal_heap::dense_link_free_space_header(),
        widths.offsets,
        widths.lengths,
    )
    .unwrap();
    let header_bytes = fractal_heap::dense_link_free_space_manager();
    let header_ref = FreeSpaceManagerHeaderRef::new(widths, PhysicalAddress::new(1403)).unwrap();
    let frame = header_ref
        .parse(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
        .unwrap();
    assert_eq!(
        frame.section_list_addr(),
        DecodedStoredAddress::Defined(StoredAddress::new(5069))
    );
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(1403), PhysicalAddress::new(5069))
        .unwrap();
    let bytes = fractal_heap::dense_link_free_space_sections();
    let parsed = reference
        .parse(FileRegion::new(reference.parse_range(), &bytes).unwrap())
        .unwrap();
    assert_eq!((parsed.total_sections(), parsed.total_space()), (10, 10933));
    assert_eq!(parsed.sections().len(), 7);
    let FractalHeapFreeSection::FirstRow { row, indirect } = parsed.sections()[4] else {
        panic!("expected a first-row section")
    };
    assert_eq!(
        row,
        FractalHeapRowSection {
            class: FractalHeapSectionClass::FirstRow,
            offset: 0,
            size: NonZeroU64::new(491).unwrap(),
            row: 0,
            column: 0,
            blocks: NonZeroU16::new(4).unwrap()
        }
    );
    assert_eq!(
        indirect,
        FractalHeapIndirectSection {
            block_offset: 0,
            rows: 8,
            start_row: 0,
            start_column: 0,
            blocks: NonZeroU64::new(16).unwrap(),
            first: true
        }
    );
    assert_eq!(indirect.class(), FractalHeapSectionClass::Indirect);
    assert_eq!(
        manager.row_section(indirect, 3).unwrap(),
        FractalHeapRowSection {
            class: FractalHeapSectionClass::NormalRow,
            offset: 8192,
            size: NonZeroU64::new(2027).unwrap(),
            row: 3,
            column: 0,
            blocks: NonZeroU16::new(4).unwrap()
        }
    );
}

#[rstest]
#[case::two_byte(2, 2)]
#[case::four_byte(4, 4)]
#[case::eight_byte(8, 8)]
#[case::mixed_widths(8, 4)]
fn parses_single_sections_in_a_direct_root(#[case] offsets: u8, #[case] lengths: u8) {
    let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
    let mut heap = heap(widths);
    heap.current_rows_in_root_indirect_block = 0;
    heap.managed_space = heap.starting_block_size;
    let overhead = METADATA_PREFIX_LEN + u64::from(offsets) + 2;
    let records = single_record(FractalHeapSectionClass::Single as u8, overhead, 10);
    let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let parsed = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap();
    assert_eq!(
        parsed.sections(),
        &[FractalHeapFreeSection::Single {
            offset: overhead,
            size: NonZeroU64::new(10).unwrap()
        }]
    );
    assert_eq!((parsed.total_sections(), parsed.total_space()), (1, 10));
}

#[test]
fn reconstructs_normal_rows_and_indirect_children_with_ghost_totals_and_spare_capacity() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let mut heap = heap(widths);
    heap.current_rows_in_root_indirect_block = 6;
    heap.managed_space = 16384;
    let records = first_row_record(1, 113, 0, 0, 0, 24);
    let (header, list) = encoded_manager(widths, &records, 17, 1, 16, 3841, 64);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    assert_eq!(
        reference.range().length(),
        reference.parse_range().length() + 64
    );
    let parsed = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap();
    assert_eq!((parsed.total_sections(), parsed.total_space()), (17, 3841));
    let FractalHeapFreeSection::FirstRow { indirect, .. } = parsed.sections()[0] else {
        panic!("expected first row")
    };
    let child = manager.child_section(indirect, 5, 0).unwrap();
    assert_eq!(
        (
            child.class(),
            child.block_offset(),
            child.rows(),
            child.blocks()
        ),
        (FractalHeapSectionClass::Indirect, 8192, 3, 12)
    );
    assert_eq!(
        manager.row_section(child, 0).unwrap().class(),
        FractalHeapSectionClass::NormalRow
    );
}

#[rstest]
#[case::normal_row(FractalHeapSectionClass::NormalRow as u8)]
#[case::indirect(FractalHeapSectionClass::Indirect as u8)]
#[case::reserved(4)]
#[case::reserved_high(255)]
fn reports_unsupported_serialized_classes(#[case] class_id: u8) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let records = single_record(class_id, 15, 10);
    let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let err = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap_err();
    assert_eq!(
        err,
        FormatError::UnsupportedSerializedSectionClass {
            client_id: CLIENT_ID,
            class_id
        }
    );
}

#[rstest]
#[case::zero_blocks(
    0,
    0,
    0,
    0,
    0,
    113,
    "the indirect section's row, column, or block count is outside its doubling table"
)]
#[case::row_outside(
    0,
    4,
    0,
    1,
    0,
    113,
    "the indirect section's row, column, or block count is outside its doubling table"
)]
#[case::column_outside(
    0,
    0,
    4,
    1,
    0,
    113,
    "the indirect section's row, column, or block count is outside its doubling table"
)]
#[case::blocks_outside(
    0,
    3,
    3,
    2,
    0,
    113,
    "the indirect section's row, column, or block count is outside its doubling table"
)]
#[case::offset_disagrees(
    0,
    0,
    0,
    1,
    1,
    113,
    "the first-row record disagrees with its reconstructed first row"
)]
#[case::size_disagrees(
    0,
    0,
    0,
    1,
    0,
    112,
    "the first-row record disagrees with its reconstructed first row"
)]
fn rejects_first_row_geometry(
    #[case] block_offset: u64,
    #[case] row: u16,
    #[case] column: u16,
    #[case] blocks: u16,
    #[case] offset: u64,
    #[case] size: u64,
    #[case] reason: &str,
) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let mut records = first_row_record(1, size, offset, row, column, blocks);
    records[6..8].copy_from_slice(&u16::try_from(block_offset).unwrap().to_le_bytes());
    let (header, list) = encoded_manager(widths, &records, 1, 1, 0, size, 0);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let err = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap_err();
    assert_eq!(err, FormatError::InvalidFreeSpaceManager(reason.into()));
}

#[test]
fn rejects_a_short_supplied_region_separately_from_truncated_records() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let (header, list) = encoded_manager(
        widths,
        &first_row_record(1, 113, 0, 0, 0, 1),
        1,
        1,
        0,
        113,
        0,
    );
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let short = FileRange::new(PhysicalAddress::new(4096), list.len().to_u64() - 1).unwrap();
    let err = reference
        .parse(FileRegion::new(short, &list[..list.len() - 1]).unwrap())
        .unwrap_err();
    assert_eq!(
        err,
        FormatError::InsufficientRegion {
            required_start: 4096,
            required_length: reference.parse_range().length(),
            provided_start: 4096,
            provided_length: short.length()
        }
    );
    let (header, list) = encoded_manager(
        widths,
        &first_row_record(1, 113, 0, 0, 0, 1)[..12],
        1,
        1,
        0,
        113,
        0,
    );
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let err = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap_err();
    assert_eq!(
        err,
        FormatError::InvalidFreeSpaceManager(
            "a section record field runs past the used section list".into()
        )
    );
}

#[rstest]
#[case::file_client(5, 1, "the client ID is 1, not 0, the ID of the fractal heap client")]
#[case::reserved_client(5, 2, "the client ID is 2, not 0, the ID of the fractal heap client")]
#[case::count_sum(14, 2, "the heap geometry and manager header fields disagree")]
#[case::class_count(38, 3, "the heap geometry and manager header fields disagree")]
#[case::heap_width(44, 24, "the heap geometry and manager header fields disagree")]
#[case::maximum_section_size(47, 2, "the heap geometry and manager header fields disagree")]
fn rejects_client_specific_header_fields(
    #[case] at: usize,
    #[case] value: u8,
    #[case] reason: &str,
) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let (mut header, _) = encoded_manager(
        widths,
        &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
        1,
        1,
        0,
        10,
        0,
    );
    header[at] = value;
    refresh_checksum(&mut header);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    assert_eq!(
        FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap_err(),
        FormatError::InvalidFreeSpaceManager(reason.into())
    );
}

#[rstest]
#[case::ghost_count(
    5,
    1,
    4,
    964,
    "the reconstructed sections describe 4 sections and 964 bytes, not the header's 5 sections and 964 bytes"
)]
#[case::space_total(
    4,
    1,
    3,
    965,
    "the reconstructed sections describe 4 sections and 964 bytes, not the header's 4 sections and 965 bytes"
)]
#[case::count_excess(
    3,
    1,
    2,
    964,
    "the reconstructed section count exceeds the header count"
)]
fn rejects_reconstructed_count_and_space_disagreement(
    #[case] total: u64,
    #[case] serial: u64,
    #[case] ghost: u64,
    #[case] space: u64,
    #[case] reason: &str,
) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let (header, list) = encoded_manager(
        widths,
        &first_row_record(1, 113, 0, 0, 0, 16),
        total,
        serial,
        ghost,
        space,
        0,
    );
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let err = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap_err();
    assert_eq!(err, FormatError::InvalidFreeSpaceManager(reason.into()));
}

#[rstest]
#[case::single_overlap(false)]
#[case::first_row_overlap(true)]
fn rejects_overlapping_logical_sections(#[case] first_row: bool) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let records = if first_row {
        let mut records = single_record(FractalHeapSectionClass::Single as u8, 15, 10);
        records.extend(first_row_record(1, 113, 0, 0, 0, 1));
        records
    } else {
        let mut records = single_record(FractalHeapSectionClass::Single as u8, 15, 10);
        records[0] = 2;
        records.extend_from_slice(&[20, 0, 0]);
        records
    };
    let (header, list) = encoded_manager(
        widths,
        &records,
        2,
        2,
        0,
        if first_row { 123 } else { 20 },
        0,
    );
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let err = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap_err();
    assert_eq!(
        err,
        FormatError::InvalidFreeSpaceManager("the logical managed-space sections overlap".into())
    );
}

#[rstest]
#[case::zero_count(0, 0, "a record set has an invalid count or section size")]
#[case::zero_size(1, 0, "a record set has an invalid count or section size")]
#[case::count_excess(0, 2, "a record set has an invalid count or section size")]
#[case::nonzero_padding(6, 1, "the section list has nonzero bytes after its records")]
fn rejects_record_sets_and_padding(#[case] at: usize, #[case] value: u8, #[case] reason: &str) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let mut records = single_record(FractalHeapSectionClass::Single as u8, 15, 10);
    records.push(0);
    records[at] = value;
    let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    assert_eq!(
        reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err(),
        FormatError::InvalidFreeSpaceManager(reason.into())
    );
}

#[rstest]
#[case::single_header(0, 10)]
#[case::single_block_crossing(120, 10)]
fn rejects_single_sections_outside_direct_block_data(#[case] offset: u64, #[case] size: u64) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let (header, list) = encoded_manager(
        widths,
        &single_record(FractalHeapSectionClass::Single as u8, offset, size),
        1,
        1,
        0,
        size,
        0,
    );
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    assert_eq!(
        reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err(),
        FormatError::InvalidFreeSpaceManager(
            "a single section crosses a direct block's usable space".into()
        )
    );
}

#[rstest]
#[case::backlink(5, 33)]
#[case::signature(0, b'X')]
#[case::version(4, 1)]
fn rejects_common_section_list_framing(#[case] at: usize, #[case] value: u8) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let (header, mut list) = encoded_manager(
        widths,
        &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
        1,
        1,
        0,
        10,
        0,
    );
    list[at] = value;
    refresh_checksum(&mut list);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let err = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap_err();
    let expected = match at {
        0 => FormatError::InvalidFreeSpaceManager(
            "the section list signature is b\"XSSE\", not b\"FSSE\"".into(),
        ),
        4 => FormatError::UnsupportedVersion(1),
        _ => FormatError::InvalidFreeSpaceManager(
            "the section list belongs to the header at 33, not to the header at 32".into(),
        ),
    };
    assert_eq!(err, expected);
}

#[cfg(feature = "checksum")]
#[rstest]
#[case::header(true)]
#[case::section_list(false)]
fn rejects_checksum_corruption(#[case] header_corrupt: bool) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let (mut header, mut list) = encoded_manager(
        widths,
        &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
        1,
        1,
        0,
        10,
        0,
    );
    let block = if header_corrupt {
        &mut header
    } else {
        &mut list
    };
    let end = block.len() - 4;
    let expected = u32::from_le_bytes(block[end..].try_into().unwrap());
    block[6] ^= 1;
    let computed = checksum::jenkins_lookup3(&block[..end]);
    let err = if header_corrupt {
        FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap_err()
    } else {
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err()
    };
    assert_eq!(err, FormatError::ChecksumMismatch { expected, computed });
}

fn refresh_checksum(block: &mut [u8]) {
    let at = block.len() - 4;
    let checksum = checksum::jenkins_lookup3(&block[..at]);
    block[at..].copy_from_slice(&checksum.to_le_bytes());
}

#[rstest]
#[case::child_block(8192, 0, 0, 12, 8192, 3, 467)]
#[case::indirect_row(0, 5, 1, 2, 10240, 6, 934)]
fn reconstructs_first_rows_under_indirect_entries(
    #[case] block_offset: u16,
    #[case] row: u16,
    #[case] column: u16,
    #[case] blocks: u16,
    #[case] offset: u64,
    #[case] count: u64,
    #[case] space: u64,
) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let mut heap = heap(widths);
    heap.current_rows_in_root_indirect_block = 6;
    heap.managed_space = 16384;
    let mut records = first_row_record(1, 113, offset, row, column, blocks);
    records[6..8].copy_from_slice(&block_offset.to_le_bytes());
    let (header, list) = encoded_manager(widths, &records, count, 1, count - 1, space, 0);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let parsed = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap();
    assert_eq!(
        (parsed.total_sections(), parsed.total_space()),
        (count, space)
    );
    let FractalHeapFreeSection::FirstRow { row: first, .. } = parsed.sections()[0] else {
        panic!("expected first row")
    };
    assert_eq!(
        (first.offset(), first.size(), first.class()),
        (offset, 113, FractalHeapSectionClass::FirstRow)
    );
}

#[rstest]
#[case::inside_direct_block(1)]
#[case::direct_block_start(128)]
#[case::outside_heap(4096)]
fn rejects_indirect_block_offsets_without_corresponding_geometry(#[case] offset: u16) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let mut records = first_row_record(1, 113, 0, 0, 0, 1);
    records[6..8].copy_from_slice(&offset.to_le_bytes());
    let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 113, 0);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let err = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap_err();
    assert_eq!(
        err,
        if offset == 4096 {
            FormatError::InvalidFreeSpaceManager(
                "the indirect section has no corresponding heap block".into(),
            )
        } else {
            FormatError::InvalidFractalHeapGeometry
        }
    );
}

#[test]
fn parses_64_bit_logical_offsets_from_small_local_regions() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let mut heap = heap(widths);
    heap.max_heap_size = 64;
    heap.current_rows_in_root_indirect_block = 55;
    heap.managed_space = 1 << 63;
    let offset = (1u64 << 40) + 21;
    let mut records = vec![1, 10, 0];
    records.extend_from_slice(&offset.to_le_bytes());
    records.push(FractalHeapSectionClass::Single as u8);
    let (mut header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
    header[44] = 64;
    refresh_checksum(&mut header);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    let manager = FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap();
    let reference = manager
        .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
        .unwrap();
    let parsed = reference
        .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
        .unwrap();
    assert_eq!(
        parsed.sections(),
        &[FractalHeapFreeSection::Single {
            offset,
            size: NonZeroU64::new(10).unwrap()
        }]
    );
}

#[test]
fn distinguishes_heap_and_file_client_header_validation() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let heap = heap(widths);
    let (mut header, _) = encoded_manager(
        widths,
        &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
        1,
        1,
        0,
        10,
        0,
    );
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    assert_eq!(
        FreeSpaceManagerHeader::from_frame(&frame).unwrap_err(),
        FormatError::InvalidFreeSpaceManager(
            "the client ID is 0, not 1, the ID of the file client".into()
        )
    );
    header[5] = 1;
    header[38] = 3;
    refresh_checksum(&mut header);
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
    assert_eq!(
        FreeSpaceManagerHeader::from_frame(&frame)
            .unwrap()
            .total_sections(),
        1
    );
    assert_eq!(
        FractalHeapFreeSpaceManager::from_frame(frame, &heap).unwrap_err(),
        FormatError::InvalidFreeSpaceManager(
            "the client ID is 1, not 0, the ID of the fractal heap client".into()
        )
    );
}

fn heap(widths: FormatWidths) -> FractalHeapHeader {
    let mut heap = FractalHeapHeader::parse_bytes(
        &fractal_heap::Header::new(256).build(Widths {
            offset: usize::from(widths.offsets.get()),
            length: usize::from(widths.lengths.get()),
        }),
        widths.offsets,
        widths.lengths,
    )
    .unwrap();
    heap.managed_space = 4096;
    heap.current_rows_in_root_indirect_block = 4;
    heap
}

fn single_record(class: u8, offset: u64, size: u64) -> Vec<u8> {
    let mut record = vec![1];
    record.extend_from_slice(&u16::try_from(size).unwrap().to_le_bytes());
    record.extend_from_slice(&u16::try_from(offset).unwrap().to_le_bytes());
    record.push(class);
    record
}

fn first_row_record(
    count: u8,
    size: u64,
    offset: u64,
    row: u16,
    column: u16,
    blocks: u16,
) -> Vec<u8> {
    let mut record = single_record(FractalHeapSectionClass::FirstRow as u8, offset, size);
    record[0] = count;
    record.extend_from_slice(&0u16.to_le_bytes());
    record.extend_from_slice(&row.to_le_bytes());
    record.extend_from_slice(&column.to_le_bytes());
    record.extend_from_slice(&blocks.to_le_bytes());
    record
}

fn encoded_manager(
    widths: FormatWidths,
    records: &[u8],
    total: u64,
    serialized: u64,
    ghost: u64,
    space: u64,
    spare: u64,
) -> (Vec<u8>, Vec<u8>) {
    let mut list = FSM_SECTION_LIST_SIGNATURE.to_vec();
    list.push(FSM_VERSION);
    crate::bytes::write_offset(&mut list, 32, widths.offsets);
    list.extend_from_slice(records);
    list.extend_from_slice(&checksum::jenkins_lookup3(&list).to_le_bytes());
    let mut header = FSM_HEADER_SIGNATURE.to_vec();
    header.extend_from_slice(&[FSM_VERSION, CLIENT_ID]);
    for value in [space, total, serialized, ghost] {
        crate::bytes::write_length(&mut header, value, widths.lengths);
    }
    for value in [NUM_CLASSES, 80, 120, 16] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    crate::bytes::write_length(&mut header, 1024, widths.lengths);
    crate::bytes::write_offset(&mut header, 4096, widths.offsets);
    crate::bytes::write_length(&mut header, list.len().to_u64(), widths.lengths);
    crate::bytes::write_length(&mut header, list.len().to_u64() + spare, widths.lengths);
    header.extend_from_slice(&checksum::jenkins_lookup3(&header).to_le_bytes());
    (header, list)
}
// The signatures and version are defined in "Free-space Index", format specification version 4.0.
const FSM_HEADER_SIGNATURE: &[u8; 4] = b"FSHD";
const FSM_SECTION_LIST_SIGNATURE: &[u8; 4] = b"FSSE";
const FSM_VERSION: u8 = 0;
