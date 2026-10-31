//! Extensible Array traversal through checked stored-coordinate range reads.
//!
//! [`read_chunks`] reads the header and the index block, then each super block and each data block
//! that holds a slot below the header's "Max Index Set". Of a paged data block it reads the prefix
//! and the initialized pages, each run of adjacent ones in one read.

#[cfg(feature = "std")]
use alloc::vec;
#[cfg(feature = "std")]
use alloc::vec::Vec;
use core::iter::Peekable;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::AddressRange;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::ExtensibleArray;
use hdf5_pure_format::__private::ExtensibleArrayDataBlock;
use hdf5_pure_format::__private::ExtensibleArrayDataBlockRef;
use hdf5_pure_format::__private::ExtensibleArrayElements;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use hdf5_pure_format::__private::ExtensibleArrayPageRef;
use hdf5_pure_format::__private::ExtensibleArrayPages;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source;
use crate::source::RangeReader;
use crate::width::FormatWidths;

/// Reads the Extensible Array at `address` and calls `visit` with the slot and the record of each
/// element that stores a chunk address.
///
/// The caller supplies a stored-coordinate view of the opened image. Returned metadata retains
/// the source's physical ranges and ownership while the parsers use base-relative positions.
///
/// A record of an unfiltered chunk has `chunk_byte_size` as its stored size. An array whose header
/// stores the undefined index block address has no chunks, and this function reads only its
/// header. Of the other blocks and pages, it reads the allocated blocks and the initialized pages
/// whose first slot is below the header's "Max Index Set".
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if resolving a stored position exceeds [`u64::MAX`].
/// Returns [`FormatError::UnexpectedSourceEof`] if the file ends inside a block or a page,
/// [`FormatError::ValueTooLargeForPlatform`] if a block does not fit a `usize`, and the error of a
/// failed read of a [`Source`](crate::source::Source). Returns the errors
/// [`ExtensibleArrayHeaderLocator::parse`],
/// [`ExtensibleArrayIndexBlockRef::parse`](hdf5_pure_format::__private::ExtensibleArrayIndexBlockRef::parse),
/// [`ExtensibleArraySuperBlockRef::parse`](hdf5_pure_format::__private::ExtensibleArraySuperBlockRef::parse),
/// [`ExtensibleArrayDataBlockRef::parse`], and [`ExtensibleArrayPageRef::parse`] return, and the
/// error `visit` returns.
pub(crate) fn read_chunks(
    file: &(impl RangeReader<AddressRange<StoredAddress>> + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
    chunk_byte_size: u64,
    mut visit: impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let Some(index_block) = read_header(file, widths, address)?.index_block() else {
        return Ok(());
    };
    let mut visit_elements = |elements: ExtensibleArrayElements<'_>| {
        elements.records(chunk_byte_size).try_for_each(|record| {
            let (slot, record) = record?;
            visit(slot, record)
        })
    };
    let range = index_block.parse_range();
    let bytes = file.read_range(source::legacy_stored_range(range)?)?;
    let index_block = index_block.parse(source::legacy_framed_region(range, &bytes)?)?;
    visit_elements(index_block.elements())?;
    for block in index_block.data_blocks() {
        let block = block?;
        if block.holds_set_slots() {
            read_data_block(file, block, &mut visit_elements)?;
        }
    }
    for super_block in index_block.super_blocks() {
        let super_block = super_block?;
        if !super_block.holds_set_slots() {
            continue;
        }
        let range = super_block.parse_range();
        let bytes = file.read_range(source::legacy_stored_range(range)?)?;
        for block in super_block
            .parse(source::legacy_framed_region(range, &bytes)?)?
            .data_blocks()
        {
            let block = block?;
            if block.holds_set_slots() {
                read_data_block(file, block, &mut visit_elements)?;
            }
        }
    }
    Ok(())
}

/// Returns the stored-coordinate ranges of the header and every allocated block of the
/// Extensible Array at `address`.
///
/// The spans cover the index, without the chunks it points at, whatever the header's "Max Index
/// Set". This function reads the header, the index block, and each super block, and no data block.
/// An array whose header stores the undefined index block address has the header's span only.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if resolving a stored position exceeds [`u64::MAX`].
/// Returns [`FormatError::UnexpectedSourceEof`] if the file ends inside the header, the index block,
/// or a super block, [`FormatError::ValueTooLargeForPlatform`] if the index block or a super block
/// does not fit a `usize`, and the error of a failed read of a
/// [`Source`](crate::source::Source). Returns the errors [`ExtensibleArrayHeaderLocator::parse`],
/// [`ExtensibleArrayIndexBlockRef::parse`](hdf5_pure_format::__private::ExtensibleArrayIndexBlockRef::parse),
/// and
/// [`ExtensibleArraySuperBlockRef::parse`](hdf5_pure_format::__private::ExtensibleArraySuperBlockRef::parse)
/// return.
#[cfg(feature = "std")]
pub(crate) fn index_spans(
    file: &(impl RangeReader<AddressRange<StoredAddress>> + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
) -> Result<Vec<AddressRange<StoredAddress>>, FormatError> {
    let header = read_header(file, widths, address)?;
    let mut spans = vec![source::legacy_stored_range(header.range())?];
    let Some(index_block) = header.index_block() else {
        return Ok(spans);
    };
    let range = index_block.parse_range();
    spans.push(source::legacy_stored_range(range)?);
    let bytes = file.read_range(source::legacy_stored_range(range)?)?;
    let index_block = index_block.parse(source::legacy_framed_region(range, &bytes)?)?;
    for block in index_block.data_blocks() {
        spans.push(source::legacy_stored_range(block?.range())?);
    }
    for super_block in index_block.super_blocks() {
        let super_block = super_block?;
        let range = super_block.parse_range();
        spans.push(source::legacy_stored_range(range)?);
        let bytes = file.read_range(source::legacy_stored_range(range)?)?;
        for block in super_block
            .parse(source::legacy_framed_region(range, &bytes)?)?
            .data_blocks()
        {
            spans.push(source::legacy_stored_range(block?.range())?);
        }
    }
    Ok(spans)
}

/// Reads and parses the header of the Extensible Array at `address`.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if resolving a stored position exceeds [`u64::MAX`].
/// Returns [`FormatError::UnexpectedSourceEof`] if the file ends inside the header, the error of a
/// failed read of a [`Source`](crate::source::Source), and the errors
/// [`ExtensibleArrayHeaderLocator::parse`] returns.
pub(crate) fn read_header(
    file: &(impl RangeReader<AddressRange<StoredAddress>> + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
) -> Result<ExtensibleArray, FormatError> {
    let locator = ExtensibleArrayHeaderLocator::new(widths, BaseAddress::ZERO, address);
    let range = locator.parse_range()?;
    locator.parse(source::legacy_framed_region(
        range,
        &file.read_range(source::legacy_stored_range(range)?)?,
    )?)
}

// Reads the prefix of a paged data block with the initialized pages that directly follow it, and
// each later run of adjacent initialized pages, with one read each, while the length of a read
// fits a `usize`. No uninitialized page is read.
fn read_data_block(
    file: &(impl RangeReader<AddressRange<StoredAddress>> + ?Sized),
    block: ExtensibleArrayDataBlockRef<'_>,
    visit_elements: &mut impl FnMut(ExtensibleArrayElements<'_>) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let planned = block.pages()?;
    let (range, len) = extend_run(
        block.parse_range(),
        &mut planned
            .iter()
            .flat_map(ExtensibleArrayPages::pages)
            .peekable(),
    );
    let bytes = file.read_range(source::legacy_stored_range(range)?)?;
    let region = source::legacy_framed_region(range, &bytes)?;
    let pages = match block.parse(region)? {
        ExtensibleArrayDataBlock::Elements(elements) => return visit_elements(elements),
        ExtensibleArrayDataBlock::Pages(pages) => pages,
    };
    let mut pending = pages.pages().peekable();
    visit_pages(region, pending.by_ref().take(len), visit_elements)?;
    loop {
        let run = pending.clone();
        let Some(first) = pending.next().transpose()? else {
            return Ok(());
        };
        let (range, len) = extend_run(first.parse_range(), &mut pending);
        let bytes = file.read_range(source::legacy_stored_range(range)?)?;
        visit_pages(
            source::legacy_framed_region(range, &bytes)?,
            run.take(len + 1),
            visit_elements,
        )?;
    }
}

// Returns `run` extended by each next page that starts where it ends, and the number of pages it
// takes from `pending`.
fn extend_run(
    mut run: FileRange,
    pending: &mut Peekable<impl Iterator<Item = Result<ExtensibleArrayPageRef, FormatError>>>,
) -> (FileRange, usize) {
    let mut len = 0;
    while let Some(extended) = pending
        .peek()
        .and_then(|page| page.as_ref().ok())
        .and_then(|page| joined(run, page.parse_range()))
    {
        pending.next();
        run = extended;
        len += 1;
    }
    (run, len)
}

// Returns `run` with `page` appended, or `None` if `page` does not start where `run` ends or the
// joined length does not fit a `usize`.
fn joined(run: FileRange, page: FileRange) -> Option<FileRange> {
    let len = run.length().checked_add(page.length())?;
    if page.start() != run.end() || len.to_usize().is_err() {
        return None;
    }
    FileRange::new(run.start(), len).ok()
}

/// Calls `visit_elements` with the elements of each page in `pages`, parsed from `region`.
fn visit_pages(
    region: FileRegion<'_>,
    mut pages: impl Iterator<Item = Result<ExtensibleArrayPageRef, FormatError>>,
    visit_elements: &mut impl FnMut(ExtensibleArrayElements<'_>) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    pages.try_for_each(|page| visit_elements(page?.parse(region)?))
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hdf5_pure_core::PhysicalAddress;
    use hdf5_pure_format::__private;
    use hdf5_pure_format::__private::IndexSlots;
    use hdf5_pure_format::__private::SlotOccupancy;
    use rstest::rstest;
    use test_util::checksum;

    use super::*;
    use crate::address::AddressContext;
    use crate::address::StoredAddressExt;
    use crate::array_test;
    use crate::placed_source::PlacedSource;
    use crate::source;
    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

    const WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    const ARRAY_AT: u64 = 0x80;

    const HEADER_LEN: u64 = 12 + 6 * 8 + 8 + 4;

    // The prefix, 4 elements, 6 data block addresses, 25 super block addresses, and the checksum.
    const INDEX_BLOCK_LEN: u64 = 14 + 4 * 8 + (6 + 25) * 8 + 4;

    const FIRST_PAGED_SLOT: u64 = 131_060;

    // The prefix, the block offset, and the checksum of a paged data block.
    const PAGED_PREFIX_LEN: u64 = 14 + 4 + 4;

    const PAGE_LEN: u64 = 1_024 * 8 + 4;

    fn image_of(slots: &IndexSlots<'_>, has_filters: bool) -> Vec<u8> {
        let array = __private::build_extensible_array_at(
            slots,
            8,
            WIDTHS.offsets,
            WIDTHS.lengths,
            has_filters,
            StoredAddress::new(ARRAY_AT),
        )
        .unwrap();
        let mut image = vec![0; usize::try_from(ARRAY_AT).unwrap()];
        image.extend_from_slice(&array);
        image
    }

    fn image(n: u64, has_filters: bool) -> Vec<u8> {
        image_of(
            &IndexSlots::dense(&array_test::chunks(n, has_filters)),
            has_filters,
        )
    }

    fn set_max_index_set(image: &mut [u8], max_index_set: u64) {
        let header = usize::try_from(ARRAY_AT).unwrap();
        image[header + 12 + 4 * 8..header + 12 + 5 * 8]
            .copy_from_slice(&max_index_set.to_le_bytes());
        checksum::restamp(image, header, usize::try_from(HEADER_LEN).unwrap());
    }

    fn buffered_records(image: &[u8], base: u64) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_chunks(
            &source::PhysicalSource::for_test(
                &source::BytesSource::new(image),
                AddressContext::from_effective_base(BaseAddress::new(base)),
            )
            .stored(),
            WIDTHS,
            StoredAddress::new(ARRAY_AT),
            8,
            |slot, record| {
                records.push((slot, record));
                Ok(())
            },
        )?;
        Ok(records)
    }

    fn records(source: &PlacedSource) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_chunks(
            &source.framed(),
            WIDTHS,
            StoredAddress::new(ARRAY_AT),
            8,
            |slot, record| {
                records.push((slot, record));
                Ok(())
            },
        )?;
        Ok(records)
    }

    #[rstest]
    fn a_buffered_read_reports_every_chunk(
        #[values(0, 512)] base: u64,
        #[values(0, 4, 20, 2_000, 140_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let mut file = vec![0; usize::try_from(base).unwrap()];
        file.extend_from_slice(&image(n, has_filters));
        assert_eq!(
            buffered_records(&file, base).unwrap(),
            array_test::slotted(array_test::chunks(n, has_filters))
        );
    }

    #[rstest]
    fn a_streamed_read_reads_each_block_and_each_run_of_pages_once(
        #[values(0, 512, 1 << 40)] base: u64,
        #[values(0, 4, 20, 2_000, 140_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let source = PlacedSource::new(base, image(n, has_filters));
        assert_eq!(
            records(&source).unwrap(),
            array_test::slotted(array_test::chunks(n, has_filters))
        );
        let stats = __private::extensible_array_layout(
            SlotOccupancy::Dense(n),
            n,
            8,
            WIDTHS.offsets,
            WIDTHS.lengths,
            has_filters,
        )
        .stats;
        let mut reads = source.reads.into_inner();
        assert_eq!(
            reads.len().to_u64(),
            2 + stats.nsuper_blks + stats.ndata_blks
        );
        reads.sort_unstable();
        for pair in reads.windows(2) {
            assert!(
                pair[0].0 + pair[0].1.to_u64() <= pair[1].0,
                "{pair:?} overlap"
            );
        }
    }

    #[rstest]
    #[case::separate_pages(
        &[FIRST_PAGED_SLOT + 1_030, FIRST_PAGED_SLOT + 3 * 2_048 + 10],
        &[PAGED_PREFIX_LEN, PAGE_LEN, PAGED_PREFIX_LEN + PAGE_LEN]
    )]
    #[case::adjacent_pages(
        &[FIRST_PAGED_SLOT + 5, FIRST_PAGED_SLOT + 1_030],
        &[PAGED_PREFIX_LEN + 2 * PAGE_LEN]
    )]
    fn a_streamed_read_skips_unallocated_blocks_and_uninitialized_pages(
        #[case] occupied: &[u64],
        #[case] data_block_reads: &[u64],
    ) {
        let chunks = array_test::chunks(occupied.len().to_u64(), false);
        let slots = IndexSlots::new(&chunks, occupied, 140_000).unwrap();
        let source = PlacedSource::new(512, image_of(&slots, false));
        assert_eq!(
            records(&source).unwrap(),
            occupied.iter().copied().zip(chunks).collect::<Vec<_>>()
        );
        // The super block's prefix, block offset, page-init bitmap, and 64 data block addresses.
        let super_block_len = 14 + 4 + 64 + 64 * 8 + 4;
        let mut expected = vec![HEADER_LEN, INDEX_BLOCK_LEN, super_block_len];
        expected.extend_from_slice(data_block_reads);
        assert_eq!(source.read_lengths(), expected);
    }

    #[rstest]
    #[case::index_block(2, &[HEADER_LEN, INDEX_BLOCK_LEN])]
    #[case::first_data_block(5, &[HEADER_LEN, INDEX_BLOCK_LEN, 14 + 4 + 16 * 8 + 4])]
    // The six data blocks the index block holds the addresses of, then super block 4 and the
    // first of its data blocks.
    #[case::first_super_block(
        300,
        &[HEADER_LEN, INDEX_BLOCK_LEN, 150, 278, 278, 278, 534, 534, 14 + 4 + 4 * 8 + 4, 534]
    )]
    fn a_streamed_read_stops_at_the_max_index_set(
        #[case] max_index_set: u64,
        #[case] reads: &[u64],
    ) {
        let mut bytes = image(2_000, false);
        set_max_index_set(&mut bytes, max_index_set);
        let source = PlacedSource::new(0, bytes);
        assert_eq!(
            records(&source).unwrap(),
            array_test::slotted(array_test::chunks(max_index_set, false))
        );
        assert_eq!(source.read_lengths(), reads);
    }

    #[rstest]
    #[case::adjacent(100, 100, Some(120))]
    #[case::past_a_gap(100, 101, None)]
    fn a_page_joins_the_run_it_follows(
        #[case] run_len: u64,
        #[case] page_start: u64,
        #[case] joined_len: Option<u64>,
    ) {
        let run = FileRange::new(PhysicalAddress::new(0), run_len).unwrap();
        let page = FileRange::new(PhysicalAddress::new(page_start), 20).unwrap();
        assert_eq!(
            joined(run, page),
            joined_len.map(|len| FileRange::new(PhysicalAddress::new(0), len).unwrap())
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn a_page_that_takes_a_run_past_usize_max_starts_a_run_of_its_own() {
        let run = FileRange::new(PhysicalAddress::new(0), u64::from(u32::MAX) - 10).unwrap();
        let page = FileRange::new(run.end(), 20).unwrap();
        assert_eq!(joined(run, page), None);
    }

    #[derive(Clone, Copy, Debug)]
    enum Truncated {
        Header,
        IndexBlock,
        SuperBlock,
        DataBlock,
        Page,
    }

    // Data block 0 of super block 13 moves to the end of the file, past the super block, so that a
    // cut through its last page leaves every other block whole.
    fn truncated(truncated: Truncated) -> (Vec<u8>, u64) {
        let (n, end) = match truncated {
            Truncated::Header => (20, ARRAY_AT + HEADER_LEN),
            Truncated::IndexBlock => (20, ARRAY_AT + HEADER_LEN + INDEX_BLOCK_LEN),
            // One data block of 16 elements.
            Truncated::DataBlock => (20, ARRAY_AT + HEADER_LEN + INDEX_BLOCK_LEN + 150),
            Truncated::SuperBlock | Truncated::Page => (FIRST_PAGED_SLOT + 2_048, 0),
        };
        let mut bytes = image(n, false);
        let end = match truncated {
            Truncated::Header | Truncated::IndexBlock | Truncated::DataBlock => end,
            Truncated::SuperBlock => bytes.len().to_u64(),
            Truncated::Page => {
                let block_len = PAGED_PREFIX_LEN + 2 * PAGE_LEN;
                // The writer places the super block after its one data block.
                let super_block_len = 14 + 4 + 64 + 64 * 8 + 4;
                let super_block = bytes.len() - usize::try_from(super_block_len).unwrap();
                let block = super_block - usize::try_from(block_len).unwrap();
                let moved = bytes.len().to_u64();
                let copy = bytes[block..super_block].to_vec();
                bytes.extend_from_slice(&copy);
                bytes[super_block + 14 + 4 + 64..super_block + 14 + 4 + 64 + 8]
                    .copy_from_slice(&moved.to_le_bytes());
                checksum::restamp(
                    &mut bytes,
                    super_block,
                    usize::try_from(super_block_len).unwrap(),
                );
                bytes.len().to_u64()
            }
        };
        bytes.truncate(usize::try_from(end - 1).unwrap());
        (bytes, end)
    }

    #[rstest]
    fn a_read_past_the_end_of_the_file_is_rejected(
        #[values(
            Truncated::Header,
            Truncated::IndexBlock,
            Truncated::SuperBlock,
            Truncated::DataBlock,
            Truncated::Page
        )]
        cut: Truncated,
    ) {
        let (bytes, end) = truncated(cut);
        let expected = FormatError::UnexpectedSourceEof {
            expected: end,
            available: bytes.len().to_u64(),
        };
        assert_eq!(buffered_records(&bytes, 0).unwrap_err(), expected);
        assert_eq!(records(&PlacedSource::new(0, bytes)).unwrap_err(), expected);
    }

    #[test]
    fn an_array_without_an_index_block_has_no_chunks_and_only_its_header() {
        let mut bytes = image(5, false);
        let header = usize::try_from(ARRAY_AT + 12 + 6 * 8).unwrap();
        bytes[header..header + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        checksum::restamp(
            &mut bytes,
            usize::try_from(ARRAY_AT).unwrap(),
            usize::try_from(HEADER_LEN).unwrap(),
        );
        let source = PlacedSource::new(0, bytes);
        assert_eq!(records(&source).unwrap(), Vec::new());
        assert_eq!(source.read_lengths(), vec![HEADER_LEN]);
        #[cfg(feature = "std")]
        assert_eq!(
            index_spans(&source.framed(), WIDTHS, StoredAddress::new(ARRAY_AT)).unwrap(),
            stored_ranges(vec![(ARRAY_AT, HEADER_LEN)])
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    fn the_index_spans_cover_every_block_and_read_no_data_block(
        #[values(0, 512)] base: u64,
        #[values(4, 20, 2_000, 140_000)] n: u64,
        #[values(false, true)] has_filters: bool,
        #[values(None, Some(2), Some(5), Some(1_000))] max_index_set: Option<u64>,
    ) {
        let mut bytes = image(n, has_filters);
        if let Some(max_index_set) = max_index_set {
            set_max_index_set(&mut bytes, max_index_set);
        }
        let end = bytes.len().to_u64();
        let source = PlacedSource::new(base, bytes);
        let mut spans =
            index_spans(&source.framed(), WIDTHS, StoredAddress::new(ARRAY_AT)).unwrap();
        let stats = __private::extensible_array_layout(
            SlotOccupancy::Dense(n),
            n,
            8,
            WIDTHS.offsets,
            WIDTHS.lengths,
            has_filters,
        )
        .stats;
        assert_eq!(
            spans.len().to_u64(),
            2 + stats.nsuper_blks + stats.ndata_blks
        );
        spans.sort_unstable();
        assert_eq!(spans[0].start(), StoredAddress::new(ARRAY_AT));
        for pair in spans.windows(2) {
            assert_eq!(pair[0].end(), pair[1].start());
        }
        assert_eq!(spans.last().unwrap().end(), StoredAddress::new(end));
        assert_eq!(
            source.reads.into_inner().len().to_u64(),
            2 + stats.nsuper_blks
        );
    }

    #[cfg(all(feature = "std", feature = "checksum"))]
    #[rstest]
    // Super block 4, the last block of 300 elements, holds 4 data block addresses.
    #[case::header(ARRAY_AT, HEADER_LEN)]
    #[case::index_block(ARRAY_AT + HEADER_LEN, INDEX_BLOCK_LEN)]
    #[case::super_block(ARRAY_AT + 2_956, 14 + 4 + 4 * 8 + 4)]
    fn the_index_spans_reject_a_corrupted_checksum(#[case] start: u64, #[case] len: u64) {
        let mut bytes = image(300, false);
        let start = usize::try_from(start).unwrap();
        let end = start + usize::try_from(len).unwrap();
        bytes[end - 1] ^= 0x01;
        let stored = u32::from_le_bytes(bytes[end - 4..end].try_into().unwrap());
        let computed = hdf5_pure_format::__private::jenkins_lookup3(&bytes[start..end - 4]);
        assert_eq!(
            index_spans(
                &source::zero_base_source(&source::BytesSource::new(bytes.as_slice())),
                WIDTHS,
                StoredAddress::new(ARRAY_AT)
            )
            .unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            }
        );
    }
    fn stored_ranges(
        bounds: impl IntoIterator<Item = (u64, u64)>,
    ) -> Vec<AddressRange<StoredAddress>> {
        bounds
            .into_iter()
            .map(|(start, length)| AddressRange::new(StoredAddress::new(start), length).unwrap())
            .collect()
    }
}
