//! The Fixed Array reads over a byte slice or a [`Source`](crate::source::Source).
//!
//! [`read_chunks`] reads unpaged entry windows or initialized pages. With the `checksum` feature,
//! the reader verifies the entire unpaged block in bounded reads before visiting any records.

#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::iter;

use core::num::NonZeroUsize;

#[cfg(feature = "checksum")]
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FixedArrayDataBlock;
use hdf5_pure_format::__private::FixedArrayElements;
use hdf5_pure_format::__private::FixedArrayHeader;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use hdf5_pure_format::__private::FixedArrayUnpaged;
#[cfg(feature = "checksum")]
use hdf5_pure_format::__private::JenkinsLookup3;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::source;
use crate::source::RangeReader;
use crate::width::FormatWidths;

/// Reads the Fixed Array at `address` and calls `visit` with the slot and the record of each
/// element that stores a chunk address.
///
/// A record of an unfiltered chunk has `chunk_byte_size` as its stored size. An array whose header
/// stores the undefined data block address has no chunks, and this function reads only its
/// header. With the `checksum` feature, the reader verifies the entire unpaged block before
/// calling `visit`, then reads its entries in bounded windows.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if the file ends inside the header, the data block, or
/// a page, and the error of a failed read of a [`Source`](crate::source::Source). Returns the
/// errors [`FixedArrayHeaderLocator::parse`],
/// [`FixedArrayDataBlockRef::parse`](hdf5_pure_format::__private::FixedArrayDataBlockRef::parse),
/// [`FixedArrayDataBlockRef::parse_unpaged_prefix`][unpaged],
/// and [`FixedArrayPageRef::parse`](hdf5_pure_format::__private::FixedArrayPageRef::parse)
/// return, and the error `visit` returns.
///
/// [unpaged]: hdf5_pure_format::__private::FixedArrayDataBlockRef::parse_unpaged_prefix
pub(crate) fn read_chunks(
    file: &(impl RangeReader + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
    chunk_byte_size: u64,
    mut visit: impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let Some(block) = read_header(file, widths, address)?.data_block() else {
        return Ok(());
    };
    let mut visit_elements = |elements: FixedArrayElements<'_>| {
        elements.records(chunk_byte_size).try_for_each(|record| {
            let (slot, record) = record?;
            visit(slot, record)
        })
    };
    if let Some(range) = block.unpaged_prefix_range()? {
        let bytes = file.read_range(range)?;
        let unpaged = block.parse_unpaged_prefix(source::legacy_framed_region(range, &bytes)?)?;
        verify_unpaged(file, unpaged)?;
        let mut first_slot = 0;
        while let Some(window) = unpaged.window(first_slot, ENTRY_WINDOW_ENTRIES)? {
            let range = window.parse_range();
            let bytes = file.read_range(range)?;
            visit_elements(window.parse(source::legacy_framed_region(range, &bytes)?)?)?;
            first_slot = window.end_slot();
        }
        return Ok(());
    }
    let range = block.parse_range()?;
    let bytes = file.read_range(range)?;
    match block.parse(source::legacy_framed_region(range, &bytes)?)? {
        FixedArrayDataBlock::Elements(elements) => visit_elements(elements),
        FixedArrayDataBlock::PageBitmap(bitmap) => bitmap.pages().try_for_each(|page| {
            let page = page?;
            let range = page.parse_range();
            let bytes = file.read_range(range)?;
            visit_elements(page.parse(source::legacy_framed_region(range, &bytes)?)?)
        }),
    }
}

/// Returns the `(address, length)` spans of the header and the data block of the Fixed Array at
/// `address`.
///
/// The spans cover the index, without the chunks it points at, and this function reads only the
/// header. An array whose header stores the undefined data block address has the header's span
/// only.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if the file ends inside the header, the error of a
/// failed read of a [`Source`](crate::source::Source), and the errors
/// [`FixedArrayHeaderLocator::parse`] returns.
#[cfg(feature = "std")]
pub(crate) fn index_spans(
    file: &(impl RangeReader + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let header = read_header(file, widths, address)?;
    Ok(iter::once(header.range())
        .chain(header.data_block().map(|block| block.range()))
        .map(|range| (range.start().get(), range.length()))
        .collect())
}

/// Reads the trailing checksum and, with the `checksum` feature, verifies the prefix and entries.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if the file ends inside the block, and, with the
/// `checksum` feature, [`FormatError::ChecksumMismatch`] if the checksum differs. Propagates
/// source read errors.
fn verify_unpaged(
    file: &(impl RangeReader + ?Sized),
    unpaged: FixedArrayUnpaged,
) -> Result<(), FormatError> {
    let range = unpaged.checksum_range()?;
    let bytes = file.read_range(range)?;
    #[cfg(feature = "checksum")]
    {
        let expected = u32::from_le_bytes(bytes.as_ref().try_into().map_err(|_error| {
            FormatError::Internal("a Fixed Array checksum does not have four bytes".into())
        })?);
        let range = unpaged.checksummed_range()?;
        let mut checksum = JenkinsLookup3::new(range.length());
        let mut start = range.start();
        while start < range.end() {
            let length = (range.end().get() - start.get()).min(CHECKSUM_WINDOW_BYTES);
            let window = FileRange::new(start, length)?;
            checksum.update(&file.read_range(window)?)?;
            start = window.end();
        }
        let computed = checksum.finish()?;
        if computed != expected {
            return Err(FormatError::ChecksumMismatch { expected, computed });
        }
    }
    #[cfg(not(feature = "checksum"))]
    drop(bytes);
    Ok(())
}

// The entry count is a nonzero constant.
const ENTRY_WINDOW_ENTRIES: NonZeroUsize = NonZeroUsize::new(256).unwrap();
#[cfg(feature = "checksum")]
const CHECKSUM_WINDOW_BYTES: u64 = 4_096;

fn read_header(
    file: &(impl RangeReader + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
) -> Result<FixedArrayHeader, FormatError> {
    let locator = FixedArrayHeaderLocator::new(widths, BaseAddress::ZERO, address);
    let range = locator.parse_range()?;
    locator.parse(source::legacy_framed_region(
        range,
        &file.read_range(range)?,
    )?)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hdf5_pure_core::__private::FileRegion;
    use hdf5_pure_format::__private::IndexSlots;

    use rstest::rstest;
    use test_util::checksum;

    use super::*;
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

    const HEADER_LEN: u64 = 28;

    fn image(n: u64, has_filters: bool) -> Vec<u8> {
        let array = hdf5_pure_format::__private::build_fixed_array_at(
            &IndexSlots::dense(&array_test::chunks(n, has_filters)),
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

    fn stride(has_filters: bool) -> u64 {
        if has_filters { 8 + 2 + 4 } else { 8 }
    }

    /// Returns the metadata reads a streamed read of `n` elements makes, where `pages` are the
    /// initialized pages of a paged data block.
    fn expected_reads(base: u64, n: u64, has_filters: bool, pages: &[u64]) -> Vec<(u64, usize)> {
        let header = base + ARRAY_AT;
        let block = header + HEADER_LEN;
        let stride = stride(has_filters);
        let prefix = 4 + 2 + 8;
        let mut reads = vec![(header, HEADER_LEN)];
        if n <= 1_024 {
            reads.push((block, prefix));
            reads.push((block + prefix + n * stride, 4));
            #[cfg(feature = "checksum")]
            for offset in (0..prefix + n * stride).step_by(4_096) {
                reads.push((block + offset, (prefix + n * stride - offset).min(4_096)));
            }
            for first in (0..n).step_by(ENTRY_WINDOW_ENTRIES.get()) {
                reads.push((
                    block + prefix + first * stride,
                    (n - first).min(ENTRY_WINDOW_ENTRIES.get() as u64) * stride,
                ));
            }
        } else {
            let bitmap = n.div_ceil(1_024).div_ceil(8);
            let pages_at = block + prefix + bitmap + 4;
            reads.push((block, prefix + bitmap + 4));
            reads.extend(pages.iter().map(|&page| {
                let entries = (n - page * 1_024).min(1_024);
                (pages_at + page * (1_024 * stride + 4), entries * stride + 4)
            }));
        }
        reads
            .into_iter()
            .map(|(at, len)| (at, usize::try_from(len).unwrap()))
            .collect()
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

    fn buffered_records(image: &[u8], base: u64) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_chunks(
            &source::framed_metadata(image, BaseAddress::new(base))?,
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

    fn unpaged_image(n: u64, has_filters: bool) -> Vec<u8> {
        let mut bytes = image(5, has_filters);
        let header = usize::try_from(ARRAY_AT).unwrap();
        bytes[header + 7] = 16;
        bytes[header + 8..header + 16].copy_from_slice(&n.to_le_bytes());
        checksum::restamp(&mut bytes, header, usize::try_from(HEADER_LEN).unwrap());
        let prefix_end = usize::try_from(ARRAY_AT + HEADER_LEN + 14).unwrap();
        bytes.truncate(prefix_end);
        for (slot, record) in array_test::chunks(n, has_filters).iter().enumerate() {
            let address = if slot % 17 == 0 {
                u64::MAX
            } else {
                record.address.get()
            };
            bytes.extend_from_slice(&address.to_le_bytes());
            if has_filters {
                bytes.extend_from_slice(&u16::try_from(record.stored_size).unwrap().to_le_bytes());
                bytes.extend_from_slice(&u32::try_from(slot % 4).unwrap().to_le_bytes());
            }
        }
        bytes.extend_from_slice(&[0; 4]);
        let block = usize::try_from(ARRAY_AT + HEADER_LEN).unwrap();
        let len = bytes.len() - block;
        checksum::restamp(&mut bytes, block, len);
        bytes
    }

    fn unpaged_expected(n: u64, has_filters: bool) -> Vec<(u64, ChunkRecord)> {
        array_test::slotted(array_test::chunks(n, has_filters))
            .into_iter()
            .filter(|(slot, _)| slot % 17 != 0)
            .map(|(slot, mut record)| {
                if has_filters {
                    record.filter_mask = u32::try_from(slot % 4).unwrap();
                }
                (slot, record)
            })
            .collect()
    }

    #[rstest]
    #[case::empty(0)]
    #[case::one(1)]
    #[case::full_window(256)]
    #[case::partial_window(4_099)]
    fn unpaged_reads_use_bounded_windows(
        #[case] n: u64,
        #[values(false, true)] has_filters: bool,
        #[values(0, 1 << 40)] base: u64,
    ) {
        let bytes = unpaged_image(n, has_filters);
        let block = read_header(bytes.as_slice(), WIDTHS, StoredAddress::new(ARRAY_AT))
            .unwrap()
            .data_block()
            .unwrap();
        let range = block.parse_range().unwrap();
        let region = FileRegion::new(
            range,
            &bytes[usize::try_from(range.start().get()).unwrap()..],
        )
        .unwrap();
        let FixedArrayDataBlock::Elements(elements) = block.parse(region).unwrap() else {
            panic!("expected unpaged elements");
        };
        let expected = unpaged_expected(n, has_filters);
        assert_eq!(
            elements.records(8).collect::<Result<Vec<_>, _>>().unwrap(),
            expected
        );
        assert_eq!(buffered_records(&bytes, 0).unwrap(), expected);
        let source = PlacedSource::new(base, bytes);
        assert_eq!(records(&source).unwrap(), expected);
        let reads = source.reads.take();
        assert!(
            reads.iter().all(|(_, length)| *length <= 4_096),
            "{reads:?}"
        );
        if n > ENTRY_WINDOW_ENTRIES.get() as u64 {
            let last = reads.last().unwrap();
            assert_eq!(
                last.1,
                usize::try_from((n % ENTRY_WINDOW_ENTRIES.get() as u64) * stride(has_filters))
                    .unwrap()
            );
            assert_eq!(
                last.0,
                base + ARRAY_AT
                    + HEADER_LEN
                    + 14
                    + (n - n % ENTRY_WINDOW_ENTRIES.get() as u64) * stride(has_filters)
            );
        }
    }

    #[rstest]
    #[case::early(14 + 8)]
    #[case::late(14 + 4_098 * 8)]
    fn unpaged_checksum_covers_every_window_before_visiting(#[case] field: usize) {
        let mut bytes = unpaged_image(4_099, false);
        let block = usize::try_from(ARRAY_AT + HEADER_LEN).unwrap();
        bytes[block + field] ^= 1;
        let expected = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
        let computed = hdf5_pure_format::__private::jenkins_lookup3(&bytes[block..bytes.len() - 4]);
        let source = PlacedSource::new(0, bytes);
        let mut visits = 0;
        let result = read_chunks(
            &source.framed(),
            WIDTHS,
            StoredAddress::new(ARRAY_AT),
            8,
            |_, _| {
                visits += 1;
                Ok(())
            },
        );
        if cfg!(feature = "checksum") {
            assert_eq!(
                result.unwrap_err(),
                FormatError::ChecksumMismatch { expected, computed }
            );
            assert_eq!(visits, 0);
        } else {
            result.unwrap();
            assert_eq!(visits, unpaged_expected(4_099, false).len());
        }
    }

    #[rstest]
    #[case::signature(0, b'X', FormatError::InvalidFixedArraySignature)]
    #[case::version(4, 1, FormatError::InvalidFixedArrayVersion(1))]
    #[case::client(5, 1, FormatError::InvalidFixedArrayBlockClient { expected: 0, actual: 1 })]
    #[case::owner(6, 0x81, FormatError::InvalidFixedArrayBlockOwner { expected: ARRAY_AT, actual: 0x81 })]
    fn unpaged_reads_reject_invalid_prefixes(
        #[case] field: usize,
        #[case] value: u8,
        #[case] expected: FormatError,
    ) {
        let mut bytes = unpaged_image(4_099, false);
        bytes[usize::try_from(ARRAY_AT + HEADER_LEN).unwrap() + field] = value;
        assert_eq!(buffered_records(&bytes, 0).unwrap_err(), expected);
        assert_eq!(records(&PlacedSource::new(0, bytes)).unwrap_err(), expected);
    }

    #[rstest]
    fn a_buffered_read_reports_every_chunk(
        #[values(0, 512)] base: u64,
        #[values(0, 5, 3_000)] n: u64,
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
    fn a_streamed_read_requests_each_metadata_range(
        #[values(0, 512, 1 << 40)] base: u64,
        #[values(0, 5, 3_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let source = PlacedSource::new(base, image(n, has_filters));
        assert_eq!(
            records(&source).unwrap(),
            array_test::slotted(array_test::chunks(n, has_filters))
        );
        assert_eq!(
            source.reads.into_inner(),
            expected_reads(base, n, has_filters, &[0, 1, 2])
        );
    }

    #[rstest]
    #[case::second(0b1011_0000, &[0, 2, 3])]
    #[case::last(0b1110_0000, &[0, 1, 2])]
    fn a_streamed_read_skips_an_uninitialized_page(#[case] bitmap: u8, #[case] pages: &[u64]) {
        let mut bytes = image(4_000, false);
        let block = usize::try_from(ARRAY_AT + HEADER_LEN).unwrap();
        bytes[block + 4 + 2 + 8] = bitmap;
        checksum::restamp(&mut bytes, block, 4 + 2 + 8 + 1 + 4);
        let source = PlacedSource::new(512, bytes);
        let expected: Vec<(u64, ChunkRecord)> =
            array_test::slotted(array_test::chunks(4_000, false))
                .into_iter()
                .filter(|(slot, _)| pages.contains(&(slot / 1_024)))
                .collect();
        assert_eq!(records(&source).unwrap(), expected);
        assert_eq!(
            source.reads.into_inner(),
            expected_reads(512, 4_000, false, pages)
        );
    }

    // Block lengths: 58 unpaged, and 19 for the prefix, 8_196 for a full page, 7_620 for the last.
    #[rstest]
    #[case::header(5, ARRAY_AT + HEADER_LEN - 1, ARRAY_AT + HEADER_LEN)]
    #[case::unpaged_prefix(5, ARRAY_AT + HEADER_LEN + 13, ARRAY_AT + HEADER_LEN + 14)]
    #[case::unpaged_entries(5, ARRAY_AT + HEADER_LEN + 20, ARRAY_AT + HEADER_LEN + 58)]
    #[case::unpaged_checksum(5, ARRAY_AT + HEADER_LEN + 57, ARRAY_AT + HEADER_LEN + 58)]
    #[case::prefix(3_000, ARRAY_AT + HEADER_LEN + 18, ARRAY_AT + HEADER_LEN + 19)]
    #[case::last_page(
        3_000,
        ARRAY_AT + HEADER_LEN + 19 + 2 * 8_196 + 7_619,
        ARRAY_AT + HEADER_LEN + 19 + 2 * 8_196 + 7_620
    )]
    fn a_read_past_the_end_of_the_file_is_rejected(
        #[case] n: u64,
        #[case] len: u64,
        #[case] end: u64,
    ) {
        let mut bytes = image(n, false);
        bytes.truncate(usize::try_from(len).unwrap());
        let expected = FormatError::UnexpectedSourceEof {
            expected: end,
            available: len,
        };
        assert_eq!(buffered_records(&bytes, 0).unwrap_err(), expected);
        assert_eq!(records(&PlacedSource::new(0, bytes)).unwrap_err(), expected);
    }

    #[test]
    fn an_array_without_a_data_block_has_no_chunks_and_only_its_header() {
        let mut bytes = image(5, false);
        let header = usize::try_from(ARRAY_AT).unwrap();
        bytes[header + 16..header + 24].copy_from_slice(&u64::MAX.to_le_bytes());
        checksum::restamp(&mut bytes, header, usize::try_from(HEADER_LEN).unwrap());
        let source = PlacedSource::new(0, bytes);
        assert_eq!(records(&source).unwrap(), Vec::new());
        assert_eq!(
            source.reads.take(),
            vec![(ARRAY_AT, usize::try_from(HEADER_LEN).unwrap())]
        );
        #[cfg(feature = "std")]
        assert_eq!(
            index_spans(&source.framed(), WIDTHS, StoredAddress::new(ARRAY_AT)).unwrap(),
            vec![(ARRAY_AT, HEADER_LEN)]
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    fn the_index_spans_tile_the_array(
        #[values(0, 512)] base: u64,
        #[values(5, 3_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let bytes = image(n, has_filters);
        let end = base + bytes.len() as u64;
        let source = PlacedSource::new(base, bytes);
        let spans = index_spans(&source.framed(), WIDTHS, StoredAddress::new(ARRAY_AT)).unwrap();
        assert_eq!(
            spans,
            vec![
                (ARRAY_AT, HEADER_LEN),
                (ARRAY_AT + HEADER_LEN, end - base - ARRAY_AT - HEADER_LEN)
            ]
        );
        assert_eq!(
            source.reads.into_inner(),
            vec![(base + ARRAY_AT, usize::try_from(HEADER_LEN).unwrap())]
        );
    }
}
