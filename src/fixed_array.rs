//! The Fixed Array reads over a byte slice or a [`Source`](crate::source::Source).
//!
//! [`read_chunks`] reads the header, then the whole data block if it is unpaged, or its prefix
//! and page-init bitmap and then each initialized page if it is paged.

#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::iter;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FixedArrayDataBlock;
use hdf5_pure_format::__private::FixedArrayElements;
use hdf5_pure_format::__private::FixedArrayHeader;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::source::RangeReader;
use crate::width::FormatWidths;

/// Reads the Fixed Array at `address` and calls `visit` with the slot and the record of each
/// element that stores a chunk address.
///
/// A record of an unfiltered chunk has `chunk_byte_size` as its stored size. An array whose header
/// stores the undefined data block address has no chunks, and this function reads only its
/// header.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the file ends inside the header, the data block, or
/// a page, and the error of a failed read of a [`Source`](crate::source::Source). Returns the
/// errors [`FixedArrayHeaderLocator::parse`],
/// [`FixedArrayDataBlockRef::parse`](hdf5_pure_format::__private::FixedArrayDataBlockRef::parse),
/// and [`FixedArrayPageRef::parse`](hdf5_pure_format::__private::FixedArrayPageRef::parse)
/// return, and the error `visit` returns.
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
    let range = block.parse_range()?;
    let bytes = file.read_range(range)?;
    match block.parse(FileRegion::new(range, &bytes)?)? {
        FixedArrayDataBlock::Elements(elements) => visit_elements(elements),
        FixedArrayDataBlock::PageBitmap(bitmap) => bitmap.pages().try_for_each(|page| {
            let page = page?;
            let range = page.parse_range();
            let bytes = file.read_range(range)?;
            visit_elements(page.parse(FileRegion::new(range, &bytes)?)?)
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
/// Returns [`FormatError::UnexpectedEof`] if the file ends inside the header, the error of a
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

fn read_header(
    file: &(impl RangeReader + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
) -> Result<FixedArrayHeader, FormatError> {
    let locator = FixedArrayHeaderLocator::new(widths, BaseAddress::ZERO, address);
    let range = locator.parse_range()?;
    locator.parse(FileRegion::new(range, &file.read_range(range)?)?)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    use hdf5_pure_format::__private::IndexSlots;
    use hdf5_pure_format::__private::build_fixed_array_at;
    use rstest::rstest;
    use test_util::checksum::restamp;

    use super::*;
    use crate::address::StoredAddressExt;
    use crate::source;
    use crate::source::BaseOffsetSource;
    use crate::source::BytesSource;
    use crate::source::Source;
    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

    const WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    const ARRAY_AT: u64 = 0x80;

    const HEADER_LEN: u64 = 28;

    fn chunks(n: u64, has_filters: bool) -> Vec<ChunkRecord> {
        (0..n)
            .map(|i| ChunkRecord {
                address: StoredAddress::new(0x100 + i * 8),
                stored_size: if has_filters { 8 + (i % 7) } else { 8 },
                filter_mask: 0,
            })
            .collect()
    }

    fn slotted(chunks: Vec<ChunkRecord>) -> Vec<(u64, ChunkRecord)> {
        (0..).zip(chunks).collect()
    }

    fn image(n: u64, has_filters: bool) -> Vec<u8> {
        let array = build_fixed_array_at(
            &IndexSlots::dense(&chunks(n, has_filters)),
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
            reads.push((block, prefix + n * stride + 4));
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

    /// A source whose bytes start at address `start` and that records each metadata read.
    struct PlacedSource {
        start: u64,
        bytes: Vec<u8>,
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl PlacedSource {
        fn new(start: u64, bytes: Vec<u8>) -> Self {
            Self {
                start,
                bytes,
                reads: RefCell::new(Vec::new()),
            }
        }

        fn framed(&self) -> BaseOffsetSource<'_, Self> {
            BaseOffsetSource {
                inner: self,
                base: BaseAddress::new(self.start),
            }
        }

        fn records(&self) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
            let mut records = Vec::new();
            read_chunks(
                &self.framed(),
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
    }

    impl Source for PlacedSource {
        fn len(&self) -> u64 {
            self.start + self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            BytesSource::new(&self.bytes).read_at(offset - self.start, buf)
        }

        fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            self.reads.borrow_mut().push((offset, len));
            self.read_exact_at(offset, len)
        }
    }

    fn buffered_records(image: &[u8], base: u64) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_chunks(
            source::frame(image, BaseAddress::new(base))?,
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
        #[values(0, 5, 3_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let mut file = vec![0; usize::try_from(base).unwrap()];
        file.extend_from_slice(&image(n, has_filters));
        assert_eq!(
            buffered_records(&file, base).unwrap(),
            slotted(chunks(n, has_filters))
        );
    }

    #[rstest]
    fn a_streamed_read_reads_the_header_the_data_block_and_each_page(
        #[values(0, 512, 1 << 40)] base: u64,
        #[values(0, 5, 3_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let source = PlacedSource::new(base, image(n, has_filters));
        assert_eq!(source.records().unwrap(), slotted(chunks(n, has_filters)));
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
        restamp(&mut bytes, block, 4 + 2 + 8 + 1 + 4);
        let source = PlacedSource::new(512, bytes);
        let expected: Vec<(u64, ChunkRecord)> = slotted(chunks(4_000, false))
            .into_iter()
            .filter(|(slot, _)| pages.contains(&(slot / 1_024)))
            .collect();
        assert_eq!(source.records().unwrap(), expected);
        assert_eq!(
            source.reads.into_inner(),
            expected_reads(512, 4_000, false, pages)
        );
    }

    // Block lengths: 58 unpaged, and 19 for the prefix, 8_196 for a full page, 7_620 for the last.
    #[rstest]
    #[case::header(5, ARRAY_AT + HEADER_LEN - 1, ARRAY_AT + HEADER_LEN)]
    #[case::unpaged_data_block(5, ARRAY_AT + HEADER_LEN + 57, ARRAY_AT + HEADER_LEN + 58)]
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
        let expected = FormatError::UnexpectedEof {
            expected: usize::try_from(end).unwrap(),
            available: usize::try_from(len).unwrap(),
        };
        assert_eq!(buffered_records(&bytes, 0).unwrap_err(), expected);
        assert_eq!(PlacedSource::new(0, bytes).records().unwrap_err(), expected);
    }

    #[test]
    fn an_array_without_a_data_block_has_no_chunks_and_only_its_header() {
        let mut bytes = image(5, false);
        let header = usize::try_from(ARRAY_AT).unwrap();
        bytes[header + 16..header + 24].copy_from_slice(&u64::MAX.to_le_bytes());
        restamp(&mut bytes, header, usize::try_from(HEADER_LEN).unwrap());
        let source = PlacedSource::new(0, bytes);
        assert_eq!(source.records().unwrap(), Vec::new());
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
