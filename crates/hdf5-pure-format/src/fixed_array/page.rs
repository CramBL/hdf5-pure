use core::num::NonZeroU8;
use core::num::NonZeroUsize;
use core::ops::Range;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;

use super::data_block;
use super::data_block::FixedArrayDataBlockRef;
use super::data_block::FixedArrayElements;
use super::data_block::FixedArrayEntryWindow;
use super::framing;
use super::framing::CHECKSUM_LEN;
use super::geometry;
use super::geometry::Paging;
use super::mutation;
use crate::bytes;
use crate::checksum;
use crate::checksum::JenkinsLookup3;
use crate::chunk_record;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;

/// The page-init bitmap of a paged Fixed Array data block, which marks the pages that hold
/// elements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayPageBitmap<'a> {
    block: PagedBlock,
    bitmap: &'a [u8],
}

impl<'a> FixedArrayPageBitmap<'a> {
    /// Returns the reference to each page the bitmap marks initialized, in page order.
    ///
    /// Page 0 is the most significant bit of the first byte.
    pub fn pages(&self) -> impl Iterator<Item = Result<FixedArrayPageRef, FormatError>> + 'a {
        let Self { block, bitmap } = *self;
        (0..block.paging.pages())
            .filter(move |&index| chunk_record::page_initialized(bitmap, index))
            .map(move |index| block.page(index))
    }
}

impl FixedArrayPageBitmap<'_> {
    /// Returns the page containing `slot` and whether the bitmap marks it initialized.
    ///
    /// Returns `None` if `slot` is at or beyond the array's entry count. The caller interprets
    /// existing page elements only when the initialization bit is set.
    ///
    /// # Errors
    ///
    /// Returns the geometry and range errors of [`FixedArrayDataBlockRef::page_for_slot`].
    pub fn page_for_slot(
        &self,
        slot: u64,
    ) -> Result<Option<(FixedArrayPageRef, bool)>, FormatError> {
        let Self { block, bitmap } = *self;
        block
            .index(slot)
            .map(|index| {
                Ok((
                    block.page(index)?,
                    chunk_record::page_initialized(bitmap, index),
                ))
            })
            .transpose()
    }
}

/// The position of one page of a paged Fixed Array data block, from
/// [`FixedArrayPageBitmap::pages`] or [`FixedArrayDataBlockRef::page_for_slot`].
///
/// A reference describes the reserved range, including for an uninitialized page. The caller
/// checks the page-init bitmap before reading existing elements.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayPageRef {
    range: FileRange,
    first_slot: u64,
    encoding: ElementEncoding,
}

impl FixedArrayPageRef {
    /// Parses the page in `region` and returns its elements.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), and, under the `checksum` feature,
    /// [`FormatError::ChecksumMismatch`] if the checksum does not match the page's bytes.
    pub fn parse<'a>(&self, region: FileRegion<'a>) -> Result<FixedArrayElements<'a>, FormatError> {
        let Self {
            range,
            first_slot,
            encoding,
        } = *self;
        let bytes = region.subregion(range)?.bytes();
        checksum::verify_trailing(bytes)?;
        Ok(FixedArrayElements::new(
            framing::without_checksum(bytes)?,
            first_slot,
            encoding,
        ))
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the page's elements and its
    /// checksum.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
    /// Returns the page's entry bytes, excluding its trailing checksum.
    ///
    /// The page of a version 0 array is defined in "The Fixed Array Index" of the
    /// [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if the page is shorter than its checksum,
    /// or [`FormatError::OffsetOverflow`] if a physical range overflows.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr
    pub fn checksummed_range(&self) -> Result<FileRange, FormatError> {
        framing::checksum_ranges(self.range).map(|(body, _)| body)
    }

    /// Returns the checksum field following [`Self::checksummed_range`].
    ///
    /// # Errors
    ///
    /// Returns the geometry and range errors of [`Self::checksummed_range`].
    pub fn checksum_range(&self) -> Result<FileRange, FormatError> {
        framing::checksum_ranges(self.range).map(|(_, checksum)| checksum)
    }

    /// Emits a complete initialized page with `records` in `window` and empty entries elsewhere.
    ///
    /// Checks the window and every record before the first sink call. Writes the checksum after
    /// the entries, independently of the `checksum` feature. Uses a 4 KiB scratch buffer and
    /// returns sink errors unchanged. The caller publishes the page before its initialization bit.
    ///
    /// # Errors
    ///
    /// Returns an error converted from [`FormatError::FixedArrayEntryCount`] if counts differ, or
    /// [`FormatError::InvalidFixedArrayGeometry`] if the window is empty or belongs to another
    /// page. Returns converted address and stored-size errors from
    /// [`FixedArrayEntryWindow::encode_records`].
    /// A sink error may follow partial output.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroUsize;
    /// use hdf5_pure_core::{BaseAddress, PhysicalAddress, StoredAddress};
    /// use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let widths = format::FormatWidths::from_sizes(2, 4)?;
    /// let placed = format::FixedArrayPlan::new(10, 16, widths, false)?.with_page_bits(2)?
    ///     .place(StoredAddress::new(64), StoredAddress::new(256))?;
    /// let bytes = placed.serialize_header()?;
    /// let locator = format::FixedArrayHeaderLocator::new(
    ///     widths, BaseAddress::ZERO, StoredAddress::new(64),
    /// );
    /// let header = locator.parse(FileRegion::new(
    ///     FileRange::new(PhysicalAddress::new(64), bytes.len() as u64)?, &bytes,
    /// )?)?;
    /// let block = header.data_block().unwrap();
    /// let mut data = Vec::new();
    /// placed.emit_empty_data_block::<FormatError>(|bytes| {
    ///     data.extend_from_slice(bytes);
    ///     Ok(())
    /// })?;
    /// assert_eq!(block.prefix_range()?.length(), 8);
    /// block.parse_prefix(FileRegion::new(block.prefix_range()?, &data[..8])?)?;
    /// assert_eq!(block.checksummed_range()?.length(), 9);
    /// assert_eq!(block.checksum_range()?.length(), 4);
    /// let page = block.page_for_slot(9)?.unwrap();
    /// let window = page.window(9, NonZeroUsize::MIN)?.unwrap();
    /// let record = format::ChunkRecord {
    ///     address: StoredAddress::new(0x1234), stored_size: 16, filter_mask: 0,
    /// };
    /// let mut output = Vec::new();
    /// page.emit_records::<FormatError>(&window, &[Some(record)], |bytes| {
    ///     output.extend_from_slice(bytes);
    ///     Ok(())
    /// })?;
    /// assert_eq!(&output[..4], &[0xff, 0xff, 0x34, 0x12]);
    /// assert_eq!(page.checksummed_range()?.length(), 4);
    /// assert_eq!(page.checksum_range()?.length(), 4);
    /// assert_eq!(&output[4..], &format::jenkins_lookup3(&output[..4]).to_le_bytes());
    /// let parsed = page.parse(FileRegion::new(page.parse_range(), &output)?)?;
    /// assert_eq!(parsed.records(16).collect::<Result<Vec<_>, _>>()?, [(9, record)]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn emit_records<E: From<FormatError>>(
        &self,
        window: &FixedArrayEntryWindow,
        records: &[Option<ChunkRecord>],
        mut write: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        if records.len().to_u64() != window.entries() {
            return Err(FormatError::FixedArrayEntryCount {
                expected: window.entries(),
                actual: records.len().to_u64(),
            }
            .into());
        }
        let maximum =
            NonZeroUsize::new(records.len()).ok_or(FormatError::InvalidFixedArrayGeometry)?;
        if self.window(window.first_slot(), maximum)? != Some(*window) {
            return Err(FormatError::InvalidFixedArrayGeometry.into());
        }
        for &record in records {
            self.encoding.encode(record)?;
        }
        let slots = self.slots()?;
        let stride = u64::from(self.encoding.stride().get());
        let empty = self.encoding.encode(None)?;
        let mut checksum = JenkinsLookup3::new(self.checksummed_range()?.length());
        mutation::emit_repeated(
            &mut write,
            Some(&mut checksum),
            empty.as_bytes(),
            geometry::product(window.first_slot() - slots.start, stride)?,
        )?;
        mutation::emit_records(&mut write, &mut checksum, self.encoding, records)?;
        mutation::emit_repeated(
            &mut write,
            Some(&mut checksum),
            empty.as_bytes(),
            geometry::product(slots.end - window.end_slot(), stride)?,
        )?;
        write(&checksum.finish()?.to_le_bytes())
    }

    /// Encodes unallocated elements and their checksum into `output`.
    ///
    /// Returns the encoded page length, preserving trailing bytes. The caller sets the page's
    /// bitmap bit only after publishing the encoded page. See
    /// [`FixedArrayEntryWindow::encode_records`] for an example of preparing a page.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is too short, or
    /// [`FormatError::ValueTooLargeForPlatform`] if the page length exceeds [`usize::MAX`].
    /// Preserves `output` on error.
    pub fn encode_empty(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        let Self {
            range,
            first_slot: _,
            encoding,
        } = *self;
        let length = range.length().to_usize()?;
        bytes::ensure_len(output, 0, length)?;
        let body = length
            .checked_sub(CHECKSUM_LEN.to_usize()?)
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        let element = encoding.encode(None)?;
        for field in output[..body].chunks_exact_mut(element.as_bytes().len()) {
            field.copy_from_slice(element.as_bytes());
        }
        let checksum = checksum::jenkins_lookup3(&output[..body]);
        output[body..length].copy_from_slice(&checksum.to_le_bytes());
        Ok(length)
    }

    /// Returns at most `maximum` entries beginning at the array-wide slot `first`.
    ///
    /// Returns `None` if `first` is outside this page. Limits the window to the entries remaining
    /// in this page and excludes its checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if a slot offset or length overflows,
    /// or [`FormatError::OffsetOverflow`] if its physical range ends past [`u64::MAX`].
    pub fn window(
        &self,
        first: u64,
        maximum: NonZeroUsize,
    ) -> Result<Option<FixedArrayEntryWindow>, FormatError> {
        let Self {
            range,
            first_slot,
            encoding,
        } = *self;
        let slots = self.slots()?;
        if !slots.contains(&first) {
            return Ok(None);
        }
        let entries = (slots.end - first).min(maximum.get().to_u64());
        let offset = geometry::product(first - first_slot, u64::from(encoding.stride().get()))?;
        Ok(Some(FixedArrayEntryWindow::new(
            range.start().checked_add(offset)?,
            first,
            entries,
            encoding,
        )?))
    }

    /// Returns the page's array-wide slot range, excluding its checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if the range overflows or the page is
    /// shorter than its checksum.
    pub fn slots(&self) -> Result<Range<u64>, FormatError> {
        let Self {
            range,
            first_slot,
            encoding,
        } = *self;
        let body = range
            .length()
            .checked_sub(CHECKSUM_LEN)
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        let entries = body / u64::from(encoding.stride().get());
        Ok(first_slot..geometry::sum([first_slot, entries])?)
    }
}

/// A data block with the page geometry derived from its header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PagedBlock {
    block: FixedArrayDataBlockRef,
    paging: Paging,
}

impl PagedBlock {
    pub(super) fn new(block: &FixedArrayDataBlockRef) -> Option<Self> {
        Some(Self {
            block: *block,
            paging: block.geometry().paging()?,
        })
    }

    pub(super) const fn bitmap(self, bitmap: &[u8]) -> FixedArrayPageBitmap<'_> {
        FixedArrayPageBitmap {
            block: self,
            bitmap,
        }
    }

    pub(super) fn page_for_slot(
        &self,
        slot: u64,
    ) -> Result<Option<FixedArrayPageRef>, FormatError> {
        self.index(slot).map(|index| self.page(index)).transpose()
    }

    pub(super) fn bitmap_byte(
        &self,
        slot: u64,
    ) -> Result<Option<(FileRange, NonZeroU8)>, FormatError> {
        let Some(index) = self.index(slot) else {
            return Ok(None);
        };
        let address = self.block.range().start().checked_add(geometry::sum([
            data_block::data_block_prefix_len(self.block.geometry().encoding().offsets()),
            index / u64::from(u8::BITS),
        ])?)?;
        let mask = NonZeroU8::new(PAGE_ZERO_MASK >> (index % u64::from(u8::BITS)))
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        Ok(Some((FileRange::new(address, 1)?, mask)))
    }

    fn index(&self, slot: u64) -> Option<u64> {
        (slot < self.block.geometry().entries()).then(|| slot / self.paging.page_entries())
    }

    fn page(&self, index: u64) -> Result<FixedArrayPageRef, FormatError> {
        let Self { block, paging } = *self;
        let geometry = block.geometry();
        let first_slot = geometry::product(index, paging.page_entries())?;
        let entries = geometry
            .entries()
            .checked_sub(first_slot)
            .ok_or(FormatError::InvalidFixedArrayGeometry)?
            .min(paging.page_entries());
        let offset = geometry::sum([
            paging.pages_offset(geometry.encoding().offsets())?,
            geometry::product(index, self.page_len(paging.page_entries())?)?,
        ])?;
        Ok(FixedArrayPageRef {
            range: FileRange::new(
                block.range().start().checked_add(offset)?,
                self.page_len(entries)?,
            )?,
            first_slot,
            encoding: geometry.encoding(),
        })
    }

    /// Returns the length in bytes of a page of `entries` elements, with its checksum.
    fn page_len(&self, entries: u64) -> Result<u64, FormatError> {
        geometry::sum([
            geometry::product(
                entries,
                u64::from(self.block.geometry().encoding().stride().get()),
            )?,
            CHECKSUM_LEN,
        ])
    }
}

// H5VM_bit_set stores page zero in the high bit (H5VMprivate.h, HDF5 2.2.0).
const PAGE_ZERO_MASK: u8 = 0x80;
