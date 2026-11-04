//! Checked entry updates and bounded data-block and page emission for version 0 Fixed Arrays.

use core::num::NonZeroU8;
use core::num::NonZeroUsize;
use core::ops::Range;

use hdf5_pure_core::__private::FileRange;

use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::checksum::JenkinsLookup3;
use crate::chunk_record::ArrayClient;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;

use super::CHECKSUM_LEN;
use super::DATA_BLOCK_SIGNATURE;
use super::FixedArrayDataBlockRef;
use super::FixedArrayEntryWindow;
use super::FixedArrayHeader;
use super::FixedArrayPageBitmap;
use super::FixedArrayPageRef;
use super::FixedArrayPlacement;
use super::FixedArrayPlan;
use super::VERSION;

impl FixedArrayHeader {
    /// Plans an empty data block at `data_block` for this header's recorded geometry.
    ///
    /// Retains the header's stored address, widths, client, entry size, entry count, and page bits.
    /// The caller establishes ownership of the header and data-block allocations before writing.
    /// An existing data-block address does not affect the plan.
    ///
    /// # Errors
    ///
    /// Returns the placement errors of [`FixedArrayPlan::place`],
    /// [`FormatError::InvalidFixedArrayGeometry`] if the block lengths or their sum overflow, or
    /// [`FormatError::AddressBelowBase`] if the header precedes its recorded base.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{BaseAddress, PhysicalAddress, StoredAddress};
    /// use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let widths = format::FormatWidths::from_sizes(2, 4)?;
    /// let placement = format::FixedArrayPlan::new(3, 16, widths, false)?
    ///     .place(StoredAddress::new(64), StoredAddress::new(256))?;
    /// let bytes = placement.serialize_header()?;
    /// let range = FileRange::new(PhysicalAddress::new(64), bytes.len() as u64)?;
    /// let locator = format::FixedArrayHeaderLocator::new(
    ///     widths, BaseAddress::ZERO, StoredAddress::new(64),
    /// );
    /// let header = locator.parse(FileRegion::new(range, &bytes)?)?;
    /// assert_eq!(header.entries(), 3);
    /// assert_eq!(header.page_bits(), 10);
    /// let plan = header.initialization_plan()?;
    /// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [18, 18]);
    /// let replacement = header.initialization_at(StoredAddress::new(512))?;
    /// let bytes = replacement.serialize_header()?;
    /// let updated = locator.parse(FileRegion::new(range, &bytes)?)?;
    /// assert_eq!(updated.entries(), header.entries());
    /// assert_eq!(updated.page_bits(), header.page_bits());
    /// assert_eq!(updated.data_block().unwrap().range().start(), PhysicalAddress::new(512));
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn initialization_at(
        &self,
        data_block: StoredAddress,
    ) -> Result<FixedArrayPlacement, FormatError> {
        self.initialization_plan()?
            .place(self.base.relative(self.range.start())?, data_block)
    }

    /// Plans empty storage with the recorded widths, entry encoding, and page bits.
    ///
    /// Does not choose addresses or establish allocation ownership. See
    /// [`Self::initialization_at`] for an example of reusing parsed geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if a block length or their sum overflows.
    pub fn initialization_plan(&self) -> Result<FixedArrayPlan, FormatError> {
        let Self {
            range,
            widths,
            base: _,
            entry_size,
            page_bits,
            geometry,
            data_block: _,
        } = *self;
        let elements = geometry.encoding;
        let encoding = ChunkElementEncoding {
            chunk_size_bytes: match elements.client() {
                ArrayClient::Chunks => 0,
                ArrayClient::FilteredChunks { chunk_size_len } => usize::from(chunk_size_len),
            },
            elem_size: usize::from(elements.stride().get()),
            client_id: elements.client().id(),
        };
        let data_block_length = geometry.data_block_len()?;
        let plan = FixedArrayPlan {
            widths,
            geometry,
            encoding,
            entry_size,
            page_bits,
            data_block_length,
            total_length: super::sum([range.length(), data_block_length])?,
        };
        Ok(plan)
    }

    /// Returns the array's maximum entry count, including unallocated slots.
    pub const fn entries(&self) -> u64 {
        self.geometry.entries
    }

    /// Returns the base 2 logarithm of the maximum entries per page.
    pub const fn page_bits(&self) -> u8 {
        self.page_bits
    }
}

impl FixedArrayDataBlockRef {
    /// Returns the data-block bytes covered by its checksum, excluding the checksum field.
    ///
    /// For a paged version 0 block, includes only the prefix and initialization bitmap. The
    /// block is defined in "The Fixed Array Index" of the [format specification, version
    /// 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns the geometry and range errors of [`Self::parse_range`].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr
    pub fn checksummed_range(&self) -> Result<FileRange, FormatError> {
        checksum_ranges(self.parse_range()?).map(|(body, _)| body)
    }

    /// Returns the checksum field following [`Self::checksummed_range`].
    ///
    /// # Errors
    ///
    /// Returns the geometry and range errors of [`Self::parse_range`].
    pub fn checksum_range(&self) -> Result<FileRange, FormatError> {
        checksum_ranges(self.parse_range()?).map(|(_, checksum)| checksum)
    }

    /// Returns the page containing `slot`, including an uninitialized page.
    ///
    /// Returns `None` for an unpaged block or a slot at or beyond the entry count. The caller
    /// checks the page-init bitmap separately. See [`FixedArrayEntryWindow::encode_records`] for
    /// an example of preparing page entries.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if a page offset or length overflows,
    /// or [`FormatError::OffsetOverflow`] if its physical range ends past [`u64::MAX`].
    pub fn page_for_slot(&self, slot: u64) -> Result<Option<FixedArrayPageRef>, FormatError> {
        if slot >= self.geometry.entries {
            return Ok(None);
        }
        self.geometry
            .paging
            .map(|paging| self.page(paging, slot / paging.page_entries))
            .transpose()
    }

    /// Returns the physical bitmap-byte range and bit mask for the page containing `slot`.
    ///
    /// Returns `None` for an unpaged block or a slot at or beyond the entry count. The caller
    /// writes and confirms the page before setting its initialization bit.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if an offset overflows, or
    /// [`FormatError::OffsetOverflow`] if the bitmap-byte range ends past [`u64::MAX`].
    pub fn page_bitmap_byte(
        &self,
        slot: u64,
    ) -> Result<Option<(FileRange, NonZeroU8)>, FormatError> {
        let Self {
            range, geometry, ..
        } = *self;
        let Some(paging) = geometry.paging.filter(|_| slot < geometry.entries) else {
            return Ok(None);
        };
        let page = slot / paging.page_entries;
        let address = range.start().checked_add(super::sum([
            super::data_block_prefix_len(geometry.encoding.offsets()),
            page / u64::from(u8::BITS),
        ])?)?;
        let mask = NonZeroU8::new(PAGE_ZERO_MASK >> (page % u64::from(u8::BITS)))
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        Ok(Some((FileRange::new(address, 1)?, mask)))
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
        let Self {
            block,
            paging,
            bitmap,
        } = *self;
        Ok(block.page_for_slot(slot)?.map(|page| {
            (
                page,
                crate::chunk_record::page_initialized(bitmap, slot / paging.page_entries),
            )
        }))
    }
}

impl FixedArrayPageRef {
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
        checksum_ranges(self.range).map(|(body, _)| body)
    }

    /// Returns the checksum field following [`Self::checksummed_range`].
    ///
    /// # Errors
    ///
    /// Returns the geometry and range errors of [`Self::checksummed_range`].
    pub fn checksum_range(&self) -> Result<FileRange, FormatError> {
        checksum_ranges(self.range).map(|(_, checksum)| checksum)
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
        if records.len().to_u64() != window.entries {
            return Err(FormatError::FixedArrayEntryCount {
                expected: window.entries,
                actual: records.len().to_u64(),
            }
            .into());
        }
        let maximum =
            NonZeroUsize::new(records.len()).ok_or(FormatError::InvalidFixedArrayGeometry)?;
        if self.window(window.first_slot, maximum)? != Some(*window) {
            return Err(FormatError::InvalidFixedArrayGeometry.into());
        }
        for &record in records {
            self.encoding.encode(record)?;
        }
        let slots = self.slots()?;
        let stride = u64::from(self.encoding.stride().get());
        let empty = self.encoding.encode(None)?;
        let mut checksum = JenkinsLookup3::new(self.checksummed_range()?.length());
        emit_repeated(
            &mut write,
            Some(&mut checksum),
            empty.as_bytes(),
            super::product(window.first_slot - slots.start, stride)?,
        )?;
        emit_records(&mut write, &mut checksum, self.encoding, records)?;
        emit_repeated(
            &mut write,
            Some(&mut checksum),
            empty.as_bytes(),
            super::product(slots.end - window.end_slot(), stride)?,
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
        let offset = super::product(first - first_slot, u64::from(encoding.stride().get()))?;
        Ok(Some(FixedArrayEntryWindow {
            range: FileRange::new(
                range.start().checked_add(offset)?,
                super::product(entries, u64::from(encoding.stride().get()))?,
            )?,
            first_slot: first,
            entries,
            encoding,
        }))
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
        Ok(first_slot..super::sum([first_slot, entries])?)
    }
}

impl FixedArrayEntryWindow {
    /// Encodes one optional chunk record per window entry into `output`.
    ///
    /// Encodes `None` as the undefined address and, for filtered entries, zero size and mask.
    /// Unfiltered entries omit size and mask. Filtered sizes use the header's entry encoding,
    /// independently of the file's length width. Returns the byte count, preserving trailing
    /// bytes. The caller updates the containing block's or page's checksum separately.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FixedArrayEntryCount`] if the record count differs from the window's,
    /// [`FormatError::AddressTooLarge`] if an address exceeds the offset width,
    /// [`FormatError::DefinedAddressIsUndefined`] if a defined address equals the undefined
    /// marker, or [`FormatError::LengthTooLarge`] if a filtered size exceeds its field width.
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the window length exceeds
    /// [`usize::MAX`], or [`FormatError::UnexpectedEof`] if `output` is too short.
    /// Preserves `output` on error, including an error in a later record.
    ///
    /// # Examples
    ///
    /// Preparing entries in an unpaged block and in the last page of a paged block:
    ///
    /// ```
    /// use core::num::NonZeroUsize;
    /// use hdf5_pure_core::{BaseAddress, PhysicalAddress, StoredAddress};
    /// use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let widths = format::FormatWidths::from_sizes(2, 4)?;
    /// for entries in [2, 1_025] {
    ///     let placed = format::FixedArrayPlan::new(entries, 16, widths, false)?
    ///         .place(StoredAddress::new(64), StoredAddress::new(256))?;
    ///     let bytes = placed.serialize_header()?;
    ///     let locator = format::FixedArrayHeaderLocator::new(
    ///         widths, BaseAddress::ZERO, StoredAddress::new(64),
    ///     );
    ///     let header = locator.parse(FileRegion::new(
    ///         FileRange::new(PhysicalAddress::new(64), bytes.len() as u64)?, &bytes,
    ///     )?)?;
    ///     let block = header.data_block().unwrap();
    ///     let slot = entries - 1;
    ///     let maximum = NonZeroUsize::new(2).unwrap();
    ///     let window = if let Some(page) = block.page_for_slot(slot)? {
    ///         assert_eq!(page.slots()?, 1_024..1_025);
    ///         let (byte, mask) = block.page_bitmap_byte(slot)?.unwrap();
    ///         assert_eq!(byte.length(), 1);
    ///         assert_eq!(mask.get(), 0x40);
    ///         let mut empty = vec![0; usize::try_from(page.parse_range().length())?];
    ///         let length = page.encode_empty(&mut empty)?;
    ///         assert_eq!(length, empty.len());
    ///         assert!(page.parse(FileRegion::new(page.parse_range(), &empty)?)?
    ///             .records(16).next().is_none());
    ///         page.window(slot, maximum)?.unwrap()
    ///     } else {
    ///         let mut data = Vec::new();
    ///         placed.emit_empty_data_block::<FormatError>(|bytes| {
    ///             data.extend_from_slice(bytes);
    ///             Ok(())
    ///         })?;
    ///         block.parse_unpaged_prefix(FileRegion::new(block.range(), &data)?)?
    ///             .window(slot, maximum)?.unwrap()
    ///     };
    ///     assert_eq!(window.end_slot(), entries);
    ///     let mut bytes = [0xcc; 3];
    ///     assert_eq!(window.encode_records(&mut bytes, &[Some(format::ChunkRecord {
    ///         address: StoredAddress::new(0x1234), stored_size: 16, filter_mask: 0,
    ///     })])?, 2);
    ///     assert_eq!(bytes, [0x34, 0x12, 0xcc]);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn encode_records(
        &self,
        output: &mut [u8],
        records: &[Option<ChunkRecord>],
    ) -> Result<usize, FormatError> {
        let Self {
            range,
            first_slot: _,
            entries,
            encoding,
        } = *self;
        if records.len().to_u64() != entries {
            return Err(FormatError::FixedArrayEntryCount {
                expected: entries,
                actual: records.len().to_u64(),
            });
        }
        for &record in records {
            encoding.encode(record)?;
        }
        let length = range.length().to_usize()?;
        bytes::ensure_len(output, 0, length)?;
        for (field, &record) in output[..length]
            .chunks_exact_mut(usize::from(encoding.stride().get()))
            .zip(records)
        {
            field.copy_from_slice(encoding.encode(record)?.as_bytes());
        }
        Ok(length)
    }
}

impl FixedArrayPlan {
    /// Selects the page threshold as a base 2 exponent and recomputes complete block lengths.
    ///
    /// Paging starts when the entry count exceeds `2^page_bits`. See [`Self::encode_record`] for
    /// an example combining custom paging with independent field widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if `page_bits` is 64 or more, or a block
    /// length or their sum overflows.
    pub fn with_page_bits(mut self, page_bits: u8) -> Result<Self, FormatError> {
        self.geometry =
            super::Geometry::new(self.geometry.encoding, self.geometry.entries, page_bits)?;
        self.page_bits = page_bits;
        self.data_block_length = self.geometry.data_block_len()?;
        self.total_length = super::sum([super::header_len(self.widths), self.data_block_length])?;
        Ok(self)
    }

    /// Encodes one entry at the plan's retained address and filtered-size widths.
    ///
    /// Writes `None` as an undefined address with zero filtered fields. Returns the encoded byte
    /// count, preserving trailing bytes and all output on error. Filtered sizes use the entry
    /// encoding, independently of the file's length width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is too short. Returns the address and
    /// size errors of [`FixedArrayEntryWindow::encode_records`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let plan = format::FixedArrayPlan::new(
    ///     9, 16, format::FormatWidths::from_sizes(2, 4)?, true,
    /// )?.with_page_bits(3)?;
    /// assert_eq!(plan.page_bits(), 3);
    /// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [18, 93]);
    /// let mut output = [0xcc; 10];
    /// let length = plan.encode_record(&mut output, Some(format::ChunkRecord {
    ///     address: StoredAddress::new(0x1234), stored_size: 0x56, filter_mask: 3,
    /// }))?;
    /// assert_eq!(length, 8);
    /// assert_eq!(output, [0x34, 0x12, 0x56, 0, 3, 0, 0, 0, 0xcc, 0xcc]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn encode_record(
        &self,
        output: &mut [u8],
        record: Option<ChunkRecord>,
    ) -> Result<usize, FormatError> {
        let element = self.geometry.encoding.encode(record)?;
        let bytes = element.as_bytes();
        bytes::ensure_len(output, 0, bytes.len())?;
        output[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }

    /// Returns the base 2 logarithm of the maximum entries per page.
    pub const fn page_bits(&self) -> u8 {
        self.page_bits
    }
}

impl FixedArrayPlacement {
    /// Writes an empty data block to successive sink windows.
    ///
    /// Writes undefined entries for an unpaged block. For a paged block, writes a zero
    /// initialization bitmap and zero bytes across the reserved page ranges. The caller initializes
    /// a page before setting its bitmap bit. Computes the prefix or whole-block checksum
    /// regardless of the `checksum` feature.
    ///
    /// Uses a 4 KiB stack buffer and a bounded prefix allocation, independently of the entry
    /// count. The caller controls retention of the emitted bytes and publication of references.
    /// The caller obtains the header bytes from [`Self::serialize_header`].
    ///
    /// # Errors
    ///
    /// Returns a sink error unchanged, including errors after partial output. Converts
    /// [`FormatError::AllocationFailed`] if the prefix reservation fails. Geometry or length
    /// failures are reported through [`FormatError`] before the affected sink window.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let plan = format::FixedArrayPlan::new(
    ///     3, 16, format::FormatWidths::from_sizes(2, 4)?, false,
    /// )?;
    /// let expected = plan.block_lengths().nth(1).unwrap();
    /// let placed = plan.place(StoredAddress::new(64), StoredAddress::new(256))?;
    /// let mut bytes_written = 0_u64;
    /// placed.emit_empty_data_block::<FormatError>(|bytes| {
    ///     bytes_written += bytes.len() as u64;
    ///     Ok(())
    /// })?;
    /// assert_eq!(bytes_written, expected);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn emit_empty_data_block<E: From<FormatError>>(
        &self,
        mut write: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        let Self {
            plan,
            header,
            data_block,
        } = *self;
        let FixedArrayPlan {
            widths,
            geometry,
            encoding: _,
            entry_size: _,
            page_bits: _,
            data_block_length: _,
            total_length: _,
        } = plan;
        let mut prefix = super::output_buffer(super::data_block_prefix_len(widths.offsets))?;
        prefix.extend_from_slice(&DATA_BLOCK_SIGNATURE);
        prefix.push(VERSION);
        prefix.push(geometry.encoding.client().id());
        bytes::try_write_defined_offset(&mut prefix, header.start().get(), widths.offsets)?;
        let body_length = match geometry.paging {
            Some(paging) => super::sum([
                prefix.len().to_u64(),
                paging.pages.div_ceil(u64::from(u8::BITS)),
            ])?,
            None => data_block
                .length()
                .checked_sub(CHECKSUM_LEN)
                .ok_or(FormatError::InvalidFixedArrayGeometry)?,
        };
        let mut checksum = JenkinsLookup3::new(body_length);
        checksum.update(&prefix)?;
        write(&prefix)?;
        match geometry.paging {
            Some(paging) => emit_repeated(
                &mut write,
                Some(&mut checksum),
                &[0],
                paging.pages.div_ceil(u64::from(u8::BITS)),
            )?,
            None => {
                let element = geometry.encoding.encode(None)?;
                emit_repeated(
                    &mut write,
                    Some(&mut checksum),
                    element.as_bytes(),
                    super::product(
                        geometry.entries,
                        u64::from(geometry.encoding.stride().get()),
                    )?,
                )?;
            }
        }
        write(&checksum.finish()?.to_le_bytes())?;
        let tail = data_block
            .length()
            .checked_sub(super::sum([body_length, CHECKSUM_LEN])?)
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        emit_repeated(&mut write, None, &[0], tail)
    }
}

fn emit_records<E: From<FormatError>>(
    write: &mut impl FnMut(&[u8]) -> Result<(), E>,
    checksum: &mut JenkinsLookup3,
    encoding: ElementEncoding,
    records: &[Option<ChunkRecord>],
) -> Result<(), E> {
    let stride = usize::from(encoding.stride().get());
    let mut scratch = [0; EMISSION_SCRATCH_LEN];
    let maximum = scratch.len() / stride;
    if maximum == 0 {
        return Err(FormatError::InvalidFixedArrayGeometry.into());
    }
    for records in records.chunks(maximum) {
        let length = records.len() * stride;
        for (bytes, &record) in scratch[..length].chunks_exact_mut(stride).zip(records) {
            bytes.copy_from_slice(encoding.encode(record)?.as_bytes());
        }
        checksum.update(&scratch[..length])?;
        write(&scratch[..length])?;
    }
    Ok(())
}

fn checksum_ranges(range: FileRange) -> Result<(FileRange, FileRange), FormatError> {
    let body_length = range
        .length()
        .checked_sub(CHECKSUM_LEN)
        .ok_or(FormatError::InvalidFixedArrayGeometry)?;
    let body = FileRange::new(range.start(), body_length)?;
    Ok((body, FileRange::new(body.end(), CHECKSUM_LEN)?))
}

fn emit_repeated<E: From<FormatError>>(
    write: &mut impl FnMut(&[u8]) -> Result<(), E>,
    mut checksum: Option<&mut JenkinsLookup3>,
    pattern: &[u8],
    mut length: u64,
) -> Result<(), E> {
    let Some(stride) = NonZeroUsize::new(pattern.len()) else {
        return Err(FormatError::Internal("an emitted byte pattern is empty".into()).into());
    };
    let mut scratch = [0; EMISSION_SCRATCH_LEN];
    let used = scratch.len() / stride.get() * stride.get();
    if used == 0 || length % stride.get().to_u64() != 0 {
        return Err(FormatError::InvalidFixedArrayGeometry.into());
    }
    for element in scratch[..used].chunks_exact_mut(stride.get()) {
        element.copy_from_slice(pattern);
    }
    while length != 0 {
        let count = length.min(used.to_u64()).to_usize()?;
        let bytes = &scratch[..count];
        if let Some(checksum) = checksum.as_mut() {
            checksum.update(bytes)?;
        }
        write(bytes)?;
        length -= count.to_u64();
    }
    Ok(())
}

// H5VM_bit_set stores page zero in the high bit (H5VMprivate.h, HDF5 2.2.0).
const PAGE_ZERO_MASK: u8 = 0x80;
const EMISSION_SCRATCH_LEN: usize = 4_096;
