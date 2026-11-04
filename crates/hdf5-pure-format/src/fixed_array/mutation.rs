//! Checked entry updates and empty data-block emission for version 0 Fixed Arrays.

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
        let Self {
            range,
            widths,
            base,
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
        plan.place(base.relative(range.start())?, data_block)
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
