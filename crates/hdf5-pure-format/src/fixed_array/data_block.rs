use core::num::NonZeroU8;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use super::framing;
use super::framing::CHECKSUM_LEN;
use super::framing::SIGNATURE_LEN;
use super::framing::VERSION;
use super::geometry;
use super::geometry::Geometry;
use super::page::FixedArrayPageBitmap;
use super::page::FixedArrayPageRef;
use super::page::PagedBlock;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::bytes;
use crate::checksum;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The position and layout of a Fixed Array data block, from
/// [`FixedArrayHeader::data_block`](super::FixedArrayHeader::data_block).
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse). For an unpaged block, read the whole block. For a paged block,
/// read the prefix, page-init bitmap, and trailing checksum. For bounded reads of an unpaged
/// block, read
/// [`unpaged_prefix_range`](Self::unpaged_prefix_range) and pass the bytes to
/// [`parse_unpaged_prefix`](Self::parse_unpaged_prefix).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayDataBlockRef {
    base: BaseAddress,
    header: PhysicalAddress,
    range: FileRange,
    geometry: Geometry,
}

impl FixedArrayDataBlockRef {
    /// Parses the data block in `region`, and returns its elements or, in a paged block, its
    /// page-init bitmap.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidFixedArraySignature`] if the
    /// bytes do not start with `FADB`, [`FormatError::InvalidFixedArrayVersion`] if the version is
    /// not 0, [`FormatError::InvalidFixedArrayBlockClient`] if the client ID differs from the
    /// header's, and [`FormatError::InvalidFixedArrayBlockOwner`] if the header address differs
    /// from the header's own. Returns [`FormatError::OffsetOverflow`] if the base address plus the
    /// header address is past [`u64::MAX`], and, under the `checksum` feature,
    /// [`FormatError::ChecksumMismatch`] if the checksum does not match the bytes of the range.
    pub fn parse<'a>(
        &self,
        region: FileRegion<'a>,
    ) -> Result<FixedArrayDataBlock<'a>, FormatError> {
        let geometry = self.geometry;
        let bytes = region.subregion(self.parse_range()?)?.bytes();
        self.parse_prefix(region)?;
        checksum::verify_trailing(bytes)?;

        let body = framing::without_checksum(bytes)?
            .get(data_block_prefix_len(geometry.encoding().offsets()).to_usize()?..)
            .ok_or_else(|| {
                FormatError::Internal("a Fixed Array data block ends inside its prefix".into())
            })?;
        Ok(match PagedBlock::new(self) {
            Some(block) => FixedArrayDataBlock::PageBitmap(block.bitmap(body)),
            None => {
                FixedArrayDataBlock::Elements(FixedArrayElements::new(body, 0, geometry.encoding()))
            }
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] or [`FormatError::OffsetOverflow`] if
    /// the range ends past [`u64::MAX`].
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        let Self {
            range, geometry, ..
        } = *self;
        match geometry.paging() {
            Some(paging) => FileRange::new(
                range.start(),
                paging.pages_offset(geometry.encoding().offsets())?,
            ),
            None => Ok(range),
        }
    }

    /// Returns the physical range of the whole data block, the pages of a paged one included.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the prefix range for an unpaged block, or `None` for a paged block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the prefix ends past [`u64::MAX`]. The prefix
    /// is contained in the block's validated range.
    pub fn unpaged_prefix_range(&self) -> Result<Option<FileRange>, FormatError> {
        self.geometry
            .paging()
            .is_none()
            .then(|| self.prefix_range())
            .transpose()
    }

    /// Parses the signature, version, client ID, and owner of an unpaged data block.
    ///
    /// The caller verifies the checksum separately, using the ranges returned by
    /// [`FixedArrayUnpaged::checksummed_range`] and [`FixedArrayUnpaged::checksum_range`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if the block is paged,
    /// [`FormatError::InsufficientRegion`] if `region` does not contain the prefix,
    /// [`FormatError::InvalidFixedArraySignature`] if the signature is not `FADB`,
    /// [`FormatError::InvalidFixedArrayVersion`] if the version is not 0,
    /// [`FormatError::InvalidFixedArrayBlockClient`] if the client ID differs from the header's,
    /// and [`FormatError::InvalidFixedArrayBlockOwner`] if the owner differs from the header's
    /// physical address. Returns [`FormatError::OffsetOverflow`] if the base address plus the
    /// owner address is past [`u64::MAX`].
    pub fn parse_unpaged_prefix(
        &self,
        region: FileRegion<'_>,
    ) -> Result<FixedArrayUnpaged, FormatError> {
        if self.geometry.paging().is_some() {
            return Err(FormatError::InvalidFixedArrayGeometry);
        }
        self.parse_prefix(region)?;
        Ok(FixedArrayUnpaged { block: *self })
    }

    /// Returns the fixed signature, version, client, and header-address prefix range.
    ///
    /// Excludes entry bytes, the page bitmap, and checksums. Version 0 blocks are defined in
    /// "The Fixed Array Index" of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the physical range overflows.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr
    pub fn prefix_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(
            self.range.start(),
            data_block_prefix_len(self.geometry.encoding().offsets()),
        )
    }

    /// Checks the version 0 signature, client, and owner in a data-block prefix.
    ///
    /// The caller verifies the checksum separately with [`Self::checksummed_range`] and
    /// [`Self::checksum_range`]. Accepts paged and unpaged blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] for a missing prefix,
    /// [`FormatError::InvalidFixedArraySignature`] for a signature other than `FADB`,
    /// [`FormatError::InvalidFixedArrayVersion`] for a version other than 0,
    /// [`FormatError::InvalidFixedArrayBlockClient`] for a different client, or
    /// [`FormatError::InvalidFixedArrayBlockOwner`] for a different owner. Propagates range and
    /// address-conversion errors.
    pub fn parse_prefix(&self, region: FileRegion<'_>) -> Result<(), FormatError> {
        let Self {
            base,
            header,
            geometry,
            ..
        } = *self;
        let mut cursor = bytes::ByteCursor::new(region.subregion(self.prefix_range()?)?.bytes());
        if cursor.array::<{ DATA_BLOCK_SIGNATURE.len() }>()? != DATA_BLOCK_SIGNATURE {
            return Err(FormatError::InvalidFixedArraySignature);
        }
        let version = cursor.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidFixedArrayVersion(version));
        }
        let client_id = cursor.u8()?;
        let expected_client = geometry.encoding().client().id();
        if client_id != expected_client {
            return Err(FormatError::InvalidFixedArrayBlockClient {
                expected: expected_client,
                actual: client_id,
            });
        }
        let owner = base.resolve(cursor.address(geometry.encoding().offsets())?)?;
        if owner != header {
            return Err(FormatError::InvalidFixedArrayBlockOwner {
                expected: header.get(),
                actual: owner.get(),
            });
        }
        Ok(())
    }

    pub(super) const fn geometry(&self) -> Geometry {
        self.geometry
    }

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
        framing::checksum_ranges(self.parse_range()?).map(|(body, _)| body)
    }

    /// Returns the checksum field following [`Self::checksummed_range`].
    ///
    /// # Errors
    ///
    /// Returns the geometry and range errors of [`Self::parse_range`].
    pub fn checksum_range(&self) -> Result<FileRange, FormatError> {
        framing::checksum_ranges(self.parse_range()?).map(|(_, checksum)| checksum)
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
        match PagedBlock::new(self) {
            Some(block) => block.page_for_slot(slot),
            None => Ok(None),
        }
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
        match PagedBlock::new(self) {
            Some(block) => block.bitmap_byte(slot),
            None => Ok(None),
        }
    }
}

/// An unpaged Fixed Array data block whose prefix has been parsed.
///
/// The caller can read entry windows independently of the whole block's size. Parsing the
/// prefix and the windows does not verify the checksum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayUnpaged {
    block: FixedArrayDataBlockRef,
}

impl FixedArrayUnpaged {
    /// Returns a window of at most `max_entries` entries beginning at `first_slot`.
    ///
    /// Returns `None` if `first_slot` is at or beyond the array's entry count. The last window
    /// may contain fewer entries. Slots are indexed from the start of the whole array.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if a byte length overflows, or
    /// [`FormatError::OffsetOverflow`] if a physical position is past [`u64::MAX`]. Every window
    /// is contained in the block's validated range.
    pub fn window(
        &self,
        first_slot: u64,
        max_entries: NonZeroUsize,
    ) -> Result<Option<FixedArrayEntryWindow>, FormatError> {
        let FixedArrayDataBlockRef {
            range, geometry, ..
        } = self.block;
        if first_slot >= geometry.entries() {
            return Ok(None);
        }
        let entries = (geometry.entries() - first_slot).min(max_entries.get() as u64);
        let offset = geometry::sum([
            data_block_prefix_len(geometry.encoding().offsets()),
            geometry::product(first_slot, u64::from(geometry.encoding().stride().get()))?,
        ])?;
        Ok(Some(FixedArrayEntryWindow::new(
            range.start().checked_add(offset)?,
            first_slot,
            entries,
            geometry.encoding(),
        )?))
    }

    /// Returns the range of the prefix and entries covered by the block's checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the range ends past [`u64::MAX`]. The range
    /// is contained in the block's validated range.
    pub fn checksummed_range(&self) -> Result<FileRange, FormatError> {
        let range = self.block.range;
        FileRange::new(range.start(), range.length() - CHECKSUM_LEN)
    }

    /// Returns the range of the block's trailing checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the range ends past [`u64::MAX`]. The range
    /// is contained in the block's validated range.
    pub fn checksum_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(self.checksummed_range()?.end(), CHECKSUM_LEN)
    }
}

/// The physical range and logical slots of entries within a Fixed Array data block or page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayEntryWindow {
    range: FileRange,
    first_slot: u64,
    end_slot: u64,
    encoding: ElementEncoding,
}

impl FixedArrayEntryWindow {
    /// Computes the byte range and exclusive end slot of `entries` consecutive entries.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if the byte length or end slot
    /// exceeds [`u64::MAX`], or [`FormatError::OffsetOverflow`] if the byte range ends past it.
    pub(super) fn new(
        address: PhysicalAddress,
        first_slot: u64,
        entries: u64,
        encoding: ElementEncoding,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            range: FileRange::new(
                address,
                geometry::product(entries, u64::from(encoding.stride().get()))?,
            )?,
            first_slot,
            end_slot: geometry::sum([first_slot, entries])?,
            encoding,
        })
    }

    pub(super) const fn first_slot(&self) -> u64 {
        self.first_slot
    }

    pub(super) const fn entries(&self) -> u64 {
        self.end_slot - self.first_slot
    }

    /// Borrows the window's entry bytes from `region`.
    ///
    /// The caller verifies the containing block's or page's checksum separately before reading
    /// records with [`FixedArrayElements::records`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the window.
    pub fn parse<'a>(&self, region: FileRegion<'a>) -> Result<FixedArrayElements<'a>, FormatError> {
        Ok(FixedArrayElements::new(
            region.subregion(self.range)?.bytes(),
            self.first_slot,
            self.encoding,
        ))
    }

    /// Returns the range the caller reads for [`parse`](Self::parse).
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }

    /// Returns the exclusive end slot, indexed from the start of the whole array.
    pub const fn end_slot(&self) -> u64 {
        self.end_slot
    }
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
            range, encoding, ..
        } = *self;
        let entries = self.entries();
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

/// A parsed Fixed Array data block, from [`FixedArrayDataBlockRef::parse`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedArrayDataBlock<'a> {
    /// The elements of an unpaged data block.
    Elements(FixedArrayElements<'a>),
    /// The page-init bitmap of a paged data block.
    PageBitmap(FixedArrayPageBitmap<'a>),
}

/// The elements of an unpaged Fixed Array data block, an entry window, or one page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayElements<'a> {
    bytes: &'a [u8],
    first_slot: u64,
    encoding: ElementEncoding,
}

impl<'a> FixedArrayElements<'a> {
    pub(super) const fn new(bytes: &'a [u8], first_slot: u64, encoding: ElementEncoding) -> Self {
        Self {
            bytes,
            first_slot,
            encoding,
        }
    }

    /// Returns the slot and the record of each element that stores a chunk address.
    ///
    /// The slot is the element's index in the whole array. A record of an unfiltered chunk has
    /// `chunk_byte_size` as its stored size and a filter mask of 0. An element that stores the
    /// undefined address has no record.
    pub fn records(
        &self,
        chunk_byte_size: u64,
    ) -> impl Iterator<Item = Result<(u64, ChunkRecord), FormatError>> + 'a {
        let Self {
            bytes,
            first_slot,
            encoding,
        } = *self;
        encoding.records(bytes, first_slot, chunk_byte_size)
    }
}

/// The data block geometry and, when the header records an address, its physical range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BlockStorage {
    geometry: Geometry,
    range: Option<FileRange>,
}

impl BlockStorage {
    pub(super) fn new(
        geometry: Geometry,
        address: Option<PhysicalAddress>,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            geometry,
            range: address
                .map(|address| FileRange::new(address, geometry.data_block_len()?))
                .transpose()?,
        })
    }

    pub(super) const fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub(super) const fn at(
        &self,
        base: BaseAddress,
        header: PhysicalAddress,
    ) -> Option<FixedArrayDataBlockRef> {
        match self.range {
            Some(range) => Some(FixedArrayDataBlockRef {
                base,
                header,
                range,
                geometry: self.geometry,
            }),
            None => None,
        }
    }
}

/// Returns the length in bytes of a data block's signature, version, client ID, and header
/// address.
pub(super) fn data_block_prefix_len(offsets: OffsetWidth) -> u64 {
    SIGNATURE_LEN + DATA_BLOCK_BYTE_FIELDS + u64::from(offsets.get())
}

pub(super) const DATA_BLOCK_SIGNATURE: [u8; 4] = *b"FADB";
/// The version and client ID bytes of a data block.
const DATA_BLOCK_BYTE_FIELDS: u64 = 2;
