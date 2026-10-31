//! The Fixed Array chunk index, index type 3 of a version 4 data layout message.
//!
//! A Fixed Array indexes the chunks of a dataset whose maximum shape is fixed. Its header
//! (`FAHD`) points at one data block (`FADB`), which holds an element per slot. Past `2^page_bits`
//! slots the data block holds a page-init bitmap and then the elements in pages, each with its
//! own checksum.
//!
//! A caller reads the bytes of each range the parser returns and passes them back to it.
//! [`FixedArrayHeaderLocator`] returns the range of the header, [`FixedArrayDataBlockRef`] the
//! range of an unpaged data block or of a paged one's prefix and page-init bitmap, and
//! [`FixedArrayPageRef`] the range of one page. [`FixedArrayUnpaged`] describes entry windows
//! and checksum ranges for an unpaged block.
//!
//! The index is defined in "The Fixed Array Index" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr

use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::checksum;
use crate::chunk_record;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::chunk_record::IndexSlots;
use crate::chunk_record::InvalidElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The address of a Fixed Array header, from the chunk index address of a data layout message.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse), which returns the header and the reference to its data block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayHeaderLocator {
    widths: FormatWidths,
    base: BaseAddress,
    address: StoredAddress,
}

impl FixedArrayHeaderLocator {
    /// Creates the locator of the header at the base-relative `address`.
    pub const fn new(widths: FormatWidths, base: BaseAddress, address: StoredAddress) -> Self {
        Self {
            widths,
            base,
            address,
        }
    }

    /// Parses the header in `region`.
    ///
    /// An unfiltered element is the chunk address alone, whatever the header's entry size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidFixedArraySignature`] if the
    /// bytes do not start with `FAHD`, [`FormatError::InvalidFixedArrayVersion`] if the version is
    /// not 0, [`FormatError::InvalidFixedArrayClient`] if the client ID is 2 or more, and
    /// [`FormatError::InvalidFixedArrayGeometry`] if the page bits, the entry size, or the entry
    /// count is out of range. Returns [`FormatError::OffsetOverflow`] if the base address plus
    /// `address` or plus the data block address, or the end of the header or of the data block,
    /// is past [`u64::MAX`], and, under the `checksum` feature, [`FormatError::ChecksumMismatch`]
    /// if the checksum does not match the header's bytes.
    pub fn parse(&self, region: FileRegion<'_>) -> Result<FixedArrayHeader, FormatError> {
        let Self {
            widths,
            base,
            address: _,
        } = *self;
        let range = self.parse_range()?;
        let bytes = region.subregion(range)?.bytes();
        let mut fields = bytes::Fields::new(bytes, 0);
        if fields.array::<{ HEADER_SIGNATURE.len() }>()? != HEADER_SIGNATURE {
            return Err(FormatError::InvalidFixedArraySignature);
        }
        let version = fields.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidFixedArrayVersion(version));
        }
        let client_id = fields.u8()?;
        let entry_size = fields.u8()?;
        let page_bits = fields.u8()?;
        let entries = fields.length(widths.lengths)?;
        let data_block = fields.decoded_address(widths.offsets)?;
        checksum::verify_trailing(bytes)?;

        let encoding = ElementEncoding::new(widths.offsets, client_id, entry_size).map_err(
            |err| match err {
                InvalidElementEncoding::Client(id) => FormatError::InvalidFixedArrayClient(id),
                InvalidElementEncoding::EntrySize => FormatError::InvalidFixedArrayGeometry,
            },
        )?;
        let geometry = Geometry::new(encoding, entries, page_bits)?;
        let data_block = match data_block {
            DecodedStoredAddress::Defined(address) => Some(FixedArrayDataBlockRef::new(
                base,
                range.start(),
                base.resolve(address)?,
                geometry,
            )?),
            DecodedStoredAddress::Undefined => None,
        };
        Ok(FixedArrayHeader { range, data_block })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the whole header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the base address plus `address`, or the end of
    /// the header, is past [`u64::MAX`].
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        let Self {
            widths,
            base,
            address,
        } = *self;
        FileRange::new(base.resolve(address)?, header_len(widths))
    }
}

/// A Fixed Array header, from [`FixedArrayHeaderLocator::parse`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayHeader {
    range: FileRange,
    data_block: Option<FixedArrayDataBlockRef>,
}

impl FixedArrayHeader {
    /// Returns the physical range of the header.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the reference to the data block, or `None` if the header stores the undefined
    /// address, in an array without chunks.
    pub const fn data_block(&self) -> Option<FixedArrayDataBlockRef> {
        self.data_block
    }
}

/// The position and the layout of a Fixed Array data block, from [`FixedArrayHeader::data_block`].
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse). The range is the whole data block if it is unpaged, and its prefix
/// and page-init bitmap if it is paged. For bounded reads of an unpaged block, the caller reads
/// [`unpaged_prefix_range`](Self::unpaged_prefix_range) and passes the bytes to
/// [`parse_unpaged_prefix`](Self::parse_unpaged_prefix).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayDataBlockRef {
    base: BaseAddress,
    header: PhysicalAddress,
    range: FileRange,
    geometry: Geometry,
}

impl FixedArrayDataBlockRef {
    fn new(
        base: BaseAddress,
        header: PhysicalAddress,
        address: PhysicalAddress,
        geometry: Geometry,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            base,
            header,
            range: FileRange::new(address, geometry.data_block_len()?)?,
            geometry,
        })
    }

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

        let body = without_checksum(bytes)?
            .get(data_block_prefix_len(geometry.encoding.offsets()).to_usize()?..)
            .ok_or_else(|| {
                FormatError::Internal("a Fixed Array data block ends inside its prefix".into())
            })?;
        Ok(match geometry.paging {
            Some(paging) => FixedArrayDataBlock::PageBitmap(FixedArrayPageBitmap {
                block: *self,
                paging,
                bitmap: body,
            }),
            None => FixedArrayDataBlock::Elements(FixedArrayElements {
                bytes: body,
                first_slot: 0,
                encoding: geometry.encoding,
            }),
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] or [`FormatError::OffsetOverflow`] if
    /// the range ends past [`u64::MAX`]. The range of every data block a header parses to ends
    /// below it.
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        let Self {
            range, geometry, ..
        } = *self;
        match geometry.paging {
            Some(paging) => FileRange::new(
                range.start(),
                paging.pages_offset(geometry.encoding.offsets())?,
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
            .paging
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
        if self.geometry.paging.is_some() {
            return Err(FormatError::InvalidFixedArrayGeometry);
        }
        self.parse_prefix(region)?;
        Ok(FixedArrayUnpaged { block: *self })
    }

    fn prefix_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(
            self.range.start(),
            data_block_prefix_len(self.geometry.encoding.offsets()),
        )
    }

    fn parse_prefix(&self, region: FileRegion<'_>) -> Result<(), FormatError> {
        let Self {
            base,
            header,
            geometry,
            ..
        } = *self;
        let mut fields = bytes::Fields::new(region.subregion(self.prefix_range()?)?.bytes(), 0);
        if fields.array::<{ DATA_BLOCK_SIGNATURE.len() }>()? != DATA_BLOCK_SIGNATURE {
            return Err(FormatError::InvalidFixedArraySignature);
        }
        let version = fields.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidFixedArrayVersion(version));
        }
        let client_id = fields.u8()?;
        let expected_client = geometry.encoding.client().id();
        if client_id != expected_client {
            return Err(FormatError::InvalidFixedArrayBlockClient {
                expected: expected_client,
                actual: client_id,
            });
        }
        let owner = base.resolve(fields.address(geometry.encoding.offsets())?)?;
        if owner != header {
            return Err(FormatError::InvalidFixedArrayBlockOwner {
                expected: header.get(),
                actual: owner.get(),
            });
        }
        Ok(())
    }

    fn page(&self, paging: Paging, index: u64) -> Result<FixedArrayPageRef, FormatError> {
        let Self {
            range, geometry, ..
        } = *self;
        let first_slot = product(index, paging.page_entries)?;
        let entries = geometry
            .entries
            .checked_sub(first_slot)
            .ok_or(FormatError::InvalidFixedArrayGeometry)?
            .min(paging.page_entries);
        let offset = sum([
            paging.pages_offset(geometry.encoding.offsets())?,
            product(index, geometry.page_len(paging.page_entries)?)?,
        ])?;
        Ok(FixedArrayPageRef {
            range: FileRange::new(
                range.start().checked_add(offset)?,
                geometry.page_len(entries)?,
            )?,
            first_slot,
            encoding: geometry.encoding,
        })
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
        if first_slot >= geometry.entries {
            return Ok(None);
        }
        let entries = (geometry.entries - first_slot).min(max_entries.get() as u64);
        let offset = sum([
            data_block_prefix_len(geometry.encoding.offsets()),
            product(first_slot, u64::from(geometry.encoding.stride().get()))?,
        ])?;
        Ok(Some(FixedArrayEntryWindow {
            range: FileRange::new(
                range.start().checked_add(offset)?,
                product(entries, u64::from(geometry.encoding.stride().get()))?,
            )?,
            first_slot,
            entries,
            encoding: geometry.encoding,
        }))
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

/// The physical range and logical slots of entries within an unpaged Fixed Array data block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayEntryWindow {
    range: FileRange,
    first_slot: u64,
    entries: u64,
    encoding: ElementEncoding,
}

impl FixedArrayEntryWindow {
    /// Borrows the window's entry bytes from `region`.
    ///
    /// The caller verifies the whole block's checksum separately before reading records with
    /// [`FixedArrayElements::records`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the window.
    pub fn parse<'a>(&self, region: FileRegion<'a>) -> Result<FixedArrayElements<'a>, FormatError> {
        Ok(FixedArrayElements {
            bytes: region.subregion(self.range)?.bytes(),
            first_slot: self.first_slot,
            encoding: self.encoding,
        })
    }

    /// Returns the range the caller reads for [`parse`](Self::parse).
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }

    /// Returns the exclusive end slot, indexed from the start of the whole array.
    pub const fn end_slot(&self) -> u64 {
        self.first_slot + self.entries
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

/// The page-init bitmap of a paged Fixed Array data block, which marks the pages that hold
/// elements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayPageBitmap<'a> {
    block: FixedArrayDataBlockRef,
    paging: Paging,
    bitmap: &'a [u8],
}

impl<'a> FixedArrayPageBitmap<'a> {
    /// Returns the reference to each page the bitmap marks initialized, in page order.
    ///
    /// Page 0 is the most significant bit of the first byte. A page the bitmap does not mark is
    /// uninitialized and has no reference.
    pub fn pages(&self) -> impl Iterator<Item = Result<FixedArrayPageRef, FormatError>> + 'a {
        let Self {
            block,
            paging,
            bitmap,
        } = *self;
        (0..paging.pages)
            .filter(move |&index| chunk_record::page_initialized(bitmap, index))
            .map(move |index| block.page(paging, index))
    }
}

/// The position of one page of a paged Fixed Array data block, from
/// [`FixedArrayPageBitmap::pages`].
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
        Ok(FixedArrayElements {
            bytes: without_checksum(bytes)?,
            first_slot,
            encoding,
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the page's elements and its
    /// checksum.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
}

/// The elements of an unpaged Fixed Array data block, an entry window, or one page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayElements<'a> {
    bytes: &'a [u8],
    first_slot: u64,
    encoding: ElementEncoding,
}

impl<'a> FixedArrayElements<'a> {
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

/// The element encoding, the entry count, and the paging of a Fixed Array, which its header fixes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    encoding: ElementEncoding,
    /// The number of elements in the array, the maximum number of entries the header stores.
    entries: u64,
    /// `None` if the data block holds the elements itself.
    paging: Option<Paging>,
}

impl Geometry {
    fn new(encoding: ElementEncoding, entries: u64, page_bits: u8) -> Result<Self, FormatError> {
        let page_entries = 1u64
            .checked_shl(u32::from(page_bits))
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        Ok(Self {
            encoding,
            entries,
            paging: (entries > page_entries).then(|| Paging {
                page_entries,
                pages: entries.div_ceil(page_entries),
            }),
        })
    }

    fn data_block_len(&self) -> Result<u64, FormatError> {
        let Self {
            encoding,
            entries,
            paging,
        } = *self;
        let elements = product(entries, u64::from(encoding.stride().get()))?;
        match paging {
            Some(paging) => sum([
                paging.pages_offset(encoding.offsets())?,
                elements,
                product(paging.pages, CHECKSUM_LEN)?,
            ]),
            None => sum([
                data_block_prefix_len(encoding.offsets()),
                elements,
                CHECKSUM_LEN,
            ]),
        }
    }

    /// Returns the length in bytes of a page of `entries` elements, with its checksum.
    fn page_len(&self, entries: u64) -> Result<u64, FormatError> {
        sum([
            product(entries, u64::from(self.encoding.stride().get()))?,
            CHECKSUM_LEN,
        ])
    }
}

/// The pages of a paged Fixed Array data block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Paging {
    /// The number of elements in a page, 2 to the power of the header's page bits. The last page
    /// holds the remaining elements, and may hold fewer.
    page_entries: u64,
    pages: u64,
}

impl Paging {
    /// Returns the offset of the first page from the start of the data block, past the prefix,
    /// the page-init bitmap, and its checksum.
    fn pages_offset(self, offsets: OffsetWidth) -> Result<u64, FormatError> {
        sum([
            data_block_prefix_len(offsets),
            self.pages.div_ceil(8),
            CHECKSUM_LEN,
        ])
    }
}

/// Returns the length in bytes of a Fixed Array header in a file with the address and length
/// widths `widths`.
fn header_len(FormatWidths { offsets, lengths }: FormatWidths) -> u64 {
    SIGNATURE_LEN
        + HEADER_BYTE_FIELDS
        + u64::from(lengths.get())
        + u64::from(offsets.get())
        + CHECKSUM_LEN
}

/// Returns the length in bytes of a data block's signature, version, client ID, and header
/// address.
fn data_block_prefix_len(offsets: OffsetWidth) -> u64 {
    SIGNATURE_LEN + DATA_BLOCK_BYTE_FIELDS + u64::from(offsets.get())
}

/// Returns `a * b`, or [`FormatError::InvalidFixedArrayGeometry`] if the product overflows.
fn product(a: u64, b: u64) -> Result<u64, FormatError> {
    a.checked_mul(b)
        .ok_or(FormatError::InvalidFixedArrayGeometry)
}

/// Returns the sum of `terms`, or [`FormatError::InvalidFixedArrayGeometry`] if it overflows.
fn sum<const N: usize>(terms: [u64; N]) -> Result<u64, FormatError> {
    terms
        .into_iter()
        .try_fold(0, u64::checked_add)
        .ok_or(FormatError::InvalidFixedArrayGeometry)
}

fn without_checksum(bytes: &[u8]) -> Result<&[u8], FormatError> {
    bytes
        .split_last_chunk::<{ size_of::<u32>() }>()
        .map(|(body, _checksum)| body)
        .ok_or_else(|| FormatError::Internal("a Fixed Array block ends inside its checksum".into()))
}

/// The base 2 logarithm of the number of elements in a data block page of a Fixed Array the
/// writer builds, 1024 elements.
///
/// The writer stores it in the "Page Bits" field of both the header and the version 4 data layout
/// message. libhdf5 calls this value `H5D_FARRAY_MAX_DBLK_PAGE_NELMTS_BITS`. A reader takes
/// the page size from the header.
pub(crate) const FIXED_ARRAY_PAGE_BITS: u8 = 10;

/// The layout of a Fixed Array that does not depend on its address: the element encoding, the
/// paging, and the lengths.
///
/// [`fixed_array_len`] computes the layout without building the array, for a caller that reserves
/// the span of the array before writing it.
struct FixedArrayLayout {
    encoding: ChunkElementEncoding,
    /// The base 2 logarithm of `page_size`, which the header stores.
    page_bits: u8,
    /// The number of elements in a page, past which the data block is paged.
    page_size: usize,
    /// The length of the header in bytes.
    fahd_size: usize,
    /// The length of the header and the data block in bytes.
    total_len: u64,
}

/// Returns the layout of the Fixed Array that holds `slots`.
fn fixed_array_layout(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> FixedArrayLayout {
    let os = usize::from(offset_size.get());
    let num_elements = slots.len();
    let encoding = chunk_record::chunk_element_encoding(chunk_bytes, offset_size, has_filters);

    let fahd_size = 4 + 1 + 1 + 1 + 1 + usize::from(length_size.get()) + os + 4;

    // The data block is the prefix, then either the elements and one checksum, or the page-init
    // bitmap, its checksum, and the pages with a checksum each. The element bytes are the same in
    // both.
    let fadb_prefix = 4 + 1 + 1 + os;
    let page_bits = FIXED_ARRAY_PAGE_BITS;
    let page_size = 1usize << page_bits;
    let elements = num_elements * encoding.elem_size;
    let fadb_size = if num_elements <= page_size {
        fadb_prefix + elements + 4
    } else {
        let npages = num_elements.div_ceil(page_size);
        fadb_prefix + npages.div_ceil(8) + 4 + elements + npages * 4
    };

    FixedArrayLayout {
        encoding,
        page_bits,
        page_size,
        fahd_size,
        total_len: (fahd_size + fadb_size) as u64,
    }
}

/// Returns the length in bytes of the Fixed Array [`build_fixed_array_at`] builds for `slots`,
/// without building it.
pub fn fixed_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> u64 {
    fixed_array_layout(slots, chunk_bytes, offset_size, length_size, has_filters).total_len
}

/// Builds the Fixed Array for `slots` at `fa_address`: the header, then the data block.
///
/// `chunk_bytes` is the size of a chunk before filtering, which sets the width of the chunk size
/// field of a filtered element. A slot no chunk occupies stores the undefined address. Past
/// `1 << FIXED_ARRAY_PAGE_BITS` slots the data block is paged, and every page is marked
/// initialized.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the stored size of a chunk does not fit the chunk size
/// field.
pub fn build_fixed_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    fa_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = slots.len();

    let layout = fixed_array_layout(slots, chunk_bytes, offset_size, length_size, has_filters);
    let ChunkElementEncoding {
        chunk_size_bytes,
        elem_size,
        client_id,
    } = layout.encoding;
    let fahd_total_size = layout.fahd_size;
    let fadb_address = fa_address.offset(fahd_total_size as u64);

    // Build FAHD
    let mut fahd = Vec::with_capacity(fahd_total_size);
    fahd.extend_from_slice(b"FAHD");
    fahd.push(0); // version
    fahd.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an element record is an address of at most 8 bytes, a chunk size of at most \
                  8 and a 4-byte filter mask, so its size fits the 1-byte FAHD field"
    )]
    fahd.push(elem_size as u8);

    fahd.push(layout.page_bits);

    bytes::write_length(&mut fahd, num_elements.to_u64(), length_size);
    bytes::write_offset(&mut fahd, fadb_address.get(), offset_size);

    // Checksum
    let checksum = checksum::jenkins_lookup3(&fahd);
    fahd.extend_from_slice(&checksum.to_le_bytes());

    debug_assert_eq!(fahd.len(), fahd_total_size);

    let write_element = |buf: &mut Vec<u8>, chunk: Option<&ChunkRecord>| match chunk {
        Some(chunk) => chunk_record::write_chunk_element(
            buf,
            chunk,
            offset_size,
            has_filters,
            chunk_size_bytes,
        ),
        None => {
            chunk_record::write_undefined_element(buf, offset_size, has_filters, chunk_size_bytes);
            Ok(())
        }
    };

    // Build the `FADB` prefix: signature + version + `client_id` + header address.
    let mut fadb = Vec::new();
    fadb.extend_from_slice(b"FADB");
    fadb.push(0); // version
    fadb.push(client_id);
    bytes::write_offset(&mut fadb, fa_address.get(), offset_size);

    let page_size = layout.page_size;
    if num_elements <= page_size {
        // Non-paged: elements stored directly, then a single checksum.
        for slot in 0..num_elements {
            write_element(&mut fadb, slots.at(slot))?;
        }
        let fadb_checksum = checksum::jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&fadb_checksum.to_le_bytes());
    } else {
        // Paged: the page-init bitmap and its checksum, then each page and its checksum.
        let npages = num_elements.div_ceil(page_size);
        let bitmap_size = npages.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_size];
        for page in 0..npages {
            // Page 0 is the most significant bit, as `H5VM_bit_set` sets it (`H5VMprivate.h`,
            // HDF5 2.2.0).
            bitmap[page / 8] |= 1 << (7 - (page % 8));
        }
        fadb.extend_from_slice(&bitmap);
        let prefix_checksum = checksum::jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&prefix_checksum.to_le_bytes());

        for page in 0..npages {
            let start = page * page_size;
            let end = core::cmp::min(start + page_size, num_elements);
            let mut page_buf = Vec::with_capacity((end - start) * elem_size);
            for slot in start..end {
                write_element(&mut page_buf, slots.at(slot))?;
            }
            let page_checksum = checksum::jenkins_lookup3(&page_buf);
            page_buf.extend_from_slice(&page_checksum.to_le_bytes());
            fadb.extend_from_slice(&page_buf);
        }
    }

    let mut combined = fahd;
    combined.extend_from_slice(&fadb);
    // The caller reserved the length `fixed_array_len` returns.
    debug_assert_eq!(
        combined.len() as u64,
        layout.total_len,
        "a fixed array must fill the length its layout promised"
    );
    Ok(combined)
}

// The signatures, the version, and the field widths are defined in "The Fixed Array Index" of the
// format specification, version 4.0.
const HEADER_SIGNATURE: [u8; 4] = *b"FAHD";
const DATA_BLOCK_SIGNATURE: [u8; 4] = *b"FADB";
const VERSION: u8 = 0;
const SIGNATURE_LEN: u64 = 4;
/// The version, client ID, entry size, and page bits bytes of a header.
const HEADER_BYTE_FIELDS: u64 = 4;
/// The version and client ID bytes of a data block.
const DATA_BLOCK_BYTE_FIELDS: u64 = 2;
const CHECKSUM_LEN: u64 = 4;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::checksum;

    use super::*;

    const WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    const ARRAY_AT: u64 = 0x80;

    /// The Fixed Array the writer builds at `ARRAY_AT`, in an image whose first byte is at
    /// physical address `base`.
    struct Fixture {
        base: u64,
        widths: FormatWidths,
        bytes: Vec<u8>,
    }

    impl Fixture {
        fn new(n: u64, has_filters: bool) -> Self {
            Self::at(0, WIDTHS, 8, n, has_filters)
        }

        fn at(
            base: u64,
            widths: FormatWidths,
            chunk_bytes: u64,
            n: u64,
            has_filters: bool,
        ) -> Self {
            let array = build_fixed_array_at(
                &IndexSlots::dense(&chunk_record::chunks(chunk_bytes, n, has_filters)),
                chunk_bytes,
                widths.offsets,
                widths.lengths,
                has_filters,
                StoredAddress::new(ARRAY_AT),
            )
            .unwrap();
            let mut bytes = vec![0; usize::try_from(ARRAY_AT).unwrap()];
            bytes.extend_from_slice(&array);
            Self {
                base,
                widths,
                bytes,
            }
        }

        fn locator(&self) -> FixedArrayHeaderLocator {
            FixedArrayHeaderLocator::new(
                self.widths,
                BaseAddress::new(self.base),
                StoredAddress::new(ARRAY_AT),
            )
        }

        fn image(&self) -> FileRegion<'_> {
            FileRegion::new(
                FileRange::new(PhysicalAddress::new(self.base), self.bytes.len() as u64).unwrap(),
                &self.bytes,
            )
            .unwrap()
        }

        fn header(&self) -> Result<FixedArrayHeader, FormatError> {
            let locator = self.locator();
            locator.parse(self.image().subregion(locator.parse_range()?)?)
        }

        fn data_block(&self) -> FixedArrayDataBlockRef {
            self.header().unwrap().data_block().unwrap()
        }

        fn pages(&self) -> Vec<FixedArrayPageRef> {
            let block = self.data_block();
            let region = self
                .image()
                .subregion(block.parse_range().unwrap())
                .unwrap();
            let FixedArrayDataBlock::PageBitmap(bitmap) = block.parse(region).unwrap() else {
                panic!("the data block of {block:?} is not paged");
            };
            bitmap.pages().collect::<Result<_, _>>().unwrap()
        }

        fn local(&self, at: PhysicalAddress) -> usize {
            usize::try_from(at.get() - self.base).unwrap()
        }

        fn header_at(&self) -> usize {
            usize::try_from(ARRAY_AT).unwrap()
        }

        fn data_block_at(&self) -> usize {
            self.local(self.data_block().range().start())
        }

        fn restamp_header(&mut self) {
            let at = self.header_at();
            let len = usize::try_from(header_len(self.widths)).unwrap();
            checksum::restamp(&mut self.bytes, at, len);
        }

        fn restamp_prefix(&mut self) {
            let block = self.data_block();
            let at = self.local(block.range().start());
            let len = usize::try_from(block.parse_range().unwrap().length()).unwrap();
            checksum::restamp(&mut self.bytes, at, len);
        }

        fn records(&self, chunk_byte_size: u64) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
            let image = self.image();
            let Some(block) = self.header()?.data_block() else {
                return Ok(Vec::new());
            };
            match block.parse(image.subregion(block.parse_range()?)?)? {
                FixedArrayDataBlock::Elements(elements) => {
                    elements.records(chunk_byte_size).collect()
                }
                FixedArrayDataBlock::PageBitmap(bitmap) => {
                    let mut records = Vec::new();
                    for page in bitmap.pages() {
                        let page = page?;
                        for record in page
                            .parse(image.subregion(page.parse_range())?)?
                            .records(chunk_byte_size)
                        {
                            records.push(record?);
                        }
                    }
                    Ok(records)
                }
            }
        }
    }

    #[rstest]
    #[case::first(0, 14, 8_192, 1_024)]
    #[case::last_full((1 << 29) - 1_024, 4_294_959_118, 8_192, 1 << 29)]
    #[case::last_entry((1 << 29) - 1, 4_294_967_302, 8, 1 << 29)]
    fn unpaged_windows_keep_full_width_geometry(
        #[case] first_slot: u64,
        #[case] offset: u64,
        #[case] length: u64,
        #[case] end_slot: u64,
    ) {
        let mut fixture = Fixture::new(5, false);
        let at = fixture.header_at();
        fixture.bytes[at + 7] = 29;
        fixture.bytes[at + 8..at + 16].copy_from_slice(&(1u64 << 29).to_le_bytes());
        fixture.restamp_header();
        let block = fixture.data_block();
        assert_eq!(block.range().length(), 4_294_967_314);
        let prefix = block.unpaged_prefix_range().unwrap().unwrap();
        let unpaged = block
            .parse_unpaged_prefix(fixture.image().subregion(prefix).unwrap())
            .unwrap();
        let window = unpaged
            .window(first_slot, NonZeroUsize::new(1_024).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            window.parse_range().start().get(),
            block.range().start().get() + offset
        );
        assert_eq!(window.parse_range().length(), length);
        assert_eq!(window.end_slot(), end_slot);
        assert_eq!(unpaged.checksummed_range().unwrap().length(), 4_294_967_310);
        assert_eq!(
            unpaged.checksum_range().unwrap().start().get(),
            block.range().start().get() + 4_294_967_310
        );
        assert_eq!(unpaged.window(1 << 29, NonZeroUsize::MIN).unwrap(), None);
    }

    #[test]
    fn unpaged_window_parsing_preserves_large_slots() {
        let geometry = Geometry::new(
            ElementEncoding::new(
                OffsetWidth::Eight,
                chunk_record::ArrayClient::Chunks.id(),
                8,
            )
            .unwrap(),
            (1 << 32) + 4,
            33,
        )
        .unwrap();
        let block = FixedArrayDataBlockRef::new(
            BaseAddress::ZERO,
            PhysicalAddress::new(0),
            PhysicalAddress::new(28),
            geometry,
        )
        .unwrap();
        let mut prefix = Vec::from(DATA_BLOCK_SIGNATURE);
        prefix.extend_from_slice(&[VERSION, chunk_record::ArrayClient::Chunks.id()]);
        prefix.extend_from_slice(&0u64.to_le_bytes());
        let range = block.unpaged_prefix_range().unwrap().unwrap();
        let unpaged = block
            .parse_unpaged_prefix(FileRegion::new(range, &prefix).unwrap())
            .unwrap();
        let window = unpaged
            .window(1 << 32, NonZeroUsize::new(2).unwrap())
            .unwrap()
            .unwrap();
        let mut bytes = Vec::from(123u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            window
                .parse(FileRegion::new(window.parse_range(), &bytes).unwrap())
                .unwrap()
                .records(8)
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![(
                1 << 32,
                ChunkRecord {
                    address: StoredAddress::new(123),
                    stored_size: 8,
                    filter_mask: 0
                }
            )]
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
        #[values(0, 5, 1_024, 1_025, 3_000)] n: u64,
        #[values(false, true)] has_filters: bool,
        #[values(8, 300, 100_000, 1 << 32)] chunk_bytes: u64,
    ) {
        let fixture = Fixture::at(0, widths, chunk_bytes, n, has_filters);
        let read = fixture.records(chunk_bytes).unwrap();
        assert_eq!(
            read,
            chunk_record::slotted(chunk_record::chunks(chunk_bytes, n, has_filters))
        );
    }

    #[rstest]
    fn the_header_and_data_block_ranges_tile_the_array(
        #[values(0, 1, 5, 1_024, 1_025, 3_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let fixture = Fixture::new(n, has_filters);
        let header = fixture.header().unwrap();
        let block = header.data_block().unwrap();
        assert_eq!(header.range().start().get(), ARRAY_AT);
        assert_eq!(header.range().end(), block.range().start());
        assert_eq!(block.range().end().get(), fixture.bytes.len() as u64);
    }

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
                (range.start().get(), range.length(), page.first_slot)
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
    fn a_slot_that_stores_the_undefined_address_has_no_record() {
        let chunks = chunk_record::chunks(8, 3, true);
        let slots = IndexSlots::new(&chunks, &[0, 2, 4], 5).unwrap();
        let array = build_fixed_array_at(
            &slots,
            8,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            true,
            StoredAddress::new(ARRAY_AT),
        )
        .unwrap();
        let mut fixture = Fixture::new(0, false);
        fixture.bytes.truncate(fixture.header_at());
        fixture.bytes.extend_from_slice(&array);
        assert_eq!(
            fixture.records(8).unwrap(),
            vec![(0, chunks[0]), (2, chunks[1]), (4, chunks[2])]
        );
    }

    #[test]
    fn an_undefined_data_block_address_leaves_the_array_without_a_data_block() {
        let mut fixture = Fixture::new(5, false);
        let at = fixture.header_at() + 8 + 8;
        fixture.bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        fixture.restamp_header();
        let header = fixture.header().unwrap();
        assert_eq!(header.data_block(), None);
        assert_eq!(header.range().length(), header_len(WIDTHS));
    }

    #[rstest]
    #[case::signature(0, b'X', FormatError::InvalidFixedArraySignature)]
    #[case::version(4, 1, FormatError::InvalidFixedArrayVersion(1))]
    #[case::client(5, 2, FormatError::InvalidFixedArrayClient(2))]
    #[case::page_bits(7, 64, FormatError::InvalidFixedArrayGeometry)]
    #[case::filtered_entry_too_small(6, 8 + 4 - 1, FormatError::InvalidFixedArrayGeometry)]
    #[case::filtered_entry_too_large(6, 8 + 4 + 9, FormatError::InvalidFixedArrayGeometry)]
    fn a_malformed_header_is_rejected(
        #[case] field: usize,
        #[case] value: u8,
        #[case] expected: FormatError,
    ) {
        let mut fixture = Fixture::new(5, true);
        let at = fixture.header_at() + field;
        fixture.bytes[at] = value;
        fixture.restamp_header();
        assert_eq!(fixture.header().unwrap_err(), expected);
    }

    #[test]
    fn an_entry_count_whose_data_block_overflows_is_rejected() {
        let mut fixture = Fixture::new(5, false);
        let at = fixture.header_at() + 8;
        fixture.bytes[at..at + 8].copy_from_slice(&(u64::MAX / 4).to_le_bytes());
        fixture.restamp_header();
        assert_eq!(
            fixture.header().unwrap_err(),
            FormatError::InvalidFixedArrayGeometry
        );
    }

    #[rstest]
    #[case::signature(0, b"X", FormatError::InvalidFixedArraySignature)]
    #[case::version(4, &[1], FormatError::InvalidFixedArrayVersion(1))]
    #[case::client(5, &[1], FormatError::InvalidFixedArrayBlockClient { expected: 0, actual: 1 })]
    #[case::header_address(6, &[0x81], FormatError::InvalidFixedArrayBlockOwner { expected: ARRAY_AT, actual: 0x81 })]
    fn a_malformed_data_block_prefix_is_rejected(
        #[case] field: usize,
        #[case] value: &[u8],
        #[case] expected: FormatError,
        #[values(5, 3_000)] n: u64,
    ) {
        let mut fixture = Fixture::new(n, false);
        let at = fixture.data_block_at() + field;
        fixture.bytes[at..at + value.len()].copy_from_slice(value);
        fixture.restamp_prefix();
        let block = fixture.data_block();
        let region = fixture
            .image()
            .subregion(block.parse_range().unwrap())
            .unwrap();
        assert_eq!(block.parse(region).unwrap_err(), expected);
    }

    #[rstest]
    #[case::plain(0)]
    #[case::userblock(512)]
    #[case::above_u32(1 << 40)]
    fn an_array_reads_at_its_physical_position(#[case] base: u64, #[values(5, 3_000)] n: u64) {
        let fixture = Fixture::at(base, WIDTHS, 8, n, true);
        let header = fixture.header().unwrap();
        assert_eq!(header.range().start().get(), base + ARRAY_AT);
        assert_eq!(
            fixture.records(8).unwrap(),
            chunk_record::slotted(chunk_record::chunks(8, n, true))
        );
    }

    #[test]
    fn a_header_address_at_the_physical_header_position_is_rejected() {
        let base = 1 << 40;
        let mut fixture = Fixture::at(base, WIDTHS, 8, 5, false);
        let at = fixture.data_block_at() + 6;
        fixture.bytes[at..at + 8].copy_from_slice(&(base + ARRAY_AT).to_le_bytes());
        fixture.restamp_prefix();
        let block = fixture.data_block();
        let region = fixture
            .image()
            .subregion(block.parse_range().unwrap())
            .unwrap();
        assert_eq!(
            block.parse(region).unwrap_err(),
            FormatError::InvalidFixedArrayBlockOwner {
                expected: base + ARRAY_AT,
                actual: 2 * base + ARRAY_AT,
            }
        );
    }

    #[rstest]
    #[case::header(Truncated::Header)]
    #[case::prefix(Truncated::Prefix)]
    #[case::page(Truncated::Page)]
    fn a_region_short_of_its_range_is_rejected(#[case] truncated: Truncated) {
        let fixture = Fixture::new(3_000, false);
        let short = |range: FileRange| {
            fixture
                .image()
                .subregion(FileRange::new(range.start(), range.length() - 1).unwrap())
                .unwrap()
        };
        let (range, err) = match truncated {
            Truncated::Header => {
                let locator = fixture.locator();
                let range = locator.parse_range().unwrap();
                (range, locator.parse(short(range)).unwrap_err())
            }
            Truncated::Prefix => {
                let block = fixture.data_block();
                let range = block.parse_range().unwrap();
                (range, block.parse(short(range)).unwrap_err())
            }
            Truncated::Page => {
                let page = *fixture.pages().last().unwrap();
                let range = page.parse_range();
                (range, page.parse(short(range)).unwrap_err())
            }
        };
        assert_eq!(
            err,
            FormatError::InsufficientRegion {
                required_start: range.start().get(),
                required_length: range.length(),
                provided_start: range.start().get(),
                provided_length: range.length() - 1,
            }
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Truncated {
        Header,
        Prefix,
        Page,
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    #[case::header(Corrupted::Header)]
    #[case::unpaged_data_block(Corrupted::DataBlock)]
    #[case::prefix(Corrupted::Prefix)]
    #[case::first_page(Corrupted::FirstPage)]
    #[case::last_page(Corrupted::LastPage)]
    fn a_corrupted_checksum_is_rejected(
        #[case] corrupted: Corrupted,
        #[values(false, true)] has_filters: bool,
    ) {
        let n = if matches!(corrupted, Corrupted::DataBlock) {
            5
        } else {
            3_000
        };
        let mut fixture = Fixture::new(n, has_filters);
        let range = match corrupted {
            Corrupted::Header => fixture.header().unwrap().range(),
            Corrupted::DataBlock | Corrupted::Prefix => fixture.data_block().parse_range().unwrap(),
            Corrupted::FirstPage => fixture.pages()[0].parse_range(),
            Corrupted::LastPage => fixture.pages().last().unwrap().parse_range(),
        };
        let start = fixture.local(range.start());
        let end = fixture.local(range.end());
        fixture.bytes[end - 1] ^= 0x01;
        let stored = u32::from_le_bytes(fixture.bytes[end - 4..end].try_into().unwrap());
        let computed = crate::checksum::jenkins_lookup3(&fixture.bytes[start..end - 4]);
        assert_eq!(
            fixture.records(8).unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            }
        );
    }

    #[cfg(feature = "checksum")]
    #[derive(Clone, Copy, Debug)]
    enum Corrupted {
        Header,
        DataBlock,
        Prefix,
        FirstPage,
        LastPage,
    }

    #[test]
    fn build_fixed_array_valid_structure() {
        let chunks = vec![
            ChunkRecord {
                address: StoredAddress::new(0x1000),
                stored_size: 160,
                filter_mask: 0,
            },
            ChunkRecord {
                address: StoredAddress::new(0x10A0),
                stored_size: 160,
                filter_mask: 0,
            },
        ];
        let fa = build_fixed_array_at(
            &IndexSlots::dense(&chunks),
            160,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(0x2000),
        )
        .unwrap();
        // Should start with FAHD
        assert_eq!(&fa[0..4], b"FAHD");
        // FAHD size = 4+1+1+1+1+8+8+4 = 28
        // FADB starts at offset 28
        assert_eq!(&fa[28..32], b"FADB");
    }

    // Sweeps every count across the page size, where the data block becomes paged, and counts
    // that leave a partial last page.
    #[test]
    fn fixed_array_len_matches_what_it_builds() {
        fn check(
            n: u64,
            chunk_bytes: u64,
            offset_size: OffsetWidth,
            length_size: LengthWidth,
            has_filters: bool,
        ) {
            let chunks: Vec<ChunkRecord> = (0..n)
                .map(|i| ChunkRecord {
                    address: StoredAddress::new(0x1000 + i * 8),
                    stored_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let planned = fixed_array_len(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
            );
            let built = build_fixed_array_at(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
                StoredAddress::new(0x10_0000),
            )
            .unwrap();
            assert_eq!(
                planned,
                built.len() as u64,
                "planned length must match the emitted array at n={n}, \
                 chunk_bytes={chunk_bytes}, offset_size={offset_size:?}, \
                 has_filters={has_filters}"
            );
        }

        for (offset_size, length_size) in [
            (OffsetWidth::Eight, LengthWidth::Eight),
            (OffsetWidth::Four, LengthWidth::Four),
        ] {
            for &has_filters in &[false, true] {
                // The array is paged past `1 << FIXED_ARRAY_PAGE_BITS` elements.
                for n in 0..=1_100u64 {
                    check(n, 8, offset_size, length_size, has_filters);
                }
                // Several whole pages, and a count that leaves a partial one.
                for &n in &[4_096u64, 5_000, 100_000] {
                    check(n, 8, offset_size, length_size, has_filters);
                }
            }
        }

        // Each chunk size selects a different chunk size field width.
        for &chunk_bytes in &chunk_record::CHUNK_BYTES {
            for &n in &[1u64, 1_024, 1_025, 5_000] {
                check(n, chunk_bytes, OffsetWidth::Eight, LengthWidth::Eight, true);
            }
        }
    }
}
