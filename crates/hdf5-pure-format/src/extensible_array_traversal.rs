//! Traverses Extensible Array chunk indexes from a buffer or a metadata source.
//!
//! Logical element counts and range endpoints are file-width integers. A [`Region`] holds a file
//! address and a read length that fits a [`usize`].

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::num::NonZeroU64;

use super::ExtensibleArrayHeader;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::chunk_record::ChunkRecord;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;
use crate::width::OffsetWidth;

/// An Extensible Array reader with parsed block geometry and address width.
///
/// Direct data blocks are unpaged. Each paged data block contains a whole, nonzero number of
/// pages, and each block and page has a nonzero element count.
pub(super) struct Reader<'a> {
    header: &'a ExtensibleArrayHeader,
    offset_width: OffsetWidth,
    element_stride: usize,
    chunk_byte_size: u64,
    geometry: Geometry,
}

impl<'a> Reader<'a> {
    /// Parses the geometry used to traverse the blocks addressed by the index block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and
    /// [`FormatError::InvalidChunkGeometry`] if the minimum data block element count is zero,
    /// a data block contains a partial page, or a direct data block requires paging.
    pub(super) fn parse(
        header: &'a ExtensibleArrayHeader,
        offset_size: u8,
        chunk_byte_size: u64,
    ) -> Result<Self, FormatError> {
        let offset_width = OffsetWidth::try_from(offset_size)?;
        let element_stride = super::ea_elem_stride(header, offset_size);
        Ok(Self {
            header,
            offset_width,
            element_stride,
            chunk_byte_size,
            geometry: Geometry::parse(header)?,
        })
    }

    /// Calls `visit` for each occupied slot before the header's "Max Index Set".
    ///
    /// Unallocated blocks occupy their logical slots without requiring a read. Checksums cover
    /// whole blocks or pages, including elements past "Max Index Set".
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChunkedReadError`] if a block signature or element encoding is
    /// invalid, [`FormatError::UnexpectedEof`] if a block or bitmap is truncated,
    /// [`FormatError::ChecksumMismatch`] if a checksum does not match,
    /// [`FormatError::OffsetOverflow`] if a byte extent overflows a [`u64`], and
    /// [`FormatError::ValueTooLargeForPlatform`] if a read length or slice offset does not fit
    /// a [`usize`]. Also returns errors from `source` and `visit`.
    pub(super) fn read(
        &self,
        source: &impl Source,
        visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        let Self {
            header,
            offset_width,
            element_stride,
            chunk_byte_size: _,
            geometry,
        } = self;
        let os = usize::from(offset_width.get());
        let prefix = (4 + 1 + 1 + os) as u64;
        let inline_size = element_extent(u64::from(header.idx_blk_elmts), *element_stride, prefix)?;
        let address_count = geometry.direct_addresses + geometry.indirect().len() as u64;
        let index_size = element_extent(address_count, os, inline_size + 4)?;
        let region = Region::parse(header.index_block_address.get(), index_size)?;
        let block = source.read(region)?;
        self.signature(&block, b"EAIB")?;
        let block = BlockData::parse(&block, region.size)?;

        let mut remaining = Elements::published(header.max_idx_set);
        let inline = remaining.take(u64::from(header.idx_blk_elmts));
        self.elements(block.range(prefix, inline_size)?, inline, visit)?;
        if remaining.is_empty() {
            return Ok(());
        }
        let mut addresses = block.from(inline_size)?.chunks_exact(os);
        for geometry in geometry.direct() {
            for _ in 0..geometry.blocks.get() {
                let address = self.address(addresses.next())?;
                let elements = remaining.take(geometry.elements.get());
                if !address.is_undefined(offset_width.get()) && !elements.is_empty() {
                    self.data_block(
                        source,
                        address,
                        Block::Unpaged(geometry.elements),
                        elements,
                        visit,
                    )?;
                }
                if remaining.is_empty() {
                    return Ok(());
                }
            }
        }
        for geometry in geometry.indirect() {
            let address = self.address(addresses.next())?;
            let elements = remaining.take_wide(geometry.elements());
            if !address.is_undefined(offset_width.get()) && !elements.is_empty() {
                self.super_block(source, address, *geometry, elements, visit)?;
            }
            if remaining.is_empty() {
                break;
            }
        }
        Ok(())
    }

    fn super_block(
        &self,
        source: &impl Source,
        address: StoredAddress,
        geometry: SuperBlock,
        mut elements: Elements,
        visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        let os = usize::from(self.offset_width.get());
        let prefix = (4 + 1 + 1 + os + self.header.block_offset_width().bytes()) as u64;
        let bitmap_size = geometry.bitmap_size()?;
        let addresses_offset =
            prefix
                .checked_add(bitmap_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: prefix,
                    length: bitmap_size,
                })?;
        let size = element_extent(geometry.blocks.get(), os, addresses_offset + 4)?;
        let region = Region::parse(address.get(), size)?;
        let block = source.read(region)?;
        self.signature(&block, b"EASB")?;
        let block = BlockData::parse(&block, region.size)?;
        let bitmap = block.range(prefix, addresses_offset)?;
        for (local, bytes) in block.from(addresses_offset)?.chunks_exact(os).enumerate() {
            if elements.is_empty() {
                break;
            }
            let address = self.address(Some(bytes))?;
            let range = elements.take(geometry.data.elements());
            if address.is_undefined(self.offset_width.get()) {
                continue;
            }
            let block = match geometry.data {
                DataBlock::Paged {
                    page_elements,
                    pages,
                } => Block::Paged(Pages::parse(
                    self.element_stride,
                    page_elements,
                    PageBitmap::parse(bitmap, local.to_u64(), pages)?,
                )?),
                DataBlock::Unpaged { elements } => Block::Unpaged(elements),
            };
            self.data_block(source, address, block, range, visit)?;
        }
        Ok(())
    }

    fn data_block(
        &self,
        source: &impl Source,
        address: StoredAddress,
        block: Block<'_>,
        elements: Elements,
        visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        let prefix = (4
            + 1
            + 1
            + usize::from(self.offset_width.get())
            + self.header.block_offset_width().bytes()) as u64;
        match block {
            Block::Unpaged(count) => {
                let size = element_extent(count.get(), self.element_stride, prefix + 4)?;
                let region = Region::parse(address.get(), size)?;
                let block = source.read(region)?;
                self.signature(&block, b"EADB")?;
                let block = BlockData::parse(&block, region.size)?;
                self.elements(block.from(prefix)?, elements, visit)
            }
            Block::Paged(pages) => {
                let prefix = Region::parse(address.get(), prefix + 4)?;
                match pages.read_plan(prefix)? {
                    PageRead::Contiguous(region) => {
                        let block = source.read(region)?;
                        self.signature(&block, b"EADB")?;
                        BlockData::parse(&block, prefix.size)?;
                        let buffer = Buffer(&block);
                        let prefix = Region::parse(0, prefix.size.to_u64())?;
                        self.pages(&buffer, prefix, pages, elements, visit)
                    }
                    PageRead::Sparse(prefix) => {
                        let block = source.read(prefix)?;
                        self.signature(&block, b"EADB")?;
                        BlockData::parse(&block, prefix.size)?;
                        self.pages(source, prefix, pages, elements, visit)
                    }
                }
            }
        }
    }

    fn pages(
        &self,
        source: &impl Source,
        prefix: Region,
        pages: Pages<'_>,
        mut elements: Elements,
        visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        let Pages {
            elements: page_elements,
            bitmap,
            stride,
        } = pages;
        for (page, initialized) in bitmap.iter() {
            if elements.is_empty() {
                break;
            }
            let range = elements.take(page_elements.get());
            if !initialized {
                continue;
            }
            let region = prefix.following(page, stride)?;
            let block = source.read(region)?;
            let block = BlockData::parse(&block, region.size)?;
            self.elements(block.from(0)?, range, visit)?;
        }
        Ok(())
    }

    fn elements(
        &self,
        block: &[u8],
        elements: Elements,
        visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        let mut pos = 0;
        for index in elements.start..elements.end {
            let (record, consumed) = super::read_element(
                block,
                pos,
                self.header.client_id,
                self.header.element_size,
                self.offset_width.get(),
                self.chunk_byte_size,
            )?;
            if let Some(record) = record {
                visit(index, record)?;
            }
            pos += consumed;
        }
        Ok(())
    }

    fn address(&self, bytes: Option<&[u8]>) -> Result<StoredAddress, FormatError> {
        let bytes = bytes.ok_or(FormatError::Internal(
            "Extensible Array address count differs from geometry".into(),
        ))?;
        Ok(StoredAddress::new(bytes::read_offset(
            bytes,
            0,
            self.offset_width.get(),
        )?))
    }

    fn signature(&self, bytes: &[u8], expected: &[u8; 4]) -> Result<(), FormatError> {
        if bytes.get(..4) != Some(expected.as_slice()) {
            return Err(FormatError::ChunkedReadError(
                "invalid Extensible Array block signature".into(),
            ));
        }
        Ok(())
    }
}

/// Nonzero block and element counts, grouped by their addresses in the index block.
struct Geometry {
    super_blocks: Vec<SuperBlock>,
    direct_blocks: Vec<DirectBlock>,
    /// The sum of the direct data block counts.
    direct_addresses: u64,
}

impl Geometry {
    /// Parses nonzero block counts for direct and indirect data blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if the minimum data block element count
    /// is zero, a data block contains a partial page, or a direct data block requires paging.
    fn parse(header: &ExtensibleArrayHeader) -> Result<Self, FormatError> {
        let minimum = NonZeroU64::new(u64::from(header.min_dblk_nelmts)).ok_or(
            FormatError::InvalidChunkGeometry(
                "Extensible Array minimum data block element count is zero",
            ),
        )?;
        let count = u32::from(header.max_index_bits().get())
            .saturating_sub(minimum.get().trailing_zeros())
            + 1;
        let first_indirect = usize::from(header.super_blk_min_data_ptrs).min(count.to_usize()?);
        let mut super_blocks = Vec::with_capacity(count.to_usize()?);
        let mut direct_addresses = 0;
        // H5EA__hdr_init in H5EAhdr.c, HDF5 2.2.0, doubles the element count at odd
        // super block indexes and the data block count at positive even indexes.
        for index in 0..count {
            let blocks = nonzero(1u64 << (index / 2))?;
            let elements = nonzero(minimum.get() << index.div_ceil(2))?;
            let data = DataBlock::parse(elements, header.page_nelmts())?;
            super_blocks.push(SuperBlock { blocks, data });
            if (index.to_usize()?) < first_indirect {
                direct_addresses += blocks.get();
            }
        }
        let direct_blocks = super_blocks
            .drain(..first_indirect)
            .map(|super_block| {
                let DataBlock::Unpaged { elements } = super_block.data else {
                    return Err(FormatError::InvalidChunkGeometry(
                        "Extensible Array direct data block requires a page bitmap",
                    ));
                };
                Ok(DirectBlock {
                    blocks: super_block.blocks,
                    elements,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            super_blocks,
            direct_blocks,
            direct_addresses,
        })
    }

    fn direct(&self) -> &[DirectBlock] {
        &self.direct_blocks
    }

    fn indirect(&self) -> &[SuperBlock] {
        &self.super_blocks
    }
}

/// A group of equally sized, unpaged data blocks addressed by the index block.
struct DirectBlock {
    blocks: NonZeroU64,
    /// The number of elements in each data block.
    elements: NonZeroU64,
}

/// A data block's element count or its parsed page layout and initialization bitmap.
enum Block<'a> {
    Paged(Pages<'a>),
    Unpaged(NonZeroU64),
}

/// The page element count, initialization bitmap, and byte stride of a paged data block.
struct Pages<'a> {
    elements: NonZeroU64,
    bitmap: PageBitmap<'a>,
    /// The length in bytes of a page, including its checksum.
    stride: u64,
}

impl<'a> Pages<'a> {
    /// Parses the page stride, including the trailing checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the length of a page and its checksum
    /// overflows a [`u64`].
    fn parse(
        element_stride: usize,
        elements: NonZeroU64,
        bitmap: PageBitmap<'a>,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            elements,
            bitmap,
            stride: element_extent(elements.get(), element_stride, 4)?,
        })
    }

    /// Selects one read through the last initialized page when its length fits a [`usize`].
    ///
    /// Otherwise, the reader reads the prefix and initialized pages separately.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the extent through the last initialized page
    /// or its end address overflows a [`u64`].
    fn read_plan(&self, prefix: Region) -> Result<PageRead, FormatError> {
        let Self {
            elements: _,
            bitmap,
            stride,
        } = self;
        let initialized_pages = bitmap
            .iter()
            .filter_map(|(page, initialized)| initialized.then_some(page + 1))
            .last()
            .unwrap_or(0);
        let size = initialized_pages
            .checked_mul(*stride)
            .and_then(|size| size.checked_add(prefix.size.to_u64()))
            .ok_or(FormatError::OffsetOverflow {
                offset: initialized_pages,
                length: *stride,
            })?;
        Ok(match Region::parse(prefix.address, size) {
            Ok(region) => PageRead::Contiguous(region),
            Err(FormatError::ValueTooLargeForPlatform { .. }) => PageRead::Sparse(prefix),
            Err(error) => return Err(error),
        })
    }
}

/// The file regions to read for a paged data block.
enum PageRead {
    /// One region containing the prefix and all pages through the last initialized page.
    Contiguous(Region),
    /// The prefix region, followed by separate reads of initialized pages.
    Sparse(Region),
}

/// A nonzero number of equally sized data blocks addressed by a super block.
#[derive(Clone, Copy)]
struct SuperBlock {
    blocks: NonZeroU64,
    data: DataBlock,
}

impl SuperBlock {
    fn elements(self) -> u128 {
        u128::from(self.blocks.get()) * u128::from(self.data.elements())
    }

    fn bitmap_size(self) -> Result<u64, FormatError> {
        match self.data {
            DataBlock::Unpaged { .. } => Ok(0),
            DataBlock::Paged { pages, .. } => self
                .blocks
                .get()
                .checked_mul(pages.get().div_ceil(8))
                .ok_or(FormatError::OffsetOverflow {
                    offset: self.blocks.get(),
                    length: pages.get().div_ceil(8),
                }),
        }
    }
}

/// A nonzero element count, stored unpaged or divided into whole pages.
#[derive(Clone, Copy)]
enum DataBlock {
    Paged {
        page_elements: NonZeroU64,
        pages: NonZeroU64,
    },
    Unpaged {
        elements: NonZeroU64,
    },
}

impl DataBlock {
    /// Parses an unpaged element count or a nonzero number of whole pages.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if `elements` exceeds `page_elements`
    /// and is not a multiple of `page_elements`.
    fn parse(elements: NonZeroU64, page_elements: NonZeroU64) -> Result<Self, FormatError> {
        if elements > page_elements && !elements.get().is_multiple_of(page_elements.get()) {
            return Err(FormatError::InvalidChunkGeometry(
                "Extensible Array data block does not contain whole pages",
            ));
        }
        Ok(if elements <= page_elements {
            Self::Unpaged { elements }
        } else {
            Self::Paged {
                page_elements,
                pages: nonzero(elements.get() / page_elements.get())?,
            }
        })
    }

    fn elements(self) -> u64 {
        match self {
            Self::Paged {
                page_elements,
                pages,
            } => page_elements.get() * pages.get(),
            Self::Unpaged { elements } => elements.get(),
        }
    }
}

/// A half-open logical element range with `start <= end`.
///
/// Both endpoints fit a [`u64`]. Taking elements advances `start` by at most the remaining count.
#[derive(Clone, Copy)]
struct Elements {
    start: u64,
    end: u64,
}

impl Elements {
    fn published(end: u64) -> Self {
        Self { start: 0, end }
    }

    /// Removes and returns the first `count` elements, bounded by the remaining range.
    fn take(&mut self, count: u64) -> Self {
        let start = self.start;
        self.start += count.min(self.end - self.start);
        Self {
            start,
            end: self.start,
        }
    }

    /// Removes a prefix whose requested count may exceed the range of a [`u64`].
    fn take_wide(&mut self, count: u128) -> Self {
        // The minimum is bounded by the remaining u64 element count.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the count is bounded by the remaining u64 element count"
        )]
        let count = count.min(u128::from(self.end - self.start)) as u64;
        self.take(count)
    }

    fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// The initialization bits for one data block, with every page's bit present.
///
/// Bits are read from most significant to least significant within each byte, as
/// `H5VM_bit_get` reads them (`H5VMprivate.h`, HDF5 2.2.0).
struct PageBitmap<'a> {
    bytes: &'a [u8],
    /// The first page's bit offset within the first byte, in `0..8`.
    first_bit: usize,
    pages: NonZeroU64,
}

impl<'a> PageBitmap<'a> {
    /// Parses the initialization bits for data block `local` of a super block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the bit range overflows a [`u64`],
    /// [`FormatError::ValueTooLargeForPlatform`] if a byte offset does not fit a [`usize`], and
    /// [`FormatError::UnexpectedEof`] if the bitmap does not contain every page's bit.
    fn parse(bytes: &'a [u8], local: u64, pages: NonZeroU64) -> Result<Self, FormatError> {
        let start = local
            .checked_mul(pages.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: local,
                length: pages.get(),
            })?;
        let end = start
            .checked_add(pages.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: start,
                length: pages.get(),
            })?;
        let first_byte = (start / 8).to_usize()?;
        let end_byte = end.div_ceil(8).to_usize()?;
        let bytes = bytes
            .get(first_byte..end_byte)
            .ok_or(FormatError::UnexpectedEof {
                expected: end_byte,
                available: bytes.len(),
            })?;
        Ok(Self {
            bytes,
            first_bit: (start % 8).to_usize()?,
            pages,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (u64, bool)> + Clone + '_ {
        let Self {
            bytes,
            first_bit,
            pages,
        } = self;
        let bits = bytes
            .iter()
            .flat_map(|byte| (0..8).map(move |bit| byte & (0x80 >> bit) != 0))
            .skip(*first_bit);
        (0..pages.get()).zip(bits)
    }
}

/// The bytes preceding a block or page's trailing checksum.
///
/// Parsing verifies the checksum when the `checksum` feature is enabled.
struct BlockData<'a> {
    bytes: &'a [u8],
}

impl<'a> BlockData<'a> {
    /// Parses the first `size` bytes as a block or page and excludes its trailing checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `size` exceeds the available bytes or is less
    /// than 4, and [`FormatError::ChecksumMismatch`] if the `checksum` feature is enabled and
    /// the checksum does not match.
    fn parse(bytes: &'a [u8], size: usize) -> Result<Self, FormatError> {
        let bytes = bytes.get(..size).ok_or(FormatError::UnexpectedEof {
            expected: size,
            available: bytes.len(),
        })?;
        checksum::verify_trailing(bytes)?;
        let end = size.checked_sub(4).ok_or(FormatError::UnexpectedEof {
            expected: 4,
            available: size,
        })?;
        let bytes = bytes.get(..end).ok_or(FormatError::UnexpectedEof {
            expected: end,
            available: bytes.len(),
        })?;
        Ok(Self { bytes })
    }

    fn range(&self, start: u64, end: u64) -> Result<&'a [u8], FormatError> {
        let start = start.to_usize()?;
        let end = end.to_usize()?;
        self.bytes
            .get(start..end)
            .ok_or(FormatError::UnexpectedEof {
                expected: end,
                available: self.bytes.len(),
            })
    }

    fn from(&self, start: u64) -> Result<&'a [u8], FormatError> {
        let start = start.to_usize()?;
        self.bytes.get(start..).ok_or(FormatError::UnexpectedEof {
            expected: start,
            available: self.bytes.len(),
        })
    }
}

/// A file byte range with a length that fits a [`usize`] and an end address that fits a [`u64`].
#[derive(Clone, Copy)]
pub(super) struct Region {
    address: u64,
    size: usize,
}

impl Region {
    /// Parses a byte range whose length can be used for a metadata read.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the end address overflows a [`u64`], and
    /// [`FormatError::ValueTooLargeForPlatform`] if `size` does not fit a [`usize`].
    fn parse(address: u64, size: u64) -> Result<Self, FormatError> {
        address
            .checked_add(size)
            .ok_or(FormatError::OffsetOverflow {
                offset: address,
                length: size,
            })?;
        Ok(Self {
            address,
            size: size.to_usize()?,
        })
    }

    /// Returns the `index`th `stride`-byte region following this prefix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the offset or end address overflows a [`u64`],
    /// and [`FormatError::ValueTooLargeForPlatform`] if `stride` does not fit a [`usize`].
    fn following(self, index: u64, stride: u64) -> Result<Self, FormatError> {
        let offset = index
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(self.size.to_u64()))
            .ok_or(FormatError::OffsetOverflow {
                offset: index,
                length: stride,
            })?;
        let address = self
            .address
            .checked_add(offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: self.address,
                length: offset,
            })?;
        Self::parse(address, stride)
    }
}

/// Reads parsed file regions for an Extensible Array traversal.
pub(super) trait Source {
    /// Reads the bytes at the region's file address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the region extends past the available bytes,
    /// [`FormatError::ValueTooLargeForPlatform`] if a buffer offset does not fit a [`usize`],
    /// and errors from the metadata source.
    fn read(&self, region: Region) -> Result<Cow<'_, [u8]>, FormatError>;
}

/// A traversal source that borrows bytes from a whole-file buffer.
pub(super) struct Buffer<'a>(pub(super) &'a [u8]);

impl Source for Buffer<'_> {
    fn read(&self, region: Region) -> Result<Cow<'_, [u8]>, FormatError> {
        let start = region.address.to_usize()?;
        let bytes = start
            .checked_add(region.size)
            .and_then(|end| self.0.get(start..end))
            .ok_or(FormatError::UnexpectedEof {
                expected: start.saturating_add(region.size),
                available: self.0.len(),
            })?;
        Ok(Cow::Borrowed(bytes))
    }
}

/// A traversal source that reads metadata at file addresses without narrowing them.
pub(super) struct Streaming<'a, S: ?Sized>(pub(super) &'a S);

impl<S: MetadataSource + ?Sized> Source for Streaming<'_, S> {
    fn read(&self, region: Region) -> Result<Cow<'_, [u8]>, FormatError> {
        Ok(Cow::Owned(
            self.0.read_metadata_at(region.address, region.size)?,
        ))
    }
}

/// Returns the byte length of `elements` records and a fixed prefix.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the length overflows a [`u64`].
pub(super) fn element_extent(
    elements: u64,
    stride: usize,
    prefix: u64,
) -> Result<u64, FormatError> {
    elements
        .checked_mul(stride.to_u64())
        .and_then(|size| size.checked_add(prefix))
        .ok_or(FormatError::OffsetOverflow {
            offset: elements,
            length: stride.to_u64(),
        })
}

fn nonzero(value: u64) -> Result<NonZeroU64, FormatError> {
    NonZeroU64::new(value).ok_or(FormatError::InvalidChunkGeometry(
        "Extensible Array block geometry is zero",
    ))
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;

    use rstest::rstest;
    use test_util::checksum;

    use super::*;

    #[rstest]
    #[case::empty(&[], 0, 1, 1)]
    #[case::later_block(&[0xff], 1, 8, 2)]
    #[case::straddles_byte(&[0xff], 1, 5, 2)]
    fn a_page_bitmap_requires_every_declared_bit(
        #[case] bytes: &[u8],
        #[case] local: u64,
        #[case] pages: u64,
        #[case] expected: usize,
    ) {
        let Err(error) = PageBitmap::parse(bytes, local, NonZeroU64::new(pages).unwrap()) else {
            panic!("incomplete bitmap parsed");
        };
        assert_eq!(
            error,
            FormatError::UnexpectedEof {
                expected,
                available: bytes.len()
            }
        );
    }

    #[test]
    fn a_page_bitmap_iterates_only_its_block_bits() {
        let bitmap = PageBitmap::parse(&[0x05, 0x80], 1, NonZeroU64::new(5).unwrap()).unwrap();
        assert_eq!(
            bitmap.iter().collect::<Vec<_>>(),
            vec![(0, true), (1, false), (2, true), (3, true), (4, false)]
        );
    }

    #[derive(Clone, Copy)]
    enum Backend {
        Buffer,
        SparseStream,
        Stream,
    }

    #[rstest]
    #[case::buffered(Backend::Buffer)]
    #[case::streamed(Backend::Stream)]
    #[case::sparse_streamed(Backend::SparseStream)]
    fn a_sparse_index_above_u32_max_is_read(#[case] backend: Backend) {
        let base = match backend {
            Backend::SparseStream => u64::from(u32::MAX) + 1,
            Backend::Buffer | Backend::Stream => 0,
        };
        let source = sparse_array(base);
        let header =
            ExtensibleArrayHeader::parse_from_source(&source, StoredAddress::new(base), 8, 8)
                .unwrap();
        let mut records = Vec::new();
        let mut visit = |index, record| {
            records.push((index, record));
            Ok(())
        };
        match backend {
            Backend::Buffer => {
                let mut file = Vec::new();
                for (address, block) in &source.blocks {
                    file.resize(address.to_usize().unwrap() + block.len(), 0);
                    file[address.to_usize().unwrap()..].copy_from_slice(block);
                }
                super::super::read_extensible_array_chunks(&file, &header, 8, 8, &mut visit)
                    .unwrap();
            }
            Backend::Stream | Backend::SparseStream => {
                super::super::read_extensible_array_chunks_from_source(
                    &source, &header, 8, 8, &mut visit,
                )
                .unwrap();
                assert_eq!(source.reads.borrow().len(), 4);
                assert_eq!(
                    source.reads.borrow().last().unwrap().1,
                    source.blocks.last().unwrap().1.len()
                );
            }
        }
        assert_eq!(
            records,
            vec![(
                4_295_490_548,
                ChunkRecord {
                    address: StoredAddress::new(0x20),
                    stored_size: 8,
                    filter_mask: 0,
                }
            )]
        );
    }

    #[rstest]
    #[case::full_width(u64::MAX)]
    #[case::above_u32_max(u64::from(u32::MAX) + 1)]
    fn unallocated_blocks_advance_full_width_element_ranges(#[case] published: u64) {
        let source = sparse_array(0);
        let mut header =
            ExtensibleArrayHeader::parse_from_source(&source, StoredAddress::new(0), 8, 8).unwrap();
        header.max_idx_set = published;
        let reader = Reader::parse(&header, 8, 8).unwrap();
        let mut index = source.blocks[1].1.clone();
        let len = index.len();
        index[14..len - 4].fill(0xff);
        checksum::restamp(&mut index, 0, len);
        let source = SparseSource {
            blocks: vec![(header.index_block_address.get(), index)],
            reads: RefCell::new(Vec::new()),
        };
        let mut records = Vec::new();
        reader
            .read(&Streaming(&source), &mut |index, record| {
                records.push((index, record));
                Ok(())
            })
            .unwrap();
        assert_eq!(records, Vec::new());
        assert_eq!(source.reads.borrow().len(), 1);
    }

    #[test]
    fn parsing_a_zero_minimum_block_size_returns_invalid_chunk_geometry() {
        let source = sparse_array(0);
        let mut header =
            ExtensibleArrayHeader::parse_from_source(&source, StoredAddress::new(0), 8, 8).unwrap();
        header.min_dblk_nelmts = 0;
        let Err(error) = Reader::parse(&header, 8, 8) else {
            panic!("zero block size parsed");
        };
        assert_eq!(
            error,
            FormatError::InvalidChunkGeometry(
                "Extensible Array minimum data block element count is zero"
            )
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn a_paged_region_above_usize_max_uses_bounded_reads() {
        let fixture = sparse_array(0);
        let header =
            ExtensibleArrayHeader::parse_from_source(&fixture, StoredAddress::new(0), 8, 8)
                .unwrap();
        let reader = Reader::parse(&header, 8, 8).unwrap();
        let count = 524_288u64;
        let mut bitmap = vec![0; (count / 8).to_usize().unwrap()];
        *bitmap.last_mut().unwrap() = 1;
        let pages = Pages::parse(
            8,
            NonZeroU64::new(1024).unwrap(),
            PageBitmap::parse(&bitmap, 0, NonZeroU64::new(count).unwrap()).unwrap(),
        )
        .unwrap();
        let mut prefix = vec![0; 26];
        prefix[..4].copy_from_slice(b"EADB");
        checksum::restamp(&mut prefix, 0, 26);
        let mut page = vec![0xff; 1024 * 8 + 4];
        page[..8].copy_from_slice(&0x20u64.to_le_bytes());
        let len = page.len();
        checksum::restamp(&mut page, 0, len);
        let page_address = 26 + (count - 1) * pages.stride;
        assert!(page_address > u64::from(u32::MAX));
        let source = SparseSource {
            blocks: vec![(0, prefix), (page_address, page)],
            reads: RefCell::new(Vec::new()),
        };
        let mut records = Vec::new();
        reader
            .data_block(
                &Streaming(&source),
                StoredAddress::new(0),
                Block::Paged(pages),
                Elements::published(count * 1024),
                &mut |index, record| {
                    records.push((index, record));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(
            records,
            vec![(
                (count - 1) * 1024,
                ChunkRecord {
                    address: StoredAddress::new(0x20),
                    stored_size: 8,
                    filter_mask: 0,
                }
            )]
        );
        assert_eq!(*source.reads.borrow(), vec![(0, 26), (page_address, len)]);
    }

    fn sparse_array(base: u64) -> SparseSource {
        let header_size = ExtensibleArrayHeader::serialized_size(8, 8);
        let mut header = vec![0; header_size];
        header[..4].copy_from_slice(b"EAHD");
        header[6..12].copy_from_slice(&[8, 64, 4, 16, 4, 10]);
        let index = 4_295_490_548u64;
        header[44..52].copy_from_slice(&(index + 1).to_le_bytes());
        header[60..68].copy_from_slice(&(base + header_size as u64).to_le_bytes());
        checksum::restamp(&mut header, 0, header_size);
        let parsed = ExtensibleArrayHeader::parse(&header, 0, 8, 8).unwrap();
        let geometry = super::super::ExtensibleArrayGeometry::from_header(&parsed);
        let index_size =
            14 + 4 * 8 + (geometry.direct_dblk_nelmts().len() + geometry.nsblk_addrs()) * 8 + 4;
        let super_address = base + header_size as u64 + index_size as u64;
        let super_geometry = geometry.super_block_at(24, 1024);
        assert_eq!(super_geometry.ndblks, 16_384);
        assert_eq!(super_geometry.blocks.dblk_nelmts, 262_144);
        let super_size = (22 + super_geometry.bitmap_size() + super_geometry.ndblks * 8 + 4)
            .to_usize()
            .unwrap();
        let data_address = super_address + super_size as u64;
        let mut index_block = vec![0xff; index_size];
        index_block[..4].copy_from_slice(b"EAIB");
        index_block[4..6].fill(0);
        index_block[6..14].copy_from_slice(&base.to_le_bytes());
        let pointer = 14 + 4 * 8 + geometry.direct_dblk_nelmts().len() * 8 + 24 * 8;
        index_block[pointer..pointer + 8].copy_from_slice(&super_address.to_le_bytes());
        checksum::restamp(&mut index_block, 0, index_size);
        let mut super_block = vec![0xff; super_size];
        super_block[..4].copy_from_slice(b"EASB");
        super_block[4..6].fill(0);
        super_block[6..14].copy_from_slice(&base.to_le_bytes());
        super_block[14..22].copy_from_slice(&4_294_967_280u64.to_le_bytes());
        let bitmap_size = super_geometry.bitmap_size().to_usize().unwrap();
        super_block[22..22 + bitmap_size].fill(0);
        super_block[22 + 63] = 1;
        super_block[22 + bitmap_size + 8..22 + bitmap_size + 16]
            .copy_from_slice(&data_address.to_le_bytes());
        checksum::restamp(&mut super_block, 0, super_size);
        let prefix_size = 26;
        let page_size = 1024 * 8 + 4;
        let mut data = vec![0; prefix_size + 256 * page_size];
        data[..4].copy_from_slice(b"EADB");
        data[6..14].copy_from_slice(&base.to_le_bytes());
        data[14..22].copy_from_slice(&4_295_229_424u64.to_le_bytes());
        checksum::restamp(&mut data, 0, prefix_size);
        let page = prefix_size + 255 * page_size;
        data[page..page + page_size - 4].fill(0xff);
        data[page..page + 8].copy_from_slice(&0x20u64.to_le_bytes());
        checksum::restamp(&mut data, page, page_size);
        header[12..20].copy_from_slice(&1u64.to_le_bytes());
        header[20..28].copy_from_slice(&(super_size as u64).to_le_bytes());
        header[28..36].copy_from_slice(&1u64.to_le_bytes());
        header[36..44].copy_from_slice(&(data.len() as u64).to_le_bytes());
        header[52..60].copy_from_slice(&262_148u64.to_le_bytes());
        checksum::restamp(&mut header, 0, header_size);
        SparseSource {
            blocks: vec![
                (base, header),
                (parsed.index_block_address.get(), index_block),
                (super_address, super_block),
                (data_address, data),
            ],
            reads: RefCell::new(Vec::new()),
        }
    }

    struct SparseSource {
        blocks: Vec<(u64, Vec<u8>)>,
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl SparseSource {
        fn region(&self, offset: u64, len: usize) -> Result<&[u8], FormatError> {
            self.blocks
                .iter()
                .find_map(|(address, bytes)| {
                    let start = offset.checked_sub(*address)?.to_usize().ok()?;
                    bytes.get(start..start.checked_add(len)?)
                })
                .ok_or(FormatError::UnexpectedEof {
                    expected: len,
                    available: 0,
                })
        }
    }

    impl MetadataSource for SparseSource {
        fn len(&self) -> u64 {
            let (address, bytes) = self.blocks.last().unwrap();
            address + bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            let block = self.region(offset, buf.len())?;
            buf.copy_from_slice(block);
            self.reads.borrow_mut().push((offset, buf.len()));
            Ok(())
        }

        fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            let bytes = self.region(offset, len)?.to_vec();
            self.reads.borrow_mut().push((offset, len));
            Ok(bytes)
        }
    }
}
