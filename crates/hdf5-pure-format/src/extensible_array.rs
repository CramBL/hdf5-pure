//! The Extensible Array chunk index, index type 4 of a version 4 data layout message.
//!
//! An Extensible Array indexes the chunks of a dataset with one unlimited dimension. Its header
//! (`EAHD`) points at an index block (`EAIB`), which holds the first elements, the addresses of
//! the first data blocks (`EADB`), and the addresses of super blocks (`EASB`), each of which
//! addresses further data blocks. A data block of more than one page of elements is paged, and
//! the super block that addresses it holds its page-init bitmap. The index is defined in "The
//! Extensible Array Index" of the [format specification, version 4.0][spec], which calls a super
//! block a secondary block.
//!
//! A caller reads the bytes of each range the parser returns and passes them back to it.
//! [`ExtensibleArrayHeaderLocator`] returns the range of the header, and each reference to a block
//! or a page, from [`ExtensibleArrayIndexBlockRef`] to [`ExtensibleArrayPageRef`], the range a
//! caller reads for it. The reference to a super block or a data block also returns whether the
//! block holds a slot below the header's "Max Index Set", one more than the highest slot the array
//! has set.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU8;
use core::num::NonZeroU64;
use core::ops::Range;

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
use crate::chunk_record::ArrayClient;
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

/// The address of an Extensible Array header, from the chunk index address of a data layout
/// message.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse), which returns the header and the reference to its index block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayHeaderLocator {
    widths: FormatWidths,
    base: BaseAddress,
    address: StoredAddress,
}

impl ExtensibleArrayHeaderLocator {
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
    /// An unfiltered element is the chunk address alone, whatever the header's element size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidExtensibleArraySignature`] if the
    /// bytes do not start with `EAHD`, [`FormatError::InvalidExtensibleArrayVersion`] if the
    /// version is not 0, [`FormatError::InvalidExtensibleArrayClient`] if the client ID is 2 or
    /// more, and [`FormatError::InvalidExtensibleArrayGeometry`] if the element size or a creation
    /// parameter is out of range. Returns [`FormatError::OffsetOverflow`] if the base address plus
    /// `address` or plus the index block address, or the end of the header or of the index block,
    /// is past [`u64::MAX`], and, under the `checksum` feature, [`FormatError::ChecksumMismatch`]
    /// if the checksum does not match the header's bytes.
    pub fn parse(&self, region: FileRegion<'_>) -> Result<ExtensibleArray, FormatError> {
        let Self {
            widths,
            base,
            address: _,
        } = *self;
        let range = self.parse_range()?;
        let bytes = region.subregion(range)?.bytes();
        let mut fields = bytes::Fields::new(bytes, 0);
        if fields.array::<{ HEADER_SIGNATURE.len() }>()? != HEADER_SIGNATURE {
            return Err(FormatError::InvalidExtensibleArraySignature);
        }
        let version = fields.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidExtensibleArrayVersion(version));
        }
        let client_id = fields.u8()?;
        let element_size = fields.u8()?;
        let max_index_bits = fields.u8()?;
        let index_block_elements = fields.u8()?;
        let data_block_min_elements = fields.u8()?;
        let super_block_min_data_blocks = fields.u8()?;
        let page_bits = fields.u8()?;
        for _ in 0..STATISTICS_BEFORE_MAX_INDEX_SET {
            fields.length(widths.lengths)?;
        }
        let max_index_set = fields.length(widths.lengths)?;
        // The number of elements, the last statistic.
        fields.length(widths.lengths)?;
        let index_block_address = fields.address(widths.offsets)?;
        checksum::verify_trailing(bytes)?;

        let encoding = ElementEncoding::new(widths.offsets, client_id, element_size).map_err(
            |err| match err {
                InvalidElementEncoding::Client(id) => FormatError::InvalidExtensibleArrayClient(id),
                InvalidElementEncoding::EntrySize => FormatError::InvalidExtensibleArrayGeometry,
            },
        )?;
        let layout = Layout::new(
            encoding,
            max_index_bits,
            index_block_elements,
            data_block_min_elements,
            super_block_min_data_blocks,
            page_bits,
        )?;
        let header = ExtensibleArrayHeader {
            client_id,
            element_size,
            bit_geometry: ExtensibleArrayBitGeometry::parse(max_index_bits, page_bits)?,
            idx_blk_elmts: index_block_elements,
            min_dblk_nelmts: data_block_min_elements,
            super_blk_min_data_ptrs: super_block_min_data_blocks,
            max_idx_set: max_index_set,
            index_block_address,
        };
        let owner = Owner {
            base,
            header: range.start(),
            layout,
            max_index_set,
        };
        let index_block = if index_block_address.is_undefined(widths.offsets) {
            None
        } else {
            Some(ExtensibleArrayIndexBlockRef {
                owner,
                range: FileRange::new(
                    owner.resolve(index_block_address)?,
                    layout.index_block_len()?,
                )?,
            })
        };
        Ok(ExtensibleArray {
            range,
            header,
            layout,
            index_block,
        })
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

/// The expected prefix fields and physical ranges of a version 0 secondary or data block.
///
/// A caller can parse the prefix separately from reading and verifying the checksummed region.
/// For a paged data block, that region ends before the pages, while the allocation includes every
/// page. The block layouts are defined in "The Extensible Array Index" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayBlockFraming {
    owner: Owner,
    range: FileRange,
    parse_range: FileRange,
    signature: [u8; 4],
    offset: (u64, Option<u64>),
}

impl ExtensibleArrayBlockFraming {
    fn data(owner: Owner, address: PhysicalAddress, block: DataBlock) -> Result<Self, FormatError> {
        let layout = owner.layout;
        let range = FileRange::new(address, layout.data_block_len(block)?)?;
        Ok(Self {
            owner,
            range,
            parse_range: if layout.pages(block.elements).is_some() {
                FileRange::new(address, layout.pages_offset())?
            } else {
                range
            },
            signature: DATA_BLOCK_SIGNATURE,
            offset: (block.offset, block.libhdf5_offset),
        })
    }

    /// Parses the signature, version, client ID, header address, and block offset in `region`.
    ///
    /// The caller verifies the checksum over [`parse_range`](Self::parse_range) separately.
    /// For a direct data block, accepts either the block's position or the alternate offset
    /// described by [`ExtensibleArrayDataBlockRef::parse`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`prefix_range`](Self::prefix_range), [`FormatError::InvalidExtensibleArraySignature`] if
    /// the signature differs from the expected block kind, or
    /// [`FormatError::InvalidExtensibleArrayVersion`] if the version is not 0. Returns
    /// [`FormatError::InvalidExtensibleArrayBlockClient`] if the client differs from the header's,
    /// [`FormatError::InvalidExtensibleArrayBlockOwner`] if the header address differs from the
    /// array's, or [`FormatError::InvalidExtensibleArrayBlockOffset`] if the block offset is
    /// outside the accepted offsets. Returns
    /// [`FormatError::OffsetOverflow`] if the base address plus the stored header address
    /// exceeds [`u64::MAX`].
    pub fn parse_prefix(&self, region: FileRegion<'_>) -> Result<(), FormatError> {
        let Self {
            owner,
            range: _,
            parse_range: _,
            signature,
            offset,
        } = *self;
        let prefix = region.subregion(self.prefix_range()?)?;
        let mut fields = owner.parse_prefix(prefix.bytes(), signature)?;
        let (expected, alternate) = offset;
        owner.parse_block_offset(&mut fields, expected, alternate)?;
        Ok(())
    }

    /// Returns the prefix range through the block offset, excluding the checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the end exceeds [`u64::MAX`].
    pub fn prefix_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(
            self.range.start(),
            self.owner.layout.prefix_len() + u64::from(self.owner.layout.block_offset_len),
        )
    }

    /// Returns the checksummed region, including its trailing checksum.
    ///
    /// For a paged data block, the region excludes the separately checksummed pages.
    pub const fn parse_range(&self) -> FileRange {
        self.parse_range
    }

    /// Returns the complete allocation, including every page of a paged data block.
    pub const fn range(&self) -> FileRange {
        self.range
    }
}

/// The parsed header of an Extensible Array and the reference to its index block, from
/// [`ExtensibleArrayHeaderLocator::parse`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArray {
    range: FileRange,
    header: ExtensibleArrayHeader,
    layout: Layout,
    index_block: Option<ExtensibleArrayIndexBlockRef>,
}

impl ExtensibleArray {
    /// Returns the physical range of the header.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the stored header fields.
    ///
    /// The [`index_block_address`](ExtensibleArrayHeader::index_block_address) field holds a
    /// base-relative address or the file-width undefined address.
    pub const fn header(&self) -> &ExtensibleArrayHeader {
        &self.header
    }

    /// Returns the block geometry defined by the header's creation parameters.
    ///
    /// Includes the geometry of unallocated blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset
    /// exceeds [`u64::MAX`].
    pub fn geometry(&self) -> Result<ExtensibleArrayGeometry, FormatError> {
        ExtensibleArrayGeometry::from_layout(self.layout)
    }

    /// Returns whether the array indexes filtered chunks, the header's client ID 1.
    pub const fn is_filtered(&self) -> bool {
        matches!(
            self.layout.encoding.client(),
            ArrayClient::FilteredChunks { .. }
        )
    }

    /// Returns the reference to the index block, or `None` if the header stores the undefined
    /// address, in an array without chunks.
    pub const fn index_block(&self) -> Option<ExtensibleArrayIndexBlockRef> {
        self.index_block
    }
}

/// The position and the layout of an Extensible Array index block, from
/// [`ExtensibleArray::index_block`].
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayIndexBlockRef {
    owner: Owner,
    range: FileRange,
}

impl ExtensibleArrayIndexBlockRef {
    /// Parses the index block in `region`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidExtensibleArraySignature`] if the
    /// bytes do not start with `EAIB`, [`FormatError::InvalidExtensibleArrayVersion`] if the
    /// version is not 0, [`FormatError::InvalidExtensibleArrayBlockClient`] if the client ID
    /// differs from the header's, and [`FormatError::InvalidExtensibleArrayBlockOwner`] if the
    /// header address differs from the header's own. Returns
    /// [`FormatError::OffsetOverflow`] if the base address plus the header address is past
    /// [`u64::MAX`], and, under the `checksum` feature, [`FormatError::ChecksumMismatch`] if the
    /// checksum does not match the index block's bytes.
    pub fn parse<'a>(
        &self,
        region: FileRegion<'a>,
    ) -> Result<ExtensibleArrayIndexBlock<'a>, FormatError> {
        let Self { owner, range } = *self;
        let bytes = region.subregion(range)?.bytes();
        let fields = owner.parse_prefix(bytes, INDEX_BLOCK_SIGNATURE)?;
        checksum::verify_trailing(bytes)?;
        let layout = owner.layout;
        let (elements, addresses) = body(bytes, fields.pos())?
            .split_at_checked(
                layout
                    .elements_len(u64::from(layout.inline_elements))?
                    .to_usize()?,
            )
            .ok_or_else(|| internal("index block ends inside its elements"))?;
        let (data_blocks, super_blocks) = addresses
            .split_at_checked(
                layout
                    .addresses_len(layout.direct_data_blocks())?
                    .to_usize()?,
            )
            .ok_or_else(|| internal("index block ends inside its data block addresses"))?;
        Ok(ExtensibleArrayIndexBlock {
            owner,
            elements,
            data_blocks,
            super_blocks,
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the whole index block.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }

    /// Returns the complete allocation of the index block, including its checksum.
    pub const fn range(&self) -> FileRange {
        self.range
    }
}

/// A parsed Extensible Array index block, from [`ExtensibleArrayIndexBlockRef::parse`].
///
/// The index block holds the elements of the first slots, the addresses of the data blocks after
/// them, and the addresses of the super blocks that hold the addresses of the remaining data
/// blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayIndexBlock<'a> {
    owner: Owner,
    elements: &'a [u8],
    data_blocks: &'a [u8],
    super_blocks: &'a [u8],
}

impl<'a> ExtensibleArrayIndexBlock<'a> {
    /// Returns the elements the index block holds, from slot 0.
    pub fn elements(&self) -> ExtensibleArrayElements<'a> {
        ExtensibleArrayElements::below_max_index_set(self.owner, self.elements, 0)
    }

    /// Returns the reference to each data block whose address the index block holds, in slot
    /// order.
    ///
    /// A data block whose address is undefined is not allocated and has no reference.
    pub fn data_blocks(
        &self,
    ) -> impl Iterator<Item = Result<ExtensibleArrayDataBlockRef<'a>, FormatError>> + 'a {
        let Self {
            owner, data_blocks, ..
        } = *self;
        owner
            .addresses(data_blocks)
            .zip(owner.layout.direct_blocks())
            .filter_map(move |(address, block)| {
                address
                    .transpose()
                    .map(|address| ExtensibleArrayDataBlockRef::new(owner, address?, block?, None))
            })
    }

    /// Returns the reference to each super block whose address the index block holds, in slot
    /// order.
    ///
    /// A super block whose address is undefined is not allocated and has no reference.
    pub fn super_blocks(
        &self,
    ) -> impl Iterator<Item = Result<ExtensibleArraySuperBlockRef, FormatError>> + 'a {
        let Self {
            owner,
            super_blocks,
            ..
        } = *self;
        let layout = owner.layout;
        owner
            .addresses(super_blocks)
            .zip(
                (layout.first_super_block..layout.super_blocks)
                    .map(move |index| layout.super_block(index)),
            )
            .filter_map(move |(address, super_block)| {
                address
                    .transpose()
                    .map(|address| ExtensibleArraySuperBlockRef::new(owner, address?, super_block?))
            })
    }
}

/// The position and the layout of an Extensible Array super block, from
/// [`ExtensibleArrayIndexBlock::super_blocks`].
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArraySuperBlockRef {
    owner: Owner,
    range: FileRange,
    super_block: SuperBlock,
}

impl ExtensibleArraySuperBlockRef {
    fn new(
        owner: Owner,
        address: PhysicalAddress,
        super_block: SuperBlock,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            owner,
            range: FileRange::new(address, owner.layout.super_block_len(super_block)?)?,
            super_block,
        })
    }

    /// Returns the expected prefix fields and ranges for this secondary block.
    pub fn framing(&self) -> ExtensibleArrayBlockFraming {
        let Self {
            owner,
            range,
            super_block,
        } = *self;
        ExtensibleArrayBlockFraming {
            owner,
            range,
            parse_range: range,
            signature: SUPER_BLOCK_SIGNATURE,
            offset: (super_block.offset, None),
        }
    }

    /// Returns the expected prefix fields and ranges for data block `local` at `address`.
    ///
    /// `local` counts from zero within this secondary block. The caller reads `address` from
    /// its data block pointers.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if `local` is out of range or
    /// the block offset or allocation length exceeds [`u64::MAX`]. Returns
    /// [`FormatError::OffsetOverflow`] if the allocation end exceeds [`u64::MAX`].
    pub fn data_block_framing(
        &self,
        local: u64,
        address: PhysicalAddress,
    ) -> Result<ExtensibleArrayBlockFraming, FormatError> {
        let Self {
            owner,
            range: _,
            super_block,
        } = *self;
        if local >= super_block.blocks {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        ExtensibleArrayBlockFraming::data(owner, address, super_block.data_block(local)?)
    }

    /// Parses the super block in `region`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidExtensibleArraySignature`] if the
    /// bytes do not start with `EASB`, [`FormatError::InvalidExtensibleArrayVersion`] if the
    /// version is not 0, [`FormatError::InvalidExtensibleArrayBlockClient`] if the client ID
    /// differs from the header's, and [`FormatError::InvalidExtensibleArrayBlockOwner`] if the
    /// header address differs from the header's own. Returns
    /// [`FormatError::InvalidExtensibleArrayBlockOffset`] if the block offset differs from the
    /// super block's position in the array, [`FormatError::OffsetOverflow`] if the base address
    /// plus the header address is past [`u64::MAX`], and, under the `checksum` feature,
    /// [`FormatError::ChecksumMismatch`] if the checksum does not match the super block's bytes.
    pub fn parse<'a>(
        &self,
        region: FileRegion<'a>,
    ) -> Result<ExtensibleArraySuperBlock<'a>, FormatError> {
        let Self {
            owner,
            range,
            super_block,
        } = *self;
        let bytes = region.subregion(range)?.bytes();
        let mut fields = owner.parse_prefix(bytes, SUPER_BLOCK_SIGNATURE)?;
        owner.parse_block_offset(&mut fields, super_block.offset, None)?;
        checksum::verify_trailing(bytes)?;
        let (bitmap, data_blocks) = body(bytes, fields.pos())?
            .split_at_checked(owner.layout.bitmap_len(super_block)?.to_usize()?)
            .ok_or_else(|| internal("super block ends inside its page-init bitmap"))?;
        Ok(ExtensibleArraySuperBlock {
            owner,
            super_block,
            bitmap,
            data_blocks,
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the whole super block.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }

    /// Returns whether the super block holds a slot below the header's "Max Index Set".
    ///
    /// The array has set none of the slots of a super block that does not, and a caller need not
    /// read it.
    pub fn holds_set_slots(&self) -> bool {
        self.owner.holds_set_slots(self.super_block.offset)
    }

    /// Returns the slots of the data blocks whose addresses the super block holds.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the range ends past
    /// [`u64::MAX`].
    pub fn slots(&self) -> Result<Range<u64>, FormatError> {
        let SuperBlock {
            offset,
            blocks,
            block_elements,
        } = self.super_block;
        self.owner
            .layout
            .slots(offset, product(blocks, block_elements)?)
    }
}

/// A parsed Extensible Array super block, from [`ExtensibleArraySuperBlockRef::parse`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArraySuperBlock<'a> {
    owner: Owner,
    super_block: SuperBlock,
    bitmap: &'a [u8],
    data_blocks: &'a [u8],
}

impl<'a> ExtensibleArraySuperBlock<'a> {
    /// Returns the reference to each data block whose address the super block holds, in slot
    /// order.
    ///
    /// A data block whose address is undefined is not allocated and has no reference. The
    /// reference to a paged data block holds the bits of the super block's page-init bitmap that
    /// mark its pages.
    pub fn data_blocks(
        &self,
    ) -> impl Iterator<Item = Result<ExtensibleArrayDataBlockRef<'a>, FormatError>> + 'a {
        let Self {
            owner,
            super_block,
            bitmap,
            data_blocks,
        } = *self;
        owner
            .addresses(data_blocks)
            .zip((0..super_block.blocks).map(move |local| (local, super_block.data_block(local))))
            .filter_map(move |(address, (local, block))| {
                address.transpose().map(|address| {
                    let block = block?;
                    let pages = match owner.layout.pages(block.elements) {
                        Some(pages) => Some(PageBitmap {
                            bitmap,
                            first_page: product(local, pages)?,
                            pages,
                        }),
                        None => None,
                    };
                    ExtensibleArrayDataBlockRef::new(owner, address?, block, pages)
                })
            })
    }
}

/// The position and the layout of an Extensible Array data block, from
/// [`ExtensibleArrayIndexBlock::data_blocks`] or [`ExtensibleArraySuperBlock::data_blocks`].
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse). The range is the whole data block if it is unpaged, and its prefix if
/// it is paged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayDataBlockRef<'a> {
    owner: Owner,
    range: FileRange,
    parse_range: FileRange,
    block: DataBlock,
    pages: Option<PageBitmap<'a>>,
}

impl<'a> ExtensibleArrayDataBlockRef<'a> {
    fn new(
        owner: Owner,
        address: PhysicalAddress,
        block: DataBlock,
        pages: Option<PageBitmap<'a>>,
    ) -> Result<Self, FormatError> {
        let layout = owner.layout;
        let range = FileRange::new(address, layout.data_block_len(block)?)?;
        Ok(Self {
            owner,
            range,
            parse_range: match pages {
                Some(_) => FileRange::new(address, layout.pages_offset())?,
                None => range,
            },
            block,
            pages,
        })
    }

    /// Returns the expected prefix fields and ranges for this data block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the allocation length exceeds
    /// [`u64::MAX`], or [`FormatError::OffsetOverflow`] if its end exceeds [`u64::MAX`].
    pub fn framing(&self) -> Result<ExtensibleArrayBlockFraming, FormatError> {
        let Self {
            owner,
            range,
            parse_range: _,
            block,
            pages: _,
        } = *self;
        ExtensibleArrayBlockFraming::data(owner, range.start(), block)
    }

    /// Parses the data block in `region`, and returns its elements or, in a paged block, its
    /// pages.
    ///
    /// libhdf5 stores another block offset in a data block whose address the index block
    /// holds, and the parser accepts that offset as well as the block's position.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidExtensibleArraySignature`] if the
    /// bytes do not start with `EADB`, [`FormatError::InvalidExtensibleArrayVersion`] if the
    /// version is not 0, [`FormatError::InvalidExtensibleArrayBlockClient`] if the client ID
    /// differs from the header's, and [`FormatError::InvalidExtensibleArrayBlockOwner`] if the
    /// header address differs from the header's own. Returns
    /// [`FormatError::InvalidExtensibleArrayBlockOffset`] if the block offset differs from the
    /// data block's position in the array, [`FormatError::InvalidExtensibleArrayGeometry`] if the
    /// block's first slot is past [`u64::MAX`], [`FormatError::OffsetOverflow`] if the base
    /// address plus the header address, or the start of the first page of a paged block, is past
    /// [`u64::MAX`], and, under the `checksum` feature, [`FormatError::ChecksumMismatch`] if the
    /// checksum does not match the bytes of the range.
    pub fn parse<'b>(
        &self,
        region: FileRegion<'b>,
    ) -> Result<ExtensibleArrayDataBlock<'b>, FormatError>
    where
        'a: 'b,
    {
        let Self {
            owner,
            parse_range,
            block,
            ..
        } = *self;
        let bytes = region.subregion(parse_range)?.bytes();
        let mut fields = owner.parse_prefix(bytes, DATA_BLOCK_SIGNATURE)?;
        // For a data block the index block holds the address of, `H5EA__lookup_elmt` (`H5EA.c`,
        // HDF5 1.10.11 to 2.2.0) stores the offset of its super block plus its ordinal among those
        // data blocks times its element count, which differs from its position from super block 1
        // on.
        owner.parse_block_offset(&mut fields, block.offset, block.libhdf5_offset)?;
        checksum::verify_trailing(bytes)?;
        Ok(match self.pages()? {
            Some(pages) => ExtensibleArrayDataBlock::Pages(pages),
            None => {
                ExtensibleArrayDataBlock::Elements(ExtensibleArrayElements::below_max_index_set(
                    owner,
                    body(bytes, fields.pos())?,
                    owner.layout.first_slot(block.offset)?,
                ))
            }
        })
    }

    /// Returns the pages of a paged data block, or `None` if it is unpaged.
    ///
    /// A caller has the pages before it parses the data block, and can read the prefix and the
    /// pages that follow it in one read.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the first page starts past [`u64::MAX`], and
    /// [`FormatError::InvalidExtensibleArrayGeometry`] if the block's first slot is past
    /// [`u64::MAX`].
    pub fn pages(&self) -> Result<Option<ExtensibleArrayPages<'a>>, FormatError> {
        let Self {
            owner,
            range,
            block,
            pages,
            ..
        } = *self;
        pages
            .map(|bitmap| {
                Ok(ExtensibleArrayPages {
                    owner,
                    first_page: range.start().checked_add(owner.layout.pages_offset())?,
                    first_slot: owner.layout.first_slot(block.offset)?,
                    bitmap,
                })
            })
            .transpose()
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the whole data block if it is
    /// unpaged and its prefix if it is paged.
    pub const fn parse_range(&self) -> FileRange {
        self.parse_range
    }

    /// Returns the physical range of the whole data block, the pages of a paged one included.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns whether the data block holds a slot below the header's "Max Index Set".
    ///
    /// The array has set none of the slots of a data block that does not, and a caller need not
    /// read it.
    pub fn holds_set_slots(&self) -> bool {
        self.owner.holds_set_slots(self.block.offset)
    }

    /// Returns the slots of the data block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the range ends past
    /// [`u64::MAX`].
    pub fn slots(&self) -> Result<Range<u64>, FormatError> {
        self.owner
            .layout
            .slots(self.block.offset, self.block.elements)
    }
}

/// A parsed Extensible Array data block, from [`ExtensibleArrayDataBlockRef::parse`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExtensibleArrayDataBlock<'a> {
    /// The elements of an unpaged data block.
    Elements(ExtensibleArrayElements<'a>),
    /// The pages of a paged data block.
    Pages(ExtensibleArrayPages<'a>),
}

/// The pages of a paged Extensible Array data block, and the bits of its super block's page-init
/// bitmap that mark them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayPages<'a> {
    owner: Owner,
    /// The address of the first page.
    first_page: PhysicalAddress,
    /// The slot of the first element of the first page.
    first_slot: u64,
    bitmap: PageBitmap<'a>,
}

impl<'a> ExtensibleArrayPages<'a> {
    /// Returns the reference to each page the bitmap marks initialized, in page order.
    ///
    /// A page the bitmap does not mark is uninitialized and has no reference, and neither has a
    /// page whose first slot is at or past the header's "Max Index Set".
    pub fn pages(
        &self,
    ) -> impl Iterator<Item = Result<ExtensibleArrayPageRef, FormatError>> + Clone + 'a {
        let Self {
            owner,
            first_page,
            first_slot,
            bitmap,
        } = *self;
        let layout = owner.layout;
        (0..bitmap.pages)
            .take_while(move |&page| {
                page.checked_mul(layout.page_elements())
                    .and_then(|offset| offset.checked_add(first_slot))
                    .is_some_and(|slot| slot < owner.max_index_set)
            })
            .filter(move |&page| bitmap.initialized(page))
            .map(move |page| {
                let page_len = layout.page_len()?;
                Ok(ExtensibleArrayPageRef {
                    owner,
                    range: FileRange::new(
                        first_page.checked_add(product(page, page_len)?)?,
                        page_len,
                    )?,
                    first_slot: sum([first_slot, product(page, layout.page_elements())?])?,
                })
            })
    }
}

/// The position of one page of a paged Extensible Array data block, from
/// [`ExtensibleArrayPages::pages`].
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayPageRef {
    owner: Owner,
    range: FileRange,
    first_slot: u64,
}

impl ExtensibleArrayPageRef {
    /// Parses the page in `region` and returns its elements.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), and, under the `checksum` feature,
    /// [`FormatError::ChecksumMismatch`] if the checksum does not match the page's bytes.
    pub fn parse<'a>(
        &self,
        region: FileRegion<'a>,
    ) -> Result<ExtensibleArrayElements<'a>, FormatError> {
        let Self {
            owner,
            range,
            first_slot,
        } = *self;
        let bytes = region.subregion(range)?.bytes();
        checksum::verify_trailing(bytes)?;
        Ok(ExtensibleArrayElements::below_max_index_set(
            owner,
            body(bytes, 0)?,
            first_slot,
        ))
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the page's elements and its
    /// checksum.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }

    /// Returns the slots of the page.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the range ends past
    /// [`u64::MAX`].
    pub fn slots(&self) -> Result<Range<u64>, FormatError> {
        let elements = self.owner.layout.page_elements();
        let end = sum([self.first_slot, elements])?;
        Ok(self.first_slot..end)
    }
}

/// The elements of an Extensible Array index block, unpaged data block, or page.
///
/// The elements end at the header's "Max Index Set", and an element of a slot at or past it has no
/// record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayElements<'a> {
    bytes: &'a [u8],
    first_slot: u64,
    encoding: ElementEncoding,
}

impl<'a> ExtensibleArrayElements<'a> {
    fn below_max_index_set(owner: Owner, bytes: &'a [u8], first_slot: u64) -> Self {
        let encoding = owner.layout.encoding;
        let len = owner
            .max_index_set
            .saturating_sub(first_slot)
            .to_usize()
            .ok()
            .and_then(|count| count.checked_mul(usize::from(encoding.stride().get())))
            .map_or(bytes.len(), |len| len.min(bytes.len()));
        // `len` is at most `bytes.len()`, so the split is inside `bytes`.
        let (bytes, _at_or_past_max_index_set) = bytes.split_at(len);
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

/// The fields of an Extensible Array header that each of its blocks is parsed against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Owner {
    base: BaseAddress,
    /// The physical address of the header, the address a block's header address resolves to.
    header: PhysicalAddress,
    layout: Layout,
    /// The header's "Max Index Set", one more than the highest slot the array has set.
    max_index_set: u64,
}

impl Owner {
    fn resolve(self, address: StoredAddress) -> Result<PhysicalAddress, FormatError> {
        self.base.resolve(address)
    }

    fn addresses<'a>(
        self,
        addresses: &'a [u8],
    ) -> impl Iterator<Item = Result<Option<PhysicalAddress>, FormatError>> + 'a {
        let offsets = self.layout.encoding.offsets();
        addresses
            .chunks_exact(usize::from(offsets.get()))
            .map(move |field| {
                Ok(
                    match bytes::Fields::new(field, 0).decoded_address(offsets)? {
                        DecodedStoredAddress::Defined(address) => Some(self.resolve(address)?),
                        DecodedStoredAddress::Undefined => None,
                    },
                )
            })
    }

    /// Parses the signature, the version, the client ID, and the header address of a block, and
    /// returns the fields that follow them.
    fn parse_prefix<'a, const N: usize>(
        self,
        bytes: &'a [u8],
        signature: [u8; N],
    ) -> Result<bytes::Fields<'a>, FormatError> {
        let mut fields = bytes::Fields::new(bytes, 0);
        if fields.array::<N>()? != signature {
            return Err(FormatError::InvalidExtensibleArraySignature);
        }
        let version = fields.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidExtensibleArrayVersion(version));
        }
        let client_id = fields.u8()?;
        let expected = self.layout.encoding.client().id();
        if client_id != expected {
            return Err(FormatError::InvalidExtensibleArrayBlockClient {
                expected,
                actual: client_id,
            });
        }
        let owner = self.resolve(fields.address(self.layout.encoding.offsets())?)?;
        if owner != self.header {
            return Err(FormatError::InvalidExtensibleArrayBlockOwner {
                expected: self.header.get(),
                actual: owner.get(),
            });
        }
        Ok(fields)
    }

    fn holds_set_slots(self, offset: u64) -> bool {
        self.layout
            .first_slot(offset)
            .is_ok_and(|slot| slot < self.max_index_set)
    }

    /// Parses a block offset and checks that it is `expected` or, where the caller passes one,
    /// `libhdf5`.
    fn parse_block_offset(
        self,
        fields: &mut bytes::Fields<'_>,
        expected: u64,
        libhdf5: Option<u64>,
    ) -> Result<(), FormatError> {
        let actual = fields.uint(usize::from(self.layout.block_offset_len))?;
        if actual != expected && Some(actual) != libhdf5 {
            return Err(FormatError::InvalidExtensibleArrayBlockOffset { expected, actual });
        }
        Ok(())
    }
}

// Super block `u` holds `2^floor(u/2)` data blocks of `min << ceil(u/2)` elements each, as
// `H5EA__hdr_init` computes them (`H5EAhdr.c`, HDF5 2.2.0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Layout {
    encoding: ElementEncoding,
    /// The number of elements the index block holds.
    inline_elements: u8,
    /// The base 2 logarithm of the number of elements in the smallest data block.
    min_elements_bits: u32,
    /// The base 2 logarithm of the number of elements in a page.
    page_bits: u32,
    block_offset_len: u8,
    /// The index of the first super block whose address the index block holds.
    first_super_block: u32,
    /// The number of super blocks, those before `first_super_block` included.
    super_blocks: u32,
}

impl Layout {
    fn new(
        encoding: ElementEncoding,
        max_index_bits: u8,
        inline_elements: u8,
        min_elements: u8,
        min_data_blocks: u8,
        page_bits: u8,
    ) -> Result<Self, FormatError> {
        Ok(ExtensibleArrayCreationParameters::parse(&[
            max_index_bits,
            inline_elements,
            min_data_blocks,
            min_elements,
            page_bits,
        ])?
        .layout(encoding))
    }

    fn direct_blocks(self) -> impl Iterator<Item = Result<DataBlock, FormatError>> {
        (0..self.first_super_block)
            .map(move |index| self.super_block(index))
            .flat_map(|super_block| {
                let blocks = super_block
                    .as_ref()
                    .map_or(1, |super_block| super_block.blocks);
                (0..blocks)
                    .map(move |local| super_block.clone().map(|super_block| (super_block, local)))
            })
            .zip(0..)
            .map(|(block, ordinal)| {
                let (super_block, local) = block?;
                super_block.direct_data_block(local, ordinal)
            })
    }

    // `2 * (min_data_blocks - 1)`, the data blocks of the super blocks before
    // `first_super_block`.
    fn direct_data_blocks(self) -> u64 {
        // `Layout::new` rejects fewer than 2 data block addresses, so `first_super_block` is at
        // least 2 and the shift at least 1.
        2 * ((1 << (self.first_super_block / 2)) - 1)
    }

    fn super_block(self, index: u32) -> Result<SuperBlock, FormatError> {
        // Every caller passes an index below `super_blocks`, which `Layout::new` bounds at 65, so
        // the subtraction and the shifts below stay inside a `u64`.
        debug_assert!(index < self.super_blocks);
        let min_elements = 1 << self.min_elements_bits;
        // `2^index - 1`.
        let before = u64::MAX.checked_shr(u64::BITS - index).unwrap_or(0);
        Ok(SuperBlock {
            offset: product(min_elements, before)?,
            blocks: 1 << (index / 2),
            block_elements: min_elements << index.div_ceil(2),
        })
    }

    // A data block of more elements than a page is paged, as `H5EA__dblock_alloc` pages it
    // (`H5EAdblock.c`, HDF5 2.2.0).
    fn pages(self, elements: u64) -> Option<u64> {
        let block = DataBlockGeometry {
            dblk_nelmts: elements,
            page_nelmts: self.page_elements(),
        };
        block.is_paged().then(|| block.npages())
    }

    const fn page_elements(self) -> u64 {
        1 << self.page_bits
    }

    /// Returns the slot of the first element of the block at `offset`.
    fn first_slot(self, offset: u64) -> Result<u64, FormatError> {
        sum([u64::from(self.inline_elements), offset])
    }

    fn slots(self, offset: u64, elements: u64) -> Result<Range<u64>, FormatError> {
        let start = self.first_slot(offset)?;
        Ok(start..sum([start, elements])?)
    }

    fn elements_len(self, elements: u64) -> Result<u64, FormatError> {
        product(elements, u64::from(self.encoding.stride().get()))
    }

    fn addresses_len(self, addresses: u64) -> Result<u64, FormatError> {
        product(addresses, u64::from(self.encoding.offsets().get()))
    }

    fn index_block_len(self) -> Result<u64, FormatError> {
        // `Layout::new` rejects a `first_super_block` past `super_blocks`.
        sum([
            self.prefix_len(),
            self.elements_len(u64::from(self.inline_elements))?,
            self.addresses_len(self.direct_data_blocks())?,
            self.addresses_len(u64::from(self.super_blocks - self.first_super_block))?,
            CHECKSUM_LEN,
        ])
    }

    // Each data block has `ceil(pages / 8)` bytes of the bitmap, as `H5EA__sblock_alloc` sizes it
    // (`H5EAsblock.c`, HDF5 2.2.0).
    fn bitmap_len(self, super_block: SuperBlock) -> Result<u64, FormatError> {
        self.pages(super_block.block_elements)
            .map_or(Ok(0), |pages| {
                product(super_block.blocks, pages.div_ceil(8))
            })
    }

    fn super_block_len(self, super_block: SuperBlock) -> Result<u64, FormatError> {
        sum([
            self.prefix_len(),
            u64::from(self.block_offset_len),
            self.bitmap_len(super_block)?,
            self.addresses_len(super_block.blocks)?,
            CHECKSUM_LEN,
        ])
    }

    fn data_block_len(self, block: DataBlock) -> Result<u64, FormatError> {
        match self.pages(block.elements) {
            Some(pages) => sum([self.pages_offset(), product(pages, self.page_len()?)?]),
            None => sum([
                self.prefix_len(),
                u64::from(self.block_offset_len),
                self.elements_len(block.elements)?,
                CHECKSUM_LEN,
            ]),
        }
    }

    fn pages_offset(self) -> u64 {
        self.prefix_len() + u64::from(self.block_offset_len) + CHECKSUM_LEN
    }

    fn page_len(self) -> Result<u64, FormatError> {
        sum([self.elements_len(self.page_elements())?, CHECKSUM_LEN])
    }

    fn prefix_len(self) -> u64 {
        SIGNATURE_LEN + BLOCK_BYTE_FIELDS + u64::from(self.encoding.offsets().get())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SuperBlock {
    /// The block offset of the super block and of its first data block.
    offset: u64,
    blocks: u64,
    block_elements: u64,
}

impl SuperBlock {
    fn data_block(self, local: u64) -> Result<DataBlock, FormatError> {
        Ok(DataBlock {
            offset: sum([self.offset, product(local, self.block_elements)?])?,
            elements: self.block_elements,
            libhdf5_offset: None,
        })
    }

    // `ordinal` counts the data blocks the index block holds the addresses of, from the first.
    fn direct_data_block(self, local: u64, ordinal: u64) -> Result<DataBlock, FormatError> {
        Ok(DataBlock {
            libhdf5_offset: Some(sum([self.offset, product(ordinal, self.block_elements)?])?),
            ..self.data_block(local)?
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DataBlock {
    offset: u64,
    elements: u64,
    /// The block offset the C library stores in a data block whose address the index block holds,
    /// or `None` in one whose address a super block holds.
    libhdf5_offset: Option<u64>,
}

// The bits of the data blocks of a super block form one sequence, and page `p` of data block `k`
// is bit `k * pages + p`, as `H5EA__lookup_elmt` indexes it (`H5EA.c`, HDF5 2.2.0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PageBitmap<'a> {
    bitmap: &'a [u8],
    /// The bit of the first page of the data block.
    first_page: u64,
    /// The number of pages in the data block.
    pages: u64,
}

impl PageBitmap<'_> {
    fn initialized(self, page: u64) -> bool {
        self.first_page
            .checked_add(page)
            .is_some_and(|bit| chunk_record::page_initialized(self.bitmap, bit))
    }
}

fn header_len(FormatWidths { offsets, lengths }: FormatWidths) -> u64 {
    SIGNATURE_LEN
        + HEADER_BYTE_FIELDS
        + STATISTICS * u64::from(lengths.get())
        + u64::from(offsets.get())
        + CHECKSUM_LEN
}

/// Returns the bytes of a block from `pos` to its checksum.
fn body(bytes: &[u8], pos: usize) -> Result<&[u8], FormatError> {
    bytes
        .split_last_chunk::<{ size_of::<u32>() }>()
        .and_then(|(body, _checksum)| body.get(pos..))
        .ok_or_else(|| internal("block ends inside its prefix or its checksum"))
}

fn product(a: u64, b: u64) -> Result<u64, FormatError> {
    a.checked_mul(b)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)
}

fn sum<const N: usize>(terms: [u64; N]) -> Result<u64, FormatError> {
    terms
        .into_iter()
        .try_fold(0, u64::checked_add)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)
}

fn internal(detail: &str) -> FormatError {
    FormatError::Internal(format!("an Extensible Array {detail}"))
}

/// Checked block and page geometry from Extensible Array indexing information.
///
/// The five creation fields are stored in a version 4 data layout message before the header
/// address, including when that address is undefined. The fields are defined in "The Data Layout
/// Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayCreationParameters {
    bits: ExtensibleArrayBitGeometry,
    inline_elements: u8,
    min_data_blocks: NonZeroU8,
    min_elements: NonZeroU8,
}

impl ExtensibleArrayCreationParameters {
    /// Parses the first five bytes in data layout message order.
    ///
    /// The fields are the maximum element bit count, inline element count, minimum data block
    /// pointer count, minimum data block element count, and page exponent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` has fewer than five bytes, or
    /// [`FormatError::InvalidExtensibleArrayGeometry`] if the fields are outside the supported
    /// block and page geometry.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut fields = bytes::Fields::new(data, 0);
        let [
            max_index_bits,
            inline_elements,
            min_data_blocks,
            min_elements,
            page_bits,
        ] = fields.array()?;
        if !(1..=MAX_INDEX_BITS).contains(&max_index_bits)
            || !min_elements.is_power_of_two()
            || !min_data_blocks.is_power_of_two()
            || min_data_blocks < MIN_DATA_BLOCKS
            || page_bits > max_index_bits.min(MAX_PAGE_BITS)
        {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        let min_elements_bits = min_elements.trailing_zeros();
        let first_super_block = 2 * min_data_blocks.trailing_zeros();
        let super_blocks = (u32::from(max_index_bits) + 1)
            .checked_sub(min_elements_bits)
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        if first_super_block > super_blocks
            || min_elements_bits + first_super_block / 2 > u32::from(page_bits)
        {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        Ok(Self {
            bits: ExtensibleArrayBitGeometry::parse(max_index_bits, page_bits)?,
            inline_elements,
            min_data_blocks: NonZeroU8::new(min_data_blocks)
                .ok_or(FormatError::InvalidExtensibleArrayGeometry)?,
            min_elements: NonZeroU8::new(min_elements)
                .ok_or(FormatError::InvalidExtensibleArrayGeometry)?,
        })
    }

    /// Returns the writer's default creation parameters.
    ///
    /// The maximum element bit count is 32, the index block holds four elements, the minimum
    /// data block pointer count is four, the minimum data block size is 16 elements, and a page
    /// holds 1024 elements.
    pub fn canonical() -> Self {
        // These constant parameters satisfy the checked block and page geometry.
        Self::parse(&[
            EA_MAX_NELMTS_BITS,
            EA_IDX_BLK_ELMTS,
            EA_SUPER_BLK_MIN_DATA_PTRS,
            EA_MIN_DBLK_NELMTS,
            EA_MAX_DBLK_NELMTS_BITS,
        ])
        .expect("canonical Extensible Array geometry is representable")
    }

    /// Returns the creation fields in data layout message order.
    pub fn encode(self) -> [u8; 5] {
        [
            self.bits.max_index_bits().get(),
            self.inline_elements,
            self.min_data_blocks.get(),
            self.min_elements.get(),
            self.bits.page_bits.0,
        ]
    }

    /// Returns the maximum element bit count and data block page geometry.
    pub fn bit_geometry(self) -> ExtensibleArrayBitGeometry {
        self.bits
    }

    /// Returns the number of elements stored directly in the index block.
    pub fn inline_elements(self) -> u8 {
        self.inline_elements
    }

    /// Returns the minimum number of data block pointers in a super block.
    pub fn min_data_blocks(self) -> NonZeroU8 {
        self.min_data_blocks
    }

    /// Returns the minimum number of elements in a data block.
    pub fn min_elements(self) -> NonZeroU8 {
        self.min_elements
    }

    // `H5EA_SBLK_FIRST_IDX` and `H5EA_SIZEOF_OFFSET_BITS` (`H5EApkg.h`, HDF5 2.2.0)
    // determine the first super block and block-offset width.
    fn layout(self, encoding: ElementEncoding) -> Layout {
        Layout {
            encoding,
            inline_elements: self.inline_elements,
            min_elements_bits: self.min_elements.get().trailing_zeros(),
            page_bits: u32::from(self.bits.page_bits.0),
            block_offset_len: self.bits.max_index_bits().get().div_ceil(8),
            first_super_block: 2 * self.min_data_blocks.get().trailing_zeros(),
            super_blocks: u32::from(self.bits.max_index_bits().get()) + 1
                - self.min_elements.get().trailing_zeros(),
        }
    }
}

/// Geometry for serializing an empty Extensible Array header and index block.
///
/// The serialized header is followed immediately by its index block. Both structures include
/// their checksums. The index block's chunk and child block addresses are undefined. The version 0
/// header and index block are defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_format::__private::{
///     ExtensibleArrayCreationParameters, ExtensibleArrayInitialization, FormatWidths,
/// };
///
/// let creation = ExtensibleArrayCreationParameters::parse(&[24, 8, 2, 8, 8])?;
/// let initialization = ExtensibleArrayInitialization::new(
///     creation,
///     16,
///     FormatWidths::from_sizes(8, 8)?,
///     false,
/// )?;
/// let bytes = initialization.encode_at(StoredAddress::new(1024))?;
/// assert_eq!(bytes.len() as u64, initialization.allocation_len());
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub struct ExtensibleArrayInitialization {
    creation: ExtensibleArrayCreationParameters,
    widths: FormatWidths,
    encoding: ChunkElementEncoding,
    layout: Layout,
    index_block_len: usize,
}

impl ExtensibleArrayInitialization {
    /// Uses version 4 chunk-record encoding for an empty index.
    ///
    /// `chunk_bytes` is the chunk size before filtering. For filtered chunks, the chunk-size
    /// field width is derived from that size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the element encoding or block
    /// geometry cannot be represented, or [`FormatError::ValueTooLargeForPlatform`] if the index
    /// block length exceeds [`usize::MAX`].
    pub fn new(
        creation: ExtensibleArrayCreationParameters,
        chunk_bytes: u64,
        widths: FormatWidths,
        has_filters: bool,
    ) -> Result<Self, FormatError> {
        let encoding =
            chunk_record::chunk_element_encoding(chunk_bytes, widths.offsets, has_filters);
        let element_size = u8::try_from(encoding.elem_size)
            .map_err(|_error| FormatError::InvalidExtensibleArrayGeometry)?;
        let layout = creation.layout(
            ElementEncoding::new(widths.offsets, encoding.client_id, element_size)
                .map_err(|_error| FormatError::InvalidExtensibleArrayGeometry)?,
        );
        Ok(Self {
            creation,
            widths,
            encoding,
            layout,
            index_block_len: layout.index_block_len()?.to_usize()?,
        })
    }

    /// Returns the block geometry defined by the retained creation parameters.
    ///
    /// Includes the geometry of unallocated blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset
    /// exceeds [`u64::MAX`].
    pub fn geometry(&self) -> Result<ExtensibleArrayGeometry, FormatError> {
        ExtensibleArrayGeometry::from_layout(self.layout)
    }

    /// Returns the canonical chunk-record encoding used to serialize the empty index.
    pub fn element_encoding(&self) -> ChunkElementEncoding {
        self.encoding
    }

    /// Returns the combined size of the header and index block, including their checksums.
    pub fn allocation_len(&self) -> u64 {
        self.header_len() as u64 + self.index_block_len as u64
    }

    /// Serializes the header at the base-relative `address`, followed by its index block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if allocation address arithmetic
    /// overflows, [`FormatError::AddressTooLarge`] if the allocation extends past the encoded
    /// address domain, or [`FormatError::ValueTooLargeForPlatform`] if the allocation length
    /// exceeds [`usize::MAX`].
    pub fn encode_at(&self, address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        let index_address = StoredAddress::new(
            address
                .get()
                .checked_add(self.header_len() as u64)
                .ok_or(FormatError::InvalidExtensibleArrayGeometry)?,
        );
        let end = address
            .get()
            .checked_add(self.allocation_len())
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        if end > StoredAddress::undefined(self.widths.offsets).get() {
            return Err(FormatError::AddressTooLarge {
                address: end,
                offset_size: self.widths.offsets.get(),
            });
        }
        let mut header = Vec::with_capacity(self.allocation_len().to_usize()?);
        header.extend_from_slice(&HEADER_SIGNATURE);
        header.push(VERSION);
        header.push(self.encoding.client_id);
        header.push(
            u8::try_from(self.encoding.elem_size)
                .map_err(|_error| FormatError::InvalidExtensibleArrayGeometry)?,
        );
        let [max_bits, inline, min_blocks, min_elements, page_bits] = self.creation.encode();
        header.extend_from_slice(&[max_bits, inline, min_elements, min_blocks, page_bits]);
        for statistic in [0, 0, 0, 0, 0, u64::from(inline)] {
            bytes::write_length(&mut header, statistic, self.widths.lengths);
        }
        bytes::try_write_decoded_address(
            &mut header,
            DecodedStoredAddress::Defined(index_address),
            self.widths.offsets,
        )?;
        let checksum = checksum::jenkins_lookup3(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        let index_start = header.len();
        header.extend_from_slice(&INDEX_BLOCK_SIGNATURE);
        header.push(VERSION);
        header.push(self.encoding.client_id);
        bytes::try_write_decoded_address(
            &mut header,
            DecodedStoredAddress::Defined(address),
            self.widths.offsets,
        )?;
        for _ in 0..inline {
            chunk_record::write_undefined_element(
                &mut header,
                self.widths.offsets,
                self.encoding.client_id != EA_CLIENT_ID_UNFILTERED,
                self.encoding.chunk_size_bytes,
            );
        }
        let addresses = self.layout.direct_data_blocks()
            + u64::from(self.layout.super_blocks - self.layout.first_super_block);
        for _ in 0..addresses {
            bytes::try_write_decoded_address(
                &mut header,
                DecodedStoredAddress::Undefined,
                self.widths.offsets,
            )?;
        }
        let checksum = checksum::jenkins_lookup3(&header[index_start..]);
        header.extend_from_slice(&checksum.to_le_bytes());
        Ok(header)
    }

    fn header_len(&self) -> usize {
        ExtensibleArrayHeader::serialized_size_with_widths(self.widths)
    }
}

/// An Extensible Array header, signature `EAHD`, version 0.
///
/// The maximum element bit count is in `1..=64`, and the page exponent is at most that count
/// and at most 63. The page element count fits a [`u64`], and the block offset is 1 to 8 bytes
/// wide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayHeader {
    /// The client ID: 0 for unfiltered chunks and 1 for filtered ones.
    pub client_id: u8,
    /// The width in bytes of one element.
    pub element_size: u8,
    bit_geometry: ExtensibleArrayBitGeometry,
    /// The number of elements the index block holds.
    pub idx_blk_elmts: u8,
    /// The number of elements in the smallest data block.
    pub min_dblk_nelmts: u8,
    /// The fewest data block addresses a super block holds.
    pub super_blk_min_data_ptrs: u8,
    /// One more than the highest element index set, the "Max Index Set" field.
    pub max_idx_set: u64,
    /// The address of the index block.
    pub index_block_address: StoredAddress,
}

impl ExtensibleArrayHeader {
    /// Parses the Extensible Array header at `offset` in `file_data`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the header runs past the end of `file_data`,
    /// [`FormatError::ChunkedReadError`] if the signature is not `EAHD` or the version is not 0,
    /// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not
    /// 2, 4, or 8, and [`FormatError::ChecksumMismatch`] if the checksum does not match.
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if the maximum element bit count is outside
    /// `1..=64`, the page element count does not fit a [`u64`], or the page exponent exceeds the
    /// maximum element bit count.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        Self::parse_with_widths(
            file_data,
            offset,
            FormatWidths::from_sizes(offset_size, length_size)?,
        )
    }

    /// Parses the Extensible Array header at `offset` using validated format widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the header runs past the end of `file_data`,
    /// [`FormatError::ChunkedReadError`] if the signature is not `EAHD` or the version is not 0,
    /// [`FormatError::ChecksumMismatch`] if the checksum does not match, and
    /// [`FormatError::InvalidChunkGeometry`] if the stored bit geometry is invalid.
    pub fn parse_with_widths(
        file_data: &[u8],
        offset: usize,
        widths: FormatWidths,
    ) -> Result<Self, FormatError> {
        // EAHD: the signature (4), eight 1-byte fields, six statistics of `length_size` bytes,
        // the index block address, and the checksum (4).
        let min_size = Self::serialized_size_with_widths(widths);
        if min_size > file_data.len() || offset > file_data.len() - min_size {
            return Err(FormatError::UnexpectedEof {
                expected: offset.saturating_add(min_size),
                available: file_data.len(),
            });
        }

        let d = &file_data[offset..];
        if &d[0..4] != b"EAHD" {
            return Err(FormatError::ChunkedReadError(
                "invalid Extensible Array header signature".into(),
            ));
        }

        let version = d[4];
        if version != 0 {
            return Err(FormatError::ChunkedReadError(format!(
                "unsupported Extensible Array header version: {version}"
            )));
        }

        let client_id = d[5];
        let element_size = d[6];
        let max_nelmts_bits = d[7];
        let idx_blk_elmts = d[8];
        let min_dblk_nelmts = d[9];
        let super_blk_min_data_ptrs = d[10];
        let max_dblk_nelmts_bits = d[11];
        let bit_geometry =
            ExtensibleArrayBitGeometry::parse(max_nelmts_bits, max_dblk_nelmts_bits)?;

        let mut pos = 12;
        // The six statistics, in the order the header stores them:
        //   [0] `nsuper_blks`   [1] `super_blk_size`   [2] `ndata_blks`
        //   [3] `data_blk_size` [4] `max_idx_set`      [5] `nelmts`
        // The parser reads `max_idx_set`, one more than the highest element index set, which
        // bounds the elements in use. `nelmts` counts every slot of the allocated blocks.
        let ls = usize::from(widths.lengths.get());
        pos += 4 * ls; // skip [0]..[3]
        let max_idx_set = bytes::read_length_width(d, pos, widths.lengths)?; // [4]
        pos += ls;
        pos += ls; // skip [5] nelmts
        let index_block_address =
            StoredAddress::new(bytes::read_offset_width(d, pos, widths.offsets)?);

        crate::checksum::verify_trailing(&d[..min_size])?;

        Ok(ExtensibleArrayHeader {
            client_id,
            element_size,
            bit_geometry,
            idx_blk_elmts,
            min_dblk_nelmts,
            super_blk_min_data_ptrs,
            max_idx_set,
            index_block_address,
        })
    }

    /// Returns the length in bytes of a header with `offset_size`-byte addresses and
    /// `length_size`-byte lengths.
    pub fn serialized_size(offset_size: u8, length_size: u8) -> usize {
        4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * length_size as usize + offset_size as usize + 4
    }

    /// Returns the header length for validated format widths.
    pub fn serialized_size_with_widths(widths: FormatWidths) -> usize {
        4 + 8 + 6 * usize::from(widths.lengths.get()) + usize::from(widths.offsets.get()) + 4
    }

    /// Returns the parsed maximum element bit count and data block page exponent.
    pub fn bit_geometry(&self) -> ExtensibleArrayBitGeometry {
        self.bit_geometry
    }

    /// Returns the "Max Nelmts Bits" field, in `1..=64`.
    pub fn max_index_bits(&self) -> MaxIndexBits {
        self.bit_geometry.max_index_bits()
    }

    /// Returns the number of elements in a data block page, a nonzero power of two.
    pub fn page_nelmts(&self) -> NonZeroU64 {
        self.bit_geometry.page_nelmts()
    }

    /// Returns the width in bytes of a block offset, in `1..=8`.
    pub fn block_offset_width(&self) -> BlockOffsetWidth {
        self.bit_geometry.block_offset_width()
    }
}

/// The parsed maximum element bit count and data block page exponent.
///
/// The maximum element bit count is in `1..=64`, and the page exponent is in `0..=63` and at
/// most that count. The page element count is a nonzero power of two representable by a [`u64`].
/// The block-offset width is the maximum element bit count rounded up to whole bytes.
/// libhdf5 computes the block-offset width with `H5EA_SIZEOF_OFFSET_BITS` (`H5EApkg.h`,
/// HDF5 2.2.0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayBitGeometry {
    max_index_bits: MaxIndexBits,
    page_bits: PageBits,
}

impl ExtensibleArrayBitGeometry {
    fn for_writer() -> Self {
        ExtensibleArrayCreationParameters::canonical().bit_geometry()
    }

    /// Parses the maximum element bit count and the data block page exponent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if `max_index_bits` is outside `1..=64`,
    /// the page count does not fit a [`u64`], or `page_bits` exceeds `max_index_bits`.
    fn parse(max_index_bits: u8, page_bits: u8) -> Result<Self, FormatError> {
        let max_index_bits = MaxIndexBits::parse(max_index_bits)?;
        let page_bits = PageBits::parse(page_bits)?;
        if page_bits.0 > max_index_bits.get() {
            return Err(FormatError::InvalidChunkGeometry(
                "Extensible Array page exponent exceeds maximum element bit count",
            ));
        }
        Ok(Self {
            max_index_bits,
            page_bits,
        })
    }

    /// Returns the "Max Nelmts Bits" field, in `1..=64`.
    pub fn max_index_bits(self) -> MaxIndexBits {
        self.max_index_bits
    }

    /// Returns the number of elements in a data block page, a nonzero power of two.
    pub fn page_nelmts(self) -> NonZeroU64 {
        self.page_bits.nelmts()
    }

    /// Returns the maximum element bit count rounded up to a block-offset byte width.
    pub fn block_offset_width(self) -> BlockOffsetWidth {
        match self.max_index_bits.get() {
            1..=8 => BlockOffsetWidth::One,
            9..=16 => BlockOffsetWidth::Two,
            17..=24 => BlockOffsetWidth::Three,
            25..=32 => BlockOffsetWidth::Four,
            33..=40 => BlockOffsetWidth::Five,
            41..=48 => BlockOffsetWidth::Six,
            49..=56 => BlockOffsetWidth::Seven,
            _ => BlockOffsetWidth::Eight,
        }
    }
}

/// An Extensible Array header's "Max Nelmts Bits" field, in `1..=64`.
///
/// The field is defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct MaxIndexBits(NonZeroU8);

impl MaxIndexBits {
    /// Parses the maximum element bit count supported by the reader and writer.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if `bits` is outside `1..=64`.
    fn parse(bits: u8) -> Result<Self, FormatError> {
        NonZeroU8::new(bits)
            .filter(|bits| bits.get() <= 64)
            .map(Self)
            .ok_or(FormatError::InvalidChunkGeometry(
                "Extensible Array maximum element bit count must be in 1..=64",
            ))
    }

    /// Returns the bit count as a [`u8`], in `1..=64`.
    pub fn get(self) -> u8 {
        self.0.get()
    }
}

/// The base 2 logarithm of the number of elements in a data block page, in `0..=63`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
struct PageBits(u8);

impl PageBits {
    /// Parses a page exponent whose element count fits a [`u64`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if `bits` exceeds 63.
    fn parse(bits: u8) -> Result<Self, FormatError> {
        if bits > 63 {
            return Err(FormatError::InvalidChunkGeometry(
                "Extensible Array page element count does not fit u64",
            ));
        }
        Ok(Self(bits))
    }

    /// Returns the number of elements in a page, a nonzero power of two.
    fn nelmts(self) -> NonZeroU64 {
        // The constructor admits only exponents at most 63, so this power of two is nonzero.
        NonZeroU64::new(1u64 << self.0).expect("a parsed page exponent produces a nonzero count")
    }
}

/// The width in bytes of an Extensible Array block offset, in `1..=8`.
///
/// The block offset is defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum BlockOffsetWidth {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
}

impl BlockOffsetWidth {
    /// Returns the number of bytes in the field, in `1..=8`.
    pub fn bytes(self) -> usize {
        self as usize
    }

    /// Reads a little-endian block offset from the start of `data`.
    ///
    /// Reads [`bytes`](Self::bytes) bytes and ignores any remaining bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the field width.
    pub fn read(self, data: &[u8]) -> Result<u64, FormatError> {
        let size = self.bytes();
        let field = data.get(..size).ok_or(FormatError::UnexpectedEof {
            expected: size,
            available: data.len(),
        })?;
        let mut bytes = [0u8; 8];
        bytes[..size].copy_from_slice(field);
        Ok(u64::from_le_bytes(bytes))
    }
}

/// The number and the size of the data blocks of each super block of an Extensible Array, from
/// the creation parameters of its header.
///
/// Super block `i` has `2^floor(i/2)` data blocks of `min_dblk_nelmts * 2^ceil(i/2)` elements
/// each, as `H5EA__hdr_init` computes them (`H5EAhdr.c`, HDF5 2.2.0):
///
/// ```text
/// SB0: 1 x 16   SB1: 1 x 32   SB2: 2 x 32   SB3: 2 x 64   SB4: 4 x 64 ...
/// ```
///
/// The index block stores direct data block addresses for the first
/// `2 * log2(super_blk_min_data_ptrs)` super blocks, followed by addresses of the remaining super
/// blocks. The direct address count is `2 * (super_blk_min_data_ptrs - 1)`, as
/// `H5EA__iblock_alloc` computes it (`H5EAiblock.c`, HDF5 2.2.0).
#[derive(Debug, Clone)]
pub struct ExtensibleArrayGeometry {
    /// The `(ndblks, dblk_nelmts)` of each super block.
    pub(crate) sblks: Vec<(u64, u64)>,
    /// The number of elements in each data block the index block addresses.
    pub(crate) direct_dblk_nelmts: Vec<u64>,
    /// The number of super block addresses in the index block.
    pub(crate) nsblk_addrs: usize,
    /// The index in [`Self::sblks`] of the first super block addressed by the index block.
    pub(crate) first_indirect_sblk: usize,
}

impl ExtensibleArrayGeometry {
    /// Returns the geometry of the super block the `j`th super block address of the index block
    /// points at, in an array whose pages hold `page_nelmts` elements.
    ///
    /// `j` counts the super block addresses, and the first `first_indirect_sblk` super blocks have
    /// none.
    pub(crate) fn super_block_at(&self, j: usize, page_nelmts: u64) -> SuperBlockGeometry {
        let (ndblks, dblk_nelmts) = self.sblks[self.first_indirect_sblk + j];
        SuperBlockGeometry {
            ndblks,
            blocks: DataBlockGeometry {
                dblk_nelmts,
                page_nelmts,
            },
        }
    }

    /// Derives block geometry from the header's creation parameters.
    ///
    /// `offsets` is the file's address width, used to interpret the element encoding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayClient`] if the client ID is 2 or more, and
    /// [`FormatError::InvalidExtensibleArrayGeometry`] if a filtered element width or creation
    /// parameter is out of range, or the block geometry overflows [`u64`].
    pub fn from_header(
        &ExtensibleArrayHeader {
            client_id,
            element_size,
            bit_geometry,
            idx_blk_elmts,
            min_dblk_nelmts,
            super_blk_min_data_ptrs,
            max_idx_set: _,
            index_block_address: _,
        }: &ExtensibleArrayHeader,
        offsets: OffsetWidth,
    ) -> Result<Self, FormatError> {
        let encoding =
            ElementEncoding::new(offsets, client_id, element_size).map_err(
                |error| match error {
                    InvalidElementEncoding::Client(id) => {
                        FormatError::InvalidExtensibleArrayClient(id)
                    }
                    InvalidElementEncoding::EntrySize => {
                        FormatError::InvalidExtensibleArrayGeometry
                    }
                },
            )?;
        let layout = Layout::new(
            encoding,
            bit_geometry.max_index_bits().get(),
            idx_blk_elmts,
            min_dblk_nelmts,
            super_blk_min_data_ptrs,
            bit_geometry.page_bits.0,
        )?;
        Self::from_layout(layout)
    }

    /// Returns the `(ndblks, dblk_nelmts)` of each super block: the number of its data blocks and
    /// the number of elements in each.
    pub fn sblks(&self) -> &[(u64, u64)] {
        &self.sblks
    }

    /// Returns the number of elements in each data block the index block addresses.
    pub fn direct_dblk_nelmts(&self) -> &[u64] {
        &self.direct_dblk_nelmts
    }

    /// Returns the number of super block addresses in the index block.
    pub fn nsblk_addrs(&self) -> usize {
        self.nsblk_addrs
    }

    /// Returns the index in [`sblks`](Self::sblks) of the super block the first super block
    /// address points at.
    pub fn first_indirect_sblk(&self) -> usize {
        self.first_indirect_sblk
    }

    /// Derives the block tables from a checked layout.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset
    /// exceeds [`u64::MAX`].
    fn from_layout(layout: Layout) -> Result<Self, FormatError> {
        let mut sblks = Vec::with_capacity(layout.super_blocks.to_usize()?);
        for index in 0..layout.super_blocks {
            let block = layout.super_block(index)?;
            sblks.push((block.blocks, block.block_elements));
        }
        let mut direct_dblk_nelmts = Vec::new();
        for block in layout.direct_blocks() {
            direct_dblk_nelmts.push(block?.elements);
        }
        Ok(Self {
            sblks,
            direct_dblk_nelmts,
            nsblk_addrs: (layout.super_blocks - layout.first_super_block).to_usize()?,
            first_indirect_sblk: layout.first_super_block.to_usize()?,
        })
    }
}

/// The number of elements in an Extensible Array data block, and the number in one page.
///
/// A data block of more elements than a page is paged, and one of exactly a page is not, as in
/// `H5EA__dblock_alloc` (`H5EAdblock.c`, HDF5 2.2.0). The rule holds for a data block the
/// index block addresses as for one a super block addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataBlockGeometry {
    /// The number of elements in the data block.
    pub dblk_nelmts: u64,
    /// The number of elements in a page.
    pub page_nelmts: u64,
}

impl DataBlockGeometry {
    /// Returns whether the data block is stored in pages.
    pub const fn is_paged(self) -> bool {
        self.dblk_nelmts > self.page_nelmts
    }

    /// Returns the number of pages in the data block, or 0 if it is not paged.
    pub const fn npages(self) -> u64 {
        if self.is_paged() {
            self.dblk_nelmts / self.page_nelmts
        } else {
            0
        }
    }
}

/// The number of data blocks of an Extensible Array super block (`EASB`), and their geometry.
///
/// A super block whose data blocks are paged stores a page-init bitmap between its block offset
/// and its data block addresses, so the position of every address in the block depends on
/// [`DataBlockGeometry::is_paged`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperBlockGeometry {
    /// The number of data block addresses in the super block.
    pub ndblks: u64,
    /// The geometry of each of its data blocks.
    pub blocks: DataBlockGeometry,
}

impl SuperBlockGeometry {
    /// Returns the length in bytes of the page-init bitmap, or 0 if the data blocks are not paged.
    ///
    /// The length is `ceil(npages / 8)` bytes per data block. The bits form one stream indexed by
    /// `dblk_local * npages + page`, so the bits of data block `k` begin at bit `k * npages`.
    ///
    /// The length is at most `ndblks * dblk_nelmts`, and every caller computes that product or
    /// the length of the enclosing block first, so the `u64` product does not overflow.
    pub const fn bitmap_size(self) -> u64 {
        self.ndblks * self.blocks.npages().div_ceil(8)
    }
}

/// The slots of an Extensible Array that hold a chunk.
///
/// [`extensible_array_stats`] counts the blocks that hold a chunk and [`build_extensible_array_at`]
/// builds them, and the length the first computes must equal the length the second builds. Both
/// test the same `SlotOccupancy`.
#[derive(Clone, Copy)]
pub enum SlotOccupancy<'a> {
    /// Slots `0..len` are occupied, and no slot past them.
    Dense(u64),
    /// The occupied slots of an index.
    Sparse(&'a IndexSlots<'a>),
    /// The occupied slots as slot numbers in ascending order, for a caller that sizes an index
    /// before its chunks have addresses.
    Slots(&'a [u64]),
}

impl SlotOccupancy<'_> {
    /// Returns whether any of the `count` slots from `start` holds a chunk, and so whether the
    /// block over them is allocated.
    ///
    /// libhdf5 allocates a data block when it writes the first chunk into it (`H5EA.c`,
    /// HDF5 2.2.0), so an index grows with its chunks and not with the maximum shape.
    fn any_occupied(&self, start: u64, count: u64) -> bool {
        match self {
            Self::Dense(len) => start < *len,
            Self::Sparse(slots) => slots.any_occupied(start, count),
            Self::Slots(sorted) => {
                let end = start.saturating_add(count);
                let from = sorted.partition_point(|&s| s < start);
                sorted.get(from).is_some_and(|&s| s < end)
            }
        }
    }
}

/// Appends `addr` to `buf` as a little-endian field of `offset_size` bytes.
pub fn write_stored_address(buf: &mut Vec<u8>, addr: StoredAddress, offset_size: OffsetWidth) {
    bytes::write_offset(buf, addr.get(), offset_size);
}

/// Appends a block's offset in the array's element space as a little-endian field of `width`
/// bytes.
///
/// # Panics
///
/// Panics if `block_offset` does not fit the number of bytes specified by `width`.
fn write_block_offset(buf: &mut Vec<u8>, block_offset: u64, width: BlockOffsetWidth) {
    let blk_off_size = width.bytes();
    let bytes = block_offset.to_le_bytes();
    assert!(
        bytes[blk_off_size..].iter().all(|&byte| byte == 0),
        "block offset {block_offset} does not fit a {blk_off_size}-byte field"
    );
    buf.extend_from_slice(&bytes[..blk_off_size]);
}

/// Builds the data block (`EADB`) of the elements in slots `elem_start..elem_start + dblk_nelmts`,
/// with `block_offset_rel` as its block offset.
///
/// An unoccupied slot stores the undefined address. A block of more than
/// [`bits.page_nelmts()`](ExtensibleArrayBitGeometry::page_nelmts) elements is paged: its prefix
/// has its own checksum, and each page has a checksum.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the stored size of a chunk does not fit the chunk size
/// field, and [`FormatError::ValueTooLargeForPlatform`] if the page element count does not fit a
/// [`usize`].
///
/// # Panics
///
/// Panics if `block_offset_rel` does not fit the block-offset width in `bits`.
#[allow(clippy::too_many_arguments)]
pub fn encode_data_block(
    slots: &IndexSlots<'_>,
    elem_start: usize,
    dblk_nelmts: usize,
    block_offset_rel: u64,
    ea_address: StoredAddress,
    offset_size: OffsetWidth,
    has_filters: bool,
    chunk_size_bytes: usize,
    client_id: u8,
    bits: ExtensibleArrayBitGeometry,
) -> Result<Vec<u8>, FormatError> {
    let page_nelmts = bits.page_nelmts().get().to_usize()?;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EADB");
    buf.push(0); // version
    buf.push(client_id);
    write_stored_address(&mut buf, ea_address, offset_size);
    write_block_offset(&mut buf, block_offset_rel, bits.block_offset_width());

    // The counts are `usize` loop bounds, and the page arithmetic below stays in `usize`.
    let blocks = DataBlockGeometry {
        dblk_nelmts: dblk_nelmts as u64,
        page_nelmts: page_nelmts as u64,
    };
    let write_elements = |buf: &mut Vec<u8>, start: usize, count: usize| {
        for slot in start..start + count {
            match slots.at(slot) {
                Some(chunk) => chunk_record::write_chunk_element(
                    buf,
                    chunk,
                    offset_size,
                    has_filters,
                    chunk_size_bytes,
                )?,
                None => chunk_record::write_undefined_element(
                    buf,
                    offset_size,
                    has_filters,
                    chunk_size_bytes,
                ),
            }
        }
        Ok::<(), FormatError>(())
    };
    if !blocks.is_paged() {
        // Non-paged: elements inline, single checksum.
        write_elements(&mut buf, elem_start, dblk_nelmts)?;
        let cks = checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&cks.to_le_bytes());
    } else {
        // Paged: the prefix and its checksum, then every page, as the C library allocates them
        // (`H5EAdblock.c`, HDF5 2.2.0).
        let header_cks = checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&header_cks.to_le_bytes());

        for page in 0..dblk_nelmts / page_nelmts {
            let page_at = buf.len();
            write_elements(&mut buf, elem_start + page * page_nelmts, page_nelmts)?;
            let page_cks = checksum::jenkins_lookup3(&buf[page_at..]);
            buf.extend_from_slice(&page_cks.to_le_bytes());
        }
    }
    Ok(buf)
}

/// Builds the super block (`EASB`) that addresses `dblk_addrs`, with `block_offset_rel` as its
/// block offset.
///
/// A non-empty `page_bitmap` is the page-init bitmap of paged data blocks, which the super block
/// stores between its block offset and its data block addresses.
///
/// # Panics
///
/// Panics if `block_offset_rel` does not fit the number of bytes returned by
/// [`width.bytes()`](BlockOffsetWidth::bytes).
pub fn encode_super_block(
    ea_address: StoredAddress,
    block_offset_rel: u64,
    page_bitmap: &[u8],
    dblk_addrs: &[StoredAddress],
    offset_size: OffsetWidth,
    client_id: u8,
    width: BlockOffsetWidth,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EASB");
    buf.push(0); // version
    buf.push(client_id);
    write_stored_address(&mut buf, ea_address, offset_size);
    write_block_offset(&mut buf, block_offset_rel, width);
    buf.extend_from_slice(page_bitmap);
    for &addr in dblk_addrs {
        write_stored_address(&mut buf, addr, offset_size);
    }
    let cks = checksum::jenkins_lookup3(&buf);
    buf.extend_from_slice(&cks.to_le_bytes());
    buf
}

/// Returns the length in bytes of an index block (`EAIB`): the prefix, `inline_elmts` elements,
/// the data block and super block addresses, and the checksum.
///
/// [`build_extensible_array_at`] sizes the index block with it.
pub(crate) fn index_block_len(
    offset_width: OffsetWidth,
    inline_elmts: usize,
    elem_size: usize,
    ndblk_addrs: usize,
    nsblk_addrs: usize,
) -> usize {
    let os = usize::from(offset_width.get());
    4 + 1 + 1 + os // signature + version + client id + header address
        + inline_elmts * elem_size // inline element slots (always all written)
        + ndblk_addrs * os // direct data-block addresses
        + nsblk_addrs * os // super-block addresses
        + 4 // checksum
}

/// The six statistics of an Extensible Array header, in the order the header stores them.
///
/// An in-place writer that appends to an array rewrites them, and [`extensible_array_layout`] takes
/// the length of the blocks after the index block from `super_blk_size` and `data_blk_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensibleArrayStats {
    /// The number of super blocks allocated.
    pub nsuper_blks: u64,
    /// The total length in bytes of the super blocks.
    pub super_blk_size: u64,
    /// The number of data blocks allocated.
    pub ndata_blks: u64,
    /// The total length in bytes of the data blocks.
    pub data_blk_size: u64,
    /// One more than the highest element index set, the "Max Index Set" field.
    pub max_idx_set: u64,
    /// The number of element slots in the index block and the allocated data blocks.
    pub nelmts: u64,
}

/// Returns the length in bytes of a data block (`EADB`) with the geometry `blocks`, paged or not.
pub(crate) fn data_block_len(
    blocks: DataBlockGeometry,
    elem_size: usize,
    offset_width: OffsetWidth,
    blk_off_size: BlockOffsetWidth,
) -> u64 {
    let prefix = (4 + 1 + 1 + u64::from(offset_width.get())) + blk_off_size.bytes() as u64;
    if blocks.is_paged() {
        // Paged: the prefix and its checksum, then every page and its checksum.
        prefix + 4 + blocks.npages() * (blocks.page_nelmts * elem_size as u64 + 4)
    } else {
        prefix + blocks.dblk_nelmts * elem_size as u64 + 4
    }
}

/// Returns the length in bytes of a super block (`EASB`): the prefix, the page-init bitmap of
/// paged data blocks, the data block addresses, and the checksum.
///
/// The length is computed in `u64`, so the counts are not narrowed to a 32-bit `usize` before
/// they are multiplied.
pub(crate) fn super_block_len(
    geom: SuperBlockGeometry,
    offset_width: OffsetWidth,
    blk_off_size: BlockOffsetWidth,
) -> u64 {
    let os = u64::from(offset_width.get());
    let header = 4 + 1 + 1 + os + blk_off_size.bytes() as u64;
    header + geom.bitmap_size() + geom.ndblks * os + 4
}

/// Returns the header statistics of the array whose occupied slots are `occupancy`, from the
/// blocks [`build_extensible_array_at`] allocates for them.
///
/// The caller passes `idx_blk_elmts`, `bits`, and `geom` from the same header.
pub fn extensible_array_stats(
    geom: &ExtensibleArrayGeometry,
    idx_blk_elmts: u64,
    bits: ExtensibleArrayBitGeometry,
    elem_size: usize,
    offset_size: OffsetWidth,
    max_idx_set: u64,
    occupancy: SlotOccupancy<'_>,
) -> ExtensibleArrayStats {
    let page_nelmts = bits.page_nelmts().get();
    let blk_off_size = bits.block_offset_width();
    let mut s = ExtensibleArrayStats {
        nsuper_blks: 0,
        super_blk_size: 0,
        ndata_blks: 0,
        data_blk_size: 0,
        max_idx_set,
        nelmts: idx_blk_elmts,
    };
    let mut elem = idx_blk_elmts;
    for &dn in &geom.direct_dblk_nelmts {
        if occupancy.any_occupied(elem, dn) {
            let blocks = DataBlockGeometry {
                dblk_nelmts: dn,
                page_nelmts,
            };
            s.ndata_blks += 1;
            s.data_blk_size += data_block_len(blocks, elem_size, offset_size, blk_off_size);
            s.nelmts += dn;
        }
        elem += dn;
    }
    for j in 0..geom.nsblk_addrs {
        let sb = geom.super_block_at(j, page_nelmts);
        let (ndblks, dn) = (sb.ndblks, sb.blocks.dblk_nelmts);
        let span = ndblks * dn;
        if occupancy.any_occupied(elem, span) {
            s.nsuper_blks += 1;
            s.super_blk_size += super_block_len(sb, offset_size, blk_off_size);
            let mut le = elem;
            for _ in 0..ndblks {
                if occupancy.any_occupied(le, dn) {
                    s.ndata_blks += 1;
                    s.data_blk_size +=
                        data_block_len(sb.blocks, elem_size, offset_size, blk_off_size);
                    s.nelmts += dn;
                }
                le += dn;
            }
        }
        elem += span;
    }
    s
}

/// The layout of an Extensible Array that does not depend on its address: the element encoding,
/// the geometry, and the lengths.
///
/// [`extensible_array_len`] computes the layout without building the array.
/// [`build_extensible_array_at`] builds from the geometry of the layout and walks the blocks
/// itself, and a debug assertion checks its length against the layout's.
pub struct ExtensibleArrayLayout {
    encoding: ChunkElementEncoding,
    bits: ExtensibleArrayBitGeometry,
    geom: ExtensibleArrayGeometry,
    /// The number of elements in the index block, `idx_blk_elmts`.
    inline: usize,
    /// The length of the header in bytes.
    header_len: usize,
    index_block_len: usize,
    /// The header statistics.
    pub stats: ExtensibleArrayStats,
    /// The length of the array in bytes.
    total_len: u64,
}

/// Returns the layout of the Extensible Array of `num_slots` slots whose occupied slots are
/// `occupancy`.
pub fn extensible_array_layout(
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> ExtensibleArrayLayout {
    let encoding = chunk_record::chunk_element_encoding(chunk_bytes, offset_size, has_filters);
    let ChunkElementEncoding {
        elem_size,
        client_id,
        ..
    } = encoding;

    // The reader derives its geometry the same way.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    let geom_header = ExtensibleArrayHeader {
        client_id,
        element_size: elem_size as u8,
        bit_geometry: ExtensibleArrayBitGeometry::for_writer(),
        idx_blk_elmts: EA_IDX_BLK_ELMTS,
        min_dblk_nelmts: EA_MIN_DBLK_NELMTS,
        super_blk_min_data_ptrs: EA_SUPER_BLK_MIN_DATA_PTRS,
        max_idx_set: 0,
        index_block_address: StoredAddress::new(0),
    };
    // The selected element encoding matches the address width, and the constant
    // writer parameters satisfy the checked layout's bounds.
    let geom = ExtensibleArrayGeometry::from_header(&geom_header, offset_size)
        .expect("the writer's Extensible Array geometry is representable");
    let inline = geom_header.idx_blk_elmts as usize;

    let header_len = ExtensibleArrayHeader::serialized_size_with_widths(FormatWidths {
        offsets: offset_size,
        lengths: length_size,
    });
    let index_block_len = index_block_len(
        offset_size,
        inline,
        elem_size,
        geom.direct_dblk_nelmts.len(),
        geom.nsblk_addrs,
    );

    let stats = extensible_array_stats(
        &geom,
        u64::from(geom_header.idx_blk_elmts),
        geom_header.bit_geometry(),
        elem_size,
        offset_size,
        num_slots,
        occupancy,
    );
    let total_len =
        (header_len + index_block_len) as u64 + stats.data_blk_size + stats.super_blk_size;

    ExtensibleArrayLayout {
        encoding,
        bits: geom_header.bit_geometry(),
        geom,
        inline,
        header_len,
        index_block_len,
        stats,
        total_len,
    }
}

/// Returns the length in bytes of the Extensible Array [`build_extensible_array_at`] builds for
/// `slots`, without building it.
///
/// A caller that reserves the span of an array before writing it, such as an in-place writer
/// that places an array in freed space, takes the length from here.
pub fn extensible_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> u64 {
    extensible_array_layout(
        SlotOccupancy::Sparse(slots),
        slots.len() as u64,
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    )
    .total_len
}

/// Builds the Extensible Array for `slots` at `ea_address`: the header, the index block, and the
/// blocks after it.
///
/// The creation parameters are the C library's defaults. The index block holds the first
/// `idx_blk_elmts` elements and addresses the data blocks and super blocks after them. A block
/// none of whose slots holds a chunk is not allocated, and its address is undefined. A data block
/// of more than a page of elements is paged, and its super block marks the pages that hold a
/// chunk initialized.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the stored size of a chunk does not fit the chunk size
/// field, and [`FormatError::ValueTooLargeForPlatform`] if the length of the blocks or the count of
/// a block that holds a chunk does not fit a `usize`.
pub fn build_extensible_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    ea_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    let max_idx_set = slots.len();

    let layout = extensible_array_layout(
        SlotOccupancy::Sparse(slots),
        slots.len() as u64,
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    );
    let ChunkElementEncoding {
        chunk_size_bytes,
        elem_size,
        client_id,
    } = layout.encoding;
    let ExtensibleArrayLayout {
        bits,
        ref geom,
        inline,
        header_len,
        index_block_len,
        ..
    } = layout;

    let page_nelmts = bits.page_nelmts().get().to_usize()?;

    let index_block_address = ea_address.offset(header_len as u64);
    let body_base = index_block_address.offset(index_block_len as u64);

    let undef_addr = StoredAddress::undefined(offset_size);

    // ---- Build the body (direct data blocks, then super blocks) -----------
    // Each block's address is computed from `body_base`, so the body can be
    // built before the index block that references it.
    let mut body: Vec<u8> =
        Vec::with_capacity((layout.stats.data_blk_size + layout.stats.super_blk_size).to_usize()?);
    let mut direct_addrs: Vec<StoredAddress> = Vec::with_capacity(geom.direct_dblk_nelmts.len());
    let mut sblk_addrs: Vec<StoredAddress> = Vec::with_capacity(geom.nsblk_addrs);

    // The header statistics.
    let mut ndata_blks: u64 = 0;
    let mut data_blk_size: u64 = 0;
    let mut nsuper_blks: u64 = 0;
    let mut super_blk_size: u64 = 0;
    // `nelmts`: the slots of the index block and of every allocated data block.
    let mut alloc_slots: u64 = inline as u64;

    // The slot past the inline slots. The array spans `extensible_array_capacity()` slots, more
    // than a 32-bit `usize` holds, so the cursor and the block spans are `u64`, and a count is
    // narrowed only for a block that holds a chunk.
    let mut elem_cursor: u64 = inline as u64;

    // The data blocks the index block addresses. A block is written only where it holds a chunk,
    // as in `extensible_array_stats`, which applies the same test.
    let occupancy = SlotOccupancy::Sparse(slots);
    for &dblk_nelmts in &geom.direct_dblk_nelmts {
        if !occupancy.any_occupied(elem_cursor, dblk_nelmts) {
            direct_addrs.push(undef_addr);
            elem_cursor += dblk_nelmts;
            continue;
        }
        let addr = body_base.offset(body.len() as u64);
        let db_bytes = encode_data_block(
            slots,
            elem_cursor.to_usize()?,
            dblk_nelmts.to_usize()?,
            elem_cursor - inline as u64,
            ea_address,
            offset_size,
            has_filters,
            chunk_size_bytes,
            client_id,
            bits,
        )?;
        ndata_blks += 1;
        data_blk_size += db_bytes.len() as u64;
        alloc_slots += dblk_nelmts;
        body.extend_from_slice(&db_bytes);
        direct_addrs.push(addr);
        elem_cursor += dblk_nelmts;
    }

    // Super blocks: addresses stored in the index block. Super-block pointer `j`
    // refers to super block `first_indirect_sblk + j`.
    for j in 0..geom.nsblk_addrs {
        let sblk_idx = geom.first_indirect_sblk + j;
        // The span of a super block can exceed a 32-bit `usize`, so it stays `u64`.
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let sb_span = ndblks * dblk_nelmts;
        if !occupancy.any_occupied(elem_cursor, sb_span) {
            sblk_addrs.push(undef_addr);
            elem_cursor += sb_span;
            continue;
        }

        // This super block holds a chunk, so the number of chunks bounds its counts.
        let sb = geom.super_block_at(j, page_nelmts as u64);
        let is_paged = sb.blocks.is_paged();
        let npages = sb.blocks.npages();
        let sb_block_offset = elem_cursor - inline as u64;
        let mut page_bitmap = vec![0u8; sb.bitmap_size().to_usize()?];

        let mut sb_dblk_addrs: Vec<StoredAddress> = Vec::with_capacity(ndblks.to_usize()?);
        let mut local_elem = elem_cursor;
        for db_local in 0..ndblks {
            if !occupancy.any_occupied(local_elem, dblk_nelmts) {
                sb_dblk_addrs.push(undef_addr);
                local_elem += dblk_nelmts;
                continue;
            }
            let addr = body_base.offset(body.len() as u64);
            let db_bytes = encode_data_block(
                slots,
                local_elem.to_usize()?,
                dblk_nelmts.to_usize()?,
                local_elem - inline as u64,
                ea_address,
                offset_size,
                has_filters,
                chunk_size_bytes,
                client_id,
                bits,
            )?;
            ndata_blks += 1;
            data_blk_size += db_bytes.len() as u64;
            alloc_slots += dblk_nelmts;
            body.extend_from_slice(&db_bytes);
            sb_dblk_addrs.push(addr);
            if is_paged {
                // Marks initialized each page that holds a chunk, as the C library marks it
                // (`H5EA.c`, HDF5 2.2.0). No test checks these bits: the tests read back an array
                // with gaps in both libraries, which also passes with every page marked.
                let base = local_elem.to_usize()?;
                for p in 0..npages.to_usize()? {
                    let page_start = base + p * page_nelmts;
                    if !(0..page_nelmts).any(|s| slots.at(page_start + s).is_some()) {
                        continue;
                    }
                    let global_page = (db_local * npages).to_usize()? + p;
                    page_bitmap[global_page / 8] |= 0x80 >> (global_page % 8);
                }
            }
            local_elem += dblk_nelmts;
        }

        let super_block_address = body_base.offset(body.len() as u64);
        let super_block = encode_super_block(
            ea_address,
            sb_block_offset,
            &page_bitmap,
            &sb_dblk_addrs,
            offset_size,
            client_id,
            bits.block_offset_width(),
        );
        nsuper_blks += 1;
        super_blk_size += super_block.len() as u64;
        body.extend_from_slice(&super_block);
        sblk_addrs.push(super_block_address);

        elem_cursor += sb_span;
    }

    // ---- Build the header (EAHD) ------------------------------------------
    let write_length = |buf: &mut Vec<u8>, val: u64| bytes::write_length(buf, val, length_size);

    let mut header_bytes = Vec::with_capacity(header_len);
    header_bytes.extend_from_slice(b"EAHD");
    header_bytes.push(0); // version
    header_bytes.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    header_bytes.push(elem_size as u8);
    header_bytes.push(bits.max_index_bits().get());
    header_bytes.push(EA_IDX_BLK_ELMTS);
    header_bytes.push(EA_MIN_DBLK_NELMTS);
    header_bytes.push(EA_SUPER_BLK_MIN_DATA_PTRS);
    header_bytes.push(bits.page_bits.0);

    // 6 statistics, in the C library's order:
    //   [0] `nsuper_blks`   [1] `super_blk_size`   [2] `ndata_blks`
    //   [3] `data_blk_size` [4] `max_idx_set`      [5] `nelmts`
    write_length(&mut header_bytes, nsuper_blks);
    write_length(&mut header_bytes, super_blk_size);
    write_length(&mut header_bytes, ndata_blks);
    write_length(&mut header_bytes, data_blk_size);
    write_length(&mut header_bytes, max_idx_set.to_u64()); // one past the last slot
    write_length(&mut header_bytes, alloc_slots); // nelmts (allocated slots)

    write_stored_address(&mut header_bytes, index_block_address, offset_size);

    let header_checksum = checksum::jenkins_lookup3(&header_bytes);
    header_bytes.extend_from_slice(&header_checksum.to_le_bytes());
    debug_assert_eq!(header_bytes.len(), header_len);

    // ---- Build the index block (EAIB) -------------------------------------
    let mut index_block = Vec::with_capacity(index_block_len);
    index_block.extend_from_slice(b"EAIB");
    index_block.push(0); // version
    index_block.push(client_id);
    write_stored_address(&mut index_block, ea_address, offset_size);

    // Inline elements: always `idx_blk_elmts` slots, the unused ones undefined.
    #[allow(clippy::needless_range_loop)]
    for i in 0..inline {
        if let Some(chunk) = slots.at(i) {
            chunk_record::write_chunk_element(
                &mut index_block,
                chunk,
                offset_size,
                has_filters,
                chunk_size_bytes,
            )?;
        } else {
            chunk_record::write_undefined_element(
                &mut index_block,
                offset_size,
                has_filters,
                chunk_size_bytes,
            );
        }
    }
    // Direct data block addresses, then super block addresses.
    for &addr in &direct_addrs {
        write_stored_address(&mut index_block, addr, offset_size);
    }
    for &addr in &sblk_addrs {
        write_stored_address(&mut index_block, addr, offset_size);
    }

    let index_block_checksum = checksum::jenkins_lookup3(&index_block);
    index_block.extend_from_slice(&index_block_checksum.to_le_bytes());
    debug_assert_eq!(index_block.len(), index_block_len);

    let mut combined = header_bytes;
    combined.extend_from_slice(&index_block);
    combined.extend_from_slice(&body);
    // The caller reserved the length `extensible_array_len` returns, and a longer array would
    // overlap the next object.
    debug_assert_eq!(
        combined.len() as u64,
        layout.total_len,
        "an extensible array must fill the length its layout promised"
    );
    Ok(combined)
}

/// Returns the capacity in slots of an Extensible Array [`build_extensible_array_at`] builds: the
/// elements of the index block, the data blocks it addresses, and the data blocks of every super
/// block.
///
/// The capacity is 8,589,934,580 slots: 4 in the index block, 240 in six data blocks, and
/// 8,589,934,336 in 25 super blocks.
pub fn extensible_array_capacity() -> u64 {
    let geom_header = ExtensibleArrayHeader {
        client_id: 0,
        element_size: 8,
        bit_geometry: ExtensibleArrayBitGeometry::for_writer(),
        idx_blk_elmts: EA_IDX_BLK_ELMTS,
        min_dblk_nelmts: EA_MIN_DBLK_NELMTS,
        super_blk_min_data_ptrs: EA_SUPER_BLK_MIN_DATA_PTRS,
        max_idx_set: 0,
        index_block_address: StoredAddress::new(0),
    };
    // Eight-byte unfiltered elements and the constant creation parameters
    // satisfy the checked layout's bounds.
    let geom = ExtensibleArrayGeometry::from_header(&geom_header, OffsetWidth::Eight)
        .expect("the writer's Extensible Array geometry is representable");
    let direct: u64 = geom.direct_dblk_nelmts.iter().sum();
    let indirect: u64 = (0..geom.nsblk_addrs)
        .map(|j| {
            let (ndblks, dn) = geom.sblks[geom.first_indirect_sblk + j];
            ndblks * dn
        })
        .sum();
    u64::from(EA_IDX_BLK_ELMTS) + direct + indirect
}

/// The client ID of an array that indexes unfiltered chunks, whose elements each store a chunk
/// address alone.
///
/// The ID is defined in the Client ID table of "The Extensible Array Index" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub const EA_CLIENT_ID_UNFILTERED: u8 = 0;

// The creation parameters the writer stores and `extensible_array_capacity` derives the capacity
// from, the C library's defaults (`H5Dpkg.h`, HDF5 2.2.0).

/// The number of bits that hold the maximum number of elements, `H5D_EARRAY_MAX_NELMTS_BITS`.
pub(crate) const EA_MAX_NELMTS_BITS: u8 = 32;
/// The number of elements in the index block, `H5D_EARRAY_IDX_BLK_ELMTS`.
pub(crate) const EA_IDX_BLK_ELMTS: u8 = 4;
/// The number of elements in the smallest data block, `H5D_EARRAY_DATA_BLK_MIN_ELMTS`.
pub(crate) const EA_MIN_DBLK_NELMTS: u8 = 16;
/// The fewest data block addresses a super block holds, `H5D_EARRAY_SUP_BLK_MIN_DATA_PTRS`.
pub(crate) const EA_SUPER_BLK_MIN_DATA_PTRS: u8 = 4;
/// The base 2 logarithm of the number of elements in a data block page,
/// `H5D_EARRAY_MAX_DBLOCK_PAGE_NELMTS_BITS`.
pub(crate) const EA_MAX_DBLK_NELMTS_BITS: u8 = 10;

// The signatures, the version, and the field widths are defined in "The Extensible Array Index"
// of the format specification, version 4.0.
const HEADER_SIGNATURE: [u8; 4] = *b"EAHD";
const INDEX_BLOCK_SIGNATURE: [u8; 4] = *b"EAIB";
const SUPER_BLOCK_SIGNATURE: [u8; 4] = *b"EASB";
const DATA_BLOCK_SIGNATURE: [u8; 4] = *b"EADB";
const VERSION: u8 = 0;
const SIGNATURE_LEN: u64 = 4;
// The version, client ID, element size, and the five creation parameters of a header.
const HEADER_BYTE_FIELDS: u64 = 8;
// The six statistics of a header, each a length, of which "Max Index Set" is the fifth.
const STATISTICS: u64 = 6;
const STATISTICS_BEFORE_MAX_INDEX_SET: u64 = 4;
// The version and client ID of a block.
const BLOCK_BYTE_FIELDS: u64 = 2;
const CHECKSUM_LEN: u64 = 4;
// The bounds `H5EA__hdr_create` checks the creation parameters against (`H5EAhdr.c`, HDF5 2.2.0).
const MAX_INDEX_BITS: u8 = 64;
const MIN_DATA_BLOCKS: u8 = 2;
// The largest page exponent whose page element count fits a `u64`.
const MAX_PAGE_BITS: u8 = 63;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::width::FormatWidths;

    // The oracle is `H5EA__dblock_alloc`, which pages a block of more elements than a page.
    #[test]
    fn a_data_block_is_paged_only_past_a_full_page() {
        let at = |dblk_nelmts| DataBlockGeometry {
            dblk_nelmts,
            page_nelmts: 16,
        };
        assert!(!at(15).is_paged(), "under a page");
        assert!(!at(16).is_paged(), "exactly a page is not paged");
        assert!(at(17).is_paged(), "past a page");
        assert_eq!(at(16).npages(), 0, "a block that is not paged has no pages");
        assert_eq!(at(32).npages(), 2);
    }

    #[test]
    fn the_page_init_bitmap_is_a_byte_per_eight_pages_per_block() {
        let sb = |ndblks, dblk_nelmts| SuperBlockGeometry {
            ndblks,
            blocks: DataBlockGeometry {
                dblk_nelmts,
                page_nelmts: 16,
            },
        };
        // Not paged: no bitmap.
        assert_eq!(sb(4, 16).bitmap_size(), 0);
        // 2 blocks x 8 pages each -> one byte per block.
        assert_eq!(sb(2, 8 * 16).bitmap_size(), 2);
        // 2 blocks x 9 pages each -> the round-up costs a second byte per block.
        assert_eq!(sb(2, 9 * 16).bitmap_size(), 4);
        // One block, two pages: a single byte, not two bits.
        assert_eq!(sb(1, 2 * 16).bitmap_size(), 1);
    }

    /// Returns the length of a header: 12 fixed bytes, six length-sized statistics, the index
    /// block address, and the checksum.
    const fn eahd_len(os: u8, ls: u8) -> usize {
        12 + 6 * ls as usize + os as usize + 4
    }
    #[test]
    fn parse_header_valid() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 0; // version
        buf[5] = 0; // `client_id` = non-filtered
        buf[6] = 8; // `element_size`
        buf[7] = 10; // `max_nelmts_bits`
        buf[8] = 2; // `idx_blk_elmts`
        buf[9] = 4; // `min_dblk_nelmts`
        buf[10] = 2; // `super_blk_min_data_ptrs`
        buf[11] = 8; // `max_dblk_nelmts_bits`
        // 6 stats fields (each 8 bytes)
        buf[12..20].copy_from_slice(&0u64.to_le_bytes()); // stat[0]
        buf[20..28].copy_from_slice(&0u64.to_le_bytes()); // stat[1]
        buf[28..36].copy_from_slice(&0u64.to_le_bytes()); // stat[2]
        buf[36..44].copy_from_slice(&0u64.to_le_bytes()); // stat[3]
        buf[44..52].copy_from_slice(&5u64.to_le_bytes()); // stat[4] = `max_idx_set`
        buf[52..60].copy_from_slice(&0u64.to_le_bytes()); // stat[5]
        buf[60..68].copy_from_slice(&0x1000u64.to_le_bytes()); // `index_block_address`
        test_util::checksum::restamp(&mut buf, 0, eahd_len(os, ls));

        let hdr = ExtensibleArrayHeader::parse(&buf, 0, os, ls).unwrap();
        assert_eq!(hdr.client_id, 0);
        assert_eq!(hdr.element_size, 8);
        assert_eq!(hdr.idx_blk_elmts, 2);
        assert_eq!(hdr.min_dblk_nelmts, 4);
        assert_eq!(hdr.max_idx_set, 5);
        assert_eq!(hdr.index_block_address, StoredAddress::new(0x1000));
    }

    #[test]
    fn bit_geometry_stores_two_bytes_with_byte_alignment() {
        assert_eq!(core::mem::size_of::<MaxIndexBits>(), 1);
        assert_eq!(core::mem::align_of::<MaxIndexBits>(), 1);
        assert_eq!(core::mem::size_of::<PageBits>(), 1);
        assert_eq!(core::mem::align_of::<PageBits>(), 1);
        assert_eq!(core::mem::size_of::<ExtensibleArrayBitGeometry>(), 2);
        assert_eq!(core::mem::align_of::<ExtensibleArrayBitGeometry>(), 1);
        assert_eq!(core::mem::size_of::<BlockOffsetWidth>(), 1);
        assert_eq!(core::mem::align_of::<BlockOffsetWidth>(), 1);
    }

    #[rstest]
    #[case::one_byte(BlockOffsetWidth::One, &[0x01], 0x01)]
    #[case::two_bytes(BlockOffsetWidth::Two, &[0x01, 0x02], 0x0201)]
    #[case::three_bytes(BlockOffsetWidth::Three, &[0x01, 0x02, 0x03], 0x03_0201)]
    #[case::four_bytes(BlockOffsetWidth::Four, &[0x01, 0x02, 0x03, 0x04], 0x0403_0201)]
    #[case::five_bytes(BlockOffsetWidth::Five, &[0x01, 0x02, 0x03, 0x04, 0x05], 0x0005_0403_0201)]
    #[case::six_bytes(BlockOffsetWidth::Six, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 0x0605_0403_0201)]
    #[case::seven_bytes(BlockOffsetWidth::Seven, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07], 0x0007_0605_0403_0201)]
    #[case::eight_bytes(BlockOffsetWidth::Eight, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08], 0x0807_0605_0403_0201)]
    fn block_offset_width_reads_the_value_and_reports_missing_bytes(
        #[case] width: BlockOffsetWidth,
        #[case] bytes: &[u8],
        #[case] value: u64,
    ) {
        assert_eq!(width.read(bytes).unwrap(), value);
        let err = width.read(&bytes[..bytes.len() - 1]).unwrap_err();
        let FormatError::UnexpectedEof {
            expected,
            available,
        } = err
        else {
            panic!("expected UnexpectedEof, got {err:?}");
        };
        assert_eq!(expected, width.bytes());
        assert_eq!(available, bytes.len() - 1);
    }

    #[rstest]
    #[case::zero_maximum(0, 10, "Extensible Array maximum element bit count must be in 1..=64")]
    #[case::maximum_65(65, 10, "Extensible Array maximum element bit count must be in 1..=64")]
    #[case::maximum_255(
        255,
        10,
        "Extensible Array maximum element bit count must be in 1..=64"
    )]
    #[case::page_64(64, 64, "Extensible Array page element count does not fit u64")]
    #[case::page_255(64, 255, "Extensible Array page element count does not fit u64")]
    #[case::page_exceeds_maximum(
        10,
        11,
        "Extensible Array page exponent exceeds maximum element bit count"
    )]
    fn malformed_bit_geometry_fails_at_the_header(
        #[case] max_bits: u8,
        #[case] page_bits: u8,
        #[case] reason: &str,
    ) {
        let mut header = empty_array();
        header[7] = max_bits;
        header[11] = page_bits;
        test_util::checksum::restamp(&mut header, 0, eahd_len(8, 8));

        let err = ExtensibleArrayHeader::parse(&header, 0, 8, 8).unwrap_err();
        let FormatError::InvalidChunkGeometry(actual) = err else {
            panic!("expected InvalidChunkGeometry, got {err:?}");
        };
        assert_eq!(actual, reason);
    }

    #[rstest]
    #[case::one_byte(1, 0, 1, 1)]
    #[case::byte_boundary(8, 8, 256, 1)]
    #[case::two_bytes(9, 9, 512, 2)]
    #[case::eight_bytes(64, 10, 1024, 8)]
    #[case::largest_page(64, 63, 1u64 << 63, 8)]
    fn parsed_bit_geometry_supplies_page_count_and_offset_width(
        #[case] max_bits: u8,
        #[case] page_bits: u8,
        #[case] page_nelmts: u64,
        #[case] offset_size: usize,
    ) {
        let mut bytes = empty_array();
        bytes[7] = max_bits;
        bytes[11] = page_bits;
        test_util::checksum::restamp(&mut bytes, 0, eahd_len(8, 8));
        let header = ExtensibleArrayHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(header.max_index_bits().get(), max_bits);
        assert_eq!(header.page_nelmts().get(), page_nelmts);
        assert_eq!(header.block_offset_width().bytes(), offset_size);

        let offset = u64::MAX >> (8 * (8 - offset_size));
        let block = encode_super_block(
            StoredAddress::new(0),
            offset,
            &[],
            &[],
            OffsetWidth::Eight,
            header.client_id,
            header.block_offset_width(),
        );
        assert_eq!(
            &block[14..14 + offset_size],
            &offset.to_le_bytes()[..offset_size]
        );
        assert_eq!(
            header.block_offset_width().read(&block[14..]).unwrap(),
            offset
        );
        checksum::verify_trailing(&block).unwrap();
    }

    fn empty_array() -> Vec<u8> {
        build_extensible_array_at(
            &IndexSlots::dense(&[]),
            8,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(0),
        )
        .unwrap()
    }

    #[test]
    fn parse_header_invalid_signature() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"XXXX");
        let result = ExtensibleArrayHeader::parse(&buf, 0, 8, 8);
        assert_eq!(
            result.unwrap_err(),
            FormatError::ChunkedReadError("invalid Extensible Array header signature".into())
        );
    }

    #[test]
    fn parse_header_invalid_version() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 1;
        let result = ExtensibleArrayHeader::parse(&buf, 0, 8, 8);
        assert_eq!(
            result.unwrap_err(),
            FormatError::ChunkedReadError("unsupported Extensible Array header version: 1".into())
        );
    }

    /// Returns the records of the Extensible Array whose header is at `header_offset` in
    /// `file_data`.
    fn records_at(
        file_data: &[u8],
        header_offset: usize,
        chunk_byte_size: u64,
    ) -> Vec<(u64, ChunkRecord)> {
        let image = FileRegion::new(
            FileRange::new(PhysicalAddress::new(0), file_data.len() as u64).unwrap(),
            file_data,
        )
        .unwrap();
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let locator = ExtensibleArrayHeaderLocator::new(
            widths,
            BaseAddress::ZERO,
            StoredAddress::new(header_offset as u64),
        );
        records_in(image, locator, chunk_byte_size).unwrap()
    }

    /// Returns the records of unfiltered chunks of `chunk_byte_size` bytes stored back to back from
    /// `base_addr`, one per slot of `slots`.
    fn back_to_back(
        slots: core::ops::Range<u64>,
        base_addr: u64,
        chunk_byte_size: u64,
    ) -> Vec<(u64, ChunkRecord)> {
        slots
            .map(|slot| {
                let record = ChunkRecord {
                    address: StoredAddress::new(base_addr + slot * chunk_byte_size),
                    stored_size: chunk_byte_size,
                    filter_mask: 0,
                };
                (slot, record)
            })
            .collect()
    }

    #[test]
    fn read_inline_only() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let num_chunks = 2usize;
        let chunk_byte_size = 20u64 * 8; // 20 elements × 8 bytes

        let mut file_data = vec![0u8; 0x3000];

        // EAHD at offset 0x100
        let header_offset = 0x100usize;
        let index_block_offset = 0x200usize;

        // Build EAHD
        file_data[header_offset..header_offset + 4].copy_from_slice(b"EAHD");
        file_data[header_offset + 4] = 0; // version
        file_data[header_offset + 5] = 0; // `client_id` = non-filtered
        file_data[header_offset + 6] = osv as u8; // `element_size`
        file_data[header_offset + 7] = 10; // `max_nelmts_bits`
        file_data[header_offset + 8] = num_chunks as u8; // `idx_blk_elmts` (all inline)
        file_data[header_offset + 9] = 4; // `min_dblk_nelmts`
        file_data[header_offset + 10] = 2; // `super_blk_min_data_ptrs`
        file_data[header_offset + 11] = 8; // `max_dblk_nelmts_bits`
        // 6 stats fields (each 8 bytes), `max_idx_set` at stat[4]
        file_data[header_offset + 44..header_offset + 52]
            .copy_from_slice(&(num_chunks as u64).to_le_bytes());
        file_data[header_offset + 60..header_offset + 68]
            .copy_from_slice(&(index_block_offset as u64).to_le_bytes());
        test_util::checksum::restamp(&mut file_data, header_offset, eahd_len(os, ls));

        // Build EAIB at `index_block_offset`
        file_data[index_block_offset..index_block_offset + 4].copy_from_slice(b"EAIB");
        file_data[index_block_offset + 4] = 0; // version
        file_data[index_block_offset + 5] = 0; // `client_id`
        file_data[index_block_offset + 6..index_block_offset + 14]
            .copy_from_slice(&(header_offset as u64).to_le_bytes());

        // Inline elements
        let elem_start = index_block_offset + 6 + osv;
        let base_addr = 0x1000u64;
        for i in 0..num_chunks {
            let addr = base_addr + i as u64 * chunk_byte_size;
            let p = elem_start + i * osv;
            file_data[p..p + osv].copy_from_slice(&addr.to_le_bytes());
        }

        // The index block is sized by the header's geometry whatever the array
        // holds: prefix, two inline slots, then a pointer per direct data block
        // (min_dblk_nelmts=4 and super_blk_min_data_ptrs=2 give two) and one per
        // super block (nine levels less the two taken as direct, so seven).
        // The pointers stay zero, and the read stops at the inline slots.
        test_util::checksum::restamp(
            &mut file_data,
            index_block_offset,
            (6 + osv) + 2 * osv + 2 * osv + 7 * osv + 4,
        );

        assert_eq!(
            records_at(&file_data, header_offset, chunk_byte_size),
            back_to_back(0..2, base_addr, chunk_byte_size)
        );
    }

    #[test]
    fn read_inline_plus_data_blocks() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let chunk_byte_size = 10u64 * 8; // 10 elements × 8 bytes
        let idx_blk_elmts = 2u8;
        let min_dblk_nelmts = 2u8;
        let sblk_min = 2u8;
        let total_chunks = 4usize; // 2 inline and 2 in the first data block

        let mut file_data = vec![0u8; 0x5000];
        let header_offset = 0x100usize;
        let index_block_offset = 0x200usize;
        let data_block_offset = 0x300usize;

        // EAHD
        file_data[header_offset..header_offset + 4].copy_from_slice(b"EAHD");
        file_data[header_offset + 4] = 0;
        file_data[header_offset + 5] = 0; // `client_id`
        file_data[header_offset + 6] = osv as u8; // `element_size`
        file_data[header_offset + 7] = 10;
        file_data[header_offset + 8] = idx_blk_elmts;
        file_data[header_offset + 9] = min_dblk_nelmts;
        file_data[header_offset + 10] = sblk_min;
        file_data[header_offset + 11] = 8;
        // 6 stats fields (each 8 bytes), `max_idx_set` at stat[4] (offset 12 + 4*8 = 44)
        file_data[header_offset + 44..header_offset + 52]
            .copy_from_slice(&(total_chunks as u64).to_le_bytes());
        // The index block address at offset 12 + 6*8 = 60
        file_data[header_offset + 60..header_offset + 68]
            .copy_from_slice(&(index_block_offset as u64).to_le_bytes());
        test_util::checksum::restamp(&mut file_data, header_offset, eahd_len(os, ls));

        // EAIB
        file_data[index_block_offset..index_block_offset + 4].copy_from_slice(b"EAIB");
        file_data[index_block_offset + 4] = 0;
        file_data[index_block_offset + 5] = 0;
        file_data[index_block_offset + 6..index_block_offset + 14]
            .copy_from_slice(&(header_offset as u64).to_le_bytes());

        let mut pos = index_block_offset + 6 + osv;

        // Inline elements (2 chunks)
        let base_addr = 0x1000u64;
        for i in 0..idx_blk_elmts as usize {
            let addr = base_addr + i as u64 * chunk_byte_size;
            file_data[pos..pos + osv].copy_from_slice(&addr.to_le_bytes());
            pos += osv;
        }

        // For minimum two data block pointers, the direct blocks hold 2 and 4 elements.
        let n_direct_dblks = 2;
        file_data[pos..pos + osv].copy_from_slice(&(data_block_offset as u64).to_le_bytes());
        pos += osv;
        for _ in 1..n_direct_dblks {
            file_data[pos..pos + osv].copy_from_slice(&u64::MAX.to_le_bytes());
            pos += osv;
        }

        // EADB at `data_block_offset` (`min_dblk_nelmts` elements)
        file_data[data_block_offset..data_block_offset + 4].copy_from_slice(b"EADB");
        file_data[data_block_offset + 4] = 0;
        file_data[data_block_offset + 5] = 0;
        file_data[data_block_offset + 6..data_block_offset + 14]
            .copy_from_slice(&(header_offset as u64).to_le_bytes());
        // The block offset takes ceil(10 / 8) = 2 bytes, and is 0 for the first data block.
        let blk_off_size = (10usize).div_ceil(8); // max_nelmts_bits=10
        let mut dbpos = data_block_offset + 6 + osv + blk_off_size;
        for i in 0..min_dblk_nelmts as usize {
            let addr = base_addr + (idx_blk_elmts as u64 + i as u64) * chunk_byte_size;
            file_data[dbpos..dbpos + osv].copy_from_slice(&addr.to_le_bytes());
            dbpos += osv;
        }

        // Prefix, two inline slots, the two direct pointers written above, and
        // one pointer per remaining super block (ten levels less the two taken
        // as direct).
        test_util::checksum::restamp(
            &mut file_data,
            index_block_offset,
            (6 + osv) + 2 * osv + n_direct_dblks * osv + 8 * osv + 4,
        );
        // Prefix, block offset, `min_dblk_nelmts` element slots, checksum.
        test_util::checksum::restamp(
            &mut file_data,
            data_block_offset,
            (6 + osv) + blk_off_size + min_dblk_nelmts as usize * osv + 4,
        );

        assert_eq!(
            records_at(&file_data, header_offset, chunk_byte_size),
            back_to_back(0..4, base_addr, chunk_byte_size)
        );
    }

    #[test]
    fn header_serialized_size() {
        // 12 fixed + 6*8 stats + 8 address + 4 checksum = 72
        assert_eq!(ExtensibleArrayHeader::serialized_size(8, 8), 72);
        // 12 fixed + 6*4 stats + 4 address + 4 checksum = 44
        assert_eq!(ExtensibleArrayHeader::serialized_size(4, 4), 44);
    }

    #[test]
    fn build_extensible_array_valid_structure() {
        let chunks = vec![
            ChunkRecord {
                address: StoredAddress::new(0x1000),
                stored_size: 80,
                filter_mask: 0,
            },
            ChunkRecord {
                address: StoredAddress::new(0x1050),
                stored_size: 80,
                filter_mask: 0,
            },
        ];
        let ea = build_extensible_array_at(
            &IndexSlots::dense(&chunks),
            80,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(0x2000),
        )
        .unwrap();
        assert_eq!(&ea[0..4], b"EAHD");
        // `EAIB` follows `EAHD`: 12 fixed + 6*8 stats + 8 address + 4 checksum = 72
        let header_len = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * 8 + 8 + 4;
        assert_eq!(&ea[header_len..header_len + 4], b"EAIB");
    }

    #[test]
    fn extensible_array_stats_matches_what_it_builds() {
        let geom_header = ExtensibleArrayHeader {
            client_id: 0,
            element_size: 8,
            bit_geometry: ExtensibleArrayBitGeometry::for_writer(),
            idx_blk_elmts: 4,
            min_dblk_nelmts: 16,
            super_blk_min_data_ptrs: 4,
            max_idx_set: 0,
            index_block_address: StoredAddress::new(0),
        };
        let geom = ExtensibleArrayGeometry::from_header(&geom_header, OffsetWidth::Eight).unwrap();
        for &n in &[1u64, 4, 20, 100, 244, 300, 2000, 50000, 131056, 140000] {
            let chunks: Vec<ChunkRecord> = (0..n)
                .map(|i| ChunkRecord {
                    address: StoredAddress::new(0x1000 + i * 8),
                    stored_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let ea = build_extensible_array_at(
                &IndexSlots::dense(&chunks),
                8,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                false,
                StoredAddress::new(0x100000),
            )
            .unwrap();
            // Parse the 6 stats from the EAHD (12-byte fixed prefix, then 6 * ls).
            let stat =
                |k: usize| u64::from_le_bytes(ea[12 + k * 8..12 + k * 8 + 8].try_into().unwrap());
            let built = super::ExtensibleArrayStats {
                nsuper_blks: stat(0),
                super_blk_size: stat(1),
                ndata_blks: stat(2),
                data_blk_size: stat(3),
                max_idx_set: stat(4),
                nelmts: stat(5),
            };
            let computed = super::extensible_array_stats(
                &geom,
                u64::from(geom_header.idx_blk_elmts),
                geom_header.bit_geometry(),
                8,
                OffsetWidth::Eight,
                n,
                SlotOccupancy::Dense(n),
            );
            assert_eq!(computed, built, "stats mismatch at n={n}");
        }
    }

    // Sweeps every count through the first super block, then the deeper super blocks and the
    // first paged data block.
    #[test]
    fn extensible_array_len_matches_what_it_builds() {
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
            let planned = extensible_array_len(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
            );
            let built = build_extensible_array_at(
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
                // Contiguous across the inline, direct-block and first
                // super-block ranges.
                for n in 0..=250u64 {
                    check(n, 8, offset_size, length_size, has_filters);
                }
                // The deeper super blocks, and the paged boundary: 131,060 is the
                // last element the unpaged super block 12 holds, 131,061 the first
                // that allocates a paged data block.
                for &n in &[300u64, 2_000, 50_000, 131_060, 131_061, 140_000] {
                    check(n, 8, offset_size, length_size, has_filters);
                }
            }
        }

        // The chunk size field of a filtered element, and with it every block, grows with the chunk
        // size.
        for &chunk_bytes in &chunk_record::CHUNK_BYTES {
            for &n in &[1u64, 5, 244, 300, 2_000] {
                check(n, chunk_bytes, OffsetWidth::Eight, LengthWidth::Eight, true);
            }
        }
        // The four chunk sizes select four different widths, measured on an array of no chunks,
        // whose width comes from `chunk_bytes` alone.
        let widths: Vec<usize> = chunk_record::CHUNK_BYTES
            .iter()
            .map(|&chunk_bytes| {
                super::extensible_array_layout(
                    SlotOccupancy::Dense(0),
                    0,
                    chunk_bytes,
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                    true,
                )
                .encoding
                .chunk_size_bytes
            })
            .collect();
        let mut distinct = widths.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            chunk_record::CHUNK_BYTES.len(),
            "each chunk size must select a different compressed-size field width, got {widths:?}"
        );
    }

    const WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    const ARRAY_AT: u64 = 0x80;

    const FIRST_PAGED_SLOT: u64 = 131_060;

    // "The Extensible Array Index" in format specification, version 4.0, defines the version 0
    // header field order and the four-byte filter mask for version 4 chunked layouts.
    const HEADER_MINIMUM_POINTERS: usize = 10;
    const FILTER_MASK_BYTES: u8 = 4;

    /// The Extensible Array the writer builds at `ARRAY_AT`, in an image whose first byte is at
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
            Self::with_slots(
                base,
                widths,
                chunk_bytes,
                &IndexSlots::dense(&chunk_record::chunks(chunk_bytes, n, has_filters)),
                has_filters,
            )
        }

        fn with_slots(
            base: u64,
            widths: FormatWidths,
            chunk_bytes: u64,
            slots: &IndexSlots<'_>,
            has_filters: bool,
        ) -> Self {
            let array = build_extensible_array_at(
                slots,
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

        fn locator(&self) -> ExtensibleArrayHeaderLocator {
            ExtensibleArrayHeaderLocator::new(
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

        fn read(&self, range: FileRange) -> Result<FileRegion<'_>, FormatError> {
            self.image().subregion(range)
        }

        fn header(&self) -> Result<ExtensibleArray, FormatError> {
            let locator = self.locator();
            locator.parse(self.read(locator.parse_range()?)?)
        }

        fn index_block(&self) -> ExtensibleArrayIndexBlock<'_> {
            let block = self.header().unwrap().index_block().unwrap();
            block
                .parse(self.read(block.parse_range()).unwrap())
                .unwrap()
        }

        fn super_blocks(&self) -> Vec<ExtensibleArraySuperBlockRef> {
            self.index_block()
                .super_blocks()
                .collect::<Result<_, _>>()
                .unwrap()
        }

        fn data_blocks(&self) -> Vec<ExtensibleArrayDataBlockRef<'_>> {
            let mut blocks: Vec<ExtensibleArrayDataBlockRef<'_>> = self
                .index_block()
                .data_blocks()
                .collect::<Result<_, _>>()
                .unwrap();
            for super_block in self.super_blocks() {
                let super_block = super_block
                    .parse(self.read(super_block.parse_range()).unwrap())
                    .unwrap();
                for block in super_block.data_blocks() {
                    blocks.push(block.unwrap());
                }
            }
            blocks
        }

        fn paged_data_blocks(&self) -> Vec<ExtensibleArrayDataBlockRef<'_>> {
            self.data_blocks()
                .into_iter()
                .filter(|block| block.parse_range() != block.range())
                .collect()
        }

        fn pages(&self, block: ExtensibleArrayDataBlockRef<'_>) -> Vec<ExtensibleArrayPageRef> {
            let ExtensibleArrayDataBlock::Pages(pages) = block
                .parse(self.read(block.parse_range()).unwrap())
                .unwrap()
            else {
                panic!("the data block {block:?} is not paged");
            };
            pages.pages().collect::<Result<_, _>>().unwrap()
        }

        fn local(&self, at: PhysicalAddress) -> usize {
            usize::try_from(at.get() - self.base).unwrap()
        }

        fn restamp(&mut self, range: FileRange) {
            let at = self.local(range.start());
            let len = usize::try_from(range.length()).unwrap();
            test_util::checksum::restamp(&mut self.bytes, at, len);
        }

        fn records(&self, chunk_byte_size: u64) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
            records_in(self.image(), self.locator(), chunk_byte_size)
        }
    }

    /// Returns the records of the index block and of each block and page that holds a slot below
    /// the header's "Max Index Set", each parsed from the range its reference returns.
    fn records_in(
        image: FileRegion<'_>,
        locator: ExtensibleArrayHeaderLocator,
        chunk_byte_size: u64,
    ) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let read = |range: FileRange| image.subregion(range);
        let Some(index_block) = locator.parse(read(locator.parse_range()?)?)?.index_block() else {
            return Ok(Vec::new());
        };
        let index_block = index_block.parse(read(index_block.parse_range())?)?;
        let mut records: Vec<(u64, ChunkRecord)> = index_block
            .elements()
            .records(chunk_byte_size)
            .collect::<Result<_, _>>()?;
        let mut visit = |block: ExtensibleArrayDataBlockRef<'_>| {
            match block.parse(read(block.parse_range())?)? {
                ExtensibleArrayDataBlock::Elements(elements) => {
                    for record in elements.records(chunk_byte_size) {
                        records.push(record?);
                    }
                }
                ExtensibleArrayDataBlock::Pages(pages) => {
                    for page in pages.pages() {
                        let page = page?;
                        for record in page
                            .parse(read(page.parse_range())?)?
                            .records(chunk_byte_size)
                        {
                            records.push(record?);
                        }
                    }
                }
            }
            Ok::<_, FormatError>(())
        };
        for block in index_block.data_blocks() {
            let block = block?;
            if block.holds_set_slots() {
                visit(block)?;
            }
        }
        for super_block in index_block.super_blocks() {
            let super_block = super_block?;
            if !super_block.holds_set_slots() {
                continue;
            }
            for block in super_block
                .parse(read(super_block.parse_range())?)?
                .data_blocks()
            {
                let block = block?;
                if block.holds_set_slots() {
                    visit(block)?;
                }
            }
        }
        Ok(records)
    }

    #[rstest]
    fn an_array_reads_back_at_every_width(
        #[values(
            FormatWidths { offsets: OffsetWidth::Two, lengths: LengthWidth::Two },
            FormatWidths { offsets: OffsetWidth::Four, lengths: LengthWidth::Four },
            FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight }
        )]
        widths: FormatWidths,
        #[values(0, 3, 4, 20, 244, 2_000)] n: u64,
        #[values(false, true)] has_filters: bool,
        #[values(8, 300, 100_000, 1 << 32)] chunk_bytes: u64,
    ) {
        let fixture = Fixture::at(0, widths, chunk_bytes, n, has_filters);
        assert_eq!(
            fixture.records(chunk_bytes).unwrap(),
            chunk_record::slotted(chunk_record::chunks(chunk_bytes, n, has_filters))
        );
    }

    #[rstest]
    fn a_paged_array_reads_back(
        #[values(
            FormatWidths { offsets: OffsetWidth::Four, lengths: LengthWidth::Four },
            FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight }
        )]
        widths: FormatWidths,
        #[values(false, true)] has_filters: bool,
    ) {
        let n = 140_000;
        let fixture = Fixture::at(0, widths, 8, n, has_filters);
        assert!(!fixture.paged_data_blocks().is_empty());
        assert_eq!(
            fixture.records(8).unwrap(),
            chunk_record::slotted(chunk_record::chunks(8, n, has_filters))
        );
    }

    #[rstest]
    fn the_block_ranges_tile_the_array(#[values(4, 244, 2_000, 140_000)] n: u64) {
        let fixture = Fixture::new(n, false);
        let stats = extensible_array_layout(
            SlotOccupancy::Dense(n),
            n,
            8,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
        )
        .stats;
        let header = fixture.header().unwrap();
        let mut ranges = vec![header.range(), header.index_block().unwrap().parse_range()];
        ranges.extend(
            fixture
                .super_blocks()
                .iter()
                .map(ExtensibleArraySuperBlockRef::parse_range),
        );
        ranges.extend(
            fixture
                .data_blocks()
                .iter()
                .map(ExtensibleArrayDataBlockRef::range),
        );
        assert_eq!(
            ranges.len() as u64,
            2 + stats.nsuper_blks + stats.ndata_blks
        );
        ranges.sort_by_key(|range| range.start());
        assert_eq!(ranges[0].start().get(), ARRAY_AT);
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end(), pair[1].start());
        }
        assert_eq!(
            ranges.last().unwrap().end().get(),
            fixture.bytes.len() as u64
        );
    }

    #[test]
    fn the_pages_of_a_paged_data_block_follow_its_prefix() {
        let fixture = Fixture::new(140_000, false);
        let block = fixture.paged_data_blocks()[0];
        let prefix = block.parse_range();
        assert_eq!(prefix.length(), 4 + 2 + 8 + 4 + 4);
        assert_eq!(
            block.slots().unwrap(),
            FIRST_PAGED_SLOT..FIRST_PAGED_SLOT + 2_048
        );
        let placed = |page: ExtensibleArrayPageRef| {
            let range = page.parse_range();
            (range.start().get(), range.length(), page.slots().unwrap())
        };
        let unparsed: Vec<(u64, u64, Range<u64>)> = block
            .pages()
            .unwrap()
            .unwrap()
            .pages()
            .map(|page| placed(page.unwrap()))
            .collect();
        let pages: Vec<(u64, u64, Range<u64>)> =
            fixture.pages(block).into_iter().map(placed).collect();
        let start = prefix.end().get();
        assert_eq!(fixture.data_blocks()[0].pages().unwrap(), None);
        let expected = vec![
            (
                start,
                1_024 * 8 + 4,
                FIRST_PAGED_SLOT..FIRST_PAGED_SLOT + 1_024,
            ),
            (
                start + 1_024 * 8 + 4,
                1_024 * 8 + 4,
                FIRST_PAGED_SLOT + 1_024..FIRST_PAGED_SLOT + 2_048,
            ),
        ];
        assert_eq!(unparsed, expected);
        assert_eq!(pages, expected);
        assert_eq!(pages[1].0 + pages[1].1, block.range().end().get());
    }

    #[rstest]
    #[case::first(0b0111_0000, &[1, 2, 3])]
    #[case::second(0b1011_0000, &[0, 2, 3])]
    #[case::every_page(0b0000_0000, &[])]
    fn a_cleared_page_bit_skips_its_page(
        #[case] bitmap: u8,
        #[case] pages: &[u64],
        #[values(false, true)] has_filters: bool,
    ) {
        let n = FIRST_PAGED_SLOT + 4_096;
        let mut fixture = Fixture::new(n, has_filters);
        let super_block = fixture.super_blocks()[9].parse_range();
        let at = fixture.local(super_block.start()) + 4 + 2 + 8 + 4;
        fixture.bytes[at] = bitmap;
        fixture.restamp(super_block);
        let expected: Vec<(u64, ChunkRecord)> =
            chunk_record::slotted(chunk_record::chunks(8, n, has_filters))
                .into_iter()
                .filter(|&(slot, _)| {
                    slot < FIRST_PAGED_SLOT || pages.contains(&((slot - FIRST_PAGED_SLOT) / 1_024))
                })
                .collect();
        assert_eq!(fixture.records(8).unwrap(), expected);
    }

    #[rstest]
    #[case::inline_elements(&[1, 3], 4)]
    #[case::direct_data_blocks(&[0, 30, 200], 244)]
    #[case::super_blocks(&[2, 1_500, 1_999], 2_000)]
    #[case::pages(&[FIRST_PAGED_SLOT + 1_030, FIRST_PAGED_SLOT + 3 * 2_048 + 10], 140_000)]
    fn an_array_with_unallocated_blocks_reads_its_chunks(
        #[case] occupied: &[u64],
        #[case] len: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let records = chunk_record::chunks(8, occupied.len() as u64, has_filters);
        let slots = IndexSlots::new(&records, occupied, len).unwrap();
        let fixture = Fixture::with_slots(0, WIDTHS, 8, &slots, has_filters);
        assert_eq!(
            fixture.records(8).unwrap(),
            occupied.iter().copied().zip(records).collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_unallocated_block_and_an_uninitialized_page_have_no_reference() {
        let records = chunk_record::chunks(8, 2, false);
        let occupied = [FIRST_PAGED_SLOT + 1_030, FIRST_PAGED_SLOT + 3 * 2_048 + 10];
        let slots = IndexSlots::new(&records, &occupied, 140_000).unwrap();
        let fixture = Fixture::with_slots(0, WIDTHS, 8, &slots, false);
        assert_eq!(fixture.super_blocks().len(), 1);
        let blocks = fixture.data_blocks();
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.slots().unwrap())
                .collect::<Vec<_>>(),
            vec![
                FIRST_PAGED_SLOT..FIRST_PAGED_SLOT + 2_048,
                FIRST_PAGED_SLOT + 3 * 2_048..FIRST_PAGED_SLOT + 4 * 2_048,
            ]
        );
        let pages: Vec<Range<u64>> = blocks
            .iter()
            .flat_map(|&block| fixture.pages(block))
            .map(|page| page.slots().unwrap())
            .collect();
        assert_eq!(
            pages,
            vec![
                FIRST_PAGED_SLOT + 1_024..FIRST_PAGED_SLOT + 2_048,
                FIRST_PAGED_SLOT + 3 * 2_048..FIRST_PAGED_SLOT + 3 * 2_048 + 1_024,
            ]
        );
    }

    #[rstest]
    #[case::none(0)]
    #[case::inside_the_index_block(2)]
    #[case::at_the_first_data_block(4)]
    #[case::inside_a_data_block(5)]
    #[case::inside_a_super_block(1_000)]
    #[case::inside_a_page(FIRST_PAGED_SLOT + 1_500)]
    #[case::past_the_last_chunk(200_000)]
    #[case::full_width(u64::MAX)]
    fn the_records_stop_at_the_max_index_set(#[case] max_index_set: u64) {
        let n = FIRST_PAGED_SLOT + 2_048;
        let mut fixture = Fixture::new(n, true);
        let header = fixture.header().unwrap().range();
        let at = fixture.local(header.start()) + 12 + 4 * 8;
        fixture.bytes[at..at + 8].copy_from_slice(&max_index_set.to_le_bytes());
        fixture.restamp(header);
        let expected: Vec<(u64, ChunkRecord)> =
            chunk_record::slotted(chunk_record::chunks(8, n, true))
                .into_iter()
                .take(usize::try_from(max_index_set).unwrap_or(usize::MAX))
                .collect();
        assert_eq!(fixture.records(8).unwrap(), expected);
    }

    #[rstest]
    #[case::inside_the_index_block(2, 0, 0, 0)]
    #[case::inside_the_first_data_block(5, 1, 0, 0)]
    #[case::inside_the_first_super_block(300, 6 + 1, 1, 0)]
    #[case::inside_the_first_page(FIRST_PAGED_SLOT + 10, 6 + 184 + 1, 10, 1)]
    fn a_block_or_page_past_the_max_index_set_holds_no_set_slots(
        #[case] max_index_set: u64,
        #[case] data_blocks: usize,
        #[case] super_blocks: usize,
        #[case] pages: usize,
    ) {
        let mut fixture = Fixture::new(FIRST_PAGED_SLOT + 4_096, false);
        let header = fixture.header().unwrap().range();
        let at = fixture.local(header.start()) + 12 + 4 * 8;
        fixture.bytes[at..at + 8].copy_from_slice(&max_index_set.to_le_bytes());
        fixture.restamp(header);
        let all_super_blocks = fixture.super_blocks();
        assert_eq!(all_super_blocks.len(), 10);
        assert_eq!(
            all_super_blocks
                .iter()
                .filter(|block| block.holds_set_slots())
                .count(),
            super_blocks
        );
        let all_data_blocks = fixture.data_blocks();
        assert_eq!(all_data_blocks.len(), 6 + 184 + 2);
        assert_eq!(
            all_data_blocks
                .iter()
                .filter(|block| block.holds_set_slots())
                .count(),
            data_blocks
        );
        assert_eq!(
            fixture
                .paged_data_blocks()
                .into_iter()
                .map(|block| fixture.pages(block).len())
                .sum::<usize>(),
            pages
        );
    }

    #[test]
    fn a_slot_parses_from_the_regions_on_its_path_alone() {
        let n = 140_000;
        let slot = FIRST_PAGED_SLOT + 2 * 2_048 + 1_100;
        let fixture = Fixture::new(n, true);
        let only = |range: FileRange| fixture.read(range).unwrap();
        let locator = fixture.locator();
        let header = locator.parse(only(locator.parse_range().unwrap())).unwrap();
        let index_block = header.index_block().unwrap();
        let index_block = index_block.parse(only(index_block.parse_range())).unwrap();
        let super_block = index_block
            .super_blocks()
            .map(Result::unwrap)
            .find(|block| block.slots().unwrap().contains(&slot))
            .unwrap();
        let super_block = super_block.parse(only(super_block.parse_range())).unwrap();
        let data_block = super_block
            .data_blocks()
            .map(Result::unwrap)
            .find(|block| block.slots().unwrap().contains(&slot))
            .unwrap();
        let ExtensibleArrayDataBlock::Pages(pages) =
            data_block.parse(only(data_block.parse_range())).unwrap()
        else {
            panic!("the data block {data_block:?} is not paged");
        };
        let page = pages
            .pages()
            .map(Result::unwrap)
            .find(|page| page.slots().unwrap().contains(&slot))
            .unwrap();
        let records: Vec<(u64, ChunkRecord)> = page
            .parse(only(page.parse_range()))
            .unwrap()
            .records(8)
            .collect::<Result<_, _>>()
            .unwrap();
        let expected: Vec<(u64, ChunkRecord)> =
            chunk_record::slotted(chunk_record::chunks(8, n, true))
                .into_iter()
                .skip(usize::try_from(FIRST_PAGED_SLOT + 2 * 2_048 + 1_024).unwrap())
                .take(1_024)
                .collect();
        assert_eq!(records, expected);
    }

    #[test]
    fn an_undefined_index_block_address_leaves_the_array_without_an_index_block() {
        let mut fixture = Fixture::new(5, false);
        let header = fixture.header().unwrap().range();
        let at = fixture.local(header.start()) + 12 + 6 * 8;
        fixture.bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        fixture.restamp(header);
        let array = fixture.header().unwrap();
        assert_eq!(array.index_block(), None);
        assert!(
            array
                .header()
                .index_block_address
                .is_undefined(WIDTHS.offsets)
        );
        assert_eq!(array.header().max_idx_set, 5);
        assert_eq!(array.range().length(), 12 + 6 * 8 + 8 + 4);
        assert_eq!(fixture.records(8).unwrap(), Vec::new());
    }

    #[rstest]
    #[case::unfiltered(false)]
    #[case::filtered(true)]
    fn the_header_reports_whether_its_chunks_are_filtered(#[case] has_filters: bool) {
        assert_eq!(
            Fixture::new(5, has_filters).header().unwrap().is_filtered(),
            has_filters
        );
    }

    #[rstest]
    #[case::two_byte_fields(OffsetWidth::Two, LengthWidth::Two)]
    #[case::four_byte_fields(OffsetWidth::Four, LengthWidth::Four)]
    #[case::eight_byte_fields(OffsetWidth::Eight, LengthWidth::Eight)]
    #[case::short_offsets_long_lengths(OffsetWidth::Four, LengthWidth::Eight)]
    #[case::long_offsets_short_lengths(OffsetWidth::Eight, LengthWidth::Four)]
    fn the_region_parser_preserves_append_metadata_and_geometry(
        #[case] offsets: OffsetWidth,
        #[case] lengths: LengthWidth,
        #[values(0, 1 << 40)] base: u64,
        #[values(false, true)] filtered: bool,
        #[values(
            (2, &[16, 32][..], 2, 27),
            (4, &[16, 32, 32, 32, 64, 64][..], 4, 25),
            (
                8,
                &[16, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128][..],
                6,
                23
            ),
            (
                16,
                &[
                    16, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128,
                    128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256,
                    256, 256, 256, 256
                ][..],
                8,
                21
            )
        )]
        expected: (u8, &[u64], usize, usize),
    ) {
        let (minimum_pointers, direct_blocks, first_secondary, secondary_pointers) = expected;
        let widths = FormatWidths { offsets, lengths };
        let mut fixture = Fixture::at(base, widths, 8, 20, filtered);
        let range = fixture.locator().parse_range().unwrap();
        let local = fixture.local(range.start());
        fixture.bytes[local + HEADER_MINIMUM_POINTERS] = minimum_pointers;
        fixture.restamp(range);
        let header_len = SIGNATURE_LEN
            + HEADER_BYTE_FIELDS
            + STATISTICS * u64::from(lengths.get())
            + u64::from(offsets.get())
            + CHECKSUM_LEN;
        let expected = ExtensibleArrayHeader {
            client_id: u8::from(filtered),
            element_size: offsets.get() + if filtered { 2 + FILTER_MASK_BYTES } else { 0 },
            bit_geometry: ExtensibleArrayBitGeometry {
                max_index_bits: MaxIndexBits(NonZeroU8::new(32).unwrap()),
                page_bits: PageBits(10),
            },
            idx_blk_elmts: 4,
            min_dblk_nelmts: 16,
            super_blk_min_data_ptrs: minimum_pointers,
            max_idx_set: 20,
            index_block_address: StoredAddress::new(ARRAY_AT + header_len),
        };
        let array = fixture.header().unwrap();
        assert_eq!(*array.header(), expected);
        assert_eq!(array.range().start().get(), base + ARRAY_AT);
        assert_eq!(array.range().length(), header_len);
        assert_eq!(
            array.index_block().unwrap().parse_range().start().get(),
            base + expected.index_block_address.get()
        );
        let actual = array.geometry().unwrap();
        assert_eq!(
            actual.sblks(),
            &[
                (1, 16),
                (1, 32),
                (2, 32),
                (2, 64),
                (4, 64),
                (4, 128),
                (8, 128),
                (8, 256),
                (16, 256),
                (16, 512),
                (32, 512),
                (32, 1_024),
                (64, 1_024),
                (64, 2_048),
                (128, 2_048),
                (128, 4_096),
                (256, 4_096),
                (256, 8_192),
                (512, 8_192),
                (512, 16_384),
                (1_024, 16_384),
                (1_024, 32_768),
                (2_048, 32_768),
                (2_048, 65_536),
                (4_096, 65_536),
                (4_096, 131_072),
                (8_192, 131_072),
                (8_192, 262_144),
                (16_384, 262_144),
            ]
        );
        assert_eq!(actual.direct_dblk_nelmts(), direct_blocks);
        assert_eq!(actual.nsblk_addrs(), secondary_pointers);
        assert_eq!(actual.first_indirect_sblk(), first_secondary);
    }

    #[rstest]
    #[case::signature(&[(0, b'X')], FormatError::InvalidExtensibleArraySignature)]
    #[case::version(&[(4, 1)], FormatError::InvalidExtensibleArrayVersion(1))]
    #[case::client(&[(5, 2)], FormatError::InvalidExtensibleArrayClient(2))]
    #[case::filtered_element_too_small(&[(6, 8 + 4 - 1)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::filtered_element_too_large(&[(6, 8 + 4 + 9)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::no_index_bits(&[(7, 0)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::index_bits_past_64(&[(7, 65)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::no_min_elements(&[(9, 0)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::min_elements_not_a_power_of_two(&[(9, 24)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::min_elements_past_the_index_bits(&[(7, 2), (11, 2)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::one_min_data_block(&[(10, 1)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::min_data_blocks_not_a_power_of_two(&[(10, 6)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::direct_data_blocks_past_the_super_blocks(&[(7, 8), (10, 16), (11, 8)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::paged_direct_data_block(&[(11, 5)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::page_bits_past_the_index_bits(&[(7, 9), (11, 10)], FormatError::InvalidExtensibleArrayGeometry)]
    #[case::page_bits_64(&[(7, 64), (11, 64)], FormatError::InvalidExtensibleArrayGeometry)]
    fn a_malformed_header_is_rejected(
        #[case] fields: &[(usize, u8)],
        #[case] expected: FormatError,
    ) {
        let mut fixture = Fixture::new(5, true);
        let header = fixture.header().unwrap().range();
        for &(field, value) in fields {
            let at = fixture.local(header.start()) + field;
            fixture.bytes[at] = value;
        }
        fixture.restamp(header);
        assert_eq!(fixture.header().unwrap_err(), expected);
    }

    #[derive(Clone, Copy, Debug)]
    enum Block {
        Index,
        Super,
        Data,
        PagedData,
    }

    impl Block {
        fn fixture(self) -> Fixture {
            Fixture::new(
                match self {
                    Self::Index | Self::Data => 20,
                    Self::Super => 2_000,
                    Self::PagedData => 140_000,
                },
                false,
            )
        }

        fn range(self, fixture: &Fixture) -> FileRange {
            match self {
                Self::Index => fixture
                    .header()
                    .unwrap()
                    .index_block()
                    .unwrap()
                    .parse_range(),
                Self::Super => fixture.super_blocks()[0].parse_range(),
                Self::Data => fixture.data_blocks()[0].parse_range(),
                Self::PagedData => fixture.paged_data_blocks()[0].parse_range(),
            }
        }

        fn offset(self) -> u64 {
            match self {
                Self::Index | Self::Data => 0,
                Self::Super => 240,
                Self::PagedData => FIRST_PAGED_SLOT - 4,
            }
        }

        fn parse(self, fixture: &Fixture, range: FileRange) -> FormatError {
            let region = fixture.read(range).unwrap();
            match self {
                Self::Index => {
                    let block = fixture.header().unwrap().index_block().unwrap();
                    block.parse(region).unwrap_err()
                }
                Self::Super => fixture.super_blocks()[0].parse(region).unwrap_err(),
                Self::Data => fixture.data_blocks()[0].parse(region).unwrap_err(),
                Self::PagedData => fixture.paged_data_blocks()[0].parse(region).unwrap_err(),
            }
        }
    }

    #[rstest]
    #[case::signature(0, b"X", FormatError::InvalidExtensibleArraySignature)]
    #[case::version(4, &[1], FormatError::InvalidExtensibleArrayVersion(1))]
    #[case::client(5, &[1], FormatError::InvalidExtensibleArrayBlockClient { expected: 0, actual: 1 })]
    #[case::header_address(6, &[0x81], FormatError::InvalidExtensibleArrayBlockOwner { expected: ARRAY_AT, actual: 0x81 })]
    fn a_malformed_block_prefix_is_rejected(
        #[case] field: usize,
        #[case] value: &[u8],
        #[case] expected: FormatError,
        #[values(Block::Index, Block::Super, Block::Data, Block::PagedData)] block: Block,
    ) {
        let mut fixture = block.fixture();
        let range = block.range(&fixture);
        let at = fixture.local(range.start()) + field;
        fixture.bytes[at..at + value.len()].copy_from_slice(value);
        fixture.restamp(range);
        assert_eq!(block.parse(&fixture, range), expected);
    }

    #[rstest]
    fn a_block_offset_other_than_its_position_is_rejected(
        #[values(Block::Super, Block::Data, Block::PagedData)] block: Block,
    ) {
        let mut fixture = block.fixture();
        let range = block.range(&fixture);
        let at = fixture.local(range.start()) + 4 + 2 + 8;
        let offset = block.offset();
        fixture.bytes[at..at + 4]
            .copy_from_slice(&u32::try_from(offset + 1).unwrap().to_le_bytes());
        fixture.restamp(range);
        assert_eq!(
            block.parse(&fixture, range),
            FormatError::InvalidExtensibleArrayBlockOffset {
                expected: offset,
                actual: offset + 1,
            }
        );
    }

    // Data block 1 is the one data block of super block 1, at offset 16, and data block 3 the
    // second of super block 2, at offset 80.
    #[rstest]
    #[case::spec_value(1, 16, None)]
    #[case::libhdf5_value(1, 48, None)]
    #[case::other_value(1, 17, Some(16))]
    #[case::spec_value_of_a_later_block(3, 80, None)]
    #[case::libhdf5_value_of_a_later_block(3, 48 + 3 * 32, None)]
    #[case::other_value_of_a_later_block(3, 48 + 2 * 32 + 1, Some(80))]
    fn a_data_block_the_index_block_holds_takes_either_offset(
        #[case] block: usize,
        #[case] offset: u32,
        #[case] rejected: Option<u64>,
    ) {
        let mut fixture = Fixture::new(244, true);
        let range = fixture.data_blocks()[block].parse_range();
        let at = fixture.local(range.start()) + 4 + 2 + 8;
        fixture.bytes[at..at + 4].copy_from_slice(&offset.to_le_bytes());
        fixture.restamp(range);
        let records = fixture.records(8);
        match rejected {
            Some(expected) => assert_eq!(
                records.unwrap_err(),
                FormatError::InvalidExtensibleArrayBlockOffset {
                    expected,
                    actual: u64::from(offset),
                }
            ),
            None => assert_eq!(
                records.unwrap(),
                chunk_record::slotted(chunk_record::chunks(8, 244, true))
            ),
        }
    }

    // Data block 1 of super block 4 is the 8th data block in the array, and starts 64 elements
    // past the super block's offset of 240.
    #[test]
    fn a_data_block_a_super_block_holds_takes_its_offset_alone() {
        let mut fixture = Fixture::new(500, false);
        let block = fixture.data_blocks()[6 + 1];
        assert_eq!(block.slots().unwrap(), 4 + 240 + 64..4 + 240 + 2 * 64);
        let range = block.parse_range();
        let at = fixture.local(range.start()) + 4 + 2 + 8;
        let ordinal_offset = 240u32 + 7 * 64;
        fixture.bytes[at..at + 4].copy_from_slice(&ordinal_offset.to_le_bytes());
        fixture.restamp(range);
        assert_eq!(
            fixture.records(8).unwrap_err(),
            FormatError::InvalidExtensibleArrayBlockOffset {
                expected: 240 + 64,
                actual: u64::from(ordinal_offset),
            }
        );
    }

    #[rstest]
    #[case::zero_minimum_elements(0, 4, 10)]
    #[case::non_power_of_two_elements(3, 4, 10)]
    #[case::zero_minimum_pointers(16, 0, 10)]
    #[case::single_minimum_pointer(16, 1, 10)]
    #[case::non_power_of_two_pointers(16, 3, 10)]
    #[case::direct_block_exceeds_page(16, 4, 4)]
    fn append_geometry_rejects_invalid_creation_parameters(
        #[case] min_elements: u8,
        #[case] minimum_pointers: u8,
        #[case] page_bits: u8,
    ) {
        let header = ExtensibleArrayHeader {
            client_id: EA_CLIENT_ID_UNFILTERED,
            element_size: OffsetWidth::Eight.get(),
            bit_geometry: ExtensibleArrayBitGeometry::parse(EA_MAX_NELMTS_BITS, page_bits).unwrap(),
            idx_blk_elmts: EA_IDX_BLK_ELMTS,
            min_dblk_nelmts: min_elements,
            super_blk_min_data_ptrs: minimum_pointers,
            max_idx_set: 0,
            index_block_address: StoredAddress::new(0),
        };
        assert_eq!(
            ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap_err(),
            FormatError::InvalidExtensibleArrayGeometry
        );
    }

    #[rstest]
    fn shared_geometry_agrees_on_block_page_and_bitmap_lengths(
        #[values(2, 4, 8, 16)] minimum_pointers: u8,
        #[values(OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
        #[values(false, true)] filtered: bool,
    ) {
        let element_size = offsets.get() + if filtered { 6 } else { 0 };
        let client_id = if filtered {
            ArrayClient::FilteredChunks { chunk_size_len: 2 }.id()
        } else {
            EA_CLIENT_ID_UNFILTERED
        };
        let header = ExtensibleArrayHeader {
            client_id,
            element_size,
            bit_geometry: ExtensibleArrayBitGeometry::for_writer(),
            idx_blk_elmts: EA_IDX_BLK_ELMTS,
            min_dblk_nelmts: EA_MIN_DBLK_NELMTS,
            super_blk_min_data_ptrs: minimum_pointers,
            max_idx_set: 0,
            index_block_address: StoredAddress::new(0),
        };
        let geometry = ExtensibleArrayGeometry::from_header(&header, offsets).unwrap();
        let layout = Layout::new(
            ElementEncoding::new(offsets, client_id, element_size).unwrap(),
            EA_MAX_NELMTS_BITS,
            EA_IDX_BLK_ELMTS,
            EA_MIN_DBLK_NELMTS,
            minimum_pointers,
            EA_MAX_DBLK_NELMTS_BITS,
        )
        .unwrap();
        assert_eq!(
            layout.index_block_len().unwrap(),
            u64::try_from(index_block_len(
                offsets,
                usize::from(EA_IDX_BLK_ELMTS),
                usize::from(element_size),
                geometry.direct_dblk_nelmts().len(),
                geometry.nsblk_addrs(),
            ))
            .unwrap()
        );
        for ordinal in 0..geometry.nsblk_addrs() {
            let writer = geometry.super_block_at(ordinal, header.page_nelmts().get());
            let reader = layout
                .super_block(u32::try_from(geometry.first_indirect_sblk() + ordinal).unwrap())
                .unwrap();
            assert_eq!(layout.bitmap_len(reader).unwrap(), writer.bitmap_size());
            assert_eq!(
                layout.super_block_len(reader).unwrap(),
                super_block_len(writer, offsets, header.block_offset_width())
            );
            assert_eq!(
                layout
                    .data_block_len(reader.data_block(0).unwrap())
                    .unwrap(),
                data_block_len(
                    writer.blocks,
                    usize::from(element_size),
                    offsets,
                    header.block_offset_width()
                )
            );
            assert_eq!(
                layout.pages(reader.block_elements).unwrap_or(0),
                writer.blocks.npages()
            );
        }
    }

    #[rstest]
    #[case::two_pointer_minimum(2, 2, 2, 282)]
    #[case::four_pointer_minimum(4, 6, 4, 298)]
    #[case::eight_pointer_minimum(8, 14, 6, 346)]
    #[case::sixteen_pointer_minimum(16, 30, 8, 458)]
    fn reader_and_writer_geometry_agree_on_pointer_counts(
        #[case] min_data_pointers: u8,
        #[case] direct_blocks: usize,
        #[case] first_secondary: usize,
        #[case] index_bytes: usize,
    ) {
        let header = ExtensibleArrayHeader {
            client_id: 0,
            element_size: 8,
            bit_geometry: ExtensibleArrayBitGeometry::for_writer(),
            idx_blk_elmts: EA_IDX_BLK_ELMTS,
            min_dblk_nelmts: EA_MIN_DBLK_NELMTS,
            super_blk_min_data_ptrs: min_data_pointers,
            max_idx_set: 0,
            index_block_address: StoredAddress::new(0),
        };
        let geometry = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        assert_eq!(geometry.direct_dblk_nelmts().len(), direct_blocks);
        assert_eq!(geometry.first_indirect_sblk(), first_secondary);
        assert_eq!(
            index_block_len(
                OffsetWidth::Eight,
                usize::from(EA_IDX_BLK_ELMTS),
                8,
                geometry.direct_dblk_nelmts().len(),
                geometry.nsblk_addrs()
            ),
            index_bytes
        );
        let layout = Layout::new(
            ElementEncoding::new(OffsetWidth::Eight, 0, 8).unwrap(),
            EA_MAX_NELMTS_BITS,
            EA_IDX_BLK_ELMTS,
            EA_MIN_DBLK_NELMTS,
            min_data_pointers,
            EA_MAX_DBLK_NELMTS_BITS,
        )
        .unwrap();
        assert_eq!(
            layout
                .direct_blocks()
                .map(|block| block.map(|block| block.elements))
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            geometry.direct_dblk_nelmts()
        );
        assert_eq!(
            usize::try_from(layout.first_super_block).unwrap(),
            geometry.first_indirect_sblk()
        );
        assert_eq!(
            usize::try_from(layout.super_blocks - layout.first_super_block).unwrap(),
            geometry.nsblk_addrs()
        );
        for (index, &(blocks, elements)) in geometry.sblks().iter().enumerate() {
            let super_block = layout.super_block(u32::try_from(index).unwrap()).unwrap();
            assert_eq!(
                (super_block.blocks, super_block.block_elements),
                (blocks, elements)
            );
        }
    }

    #[rstest]
    #[case::plain(0)]
    #[case::userblock(512)]
    #[case::above_u32(1 << 40)]
    fn an_array_reads_at_its_physical_position(
        #[case] base: u64,
        #[values(2_000, 140_000)] n: u64,
    ) {
        let fixture = Fixture::at(base, WIDTHS, 8, n, true);
        assert_eq!(
            fixture.header().unwrap().range().start().get(),
            base + ARRAY_AT
        );
        assert_eq!(
            fixture.records(8).unwrap(),
            chunk_record::slotted(chunk_record::chunks(8, n, true))
        );
    }

    #[rstest]
    fn a_header_address_at_the_physical_header_position_is_rejected(
        #[values(Block::Index, Block::Super, Block::Data, Block::PagedData)] block: Block,
    ) {
        let base = 1 << 40;
        let mut fixture = block.fixture();
        fixture.base = base;
        let range = block.range(&fixture);
        let at = fixture.local(range.start()) + 6;
        fixture.bytes[at..at + 8].copy_from_slice(&(base + ARRAY_AT).to_le_bytes());
        fixture.restamp(range);
        assert_eq!(
            block.parse(&fixture, range),
            FormatError::InvalidExtensibleArrayBlockOwner {
                expected: base + ARRAY_AT,
                actual: 2 * base + ARRAY_AT,
            }
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Truncated {
        Header,
        IndexBlock,
        SuperBlock,
        DataBlock,
        PagedDataBlock,
        Page,
    }

    #[rstest]
    fn a_region_short_of_its_range_is_rejected(
        #[values(
            Truncated::Header,
            Truncated::IndexBlock,
            Truncated::SuperBlock,
            Truncated::DataBlock,
            Truncated::PagedDataBlock,
            Truncated::Page
        )]
        truncated: Truncated,
    ) {
        let fixture = Fixture::new(140_000, false);
        let short = |range: FileRange| {
            fixture
                .read(FileRange::new(range.start(), range.length() - 1).unwrap())
                .unwrap()
        };
        let (range, err) = match truncated {
            Truncated::Header => {
                let locator = fixture.locator();
                let range = locator.parse_range().unwrap();
                (range, locator.parse(short(range)).unwrap_err())
            }
            Truncated::IndexBlock => {
                let block = fixture.header().unwrap().index_block().unwrap();
                let range = block.parse_range();
                (range, block.parse(short(range)).unwrap_err())
            }
            Truncated::SuperBlock => {
                let block = fixture.super_blocks()[0];
                let range = block.parse_range();
                (range, block.parse(short(range)).unwrap_err())
            }
            Truncated::DataBlock => {
                let block = fixture.data_blocks()[0];
                let range = block.parse_range();
                (range, block.parse(short(range)).unwrap_err())
            }
            Truncated::PagedDataBlock => {
                let block = fixture.paged_data_blocks()[0];
                let range = block.parse_range();
                (range, block.parse(short(range)).unwrap_err())
            }
            Truncated::Page => {
                let page = *fixture
                    .pages(*fixture.paged_data_blocks().last().unwrap())
                    .last()
                    .unwrap();
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

    #[cfg(feature = "checksum")]
    #[derive(Clone, Copy, Debug)]
    enum Corrupted {
        Header,
        IndexBlock,
        SuperBlock,
        DataBlock,
        PagedDataBlock,
        FirstPage,
        LastPage,
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    fn a_corrupted_checksum_is_rejected(
        #[values(
            Corrupted::Header,
            Corrupted::IndexBlock,
            Corrupted::SuperBlock,
            Corrupted::DataBlock,
            Corrupted::PagedDataBlock,
            Corrupted::FirstPage,
            Corrupted::LastPage
        )]
        corrupted: Corrupted,
        #[values(false, true)] has_filters: bool,
    ) {
        let mut fixture = Fixture::new(140_000, has_filters);
        let range = match corrupted {
            Corrupted::Header => fixture.header().unwrap().range(),
            Corrupted::IndexBlock => fixture
                .header()
                .unwrap()
                .index_block()
                .unwrap()
                .parse_range(),
            Corrupted::SuperBlock => fixture.super_blocks()[0].parse_range(),
            Corrupted::DataBlock => fixture.data_blocks()[0].parse_range(),
            Corrupted::PagedDataBlock => fixture.paged_data_blocks()[0].parse_range(),
            Corrupted::FirstPage => fixture.pages(fixture.paged_data_blocks()[0])[0].parse_range(),
            Corrupted::LastPage => fixture
                .pages(*fixture.paged_data_blocks().last().unwrap())
                .last()
                .unwrap()
                .parse_range(),
        };
        let start = fixture.local(range.start());
        let end = fixture.local(range.end());
        fixture.bytes[end - 1] ^= 0x01;
        let stored = u32::from_le_bytes(fixture.bytes[end - 4..end].try_into().unwrap());
        let computed = checksum::jenkins_lookup3(&fixture.bytes[start..end - 4]);
        assert_eq!(
            fixture.records(8).unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            }
        );
    }

    #[rstest]
    #[case::two_min_data_blocks(2, &[(0, 16), (16, 32)], 2)]
    #[case::four_min_data_blocks(4, &[(0, 16), (16, 32), (48, 32), (80, 32), (112, 64), (176, 64)], 4)]
    #[case::eight_min_data_blocks(
        8,
        &[
            (0, 16), (16, 32), (48, 32), (80, 32), (112, 64), (176, 64),
            (240, 64), (304, 64), (368, 64), (432, 64),
            (496, 128), (624, 128), (752, 128), (880, 128),
        ],
        6
    )]
    fn the_index_block_holds_the_data_blocks_before_the_first_super_block(
        #[case] min_data_blocks: u8,
        #[case] direct_blocks: &[(u64, u64)],
        #[case] first_super_block: u32,
    ) {
        let encoding = ElementEncoding::new(OffsetWidth::Eight, 0, 8).unwrap();
        let layout = Layout::new(encoding, 32, 4, 16, min_data_blocks, 10).unwrap();
        assert_eq!(layout.first_super_block, first_super_block);
        assert_eq!(
            layout
                .direct_blocks()
                .map(|block| block.map(|block| (block.offset, block.elements)))
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            direct_blocks
        );
        assert_eq!(layout.direct_data_blocks(), direct_blocks.len() as u64);
        assert_eq!(
            layout.index_block_len().unwrap(),
            4 + 2
                + 8
                + 4 * 8
                + (direct_blocks.len() as u64 + u64::from(29 - first_super_block)) * 8
                + 4
        );
    }

    // An unpaged data block is the prefix, the block offset, the elements, and the checksum, and a
    // paged one the prefix, the block offset, the checksum, and the pages with a checksum each.
    #[rstest]
    #[case::no_elements(OffsetWidth::Four, 0, 4, 10, 0, 4 + 2 + 4 + 2 + 4)]
    #[case::four_byte_addresses(OffsetWidth::Four, 0, 4, 16, 255, 4 + 2 + 4 + 2 + 255 * 4 + 4)]
    #[case::filtered_four_byte_addresses(OffsetWidth::Four, 1, 10, 32, 16, 4 + 2 + 4 + 4 + 16 * 10 + 4)]
    #[case::one_page(OffsetWidth::Eight, 0, 8, 32, 1_024, 4 + 2 + 8 + 4 + 1_024 * 8 + 4)]
    #[case::two_pages(OffsetWidth::Eight, 1, 14, 32, 2_048, 4 + 2 + 8 + 4 + 4 + 2 * (1_024 * 14 + 4))]
    #[case::four_pages_four_byte_addresses(OffsetWidth::Four, 0, 4, 16, 4_096, 4 + 2 + 4 + 2 + 4 + 4 * (1_024 * 4 + 4))]
    fn a_data_block_has_the_length_of_its_fields(
        #[case] offsets: OffsetWidth,
        #[case] client_id: u8,
        #[case] element_size: u8,
        #[case] max_index_bits: u8,
        #[case] elements: u64,
        #[case] expected: u64,
    ) {
        let encoding = ElementEncoding::new(offsets, client_id, element_size).unwrap();
        let layout = Layout::new(encoding, max_index_bits, 4, 4, 2, 10).unwrap();
        assert_eq!(
            layout
                .data_block_len(DataBlock {
                    offset: 0,
                    elements,
                    libhdf5_offset: None,
                })
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::one_byte(1, 1)]
    #[case::byte_boundary(8, 1)]
    #[case::two_bytes(9, 2)]
    #[case::four_bytes(32, 4)]
    #[case::eight_bytes(64, 8)]
    fn the_block_offset_is_the_index_bits_in_whole_bytes(#[case] bits: u8, #[case] len: u8) {
        let encoding = ElementEncoding::new(OffsetWidth::Eight, 0, 8).unwrap();
        assert_eq!(
            Layout::new(encoding, bits, 0, 1, 2, 1)
                .unwrap()
                .block_offset_len,
            len
        );
    }

    // An array of 64 index bits whose one chunk is in page 255 of data block 1 of super block 28,
    // past slot `u32::MAX`, and that allocates no other block.
    struct SparseArray {
        blocks: Vec<(u64, Vec<u8>)>,
    }

    impl SparseArray {
        const SLOT: u64 = 4_295_490_548;

        fn new(base: u64) -> Self {
            let header_len = 12 + 6 * 8 + 8 + 4;
            let index_block_at = base + header_len;
            // The prefix, 4 elements, 6 data block addresses, and 57 super block addresses.
            let index_block_len = 14 + 4 * 8 + (6 + 57) * 8 + 4;
            let super_block_at = index_block_at + index_block_len;
            // The prefix, the block offset, 256 pages for each of 16_384 data blocks, the data
            // block addresses, and the checksum.
            let bitmap_len = 16_384 * 32;
            let super_block_len = 14 + 8 + bitmap_len + 16_384 * 8 + 4;
            let data_block_at = super_block_at + super_block_len;

            let mut header = Vec::new();
            header.extend_from_slice(b"EAHD");
            header.extend_from_slice(&[0, 0, 8, 64, 4, 16, 4, 10]);
            for statistic in [
                1,
                super_block_len,
                1,
                26 + 256 * 8_196,
                Self::SLOT + 1,
                4 + 262_144,
            ] {
                header.extend_from_slice(&statistic.to_le_bytes());
            }
            header.extend_from_slice(&index_block_at.to_le_bytes());
            test_util::checksum::append(&mut header);

            let mut index_block = Vec::new();
            index_block.extend_from_slice(b"EAIB");
            index_block.extend_from_slice(&[0, 0]);
            index_block.extend_from_slice(&base.to_le_bytes());
            index_block.extend_from_slice(&[0xff; 4 * 8]);
            for block in 0..6 + 57 {
                let address = if block == 6 + 24 {
                    super_block_at
                } else {
                    u64::MAX
                };
                index_block.extend_from_slice(&address.to_le_bytes());
            }
            test_util::checksum::append(&mut index_block);

            let mut super_block = Vec::new();
            super_block.extend_from_slice(b"EASB");
            super_block.extend_from_slice(&[0, 0]);
            super_block.extend_from_slice(&base.to_le_bytes());
            super_block.extend_from_slice(&4_294_967_280u64.to_le_bytes());
            let mut bitmap = vec![0; usize::try_from(bitmap_len).unwrap()];
            bitmap[63] = 0x01;
            super_block.extend_from_slice(&bitmap);
            for data_block in 0..16_384 {
                let address = if data_block == 1 {
                    data_block_at
                } else {
                    u64::MAX
                };
                super_block.extend_from_slice(&address.to_le_bytes());
            }
            test_util::checksum::append(&mut super_block);

            let mut data_block = Vec::new();
            data_block.extend_from_slice(b"EADB");
            data_block.extend_from_slice(&[0, 0]);
            data_block.extend_from_slice(&base.to_le_bytes());
            data_block.extend_from_slice(&4_295_229_424u64.to_le_bytes());
            test_util::checksum::append(&mut data_block);
            let mut page = vec![0xff; 1_024 * 8];
            page[..8].copy_from_slice(&0x20u64.to_le_bytes());
            test_util::checksum::append(&mut page);
            let page_at = data_block_at + 26 + 255 * 8_196;

            Self {
                blocks: vec![
                    (base, header),
                    (index_block_at, index_block),
                    (super_block_at, super_block),
                    (data_block_at, data_block),
                    (page_at, page),
                ],
            }
        }

        fn read(&self, range: FileRange) -> FileRegion<'_> {
            let (at, bytes) = self
                .blocks
                .iter()
                .find(|&&(at, _)| at == range.start().get())
                .unwrap();
            FileRegion::new(
                FileRange::new(PhysicalAddress::new(*at), bytes.len() as u64).unwrap(),
                bytes,
            )
            .unwrap()
            .subregion(range)
            .unwrap()
        }
    }

    #[rstest]
    fn a_slot_past_u32_max_is_read(#[values(0, 1 << 32)] base: u64) {
        let array = SparseArray::new(base);
        let locator =
            ExtensibleArrayHeaderLocator::new(WIDTHS, BaseAddress::ZERO, StoredAddress::new(base));
        let header = locator
            .parse(array.read(locator.parse_range().unwrap()))
            .unwrap();
        let index_block = header.index_block().unwrap();
        let index_block = index_block
            .parse(array.read(index_block.parse_range()))
            .unwrap();
        assert_eq!(index_block.data_blocks().count(), 0);
        let super_blocks: Vec<ExtensibleArraySuperBlockRef> = index_block
            .super_blocks()
            .collect::<Result<_, _>>()
            .unwrap();
        let &[super_block] = &super_blocks[..] else {
            panic!("expected one super block, got {super_blocks:?}");
        };
        let super_block = super_block
            .parse(array.read(super_block.parse_range()))
            .unwrap();
        let data_blocks: Vec<ExtensibleArrayDataBlockRef<'_>> =
            super_block.data_blocks().collect::<Result<_, _>>().unwrap();
        let &[data_block] = &data_blocks[..] else {
            panic!("expected one data block, got {data_blocks:?}");
        };
        let ExtensibleArrayDataBlock::Pages(pages) = data_block
            .parse(array.read(data_block.parse_range()))
            .unwrap()
        else {
            panic!("the data block {data_block:?} is not paged");
        };
        let pages: Vec<ExtensibleArrayPageRef> = pages.pages().collect::<Result<_, _>>().unwrap();
        let &[page] = &pages[..] else {
            panic!("expected one page, got {pages:?}");
        };
        let records: Vec<(u64, ChunkRecord)> = page
            .parse(array.read(page.parse_range()))
            .unwrap()
            .records(8)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            records,
            vec![(
                SparseArray::SLOT,
                ChunkRecord {
                    address: StoredAddress::new(0x20),
                    stored_size: 8,
                    filter_mask: 0,
                }
            )]
        );
    }
}
