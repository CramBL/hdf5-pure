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
use alloc::vec::Vec;
use core::ops::Range;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use self::write::ExtensibleArrayPlan;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::chunk_record;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::chunk_record::IndexSlots;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[cfg(test)]
mod measurement_sizes;

mod creation;
mod geometry;
mod header;
mod statistics;
pub(crate) mod write;

pub use creation::{
    BlockOffsetWidth, ExtensibleArrayBitGeometry, ExtensibleArrayCreationParameters, MaxIndexBits,
    PageElementCount,
};
pub(crate) use creation::{EA_IDX_BLK_ELMTS, EA_MAX_NELMTS_BITS, EA_MIN_DBLK_NELMTS};
use geometry::{DataBlock, Layout, SuperBlock};
pub use geometry::{DataBlockGeometry, ExtensibleArrayGeometry, SuperBlockGeometry};
pub use header::{ExtensibleArray, ExtensibleArrayHeader, ExtensibleArrayHeaderLocator};
use header::{HEADER_BYTE_FIELDS, HEADER_PREFIX_LEN, HEADER_SIGNATURE};
pub use statistics::ExtensibleArrayStats;

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
            parse_range: if layout.pages(block.elements()).is_some() {
                FileRange::new(address, layout.pages_offset())?
            } else {
                range
            },
            signature: DATA_BLOCK_SIGNATURE,
            offset: (block.offset(), block.libhdf5_offset()),
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
        let mut cursor = owner.parse_prefix(prefix.bytes(), signature)?;
        let (expected, alternate) = offset;
        owner.parse_block_offset(&mut cursor, expected, alternate)?;
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
            self.owner.layout.prefix_len() + u64::from(self.owner.layout.block_offset_len()),
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
        let cursor = owner.parse_prefix(bytes, INDEX_BLOCK_SIGNATURE)?;
        checksum::verify_trailing(bytes)?;
        let layout = owner.layout;
        let (elements, addresses) = body(bytes, cursor.position())?
            .split_at_checked(
                layout
                    .elements_len(u64::from(layout.inline_elements()))?
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
                (layout.first_super_block()..layout.super_blocks())
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
            offset: (super_block.offset(), None),
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
        if local >= super_block.blocks() {
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
        let mut cursor = owner.parse_prefix(bytes, SUPER_BLOCK_SIGNATURE)?;
        owner.parse_block_offset(&mut cursor, super_block.offset(), None)?;
        checksum::verify_trailing(bytes)?;
        let (bitmap, data_blocks) = body(bytes, cursor.position())?
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
        self.owner.holds_set_slots(self.super_block.offset())
    }

    /// Returns the slots of the data blocks whose addresses the super block holds.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the range ends past
    /// [`u64::MAX`].
    pub fn slots(&self) -> Result<Range<u64>, FormatError> {
        self.super_block.slots(self.owner.layout)
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
            .zip((0..super_block.blocks()).map(move |local| (local, super_block.data_block(local))))
            .filter_map(move |(address, (local, block))| {
                address.transpose().map(|address| {
                    let block = block?;
                    let pages = match owner.layout.pages(block.elements()) {
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
        let mut cursor = owner.parse_prefix(bytes, DATA_BLOCK_SIGNATURE)?;
        // For a data block the index block holds the address of, `H5EA__lookup_elmt` (`H5EA.c`,
        // HDF5 1.10.11 to 2.2.0) stores the offset of its super block plus its ordinal among those
        // data blocks times its element count, which differs from its position from super block 1
        // on.
        owner.parse_block_offset(&mut cursor, block.offset(), block.libhdf5_offset())?;
        checksum::verify_trailing(bytes)?;
        Ok(match self.pages()? {
            Some(pages) => ExtensibleArrayDataBlock::Pages(pages),
            None => {
                ExtensibleArrayDataBlock::Elements(ExtensibleArrayElements::below_max_index_set(
                    owner,
                    body(bytes, cursor.position())?,
                    owner.layout.first_slot(block.offset())?,
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
                    first_slot: owner.layout.first_slot(block.offset())?,
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
        self.owner.holds_set_slots(self.block.offset())
    }

    /// Returns the slots of the data block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the range ends past
    /// [`u64::MAX`].
    pub fn slots(&self) -> Result<Range<u64>, FormatError> {
        self.block.slots(self.owner.layout)
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
        let page_elements = layout.page_nelmts().get();
        (0..bitmap.pages)
            .take_while(move |&page| {
                page.checked_mul(page_elements)
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
                    first_slot: sum([first_slot, product(page, page_elements)?])?,
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
        let elements = self.owner.layout.page_nelmts().get();
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
        let encoding = owner.layout.encoding();
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
        let offsets = self.layout.encoding().offsets();
        addresses
            .chunks_exact(usize::from(offsets.get()))
            .map(move |field| {
                Ok(
                    match bytes::ByteCursor::new(field).decoded_address(offsets)? {
                        DecodedStoredAddress::Defined(address) => Some(self.resolve(address)?),
                        DecodedStoredAddress::Undefined => None,
                    },
                )
            })
    }

    /// Parses the signature, the version, the client ID, and the header address of a block, and
    /// returns a cursor after those fields.
    fn parse_prefix<'a, const N: usize>(
        self,
        bytes: &'a [u8],
        signature: [u8; N],
    ) -> Result<bytes::ByteCursor<'a>, FormatError> {
        let mut cursor = bytes::ByteCursor::new(bytes);
        if cursor.array::<N>()? != signature {
            return Err(FormatError::InvalidExtensibleArraySignature);
        }
        let version = cursor.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidExtensibleArrayVersion(version));
        }
        let client_id = cursor.u8()?;
        let expected = self.layout.encoding().client().id();
        if client_id != expected {
            return Err(FormatError::InvalidExtensibleArrayBlockClient {
                expected,
                actual: client_id,
            });
        }
        let owner = self.resolve(cursor.address(self.layout.encoding().offsets())?)?;
        if owner != self.header {
            return Err(FormatError::InvalidExtensibleArrayBlockOwner {
                expected: self.header.get(),
                actual: owner.get(),
            });
        }
        Ok(cursor)
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
        cursor: &mut bytes::ByteCursor<'_>,
        expected: u64,
        libhdf5: Option<u64>,
    ) -> Result<(), FormatError> {
        let actual = cursor.uint(usize::from(self.layout.block_offset_len()))?;
        if actual != expected && Some(actual) != libhdf5 {
            return Err(FormatError::InvalidExtensibleArrayBlockOffset { expected, actual });
        }
        Ok(())
    }
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

/// A retained plan for serializing an empty Extensible Array header and index block.
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
/// assert_eq!(initialization.plan().blocks().len(), 2);
/// let bytes = initialization.encode_at(StoredAddress::new(1024))?;
/// assert_eq!(bytes.len() as u64, initialization.allocation_len());
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub struct ExtensibleArrayInitialization {
    plan: ExtensibleArrayPlan,
}

impl ExtensibleArrayInitialization {
    /// Uses version 4 chunk-record encoding for an empty index.
    ///
    /// `chunk_bytes` is the chunk size before filtering. For filtered chunks, the chunk-size
    /// field width is derived from that size.
    ///
    /// # Errors
    ///
    /// Returns the geometry and width errors of [`ExtensibleArrayPlan::new`].
    pub fn new(
        creation: ExtensibleArrayCreationParameters,
        chunk_bytes: u64,
        widths: FormatWidths,
        has_filters: bool,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            plan: ExtensibleArrayPlan::new(
                creation,
                SlotOccupancy::Dense(0),
                0,
                chunk_bytes,
                widths,
                has_filters,
            )?,
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
        self.plan.geometry()
    }

    /// Returns the canonical chunk-record encoding used to serialize the empty index.
    pub fn element_encoding(&self) -> ChunkElementEncoding {
        self.plan.element_encoding()
    }

    /// Returns the combined size of the header and index block, including their checksums.
    pub fn allocation_len(&self) -> u64 {
        self.plan.total_len()
    }

    /// Serializes the header at the base-relative `address`, followed by its index block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the allocation's end exceeds
    /// [`u64::MAX`]. After this check succeeds, returns [`FormatError::DefinedAddressIsUndefined`]
    /// if `address` is undefined, or [`FormatError::AddressTooLarge`] if its start or end exceeds
    /// the address width. Returns
    /// [`FormatError::ValueTooLargeForPlatform`] if the allocation length does not fit [`usize`],
    /// [`FormatError::AllocationFailed`] if a scratch or output reservation fails, or
    /// [`FormatError::Internal`] if the encoding differs from the planned length.
    pub fn encode_at(&self, address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        address
            .get()
            .checked_add(self.allocation_len())
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        self.plan
            .encode_contiguous(&IndexSlots::dense(&[]), address)
    }

    /// Returns the retained plan for the empty header and index block.
    pub fn plan(&self) -> &ExtensibleArrayPlan {
        &self.plan
    }
}

/// The slots of an Extensible Array that hold a chunk.
///
/// [`extensible_array_layout`] counts the blocks that hold a chunk and [`build_extensible_array_at`]
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

/// Header statistics and complete encoded length from Extensible Array measurement.
///
/// The statistics and total length are computed through the same block traversal as
/// [`ExtensibleArrayPlan`].
pub struct ExtensibleArrayLayout {
    /// The header statistics.
    pub stats: ExtensibleArrayStats,
    /// The length of the array in bytes.
    total_len: u64,
}

/// Returns the layout of the Extensible Array of `num_slots` slots whose occupied slots are
/// `occupancy`.
///
/// The creation parameters are [`ExtensibleArrayCreationParameters::canonical`]. Measurement
/// retains numeric totals in the returned value.
///
/// # Errors
///
/// Returns [`FormatError::ExtensibleArraySlotBounds`] if an occupied slot is outside the
/// declared span, [`FormatError::ExtensibleArraySlotCount`] if a sparse index's span differs
/// from `num_slots`, or [`FormatError::ExtensibleArraySlotOrder`] if explicit slots are not
/// strictly increasing. Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the span,
/// block offsets, or complete lengths exceed the creation geometry or checked `u64` arithmetic,
/// or [`FormatError::LengthTooLarge`] if a header statistic does not fit `length_size`.
pub fn extensible_array_layout(
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> Result<ExtensibleArrayLayout, FormatError> {
    write::measure(
        occupancy,
        num_slots,
        chunk_bytes,
        FormatWidths {
            offsets: offset_size,
            lengths: length_size,
        },
        has_filters,
    )
}

/// Returns the length in bytes of the Extensible Array [`build_extensible_array_at`] builds for
/// `slots`, without building it.
///
/// A caller that reserves the span of an array before writing it, such as an in-place writer
/// that places an array in freed space, takes the length from here.
///
/// # Errors
///
/// Returns the geometry and width errors of [`extensible_array_layout`].
pub fn extensible_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> Result<u64, FormatError> {
    Ok(extensible_array_layout(
        SlotOccupancy::Sparse(slots),
        slots.len().to_u64(),
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    )?
    .total_len)
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
/// Returns the planning errors of [`ExtensibleArrayPlan::new`] and record errors of
/// [`ExtensibleArrayPlacement::encoding`](write::ExtensibleArrayPlacement::encoding).
/// Returns [`FormatError::DefinedAddressIsUndefined`] if `ea_address` is undefined,
/// [`FormatError::AddressTooLarge`] if the allocation's start or end exceeds the address width,
/// [`FormatError::OffsetOverflow`] if its end exceeds [`u64::MAX`], or
/// [`FormatError::ValueTooLargeForPlatform`] if the output length does not fit [`usize`]. Returns
/// [`FormatError::AllocationFailed`] if a scratch or output reservation fails, or
/// [`FormatError::Internal`] if the encoding differs from the planned length.
pub fn build_extensible_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    ea_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    ExtensibleArrayPlan::new(
        ExtensibleArrayCreationParameters::canonical(),
        SlotOccupancy::Sparse(slots),
        slots.len().to_u64(),
        chunk_bytes,
        FormatWidths {
            offsets: offset_size,
            lengths: length_size,
        },
        has_filters,
    )?
    .encode_contiguous(slots, ea_address)
}

/// Returns the capacity in slots of an Extensible Array [`build_extensible_array_at`] builds: the
/// elements of the index block, the data blocks it addresses, and the data blocks of every super
/// block.
///
/// The capacity is 8,589,934,580 slots: 4 in the index block, 240 in six data blocks, and
/// 8,589,934,336 in 25 super blocks.
pub fn extensible_array_capacity() -> u64 {
    // Each super block holds twice as many slots as the previous one. The sum starts at
    // `EA_MIN_DBLK_NELMTS` and includes a final block of `2^EA_MAX_NELMTS_BITS` slots.
    const {
        assert!((EA_MAX_NELMTS_BITS as u32) < u64::BITS - 1);
        (1u64 << (EA_MAX_NELMTS_BITS + 1)) - EA_MIN_DBLK_NELMTS as u64 + EA_IDX_BLK_ELMTS as u64
    }
}

/// The client ID of an array that indexes unfiltered chunks, whose elements each store a chunk
/// address alone.
///
/// The ID is defined in the Client ID table of "The Extensible Array Index" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub const EA_CLIENT_ID_UNFILTERED: u8 = 0;

// The signatures, the version, and the field widths are defined in "The Extensible Array Index"
// of the format specification, version 4.0.
const INDEX_BLOCK_SIGNATURE: [u8; 4] = *b"EAIB";
const SUPER_BLOCK_SIGNATURE: [u8; 4] = *b"EASB";
const DATA_BLOCK_SIGNATURE: [u8; 4] = *b"EADB";
const VERSION: u8 = 0;
// The version and client ID of a block.
const BLOCK_BYTE_FIELDS: usize = 2;
const CHECKSUM_LEN: usize = 4;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use rstest::rstest;

    use super::statistics::STATISTICS;
    use super::*;
    use crate::address::StoredAddressExt;
    use crate::width::FormatWidths;

    /// Returns the length of a header: 12 fixed bytes, six length-sized statistics, the index
    /// block address, and the checksum.
    const fn eahd_len(os: u8, ls: u8) -> usize {
        12 + 6 * ls as usize + os as usize + 4
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
            let computed = super::extensible_array_layout(
                SlotOccupancy::Dense(n),
                n,
                8,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                false,
            )
            .unwrap()
            .stats;
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
            )
            .unwrap();
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
                .unwrap();
                chunk_record::chunk_element_encoding(chunk_bytes, OffsetWidth::Eight, true)
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
    pub(super) fn records_in(
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
        .unwrap()
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
                .index_block_address()
                .is_undefined(WIDTHS.offsets)
        );
        assert_eq!(array.header().stored_statistics().max_idx_set, 5);
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
        let header_len = HEADER_PREFIX_LEN.to_u64()
            + u64::from(STATISTICS) * u64::from(lengths.get())
            + u64::from(offsets.get())
            + CHECKSUM_LEN.to_u64();
        let array = fixture.header().unwrap();
        let header = array.header();
        assert_eq!(header.client_id(), u8::from(filtered));
        assert_eq!(
            header.element_size(),
            offsets.get() + if filtered { 2 + FILTER_MASK_BYTES } else { 0 },
        );
        assert_eq!(
            header.creation_parameters().encode(),
            [32, 4, minimum_pointers, 16, 10]
        );
        assert_eq!(header.stored_statistics().max_idx_set, 20);
        assert_eq!(
            header.index_block_address(),
            StoredAddress::new(ARRAY_AT + header_len)
        );
        assert_eq!(array.range().start().get(), base + ARRAY_AT);
        assert_eq!(array.range().length(), header_len);
        assert_eq!(
            array.index_block().unwrap().parse_range().start().get(),
            base + ARRAY_AT + header_len,
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
    #[case::zero_min_data_blocks(&[(10, 0)], FormatError::InvalidExtensibleArrayGeometry)]
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
    #[rstest]
    #[case::maximum_end(false)]
    #[case::overflow(true)]
    fn builder_checks_the_complete_exclusive_end(#[case] overflow: bool) {
        let chunks = [ChunkRecord {
            address: StoredAddress::new(16),
            stored_size: 8,
            filter_mask: 0,
        }; 2];
        let slots = IndexSlots::dense(&chunks);
        let length =
            extensible_array_len(&slots, 8, OffsetWidth::Eight, LengthWidth::Eight, false).unwrap();
        let start = u64::MAX - length + u64::from(overflow);
        let result = build_extensible_array_at(
            &slots,
            8,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(start),
        );
        if overflow {
            assert_eq!(
                result.unwrap_err(),
                FormatError::OffsetOverflow {
                    offset: start,
                    length
                }
            );
        } else {
            let bytes = result.unwrap();
            assert_eq!(bytes.len() as u64, length);
            assert_eq!(&bytes[..HEADER_SIGNATURE.len()], &HEADER_SIGNATURE);
            assert_eq!(
                StoredAddress::new(start).checked_add(length).unwrap(),
                StoredAddress::new(u64::MAX)
            );
        }
    }
}
