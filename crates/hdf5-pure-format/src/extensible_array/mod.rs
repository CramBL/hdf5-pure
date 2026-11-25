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
use core::num::NonZeroU8;
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
use crate::chunk_record::ArrayClient;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::chunk_record::IndexSlots;
use crate::chunk_record::InvalidElementEncoding;
use crate::codec::Decode;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[cfg(test)]
mod measurement_sizes;

pub(crate) mod write;

/// The address of an Extensible Array header, from the chunk index address of a data layout
/// message.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse), which returns the header and any index-block reference.
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

    /// Parses the complete header from [`Self::parse_range`] within `region`.
    ///
    /// Uses the same framing and checksum policy as [`ExtensibleArrayHeader::parse_with_widths`].
    /// Resolves a defined index-block address against the base and derives the block's range
    /// from the checked creation parameters and element encoding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the header address or end exceeds [`u64::MAX`],
    /// or [`FormatError::InsufficientRegion`] if `region` omits any part of the header range.
    /// Within that range, reports the header field and checksum errors documented by
    /// [`ExtensibleArrayHeader::parse_with_widths`]. After decoding a header with a defined
    /// index-block address, returns [`FormatError::OffsetOverflow`] if the resolved index-block
    /// address or end exceeds [`u64::MAX`].
    pub fn parse(&self, region: FileRegion<'_>) -> Result<ExtensibleArray, FormatError> {
        let range = self.parse_range()?;
        let bytes = region.subregion(range)?.bytes();
        let header = ExtensibleArrayHeader::decode(bytes, self.widths)?;
        let index_block_address = header.index_block_address();
        let index_block = if index_block_address.is_undefined(header.encoding.offsets()) {
            None
        } else {
            let owner = Owner {
                base: self.base,
                header: range.start(),
                layout: header.layout(),
                max_index_set: header.stored_statistics().max_idx_set,
            };
            Some(ExtensibleArrayIndexBlockRef {
                owner,
                range: FileRange::new(
                    owner.resolve(index_block_address)?,
                    owner.layout.index_block_len()?,
                )?,
            })
        };
        Ok(ExtensibleArray {
            range,
            header,
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
    index_block: Option<ExtensibleArrayIndexBlockRef>,
}

impl ExtensibleArray {
    /// Returns the physical range of the header.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the stored header fields.
    ///
    /// The [`ExtensibleArrayHeader::index_block_address`] is base-relative or undefined.
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
        ExtensibleArrayGeometry::from_header(&self.header)
    }

    /// Returns whether the array indexes filtered chunks, the header's client ID 1.
    pub const fn is_filtered(&self) -> bool {
        matches!(
            self.header.encoding.client(),
            ArrayClient::FilteredChunks { .. }
        )
    }

    /// Returns the index-block reference, or [`None`] when the header stores the undefined address.
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
        let cursor = owner.parse_prefix(bytes, INDEX_BLOCK_SIGNATURE)?;
        checksum::verify_trailing(bytes)?;
        let layout = owner.layout;
        let (elements, addresses) = body(bytes, cursor.position())?
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
        let mut cursor = owner.parse_prefix(bytes, SUPER_BLOCK_SIGNATURE)?;
        owner.parse_block_offset(&mut cursor, super_block.offset, None)?;
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
        let mut cursor = owner.parse_prefix(bytes, DATA_BLOCK_SIGNATURE)?;
        // For a data block the index block holds the address of, `H5EA__lookup_elmt` (`H5EA.c`,
        // HDF5 1.10.11 to 2.2.0) stores the offset of its super block plus its ordinal among those
        // data blocks times its element count, which differs from its position from super block 1
        // on.
        owner.parse_block_offset(&mut cursor, block.offset, block.libhdf5_offset)?;
        checksum::verify_trailing(bytes)?;
        Ok(match self.pages()? {
            Some(pages) => ExtensibleArrayDataBlock::Pages(pages),
            None => {
                ExtensibleArrayDataBlock::Elements(ExtensibleArrayElements::below_max_index_set(
                    owner,
                    body(bytes, cursor.position())?,
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
        let page_elements = layout.page_nelmts.get();
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
        let elements = self.owner.layout.page_nelmts.get();
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
        let expected = self.layout.encoding.client().id();
        if client_id != expected {
            return Err(FormatError::InvalidExtensibleArrayBlockClient {
                expected,
                actual: client_id,
            });
        }
        let owner = self.resolve(cursor.address(self.layout.encoding.offsets())?)?;
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
        let actual = cursor.uint(usize::from(self.layout.block_offset_len))?;
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
    page_nelmts: PageElementCount,
    block_offset_len: u8,
    /// The index of the first super block whose address the index block holds.
    first_super_block: u32,
    /// The number of super blocks, those before `first_super_block` included.
    super_blocks: u32,
}

impl Layout {
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
        // Construction requires at least two data block pointers per super block, so
        // `first_super_block` is at least 2 and the shift is at least 1.
        2 * ((1 << (self.first_super_block / 2)) - 1)
    }

    fn super_block(self, index: u32) -> Result<SuperBlock, FormatError> {
        // Every caller passes an index below `super_blocks`, whose checked value is at most 65, so
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
        let pages = DataBlockGeometry {
            dblk_nelmts: elements,
            page_nelmts: self.page_nelmts,
        }
        .npages();
        (pages != 0).then_some(pages)
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
        // `ExtensibleArrayCreationParameters::from_bit_geometry` checks that `first_super_block`
        // is at most `super_blocks`.
        sum([
            self.prefix_len(),
            self.elements_len(u64::from(self.inline_elements))?,
            self.addresses_len(self.direct_data_blocks())?,
            self.addresses_len(u64::from(self.super_blocks - self.first_super_block))?,
            CHECKSUM_LEN.to_u64(),
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
            CHECKSUM_LEN.to_u64(),
        ])
    }

    fn data_block_len(self, block: DataBlock) -> Result<u64, FormatError> {
        match self.pages(block.elements) {
            Some(pages) => sum([self.pages_offset(), product(pages, self.page_len()?)?]),
            None => sum([
                self.prefix_len(),
                u64::from(self.block_offset_len),
                self.elements_len(block.elements)?,
                CHECKSUM_LEN.to_u64(),
            ]),
        }
    }

    fn pages_offset(self) -> u64 {
        self.pages_offset_usize().to_u64()
    }

    fn page_len(self) -> Result<u64, FormatError> {
        sum([
            self.elements_len(self.page_nelmts.get())?,
            CHECKSUM_LEN.to_u64(),
        ])
    }

    fn prefix_len(self) -> u64 {
        self.prefix_len_usize().to_u64()
    }

    fn pages_offset_usize(self) -> usize {
        self.prefix_len_usize() + usize::from(self.block_offset_len) + CHECKSUM_LEN
    }

    fn prefix_len_usize(self) -> usize {
        INDEX_BLOCK_SIGNATURE.len() + BLOCK_BYTE_FIELDS + usize::from(self.encoding.offsets().get())
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

fn header_len(widths: FormatWidths) -> u64 {
    ExtensibleArrayHeader::serialized_size_with_widths(widths).to_u64()
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
        let fields = bytes::ByteCursor::new(data).array()?;
        Self::from_fields(fields).ok_or(FormatError::InvalidExtensibleArrayGeometry)
    }

    /// Returns the writer's default creation parameters.
    ///
    /// The maximum element bit count is 32, the index block holds four elements, the minimum
    /// data block pointer count is four, the minimum data block size is 16 elements, and a page
    /// holds 1024 elements.
    pub fn canonical() -> Self {
        const {
            // Constant evaluation checks these creation fields before the writer uses them.
            Self::from_fields([
                EA_MAX_NELMTS_BITS,
                EA_IDX_BLK_ELMTS,
                EA_SUPER_BLK_MIN_DATA_PTRS,
                EA_MIN_DBLK_NELMTS,
                EA_MAX_DBLK_NELMTS_BITS,
            ])
            .expect("the writer's creation fields have valid geometry")
        }
    }

    /// Returns the creation fields in data layout message order.
    pub fn encode(self) -> [u8; 5] {
        [
            self.bits.max_index_bits().get(),
            self.inline_elements,
            self.min_data_blocks.get(),
            self.min_elements.get(),
            self.bits.page_nelmts().exponent(),
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

    /// Constructs creation parameters from the five fields in data layout message order.
    ///
    /// Returns `None` if the fields describe an unsupported block or page geometry.
    const fn from_fields(fields: [u8; 5]) -> Option<Self> {
        let [
            max_index_bits,
            inline_elements,
            min_data_blocks,
            min_elements,
            page_bits,
        ] = fields;
        let Ok(bits) = ExtensibleArrayBitGeometry::from_fields(max_index_bits, page_bits) else {
            return None;
        };
        Self::from_bit_geometry(bits, inline_elements, min_data_blocks, min_elements)
    }

    /// Combines checked bit geometry with the counts for index and data blocks.
    ///
    /// Returns `None` if the minimum block element or pointer count is unsupported, or if the
    /// block geometry exceeds the array or page bounds.
    const fn from_bit_geometry(
        bits: ExtensibleArrayBitGeometry,
        inline_elements: u8,
        min_data_blocks: u8,
        min_elements: u8,
    ) -> Option<Self> {
        let (Some(min_data_blocks), Some(min_elements)) = (
            NonZeroU8::new(min_data_blocks),
            NonZeroU8::new(min_elements),
        ) else {
            return None;
        };
        if !min_elements.get().is_power_of_two()
            || !min_data_blocks.get().is_power_of_two()
            || min_data_blocks.get() < MIN_DATA_BLOCKS
        {
            return None;
        }
        let min_elements_bits = min_elements.get().trailing_zeros();
        let first_super_block = 2 * min_data_blocks.get().trailing_zeros();
        let Some(super_blocks) =
            (bits.max_index_bits().get() as u32 + 1).checked_sub(min_elements_bits)
        else {
            return None;
        };
        if first_super_block > super_blocks
            || min_elements_bits + first_super_block / 2 > bits.page_nelmts().exponent() as u32
        {
            return None;
        }
        Some(Self {
            bits,
            inline_elements,
            min_data_blocks,
            min_elements,
        })
    }

    // `H5EA_SBLK_FIRST_IDX` and `H5EA_SIZEOF_OFFSET_BITS` (`H5EApkg.h`, HDF5 2.2.0)
    // determine the first super block and block-offset width.
    fn layout(self, encoding: ElementEncoding) -> Layout {
        Layout {
            encoding,
            inline_elements: self.inline_elements,
            min_elements_bits: self.min_elements.get().trailing_zeros(),
            page_nelmts: self.bits.page_nelmts(),
            block_offset_len: self.bits.max_index_bits().get().div_ceil(8),
            first_super_block: 2 * self.min_data_blocks.get().trailing_zeros(),
            super_blocks: u32::from(self.bits.max_index_bits().get()) + 1
                - self.min_elements.get().trailing_zeros(),
        }
    }
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

/// An Extensible Array header, signature `EAHD`, version 0.
///
/// The maximum element bit count is in `1..=64`, and the page exponent is at most that count
/// and at most 63. The page element count fits a [`u64`], and the block offset is 1 to 8 bytes
/// wide.
///
/// "The Extensible Array Index" of the [format specification, version 4.0][spec]
/// defines the header.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayHeader {
    encoding: ElementEncoding,
    element_size: u8,
    creation: ExtensibleArrayCreationParameters,
    stored_statistics: ExtensibleArrayStats,
    index_block_address: StoredAddress,
}

impl ExtensibleArrayHeader {
    /// Parses the Extensible Array header at `offset` in `file_data`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] for an unsupported `offset_size` or
    /// [`FormatError::InvalidLengthSize`] for an unsupported `length_size`. The supported
    /// widths are 2, 4, and 8 bytes. Checks the offset width before the length width and
    /// checks both before reading the header. See [`Self::parse_with_widths`] for errors
    /// from the header bytes.
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

    /// Parses the complete header at `offset` and ignores bytes after its checksum.
    ///
    /// Reads the signature and version before checking the checksum. With the `checksum`
    /// feature, verifies the four-byte Jenkins checksum over all preceding header bytes.
    /// Then checks the client ID, the element size for filtered chunks, and the creation parameters
    /// in that order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the complete header is unavailable. This
    /// error reports the required end relative to the start of `file_data` and the input length.
    /// The parser caps the required end at [`usize::MAX`].
    ///
    /// Returns [`FormatError::InvalidExtensibleArraySignature`] for a signature other than
    /// `EAHD`, [`FormatError::InvalidExtensibleArrayVersion`] for a version other than 0,
    /// or [`FormatError::ChecksumMismatch`] if verification is enabled and the checksum differs.
    /// Returns [`FormatError::InvalidExtensibleArrayClient`] for a client ID other than 0 or 1,
    /// or [`FormatError::InvalidExtensibleArrayGeometry`] for an unsupported filtered element
    /// size or creation parameter.
    pub fn parse_with_widths(
        file_data: &[u8],
        offset: usize,
        widths: FormatWidths,
    ) -> Result<Self, FormatError> {
        let input = file_data
            .get(offset..)
            .ok_or_else(|| FormatError::UnexpectedEof {
                expected: offset.saturating_add(Self::serialized_size_with_widths(widths)),
                available: file_data.len(),
            })?;
        Self::decode(input, widths).map_err(|error| match error {
            FormatError::UnexpectedEof { expected, .. } => FormatError::UnexpectedEof {
                expected: offset.saturating_add(expected),
                available: file_data.len(),
            },
            error => error,
        })
    }

    /// Returns the length in bytes of a header with `offset_size`-byte addresses and
    /// `length_size`-byte lengths.
    pub fn serialized_size(offset_size: u8, length_size: u8) -> usize {
        4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * length_size as usize + offset_size as usize + 4
    }

    /// Returns the header length for validated format widths.
    pub fn serialized_size_with_widths(widths: FormatWidths) -> usize {
        HEADER_PREFIX_LEN
            + ExtensibleArrayStats::fields_len(widths.lengths)
            + usize::from(widths.offsets.get())
            + CHECKSUM_LEN
    }

    /// Returns 0 for unfiltered chunks or 1 for filtered chunks.
    pub const fn client_id(&self) -> u8 {
        self.encoding.client().id()
    }

    /// Returns the stored element size in bytes.
    ///
    /// For unfiltered chunks, the reader uses the file's offset width to read the entries.
    pub const fn element_size(&self) -> u8 {
        self.element_size
    }

    /// Returns the checked creation parameters from the header.
    pub const fn creation_parameters(&self) -> ExtensibleArrayCreationParameters {
        self.creation
    }

    /// Returns the parsed maximum element bit count and data block page exponent.
    pub fn bit_geometry(&self) -> ExtensibleArrayBitGeometry {
        self.creation.bit_geometry()
    }

    /// Returns the "Max Nelmts Bits" field, in `1..=64`.
    pub fn max_index_bits(&self) -> MaxIndexBits {
        self.bit_geometry().max_index_bits()
    }

    /// Returns the number of elements in a data block page, a nonzero power of two.
    pub const fn page_nelmts(&self) -> PageElementCount {
        self.creation.bits.page_nelmts()
    }

    /// Returns the width in bytes of a block offset, in `1..=8`.
    pub fn block_offset_width(&self) -> BlockOffsetWidth {
        self.bit_geometry().block_offset_width()
    }

    /// Returns the six statistics as stored in the header.
    ///
    /// The parser reads these values from the header bytes alone. Use
    /// [`ExtensibleArrayStats::encode_fields`] to reproduce the fields at the file's length width.
    ///
    /// # Examples
    ///
    /// Reads and re-encodes the statistics of a header with two-byte lengths and four-byte addresses.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{ExtensibleArrayHeader, FormatWidths};
    ///
    /// let bytes = [
    ///     b'E', b'A', b'H', b'D', 0, 0, 4, 32, 4, 16, 4, 10,
    ///     1, 0, 0x80, 1, 3, 0, 0, 9, 5, 0, 6, 0,
    ///     0, 0x10, 0, 0, 0x73, 0x63, 0xec, 0x2b,
    /// ];
    /// let widths = FormatWidths::from_sizes(4, 2)?;
    /// let header = ExtensibleArrayHeader::parse_with_widths(&bytes, 0, widths)?;
    /// let stats = header.stored_statistics();
    /// assert_eq!(stats.max_idx_set, 5);
    /// assert_eq!(stats.nelmts, 6);
    ///
    /// let mut fields = [0; 12];
    /// assert_eq!(stats.encode_fields(&mut fields, widths.lengths)?, fields.len());
    /// assert_eq!(fields, bytes[12..24]);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub const fn stored_statistics(&self) -> &ExtensibleArrayStats {
        &self.stored_statistics
    }

    /// Returns the stored index-block address, including the file-width undefined address.
    pub const fn index_block_address(&self) -> StoredAddress {
        self.index_block_address
    }

    fn layout(&self) -> Layout {
        self.creation.layout(self.encoding)
    }
}

impl<'input> Decode<'input> for ExtensibleArrayHeader {
    type Context = FormatWidths;
    type Error = FormatError;

    fn decode(input: &'input [u8], widths: FormatWidths) -> Result<Self, Self::Error> {
        let header =
            bytes::ByteCursor::new(input).take(Self::serialized_size_with_widths(widths))?;
        let mut cursor = bytes::ByteCursor::new(header);
        if *cursor.take_array()? != HEADER_SIGNATURE {
            return Err(FormatError::InvalidExtensibleArraySignature);
        }
        let [
            version,
            client_id,
            element_size,
            max_index_bits,
            inline_elements,
            min_elements,
            min_data_blocks,
            page_bits,
        ] = cursor.array::<HEADER_BYTE_FIELDS>()?;
        if version != VERSION {
            return Err(FormatError::InvalidExtensibleArrayVersion(version));
        }
        checksum::verify_trailing(header)?;

        let encoding =
            ElementEncoding::new(widths.offsets, client_id, element_size).map_err(|error| {
                match error {
                    InvalidElementEncoding::Client(id) => {
                        FormatError::InvalidExtensibleArrayClient(id)
                    }
                    InvalidElementEncoding::EntrySize => {
                        FormatError::InvalidExtensibleArrayGeometry
                    }
                }
            })?;
        let creation = ExtensibleArrayCreationParameters::from_fields([
            max_index_bits,
            inline_elements,
            min_data_blocks,
            min_elements,
            page_bits,
        ])
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        let stored_statistics = ExtensibleArrayStats::decode_fields(&mut cursor, widths.lengths)?;
        let index_block_address = cursor.address(widths.offsets)?;
        Ok(Self {
            encoding,
            element_size,
            creation,
            stored_statistics,
            index_block_address,
        })
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
    page_nelmts: PageElementCount,
}

impl ExtensibleArrayBitGeometry {
    /// Returns the "Max Nelmts Bits" field, in `1..=64`.
    pub const fn max_index_bits(self) -> MaxIndexBits {
        self.max_index_bits
    }

    /// Returns the number of elements in a data block page, a nonzero power of two.
    pub const fn page_nelmts(self) -> PageElementCount {
        self.page_nelmts
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

    /// Constructs checked bit geometry from the maximum element bit count and page exponent.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidBitGeometry::MaximumIndexBits`] if the maximum bit count is outside
    /// `1..=64`, [`InvalidBitGeometry::PageElementCount`] if the page element count does not fit in
    /// [`u64`], or [`InvalidBitGeometry::PageExponentExceedsIndexBits`] if the page exponent
    /// exceeds the maximum bit count.
    const fn from_fields(max_index_bits: u8, page_bits: u8) -> Result<Self, InvalidBitGeometry> {
        let max_index_bits = match NonZeroU8::new(max_index_bits) {
            Some(bits) if bits.get() <= MAX_INDEX_BITS => MaxIndexBits(bits),
            _ => return Err(InvalidBitGeometry::MaximumIndexBits),
        };
        if page_bits > MAX_PAGE_BITS {
            return Err(InvalidBitGeometry::PageElementCount);
        }
        if page_bits > max_index_bits.get() {
            return Err(InvalidBitGeometry::PageExponentExceedsIndexBits);
        }
        Ok(Self {
            max_index_bits,
            page_nelmts: PageElementCount(page_bits),
        })
    }
}

#[derive(Debug, Eq, PartialEq)]
enum InvalidBitGeometry {
    MaximumIndexBits,
    PageElementCount,
    PageExponentExceedsIndexBits,
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
    /// Returns the bit count as a [`u8`], in `1..=64`.
    pub const fn get(self) -> u8 {
        self.0.get()
    }
}

/// A data block page's element count, a nonzero power of two.
///
/// Stores the base 2 exponent in `0..=63`. The version 0 Extensible Array header's
/// `Max Dblk Page Nelmts Bits` field is defined in "The Extensible Array Index" of the
/// [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
///
/// let geometry = ExtensibleArrayCreationParameters::canonical().bit_geometry();
/// let count = geometry.page_nelmts();
/// assert_eq!(count.get(), 1024);
/// assert_eq!(count.exponent(), 10);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct PageElementCount(u8);

impl PageElementCount {
    /// Returns the number of elements in a page, a nonzero power of two.
    pub const fn get(self) -> u64 {
        1 << self.0
    }

    /// Returns the base 2 exponent stored in the page size field.
    pub const fn exponent(self) -> u8 {
        self.0
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
    #[cfg(test)]
    pub(crate) fn super_block_at(
        &self,
        j: usize,
        page_nelmts: PageElementCount,
    ) -> SuperBlockGeometry {
        let (ndblks, dblk_nelmts) = self.sblks[self.first_indirect_sblk + j];
        SuperBlockGeometry {
            ndblks,
            blocks: DataBlockGeometry {
                dblk_nelmts,
                page_nelmts,
            },
        }
    }

    /// Derives block geometry from the header's checked creation parameters.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset
    /// exceeds [`u64::MAX`].
    pub fn from_header(header: &ExtensibleArrayHeader) -> Result<Self, FormatError> {
        Self::from_layout(header.layout())
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
    /// The checked element count per page.
    pub page_nelmts: PageElementCount,
}

impl DataBlockGeometry {
    /// Returns whether the data block is stored in pages.
    pub const fn is_paged(self) -> bool {
        self.dblk_nelmts > self.page_nelmts.get()
    }

    /// Returns the number of pages in the data block, or zero for an unpaged block.
    pub const fn npages(self) -> u64 {
        let page_nelmts = self.page_nelmts.get();
        if self.dblk_nelmts > page_nelmts {
            self.dblk_nelmts / page_nelmts
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

/// Appends `addr` to `buf` as a little-endian field of `offset_size` bytes.
///
/// Encodes [`u64::MAX`] as the undefined address.
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if `addr` exceeds `offset_size` and is not
/// [`u64::MAX`]. Leaves `buf` unchanged on error.
pub fn write_stored_address(
    buf: &mut Vec<u8>,
    addr: StoredAddress,
    offset_size: OffsetWidth,
) -> Result<(), FormatError> {
    bytes::try_write_offset(buf, addr.get(), offset_size)
}

/// Appends a block's offset in the array's element space as a little-endian field of `width`
/// bytes.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForRepresentation`] if `block_offset` exceeds `width`.
fn write_block_offset(
    buf: &mut Vec<u8>,
    block_offset: u64,
    width: BlockOffsetWidth,
) -> Result<(), FormatError> {
    let blk_off_size = width.bytes();
    let bytes = block_offset.to_le_bytes();
    if bytes[blk_off_size..].iter().any(|&byte| byte != 0) {
        return Err(FormatError::ValueTooLargeForRepresentation {
            value: block_offset,
            target: "Extensible Array block offset",
        });
    }
    buf.extend_from_slice(&bytes[..blk_off_size]);
    Ok(())
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
/// Returns [`FormatError::AddressTooLarge`] if an occupied chunk's address exceeds `offset_size`,
/// or `ea_address` exceeds `offset_size` and is not [`u64::MAX`]. Returns
/// [`FormatError::DefinedAddressIsUndefined`] if an occupied chunk's address equals that width's
/// undefined marker, or [`FormatError::ValueTooLargeForRepresentation`] if `block_offset_rel`
/// exceeds the block-offset width in `bits`.
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
    let page_element_count = bits.page_nelmts();
    let page_nelmts = page_element_count.get().to_usize()?;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EADB");
    buf.push(0); // version
    buf.push(client_id);
    write_stored_address(&mut buf, ea_address, offset_size)?;
    write_block_offset(&mut buf, block_offset_rel, bits.block_offset_width())?;

    // The counts are `usize` loop bounds, and the page arithmetic below stays in `usize`.
    let blocks = DataBlockGeometry {
        dblk_nelmts: dblk_nelmts as u64,
        page_nelmts: page_element_count,
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
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if an address exceeds `offset_size` and is not
/// [`u64::MAX`], or [`FormatError::ValueTooLargeForRepresentation`] if `block_offset_rel` exceeds
/// `width`.
pub fn encode_super_block(
    ea_address: StoredAddress,
    block_offset_rel: u64,
    page_bitmap: &[u8],
    dblk_addrs: &[StoredAddress],
    offset_size: OffsetWidth,
    client_id: u8,
    width: BlockOffsetWidth,
) -> Result<Vec<u8>, FormatError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EASB");
    buf.push(0); // version
    buf.push(client_id);
    write_stored_address(&mut buf, ea_address, offset_size)?;
    write_block_offset(&mut buf, block_offset_rel, width)?;
    buf.extend_from_slice(page_bitmap);
    for &addr in dblk_addrs {
        write_stored_address(&mut buf, addr, offset_size)?;
    }
    let cks = checksum::jenkins_lookup3(&buf);
    buf.extend_from_slice(&cks.to_le_bytes());
    Ok(buf)
}

/// The six statistics of an Extensible Array header, in the order the header stores them.
///
/// [`ExtensibleArrayHeader::stored_statistics`] reports the values read from a header.
/// [`extensible_array_layout`] derives statistics from the occupied slots and creation geometry.
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

impl ExtensibleArrayStats {
    /// Returns the statistics' byte range relative to the Extensible Array header's start.
    ///
    /// The fields precede the index-block address and checksum in a version 0 header, as defined
    /// in "The Extensible Array Index" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
    pub fn fields_range(lengths: LengthWidth) -> Range<u64> {
        let start = HEADER_PREFIX_LEN.to_u64();
        start..start + u64::from(STATISTICS) * u64::from(lengths.get())
    }

    /// Encodes the statistics into the start of `output` and returns the byte count.
    ///
    /// Writes each field in header order at `lengths`, preserving bytes after the fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if a statistic exceeds `lengths`, or
    /// [`FormatError::UnexpectedEof`] if `output` is too short. Preserves `output` on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private as format, FormatError, LengthWidth};
    ///
    /// let stats = format::ExtensibleArrayStats {
    ///     nsuper_blks: 1,
    ///     super_blk_size: 2,
    ///     ndata_blks: 3,
    ///     data_blk_size: 4,
    ///     max_idx_set: 5,
    ///     nelmts: 6,
    /// };
    /// let mut fields = [0xcc; format::ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN];
    /// let used = stats.encode_fields(&mut fields, LengthWidth::Two)?;
    /// assert_eq!(used, 12);
    /// assert_eq!(&fields[..used], &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    /// assert_eq!(format::ExtensibleArrayStats::fields_range(LengthWidth::Two), 12..24);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn encode_fields(
        &self,
        output: &mut [u8],
        lengths: LengthWidth,
    ) -> Result<usize, FormatError> {
        self.require_field_fits(lengths)?;
        let size = Self::fields_len(lengths);
        let available = output.len();
        let fields = output.get_mut(..size).ok_or(FormatError::UnexpectedEof {
            expected: size,
            available,
        })?;
        self.write_fields(fields, lengths);
        Ok(size)
    }

    /// Reads the six consecutive length fields in header order.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if the remaining input is shorter than six length fields.
    fn decode_fields(
        cursor: &mut bytes::ByteCursor<'_>,
        lengths: LengthWidth,
    ) -> Result<Self, bytes::UnexpectedEof> {
        let [
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        ] = match lengths {
            LengthWidth::Two => {
                Self::decode_values(cursor, |field| u64::from(u16::from_le_bytes(field)))?
            }
            LengthWidth::Four => {
                Self::decode_values(cursor, |field| u64::from(u32::from_le_bytes(field)))?
            }
            LengthWidth::Eight => Self::decode_values(cursor, u64::from_le_bytes)?,
        };
        Ok(Self {
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        })
    }

    fn decode_values<const N: usize>(
        cursor: &mut bytes::ByteCursor<'_>,
        decode: impl Fn([u8; N]) -> u64,
    ) -> Result<[u64; STATISTICS as usize], bytes::UnexpectedEof> {
        let fields = cursor.take_arrays::<N>(usize::from(STATISTICS))?;
        let mut values = [0; STATISTICS as usize];
        for (value, field) in values.iter_mut().zip(fields) {
            *value = decode(*field);
        }
        Ok(values)
    }

    fn require_field_fits(&self, lengths: LengthWidth) -> Result<(), FormatError> {
        for value in self.field_values() {
            bytes::require_length(value, lengths)?;
        }
        Ok(())
    }

    /// Writes the statistics into a slice of exactly [`Self::fields_len`] bytes.
    ///
    /// The caller checks that each statistic fits `lengths`.
    fn write_fields(&self, output: &mut [u8], lengths: LengthWidth) {
        let width = usize::from(lengths.get());
        for (field, value) in output.chunks_exact_mut(width).zip(self.field_values()) {
            field.copy_from_slice(&value.to_le_bytes()[..width]);
        }
    }

    fn fields_len(lengths: LengthWidth) -> usize {
        usize::from(STATISTICS) * usize::from(lengths.get())
    }

    fn field_values(&self) -> [u64; STATISTICS as usize] {
        let Self {
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        } = *self;
        [
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        ]
    }

    /// The maximum encoded statistics length across the supported length widths.
    pub const MAX_ENCODED_FIELDS_LEN: usize =
        STATISTICS as usize * LengthWidth::Eight.get() as usize;
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
// The version, client ID, element size, and the five creation parameters of a header.
const HEADER_BYTE_FIELDS: usize = 8;
const HEADER_PREFIX_LEN: usize = HEADER_SIGNATURE.len() + HEADER_BYTE_FIELDS;
// The six statistics of a header, each a length, of which "Max Index Set" is the fifth.
const STATISTICS: u8 = 6;
// The version and client ID of a block.
const BLOCK_BYTE_FIELDS: usize = 2;
const CHECKSUM_LEN: usize = 4;
// The bounds `H5EA__hdr_create` checks the creation parameters against (`H5EAhdr.c`, HDF5 2.2.0).
const MAX_INDEX_BITS: u8 = 64;
const MIN_DATA_BLOCKS: u8 = 2;
// The largest page exponent whose page element count fits a `u64`.
const MAX_PAGE_BITS: u8 = 63;

const _: () = {
    assert!(MAX_INDEX_BITS as u32 == u64::BITS);
    assert!(MAX_PAGE_BITS as u32 + 1 == u64::BITS);
};

#[cfg(test)]
mod tests {
    use alloc::vec;

    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddressExt;
    use crate::width::FormatWidths;

    #[rstest::rstest]
    #[case::two(LengthWidth::Two, u64::from(u16::MAX))]
    #[case::four(LengthWidth::Four, u64::from(u32::MAX))]
    #[case::eight(LengthWidth::Eight, u64::MAX)]
    fn statistics_encode_exact_width_limits(#[case] lengths: LengthWidth, #[case] maximum: u64) {
        let stats = ExtensibleArrayStats {
            nsuper_blks: maximum,
            super_blk_size: maximum,
            ndata_blks: maximum,
            data_blk_size: maximum,
            max_idx_set: maximum,
            nelmts: maximum,
        };
        let mut output = [0xcc; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN];
        let used = stats.encode_fields(&mut output, lengths).unwrap();
        assert_eq!(used, usize::from(lengths.get()) * STATISTICS as usize);
        assert_eq!(&output[..used], vec![0xff; used]);
        assert_eq!(&output[used..], vec![0xcc; output.len() - used]);
    }

    #[rstest::rstest]
    #[case::secondary_count(0)]
    #[case::secondary_bytes(1)]
    #[case::data_count(2)]
    #[case::data_bytes(3)]
    #[case::maximum_index(4)]
    #[case::elements(5)]
    fn statistics_reject_overflow_before_modifying_any_field(
        #[case] field: usize,
        #[values(LengthWidth::Two, LengthWidth::Four)] lengths: LengthWidth,
    ) {
        let value = match lengths {
            LengthWidth::Two => 65_536,
            LengthWidth::Four => 4_294_967_296,
            LengthWidth::Eight => panic!("eight-byte lengths have no larger u64 value"),
        };
        let mut fields = [0; STATISTICS as usize];
        fields[field] = value;
        let [
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        ] = fields;
        let stats = ExtensibleArrayStats {
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        };
        let mut output = [0xcc; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN];
        assert_eq!(
            stats.encode_fields(&mut output, lengths).unwrap_err(),
            FormatError::LengthTooLarge {
                length: value,
                length_size: lengths.get()
            }
        );
        assert_eq!(output, [0xcc; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN]);
    }

    #[rstest]
    #[case::two(LengthWidth::Two)]
    #[case::four(LengthWidth::Four)]
    #[case::eight(LengthWidth::Eight)]
    fn statistics_reject_a_short_output_before_modifying_any_field(#[case] lengths: LengthWidth) {
        let stats = ExtensibleArrayStats {
            nsuper_blks: 1,
            super_blk_size: 2,
            ndata_blks: 3,
            data_blk_size: 4,
            max_idx_set: 5,
            nelmts: 6,
        };
        let size = usize::from(lengths.get()) * STATISTICS as usize;
        let mut output = vec![0xcc; size - 1];
        assert_eq!(
            stats.encode_fields(&mut output, lengths).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: size,
                available: size - 1
            }
        );
        assert_eq!(output, vec![0xcc; size - 1]);
    }

    // The oracle is `H5EA__dblock_alloc`, which pages a block of more elements than a page.
    #[test]
    fn a_data_block_is_paged_only_past_a_full_page() {
        let page_nelmts = ExtensibleArrayBitGeometry::from_fields(32, 4)
            .unwrap()
            .page_nelmts();
        let at = |dblk_nelmts| DataBlockGeometry {
            dblk_nelmts,
            page_nelmts,
        };
        assert!(!at(15).is_paged(), "under a page");
        assert!(!at(16).is_paged(), "exactly a page is not paged");
        assert!(at(17).is_paged(), "past a page");
        assert_eq!(at(16).npages(), 0, "a block that is not paged has no pages");
        assert_eq!(at(32).npages(), 2);
    }

    #[test]
    fn the_page_init_bitmap_is_a_byte_per_eight_pages_per_block() {
        let page_nelmts = ExtensibleArrayBitGeometry::from_fields(32, 4)
            .unwrap()
            .page_nelmts();
        let sb = |ndblks, dblk_nelmts| SuperBlockGeometry {
            ndblks,
            blocks: DataBlockGeometry {
                dblk_nelmts,
                page_nelmts,
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
    fn bit_geometry_stores_two_bytes_with_byte_alignment() {
        assert_eq!(core::mem::size_of::<MaxIndexBits>(), 1);
        assert_eq!(core::mem::align_of::<MaxIndexBits>(), 1);
        assert_eq!(core::mem::size_of::<PageElementCount>(), 1);
        assert_eq!(core::mem::align_of::<PageElementCount>(), 1);
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
    #[case::zero_maximum(0, 10, InvalidBitGeometry::MaximumIndexBits)]
    #[case::maximum_65(65, 10, InvalidBitGeometry::MaximumIndexBits)]
    #[case::maximum_255(255, 10, InvalidBitGeometry::MaximumIndexBits)]
    #[case::invalid_maximum_precedes_page_overflow(0, 64, InvalidBitGeometry::MaximumIndexBits)]
    #[case::page_64(64, 64, InvalidBitGeometry::PageElementCount)]
    #[case::page_overflow_precedes_exponent_above_maximum(
        10,
        64,
        InvalidBitGeometry::PageElementCount
    )]
    #[case::page_255(64, 255, InvalidBitGeometry::PageElementCount)]
    #[case::page_exceeds_maximum(10, 11, InvalidBitGeometry::PageExponentExceedsIndexBits)]
    fn malformed_bit_geometry_is_rejected(
        #[case] max_bits: u8,
        #[case] page_bits: u8,
        #[case] expected: InvalidBitGeometry,
    ) {
        assert_eq!(
            ExtensibleArrayBitGeometry::from_fields(max_bits, page_bits).unwrap_err(),
            expected,
        );
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
        let geometry = ExtensibleArrayBitGeometry::from_fields(max_bits, page_bits).unwrap();
        assert_eq!(geometry.max_index_bits().get(), max_bits);
        let page_count = geometry.page_nelmts();
        assert_eq!(page_count.exponent(), page_bits);
        assert_eq!(page_count.get(), page_nelmts);
        assert_eq!(geometry.block_offset_width().bytes(), offset_size);

        let offset = u64::MAX >> (8 * (8 - offset_size));
        let block = encode_super_block(
            StoredAddress::new(0),
            offset,
            &[],
            &[],
            OffsetWidth::Eight,
            EA_CLIENT_ID_UNFILTERED,
            geometry.block_offset_width(),
        )
        .unwrap();
        assert_eq!(
            &block[14..14 + offset_size],
            &offset.to_le_bytes()[..offset_size]
        );
        assert_eq!(
            geometry.block_offset_width().read(&block[14..]).unwrap(),
            offset
        );
        checksum::verify_trailing(&block).unwrap();
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
        let creation = ExtensibleArrayCreationParameters::parse(&[
            EA_MAX_NELMTS_BITS,
            EA_IDX_BLK_ELMTS,
            minimum_pointers,
            EA_MIN_DBLK_NELMTS,
            EA_MAX_DBLK_NELMTS_BITS,
        ])
        .unwrap();
        let encoding = ElementEncoding::new(offsets, client_id, element_size).unwrap();
        let layout = creation.layout(encoding);
        let geometry = ExtensibleArrayGeometry::from_layout(layout).unwrap();
        assert_eq!(
            layout.index_block_len().unwrap(),
            u64::try_from(measurement_sizes::index_block_len(
                offsets,
                usize::from(EA_IDX_BLK_ELMTS),
                usize::from(element_size),
                geometry.direct_dblk_nelmts().len(),
                geometry.nsblk_addrs(),
            ))
            .unwrap()
        );
        for ordinal in 0..geometry.nsblk_addrs() {
            let writer = geometry.super_block_at(ordinal, layout.page_nelmts);
            let reader = layout
                .super_block(u32::try_from(geometry.first_indirect_sblk() + ordinal).unwrap())
                .unwrap();
            assert_eq!(layout.bitmap_len(reader).unwrap(), writer.bitmap_size());
            assert_eq!(
                layout.super_block_len(reader).unwrap(),
                measurement_sizes::super_block_len(
                    writer,
                    offsets,
                    creation.bit_geometry().block_offset_width()
                )
            );
            assert_eq!(
                layout
                    .data_block_len(reader.data_block(0).unwrap())
                    .unwrap(),
                measurement_sizes::data_block_len(
                    writer.blocks,
                    usize::from(element_size),
                    offsets,
                    creation.bit_geometry().block_offset_width()
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
        let layout = layout_from_fields(
            ElementEncoding::new(OffsetWidth::Eight, 0, 8).unwrap(),
            [
                EA_MAX_NELMTS_BITS,
                EA_IDX_BLK_ELMTS,
                min_data_pointers,
                EA_MIN_DBLK_NELMTS,
                EA_MAX_DBLK_NELMTS_BITS,
            ],
        );
        let geometry = ExtensibleArrayGeometry::from_layout(layout).unwrap();
        assert_eq!(geometry.direct_dblk_nelmts().len(), direct_blocks);
        assert_eq!(geometry.first_indirect_sblk(), first_secondary);
        assert_eq!(
            measurement_sizes::index_block_len(
                OffsetWidth::Eight,
                usize::from(EA_IDX_BLK_ELMTS),
                8,
                geometry.direct_dblk_nelmts().len(),
                geometry.nsblk_addrs()
            ),
            index_bytes
        );
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
        let layout = layout_from_fields(encoding, [32, 4, min_data_blocks, 16, 10]);
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
        let layout = layout_from_fields(encoding, [max_index_bits, 4, 2, 4, 10]);
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
            layout_from_fields(encoding, [bits, 0, 2, 1, 1]).block_offset_len,
            len
        );
    }

    fn layout_from_fields(encoding: ElementEncoding, fields: [u8; 5]) -> Layout {
        ExtensibleArrayCreationParameters::parse(&fields)
            .unwrap()
            .layout(encoding)
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
