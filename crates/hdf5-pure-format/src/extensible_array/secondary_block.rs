use core::ops::Range;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use super::block::{self, ExtensibleArrayBlockFraming, Owner};
use super::data_block::ExtensibleArrayDataBlockRef;
use super::geometry::SuperBlock;
use super::page::PageBitmap;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;

/// The position and the layout of an Extensible Array super block, from
/// [`super::ExtensibleArrayIndexBlock::super_blocks`].
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
    pub(super) fn new(
        owner: Owner,
        address: PhysicalAddress,
        super_block: SuperBlock,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            owner,
            range: FileRange::new(address, owner.layout().super_block_len(super_block)?)?,
            super_block,
        })
    }

    /// Returns the expected prefix fields and ranges for this secondary block.
    pub fn framing(&self) -> ExtensibleArrayBlockFraming {
        ExtensibleArrayBlockFraming::secondary(self)
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
        let (bitmap, data_blocks) = block::body(bytes, cursor.position())?
            .split_at_checked(owner.layout().bitmap_len(super_block)?.to_usize()?)
            .ok_or_else(|| super::internal("super block ends inside its page-init bitmap"))?;
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
        self.super_block.slots(self.owner.layout())
    }

    pub(super) fn owner(&self) -> Owner {
        self.owner
    }

    pub(super) fn offset(&self) -> u64 {
        self.super_block.offset()
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
                    let pages = match owner.layout().pages(block.elements()) {
                        Some(pages) => Some(PageBitmap::for_block(bitmap, local, pages)?),
                        None => None,
                    };
                    ExtensibleArrayDataBlockRef::new(owner, address?, block, pages)
                })
            })
    }
}

pub(super) const SUPER_BLOCK_SIGNATURE: [u8; 4] = *b"EASB";
