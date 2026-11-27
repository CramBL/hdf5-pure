use hdf5_pure_core::__private::{FileRange, FileRegion};

use super::block::{self, Owner};
use super::data_block::ExtensibleArrayDataBlockRef;
use super::elements::ExtensibleArrayElements;
use super::secondary_block::ExtensibleArraySuperBlockRef;
use crate::address::StoredAddress;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;

/// The position and the layout of an Extensible Array index block, from
/// [`super::ExtensibleArray::index_block`].
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayIndexBlockRef {
    owner: Owner,
    range: FileRange,
}

impl ExtensibleArrayIndexBlockRef {
    pub(super) fn new(owner: Owner, address: StoredAddress) -> Result<Self, FormatError> {
        Ok(Self {
            owner,
            range: FileRange::new(owner.resolve(address)?, owner.layout().index_block_len()?)?,
        })
    }

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
        let layout = owner.layout();
        let (elements, addresses) = block::body(bytes, cursor.position())?
            .split_at_checked(
                layout
                    .elements_len(u64::from(layout.inline_elements()))?
                    .to_usize()?,
            )
            .ok_or_else(|| block::internal("index block ends inside its elements"))?;
        let (data_blocks, super_blocks) = addresses
            .split_at_checked(
                layout
                    .addresses_len(layout.direct_data_blocks())?
                    .to_usize()?,
            )
            .ok_or_else(|| block::internal("index block ends inside its data block addresses"))?;
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
            .zip(owner.layout().direct_blocks())
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
        let layout = owner.layout();
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

pub(super) const INDEX_BLOCK_SIGNATURE: [u8; 4] = *b"EAIB";
