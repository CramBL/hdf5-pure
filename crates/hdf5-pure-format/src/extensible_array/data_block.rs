use core::ops::Range;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use super::block::{self, ExtensibleArrayBlockFraming, Owner};
use super::elements::ExtensibleArrayElements;
use super::geometry::DataBlock;
use super::page::{ExtensibleArrayPages, PageBitmap};
use crate::checksum;
use crate::error::FormatError;

/// The position and the layout of an Extensible Array data block, from
/// [`super::ExtensibleArrayIndexBlock::data_blocks`] or
/// [`super::ExtensibleArraySuperBlock::data_blocks`].
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
    pub(super) fn new(
        owner: Owner,
        address: PhysicalAddress,
        block: DataBlock,
        pages: Option<PageBitmap<'a>>,
    ) -> Result<Self, FormatError> {
        let layout = owner.layout();
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
    /// [`FormatError::InvalidExtensibleArrayBlockOffset`] if the block offset matches neither its
    /// position nor an accepted libhdf5 offset, [`FormatError::InvalidExtensibleArrayGeometry`] if the
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
                    block::body(bytes, cursor.position())?,
                    owner.layout().first_slot(block.offset())?,
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
            .map(|bitmap| ExtensibleArrayPages::new(owner, range, block, bitmap))
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
        self.block.slots(self.owner.layout())
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

pub(super) const DATA_BLOCK_SIGNATURE: [u8; 4] = *b"EADB";
