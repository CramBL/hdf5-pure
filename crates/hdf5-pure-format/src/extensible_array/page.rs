use core::ops::Range;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use super::block::{self, Owner};
use super::elements::ExtensibleArrayElements;
use super::geometry::DataBlock;
use crate::error::FormatError;
use crate::{checksum, chunk_record};

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
    pub(super) fn new(
        owner: Owner,
        range: FileRange,
        block: DataBlock,
        bitmap: PageBitmap<'a>,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            owner,
            first_page: range.start().checked_add(owner.layout().pages_offset())?,
            first_slot: owner.layout().first_slot(block.offset())?,
            bitmap,
        })
    }

    /// Returns the initialized pages whose first slot is below the header's "Max Index Set",
    /// in page order.
    pub fn pages(
        &self,
    ) -> impl Iterator<Item = Result<ExtensibleArrayPageRef, FormatError>> + Clone + 'a {
        let Self {
            owner,
            first_page,
            first_slot,
            bitmap,
        } = *self;
        let layout = owner.layout();
        let page_elements = layout.page_nelmts().get();
        (0..bitmap.pages)
            .take_while(move |&page| {
                page.checked_mul(page_elements)
                    .and_then(|offset| offset.checked_add(first_slot))
                    .is_some_and(|slot| slot < owner.max_index_set())
            })
            .filter(move |&page| bitmap.initialized(page))
            .map(move |page| {
                let page_len = layout.page_len()?;
                Ok(ExtensibleArrayPageRef {
                    owner,
                    range: FileRange::new(
                        first_page.checked_add(super::product(page, page_len)?)?,
                        page_len,
                    )?,
                    first_slot: super::sum([first_slot, super::product(page, page_elements)?])?,
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
            block::body(bytes, 0)?,
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
        let elements = self.owner.layout().page_nelmts().get();
        let end = super::sum([self.first_slot, elements])?;
        Ok(self.first_slot..end)
    }
}

// The bits of the data blocks of a super block form one sequence, and page `p` of data block `k`
// is bit `k * pages + p`, as `H5EA__lookup_elmt` indexes it (`H5EA.c`, HDF5 2.2.0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PageBitmap<'a> {
    bitmap: &'a [u8],
    /// The bit of the first page of the data block.
    first_page: u64,
    /// The number of pages in the data block.
    pages: u64,
}

impl<'a> PageBitmap<'a> {
    pub(super) fn for_block(bitmap: &'a [u8], local: u64, pages: u64) -> Result<Self, FormatError> {
        Ok(Self {
            bitmap,
            first_page: super::product(local, pages)?,
            pages,
        })
    }

    fn initialized(self, page: u64) -> bool {
        self.first_page
            .checked_add(page)
            .is_some_and(|bit| chunk_record::page_initialized(self.bitmap, bit))
    }
}
