use alloc::vec::Vec;
use core::ops::Range;

use super::creation::{ExtensibleArrayCreationParameters, PageElementCount};
use super::header::ExtensibleArrayHeader;
use super::{BLOCK_BYTE_FIELDS, CHECKSUM_LEN, INDEX_BLOCK_SIGNATURE};
use crate::chunk_record::ElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;

// Super block `u` holds `2^floor(u/2)` data blocks of `min << ceil(u/2)` elements each, as
// `H5EA__hdr_init` computes them (`H5EAhdr.c`, HDF5 2.2.0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Layout {
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
    // `H5EA_SBLK_FIRST_IDX` and `H5EA_SIZEOF_OFFSET_BITS` (`H5EApkg.h`, HDF5 2.2.0)
    // determine the first super block and block-offset width.
    pub(super) fn new(
        creation: ExtensibleArrayCreationParameters,
        encoding: ElementEncoding,
    ) -> Self {
        Self {
            encoding,
            inline_elements: creation.inline_elements(),
            min_elements_bits: creation.min_elements().get().trailing_zeros(),
            page_nelmts: creation.bit_geometry().page_nelmts(),
            block_offset_len: creation.bit_geometry().max_index_bits().get().div_ceil(8),
            first_super_block: 2 * creation.min_data_blocks().get().trailing_zeros(),
            super_blocks: u32::from(creation.bit_geometry().max_index_bits().get()) + 1
                - creation.min_elements().get().trailing_zeros(),
        }
    }

    pub(super) fn encoding(self) -> ElementEncoding {
        self.encoding
    }

    pub(super) fn inline_elements(self) -> u8 {
        self.inline_elements
    }

    pub(super) fn page_nelmts(self) -> PageElementCount {
        self.page_nelmts
    }

    pub(super) fn block_offset_len(self) -> u8 {
        self.block_offset_len
    }

    pub(super) fn first_super_block(self) -> u32 {
        self.first_super_block
    }

    pub(super) fn super_blocks(self) -> u32 {
        self.super_blocks
    }

    pub(super) fn direct_blocks(self) -> impl Iterator<Item = Result<DataBlock, FormatError>> {
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
    pub(super) fn direct_data_blocks(self) -> u64 {
        // Construction requires at least two data block pointers per super block, so
        // `first_super_block` is at least 2 and the shift is at least 1.
        2 * ((1 << (self.first_super_block / 2)) - 1)
    }

    pub(super) fn super_block(self, index: u32) -> Result<SuperBlock, FormatError> {
        // Every caller passes an index below `super_blocks`, whose checked value is at most 65, so
        // the subtraction and the shifts below stay inside a `u64`.
        debug_assert!(index < self.super_blocks);
        let min_elements = 1 << self.min_elements_bits;
        // `2^index - 1`.
        let before = u64::MAX.checked_shr(u64::BITS - index).unwrap_or(0);
        Ok(SuperBlock {
            offset: super::product(min_elements, before)?,
            blocks: 1 << (index / 2),
            block_elements: min_elements << index.div_ceil(2),
        })
    }

    // A data block of more elements than a page is paged, as `H5EA__dblock_alloc` pages it
    // (`H5EAdblock.c`, HDF5 2.2.0).
    pub(super) fn pages(self, elements: u64) -> Option<u64> {
        let pages = DataBlockGeometry {
            dblk_nelmts: elements,
            page_nelmts: self.page_nelmts,
        }
        .npages();
        (pages != 0).then_some(pages)
    }

    /// Returns the slot of the first element of the block at `offset`.
    pub(super) fn first_slot(self, offset: u64) -> Result<u64, FormatError> {
        super::sum([u64::from(self.inline_elements), offset])
    }

    pub(super) fn slots(self, offset: u64, elements: u64) -> Result<Range<u64>, FormatError> {
        let start = self.first_slot(offset)?;
        Ok(start..super::sum([start, elements])?)
    }

    pub(super) fn elements_len(self, elements: u64) -> Result<u64, FormatError> {
        super::product(elements, u64::from(self.encoding.stride().get()))
    }

    pub(super) fn addresses_len(self, addresses: u64) -> Result<u64, FormatError> {
        super::product(addresses, u64::from(self.encoding.offsets().get()))
    }

    pub(super) fn index_block_len(self) -> Result<u64, FormatError> {
        // `ExtensibleArrayCreationParameters::from_bit_geometry` checks that `first_super_block`
        // is at most `super_blocks`.
        super::sum([
            self.prefix_len(),
            self.elements_len(u64::from(self.inline_elements))?,
            self.addresses_len(self.direct_data_blocks())?,
            self.addresses_len(u64::from(self.super_blocks - self.first_super_block))?,
            CHECKSUM_LEN.to_u64(),
        ])
    }

    // Each data block has `ceil(pages / 8)` bytes of the bitmap, as `H5EA__sblock_alloc` sizes it
    // (`H5EAsblock.c`, HDF5 2.2.0).
    pub(super) fn bitmap_len(self, super_block: SuperBlock) -> Result<u64, FormatError> {
        self.pages(super_block.block_elements)
            .map_or(Ok(0), |pages| {
                super::product(super_block.blocks, pages.div_ceil(8))
            })
    }

    pub(super) fn super_block_len(self, super_block: SuperBlock) -> Result<u64, FormatError> {
        super::sum([
            self.prefix_len(),
            u64::from(self.block_offset_len),
            self.bitmap_len(super_block)?,
            self.addresses_len(super_block.blocks)?,
            CHECKSUM_LEN.to_u64(),
        ])
    }

    pub(super) fn data_block_len(self, block: DataBlock) -> Result<u64, FormatError> {
        match self.pages(block.elements) {
            Some(pages) => super::sum([
                self.pages_offset(),
                super::product(pages, self.page_len()?)?,
            ]),
            None => super::sum([
                self.prefix_len(),
                u64::from(self.block_offset_len),
                self.elements_len(block.elements)?,
                CHECKSUM_LEN.to_u64(),
            ]),
        }
    }

    pub(super) fn pages_offset(self) -> u64 {
        self.pages_offset_usize().to_u64()
    }

    pub(super) fn page_len(self) -> Result<u64, FormatError> {
        super::sum([
            self.elements_len(self.page_nelmts.get())?,
            CHECKSUM_LEN.to_u64(),
        ])
    }

    pub(super) fn prefix_len(self) -> u64 {
        self.prefix_len_usize().to_u64()
    }

    pub(super) fn pages_offset_usize(self) -> usize {
        self.prefix_len_usize() + usize::from(self.block_offset_len) + CHECKSUM_LEN
    }

    pub(super) fn prefix_len_usize(self) -> usize {
        INDEX_BLOCK_SIGNATURE.len() + BLOCK_BYTE_FIELDS + usize::from(self.encoding.offsets().get())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SuperBlock {
    /// The block offset of the super block and of its first data block.
    pub(super) offset: u64,
    pub(super) blocks: u64,
    pub(super) block_elements: u64,
}

impl SuperBlock {
    pub(super) fn data_block(self, local: u64) -> Result<DataBlock, FormatError> {
        Ok(DataBlock {
            offset: super::sum([self.offset, super::product(local, self.block_elements)?])?,
            elements: self.block_elements,
            libhdf5_offset: None,
        })
    }

    // `ordinal` counts the data blocks the index block holds the addresses of, from the first.
    fn direct_data_block(self, local: u64, ordinal: u64) -> Result<DataBlock, FormatError> {
        Ok(DataBlock {
            libhdf5_offset: Some(super::sum([
                self.offset,
                super::product(ordinal, self.block_elements)?,
            ])?),
            ..self.data_block(local)?
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DataBlock {
    pub(super) offset: u64,
    pub(super) elements: u64,
    /// The block offset the C library stores in a data block whose address the index block holds,
    /// or `None` in one whose address a super block holds.
    pub(super) libhdf5_offset: Option<u64>,
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
    sblks: Vec<(u64, u64)>,
    /// The number of elements in each data block the index block addresses.
    direct_dblk_nelmts: Vec<u64>,
    /// The number of super block addresses in the index block.
    nsblk_addrs: usize,
    /// The index in [`Self::sblks`] of the first super block addressed by the index block.
    first_indirect_sblk: usize,
}

impl ExtensibleArrayGeometry {
    /// Returns the geometry of the super block the `j`th super block address of the index block
    /// points at, in an array whose pages hold `page_nelmts` elements.
    ///
    /// `j` counts the super block addresses, and the first `first_indirect_sblk` super blocks have
    /// none.
    #[cfg(test)]
    fn super_block_at(&self, j: usize, page_nelmts: PageElementCount) -> SuperBlockGeometry {
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
    pub(super) fn from_layout(layout: Layout) -> Result<Self, FormatError> {
        let mut sblks = Vec::with_capacity(layout.super_blocks().to_usize()?);
        for index in 0..layout.super_blocks() {
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
            nsblk_addrs: (layout.super_blocks() - layout.first_super_block()).to_usize()?,
            first_indirect_sblk: layout.first_super_block().to_usize()?,
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
    /// The total bitmap length must fit in a [`u64`].
    ///
    /// # Panics
    ///
    /// Panics if overflow checks are enabled and the total length exceeds [`u64::MAX`].
    pub const fn bitmap_size(self) -> u64 {
        self.ndblks * self.blocks.npages().div_ceil(8)
    }
}

#[cfg(test)]
#[path = "tests/geometry.rs"]
mod tests;
