use hdf5_pure_core::__private::DecodedStoredAddress;

use crate::address::StoredAddress;
use crate::bytes;
use crate::error::FormatError;
use crate::fractal_heap::{
    FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED, FractalHeapFiltering, FractalHeapHeader,
    FractalHeapLayout,
};
use crate::width::{LengthWidth, OffsetWidth};

use super::direct::FractalHeapDirectBlock;
use super::indirect::FractalHeapIndirectBlock;

/// Describes managed blocks using a heap header's doubling table and filter metadata.
///
/// The caller obtains each block's stored address, expected heap offset, and size, then
/// converts the address to a physical position before parsing the supplied
/// [`FileRegion`](hdf5_pure_core::__private::FileRegion).
///
/// # Examples
///
/// ```
/// # use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion};
/// # use hdf5_pure_core::PhysicalAddress;
/// # use hdf5_pure_core::BaseAddress;
/// # use hdf5_pure_format::__private::{FractalHeapBlock, FractalHeapBlockLayout, FractalHeapHeader, LengthWidth, OffsetWidth};
/// # use hdf5_pure_format::FormatError;
/// # use test_util::{fractal_heap, widths::Widths};
/// # let fixture = fractal_heap::Header::new(256);
/// # let encoded = fixture.build(Widths::EIGHT);
/// # let header = FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight)?;
/// # let bytes = fixture.direct_block(96, &[7; 113], Widths::EIGHT);
/// # let image = FileRegion::new(FileRange::new(PhysicalAddress::new(768), 128)?, &bytes)?;
/// let base = BaseAddress::new(512);
/// let blocks = FractalHeapBlockLayout::new(
///     &header, OffsetWidth::Eight, LengthWidth::Eight, base.relative(PhysicalAddress::new(608))?,
/// )?;
/// let mut pending = blocks.root()?.into_iter().collect::<Vec<_>>();
/// while let Some(block) = pending.pop() {
///     match block {
///         FractalHeapBlock::Direct(block) => {
///             let physical = base.resolve(block.address())?;
///             let reference = block.at(physical)?;
///             reference.parse_prefix(image.subregion(reference.prefix_range()?)?)?;
///             let view = reference.parse(image.subregion(reference.parse_range())?)?;
///             assert_eq!(view.data(), &[7; 113]);
///         }
///         FractalHeapBlock::Indirect(block) => {
///             let physical = base.resolve(block.address())?;
///             let reference = block.at(physical)?;
///             let view = reference.parse(image.subregion(reference.parse_range())?)?;
///             pending.extend(view.children().collect::<Result<Vec<_>, _>>()?);
///         }
///     }
/// }
/// # Ok::<(), FormatError>(())
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapBlockLayout {
    layout: FractalHeapLayout,
    owner: StoredAddress,
    root: DecodedStoredAddress,
    root_rows: u16,
    starting_block_size: u64,
    heap_limit: Option<u64>,
    checksummed: bool,
    filtering: FractalHeapFiltering,
}

impl FractalHeapBlockLayout {
    /// Validates block geometry for the heap header at `owner`.
    ///
    /// `owner` is the header's base-relative address. The widths are those parsed from the
    /// superblock.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if an address does not fit `offsets`,
    /// `owner` is undefined, or the doubling table, root block, or filter metadata is invalid.
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn new(
        header: &FractalHeapHeader,
        offsets: OffsetWidth,
        lengths: LengthWidth,
        owner: StoredAddress,
    ) -> Result<Self, FormatError> {
        if owner.get() >= StoredAddress::undefined(offsets).get()
            || header.root_block_address.get() > StoredAddress::undefined(offsets).get()
        {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let layout = FractalHeapLayout::new(header, offsets, lengths)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        if header.current_rows_in_root_indirect_block > layout.max_root_rows() {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let heap = Self {
            layout,
            owner,
            root: if header.root_block_address.is_undefined(offsets) {
                DecodedStoredAddress::Undefined
            } else {
                DecodedStoredAddress::Defined(header.root_block_address)
            },
            root_rows: header.current_rows_in_root_indirect_block,
            starting_block_size: header.starting_block_size,
            heap_limit: (header.max_heap_size < 64)
                .then(|| 1u64 << u32::from(header.max_heap_size)),
            checksummed: header.flags & FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED != 0,
            filtering: header.filtering.clone(),
        };
        heap.root()?;
        Ok(heap)
    }

    /// Returns the root managed block, or [`None`] if its stored address is undefined.
    ///
    /// Huge objects may exist in a heap without a managed root block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the root block's geometry or
    /// filter metadata is invalid.
    pub fn root(&self) -> Result<Option<FractalHeapBlock<'_>>, FormatError> {
        let DecodedStoredAddress::Defined(root) = self.root else {
            return Ok(None);
        };
        Ok(Some(if self.root_rows == 0 {
            let (encoded_size, mask) = match self.filtering {
                FractalHeapFiltering::Unfiltered => (self.starting_block_size, None),
                FractalHeapFiltering::Filtered {
                    root_direct_block_size,
                    root_filter_mask,
                    ..
                } => (root_direct_block_size, Some(root_filter_mask)),
            };
            FractalHeapBlock::Direct(self.direct(
                root,
                0,
                self.starting_block_size,
                encoded_size,
                mask,
            )?)
        } else {
            FractalHeapBlock::Indirect(self.indirect(root, 0, self.root_rows)?)
        }))
    }

    /// Returns the doubling-table geometry used to derive block sizes and offsets.
    pub fn layout(&self) -> &FractalHeapLayout {
        &self.layout
    }

    pub(super) fn checksummed(&self) -> bool {
        self.checksummed
    }

    pub(super) fn filtering(&self) -> &FractalHeapFiltering {
        &self.filtering
    }

    pub(super) fn contains(&self, start: u64, length: u64) -> Result<(), FormatError> {
        let end = start
            .checked_add(length)
            .ok_or(FormatError::InvalidFractalHeapGeometry)?;
        if self.heap_limit.is_some_and(|limit| end > limit) {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(())
    }

    pub(super) fn parse_prefix(
        &self,
        block: &[u8],
        signature: &[u8; 4],
        heap_offset: u64,
    ) -> Result<(), FormatError> {
        let mut cursor = bytes::ByteCursor::new(block);
        if cursor.array::<4>()? != *signature {
            return Err(FormatError::InvalidFractalHeapSignature);
        }
        let version = cursor.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidFractalHeapVersion(version));
        }
        let owner = cursor.address(self.layout.offset_width())?;
        if owner != self.owner {
            return Err(FormatError::InvalidFractalHeapBlockOwner {
                expected: self.owner.get(),
                actual: owner.get(),
            });
        }
        let mut offset = 0u64;
        for shift in 0..self.layout.block_offset_size() {
            offset |= u64::from(cursor.u8()?) << (shift * 8);
        }
        if offset != heap_offset {
            return Err(FormatError::InvalidFractalHeapBlockOffset {
                expected: heap_offset,
                actual: offset,
            });
        }
        Ok(())
    }

    pub(super) fn prefix_len(&self) -> usize {
        SIGNATURE_LEN
            + 1
            + usize::from(self.layout.offset_width().get())
            + self.layout.block_offset_size()
    }
}

/// Describes a managed block at a stored address in the heap's doubling table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapBlock<'a> {
    /// A block containing managed-object data.
    Direct(FractalHeapDirectBlock<'a>),
    /// A block containing child block entries.
    Indirect(FractalHeapIndirectBlock<'a>),
}

// The block signatures, version, and checksum width are defined in "Fractal Heap"
// of the format specification, version 4.0.
pub(crate) const DIRECT_SIGNATURE: [u8; 4] = *b"FHDB";
pub(crate) const INDIRECT_SIGNATURE: [u8; 4] = *b"FHIB";
pub(crate) const VERSION: u8 = 0;
const SIGNATURE_LEN: usize = 4;
pub(crate) const CHECKSUM_LEN: usize = 4;
pub(super) const FILTER_MASK_LEN: usize = 4;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
