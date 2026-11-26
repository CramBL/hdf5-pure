use hdf5_pure_core::__private::{DecodedStoredAddress, FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::error::FormatError;

use super::layout::FILTER_MASK_LEN;
use super::{FractalHeapBlock, FractalHeapBlockLayout, INDIRECT_SIGNATURE};

/// Describes an indirect block's stored address, heap offset, and row count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIndirectBlock<'a> {
    heap: &'a FractalHeapBlockLayout,
    address: StoredAddress,
    heap_offset: u64,
    rows: u16,
}

impl<'a> FractalHeapIndirectBlock<'a> {
    /// Associates the complete indirect-block allocation with a physical address.
    ///
    /// The caller resolves [`Self::address`] using the opened image's effective base.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the block size cannot be
    /// derived, or [`FormatError::OffsetOverflow`] if the allocation's end exceeds [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn at(
        self,
        address: PhysicalAddress,
    ) -> Result<FractalHeapIndirectBlockRef<'a>, FormatError> {
        let size = self
            .heap
            .layout()
            .indirect_block_size(self.rows)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        Ok(FractalHeapIndirectBlockRef {
            block: self,
            range: FileRange::new(address, size)?,
        })
    }

    /// Returns the block's base-relative file address.
    pub const fn address(&self) -> StoredAddress {
        self.address
    }
    /// Returns the block's offset in the heap's managed address space, in bytes.
    pub const fn heap_offset(&self) -> u64 {
        self.heap_offset
    }
    /// Returns the number of doubling-table rows encoded in this block.
    pub const fn rows(&self) -> u16 {
        self.rows
    }
}

/// Describes an indirect block's physical allocation and expected structural fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIndirectBlockRef<'a> {
    block: FractalHeapIndirectBlock<'a>,
    range: FileRange,
}

impl<'heap> FractalHeapIndirectBlockRef<'heap> {
    /// Parses the block prefix and borrows the encoded child entries.
    ///
    /// Validates the signature, version, owner, and heap offset. With the `checksum` feature,
    /// validates the trailing checksum. Child geometry is validated by
    /// [`FractalHeapIndirectBlockView::child`] or while iterating
    /// [`FractalHeapIndirectBlockView::children`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`Self::parse_range`], or [`FormatError::InvalidFractalHeapSignature`],
    /// [`FormatError::InvalidFractalHeapVersion`], [`FormatError::InvalidFractalHeapBlockOwner`],
    /// or [`FormatError::InvalidFractalHeapBlockOffset`] if a prefix field is invalid.
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if checksum validation fails.
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn parse<'bytes>(
        &self,
        region: FileRegion<'bytes>,
    ) -> Result<FractalHeapIndirectBlockView<'heap, 'bytes>, FormatError> {
        let bytes = region.subregion(self.range)?.bytes();
        self.block
            .heap
            .parse_prefix(bytes, &INDIRECT_SIGNATURE, self.block.heap_offset)?;
        checksum::verify_trailing(bytes)?;
        Ok(FractalHeapIndirectBlockView {
            block: self.block,
            bytes,
        })
    }

    /// Returns the complete physical allocation, including the trailing checksum.
    pub const fn range(&self) -> FileRange {
        self.range
    }
    /// Returns the physical bytes required for parsing, equal to [`Self::range`].
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
}

/// Borrows an indirect block after prefix and enabled checksum validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIndirectBlockView<'heap, 'bytes> {
    block: FractalHeapIndirectBlock<'heap>,
    bytes: &'bytes [u8],
}

impl<'heap, 'bytes> FractalHeapIndirectBlockView<'heap, 'bytes> {
    /// Iterates allocated children in row order, then column order.
    ///
    /// Undefined entries are omitted. Each item is validated as in [`Self::child`].
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn children(
        &self,
    ) -> impl Iterator<Item = Result<FractalHeapBlock<'heap>, FormatError>> + '_ {
        (0..usize::from(self.block.rows)).flat_map(move |row| {
            (0..self.block.heap.layout().table_width())
                .filter_map(move |column| self.child(row, column).transpose())
        })
    }

    /// Derives the child at a zero-based doubling-table slot.
    ///
    /// Returns [`None`] for an undefined address with valid unused filter metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the slot is out of bounds,
    /// the child exceeds the heap address space, or the child size, row count, or filter
    /// metadata is invalid.
    pub fn child(
        &self,
        row: usize,
        column: usize,
    ) -> Result<Option<FractalHeapBlock<'heap>>, FormatError> {
        let FractalHeapIndirectBlock {
            heap,
            heap_offset,
            rows,
            address: _,
        } = self.block;
        let layout = heap.layout();
        if row >= usize::from(rows) || column >= layout.table_width() {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let direct = row < layout.direct_rows();
        let direct_count = usize::from(rows).min(layout.direct_rows()) * layout.table_width();
        let entry = row * layout.table_width() + column;
        let offset_size = usize::from(layout.offset_width().get());
        let filter_size = if heap.filtering().is_filtered() {
            usize::from(layout.length_width().get()) + FILTER_MASK_LEN
        } else {
            0
        };
        let pos = heap.prefix_len() + entry * offset_size + entry.min(direct_count) * filter_size;
        let address = crate::width::decode_stored_address(layout.offset_width(), self.bytes, pos)?;
        let filtered = if direct && filter_size != 0 {
            let mut cursor =
                bytes::ByteCursor::with_position(self.bytes, pos.saturating_add(offset_size))?;
            Some((cursor.length(layout.length_width())?, cursor.u32()?))
        } else {
            None
        };
        let DecodedStoredAddress::Defined(address) = address else {
            if filtered.is_some_and(|(size, _mask)| size != 0) {
                return Err(FormatError::InvalidFractalHeapGeometry);
            }
            return Ok(None);
        };
        let offset = heap_offset
            .checked_add(
                layout
                    .slot_offset(row, column)
                    .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?,
            )
            .ok_or(FormatError::InvalidFractalHeapGeometry)?;
        let size = layout
            .block_size_for_row(row)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        Ok(Some(if direct {
            let (encoded_size, mask) = match filtered {
                Some((encoded_size, mask)) => (encoded_size, Some(mask)),
                None => (size, None),
            };
            FractalHeapBlock::Direct(heap.direct(address, offset, size, encoded_size, mask)?)
        } else {
            let child_rows = layout
                .child_indirect_rows(row)
                .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
            if child_rows >= rows
                || layout
                    .indirect_heap_size(child_rows)
                    .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?
                    != size
            {
                return Err(FormatError::InvalidFractalHeapGeometry);
            }
            FractalHeapBlock::Indirect(heap.indirect(address, offset, child_rows)?)
        }))
    }

    /// Returns the complete encoded block, including its prefix and checksum.
    pub const fn bytes(&self) -> &'bytes [u8] {
        self.bytes
    }
}

impl FractalHeapBlockLayout {
    pub(super) fn indirect(
        &self,
        address: StoredAddress,
        heap_offset: u64,
        rows: u16,
    ) -> Result<FractalHeapIndirectBlock<'_>, FormatError> {
        let span = self
            .layout()
            .indirect_heap_size(rows)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        self.contains(heap_offset, span)?;
        Ok(FractalHeapIndirectBlock {
            heap: self,
            address,
            heap_offset,
            rows,
        })
    }
}
