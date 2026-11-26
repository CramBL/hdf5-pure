use alloc::vec::Vec;

use super::geometry::FractalHeapGeometry;
use super::geometry::FractalHeapLayoutError;
use super::header::FractalHeapHeader;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::error::FormatError;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The doubling-table geometry and encoded field widths of a fractal heap.
///
/// The layout derives the managed-block geometry from a [`FractalHeapHeader`] once and provides
/// the same checked row, span, and encoded-size calculations to object lookup and storage
/// ownership traversal. Callers perform file I/O and traversal. The doubling table is defined in
/// "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapLayout {
    offset_width: OffsetWidth,
    length_width: LengthWidth,
    block_offset_size: usize,
    geometry: FractalHeapGeometry,
    filtered: bool,
    indirect_block_sizes: Vec<u64>,
}

impl FractalHeapLayout {
    /// Derives and validates the doubling-table geometry declared by `header`.
    ///
    /// `offset_width` and `length_width` are the file's address and length field widths.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if the header fields contradict the
    /// doubling-table constraints, or if a row size, row offset, or encoded indirect-block size
    /// exceeds [`u64::MAX`].
    pub fn new(
        header: &FractalHeapHeader,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<Self, FractalHeapLayoutError> {
        let geometry = FractalHeapGeometry::new(
            header.table_width,
            header.starting_block_size,
            header.max_direct_block_size,
            header.max_heap_size,
        )?;
        let max_root_rows = geometry.max_root_rows();
        if header.start_root_rows > max_root_rows {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let direct_rows = geometry.direct_rows();
        let table_width_u64 = u64::from(header.table_width);
        let block_offset_size = usize::from(header.max_heap_size).div_ceil(8);

        let prefix = 5u64
            .checked_add(u64::from(offset_width.get()))
            .and_then(|n| n.checked_add(u64::try_from(block_offset_size).ok()?))
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        let direct_entry_size = u64::from(offset_width.get())
            .checked_add(if !header.filtering.is_filtered() {
                0
            } else {
                u64::from(length_width.get()) + 4
            })
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        let indirect_entry_size = u64::from(offset_width.get());
        let mut indirect_block_sizes = Vec::with_capacity(usize::from(max_root_rows));
        for nrows in 1..=usize::from(max_root_rows) {
            let direct = nrows.min(direct_rows);
            let indirect = nrows - direct;
            let entry_row_bytes = u64::try_from(direct)
                .ok()
                .and_then(|rows| rows.checked_mul(direct_entry_size))
                .and_then(|bytes| {
                    u64::try_from(indirect)
                        .ok()
                        .and_then(|rows| rows.checked_mul(indirect_entry_size))
                        .and_then(|more| bytes.checked_add(more))
                })
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            let entries = entry_row_bytes
                .checked_mul(table_width_u64)
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            let size = prefix
                .checked_add(entries)
                .and_then(|n| n.checked_add(4))
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            indirect_block_sizes.push(size);
        }

        Ok(Self {
            offset_width,
            length_width,
            block_offset_size,
            geometry,
            filtered: header.filtering.is_filtered(),
            indirect_block_sizes,
        })
    }

    /// Returns the heap's doubling-table geometry independently of its encoded field widths.
    pub const fn geometry(&self) -> &FractalHeapGeometry {
        &self.geometry
    }

    /// Returns the file's address field width.
    pub const fn offset_width(&self) -> OffsetWidth {
        self.offset_width
    }

    /// Returns the file's length field width.
    pub const fn length_width(&self) -> LengthWidth {
        self.length_width
    }

    /// Returns the width in bytes of a managed block's "Block Offset" field.
    pub const fn block_offset_size(&self) -> usize {
        self.block_offset_size
    }

    /// Returns the number of child slots in each doubling-table row.
    pub const fn table_width(&self) -> usize {
        self.geometry.table_width()
    }

    /// Returns the number of doubling-table rows whose entries address direct blocks.
    pub const fn direct_rows(&self) -> usize {
        self.geometry.direct_rows()
    }

    /// Returns the maximum legal number of rows in the root indirect block.
    pub const fn max_root_rows(&self) -> u16 {
        self.geometry.max_root_rows()
    }

    /// Returns the size in bytes of the managed address space covered by one slot in `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` is outside the managed heap's
    /// legal doubling-table rows.
    pub fn block_size_for_row(&self, row: usize) -> Result<u64, FractalHeapLayoutError> {
        self.geometry.block_size_for_row(row)
    }

    /// Returns the row count of the child indirect blocks addressed by doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` is a direct row or lies outside
    /// the managed heap's legal doubling-table rows.
    pub fn child_indirect_rows(&self, row: usize) -> Result<u16, FractalHeapLayoutError> {
        self.geometry.child_indirect_rows(row)
    }

    /// Returns the managed heap-address space represented by an indirect block of `nrows` rows.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `nrows` is zero, exceeds the
    /// maximum root row count, or covers more than [`u64::MAX`] bytes of managed address space.
    pub fn indirect_heap_size(&self, nrows: u16) -> Result<u64, FractalHeapLayoutError> {
        self.geometry.indirect_heap_size(nrows)
    }

    /// Returns the exact encoded allocation size of an indirect block of `nrows` rows.
    ///
    /// The size includes the `FHIB` signature, version, heap-header address, block heap offset,
    /// child entries, and trailing checksum. Filtered heaps use their wider direct-child entries.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `nrows` is zero or exceeds the
    /// maximum root row count.
    pub fn indirect_block_size(&self, nrows: u16) -> Result<u64, FractalHeapLayoutError> {
        let index = usize::from(nrows)
            .checked_sub(1)
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        self.indirect_block_sizes
            .get(index)
            .copied()
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the encoded indirect-block length through the end of its child entries.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `nrows` is zero or exceeds the
    /// maximum root row count.
    pub fn indirect_block_entries_len(&self, nrows: u16) -> Result<u64, FractalHeapLayoutError> {
        self.indirect_block_size(nrows)?
            .checked_sub(4)
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the heap offset of the first child slot in doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` lies outside the managed
    /// heap's legal doubling-table rows.
    pub fn row_offset(&self, row: usize) -> Result<u64, FractalHeapLayoutError> {
        self.geometry.row_offset(row)
    }

    /// Returns the heap offset of child slot `column` in doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if the row or column lies outside the
    /// doubling table.
    pub fn slot_offset(&self, row: usize, column: usize) -> Result<u64, FractalHeapLayoutError> {
        self.geometry.slot_offset(row, column)
    }

    /// Returns the allocated child whose managed heap space contains `target_offset`.
    ///
    /// `block` begins at the `FHIB` signature, and `iblock_heap_offset` is the managed heap offset
    /// represented by the indirect block itself.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapSignature`] if `block` lacks the `FHIB` signature,
    /// [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters managed blocks, or
    /// [`FormatError::UnexpectedEof`] if the signature or a child address is truncated.
    /// Returns [`FormatError::ChunkedReadError`] if the row count is invalid or a child
    /// offset or extent overflows.
    pub fn find_child_for_offset(
        &self,
        block: &[u8],
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
    ) -> Result<Option<FractalHeapChild>, FormatError> {
        bytes::ensure_len(block, 0, 4)?;
        if &block[..4] != b"FHIB" {
            return Err(FormatError::InvalidFractalHeapSignature);
        }
        if self.filtered {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if nrows == 0 || nrows > self.max_root_rows() {
            return Err(Self::geometry_format_error());
        }

        let entries_at = 5usize
            .checked_add(usize::from(self.offset_width.get()))
            .and_then(|n| n.checked_add(self.block_offset_size))
            .ok_or_else(Self::geometry_format_error)?;
        let direct_rows = usize::from(nrows).min(self.direct_rows());
        let offset_size = usize::from(self.offset_width.get());

        for row in 0..usize::from(nrows) {
            let slot_size = self
                .block_size_for_row(row)
                .map_err(|_error| Self::geometry_format_error())?;
            let child_rows = if row < direct_rows {
                None
            } else {
                Some(
                    self.child_indirect_rows(row)
                        .map_err(|_error| Self::geometry_format_error())?,
                )
            };
            for column in 0..self.table_width() {
                let entry = row
                    .checked_mul(self.table_width())
                    .and_then(|n| n.checked_add(column))
                    .ok_or_else(Self::geometry_format_error)?;
                let pos = entries_at
                    .checked_add(
                        entry
                            .checked_mul(offset_size)
                            .ok_or_else(Self::geometry_format_error)?,
                    )
                    .ok_or_else(Self::geometry_format_error)?;
                let child_addr = bytes::read_offset_width(block, pos, self.offset_width)?;
                if StoredAddress::new(child_addr).is_undefined(self.offset_width) {
                    continue;
                }
                let child_heap_offset = iblock_heap_offset
                    .checked_add(
                        self.slot_offset(row, column)
                            .map_err(|_error| Self::geometry_format_error())?,
                    )
                    .ok_or_else(Self::geometry_format_error)?;
                let span = match child_rows {
                    Some(rows) => self
                        .indirect_heap_size(rows)
                        .map_err(|_error| Self::geometry_format_error())?,
                    None => slot_size,
                };
                let child_end = child_heap_offset
                    .checked_add(span)
                    .ok_or_else(Self::geometry_format_error)?;
                if target_offset < child_heap_offset || target_offset >= child_end {
                    continue;
                }
                return Ok(Some(match child_rows {
                    Some(nrows) => FractalHeapChild::Indirect {
                        addr: StoredAddress::new(child_addr),
                        nrows,
                        heap_offset: child_heap_offset,
                    },
                    None => FractalHeapChild::Direct {
                        addr: StoredAddress::new(child_addr),
                        block_size: slot_size,
                        heap_offset: child_heap_offset,
                    },
                }));
            }
        }
        Ok(None)
    }

    fn geometry_format_error() -> FormatError {
        FormatError::ChunkedReadError("fractal heap: invalid doubling-table geometry".into())
    }
}

/// A child of a fractal heap indirect block: a direct block, which holds objects, or an indirect
/// block one level down.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapChild {
    /// A direct block.
    Direct {
        /// The address of the block.
        addr: StoredAddress,
        /// The size of the block in bytes.
        block_size: u64,
        /// The heap offset the space of the block begins at.
        heap_offset: u64,
    },
    /// An indirect block.
    Indirect {
        /// The address of the block.
        addr: StoredAddress,
        /// The number of rows of the block.
        nrows: u16,
        /// The heap offset the space of the block begins at.
        heap_offset: u64,
    },
}

#[cfg(test)]
#[path = "tests/layout.rs"]
mod tests;
