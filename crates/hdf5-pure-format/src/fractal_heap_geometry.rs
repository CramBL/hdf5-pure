//! Checked doubling-table geometry in a fractal heap's managed address space.

use alloc::vec::Vec;

use crate::fractal_heap::FractalHeapLayoutError;

/// Describes the doubling table of a fractal heap independently of file placement and field widths.
///
/// Each row has the same number of child slots. Blocks in the first two rows have the starting
/// size, and each subsequent row has blocks twice the size of the preceding row. Rows whose
/// blocks exceed the maximum direct-block size refer to indirect blocks.
///
/// The doubling table is defined in "Fractal Heap" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapGeometry {
    table_width: usize,
    direct_rows: usize,
    max_root_rows: u16,
    first_row_bits: u32,
    starting_block_size: u64,
    max_direct_block_size: u64,
    max_heap_size: u16,
    row_block_sizes: Vec<u64>,
    row_offsets: Vec<u64>,
}

impl FractalHeapGeometry {
    /// Creates geometry from the heap's table width, block sizes, and heap-offset bit count.
    ///
    /// `table_width` is the number of slots per row. Both block sizes are in bytes, and
    /// `max_heap_size` is the header's "Maximum Heap Size" field, in bits.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if the table width or either block
    /// size is zero or is not a power of two, the maximum direct-block size is smaller than the
    /// starting size, or `max_heap_size` is outside `1..=64`. Also returns this error if the
    /// doubling-table rows are inconsistent with these parameters or a row calculation overflows.
    pub fn new(
        table_width: u16,
        starting_block_size: u64,
        max_direct_block_size: u64,
        max_heap_size: u16,
    ) -> Result<Self, FractalHeapLayoutError> {
        if table_width == 0
            || !table_width.is_power_of_two()
            || starting_block_size == 0
            || !starting_block_size.is_power_of_two()
            || max_direct_block_size < starting_block_size
            || !max_direct_block_size.is_power_of_two()
            || max_heap_size == 0
            || max_heap_size > 64
        {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }

        let start_bits = starting_block_size.trailing_zeros();
        let max_direct_bits = max_direct_block_size.trailing_zeros();
        let first_row_bits = start_bits
            .checked_add(u64::from(table_width).trailing_zeros())
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        let heap_bits = u32::from(max_heap_size);
        if heap_bits < first_row_bits {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let max_root_rows = u16::try_from(heap_bits - first_row_bits + 1)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
        let direct_rows = usize::try_from(max_direct_bits - start_bits + 2)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
        let table_width_u64 = u64::from(table_width);
        let table_width = usize::from(table_width);

        let max_root_rows_usize = usize::from(max_root_rows);
        let mut row_block_sizes = Vec::with_capacity(max_root_rows_usize);
        let mut row_offsets = Vec::with_capacity(max_root_rows_usize);
        let mut row_offset = 0u64;
        for row in 0..max_root_rows_usize {
            row_offsets.push(row_offset);
            let block_size = if row <= 1 {
                starting_block_size
            } else {
                let shift = u32::try_from(row - 1)
                    .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
                let multiplier = 1u64
                    .checked_shl(shift)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
                starting_block_size
                    .checked_mul(multiplier)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?
            };
            let row_span = block_size
                .checked_mul(table_width_u64)
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            row_block_sizes.push(block_size);
            if row + 1 < max_root_rows_usize {
                row_offset = row_offset
                    .checked_add(row_span)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            }
        }

        for (row, &block_size) in row_block_sizes
            .iter()
            .enumerate()
            .take(max_root_rows_usize)
            .skip(direct_rows)
        {
            let size_bits = block_size.trailing_zeros();
            if size_bits < first_row_bits {
                return Err(FractalHeapLayoutError::InvalidGeometry);
            }
            let child_rows = usize::try_from(size_bits - first_row_bits + 1)
                .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
            if child_rows == 0
                || child_rows > row
                || row_offsets.get(child_rows).copied() != Some(block_size)
            {
                return Err(FractalHeapLayoutError::InvalidGeometry);
            }
        }

        Ok(Self {
            table_width,
            direct_rows,
            max_root_rows,
            first_row_bits,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            row_block_sizes,
            row_offsets,
        })
    }

    /// Returns the size in bytes of blocks in the first two doubling-table rows.
    pub const fn starting_block_size(&self) -> u64 {
        self.starting_block_size
    }

    /// Returns the largest direct-block size permitted by the block-size limits.
    pub const fn max_direct_block_size(&self) -> u64 {
        self.max_direct_block_size
    }

    /// Returns the bit count of the managed heap address space.
    pub const fn max_heap_size(&self) -> u16 {
        self.max_heap_size
    }

    /// Returns the number of child slots in each row.
    pub const fn table_width(&self) -> usize {
        self.table_width
    }

    /// Returns the number of direct-block rows allowed by the block-size limits.
    pub const fn direct_rows(&self) -> usize {
        self.direct_rows
    }

    /// Returns the maximum number of rows in the root indirect block.
    pub const fn max_root_rows(&self) -> u16 {
        self.max_root_rows
    }

    /// Returns the size in bytes represented by one child slot in `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` is outside the doubling table.
    pub fn block_size_for_row(&self, row: usize) -> Result<u64, FractalHeapLayoutError> {
        self.row_block_sizes
            .get(row)
            .copied()
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the row count of an indirect block referred to by a slot in `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` refers to direct blocks or
    /// lies outside the doubling table.
    pub fn child_indirect_rows(&self, row: usize) -> Result<u16, FractalHeapLayoutError> {
        if row < self.direct_rows {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let size = self.block_size_for_row(row)?;
        let size_bits = size.trailing_zeros();
        if size_bits < self.first_row_bits {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        u16::try_from(size_bits - self.first_row_bits + 1)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the managed address-space size represented by `nrows` doubling-table rows.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `nrows` is zero, exceeds
    /// [`max_root_rows`](Self::max_root_rows), or represents a size greater than [`u64::MAX`].
    pub fn indirect_heap_size(&self, nrows: u16) -> Result<u64, FractalHeapLayoutError> {
        if nrows == 0 || nrows > self.max_root_rows {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let last_row = usize::from(nrows - 1);
        let start = self.row_offset(last_row)?;
        let span = self
            .block_size_for_row(last_row)?
            .checked_mul(
                u64::try_from(self.table_width)
                    .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?,
            )
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        start
            .checked_add(span)
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the first slot's offset in `row` within the indirect block's managed space.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` is outside the doubling table.
    pub fn row_offset(&self, row: usize) -> Result<u64, FractalHeapLayoutError> {
        self.row_offsets
            .get(row)
            .copied()
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns a child slot's offset within its indirect block's managed space.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` or `column` is outside the
    /// doubling table, or the slot offset exceeds [`u64::MAX`].
    pub fn slot_offset(&self, row: usize, column: usize) -> Result<u64, FractalHeapLayoutError> {
        if column >= self.table_width {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let row_offset = self.row_offset(row)?;
        let block_size = self.block_size_for_row(row)?;
        let column = u64::try_from(column)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
        row_offset
            .checked_add(
                column
                    .checked_mul(block_size)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?,
            )
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns a child slot's heap offset for an indirect block at heap offset `parent`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` or `column` is outside the
    /// doubling table, or the resulting heap offset exceeds [`u64::MAX`].
    pub fn slot_heap_offset(
        &self,
        parent: u64,
        row: usize,
        column: usize,
    ) -> Result<u64, FractalHeapLayoutError> {
        parent
            .checked_add(self.slot_offset(row, column)?)
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::group(512, 32, 9, 22)]
    #[case::attribute(1024, 40, 8, 29)]
    fn geometry_derives_group_and_attribute_rows(
        #[case] start: u64,
        #[case] bits: u16,
        #[case] direct: usize,
        #[case] root: u16,
    ) {
        let geometry = FractalHeapGeometry::new(4, start, 65_536, bits).unwrap();
        assert_eq!(geometry.table_width(), 4);
        assert_eq!(geometry.direct_rows(), direct);
        assert_eq!(geometry.max_root_rows(), root);
        assert_eq!(geometry.block_size_for_row(0), Ok(start));
        assert_eq!(geometry.block_size_for_row(1), Ok(start));
        assert_eq!(geometry.block_size_for_row(2), Ok(start * 2));
        assert_eq!(geometry.row_offset(0), Ok(0));
        assert_eq!(geometry.row_offset(1), Ok(start * 4));
        assert_eq!(geometry.row_offset(2), Ok(start * 8));
        assert_eq!(geometry.slot_heap_offset(123, 2, 3), Ok(123 + start * 14));
        assert_eq!(geometry.block_size_for_row(direct), Ok(131_072));
        assert_eq!(
            geometry.child_indirect_rows(direct),
            Ok(if start == 512 { 7 } else { 6 })
        );
        assert_eq!(geometry.indirect_heap_size(root), Ok(1u64 << bits));
    }

    #[rstest]
    #[case::zero_width(0, 512, 65_536, 32)]
    #[case::non_power_width(3, 512, 65_536, 32)]
    #[case::zero_start(4, 0, 65_536, 32)]
    #[case::non_power_start(4, 513, 65_536, 32)]
    #[case::zero_max_direct(4, 512, 0, 32)]
    #[case::small_max_direct(4, 512, 256, 32)]
    #[case::non_power_max_direct(4, 512, 65_537, 32)]
    #[case::zero_heap_bits(4, 512, 65_536, 0)]
    #[case::wide_heap_bits(4, 512, 65_536, 65)]
    #[case::short_heap_bits(4, 512, 65_536, 10)]
    #[case::row_span_overflow(4, 1 << 63, 1 << 63, 64)]
    fn invalid_geometry_is_rejected(
        #[case] width: u16,
        #[case] start: u64,
        #[case] max_direct: u64,
        #[case] bits: u16,
    ) {
        assert_eq!(
            FractalHeapGeometry::new(width, start, max_direct, bits),
            Err(FractalHeapLayoutError::InvalidGeometry),
        );
    }

    #[rstest]
    #[case::invalid_row(22, 0, 0)]
    #[case::invalid_column(0, 4, 0)]
    #[case::parent_overflow(1, 0, u64::MAX)]
    fn invalid_slots_and_parent_overflow_are_rejected(
        #[case] row: usize,
        #[case] column: usize,
        #[case] parent: u64,
    ) {
        let geometry = FractalHeapGeometry::new(4, 512, 65_536, 32).unwrap();
        assert_eq!(
            geometry.slot_heap_offset(parent, row, column),
            Err(FractalHeapLayoutError::InvalidGeometry)
        );
    }

    #[test]
    fn a_64_bit_heap_preserves_representable_offsets_and_rejects_its_overflowing_span() {
        let geometry = FractalHeapGeometry::new(4, 512, 65_536, 64).unwrap();
        let last = usize::from(geometry.max_root_rows() - 1);
        assert_eq!(geometry.row_offset(last), Ok(1u64 << 63));
        assert_eq!(geometry.slot_offset(last, 3), Ok(0xE000_0000_0000_0000));
        assert_eq!(
            geometry.indirect_heap_size(geometry.max_root_rows()),
            Err(FractalHeapLayoutError::InvalidGeometry)
        );
    }

    #[rstest]
    #[case::zero(0)]
    #[case::past_root(23)]
    fn invalid_indirect_row_counts_are_rejected(#[case] rows: u16) {
        let geometry = FractalHeapGeometry::new(4, 512, 65_536, 32).unwrap();
        assert_eq!(
            geometry.indirect_heap_size(rows),
            Err(FractalHeapLayoutError::InvalidGeometry)
        );
    }
}
