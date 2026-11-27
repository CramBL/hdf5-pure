use core::ops::Range;

use super::geometry::{DataBlock, Layout, SuperBlock};
use super::header::HEADER_PREFIX_LEN;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::{FormatWidths, LengthWidth};

/// The six statistics of an Extensible Array header, in the order the header stores them.
///
/// [`ExtensibleArrayHeader::stored_statistics`](super::ExtensibleArrayHeader::stored_statistics)
/// reports the values read from a header.
/// [`extensible_array_layout`](super::extensible_array_layout) derives statistics from the occupied
/// slots and creation geometry.
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
    pub(super) fn decode_fields(
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
    pub(super) fn write_fields(&self, output: &mut [u8], lengths: LengthWidth) {
        let width = usize::from(lengths.get());
        for (field, value) in output.chunks_exact_mut(width).zip(self.field_values()) {
            field.copy_from_slice(&value.to_le_bytes()[..width]);
        }
    }

    pub(super) fn fields_len(lengths: LengthWidth) -> usize {
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

// The six statistics of a header, each a length, of which "Max Index Set" is the fifth.
pub(super) const STATISTICS: u8 = 6;

impl ExtensibleArrayStats {
    /// Starts the allocation counts with the index block's inline slots and no allocated blocks.
    ///
    /// The recorded maximum index is the caller's `num_slots`.
    pub(super) fn empty(layout: Layout, num_slots: u64) -> Self {
        Self {
            nsuper_blks: 0,
            super_blk_size: 0,
            ndata_blks: 0,
            data_blk_size: 0,
            max_idx_set: num_slots,
            nelmts: u64::from(layout.inline_elements()),
        }
    }

    /// Adds a complete data block's counts and returns its checked length and logical slots.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the block offset exceeds its
    /// field width or the length, slots, or statistics overflow [`u64`].
    pub(super) fn data(
        &mut self,
        layout: Layout,
        block: DataBlock,
    ) -> Result<(u64, Range<u64>), FormatError> {
        let offset_len = usize::from(layout.block_offset_len());
        if block.offset().to_le_bytes()[offset_len..]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        let length = layout.data_block_len(block)?;
        let slots = block.slots(layout)?;
        self.ndata_blks = super::geometry::sum([self.ndata_blks, 1])?;
        self.data_blk_size = super::geometry::sum([self.data_blk_size, length])?;
        self.nelmts = super::geometry::sum([self.nelmts, block.elements()])?;
        Ok((length, slots))
    }

    /// Adds a secondary block's counts and returns its complete length and logical slots.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if its length, logical range, or
    /// accumulated statistics overflow [`u64`].
    pub(super) fn secondary(
        &mut self,
        layout: Layout,
        block: SuperBlock,
    ) -> Result<(u64, Range<u64>), FormatError> {
        let length = layout.super_block_len(block)?;
        let slots = block.slots(layout)?;
        self.nsuper_blks = super::geometry::sum([self.nsuper_blks, 1])?;
        self.super_blk_size = super::geometry::sum([self.super_blk_size, length])?;
        Ok((length, slots))
    }

    /// Checks that every header statistic fits the encoded length width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if a statistic exceeds `widths.lengths`.
    pub(super) fn with_widths(self, widths: FormatWidths) -> Result<Self, FormatError> {
        self.require_field_fits(widths.lengths)?;
        Ok(self)
    }

    /// Returns the combined byte length of the header, index block, and allocated array blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the length exceeds [`u64::MAX`].
    pub(super) fn total_len(
        self,
        layout: Layout,
        widths: FormatWidths,
    ) -> Result<u64, FormatError> {
        super::geometry::sum([
            super::header::header_len(widths),
            layout.index_block_len()?,
            self.super_blk_size,
            self.data_blk_size,
        ])
    }
}

#[cfg(test)]
#[path = "tests/statistics.rs"]
mod tests;
