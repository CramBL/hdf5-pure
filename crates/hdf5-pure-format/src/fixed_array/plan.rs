use super::FixedArrayPlacement;
use super::geometry;
use super::geometry::Geometry;
use super::header;
use crate::address::StoredAddress;
use crate::bytes;
use crate::chunk_record;
use crate::chunk_record::ChunkRecord;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
use crate::width::FormatWidths;
use crate::width::LengthWidth;

/// The checked lengths and element encoding of a Fixed Array to serialize.
///
/// Planning uses `u64` lengths and does not allocate output buffers. The header and data block
/// can be placed independently with [`place`](Self::place).
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_format::__private::ChunkRecord;
/// use hdf5_pure_format::__private::FixedArrayPlan;
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::IndexSlots;
///
/// let chunks = [ChunkRecord {
///     address: StoredAddress::new(4_096),
///     stored_size: 128,
///     filter_mask: 0,
/// }];
/// let plan = FixedArrayPlan::new(1, 128, FormatWidths::from_sizes(8, 8)?, false)?;
/// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [28, 26]);
///
/// let placement = plan.place(StoredAddress::new(1_024), StoredAddress::new(512))?;
/// let [(header_at, header), (data_at, data)] =
///     placement.serialize_blocks(&IndexSlots::dense(&chunks))?;
/// assert_eq!((header_at.get(), header.len()), (1_024, 28));
/// assert_eq!((data_at.get(), data.len()), (512, 26));
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FixedArrayPlan {
    lengths: LengthWidth,
    geometry: Geometry,
    data_block_length: u64,
    total_length: u64,
}

impl FixedArrayPlan {
    /// Plans an index of `entries` slots, including unallocated slots.
    ///
    /// `chunk_bytes` is the unfiltered chunk size. For filtered chunks, the writer derives the
    /// stored size field width from this size. The data block is paged past 1,024 slots.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if `entries` does not fit `widths.lengths`, or
    /// [`FormatError::InvalidFixedArrayGeometry`] if a block length or their sum exceeds
    /// [`u64::MAX`].
    pub fn new(
        entries: u64,
        chunk_bytes: u64,
        widths: FormatWidths,
        has_filters: bool,
    ) -> Result<Self, FormatError> {
        let encoding =
            chunk_record::chunk_element_encoding(chunk_bytes, widths.offsets, has_filters);
        let entry_size = encoding
            .elem_size
            .narrow_or_else(|| FormatError::InvalidFixedArrayGeometry)?;
        let elements = geometry::element_encoding(widths.offsets, encoding.client_id, entry_size)?;
        let geometry = Geometry::new(elements, entries, FIXED_ARRAY_PAGE_BITS)?;
        Self::from_geometry(widths.lengths, geometry)
    }

    /// Returns complete allocation lengths in header, then data block order.
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> {
        self.block_sizes().into_iter()
    }

    /// Returns the sum of the block lengths, excluding any gaps between their placements.
    pub fn total_len(&self) -> u64 {
        self.total_length
    }

    /// Checks disjoint complete block ranges at the supplied stored addresses.
    ///
    /// The header may precede or follow the data block. Both starts must be defined addresses,
    /// and both exclusive ends must fit the address width. An end may equal the undefined
    /// address value. The caller establishes ownership of the ranges.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if a start is the undefined address,
    /// [`FormatError::AddressTooLarge`] if a start or end exceeds the address width,
    /// [`FormatError::OffsetOverflow`] if an end exceeds [`u64::MAX`], or
    /// [`FormatError::FixedArrayBlockOverlap`] if the ranges overlap.
    pub fn place(
        self,
        header: StoredAddress,
        data_block: StoredAddress,
    ) -> Result<FixedArrayPlacement, FormatError> {
        FixedArrayPlacement::new(self, header, data_block)
    }

    /// Selects the page threshold as a base 2 exponent and recomputes complete block lengths.
    ///
    /// Paging starts when the entry count exceeds `2^page_bits`. See [`Self::encode_record`] for
    /// an example combining custom paging with independent field widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if `page_bits` is 64 or more, or a block
    /// length or their sum overflows.
    pub fn with_page_bits(self, page_bits: u8) -> Result<Self, FormatError> {
        Self::from_geometry(
            self.lengths,
            Geometry::new(self.geometry.encoding(), self.geometry.entries(), page_bits)?,
        )
    }

    /// Encodes one entry at the plan's retained address and filtered-size widths.
    ///
    /// Writes `None` as an undefined address with zero filtered fields. Returns the encoded byte
    /// count, preserving trailing bytes and all output on error. Filtered sizes use the entry
    /// encoding, independently of the file's length width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is too short. Returns the address and
    /// size errors of [`super::FixedArrayEntryWindow::encode_records`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let plan = format::FixedArrayPlan::new(
    ///     9, 16, format::FormatWidths::from_sizes(2, 4)?, true,
    /// )?.with_page_bits(3)?;
    /// assert_eq!(plan.page_bits(), 3);
    /// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [18, 93]);
    /// let mut output = [0xcc; 10];
    /// let length = plan.encode_record(&mut output, Some(format::ChunkRecord {
    ///     address: StoredAddress::new(0x1234), stored_size: 0x56, filter_mask: 3,
    /// }))?;
    /// assert_eq!(length, 8);
    /// assert_eq!(output, [0x34, 0x12, 0x56, 0, 3, 0, 0, 0, 0xcc, 0xcc]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn encode_record(
        &self,
        output: &mut [u8],
        record: Option<ChunkRecord>,
    ) -> Result<usize, FormatError> {
        let element = self.geometry.encoding().encode(record)?;
        let bytes = element.as_bytes();
        bytes::ensure_len(output, 0, bytes.len())?;
        output[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }

    /// Returns the base 2 logarithm of the maximum entries per page.
    pub const fn page_bits(&self) -> u8 {
        self.geometry.page_bits()
    }

    /// Plans storage for the entry encoding, entry count, and page size in `geometry`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if the entry count does not fit `lengths`, or
    /// [`FormatError::InvalidFixedArrayGeometry`] if a block length or their sum overflows.
    pub(super) fn from_geometry(
        lengths: LengthWidth,
        geometry: Geometry,
    ) -> Result<Self, FormatError> {
        let entries = geometry.entries();
        if !width::length_width_holds(lengths, entries) {
            return Err(FormatError::LengthTooLarge {
                length: entries,
                length_size: lengths.get(),
            });
        }
        let data_block_length = geometry.data_block_len()?;
        let widths = FormatWidths {
            offsets: geometry.encoding().offsets(),
            lengths,
        };
        let total_length = geometry::sum([header::header_len(widths), data_block_length])?;
        Ok(Self {
            lengths,
            geometry,
            data_block_length,
            total_length,
        })
    }

    pub(super) fn block_sizes(&self) -> [u64; 2] {
        [header::header_len(self.widths()), self.data_block_length]
    }

    pub(super) fn widths(&self) -> FormatWidths {
        FormatWidths {
            offsets: self.geometry.encoding().offsets(),
            lengths: self.lengths,
        }
    }

    pub(super) fn geometry(&self) -> Geometry {
        self.geometry
    }
}

/// The default Fixed Array page size exponent, for 1,024 entries per page.
///
/// The version 4 data layout writer and [`FixedArrayPlan::new`] use this default.
/// libhdf5 calls it `H5D_FARRAY_MAX_DBLK_PAGE_NELMTS_BITS`.
pub(crate) const FIXED_ARRAY_PAGE_BITS: u8 = 10;
