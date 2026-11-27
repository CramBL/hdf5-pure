//! The Fixed Array chunk index, index type 3 of a version 4 data layout message.
//!
//! A Fixed Array indexes the chunks of a dataset whose maximum shape is fixed. Its header
//! (`FAHD`) points at one data block (`FADB`), which holds an element per slot. Past `2^page_bits`
//! slots the data block holds a page-init bitmap and then the elements in pages, each with its
//! own checksum.
//!
//! A caller reads the bytes of each range the parser returns and passes them back to it.
//! [`FixedArrayHeaderLocator`] returns the range of the header, [`FixedArrayDataBlockRef`] the
//! range of an unpaged block or of a paged block's prefix, page-init bitmap, and checksum.
//! [`FixedArrayPageRef`] gives the range of one page. [`FixedArrayUnpaged`] describes entry
//! windows and checksum ranges for an unpaged block.
//!
//! A writer plans the complete header and data block with [`FixedArrayPlan`] and supplies
//! their stored addresses to [`FixedArrayPlan::place`]. The data block includes all its pages.
//! The caller establishes allocation ownership and placement policy.
//!
//! The index is defined in "The Fixed Array Index" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr

use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::AddressRange;

use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::chunk_record;
use crate::chunk_record::ArrayClient;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::IndexSlots;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;
use data_block::DATA_BLOCK_SIGNATURE;
use framing::VERSION;
use geometry::Geometry;
use header::HEADER_SIGNATURE;

mod data_block;
mod framing;
mod geometry;
mod header;
mod mutation;
mod page;

pub use data_block::FixedArrayDataBlock;
pub use data_block::FixedArrayDataBlockRef;
pub use data_block::FixedArrayElements;
pub use data_block::FixedArrayEntryWindow;
pub use data_block::FixedArrayUnpaged;
pub use header::FixedArrayHeader;
pub use header::FixedArrayHeaderLocator;
pub use page::FixedArrayPageBitmap;
pub use page::FixedArrayPageRef;

/// The base 2 logarithm of the number of elements in a data block page of a Fixed Array the
/// writer builds, 1024 elements.
///
/// The writer stores it in the "Page Bits" field of both the header and the version 4 data layout
/// message. libhdf5 calls this value `H5D_FARRAY_MAX_DBLK_PAGE_NELMTS_BITS`. A reader takes
/// the page size from the header.
pub(crate) const FIXED_ARRAY_PAGE_BITS: u8 = 10;

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
    widths: FormatWidths,
    geometry: Geometry,
    encoding: ChunkElementEncoding,
    entry_size: u8,
    page_bits: u8,
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
        if !width::length_width_holds(widths.lengths, entries) {
            return Err(FormatError::LengthTooLarge {
                length: entries,
                length_size: widths.lengths.get(),
            });
        }
        let encoding =
            chunk_record::chunk_element_encoding(chunk_bytes, widths.offsets, has_filters);
        let entry_size = encoding
            .elem_size
            .narrow_or_else(|| FormatError::InvalidFixedArrayGeometry)?;
        let elements = geometry::element_encoding(widths.offsets, encoding.client_id, entry_size)?;
        let geometry = Geometry::new(elements, entries, FIXED_ARRAY_PAGE_BITS)?;
        let data_block_length = geometry.data_block_len()?;
        let total_length = geometry::sum([header::header_len(widths), data_block_length])?;
        Ok(Self {
            widths,
            geometry,
            encoding,
            entry_size,
            page_bits: FIXED_ARRAY_PAGE_BITS,
            data_block_length,
            total_length,
        })
    }

    /// Returns complete allocation lengths in header, then data block order.
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> {
        [header::header_len(self.widths), self.data_block_length].into_iter()
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
        let header = self.range(header, header::header_len(self.widths))?;
        let data_block = self.range(data_block, self.data_block_length)?;
        if header.overlaps(data_block) {
            return Err(FormatError::FixedArrayBlockOverlap {
                header: header.start().get(),
                data_block: data_block.start().get(),
            });
        }
        Ok(FixedArrayPlacement {
            plan: self,
            header,
            data_block,
        })
    }

    fn range(
        &self,
        address: StoredAddress,
        length: u64,
    ) -> Result<AddressRange<StoredAddress>, FormatError> {
        let maximum = StoredAddress::undefined(self.widths.offsets).get();
        if address.get() == maximum {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: self.widths.offsets.get(),
            });
        }
        if address.get() > maximum {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: self.widths.offsets.get(),
            });
        }
        let range = AddressRange::new(address, length)?;
        if range.end().get() > maximum {
            return Err(FormatError::AddressTooLarge {
                address: range.end().get(),
                offset_size: self.widths.offsets.get(),
            });
        }
        Ok(range)
    }
}

/// Disjoint complete Fixed Array block ranges whose stored addresses fit the encoding width.
///
/// A paged data block occupies one range, including its prefix, bitmap, pages, and checksums.
/// Allocation ownership and placement policy are established by the caller.
#[derive(Clone, Copy, Debug)]
pub struct FixedArrayPlacement {
    plan: FixedArrayPlan,
    header: AddressRange<StoredAddress>,
    data_block: AddressRange<StoredAddress>,
}

impl FixedArrayPlacement {
    /// Returns each block's stored address and complete length, header first.
    ///
    /// The order is independent of the blocks' address order.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> {
        let Self {
            plan: _,
            header,
            data_block,
        } = *self;
        [header, data_block]
            .into_iter()
            .map(|range| (range.start(), range.length()))
    }

    /// Serializes the header and data block separately, with their reciprocal stored references.
    ///
    /// The returned buffers follow header, then data block order. Unallocated slots contain the
    /// undefined address. All pages are initialized, and the last page holds only its remaining
    /// entries. The writer computes all checksums regardless of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FixedArrayEntryCount`] if the supplied slot count differs from
    /// the planned count, [`FormatError::DefinedAddressIsUndefined`] if a chunk address is
    /// undefined, [`FormatError::AddressTooLarge`] if a chunk address exceeds the address width,
    /// or [`FormatError::Internal`] if a filtered chunk's stored size exceeds its field width.
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if an output length does not fit
    /// [`usize`], or [`FormatError::AllocationFailed`] if an output buffer reservation fails.
    pub fn serialize_blocks(
        &self,
        slots: &IndexSlots<'_>,
    ) -> Result<[(StoredAddress, Vec<u8>); 2], FormatError> {
        let Self {
            plan,
            header,
            data_block,
        } = *self;
        let FixedArrayPlan {
            widths: _,
            geometry,
            encoding: _,
            entry_size: _,
            page_bits: _,
            data_block_length: _,
            total_length: _,
        } = plan;
        let actual = slots.len().to_u64();
        if actual != geometry.entries() {
            return Err(FormatError::FixedArrayEntryCount {
                expected: geometry.entries(),
                actual,
            });
        }
        Ok([
            (header.start(), self.serialize_header()?),
            (data_block.start(), self.serialize_data_block(slots)?),
        ])
    }

    /// Serializes the version 0 header with the placed data-block address and its checksum.
    ///
    /// The caller obtains both blocks from [`Self::serialize_blocks`] or emits an empty data
    /// block with [`Self::emit_empty_data_block`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the header length exceeds
    /// [`usize::MAX`], or [`FormatError::AllocationFailed`] if its buffer reservation fails.
    /// Placement and planning check the encoded address and entry count.
    pub fn serialize_header(&self) -> Result<Vec<u8>, FormatError> {
        let Self {
            plan,
            header,
            data_block,
        } = *self;
        let FixedArrayPlan {
            widths,
            geometry,
            encoding,
            entry_size,
            page_bits,
            data_block_length: _,
            total_length: _,
        } = plan;
        let mut bytes = output_buffer(header.length())?;
        bytes.extend_from_slice(&HEADER_SIGNATURE);
        bytes.push(VERSION);
        bytes.push(encoding.client_id);
        bytes.push(entry_size);
        bytes.push(page_bits);
        bytes::try_write_length(&mut bytes, geometry.entries(), widths.lengths)?;
        bytes::try_write_decoded_address(
            &mut bytes,
            DecodedStoredAddress::Defined(data_block.start()),
            widths.offsets,
        )?;
        let checksum = checksum::jenkins_lookup3(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(bytes.len().to_u64(), header.length());
        Ok(bytes)
    }

    fn serialize_data_block(&self, slots: &IndexSlots<'_>) -> Result<Vec<u8>, FormatError> {
        let Self {
            plan,
            header,
            data_block,
        } = *self;
        let FixedArrayPlan {
            widths,
            geometry,
            encoding,
            entry_size: _,
            page_bits: _,
            data_block_length: _,
            total_length: _,
        } = plan;
        let mut bytes = output_buffer(data_block.length())?;
        bytes.extend_from_slice(&DATA_BLOCK_SIGNATURE);
        bytes.push(VERSION);
        bytes.push(encoding.client_id);
        bytes::try_write_decoded_address(
            &mut bytes,
            DecodedStoredAddress::Defined(header.start()),
            widths.offsets,
        )?;
        match geometry.paging() {
            None => {
                for slot in 0..slots.len() {
                    self.write_element(&mut bytes, slots.at(slot))?;
                }
                let checksum = checksum::jenkins_lookup3(&bytes);
                bytes.extend_from_slice(&checksum.to_le_bytes());
            }
            Some(paging) => {
                let pages = paging.pages().to_usize()?;
                let mut bitmap = vec![0; paging.pages().div_ceil(8).to_usize()?];
                for page in 0..pages {
                    // Page 0 occupies the most significant bit. H5VM_bit_set uses this order
                    // in H5VMprivate.h, HDF5 2.2.0.
                    bitmap[page / 8] |= 1 << (7 - page % 8);
                }
                bytes.extend_from_slice(&bitmap);
                let checksum = checksum::jenkins_lookup3(&bytes);
                bytes.extend_from_slice(&checksum.to_le_bytes());
                let page_size = paging.page_entries().to_usize()?;
                for page in 0..pages {
                    let start = page * page_size;
                    let end = start.saturating_add(page_size).min(slots.len());
                    let mut page_bytes = Vec::with_capacity((end - start) * encoding.elem_size);
                    for slot in start..end {
                        self.write_element(&mut page_bytes, slots.at(slot))?;
                    }
                    let checksum = checksum::jenkins_lookup3(&page_bytes);
                    page_bytes.extend_from_slice(&checksum.to_le_bytes());
                    bytes.extend_from_slice(&page_bytes);
                }
            }
        }
        debug_assert_eq!(bytes.len().to_u64(), data_block.length());
        Ok(bytes)
    }

    fn write_element(
        &self,
        bytes: &mut Vec<u8>,
        chunk: Option<&ChunkRecord>,
    ) -> Result<(), FormatError> {
        let Self {
            plan,
            header: _,
            data_block: _,
        } = *self;
        let FixedArrayPlan {
            widths,
            geometry,
            encoding,
            entry_size: _,
            page_bits: _,
            data_block_length: _,
            total_length: _,
        } = plan;
        let filtered = matches!(
            geometry.encoding().client(),
            ArrayClient::FilteredChunks { .. }
        );
        match chunk {
            Some(chunk) => {
                plan.range(chunk.address, 0)?;
                chunk_record::write_chunk_element(
                    bytes,
                    chunk,
                    widths.offsets,
                    filtered,
                    encoding.chunk_size_bytes,
                )
            }
            None => {
                chunk_record::write_undefined_element(
                    bytes,
                    widths.offsets,
                    filtered,
                    encoding.chunk_size_bytes,
                );
                Ok(())
            }
        }
    }
}

/// Reserves capacity for `length` bytes in an empty buffer.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if `length` does not fit [`usize`], or
/// [`FormatError::AllocationFailed`] if the capacity reservation fails.
fn output_buffer(length: u64) -> Result<Vec<u8>, FormatError> {
    let values = length.to_usize()?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(values)
        .map_err(|source| FormatError::AllocationFailed { values, source })?;
    Ok(bytes)
}

/// Returns the length in bytes of the Fixed Array [`build_fixed_array_at`] builds for `slots`,
/// without building it.
///
/// # Errors
///
/// Returns the geometry and length errors of [`FixedArrayPlan::new`].
pub fn fixed_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> Result<u64, FormatError> {
    Ok(FixedArrayPlan::new(
        slots.len().to_u64(),
        chunk_bytes,
        FormatWidths {
            offsets: offset_size,
            lengths: length_size,
        },
        has_filters,
    )?
    .total_len())
}

/// Builds the Fixed Array for `slots` at `fa_address`: the header, then the data block.
///
/// `chunk_bytes` is the size of a chunk before filtering, which sets the width of the chunk size
/// field of a filtered element. A slot no chunk occupies stores the undefined address. Past
/// 1,024 slots the data block is paged, and every page is marked
/// initialized.
///
/// # Errors
///
/// Returns the planning, placement, and serialization errors of [`FixedArrayPlan::new`],
/// [`FixedArrayPlan::place`], and [`FixedArrayPlacement::serialize_blocks`]. Also returns
/// [`FormatError::ValueTooLargeForPlatform`] if the combined length does not fit [`usize`], or
/// [`FormatError::AllocationFailed`] if the combined buffer reservation fails.
pub fn build_fixed_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    fa_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    let plan = FixedArrayPlan::new(
        slots.len().to_u64(),
        chunk_bytes,
        FormatWidths {
            offsets: offset_size,
            lengths: length_size,
        },
        has_filters,
    )?;
    fa_address.checked_add(plan.total_len())?;
    let data_block = fa_address.checked_add(header::header_len(plan.widths))?;
    let [(header, mut bytes), (_, data)] = plan
        .place(fa_address, data_block)?
        .serialize_blocks(slots)?;
    debug_assert_eq!(header, fa_address);
    let values = plan.total_len().to_usize()?;
    bytes
        .try_reserve_exact(data.len())
        .map_err(|source| FormatError::AllocationFailed { values, source })?;
    bytes.extend_from_slice(&data);
    debug_assert_eq!(bytes.len().to_u64(), plan.total_len());
    Ok(bytes)
}

// The signatures, the version, and the field widths are defined in "The Fixed Array Index" of the
// format specification, version 4.0.
#[cfg(test)]
mod tests;
