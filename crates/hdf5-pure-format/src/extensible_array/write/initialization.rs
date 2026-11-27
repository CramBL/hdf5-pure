use alloc::vec::Vec;

use super::occupancy::SlotOccupancy;
use super::plan::ExtensibleArrayPlan;
use crate::address::StoredAddress;
use crate::chunk_record::{ChunkElementEncoding, IndexSlots};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::extensible_array::{ExtensibleArrayCreationParameters, ExtensibleArrayGeometry};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

/// A retained plan for serializing an empty Extensible Array header and index block.
///
/// The serialized header is followed immediately by its index block. Both structures include
/// their checksums. The index block's chunk and child block addresses are undefined. The version 0
/// header and index block are defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_format::__private::{
///     ExtensibleArrayCreationParameters, ExtensibleArrayInitialization, FormatWidths,
/// };
///
/// let creation = ExtensibleArrayCreationParameters::parse(&[24, 8, 2, 8, 8])?;
/// let initialization = ExtensibleArrayInitialization::new(
///     creation,
///     16,
///     FormatWidths::from_sizes(8, 8)?,
///     false,
/// )?;
/// assert_eq!(initialization.plan().blocks().len(), 2);
/// let bytes = initialization.encode_at(StoredAddress::new(1024))?;
/// assert_eq!(bytes.len() as u64, initialization.allocation_len());
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub struct ExtensibleArrayInitialization {
    plan: ExtensibleArrayPlan,
}

impl ExtensibleArrayInitialization {
    /// Uses version 4 chunk-record encoding for an empty index.
    ///
    /// `chunk_bytes` is the chunk size before filtering. For filtered chunks, the chunk-size
    /// field width is derived from that size.
    ///
    /// # Errors
    ///
    /// Returns the geometry and width errors of [`ExtensibleArrayPlan::new`].
    pub fn new(
        creation: ExtensibleArrayCreationParameters,
        chunk_bytes: u64,
        widths: FormatWidths,
        has_filters: bool,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            plan: ExtensibleArrayPlan::new(
                creation,
                SlotOccupancy::Dense(0),
                0,
                chunk_bytes,
                widths,
                has_filters,
            )?,
        })
    }

    /// Returns the block geometry defined by the retained creation parameters.
    ///
    /// Includes the geometry of unallocated blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset
    /// exceeds [`u64::MAX`].
    pub fn geometry(&self) -> Result<ExtensibleArrayGeometry, FormatError> {
        self.plan.geometry()
    }

    /// Returns the canonical chunk-record encoding used to serialize the empty index.
    pub fn element_encoding(&self) -> ChunkElementEncoding {
        self.plan.element_encoding()
    }

    /// Returns the combined size of the header and index block, including their checksums.
    pub fn allocation_len(&self) -> u64 {
        self.plan.total_len()
    }

    /// Serializes the header at the base-relative `address`, followed by its index block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the allocation's end exceeds
    /// [`u64::MAX`]. After this check succeeds, returns [`FormatError::DefinedAddressIsUndefined`]
    /// if `address` is undefined, or [`FormatError::AddressTooLarge`] if its start or end exceeds
    /// the address width. Returns
    /// [`FormatError::ValueTooLargeForPlatform`] if the allocation length does not fit [`usize`],
    /// [`FormatError::AllocationFailed`] if a scratch or output reservation fails, or
    /// [`FormatError::Internal`] if the encoding differs from the planned length.
    pub fn encode_at(&self, address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        address
            .get()
            .checked_add(self.allocation_len())
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        self.plan
            .encode_contiguous(&IndexSlots::dense(&[]), address)
    }

    /// Returns the retained plan for the empty header and index block.
    pub fn plan(&self) -> &ExtensibleArrayPlan {
        &self.plan
    }
}

/// Builds a contiguous Extensible Array for `slots` at `ea_address`.
///
/// The writer uses the canonical creation parameters and always allocates the header and index
/// block. It allocates a data block or super block only if its slot range contains a chunk.
/// For an unallocated block, it writes an undefined address. For a paged data block, it marks
/// each page containing a chunk as initialized in the super block.
///
/// # Errors
///
/// Returns the planning errors of [`ExtensibleArrayPlan::new`] and record errors of
/// [`ExtensibleArrayPlacement::encoding`](super::placement::ExtensibleArrayPlacement::encoding).
/// Returns [`FormatError::DefinedAddressIsUndefined`] if `ea_address` is undefined,
/// [`FormatError::AddressTooLarge`] if the allocation's start or end exceeds the address width,
/// [`FormatError::OffsetOverflow`] if its end exceeds [`u64::MAX`], or
/// [`FormatError::ValueTooLargeForPlatform`] if the output length does not fit [`usize`]. Returns
/// [`FormatError::AllocationFailed`] if a scratch or output reservation fails, or
/// [`FormatError::Internal`] if the encoding differs from the planned length.
pub fn build_extensible_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    ea_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    ExtensibleArrayPlan::new(
        ExtensibleArrayCreationParameters::canonical(),
        SlotOccupancy::Sparse(slots),
        slots.len().to_u64(),
        chunk_bytes,
        FormatWidths {
            offsets: offset_size,
            lengths: length_size,
        },
        has_filters,
    )?
    .encode_contiguous(slots, ea_address)
}
