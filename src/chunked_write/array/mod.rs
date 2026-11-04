use alloc::vec::Vec;
use core::convert::Infallible;

use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayPlan;
use hdf5_pure_format::__private::FixedArrayPlan;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::SlotOccupancy;

use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::file_emission::EmissionError;
use crate::width::FormatWidths;

use super::ChunkArrayGeometry;
use super::ChunkArrayKind;
use super::EmptyChunkIndex;
use super::INDEX_LENGTH_SIZE;
use super::INDEX_OFFSET_SIZE;

/// Address-independent block geometry for an allocated chunk array.
///
/// Empty arrays have scalar geometry. Format plans for occupied arrays include individual block
/// lengths.
pub(crate) enum ArrayMeasure {
    Empty(EmptyChunkIndex),
    Extensible(ExtensibleArrayPlan),
    Fixed(FixedArrayPlan),
}

impl ArrayMeasure {
    /// Plans array blocks, retaining scalar geometry for an empty array.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`ExtensibleArrayPlan::new`] and [`FixedArrayPlan::new`].
    pub(crate) fn new(geometry: ChunkArrayGeometry<'_>) -> Result<Self, FormatError> {
        let ChunkArrayGeometry {
            kind,
            occupied,
            slots,
            chunk_bytes,
            filtered,
        } = geometry;
        if occupied.is_empty() {
            return Ok(Self::Empty(EmptyChunkIndex::new(
                kind,
                chunk_bytes,
                slots,
                filtered,
            )));
        }
        Self::plan(ChunkArrayGeometry {
            kind,
            occupied,
            slots,
            chunk_bytes,
            filtered,
        })
    }

    /// Calls `visit` with each complete block's byte length in serialization order.
    ///
    /// The writer passes one length for each address required by [`Self::write_at`].
    ///
    /// # Errors
    ///
    /// Returns the first error from `visit`. Propagates array planning errors for an empty array.
    pub(crate) fn visit_blocks(
        &self,
        visit: impl FnMut(u64) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        match self {
            Self::Empty(index) => Self::plan(index.geometry())?.visit_blocks(visit),
            Self::Extensible(plan) => plan.block_lengths().try_for_each(visit),
            Self::Fixed(plan) => plan.block_lengths().try_for_each(visit),
        }
    }

    /// Returns the array representation described by this measurement.
    pub(crate) fn kind(&self) -> ChunkArrayKind {
        match self {
            Self::Empty(index) => index.geometry().kind,
            Self::Extensible(_) => ChunkArrayKind::ExtensibleArray,
            Self::Fixed(_) => ChunkArrayKind::FixedArray,
        }
    }

    /// Serializes complete array blocks at consecutive stored addresses.
    ///
    /// Calls `emit` in [`Self::visit_blocks`] order, stopping at its first error. A failure can
    /// leave partial output. Uses the block buffering described by [`Self::write_at`].
    ///
    /// # Errors
    ///
    /// Returns callback errors as [`EmissionError::Sink`]. Propagates allocation, address,
    /// planning, slot, and encoding errors as [`EmissionError::Format`].
    pub(crate) fn write_consecutive<E>(
        &self,
        slots: &IndexSlots<'_>,
        address: StoredAddress,
        emit: impl FnMut(StoredAddress, &[u8]) -> Result<(), E>,
    ) -> Result<(), EmissionError<E>> {
        if let Self::Empty(index) = self {
            return Self::plan(index.geometry())?.write_consecutive(slots, address, emit);
        }
        self.write_at(slots, &self.addresses_at(address)?, emit)
    }

    /// Serializes each complete block at its assigned stored address.
    ///
    /// The caller supplies `addresses` in [`Self::visit_blocks`] order and `slots` with the
    /// planned slot span and occupied slots. Calls `emit` in that order, stopping at the first
    /// error. A failure can leave partial output. The writer reuses one block buffer for an
    /// extensible array and buffers both fixed-array blocks.
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Sink`] if `emit` fails. Returns [`EmissionError::Format`] with
    /// [`FormatError::Internal`] if a fixed array has other than two addresses. Propagates
    /// array planning, placement, slot, and encoding errors as [`EmissionError::Format`].
    pub(crate) fn write_at<E>(
        &self,
        slots: &IndexSlots<'_>,
        addresses: &[StoredAddress],
        mut emit: impl FnMut(StoredAddress, &[u8]) -> Result<(), E>,
    ) -> Result<(), EmissionError<E>> {
        match self {
            Self::Empty(index) => Self::plan(index.geometry())?.write_at(slots, addresses, emit),
            Self::Extensible(plan) => {
                let placement = plan.place(addresses)?;
                let encoding = placement.encoding(slots)?;
                let mut bytes = Vec::new();
                for block in encoding.blocks() {
                    block.encode_into(&mut bytes)?;
                    emit(block.address(), &bytes).map_err(EmissionError::Sink)?;
                }
                Ok(())
            }
            Self::Fixed(plan) => {
                let [header, data] = addresses else {
                    return Err(FormatError::Internal(
                        "a fixed array requires two complete block placements".into(),
                    )
                    .into());
                };
                for (address, bytes) in plan.place(*header, *data)?.serialize_blocks(slots)? {
                    emit(address, &bytes).map_err(EmissionError::Sink)?;
                }
                Ok(())
            }
        }
    }

    /// Serializes the array into one buffer with consecutive block addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the exclusive end exceeds [`u64::MAX`],
    /// [`FormatError::ValueTooLargeForPlatform`] if the length exceeds [`usize::MAX`],
    /// [`FormatError::AllocationFailed`] if an address or output reservation fails, or
    /// [`FormatError::Internal`] if the encoded length differs from the planned length.
    /// Propagates array planning errors and the format errors of [`Self::write_at`].
    pub(crate) fn build_at(
        &self,
        slots: &IndexSlots<'_>,
        address: StoredAddress,
    ) -> Result<Vec<u8>, FormatError> {
        if let Self::Empty(index) = self {
            return Self::plan(index.geometry())?.build_at(slots, address);
        }
        let addresses = self.addresses_at(address)?;
        let length = self.length()?.to_usize()?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|source| FormatError::AllocationFailed {
                values: length,
                source,
            })?;
        self.write_at(slots, &addresses, |_, block| {
            bytes.extend_from_slice(block);
            Ok::<(), Infallible>(())
        })
        .map_err(EmissionError::into_format)?;
        if bytes.len() != length {
            return Err(FormatError::Internal(
                "array encoding differs from its complete block lengths".into(),
            ));
        }
        Ok(bytes)
    }

    /// Returns the combined length of the array blocks.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`EmptyChunkIndex::length`].
    pub(crate) fn length(&self) -> Result<u64, FormatError> {
        Ok(match self {
            Self::Empty(index) => return index.length(),
            Self::Extensible(plan) => plan.total_len(),
            Self::Fixed(plan) => plan.total_len(),
        })
    }

    /// Assigns consecutive stored addresses in [`Self::visit_blocks`] order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a block's exclusive end exceeds [`u64::MAX`],
    /// or [`FormatError::AllocationFailed`] if reserving the address vector fails. Propagates
    /// planning errors from [`Self::visit_blocks`].
    fn addresses_at(&self, address: StoredAddress) -> Result<Vec<StoredAddress>, FormatError> {
        let mut addresses = Vec::new();
        let mut cursor = address;
        self.visit_blocks(|length| {
            addresses
                .try_reserve(1)
                .map_err(|source| FormatError::AllocationFailed {
                    values: addresses.len() + 1,
                    source,
                })?;
            addresses.push(cursor);
            cursor = cursor.checked_add(length)?;
            Ok(())
        })?;
        Ok(addresses)
    }

    /// Plans individual blocks using the writer's address and length widths.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`ExtensibleArrayPlan::new`] and [`FixedArrayPlan::new`].
    fn plan(geometry: ChunkArrayGeometry<'_>) -> Result<Self, FormatError> {
        let ChunkArrayGeometry {
            kind,
            occupied,
            slots,
            chunk_bytes,
            filtered,
        } = geometry;
        let widths = FormatWidths {
            offsets: INDEX_OFFSET_SIZE,
            lengths: INDEX_LENGTH_SIZE,
        };
        Ok(match kind {
            ChunkArrayKind::ExtensibleArray => Self::Extensible(ExtensibleArrayPlan::new(
                ExtensibleArrayCreationParameters::canonical(),
                SlotOccupancy::Slots(&occupied),
                slots,
                chunk_bytes,
                widths,
                filtered,
            )?),
            ChunkArrayKind::FixedArray => {
                Self::Fixed(FixedArrayPlan::new(slots, chunk_bytes, widths, filtered)?)
            }
        })
    }
}

#[cfg(test)]
mod tests;
