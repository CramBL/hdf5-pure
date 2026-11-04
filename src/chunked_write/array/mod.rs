use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayPlan;
use hdf5_pure_format::__private::FixedArrayPlan;
use hdf5_pure_format::__private::SlotOccupancy;

use crate::error::FormatError;
use crate::file_writer::WIDTHS;

use super::ChunkArrayGeometry;
use super::ChunkArrayKind;
use super::EmptyChunkIndex;

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
        Ok(match kind {
            ChunkArrayKind::ExtensibleArray => Self::Extensible(ExtensibleArrayPlan::new(
                ExtensibleArrayCreationParameters::canonical(),
                SlotOccupancy::Slots(&occupied),
                slots,
                chunk_bytes,
                WIDTHS,
                filtered,
            )?),
            ChunkArrayKind::FixedArray => {
                Self::Fixed(FixedArrayPlan::new(slots, chunk_bytes, WIDTHS, filtered)?)
            }
        })
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
}
