//! Checked chunk-array geometry, metadata requests, and reservation-backed emission.

use core::num::NonZeroU64;

use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayPlan;
use hdf5_pure_format::__private::FixedArrayPlan;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::SlotOccupancy;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ReservationRequest;

use crate::address::StoredAddress;
use crate::chunked_write::ChunkArrayKind;
use crate::error::Error;
use crate::error::FormatError;
use crate::width::FormatWidths;

use super::block::BlockRequest;
use super::block::EncodingDependency;
use super::reservation::PlannedReservation;
use super::reservation::ReservationPlan;

/// Checked block geometry for a fixed or extensible chunk array.
///
/// The caller reserves the metadata requests from [`Self::requests`] before passing borrowed
/// [`IndexSlots`] to [`Self::write`]. The writer obtains complete block lengths from the matching
/// format-layer plan and serializes each block through its encoder.
pub(super) enum ArrayIndexPlan {
    /// An extensible array with canonical creation parameters.
    Extensible(ExtensibleArrayPlan),
    /// A fixed array covering its complete logical slot span.
    Fixed(FixedArrayPlan),
}

impl ArrayIndexPlan {
    /// Plans complete array-index blocks for `slots` logical slots.
    ///
    /// Extensible arrays are planned with canonical creation parameters and `occupancy`. Fixed
    /// arrays cover every slot. `chunk_bytes` is the unfiltered chunk size used to derive the element
    /// encoding.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ExtensibleArrayPlan::new`] or [`FixedArrayPlan::new`].
    pub(super) fn new(
        kind: ChunkArrayKind,
        occupancy: SlotOccupancy<'_>,
        slots: u64,
        chunk_bytes: u64,
        widths: FormatWidths,
        filtered: bool,
    ) -> Result<Self, FormatError> {
        Ok(match kind {
            ChunkArrayKind::ExtensibleArray => Self::Extensible(ExtensibleArrayPlan::new(
                ExtensibleArrayCreationParameters::canonical(),
                occupancy,
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

    /// Returns complete metadata requests in format block order.
    ///
    /// The encoded length equals the complete allocation length. Encoding depends on the
    /// reservation addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a complete block length is zero.
    pub(super) fn requests(&self) -> Result<Vec<BlockRequest>, FormatError> {
        match self {
            Self::Extensible(plan) => plan.block_lengths().map(Self::request).collect(),
            Self::Fixed(plan) => plan.block_lengths().map(Self::request).collect(),
        }
    }

    /// Writes array blocks to matching reservations and returns the stored header address.
    ///
    /// The pending prefix must match [`Self::requests`] exactly. Previously emitted payload
    /// reservations may precede it, and later reservations remain pending. The reservation plan's
    /// address context is used for stored references. Chunk records remain borrowed. Parser child
    /// pages remain within their parent allocation.
    ///
    /// On error, the caller retains all reservations through `reservations`, including any partly
    /// emitted blocks.
    ///
    /// # Working memory
    ///
    /// The writer reuses one complete-block byte buffer and prepares a child reference index once
    /// for an extensible array. For a fixed array, the writer retains both complete block buffers
    /// together. Placement and reference metadata are separate from those byte buffers.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if pending reservations differ from the complete metadata
    /// requests. Returns errors from checked address conversion, format placement and encoding,
    /// or [`ReservationPlan::write_next`].
    pub(super) fn write(
        &self,
        slots: &IndexSlots<'_>,
        reservations: &mut ReservationPlan<'_>,
    ) -> Result<StoredAddress, Error> {
        let addresses = reservations.addresses();
        let count = match self {
            Self::Extensible(plan) => plan.block_lengths().len(),
            Self::Fixed(plan) => plan.block_lengths().len(),
        };
        let pending = reservations.pending().get(..count).ok_or_else(|| {
            FormatError::Internal("array index reservations omit complete blocks".into())
        })?;
        match self {
            Self::Extensible(plan) => Self::match_requests(plan.block_lengths(), pending)?,
            Self::Fixed(plan) => Self::match_requests(plan.block_lengths(), pending)?,
        }
        let starts = pending
            .iter()
            .map(|reservation| addresses.relative(reservation.range().start()))
            .collect::<Result<Vec<_>, _>>()?;
        let first = starts.first().copied().ok_or_else(|| {
            FormatError::Internal("an array index has no header reservation".into())
        })?;
        match self {
            Self::Extensible(plan) => {
                let placement = plan.place(&starts)?;
                let encoding = placement.encoding(slots)?;
                let mut bytes = Vec::new();
                for block in encoding.blocks() {
                    block.encode_into(&mut bytes)?;
                    reservations.write_next(&bytes)?;
                }
            }
            Self::Fixed(plan) => {
                let [header, data] = starts.as_slice() else {
                    return Err(FormatError::Internal(
                        "a fixed array requires two complete block reservations".into(),
                    )
                    .into());
                };
                for (_, bytes) in plan.place(*header, *data)?.serialize_blocks(slots)? {
                    reservations.write_next(&bytes)?;
                }
            }
        }
        Ok(first)
    }

    /// Creates a complete metadata request that depends on reservation addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `length` is zero.
    fn request(length: u64) -> Result<BlockRequest, FormatError> {
        NonZeroU64::new(length)
            .map(|length| {
                BlockRequest::complete(
                    ReservationRequest::new(length, AllocationClass::Metadata).for_block(),
                    EncodingDependency::Reservations,
                )
            })
            .ok_or_else(|| FormatError::Internal("an array index block is empty".into()))
    }

    /// Checks complete request equality for every pending array block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if request counts or descriptors differ.
    fn match_requests(
        lengths: impl ExactSizeIterator<Item = u64>,
        reservations: &[PlannedReservation],
    ) -> Result<(), FormatError> {
        if lengths.len() != reservations.len() {
            return Err(FormatError::Internal(
                "array index reservations differ from its complete blocks".into(),
            ));
        }
        for (length, reservation) in lengths.zip(reservations) {
            let request = Self::request(length)?;
            if request != reservation.block_request() {
                return Err(FormatError::Internal(
                    "an array index reservation differs from its block request".into(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
