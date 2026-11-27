use alloc::vec::Vec;

use super::FixedArrayPlan;
use super::header;
use crate::address::StoredAddress;
use crate::chunk_record::IndexSlots;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

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
/// [`FixedArrayPlan::place`], and [`super::FixedArrayPlacement::serialize_blocks`]. Also returns
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
    let data_block = fa_address.checked_add(header::header_len(plan.widths()))?;
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
