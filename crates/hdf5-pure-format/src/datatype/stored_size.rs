//! Datatype element sizes and conversions to host-sized integers.

use core::num::{NonZeroU32, NonZeroUsize};

use hdf5_pure_core::{__private, Datatype};

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// Returns the datatype's element size as a nonzero host-sized integer.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes
/// per element, or [`FormatError::ValueTooLargeForPlatform`] if its size exceeds
/// the target's `usize` range.
pub fn element_size_usize(datatype: &Datatype) -> Result<NonZeroUsize, FormatError> {
    usize::try_from(datatype.element_size()?.get())
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or(FormatError::ValueTooLargeForPlatform {
            value: u64::from(datatype.type_size()),
            target: "usize",
        })
}

/// Returns the nonzero size in bytes of one value stored at `offset_width`.
///
/// Uses [`stored_type_size`] to account for variable-length descriptors and array elements.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes, or the errors of
/// [`stored_type_size`] if an array size exceeds its encoded field.
pub fn stored_element_size(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<NonZeroU32, FormatError> {
    NonZeroU32::new(stored_type_size(datatype, offset_width)?)
        .map_or_else(|| datatype.element_size(), Ok)
}

/// Returns the size in bytes of one stored value, including zero-sized types.
///
/// Computes variable-length descriptor sizes at `offset_width` and array sizes from their base
/// types and dimensions. Uses [`Datatype::type_size`] for other types, including declared
/// compound record sizes.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte datatype size
/// field.
pub fn stored_type_size(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<u32, FormatError> {
    __private::stored_type_size(datatype, offset_width)
}

/// Returns the stored element size as a nonzero host-sized integer.
///
/// # Errors
///
/// Returns the errors of [`stored_element_size`], or [`FormatError::ValueTooLargeForPlatform`]
/// if the size exceeds the target's [`usize`] range.
pub fn stored_element_size_usize(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<NonZeroUsize, FormatError> {
    stored_element_size(datatype, offset_width)?.narrow()
}
