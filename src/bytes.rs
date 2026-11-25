use alloc::vec::Vec;

pub use hdf5_pure_format::__private::ensure_len;

use crate::error::FormatError;

/// Sets the buffer length to `len`, zeroing added bytes and reusing its capacity.
///
/// # Errors
///
/// Returns [`FormatError::AllocationFailed`] if reserving space fails, leaving the buffer
/// unchanged.
pub(crate) fn resize_buffer(buffer: &mut Vec<u8>, len: usize) -> Result<(), FormatError> {
    buffer
        .try_reserve_exact(len.saturating_sub(buffer.len()))
        .map_err(|source| FormatError::AllocationFailed {
            values: len,
            source,
        })?;
    buffer.resize(len, 0);
    Ok(())
}
