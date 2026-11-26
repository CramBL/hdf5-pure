use crate::bytes;
use crate::error::FormatError;

/// Reads the little-endian unsigned integer of `width` bytes at `pos`, for a width of 1 to 8.
pub(super) fn read_var_uint(data: &[u8], pos: usize, width: usize) -> Result<u64, FormatError> {
    bytes::ensure_len(data, pos, width)?;
    let mut val = 0u64;
    for i in 0..width {
        val |= (data[pos + i] as u64) << (i * 8);
    }
    Ok(val)
}
