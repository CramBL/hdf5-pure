//! Zero-filled image ranges emitted with a reusable scratch buffer.

use hdf5_pure_core::__private::FileRange;

use crate::convert::Narrow;
use crate::error::{Error, FormatError};

/// Returns whether a nonempty range starts at the current image length.
///
/// Returns `false` for an empty range at any address.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if a nonempty range extends beyond `length`
/// without starting there.
pub(super) fn appends(range: FileRange, length: u64) -> Result<bool, Error> {
    if range.length() == 0 {
        return Ok(false);
    }
    if range.start().get() == length {
        return Ok(true);
    }
    if range.end().get() > length {
        return Err(FormatError::UnexpectedSourceEof {
            expected: range.end().get(),
            available: length,
        }
        .into());
    }
    Ok(false)
}

/// Writes consecutive zero batches through `write`, reusing at most 64 KiB of scratch.
///
/// Calls `write` with each batch's physical offset and bytes. An empty range requires no
/// allocation or callback. Offsets and the complete range length remain [`u64`] values.
///
/// # Errors
///
/// Returns the callback's error if writing a batch fails. Stops calling `write` after that error.
pub(super) fn emit(
    range: FileRange,
    mut write: impl FnMut(u64, &[u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    if range.length() == 0 {
        return Ok(());
    }
    let bytes = vec![0; scratch_length(range)];
    let mut offset = range.start().get();
    while offset < range.end().get() {
        let length = (range.end().get() - offset)
            .min(bytes.len() as u64)
            .to_usize()?;
        write(offset, &bytes[..length])?;
        offset += length as u64;
    }
    Ok(())
}

/// Returns the smaller of the range length and the zero batch limit.
pub(crate) fn scratch_length(range: FileRange) -> usize {
    // The minimum is bounded by ZERO_BATCH_BYTES, which is represented by usize.
    usize::try_from(range.length().min(ZERO_BATCH_BYTES as u64))
        .expect("a bounded zero batch fits in usize")
}

/// The maximum scratch buffer length for one zero-emission request.
const ZERO_BATCH_BYTES: usize = 64 * 1024;

#[cfg(test)]
mod tests;
