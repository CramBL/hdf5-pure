use hdf5_pure_core::__private::FileRange;

use crate::error::FormatError;

pub(super) fn without_checksum(bytes: &[u8]) -> Result<&[u8], FormatError> {
    bytes
        .split_last_chunk::<{ size_of::<u32>() }>()
        .map(|(body, _checksum)| body)
        .ok_or_else(|| FormatError::Internal("a Fixed Array block ends inside its checksum".into()))
}

pub(super) fn checksum_ranges(range: FileRange) -> Result<(FileRange, FileRange), FormatError> {
    let body_length = range
        .length()
        .checked_sub(CHECKSUM_LEN)
        .ok_or(FormatError::InvalidFixedArrayGeometry)?;
    let body = FileRange::new(range.start(), body_length)?;
    Ok((body, FileRange::new(body.end(), CHECKSUM_LEN)?))
}

pub(super) const VERSION: u8 = 0;
pub(super) const SIGNATURE_LEN: u64 = 4;
pub(super) const CHECKSUM_LEN: u64 = 4;
