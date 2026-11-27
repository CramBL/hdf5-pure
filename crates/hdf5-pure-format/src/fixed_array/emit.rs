use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::checksum::JenkinsLookup3;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;

/// Reserves capacity for `length` bytes in an empty buffer.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if `length` does not fit [`usize`], or
/// [`FormatError::AllocationFailed`] if the capacity reservation fails.
pub(super) fn output_buffer(length: u64) -> Result<Vec<u8>, FormatError> {
    let values = length.to_usize()?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(values)
        .map_err(|source| FormatError::AllocationFailed { values, source })?;
    Ok(bytes)
}

pub(super) fn emit_records<E: From<FormatError>>(
    write: &mut impl FnMut(&[u8]) -> Result<(), E>,
    checksum: &mut JenkinsLookup3,
    encoding: ElementEncoding,
    records: &[Option<ChunkRecord>],
) -> Result<(), E> {
    let stride = usize::from(encoding.stride().get());
    let mut scratch = [0; EMISSION_SCRATCH_LEN];
    let maximum = scratch.len() / stride;
    if maximum == 0 {
        return Err(FormatError::InvalidFixedArrayGeometry.into());
    }
    for records in records.chunks(maximum) {
        let length = records.len() * stride;
        for (bytes, &record) in scratch[..length].chunks_exact_mut(stride).zip(records) {
            bytes.copy_from_slice(encoding.encode(record)?.as_bytes());
        }
        checksum.update(&scratch[..length])?;
        write(&scratch[..length])?;
    }
    Ok(())
}

pub(super) fn emit_repeated<E: From<FormatError>>(
    write: &mut impl FnMut(&[u8]) -> Result<(), E>,
    mut checksum: Option<&mut JenkinsLookup3>,
    pattern: &[u8],
    mut length: u64,
) -> Result<(), E> {
    let Some(stride) = NonZeroUsize::new(pattern.len()) else {
        return Err(FormatError::Internal("an emitted byte pattern is empty".into()).into());
    };
    let mut scratch = [0; EMISSION_SCRATCH_LEN];
    let used = scratch.len() / stride.get() * stride.get();
    if used == 0 || length % stride.get().to_u64() != 0 {
        return Err(FormatError::InvalidFixedArrayGeometry.into());
    }
    for element in scratch[..used].chunks_exact_mut(stride.get()) {
        element.copy_from_slice(pattern);
    }
    while length != 0 {
        let count = length.min(used.to_u64()).to_usize()?;
        let bytes = &scratch[..count];
        if let Some(checksum) = checksum.as_mut() {
            checksum.update(bytes)?;
        }
        write(bytes)?;
        length -= count.to_u64();
    }
    Ok(())
}

const EMISSION_SCRATCH_LEN: usize = 4_096;
