//! Source reads with one acquired physical metadata range.

use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::MetadataSource;

use crate::convert;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source;
use crate::source::Source;

/// A source view that reuses one acquired metadata range.
///
/// Reads use physical file positions and the source length recorded at acquisition. The caller
/// must keep the source contents stable while using this view. Object identity, allocation
/// ownership, and permission to write require separate evidence.
///
/// # Performance
///
/// Retains one [`MetadataBytes`] owner. Contained metadata reads borrow or share its backing
/// bytes. For overlapping reads, copies the acquired bytes and reads only uncovered ranges from
/// the source. Metadata reads use [`Source::read_metadata`], and buffer reads use
/// [`Source::read_at`]. Joining an overlapping metadata read allocates a buffer as long as the read,
/// with additional ownership bookkeeping and allocations made by the source.
pub(super) struct CurrentMetadata<'a, S: Source + ?Sized> {
    source: &'a S,
    bytes: MetadataBytes<'a>,
    length: u64,
}

impl<'a, S: Source + ?Sized> CurrentMetadata<'a, S> {
    /// Acquires `range` from `source` after checking its complete extent.
    ///
    /// # Errors
    ///
    /// Returns [`CurrentMetadataError::Read`] if the range extends past the source length, its
    /// length exceeds [`usize::MAX`], or the source read fails. Returns
    /// [`CurrentMetadataError::Range`] if the returned metadata has a different physical range.
    pub(super) fn new(source: &'a S, range: FileRange) -> Result<Self, CurrentMetadataError> {
        let length = source.len();
        if range.end().get() > length {
            return Err(CurrentMetadataError::Read(source::unexpected_eof(
                range, length,
            )));
        }
        let bytes = source
            .read_metadata(
                range.start().get(),
                range
                    .length()
                    .to_usize()
                    .map_err(CurrentMetadataError::Read)?,
            )
            .map_err(CurrentMetadataError::Read)?;
        if bytes.range() != range {
            return Err(CurrentMetadataError::Range {
                expected: range,
                provided: bytes.range(),
            });
        }
        Ok(Self {
            source,
            bytes,
            length,
        })
    }

    /// Borrows the acquired bytes with their physical range.
    pub(super) fn region(&self) -> FileRegion<'_> {
        self.bytes.region()
    }

    /// Returns the capacity of the acquired owner's backing allocation, or zero for borrowed bytes.
    ///
    /// Includes capacity outside the acquired range and excludes ownership bookkeeping and
    /// allocations retained separately by the source.
    pub(super) fn retained_bytes(&self) -> usize {
        self.bytes.retained_bytes()
    }

    fn external(&self, range: FileRange) -> Result<MetadataBytes<'_>, FormatError> {
        let bytes = self
            .source
            .read_metadata(range.start().get(), range.length().to_usize()?)?;
        FileRegion::new(range, bytes.as_ref())?;
        bytes.region().subregion(range)?;
        Ok(bytes)
    }

    fn request(&self, offset: u64, length: usize) -> Result<FileRange, FormatError> {
        let range = FileRange::new(PhysicalAddress::new(offset), length.to_u64())?;
        if range.end().get() > self.length {
            return Err(source::unexpected_eof(range, self.length));
        }
        Ok(range)
    }
}

impl<S: Source + ?Sized> MetadataSource for CurrentMetadata<'_, S> {
    type Bytes<'a>
        = MetadataBytes<'a>
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, length: usize) -> Result<Self::Bytes<'_>, FormatError> {
        let request = self.request(offset, length)?;
        let Self {
            source: _,
            bytes,
            length: _,
        } = self;
        if length == 0 {
            return MetadataBytes::borrowed(request, &[]);
        }
        if bytes.range().contains(request) {
            return bytes.subregion(request);
        }
        let Some(parts) = ReadParts::new(request, bytes.range())? else {
            return self.external(request);
        };
        let ReadParts {
            prefix,
            cached,
            suffix,
        } = parts;
        let mut joined = vec![0; length];
        for part in [prefix, suffix] {
            if part.length() > 0 {
                let current = self.external(part)?;
                local_mut(&mut joined, request, part)?.copy_from_slice(current.as_ref());
            }
        }
        local_mut(&mut joined, request, cached)?
            .copy_from_slice(bytes.region().subregion(cached)?.bytes());
        MetadataBytes::owned(request, joined)
    }

    fn len(&self) -> u64 {
        self.length
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        let request = self.request(offset, buffer.len())?;
        let Self {
            source,
            bytes,
            length: _,
        } = self;
        if buffer.is_empty() {
            return Ok(());
        }
        let Some(parts) = ReadParts::new(request, bytes.range())? else {
            return source.read_at(offset, buffer);
        };
        let ReadParts {
            prefix,
            cached,
            suffix,
        } = parts;
        if prefix.length() > 0 {
            source.read_at(prefix.start().get(), local_mut(buffer, request, prefix)?)?;
        }
        local_mut(buffer, request, cached)?
            .copy_from_slice(bytes.region().subregion(cached)?.bytes());
        if suffix.length() > 0 {
            source.read_at(suffix.start().get(), local_mut(buffer, request, suffix)?)?;
        }
        Ok(())
    }

    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.read_metadata(offset, length)?.into_vec())
    }
}

/// A request split into an acquired intersection and uncovered prefix and suffix.
struct ReadParts {
    prefix: FileRange,
    cached: FileRange,
    suffix: FileRange,
}

impl ReadParts {
    fn new(request: FileRange, acquired: FileRange) -> Result<Option<Self>, FormatError> {
        let Some(cached) = request.intersection(acquired) else {
            return Ok(None);
        };
        Ok(Some(Self {
            prefix: FileRange::new(
                request.start(),
                cached.start().get() - request.start().get(),
            )?,
            cached,
            suffix: FileRange::new(cached.end(), request.end().get() - cached.end().get())?,
        }))
    }
}

/// An unsuccessful source read or a metadata range mismatch during acquisition.
#[derive(Debug)]
pub(super) enum CurrentMetadataError {
    /// The source returned metadata for a different physical range.
    Range {
        /// The physical range passed by the caller.
        expected: FileRange,
        /// The physical range of the returned metadata.
        provided: FileRange,
    },
    /// An extent, local-length conversion, or source-read error.
    Read(FormatError),
}

fn local_mut(
    buffer: &mut [u8],
    whole: FileRange,
    part: FileRange,
) -> Result<&mut [u8], FormatError> {
    FileRegion::new(whole, buffer)?.subregion(part)?;
    let range = convert::slice_range(part.start().get() - whole.start().get(), part.length())?;
    buffer
        .get_mut(range)
        .ok_or(FormatError::InsufficientRegion {
            required_start: part.start().get(),
            required_length: part.length(),
            provided_start: whole.start().get(),
            provided_length: whole.length(),
        })
}

#[cfg(test)]
mod tests;
