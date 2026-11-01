#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

use hdf5_pure_core::__private::{FileRange, FileRegion, MetadataFrame};
use hdf5_pure_core::AddressRange;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::MetadataSource;

use crate::address::BaseAddressExt;
#[cfg(any(test, doc))]
use crate::address::StoredAddressExt;
use crate::address::{BaseAddress, StoredAddress};
use crate::convert;
use crate::convert::Narrow;
use crate::error::FormatError;

#[cfg(doc)]
use super::StoredSource;

/// Default maximum size of one entry admitted to a streaming metadata cache.
pub const DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES: usize = 64 * 1024;

/// Initial metadata-cache settings for streaming file access.
///
/// This is the `hdf5-pure` counterpart to the memory-budget portion of HDF5's
/// `H5Pset_mdc_config`: it bounds the bytes retained for parsed metadata reads
/// while a file is opened through [`crate::File::open_streaming_with_options`].
/// Raw dataset payload reads use [`Source::read_exact_at`] and are not
/// admitted to this cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataCacheConfig {
    max_bytes: usize,
    max_entry_bytes: usize,
}

impl MetadataCacheConfig {
    /// Creates a metadata-cache configuration with a total byte budget.
    ///
    /// The default entry cap is the smaller of 64 KiB and the total byte budget. Use
    /// [`with_max_entry_bytes`](Self::with_max_entry_bytes) to change the entry cap.
    pub const fn new(max_bytes: usize) -> Self {
        let max_entry_bytes = if max_bytes < DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES {
            max_bytes
        } else {
            DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES
        };
        Self {
            max_bytes,
            max_entry_bytes,
        }
    }

    /// Returns a configuration that disables metadata caching.
    pub const fn disabled() -> Self {
        Self {
            max_bytes: 0,
            max_entry_bytes: 0,
        }
    }

    /// Sets the maximum size of a single metadata read admitted to the cache.
    pub const fn with_max_entry_bytes(mut self, max_entry_bytes: usize) -> Self {
        self.max_entry_bytes = max_entry_bytes;
        self
    }

    /// Returns the total metadata-cache byte budget.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Returns the maximum size of one cached metadata entry.
    pub const fn max_entry_bytes(&self) -> usize {
        self.max_entry_bytes
    }

    /// Returns whether metadata caching is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.max_bytes > 0 && self.max_entry_bytes > 0
    }
}

impl Default for MetadataCacheConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Cumulative metadata-cache counters and current occupancy.
///
/// Returned by [`File::metadata_cache_stats`](crate::File::metadata_cache_stats). Counters
/// accumulate from file opening or the last call to
/// [`File::reset_metadata_cache_stats`](crate::File::reset_metadata_cache_stats).
/// [`entries`](Self::entries) and [`bytes`](Self::bytes) describe the currently retained entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MetadataCacheStats {
    hits: u64,
    misses: u64,
    oversize_reads: u64,
    evictions: u64,
    invalidations: u64,
    entries: usize,
    bytes: usize,
}

impl MetadataCacheStats {
    /// Returns the number of metadata reads served from the cache.
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Returns the number of eligible metadata reads absent from the cache.
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Returns the number of metadata reads exceeding the entry or total byte budget.
    ///
    /// These reads are included in [`reads`](Self::reads) and excluded from
    /// [`misses`](Self::misses) and [`hit_rate`](Self::hit_rate).
    pub const fn oversize_reads(&self) -> u64 {
        self.oversize_reads
    }

    /// Returns the number of entries evicted to satisfy [`MetadataCacheConfig::max_bytes`].
    pub const fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Returns the number of entries invalidated by overlapping in-place writes.
    ///
    /// Returns zero for a read-only session. Increasing the byte budget does not prevent
    /// invalidation when the caller rewrites cached metadata.
    pub const fn invalidations(&self) -> u64 {
        self.invalidations
    }

    /// Returns the number of cached entries.
    ///
    /// Dividing [`bytes`](Self::bytes) by this count gives the mean retained capacity per entry.
    pub const fn entries(&self) -> usize {
        self.entries
    }

    /// Returns the payload capacity retained by cache entries.
    ///
    /// Compare against [`MetadataCacheConfig::max_bytes`]. Owners held outside the cache may keep
    /// allocations alive after eviction. Ownership bookkeeping is excluded.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Returns the sum of hits, misses, and reads too large to admit.
    ///
    /// [`hit_rate`](Self::hit_rate) excludes reads too large to admit from its denominator.
    pub const fn reads(&self) -> u64 {
        self.hits
            .saturating_add(self.misses)
            .saturating_add(self.oversize_reads)
    }

    /// Returns the fraction of eligible metadata reads served from the cache.
    ///
    /// Returns `None` before the first eligible read. Reads too large to admit are excluded.
    pub fn hit_rate(&self) -> Option<f64> {
        let eligible = self.hits.saturating_add(self.misses);
        if eligible == 0 {
            return None;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "a hit rate is a ratio, and f64 holds these counts exactly far past any \
                      read count a process will reach"
        )]
        Some(self.hits as f64 / eligible as f64)
    }
}

/// A random-access, read-only source of the bytes of an HDF5 file.
///
/// Read offsets are byte positions measured from the beginning of the source. Implementations
/// fill the complete requested buffer or return an error.
///
/// # Implementing one
///
/// Implement [`len`](Source::len) and [`read_at`](Source::read_at). The other methods have default
/// implementations.
///
/// The reader treats a source as immutable. Its length must remain accurate and constant, and
/// repeated reads must return the same bytes. Use [`File::open_swmr`](crate::File::open_swmr) to
/// read a file that another process is appending to. Implementations over mutable handles must
/// synchronize access, as [`ReadSeekSource`] does with a mutex.
///
/// Metadata parsers call [`read_metadata`](Source::read_metadata), which may borrow resident
/// bytes or share an immutable allocation. Its default calls
/// [`read_metadata_at`](Source::read_metadata_at). A source with base-relative offsets also
/// implements [`metadata_origin`](Source::metadata_origin) to identify its physical origin.
///
/// To cache metadata reads, pass a [`MetadataCacheConfig`] to
/// [`File::from_source_with_options`](crate::File::from_source_with_options). A wrapper that
/// forwards metadata reads also forwards [`metadata_cache_stats`](Source::metadata_cache_stats)
/// and [`reset_metadata_cache_stats`](Source::reset_metadata_cache_stats).
#[expect(
    clippy::len_without_is_empty,
    reason = "`len` here is a file's byte length, not a container's count — the shape of \
              `std::fs::Metadata::len`, which ships without an `is_empty` for the same \
              reason. An HDF5 file is never empty (the signature alone is eight bytes), so \
              an `is_empty` on this trait would be a public method with no caller, and one \
              an implementation could contradict its own `len` with"
)]
pub trait Source {
    /// Returns the number of readable bytes in the source.
    ///
    /// Report the actual readable extent. The default [`read_exact_at`](Self::read_exact_at)
    /// checks this bound before allocating a buffer whose length may come from an untrusted
    /// file. An overstated extent permits allocations larger than the readable data.
    fn len(&self) -> u64;

    /// Reads exactly `buf.len()` bytes at `offset` into `buf`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the request ends past
    /// [`len`](Self::len), [`FormatError::OffsetOverflow`] if `offset + buf.len()`
    /// exceeds [`u64::MAX`], and [`FormatError::Source`] if a backend read fails.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError>;

    /// Reads `len` bytes at `offset` into a new vector.
    ///
    /// Checks `offset + len` against [`len`](Self::len) before allocating the
    /// buffer or calling [`read_at`](Self::read_at).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if `offset + len` exceeds
    /// [`u64::MAX`], and [`FormatError::UnexpectedSourceEof`] if `offset + len`
    /// exceeds the source extent. Propagates errors from [`read_at`](Self::read_at).
    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let end = offset
            .checked_add(len as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: len as u64,
            })?;
        if end > self.len() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end,
                available: self.len(),
            });
        }
        let mut buf = vec![0u8; len];
        self.read_at(offset, &mut buf)?;
        Ok(buf)
    }

    /// Reads metadata bytes through the source's metadata cache, where enabled.
    ///
    /// The default implementation performs an uncached exact read. Raw dataset
    /// payload readers intentionally call [`read_exact_at`](Self::read_exact_at)
    /// instead, so a metadata cache does not retain user data chunks.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read_exact_at`](Self::read_exact_at).
    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.read_exact_at(offset, len)
    }

    /// Returns the physical file position represented by source offset zero.
    ///
    /// The default is zero. A base-relative view adds its base to its inner source's origin.
    /// Metadata owners returned by [`read_metadata`](Self::read_metadata) use physical positions,
    /// while read offsets remain relative to this origin.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a view's physical origin exceeds [`u64::MAX`].
    fn metadata_origin(&self) -> Result<u64, FormatError> {
        Ok(0)
    }

    /// Reads metadata with immutable ownership and a checked physical range.
    ///
    /// Returns exactly `len` bytes at physical position
    /// [`metadata_origin`](Self::metadata_origin) `+ offset`. An override may borrow resident
    /// bytes or share an allocation. The default takes ownership of the vector returned by
    /// [`read_metadata_at`](Self::read_metadata_at), preserving overrides of that method.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Source`] if the owned adapter returns a different byte count,
    /// [`FormatError::OffsetOverflow`] if the physical range exceeds [`u64::MAX`], and errors from
    /// [`read_metadata_at`](Self::read_metadata_at) or [`metadata_origin`](Self::metadata_origin).
    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let bytes = self.read_metadata_at(offset, len)?;
        exact_metadata(offset, len, bytes.len())?;
        MetadataBytes::owned(metadata_range(self.metadata_origin()?, offset, len)?, bytes)
    }

    /// Returns metadata-cache statistics, or `None` if caching is disabled.
    ///
    /// The default returns `None`. A wrapper that forwards metadata reads to an inner source
    /// must also forward this method and
    /// [`reset_metadata_cache_stats`](Self::reset_metadata_cache_stats).
    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        None
    }

    /// Resets cumulative metadata-cache counters, retaining cached entries.
    ///
    /// The default does nothing. Cache implementations preserve the current entry and byte counts.
    fn reset_metadata_cache_stats(&self) {}
}

// Forward `Source` through references and boxes so `&S`, `&dyn Source`,
// and `Box<dyn Source>` are all usable wherever an `S: Source` is.
impl<S: Source + ?Sized> Source for &S {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        (**self).read_at(offset, buf)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_exact_at(offset, len)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_metadata_at(offset, len)
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        (**self).metadata_origin()
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let bytes = (**self).read_metadata(offset, len)?;
        exact_metadata(offset, len, bytes.len())?;
        Ok(bytes)
    }

    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        (**self).metadata_cache_stats()
    }

    fn reset_metadata_cache_stats(&self) {
        (**self).reset_metadata_cache_stats();
    }
}

#[cfg(feature = "std")]
impl<S: Source + ?Sized> Source for std::boxed::Box<S> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        (**self).read_at(offset, buf)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_exact_at(offset, len)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        (**self).read_metadata_at(offset, len)
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        (**self).metadata_origin()
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let bytes = (**self).read_metadata(offset, len)?;
        exact_metadata(offset, len, bytes.len())?;
        Ok(bytes)
    }

    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        (**self).metadata_cache_stats()
    }

    fn reset_metadata_cache_stats(&self) {
        (**self).reset_metadata_cache_stats();
    }
}

// ---------------------------------------------------------------------------
// Caller-supplied sources
// ---------------------------------------------------------------------------

/// Checks the byte count and physical range returned by a caller-supplied [`Source`].
///
/// Metadata parsers index returned bytes using the length passed to the read. A source that
/// returns a different extent or origin is rejected with [`FormatError::Source`] before parsing.
#[cfg(feature = "std")]
pub(crate) struct ValidatedSource<S>(S);

#[cfg(feature = "std")]
impl<S> ValidatedSource<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self(inner)
    }

    /// Rejects a buffer whose length differs from the request.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Source`] if `bytes.len()` differs from `len`.
    fn check(offset: u64, len: usize, bytes: Vec<u8>) -> Result<Vec<u8>, FormatError> {
        if bytes.len() == len {
            return Ok(bytes);
        }
        Err(FormatError::Source(std::format!(
            "the source returned {} bytes for a {len}-byte read at offset {offset}",
            bytes.len()
        )))
    }
}

#[cfg(feature = "std")]
impl<S: Source> Source for ValidatedSource<S> {
    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.0.read_at(offset, buf)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        Self::check(offset, len, self.0.read_exact_at(offset, len)?)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        Self::check(offset, len, self.0.read_metadata_at(offset, len)?)
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        self.0.metadata_origin()
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let bytes = self.0.read_metadata(offset, len)?;
        exact_metadata(offset, len, bytes.len())?;
        let expected = metadata_range(self.metadata_origin()?, offset, len)?;
        if bytes.range() != expected {
            return Err(FormatError::Source(alloc::format!(
                "the source returned metadata at physical offset {} for a read at physical offset {}",
                bytes.offset(),
                expected.start().get()
            )));
        }
        Ok(bytes)
    }

    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.0.metadata_cache_stats()
    }

    fn reset_metadata_cache_stats(&self) {
        self.0.reset_metadata_cache_stats();
    }
}

// ---------------------------------------------------------------------------
// Base-relative view
// ---------------------------------------------------------------------------

/// A base-relative adapter for readers using the [`Source`] interface.
///
/// Read offsets are relative to `base` in the inner source. Metadata reads use its cache and
/// retain its physical metadata ranges. Payload reads use [`Source::read_exact_at`] on the inner
/// source after checking the base-relative extent.
#[cfg(test)]
pub(crate) struct BaseOffsetSource<'a, S: Source + ?Sized> {
    pub(crate) inner: &'a S,
    pub(crate) base: BaseAddress,
}

/// Borrows the bytes after `base` so stored addresses index the slice directly.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if `base` exceeds the length of `bytes`.
fn frame(bytes: &[u8], base: BaseAddress) -> Result<&[u8], FormatError> {
    if base.is_zero() {
        return Ok(bytes);
    }
    if base.get() > bytes.len().to_u64() {
        return Err(FormatError::UnexpectedSourceEof {
            expected: base.get(),
            available: bytes.len().to_u64(),
        });
    }
    bytes
        .get(base.get().to_usize()?..)
        .ok_or(FormatError::UnexpectedSourceEof {
            expected: base.get(),
            available: bytes.len().to_u64(),
        })
}

#[cfg(test)]
impl<S: Source + ?Sized> Source for BaseOffsetSource<'_, S> {
    fn len(&self) -> u64 {
        self.inner.len().saturating_sub(self.base.get())
    }
    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.inner
            .read_at(self.base.resolve(StoredAddress::new(offset))?.get(), buffer)
    }
    fn read_exact_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        let range = AddressRange::new(StoredAddress::new(offset), length.to_u64())?;
        if range.end().get() > self.len() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: range.end().get(),
                available: self.len(),
            });
        }
        self.inner
            .read_exact_at(self.base.resolve(range.start())?.get(), length)
    }
    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        self.inner
            .read_metadata_at(self.base.resolve(StoredAddress::new(offset))?.get(), length)
    }
    fn metadata_origin(&self) -> Result<u64, FormatError> {
        Ok(PhysicalAddress::new(self.inner.metadata_origin()?)
            .checked_add(self.base.get())?
            .get())
    }
    fn read_metadata(&self, offset: u64, length: usize) -> Result<MetadataBytes<'_>, FormatError> {
        self.inner
            .read_metadata(self.base.resolve(StoredAddress::new(offset))?.get(), length)
    }
    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.inner.metadata_cache_stats()
    }
    fn reset_metadata_cache_stats(&self) {
        self.inner.reset_metadata_cache_stats();
    }
}

/// Adapts a [`Source`] to the [`MetadataSource`] trait that the format crate's parsers take.
///
/// Each method calls the method of the same name on the wrapped source, so the parsers read
/// metadata through the source's metadata cache, where it has one.
pub(crate) struct SourceMetadata<'a, S: Source + ?Sized>(pub(crate) &'a S);

impl<S: Source + ?Sized> MetadataSource for SourceMetadata<'_, S> {
    type Bytes<'a>
        = MetadataBytes<'a>
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
        self.0.read_metadata(offset, len)
    }

    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.0.read_at(offset, buf)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.0.read_metadata_at(offset, len)
    }
}

/// Metadata reads with requests in the coordinate system of `R`.
///
/// Returned [`MetadataBytes`] retain their physical ranges. Implementations for byte slices
/// and [`Source`] use [`FileRange`]. A [`StoredSource`] requires a checked stored-coordinate range.
pub(crate) trait RangeReader<R = FileRange> {
    /// Returns the bytes of `range`, borrowed from a byte slice or read from a [`Source`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the file ends inside `range`,
    /// [`FormatError::ValueTooLargeForPlatform`] if the range length exceeds [`usize::MAX`],
    /// [`FormatError::OffsetOverflow`] if resolving a stored start exceeds [`u64::MAX`], and
    /// errors from reading a [`Source`].
    fn read_range(&self, range: R) -> Result<MetadataBytes<'_>, FormatError>;
}

impl RangeReader for [u8] {
    fn read_range(&self, range: FileRange) -> Result<MetadataBytes<'_>, FormatError> {
        MetadataBytes::borrowed(range, bytes_in(self, range)?)
    }
}

impl<S: Source + ?Sized> RangeReader for S {
    fn read_range(&self, range: FileRange) -> Result<MetadataBytes<'_>, FormatError> {
        self.read_metadata(range.start().get(), range.length().to_usize()?)
    }
}

impl Source for MetadataFrame<'_> {
    fn len(&self) -> u64 {
        self.bytes().len().to_u64()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        BytesSource::new(self.bytes()).read_at(offset, buf)
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        Ok(self.origin().get())
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let physical = metadata_range(self.origin().get(), offset, len)?;
        let local = metadata_range(0, offset, len)?;
        MetadataBytes::borrowed(physical, bytes_in(self.bytes(), local)?)
    }
}

/// Borrows a base-relative slice with its physical origin.
///
/// # Errors
///
/// Returns the errors of [`frame`] and [`MetadataFrame::new`].
pub(crate) fn framed_metadata(
    bytes: &[u8],
    base: BaseAddress,
) -> Result<MetadataFrame<'_>, FormatError> {
    MetadataFrame::new(PhysicalAddress::new(base.get()), frame(bytes, base)?)
}

/// Interprets a legacy locator's base-relative endpoints as stored positions.
///
/// The caller initializes the locator with [`BaseAddressExt::ZERO`]. The numeric endpoints are
/// stored positions, and the metadata owner's physical range is maintained separately.
///
/// # Errors
///
/// Returns errors from [`BaseAddressExt::relative`] and [`AddressRange::new`].
pub(crate) fn legacy_stored_range(
    range: FileRange,
) -> Result<AddressRange<StoredAddress>, FormatError> {
    AddressRange::new(BaseAddress::ZERO.relative(range.start())?, range.length())
}

/// Borrows the owner's bytes in the coordinate frame used by a legacy locator.
///
/// `range` is the locator's range before physical-origin resolution. The owner retains its
/// physical range independently of this parser view.
///
/// # Errors
///
/// Returns [`FormatError::FileRegionLengthMismatch`] if the locator's extent differs from the
/// byte count.
pub(crate) fn legacy_framed_region<'a>(
    range: FileRange,
    bytes: &'a MetadataBytes<'_>,
) -> Result<FileRegion<'a>, FormatError> {
    FileRegion::new(range, bytes)
}

fn metadata_range(origin: u64, offset: u64, len: usize) -> Result<FileRange, FormatError> {
    FileRange::new(
        PhysicalAddress::new(origin).checked_add(offset)?,
        len.to_u64(),
    )
}

fn exact_metadata(offset: u64, len: usize, actual: usize) -> Result<(), FormatError> {
    if actual != len {
        return Err(FormatError::Source(alloc::format!(
            "the source returned {actual} bytes for a {len}-byte read at offset {offset}"
        )));
    }
    Ok(())
}

/// Returns the bytes of `range` in `bytes`, a buffer whose first byte is at address 0.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if `range` extends past `bytes`.
pub(crate) fn bytes_in(bytes: &[u8], range: FileRange) -> Result<&[u8], FormatError> {
    if range.end().get() > bytes.len().to_u64() {
        return Err(unexpected_eof(range, bytes.len().to_u64()));
    }
    bytes
        .get(convert::slice_range(range.start().get(), range.length())?)
        .ok_or_else(|| unexpected_eof(range, bytes.len().to_u64()))
}

/// Reports a request past a source's extent with its original positions.
///
/// `range` and `extent` are measured from the same source origin.
pub(crate) fn unexpected_eof(range: FileRange, extent: u64) -> FormatError {
    FormatError::UnexpectedSourceEof {
        expected: range.end().get(),
        available: extent,
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// A [`Source`] over an owned or borrowed byte buffer.
///
/// This backend is available without the `std` feature.
#[derive(Debug, Clone, Copy)]
pub struct BytesSource<T>(pub T);

impl<T: AsRef<[u8]>> BytesSource<T> {
    /// Wraps an in-memory byte buffer.
    pub fn new(bytes: T) -> Self {
        BytesSource(bytes)
    }
}

impl<T: AsRef<[u8]>> Source for BytesSource<T> {
    fn len(&self) -> u64 {
        self.0.as_ref().len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        let bytes = self.0.as_ref();
        let end = offset
            .checked_add(buf.len().to_u64())
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > bytes.len().to_u64() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end,
                available: bytes.len().to_u64(),
            });
        }
        let region = bytes.get(offset.to_usize()?..end.to_usize()?).ok_or(
            FormatError::UnexpectedSourceEof {
                expected: end,
                available: bytes.len().to_u64(),
            },
        )?;
        buf.copy_from_slice(region);
        Ok(())
    }
    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let range = metadata_range(0, offset, len)?;
        MetadataBytes::borrowed(range, bytes_in(self.0.as_ref(), range)?)
    }
}

// ---------------------------------------------------------------------------
// Metadata-caching wrapper (std)
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
struct CachedMetadataRead {
    bytes: MetadataBytes<'static>,
    last_access: u64,
}

/// A bounded least-recently-used cache of owned metadata reads.
///
/// Entries are indexed by their requested range and access order. In-place writers invalidate
/// entries that overlap a write.
#[cfg(feature = "std")]
pub(crate) struct MetadataReadCache {
    /// Entries indexed by `(offset, len)`, ordered by offset for overlap queries.
    entries: BTreeMap<(u64, usize), CachedMetadataRead>,
    /// Entry keys indexed by access tick, with the least recently used entry first.
    by_access: BTreeMap<u64, (u64, usize)>,
    current_bytes: usize,
    tick: u64,
    /// The longest `len` ever admitted, bounding how far *before* a given
    /// offset an entry that overlaps it can start. It only grows, so the window
    /// it gives can be wider than needed but never too narrow.
    longest_entry: usize,
    /// Cumulative counters, reported as [`MetadataCacheStats`].
    hits: u64,
    misses: u64,
    oversize_reads: u64,
    evictions: u64,
    invalidations: u64,
}

#[cfg(feature = "std")]
impl MetadataReadCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            by_access: BTreeMap::new(),
            current_bytes: 0,
            tick: 0,
            longest_entry: 0,
            hits: 0,
            misses: 0,
            oversize_reads: 0,
            evictions: 0,
            invalidations: 0,
        }
    }

    /// Acquires the cache lock, recovering the guard after poisoning.
    pub(crate) fn locked(lock: &std::sync::Mutex<Self>) -> std::sync::MutexGuard<'_, Self> {
        lock.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Serves a metadata read through the cache, calling `read` on a miss.
    ///
    /// Admission and hits share owned backing allocations. Borrowed bytes are returned without
    /// admission. The byte budget includes the backing vector's full capacity, and owners held
    /// outside the cache may keep that allocation alive after eviction.
    ///
    /// The lock is released before `read` and before returning the owner to a parser.
    ///
    /// # Errors
    ///
    /// Returns errors from `read`, and [`FormatError::Source`] if an eligible cache miss returns
    /// a byte count different from `len`.
    pub(crate) fn read_through<'a>(
        lock: &std::sync::Mutex<Self>,
        config: MetadataCacheConfig,
        offset: u64,
        len: usize,
        read: impl FnOnce() -> Result<MetadataBytes<'a>, FormatError>,
    ) -> Result<MetadataBytes<'a>, FormatError> {
        // Zero-length reads and a disabled cache bypass locking and counters.
        if len == 0 || !config.is_enabled() {
            return read();
        }
        if len > config.max_entry_bytes() || len > config.max_bytes() {
            Self::locked(lock).oversize_reads += 1;
            return read();
        }
        if let Some(bytes) = Self::locked(lock).get(offset, len) {
            return Ok(bytes);
        }
        let bytes = read()?;
        exact_metadata(offset, len, bytes.len())?;
        if let Some(shared) = bytes.shared() {
            if shared.retained_bytes() <= config.max_entry_bytes() {
                Self::locked(lock).insert_shared(offset, len, shared, config.max_bytes());
            }
        }
        Ok(bytes)
    }

    /// Returns cumulative counters and current occupancy.
    pub(crate) fn stats(&self) -> MetadataCacheStats {
        MetadataCacheStats {
            hits: self.hits,
            misses: self.misses,
            oversize_reads: self.oversize_reads,
            evictions: self.evictions,
            invalidations: self.invalidations,
            entries: self.entries.len(),
            bytes: self.current_bytes,
        }
    }

    /// Resets cumulative counters, retaining cached entries and occupancy counts.
    pub(crate) fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.oversize_reads = 0;
        self.evictions = 0;
        self.invalidations = 0;
    }

    /// Removes an entry and its access index, updating retained capacity.
    fn remove(&mut self, key: (u64, usize)) {
        if let Some(entry) = self.entries.remove(&key) {
            self.by_access.remove(&entry.last_access);
            self.current_bytes -= entry.bytes.retained_bytes();
        }
        debug_assert_eq!(
            self.entries.len(),
            self.by_access.len(),
            "every entry holds exactly one access row"
        );
    }

    /// Removes cached entries overlapping `[offset, offset + len)`.
    pub(crate) fn invalidate_overlapping(&mut self, offset: u64, len: usize) {
        if len == 0 {
            return;
        }
        let end = offset.saturating_add(len as u64);
        // Entries before this bound end before `offset`. The bound is at most `offset`,
        // and `end` is at least `offset`, as required by `BTreeMap::range`.
        let first = offset.saturating_sub(self.longest_entry as u64);
        let overlapping_keys: Vec<(u64, usize)> = self
            .entries
            .range((first, 0)..(end, 0))
            // The range checks `entry_offset < end`. Also require `entry_end > offset`.
            .filter(|((entry_offset, entry_len), _)| {
                entry_offset.saturating_add(*entry_len as u64) > offset
            })
            .map(|(key, _)| *key)
            .collect();
        self.invalidations += overlapping_keys.len() as u64;
        for key in overlapping_keys {
            self.remove(key);
        }
    }

    pub(crate) fn get(&mut self, offset: u64, len: usize) -> Option<MetadataBytes<'static>> {
        let key = (offset, len);
        let tick = self.tick + 1;
        let Some(entry) = self.entries.get_mut(&key) else {
            self.misses += 1;
            return None;
        };
        let previous = core::mem::replace(&mut entry.last_access, tick);
        let bytes = entry.bytes.clone();
        // Only past the lookup is this a hit, so only here does the clock move.
        self.tick = tick;
        self.hits += 1;
        self.by_access.remove(&previous);
        self.by_access.insert(tick, key);
        Some(bytes)
    }

    #[cfg(test)]
    pub(crate) fn insert(&mut self, offset: u64, len: usize, bytes: Vec<u8>, max_bytes: usize) {
        let bytes = MetadataBytes::from_vec(offset, bytes).unwrap();
        self.insert_shared(offset, len, bytes, max_bytes);
    }

    fn insert_shared(
        &mut self,
        offset: u64,
        len: usize,
        bytes: MetadataBytes<'static>,
        max_bytes: usize,
    ) {
        if len == 0 || bytes.retained_bytes() > max_bytes {
            return;
        }

        let key = (offset, len);
        self.remove(key);

        self.tick += 1;
        let tick = self.tick;
        self.longest_entry = self.longest_entry.max(len);
        self.current_bytes += bytes.retained_bytes();
        self.entries.insert(
            key,
            CachedMetadataRead {
                bytes,
                last_access: tick,
            },
        );
        self.by_access.insert(tick, key);
        debug_assert_eq!(
            self.entries.len(),
            self.by_access.len(),
            "every entry holds exactly one access row"
        );
        self.evict_to_budget(max_bytes);
    }

    fn evict_to_budget(&mut self, max_bytes: usize) {
        while self.current_bytes > max_bytes {
            let Some((_, &key)) = self.by_access.first_key_value() else {
                break;
            };
            // Count budget-driven removals as evictions. Invalidation and replacement use `remove`.
            self.evictions += 1;
            self.remove(key);
        }
    }
}

/// A [`Source`] wrapper with a bounded cache for metadata reads.
///
/// Metadata reads borrow resident bytes or share owned allocations through
/// [`Source::read_metadata`]. Only owned backing allocations are admitted. Eligible borrowed
/// reads are counted as misses. The owned adapter,
/// [`Source::read_metadata_at`], returns a vector. Raw dataset payloads use
/// [`Source::read_exact_at`] and bypass this cache.
///
/// The configured byte budget covers retained payload capacity. An owner held outside the cache
/// may keep an allocation alive after eviction. Separate reads do not form an atomic snapshot.
#[cfg(feature = "std")]
pub struct MetadataCachingSource<S> {
    inner: S,
    config: MetadataCacheConfig,
    cache: std::sync::Mutex<MetadataReadCache>,
}

#[cfg(feature = "std")]
impl<S> MetadataCachingSource<S> {
    /// Wraps a source with the supplied metadata-cache configuration.
    pub fn new(inner: S, config: MetadataCacheConfig) -> Self {
        Self {
            inner,
            config,
            cache: std::sync::Mutex::new(MetadataReadCache::new()),
        }
    }
}

#[cfg(feature = "std")]
impl<S: Source> Source for MetadataCachingSource<S> {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, buf)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.inner.read_exact_at(offset, len)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.read_metadata(offset, len)?.into_vec())
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        self.inner.metadata_origin()
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        MetadataReadCache::read_through(&self.cache, self.config, offset, len, || {
            self.inner.read_metadata(offset, len)
        })
    }

    /// Returns cache statistics, or `None` if the configuration disables caching.
    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.config
            .is_enabled()
            .then(|| MetadataReadCache::locked(&self.cache).stats())
    }

    fn reset_metadata_cache_stats(&self) {
        MetadataReadCache::locked(&self.cache).reset_stats();
    }
}

// ---------------------------------------------------------------------------
// Read + Seek backend (std)
// ---------------------------------------------------------------------------

/// A [`Source`] that seeks before each read from a [`std::io::Read`] and [`std::io::Seek`] backend.
///
/// [`Source::read_at`] reads directly into the caller's buffer. Access to the backend is serialized
/// with a [`std::sync::Mutex`].
#[cfg(feature = "std")]
pub struct ReadSeekSource<R> {
    inner: std::sync::Mutex<R>,
    len: u64,
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> ReadSeekSource<R> {
    /// Wraps a backend, measuring its length by seeking to the end.
    ///
    /// Leaves the cursor at the end. Each read seeks to its requested offset.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Source`] if seeking to the end fails.
    pub fn new(mut reader: R) -> Result<Self, FormatError> {
        let len = reader
            .seek(std::io::SeekFrom::End(0))
            .map_err(|e| FormatError::Source(format_io(&e)))?;
        Ok(ReadSeekSource {
            inner: std::sync::Mutex::new(reader),
            len,
        })
    }
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> Source for ReadSeekSource<R> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        // Check bounds before reading so a request past EOF returns an error.
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > self.len {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end,
                available: self.len,
            });
        }
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .seek(std::io::SeekFrom::Start(offset))
            .map_err(|e| FormatError::Source(format_io(&e)))?;
        guard
            .read_exact(buf)
            .map_err(|e| FormatError::Source(format_io(&e)))?;
        Ok(())
    }
}

/// Formats an [`std::io::Error`] for [`FormatError::Source`].
#[cfg(feature = "std")]
fn format_io(e: &std::io::Error) -> std::string::String {
    std::format!("{e}")
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::string::ToString;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    use core::cell::Cell;
    #[cfg(feature = "std")]
    use std::io;
    #[cfg(feature = "std")]
    use std::io::{Read, Seek, SeekFrom};

    use hdf5_pure_core::PhysicalAddress;
    #[cfg(feature = "std")]
    use rstest::rstest;

    use super::*;

    #[test]
    fn metadata_read_preserves_legacy_override_and_vector_allocation() {
        struct LegacySource {
            payload: Cell<Option<Vec<u8>>>,
            request: Cell<Option<(u64, usize)>>,
        }
        impl Source for LegacySource {
            fn len(&self) -> u64 {
                16
            }
            fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FormatError> {
                Err(FormatError::Source("unexpected payload read".into()))
            }
            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.request.set(Some((offset, len)));
                Ok(self.payload.take().unwrap())
            }
        }
        let payload = vec![1, 2, 3];
        let pointer = payload.as_ptr();
        let source = LegacySource {
            payload: Cell::new(Some(payload)),
            request: Cell::new(None),
        };
        let bytes = source.read_metadata(4, 3).unwrap();
        assert_eq!(bytes.as_ref(), &[1, 2, 3]);
        assert_eq!(bytes.offset(), 4);
        assert_eq!(bytes.as_ptr(), pointer);
        let owned = bytes.into_vec();
        assert_eq!(owned.as_ptr(), pointer);
        assert_eq!(source.request.get(), Some((4, 3)));
    }

    #[test]
    fn framed_metadata_preserves_borrow_and_resolves_nested_origins_once() {
        let data = [1, 2, 3, 4, 5, 6];
        let source = BytesSource::new(data.as_slice());
        let first = BaseOffsetSource {
            inner: &source,
            base: BaseAddress::new(1),
        };
        let second = BaseOffsetSource {
            inner: &first,
            base: BaseAddress::new(2),
        };
        let bytes = second.read_metadata(1, 2).unwrap();
        assert_eq!(second.metadata_origin().unwrap(), 3);
        assert_eq!(bytes.offset(), 4);
        assert_eq!(bytes.as_ref(), &[5, 6]);
        assert_eq!(bytes.as_ptr(), data[4..].as_ptr());
        let legacy =
            legacy_framed_region(FileRange::new(PhysicalAddress::new(1), 2).unwrap(), &bytes)
                .unwrap();
        assert_eq!(legacy.range().start().get(), 1);
        assert_eq!(legacy.bytes(), &[5, 6]);
        let frame = MetadataFrame::new(PhysicalAddress::new(3), &data[3..]).unwrap();
        let framed = frame
            .read_range(FileRange::new(PhysicalAddress::new(1), 2).unwrap())
            .unwrap();
        assert_eq!(framed.offset(), bytes.offset());
        assert_eq!(framed.as_ptr(), bytes.as_ptr());
    }

    #[cfg(feature = "std")]
    #[test]
    fn borrowed_metadata_owners_are_returned_without_admission() {
        let data = [1, 2, 3, 4];
        let source = MetadataCachingSource::new(
            BytesSource::new(data.as_slice()),
            MetadataCacheConfig::new(16),
        );
        let first = source.read_metadata(1, 2).unwrap();
        let second = source.read_metadata(1, 2).unwrap();
        assert_eq!(first.as_ref(), &[2, 3]);
        assert_eq!(second.as_ref(), first.as_ref());
        assert_eq!(first.as_ptr(), data[1..].as_ptr());
        assert_eq!(second.as_ptr(), first.as_ptr());
        assert_eq!(first.offset(), 1);
        assert_eq!(second.offset(), 1);
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(
            stats,
            MetadataCacheStats {
                hits: 0,
                misses: 2,
                oversize_reads: 0,
                evictions: 0,
                invalidations: 0,
                entries: 0,
                bytes: 0,
            }
        );
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[rstest]
    #[case::one_mib(
        1_048_576,
        "borrowed_metadata_one_mib",
        "borrowed_metadata_one_mib_control"
    )]
    #[case::four_mib(
        4_194_304,
        "borrowed_metadata_four_mib",
        "borrowed_metadata_four_mib_control"
    )]
    fn borrowed_metadata_reads_bypass_retention_without_allocating(
        #[case] length: usize,
        #[case] region: &str,
        #[case] control: &str,
    ) {
        let data = vec![7; length + 1];
        let source = MetadataCachingSource::new(
            BytesSource::new(data.as_slice()),
            MetadataCacheConfig::new(length * 2).with_max_entry_bytes(length),
        );
        assert_eq!(
            source.metadata_cache_stats().unwrap(),
            MetadataCacheStats::default()
        );
        let expected = &data[1..];
        let (copy, positive) =
            test_util::allocation::measure(control, || std::hint::black_box(expected.to_vec()));
        assert_eq!(copy, expected);
        assert!(positive.blocks > 0, "{positive}");
        assert!(positive.bytes >= length as u64, "{positive}");
        drop(copy);
        let ((first, second), measured) = test_util::allocation::measure(region, || {
            (
                source.read_metadata(1, length).unwrap(),
                source.read_metadata(1, length).unwrap(),
            )
        });
        assert_eq!(first.as_ref(), expected);
        assert_eq!(second.as_ref(), expected);
        assert_eq!(first.as_ptr(), expected.as_ptr());
        assert_eq!(second.as_ptr(), expected.as_ptr());
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(measured.bytes, 0, "{measured}");
        assert_eq!(measured.peak_bytes, 0, "{measured}");
        assert_eq!(measured.live_bytes, 0, "{measured}");
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(
            (stats.hits(), stats.misses(), stats.entries(), stats.bytes()),
            (0, 2, 0, 0)
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn metadata_cache_admission_and_forwarding_share_the_payload() {
        let source = MetadataCachingSource::new(
            ReadSeekSource::new(std::io::Cursor::new(vec![1, 2, 3, 4])).unwrap(),
            MetadataCacheConfig::new(16),
        );
        let cold = source.read_metadata(1, 2).unwrap();
        let shared: &dyn Source = &source;
        let boxed: Box<dyn Source + '_> = Box::new(shared);
        let warm = boxed.read_metadata(1, 2).unwrap();
        assert_eq!(cold.as_ref(), &[2, 3]);
        assert_eq!(warm.as_ref(), cold.as_ref());
        assert_eq!(cold.as_ptr(), warm.as_ptr());
        assert_eq!(warm.offset(), 1);
        drop(source.cache.try_lock().unwrap());
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(
            (stats.hits(), stats.misses(), stats.bytes(), stats.entries()),
            (1, 1, 2, 1)
        );
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[rstest]
    #[case::one_mib(
        1_048_576,
        "shared_metadata_cold_one",
        "shared_metadata_warm_one",
        "shared_metadata_control_one"
    )]
    #[case::four_mib(
        4_194_304,
        "shared_metadata_cold_four",
        "shared_metadata_warm_four",
        "shared_metadata_control_four"
    )]
    fn shared_metadata_admission_and_hits_allocate_no_second_payload(
        #[case] length: usize,
        #[case] cold_region: &str,
        #[case] warm_region: &str,
        #[case] control_region: &str,
    ) {
        let expected = vec![23; length];
        let source = MetadataCachingSource::new(
            ReadSeekSource::new(io::Cursor::new(expected.clone())).unwrap(),
            MetadataCacheConfig::new(length).with_max_entry_bytes(length),
        );
        let (copy, control) = test_util::allocation::measure(control_region, || {
            std::hint::black_box(expected.clone())
        });
        assert_eq!(copy, expected);
        assert!(control.blocks > 0, "{control}");
        assert!(control.bytes >= length as u64, "{control}");
        drop(copy);
        let (cold, admission) = test_util::allocation::measure(cold_region, || {
            source.read_metadata(0, length).unwrap()
        });
        assert_eq!(cold.as_ref(), expected);
        assert!(admission.bytes >= length as u64, "{admission}");
        assert!(admission.bytes < length as u64 + 16_384, "{admission}");
        assert!(admission.peak_bytes < length as u64 + 16_384, "{admission}");
        let ((first, second), warm) = test_util::allocation::measure(warm_region, || {
            (
                source.read_metadata(0, length).unwrap(),
                source.read_metadata(0, length).unwrap(),
            )
        });
        assert_eq!(first.as_ref(), expected);
        assert_eq!(second.as_ref(), expected);
        assert_eq!(first.as_ptr(), cold.as_ptr());
        assert_eq!(second.as_ptr(), cold.as_ptr());
        assert!(warm.bytes < 16_384, "{warm}");
        assert!(warm.peak_bytes < 16_384, "{warm}");
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(
            (stats.hits(), stats.misses(), stats.entries(), stats.bytes()),
            (2, 1, 1, length)
        );
        eprintln!("payload={length} admission={admission}; warm={warm}; control={control}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn evicted_metadata_owners_retain_their_bytes_without_retaining_cache_entries() {
        let source = MetadataCachingSource::new(
            ReadSeekSource::new(io::Cursor::new(b"abcdefgh".to_vec())).unwrap(),
            MetadataCacheConfig::new(4),
        );
        let first = source.read_metadata(0, 4).unwrap().shared().unwrap();
        let second = source.read_metadata(4, 4).unwrap().shared().unwrap();
        assert_eq!(first.as_ref(), b"abcd");
        assert_eq!(second.as_ref(), b"efgh");
        assert_eq!((first.offset(), second.offset()), (0, 4));
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(
            (
                stats.misses(),
                stats.evictions(),
                stats.entries(),
                stats.bytes()
            ),
            (2, 1, 1, 4)
        );
        let reloaded = source.read_metadata(0, 4).unwrap().shared().unwrap();
        assert_eq!(reloaded.as_ref(), first.as_ref());
        assert_ne!(reloaded.as_ptr(), first.as_ptr());
        assert_ne!(reloaded.as_ptr(), second.as_ptr());
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(
            (
                stats.hits(),
                stats.misses(),
                stats.evictions(),
                stats.bytes()
            ),
            (0, 3, 2, 4)
        );
        drop(source);
        assert_eq!(first.as_ref(), b"abcd");
        assert_eq!(second.as_ref(), b"efgh");
        assert_eq!(reloaded.as_ref(), b"abcd");
    }

    #[cfg(feature = "std")]
    #[test]
    fn cached_metadata_keeps_high_physical_origins_across_nested_frames() {
        struct PositionedSource {
            last_read: Cell<Option<u64>>,
        }

        impl Source for PositionedSource {
            fn len(&self) -> u64 {
                65_536
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.last_read.set(Some(offset));
                buf.fill(37);
                Ok(())
            }

            fn metadata_origin(&self) -> Result<u64, FormatError> {
                Ok(0x1_0000_0000)
            }
        }

        let source = MetadataCachingSource::new(
            ValidatedSource::new(PositionedSource {
                last_read: Cell::new(None),
            }),
            MetadataCacheConfig::new(32),
        );
        let outer = BaseOffsetSource {
            inner: &source,
            base: BaseAddress::new(512),
        };
        let inner = BaseOffsetSource {
            inner: &outer,
            base: BaseAddress::new(4096),
        };
        let range = FileRange::new(PhysicalAddress::new(17), 4).unwrap();
        let first = inner.read_range(range).unwrap();
        let second = inner.read_range(range).unwrap();
        assert_eq!(first.as_ref(), &[37; 4]);
        assert_eq!(second.as_ref(), first.as_ref());
        assert_eq!(first.offset(), 0x1_0000_1211);
        assert_eq!(second.offset(), first.offset());
        assert_eq!(first.as_ptr(), second.as_ptr());
        assert_eq!(source.inner.0.last_read.get(), Some(4625));
        let region = first.region();
        assert_eq!(region.range().start().get(), 0x1_0000_1211);
        let legacy = legacy_framed_region(range, &first).unwrap();
        assert_eq!(legacy.range(), range);
        assert_eq!(legacy.bytes().as_ptr(), first.as_ptr());
        drop(source.cache.try_lock().unwrap());
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!((stats.hits(), stats.misses(), stats.bytes()), (1, 1, 4));
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::exact_backing(4, 4, 1, 4)]
    #[case::retained_capacity_exceeds_budget(32, 4, 0, 0)]
    #[case::subregion_retains_the_larger_allocation(8, 2, 0, 0)]
    fn metadata_cache_admission_accounts_for_retained_backing(
        #[case] capacity: usize,
        #[case] len: usize,
        #[case] entries: usize,
        #[case] retained: usize,
    ) {
        let cache = std::sync::Mutex::new(MetadataReadCache::new());
        let mut payload = Vec::with_capacity(capacity);
        payload.resize(capacity.min(8), 1);
        let owner = MetadataBytes::from_vec(0, payload).unwrap();
        let owner = owner
            .subregion(FileRange::new(PhysicalAddress::new(0), len as u64).unwrap())
            .unwrap();
        let pointer = owner.as_ptr();
        let bytes =
            MetadataReadCache::read_through(&cache, MetadataCacheConfig::new(4), 0, len, || {
                Ok(owner)
            })
            .unwrap();
        assert_eq!(bytes.as_ref(), vec![1; len]);
        assert_eq!(bytes.as_ptr(), pointer);
        let stats = MetadataReadCache::locked(&cache).stats();
        assert_eq!(
            (stats.entries(), stats.bytes(), stats.misses()),
            (entries, retained, 1)
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn metadata_cache_releases_its_lock_during_a_miss() {
        let cache = std::sync::Mutex::new(MetadataReadCache::new());
        let bytes =
            MetadataReadCache::read_through(&cache, MetadataCacheConfig::new(16), 1, 2, || {
                drop(cache.try_lock().unwrap());
                MetadataBytes::from_vec(1, vec![2, 3])
            })
            .unwrap();
        assert_eq!(bytes.as_ref(), &[2, 3]);
        drop(cache.try_lock().unwrap());
        assert_eq!(
            MetadataReadCache::locked(&cache)
                .get(1, 2)
                .unwrap()
                .as_ptr(),
            bytes.as_ptr()
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::short(2, 0)]
    #[case::long(4, 0)]
    #[case::wrong_origin(3, 1)]
    fn validated_metadata_rejects_mismatched_lengths_and_origins(
        #[case] len: usize,
        #[case] origin: u64,
    ) {
        struct MismatchedMetadataSource {
            len: usize,
            origin: u64,
        }
        impl Source for MismatchedMetadataSource {
            fn len(&self) -> u64 {
                16
            }
            fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FormatError> {
                Err(FormatError::Source("unexpected payload read".into()))
            }
            fn read_metadata(
                &self,
                _offset: u64,
                _len: usize,
            ) -> Result<MetadataBytes<'_>, FormatError> {
                MetadataBytes::from_vec(self.origin, vec![0; self.len])
            }
        }
        let source = ValidatedSource::new(MismatchedMetadataSource { len, origin });
        let err = source.read_metadata(0, 3).unwrap_err();
        let expected = if len != 3 {
            format!("the source returned {len} bytes for a 3-byte read at offset 0")
        } else {
            "the source returned metadata at physical offset 1 for a read at physical offset 0"
                .into()
        };
        assert_eq!(err, FormatError::Source(expected));
    }

    #[test]
    fn bytes_source_reads_and_reports_len() {
        let data = (0u8..=255).collect::<Vec<u8>>();
        let src = BytesSource::new(data.clone());
        assert_eq!(src.len(), 256);

        let mut buf = [0u8; 4];
        src.read_at(10, &mut buf).unwrap();
        assert_eq!(buf, [10, 11, 12, 13]);

        let owned = src.read_exact_at(250, 6).unwrap();
        assert_eq!(owned, vec![250, 251, 252, 253, 254, 255]);
    }

    #[test]
    fn bytes_source_short_read_is_eof() {
        let src = BytesSource::new(vec![1u8, 2, 3]);
        let mut buf = [0u8; 4];
        let err = src.read_at(0, &mut buf).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedSourceEof {
                expected: 4,
                available: 3
            }
        );
        // Reading exactly to the end is fine.
        let mut ok = [0u8; 3];
        src.read_at(0, &mut ok).unwrap();
        assert_eq!(ok, [1, 2, 3]);
    }

    #[test]
    fn bytes_source_reads_at_end_require_an_empty_buffer() {
        let src = BytesSource::new(vec![0u8; 8]);
        let mut buf = [0u8; 1];
        assert_eq!(
            src.read_at(8, &mut buf).unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: 9,
                available: 8
            }
        );
        // Zero-length read at EOF succeeds.
        src.read_at(8, &mut []).unwrap();
    }

    #[test]
    fn read_exact_at_rejects_oversized_len_without_allocating() {
        let src = BytesSource::new(vec![1u8, 2, 3, 4]);
        assert_eq!(
            src.read_exact_at(0, usize::MAX).unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: usize::MAX.to_u64(),
                available: 4
            }
        );
        assert_eq!(src.read_exact_at(1, 3).unwrap(), vec![2, 3, 4]);
    }

    #[rstest::rstest]
    #[case::small(2, 3, 4, 5)]
    #[case::large(0x1_0000_00f0, 64, 0x1_0000_0100, 4_294_967_600)]
    fn source_preflight_preserves_positions_without_backend_reads(
        #[case] offset: u64,
        #[case] len: usize,
        #[case] available: u64,
        #[case] expected: u64,
    ) {
        let source = ReadCountingSource {
            extent: available,
            reads: Cell::new(0),
        };
        let error = FormatError::UnexpectedSourceEof {
            expected,
            available,
        };
        assert_eq!(source.read_exact_at(offset, len).unwrap_err(), error);
        assert_eq!(source.reads.get(), 0);
        let range = FileRange::new(PhysicalAddress::new(offset), len.to_u64()).unwrap();
        assert_eq!(unexpected_eof(range, available), error);
        assert_eq!(
            error.to_string(),
            alloc::format!(
                "unexpected source EOF: need end position {expected}, source extent is {available}"
            )
        );
    }

    #[rstest::rstest]
    #[case::small(4, 2, FormatError::UnexpectedSourceEof { expected: 6, available: 5 })]
    #[case::large(0x1_0000_00f0, 64, FormatError::UnexpectedSourceEof { expected: 4_294_967_600, available: 5 })]
    #[case::overflow(u64::MAX, 1, FormatError::OffsetOverflow { offset: u64::MAX, length: 1 })]
    fn buffered_source_bounds_preserve_the_requested_position(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] expected: FormatError,
    ) {
        let bytes = [0u8; 5];
        assert_eq!(
            BytesSource::new(bytes)
                .read_at(offset, &mut vec![0; length])
                .unwrap_err(),
            expected
        );
        let error = FileRange::new(PhysicalAddress::new(offset), length.to_u64())
            .and_then(|range| bytes_in(&bytes, range));
        assert_eq!(error.unwrap_err(), expected);
    }

    #[test]
    fn source_preflight_reports_address_overflow_without_backend_reads() {
        assert_eq!(
            ReadCountingSource {
                extent: u64::MAX,
                reads: Cell::new(0)
            }
            .read_exact_at(u64::MAX, 1)
            .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        );
    }

    struct ReadCountingSource {
        extent: u64,
        reads: Cell<u32>,
    }

    impl Source for ReadCountingSource {
        fn len(&self) -> u64 {
            self.extent
        }
        fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.set(self.reads.get() + 1);
            Err(FormatError::Internal("unexpected backend read".into()))
        }
    }

    #[cfg(feature = "std")]
    #[rstest::rstest]
    #[case::past_extent(0x1_0000_00f0, 64, FormatError::UnexpectedSourceEof { expected: 4_294_967_600, available: 4_294_967_552 })]
    #[case::overflow(u64::MAX, 1, FormatError::OffsetOverflow { offset: u64::MAX, length: 1 })]
    fn seek_source_preflight_preserves_positions_without_backend_access(
        #[case] offset: u64,
        #[case] len: usize,
        #[case] expected: FormatError,
    ) {
        let source = ReadSeekSource::new(ReadSeekRecordingBackend {
            reads: 0,
            seeks: Vec::new(),
        })
        .unwrap();
        assert_eq!(source.len(), 4_294_967_552);
        assert_eq!(
            source.read_at(offset, &mut vec![0; len]).unwrap_err(),
            expected
        );
        let backing = source.inner.lock().unwrap();
        assert_eq!(backing.reads, 0);
        assert_eq!(backing.seeks, vec![SeekFrom::End(0)]);
    }

    #[cfg(feature = "std")]
    struct ReadSeekRecordingBackend {
        reads: u32,
        seeks: Vec<SeekFrom>,
    }

    #[cfg(feature = "std")]
    impl Read for ReadSeekRecordingBackend {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            self.reads += 1;
            Err(io::Error::other("unexpected backend read"))
        }
    }

    #[cfg(feature = "std")]
    impl Seek for ReadSeekRecordingBackend {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.seeks.push(position);
            Ok(4_294_967_552)
        }
    }

    #[test]
    fn source_references_forward_payload_reads() {
        let src = BytesSource::new(vec![9u8, 8, 7]);
        let r: &dyn Source = &src;
        let mut buf = [0u8; 2];
        r.read_at(1, &mut buf).unwrap();
        assert_eq!(buf, [8, 7]);
    }

    #[test]
    fn source_references_forward_metadata_reads() {
        use core::cell::Cell;

        struct MetadataSource {
            metadata_reads: Cell<usize>,
        }

        impl Source for MetadataSource {
            fn len(&self) -> u64 {
                16
            }

            fn read_at(&self, _offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                buf.fill(0);
                Ok(())
            }

            fn read_metadata_at(&self, _offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.metadata_reads.set(self.metadata_reads.get() + 1);
                Ok(vec![0xAB; len])
            }
        }

        fn read_metadata_via_trait<T: Source>(source: T) -> Vec<u8> {
            source.read_metadata_at(4, 3).unwrap()
        }

        let source = MetadataSource {
            metadata_reads: Cell::new(0),
        };

        assert_eq!(read_metadata_via_trait(&source), vec![0xAB; 3]);
        assert_eq!(source.metadata_reads.get(), 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn metadata_cache_caches_only_metadata_reads() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        struct CountingSource {
            data: Vec<u8>,
            reads: Arc<AtomicUsize>,
        }

        impl Source for CountingSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.reads.fetch_add(1, Ordering::SeqCst);
                BytesSource::new(&self.data).read_at(offset, buf)
            }
        }

        let reads = Arc::new(AtomicUsize::new(0));
        let source = MetadataCachingSource::new(
            CountingSource {
                data: (0u8..16).collect(),
                reads: Arc::clone(&reads),
            },
            MetadataCacheConfig::new(16),
        );

        assert_eq!(source.read_metadata_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(source.read_metadata_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(reads.load(Ordering::SeqCst), 1);

        assert_eq!(source.read_exact_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(source.read_exact_at(4, 4).unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(reads.load(Ordering::SeqCst), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_seek_source_matches_in_memory() {
        use std::io::Cursor;
        let data = (0u8..200).collect::<Vec<u8>>();
        let mem = BytesSource::new(data.clone());
        let seek = ReadSeekSource::new(Cursor::new(data.clone())).unwrap();
        assert_eq!(seek.len(), mem.len());

        for &(off, len) in &[(0u64, 1usize), (5, 10), (199, 1), (100, 50)] {
            let a = mem.read_exact_at(off, len).unwrap();
            let b = seek.read_exact_at(off, len).unwrap();
            assert_eq!(a, b, "mismatch at offset {off} len {len}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_seek_source_past_end_is_error() {
        use std::io::Cursor;
        let seek = ReadSeekSource::new(Cursor::new(vec![1u8, 2, 3, 4])).unwrap();
        let mut buf = [0u8; 3];
        assert_eq!(
            seek.read_at(2, &mut buf).unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: 5,
                available: 4
            }
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_seek_source_is_sync() {
        // Compile-time assertion that the std backend is Send + Sync so it can
        // back a parallel reader.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ReadSeekSource<std::io::Cursor<Vec<u8>>>>();
    }

    // -----------------------------------------------------------------------
    // The bounded metadata store (issue #367)
    // -----------------------------------------------------------------------

    #[test]
    fn cache_eviction_uses_last_access_order() {
        // Room for three ten-byte entries, so the fourth displaces exactly one.
        let budget = 30;
        let mut cache = MetadataReadCache::new();
        cache.insert(0, 10, vec![0u8; 10], budget);
        cache.insert(100, 10, vec![1u8; 10], budget);
        cache.insert(200, 10, vec![2u8; 10], budget);

        // Reading the first entry makes the *second* the least recently used,
        // which is what separates an LRU from a queue.
        assert!(cache.get(0, 10).is_some());
        cache.insert(300, 10, vec![3u8; 10], budget);

        assert!(
            cache.get(0, 10).is_some(),
            "read most recently, must survive"
        );
        assert!(cache.get(100, 10).is_none(), "least recently used, must go");
        assert!(cache.get(200, 10).is_some());
        assert!(cache.get(300, 10).is_some());
    }

    #[test]
    fn cache_invalidation_removes_overlapping_entries() {
        let budget = 1024;
        let mut cache = MetadataReadCache::new();
        for offset in [0u64, 10, 20, 30] {
            cache.insert(offset, 10, vec![offset as u8; 10], budget);
        }
        // The [5, 45) entry starts before [20, 25) and overlaps it.
        cache.insert(5, 40, vec![9u8; 40], budget);

        cache.invalidate_overlapping(20, 5);

        assert!(cache.get(0, 10).is_some(), "ends at 10, short of the write");
        assert!(
            cache.get(10, 10).is_some(),
            "ends exactly where the write starts, so it shares no byte with it"
        );
        assert!(cache.get(20, 10).is_none(), "the write lands inside it");
        assert!(cache.get(30, 10).is_some(), "starts after the write ends");
        assert!(
            cache.get(5, 40).is_none(),
            "starts before the write and spans it, so a search beginning at the \
             write's own offset would walk straight past it"
        );
    }

    #[test]
    fn cache_replacement_updates_payload_and_preserves_occupancy() {
        // Exactly two ten-byte entries fit.
        let budget = 20;
        let mut cache = MetadataReadCache::new();
        cache.insert(0, 10, vec![0u8; 10], budget);
        cache.insert(0, 10, vec![1u8; 10], budget);
        cache.insert(100, 10, vec![2u8; 10], budget);

        assert_eq!(
            cache.get(0, 10).as_deref(),
            Some(&[1u8; 10][..]),
            "the later value replaces the earlier one"
        );
        assert!(
            cache.get(100, 10).is_some(),
            "a replacement that was counted twice would have evicted to make room"
        );
    }

    #[test]
    fn one_offset_at_two_lengths_holds_two_entries() {
        let budget = 1024;
        let mut cache = MetadataReadCache::new();
        cache.insert(64, 4, vec![1u8; 4], budget);
        cache.insert(64, 8, vec![2u8; 8], budget);

        assert_eq!(cache.get(64, 4).as_deref(), Some(&[1u8; 4][..]));
        assert_eq!(cache.get(64, 8).as_deref(), Some(&[2u8; 8][..]));
    }

    // -----------------------------------------------------------------------
    // What the cache reports about itself (issue #353)
    // -----------------------------------------------------------------------

    /// Creates a seek source whose byte at each offset is that offset truncated to [`u8`].
    #[cfg(feature = "std")]
    fn ramp(len: usize) -> ReadSeekSource<io::Cursor<Vec<u8>>> {
        ReadSeekSource::new(io::Cursor::new(
            (0..len).map(|i| i as u8).collect::<Vec<u8>>(),
        ))
        .unwrap()
    }

    #[cfg(feature = "std")]
    #[test]
    fn oversized_metadata_reads_increment_the_oversize_counter() {
        // The 64-byte entry cap excludes both 256-byte reads.
        let config = MetadataCacheConfig::new(4096).with_max_entry_bytes(64);
        let source = MetadataCachingSource::new(ramp(4096), config);

        source.read_metadata_at(0, 64).unwrap(); // miss, then admitted
        source.read_metadata_at(0, 64).unwrap(); // hit
        source.read_metadata_at(128, 256).unwrap(); // too large to admit
        source.read_metadata_at(128, 256).unwrap(); // and so, still too large

        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(stats.hits(), 1);
        assert_eq!(stats.misses(), 1);
        assert_eq!(stats.oversize_reads(), 2);
        assert_eq!(stats.reads(), 4);
        assert_eq!(stats.hit_rate(), Some(0.5));

        // The total byte budget is smaller than the per-entry cap.
        let budget_below_entry_cap = MetadataCacheConfig::new(128).with_max_entry_bytes(512);
        let source = MetadataCachingSource::new(ramp(4096), budget_below_entry_cap);
        source.read_metadata_at(0, 256).unwrap();
        source.read_metadata_at(0, 256).unwrap();
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(stats.oversize_reads(), 2);
        assert_eq!(stats.misses(), 0);
        assert_eq!(stats.hit_rate(), None);
    }

    #[cfg(feature = "std")]
    #[test]
    fn cache_hit_rate_requires_an_eligible_read() {
        let config = MetadataCacheConfig::new(4096).with_max_entry_bytes(64);
        let source = MetadataCachingSource::new(ramp(4096), config);

        let fresh = source.metadata_cache_stats().unwrap();
        assert_eq!(fresh.hit_rate(), None, "nothing has been read");

        source.read_metadata_at(0, 256).unwrap();
        assert_eq!(
            source.metadata_cache_stats().unwrap().hit_rate(),
            None,
            "a read the cache never saw does not make a rate out of it"
        );

        source.read_metadata_at(0, 64).unwrap();
        assert_eq!(
            source.metadata_cache_stats().unwrap().hit_rate(),
            Some(0.0),
            "one eligible read that missed *is* a rate, and the opposite reading"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn cache_counts_evictions_and_invalidations_separately() {
        // Two 64-byte entries fit. A third forces one out.
        const BUDGET: usize = 128;
        let mut cache = MetadataReadCache::new();
        cache.insert(0, 64, vec![1u8; 64], BUDGET);
        cache.insert(64, 64, vec![2u8; 64], BUDGET);

        cache.insert(0, 64, vec![1u8; 64], BUDGET);
        let replaced = cache.stats();
        assert_eq!(replaced.entries(), 2);
        assert_eq!(replaced.evictions(), 0);
        assert_eq!(replaced.invalidations(), 0);

        cache.insert(128, 64, vec![3u8; 64], BUDGET);
        let evicted = cache.stats();
        assert_eq!(evicted.evictions(), 1, "the budget forced this one");
        assert_eq!(evicted.invalidations(), 0);
        // The least recently used of the three went, leaving [0, 64) and
        // [128, 192).
        assert_eq!(evicted.entries(), 2);

        // A write across [32, 160) reaches into both survivors: one starts
        // before it, the other after.
        cache.invalidate_overlapping(32, 128);
        let invalidated = cache.stats();
        assert_eq!(invalidated.invalidations(), 2, "the write overlapped both");
        assert_eq!(
            invalidated.evictions(),
            1,
            "a write is not the budget, and a caller told to raise the budget \
             because of one would be raising it for nothing"
        );
        assert_eq!(invalidated.entries(), 0);
        assert_eq!(invalidated.bytes(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn resetting_the_counters_keeps_the_entries() {
        const BUDGET: usize = 4096;
        let mut cache = MetadataReadCache::new();
        cache.insert(0, 64, vec![1u8; 64], BUDGET);
        assert!(cache.get(0, 64).is_some());
        assert!(cache.get(512, 64).is_none());

        cache.reset_stats();

        let stats = cache.stats();
        assert_eq!(stats.hits(), 0);
        assert_eq!(stats.misses(), 0);
        assert_eq!(stats.hit_rate(), None);
        assert_eq!(stats.entries(), 1);
        assert_eq!(stats.bytes(), 64);
        assert!(cache.get(0, 64).is_some(), "the entry is still servable");
    }

    #[cfg(feature = "std")]
    #[test]
    fn disabled_cache_statistics_are_none() {
        let source = MetadataCachingSource::new(ramp(4096), MetadataCacheConfig::disabled());
        assert_eq!(
            source.read_metadata_at(0, 64).unwrap(),
            (0..64u8).collect::<Vec<u8>>()
        );
        assert_eq!(
            source.metadata_cache_stats(),
            None,
            "an all-zero snapshot would read as a cache that is on and idle"
        );
        source.reset_metadata_cache_stats();
        assert_eq!(
            source.metadata_cache_stats(),
            None,
            "and resetting one there is nothing to reset does not conjure one"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn base_offset_source_forwards_cache_statistics_and_resets() {
        let config = MetadataCacheConfig::new(4096);
        let source = MetadataCachingSource::new(ramp(4096), config);
        let framed = BaseOffsetSource {
            inner: &source,
            base: BaseAddress::new(512),
        };
        framed.read_metadata_at(0, 64).unwrap();
        framed.read_metadata_at(0, 64).unwrap();

        let stats = framed.metadata_cache_stats().expect("forwarded");
        assert_eq!((stats.hits(), stats.misses()), (1, 1));
        assert_eq!(stats, source.metadata_cache_stats().unwrap());

        framed.reset_metadata_cache_stats();
        assert_eq!(source.metadata_cache_stats().unwrap().hits(), 0);
        assert_eq!(
            source.metadata_cache_stats().unwrap().entries(),
            1,
            "reset through the view is a reset of counters, not a flush"
        );
    }
}
