//! Random-access byte sources for the reader: the [`Source`] trait and its backends.
//!
//! [`BytesSource`] reads from a buffer that holds the whole file, and [`ReadSeekSource`] from a
//! reader that seeks. [`MetadataCachingSource`] wraps either with a bounded cache of the metadata
//! reads and passes a dataset's payload through to the inner source.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

use crate::address::BaseAddress;
use crate::convert::TryToUsize;
use crate::error::FormatError;

/// Default maximum size of one entry admitted to a streaming metadata cache.
pub const DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES: usize = 64 * 1024;

/// Initial metadata-cache settings for streaming file access.
///
/// This is the `hdf5-pure` counterpart to the memory-budget portion of HDF5's
/// `H5Pset_mdc_config`: it bounds the bytes retained for parsed metadata reads
/// while a file is opened through [`crate::File::open_streaming_with_options`].
/// Raw dataset payload reads use `Source::read_exact_at` and are not
/// admitted to this cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataCacheConfig {
    max_bytes: usize,
    max_entry_bytes: usize,
}

impl MetadataCacheConfig {
    /// Create a metadata cache with the given total byte budget.
    ///
    /// Individual cached reads are capped at
    /// `DEFAULT_METADATA_CACHE_MAX_ENTRY_BYTES` (64 KiB) by default so one large
    /// heap or index block cannot monopolize the cache. Use
    /// [`with_max_entry_bytes`](Self::with_max_entry_bytes) to change that.
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

    /// Disable metadata read caching.
    pub const fn disabled() -> Self {
        Self {
            max_bytes: 0,
            max_entry_bytes: 0,
        }
    }

    /// Set the maximum size of a single metadata read admitted to the cache.
    pub const fn with_max_entry_bytes(mut self, max_entry_bytes: usize) -> Self {
        self.max_entry_bytes = max_entry_bytes;
        self
    }

    /// Return the total metadata-cache byte budget.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Return the maximum size of one cached metadata entry.
    pub const fn max_entry_bytes(&self) -> usize {
        self.max_entry_bytes
    }

    /// Whether metadata read caching is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.max_bytes > 0 && self.max_entry_bytes > 0
    }
}

impl Default for MetadataCacheConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// What a file's metadata cache has done, and what it is holding.
///
/// Returned by [`crate::File::metadata_cache_stats`]. This is the `hdf5-pure`
/// counterpart to HDF5's `H5Fget_mdc_hit_rate` and `H5Fget_mdc_size`:
/// [`entries`](Self::entries) and [`bytes`](Self::bytes) are a point-in-time
/// view of occupancy, and the counters are cumulative since the file was
/// opened or since the last
/// [`reset_metadata_cache_stats`](crate::File::reset_metadata_cache_stats).
///
/// The reason to look is that [`MetadataCacheConfig`] is a budget chosen before
/// a single read has happened, and nothing else reports whether it was the
/// right one:
///
/// - [`hit_rate`](Self::hit_rate) says whether the cache is earning its memory.
/// - [`evictions`](Self::evictions) says whether the budget is the binding
///   constraint. A hit rate below expectations with no evictions is not a
///   budget problem, and raising it will not help.
/// - [`oversize_reads`](Self::oversize_reads) says whether
///   [`max_entry_bytes`](MetadataCacheConfig::max_entry_bytes) is turning reads
///   away before they reach the cache at all.
/// - [`invalidations`](Self::invalidations) says how much of the cache a
///   read-write session is throwing away with its own writes.
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
    /// Metadata reads served from the cache.
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Metadata reads eligible for the cache that were not in it.
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Metadata reads that bypassed the cache because they exceed
    /// [`MetadataCacheConfig::max_entry_bytes`] (or the whole budget).
    ///
    /// These are counted apart from [`misses`](Self::misses) rather than folded
    /// into them: the cache was never offered the read, so charging it as a miss
    /// would report a failure at work it could not have done. They still show up
    /// in [`reads`](Self::reads).
    pub const fn oversize_reads(&self) -> u64 {
        self.oversize_reads
    }

    /// Entries dropped to stay inside [`MetadataCacheConfig::max_bytes`].
    pub const fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Entries dropped because an in-place write overlapped them.
    ///
    /// Only a read-write session invalidates; this stays zero on a read-only
    /// open. Invalidations approaching [`misses`](Self::misses) mean the session
    /// is rewriting the metadata it is caching, and a larger budget will not
    /// change that.
    pub const fn invalidations(&self) -> u64 {
        self.invalidations
    }

    /// Entries currently held.
    ///
    /// Against [`bytes`](Self::bytes) this is the mean entry size, which is what
    /// says whether a few large reads are spending the budget; pair it with
    /// [`oversize_reads`](Self::oversize_reads) to see the ones already refused.
    pub const fn entries(&self) -> usize {
        self.entries
    }

    /// Bytes currently held, to compare against
    /// [`MetadataCacheConfig::max_bytes`].
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Every metadata read through this source: hits, misses, and reads too
    /// large to admit.
    ///
    /// The last of those three is not in [`hit_rate`](Self::hit_rate)'s
    /// denominator, so `hits() / reads()` is a different figure and a lower one.
    pub const fn reads(&self) -> u64 {
        self.hits
            .saturating_add(self.misses)
            .saturating_add(self.oversize_reads)
    }

    /// The fraction of *eligible* metadata reads served from the cache, or
    /// `None` before any eligible read has happened.
    ///
    /// `None` rather than C's `0.0`, which `H5Fget_mdc_hit_rate` also returns
    /// for a cache that has missed every access: the two mean opposite things to
    /// a caller deciding whether to raise the budget, and only one of them is a
    /// reason to.
    pub fn hit_rate(&self) -> Option<f64> {
        let eligible = self.hits.saturating_add(self.misses);
        if eligible == 0 {
            return None;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "a hit rate is a ratio; f64 holds these counts exactly far past any \
                      read count a process will reach"
        )]
        Some(self.hits as f64 / eligible as f64)
    }
}

/// A random-access, read-only source of the bytes of an HDF5 file.
///
/// Offsets are `u64` (HDF5's native address width); lengths of individual reads
/// are `usize` (they must fit in a caller-provided buffer). Implementations must
/// either fill the whole request or return an error — a short read is always an
/// error, never silently truncated.
///
/// # Implementing one
///
/// Two methods carry the whole trait: [`len`](Source::len) and
/// [`read_at`](Source::read_at). The rest are conveniences with default bodies,
/// and every method added later will have one too, so an implementation written
/// today keeps compiling.
///
/// The reader treats a source as an immutable file for as long as it holds one:
/// [`len`](Source::len) is expected to stay put and to be true — the reader
/// bounds allocations with it — and bytes already read are expected to read
/// back the same. A source over something that grows underneath it — a file a
/// writer is appending to — is what [`File::open_swmr`](crate::File::open_swmr)
/// is for instead. `read_at` takes `&self` so a `File` can be shared, so an
/// implementation over a mutable handle owns its own synchronisation, the way
/// [`ReadSeekSource`] wraps its reader in a mutex.
///
/// Leave the last three defaulted unless you mean to cache.
/// [`read_metadata_at`](Source::read_metadata_at),
/// [`metadata_cache_stats`](Source::metadata_cache_stats) and
/// [`reset_metadata_cache_stats`](Source::reset_metadata_cache_stats) are the
/// seam this crate's own bounded metadata cache hangs on, and they come as a
/// set: overriding the first without the other two reports *no* cache where
/// there is a full one. To have metadata reads cached, ask for it with a
/// [`MetadataCacheConfig`] through
/// [`File::from_source_with_options`](crate::File::from_source_with_options),
/// which wraps the source in an implementation of all three.
#[expect(
    clippy::len_without_is_empty,
    reason = "`len` here is a file's byte length, not a container's count — the shape of \
              `std::fs::Metadata::len`, which ships without an `is_empty` for the same \
              reason. An HDF5 file is never empty (the signature alone is eight bytes), so \
              an `is_empty` on this trait would be a public method with no caller, and one \
              an implementation could contradict its own `len` with"
)]
pub trait Source {
    /// Total number of bytes the source can supply.
    ///
    /// Report the true length: the reader bounds allocations with it. Metadata
    /// lengths come out of the file being read, and
    /// [`read_exact_at`](Source::read_exact_at) rejects one that runs past the
    /// end *before* reserving a buffer for it, so a malformed file cannot name
    /// a multi-gigabyte read and have it reserved. A `len` larger than what the
    /// source can actually serve passes that check and the reservation happens,
    /// which leaves the file choosing the size of an allocation. A length that
    /// arrives over a channel the caller does not control — a `Content-Length`
    /// header, a size a host reports — should be held to what the source can
    /// really serve before it is reported here.
    fn len(&self) -> u64;

    /// Read exactly `buf.len()` bytes starting at absolute offset `offset`,
    /// filling `buf`.
    ///
    /// Returns [`FormatError::UnexpectedEof`] if fewer than `buf.len()` bytes are
    /// available at `offset`, [`FormatError::OffsetOverflow`] if
    /// `offset + buf.len()` overflows, [`FormatError::ValueTooLargeForPlatform`]
    /// if `offset` does not fit this platform's `usize` (for in-memory
    /// backends), or [`FormatError::Source`] for a backend I/O failure.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError>;

    /// Read `len` bytes starting at `offset` into a freshly allocated `Vec`.
    ///
    /// Convenience wrapper over [`read_at`](Source::read_at) for callers that
    /// want an owned buffer; the lazy backends keep no more than this resident.
    ///
    /// The request is bounds-checked against [`len`](Source::len) *before* the
    /// buffer is allocated. The metadata parsers feed `len` values straight from
    /// the file (a chunk-0 body size, a continuation-block length, a heap object
    /// size), so a malformed file could otherwise name a multi-gigabyte length
    /// and make this reserve `vec![0u8; len]` up front only for the read to fail
    /// EOF anyway — a cheap denial of service. Rejecting an out-of-range request
    /// before allocating avoids that; the error returned is identical to the one
    /// the underlying [`read_at`](Source::read_at) would have produced.
    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let end = offset
            .checked_add(len as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: len as u64,
            })?;
        if end > self.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end.to_usize().unwrap_or(usize::MAX),
                available: self.len().to_usize().unwrap_or(usize::MAX),
            });
        }
        let mut buf = vec![0u8; len];
        self.read_at(offset, &mut buf)?;
        Ok(buf)
    }

    /// Read metadata bytes, allowing source implementations to apply a bounded
    /// metadata cache.
    ///
    /// The default implementation performs an uncached exact read. Raw dataset
    /// payload readers intentionally call [`read_exact_at`](Self::read_exact_at)
    /// instead, so a metadata cache does not retain user data chunks.
    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.read_exact_at(offset, len)
    }

    /// What the metadata cache in front of this source has done, or `None` when
    /// it has none.
    ///
    /// The observation half of [`read_metadata_at`](Self::read_metadata_at):
    /// that method exists so an implementation *may* cache a metadata read, and
    /// this one reports whether doing so paid. The default is `None`, since most
    /// sources cache nothing.
    ///
    /// A wrapper that forwards `read_metadata_at` to an inner source must
    /// forward this too. Leaving it defaulted would have it report *no* cache
    /// where there is a full one, which reads as "caching is off" rather than as
    /// the missing forward it is.
    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        None
    }

    /// Zero that cache's cumulative counters, leaving its contents alone.
    ///
    /// The counterpart of HDF5's `H5Freset_mdc_hit_rate_stats`, for measuring
    /// one phase of a program rather than a whole run. A no-op where there is no
    /// cache, and it evicts nothing: occupancy is not a counter.
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

/// Holds a caller-supplied [`Source`] to the part of the trait's contract the
/// parsers above it cannot check for themselves.
///
/// [`Source::read_exact_at`] and [`Source::read_metadata_at`] have default
/// bodies that return exactly the bytes asked for, and every source this crate
/// builds either uses those bodies or forwards to one that does. An
/// implementation from outside may override them, and has a reason to — a
/// remote source that batches or coalesces its reads is the case
/// [`crate::File::from_source`] exists for. The parsers then index the returned
/// buffer at offsets derived from the length they *requested*, so a buffer that
/// comes back short is a slice panic inside a header parser, blaming a file
/// format for what the source did.
///
/// One length comparison per read turns that into [`FormatError::Source`],
/// which names the source instead. This wraps only what a caller hands in;
/// the crate's own sources have no override to check.
#[cfg(feature = "std")]
pub(crate) struct ValidatedSource<S>(S);

#[cfg(feature = "std")]
impl<S> ValidatedSource<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self(inner)
    }

    /// Refuse a buffer whose length is not the one that was asked for.
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

/// A [`Source`] view shifted forward by a base address: every read at a
/// base-relative `offset` is served from `inner` at `offset + base`.
///
/// Used wherever on-disk addresses are stored relative to the superblock's base
/// address rather than absolutely — the data layout's contiguous-data, chunk-index,
/// and chunk addresses on a file with a userblock, and the fractal-heap address in
/// an Attribute Info message. Presenting this shifted view lets those relative
/// addresses index it directly, exactly as an in-memory path slices the buffer at
/// `base`. For a plain (base-0) file it is the identity.
///
/// `len`/`read_at` shift by the base; `read_metadata_at` forwards to the inner
/// source at the *absolute* offset so the inner source's metadata cache is shared
/// (a chunk-index walk on a streaming userblock file would otherwise re-read every
/// node), while payload reads keep the default uncached `read_exact_at` so user
/// data does not evict metadata.
pub(crate) struct BaseOffsetSource<'a, S: Source + ?Sized> {
    pub(crate) inner: &'a S,
    pub(crate) base: BaseAddress,
}

/// A base-relative view of an in-memory file: `bytes` with its first `base` bytes
/// (the userblock) cut off, so every address stored relative to the base address
/// indexes it directly. The in-memory counterpart of [`BaseOffsetSource`], and the
/// identity for a plain file.
pub(crate) fn frame(bytes: &[u8], base: BaseAddress) -> Result<&[u8], FormatError> {
    if base.is_zero() {
        return Ok(bytes);
    }
    let start = base.get().to_usize()?;
    bytes.get(start..).ok_or(FormatError::UnexpectedEof {
        expected: start,
        available: bytes.len(),
    })
}

impl<S: Source + ?Sized> Source for BaseOffsetSource<'_, S> {
    fn len(&self) -> u64 {
        self.inner.len().saturating_sub(self.base.get())
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(self.base.absolute(offset)?, buf)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.inner
            .read_metadata_at(self.base.absolute(offset)?, len)
    }

    // The metadata reads above are the inner source's, so its cache is the one
    // to report on. A base-relative view holds none of its own.
    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.inner.metadata_cache_stats()
    }

    fn reset_metadata_cache_stats(&self) {
        self.inner.reset_metadata_cache_stats();
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// A [`Source`] over an in-memory byte buffer: anything that is
/// `AsRef<[u8]>` (`Vec<u8>`, `&[u8]`, `Box<[u8]>`, `Arc<[u8]>`, …).
///
/// This is the always-available backend that mirrors the crate's current
/// in-memory model, usable on WASM and `no_std`.
#[derive(Debug, Clone, Copy)]
pub struct BytesSource<T>(pub T);

impl<T: AsRef<[u8]>> BytesSource<T> {
    /// Wrap an in-memory byte buffer.
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
        let start = offset.to_usize()?;
        let end = start
            .checked_add(buf.len())
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > bytes.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end,
                available: bytes.len(),
            });
        }
        buf.copy_from_slice(&bytes[start..end]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Metadata-caching wrapper (std)
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
struct CachedMetadataRead {
    bytes: Vec<u8>,
    last_access: u64,
}

/// The bounded LRU store behind [`MetadataCachingSource`], also embedded
/// directly by the mirrorless write image (`crate::image::HandleImage`), which
/// must invalidate entries that overlap an in-place write.
///
/// # Why this is indexed rather than scanned (issue #367)
///
/// It held a `Vec` walked end to end by every operation, which made a *hit*
/// cost O(entries) and put the budget's useful range at a few thousand of them.
/// Measured against the positioned read a hit replaces: 9x faster at 64
/// entries, 3.2x at 1,024, then 1.2x **slower** at 4,096 and 21.9x slower at
/// 65,536. An 8 MiB budget, the figure `README.md` recommends, admits over
/// 100,000 metadata-sized reads, so the knob documented as a way to make a file
/// of many datasets read faster made one read about 30% slower.
///
/// Both maps below are therefore keyed, not searched, and the budget is a dial
/// over its whole range rather than only below a cliff.
#[cfg(feature = "std")]
pub(crate) struct MetadataReadCache {
    /// Entries by the `(offset, len)` the caller asked for. Two reads may share
    /// an offset at different lengths, so the length is part of the key.
    ///
    /// Ordered by offset first, which is what lets
    /// [`invalidate_overlapping`](Self::invalidate_overlapping) look at one
    /// bounded key range instead of every entry.
    entries: BTreeMap<(u64, usize), CachedMetadataRead>,
    /// `last_access` -> the key stamped with it, one row per entry. Its first
    /// row is the least recently used entry, which is what eviction wants.
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

    /// Take the cache's lock, treating a poisoned one as held rather than
    /// panicking: a cache is a performance aid, and a reader that panicked
    /// elsewhere leaves no invariant here for a later caller to trip over.
    pub(crate) fn locked(lock: &std::sync::Mutex<Self>) -> std::sync::MutexGuard<'_, Self> {
        lock.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Serve one [`Source::read_metadata_at`] through the cache behind `lock`,
    /// falling back to `read` and recording what happened.
    ///
    /// Both call sites that have a metadata cache — [`MetadataCachingSource`]
    /// and `crate::image::HandleImage` — go through here rather than each
    /// repeating the admission rule and its five counters; they differ only in
    /// what `read` does, which is why it is a closure.
    ///
    /// The lock is taken up to twice and never held across `read`. A metadata
    /// read is file I/O, and serializing every one of them behind this mutex
    /// would cost more than the cache saves.
    pub(crate) fn read_through(
        lock: &std::sync::Mutex<Self>,
        config: MetadataCacheConfig,
        offset: u64,
        len: usize,
        read: impl FnOnce() -> Result<Vec<u8>, FormatError>,
    ) -> Result<Vec<u8>, FormatError> {
        // A zero-length read is not a read of anything, and a disabled cache has
        // no counters worth keeping; neither is worth a lock.
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
        Self::locked(lock).insert(offset, len, bytes.clone(), config.max_bytes());
        Ok(bytes)
    }

    /// Snapshot the counters and the current occupancy.
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

    /// Zero the counters, keeping every entry. Occupancy is a measurement of the
    /// cache's contents rather than a tally of its history, so resetting the
    /// history must not disturb it.
    pub(crate) fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.oversize_reads = 0;
        self.evictions = 0;
        self.invalidations = 0;
    }

    /// Drop one entry and its access row together, keeping the two maps and the
    /// byte total in step. Every removal goes through here for that reason.
    fn remove(&mut self, key: (u64, usize)) {
        if let Some(entry) = self.entries.remove(&key) {
            self.by_access.remove(&entry.last_access);
            self.current_bytes -= entry.bytes.len();
        }
        debug_assert_eq!(
            self.entries.len(),
            self.by_access.len(),
            "every entry holds exactly one access row"
        );
    }

    /// Drop every cached entry that overlaps `[offset, offset + len)`, so a
    /// read after an in-place write never observes stale bytes.
    pub(crate) fn invalidate_overlapping(&mut self, offset: u64, len: usize) {
        if len == 0 {
            return;
        }
        let end = offset.saturating_add(len as u64);
        // An entry starting before this cannot reach `offset` at any length the
        // cache has admitted, so the search starts here rather than at the map's
        // first key. It is never above `end`, which is what `BTreeMap::range`
        // requires of its bounds: it is at most `offset`, and `end` is at least
        // `offset` even where the addition above saturates.
        let first = offset.saturating_sub(self.longest_entry as u64);
        let doomed: Vec<(u64, usize)> = self
            .entries
            .range((first, 0)..(end, 0))
            // The range settles `entry_offset < end`; this settles the other
            // half, that the entry reaches forward as far as `offset`.
            .filter(|((entry_offset, entry_len), _)| {
                entry_offset.saturating_add(*entry_len as u64) > offset
            })
            .map(|(key, _)| *key)
            .collect();
        self.invalidations += doomed.len() as u64;
        for key in doomed {
            self.remove(key);
        }
    }

    pub(crate) fn get(&mut self, offset: u64, len: usize) -> Option<Vec<u8>> {
        let key = (offset, len);
        // One tick per cached read, so a `u64` outlasts any process that could
        // run. The counter formerly wrapped, which would have inverted the very
        // ordering it exists to record.
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

    pub(crate) fn insert(&mut self, offset: u64, len: usize, bytes: Vec<u8>, max_bytes: usize) {
        if len == 0 || bytes.len() > max_bytes {
            return;
        }

        let key = (offset, len);
        // Re-reading a key replaces it. Removing first means the byte total and
        // the access index never keep a row for the value being displaced.
        self.remove(key);

        self.tick += 1;
        let tick = self.tick;
        self.longest_entry = self.longest_entry.max(len);
        self.current_bytes += bytes.len();
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
            // Counted here rather than in `remove`, which also serves
            // invalidation and replacement. Only a drop the *budget* forced is
            // an eviction, and that is the one that says to raise it.
            self.evictions += 1;
            self.remove(key);
        }
    }
}

/// A [`Source`] wrapper with a bounded cache for metadata reads.
///
/// The wrapper only caches calls to [`Source::read_metadata_at`]. Plain
/// [`Source::read_exact_at`] calls still go directly to the inner source,
/// which keeps raw dataset payloads out of the metadata cache.
#[cfg(feature = "std")]
pub struct MetadataCachingSource<S> {
    inner: S,
    config: MetadataCacheConfig,
    cache: std::sync::Mutex<MetadataReadCache>,
}

#[cfg(feature = "std")]
impl<S> MetadataCachingSource<S> {
    /// Wrap a source with the supplied metadata-cache configuration.
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
        MetadataReadCache::read_through(&self.cache, self.config, offset, len, || {
            self.inner.read_metadata_at(offset, len)
        })
    }

    /// `None` when the configuration disabled the cache, which is what the
    /// wrapper being present but inert means to a caller.
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

/// A lazy [`Source`] over any [`std::io::Read`] + [`std::io::Seek`] (a
/// [`std::fs::File`], an in-memory `Cursor`, etc.).
///
/// Each [`read_at`](Source::read_at) performs a `seek` + `read_exact`, so no
/// more than the requested bytes are ever held in memory. This is the backend
/// that lets a 32-bit host read a file larger than its address space: the
/// metadata and one working chunk fit even when the whole file does not.
///
/// The reader is wrapped in a [`std::sync::Mutex`] so the source is `Sync` and
/// `read_at` can take `&self` (seeking needs `&mut` access). This serializes
/// concurrent reads, which is correct though not maximally parallel; a future
/// backend can use positioned reads (`pread`/`seek_read`) to avoid the lock.
#[cfg(feature = "std")]
pub struct ReadSeekSource<R> {
    inner: std::sync::Mutex<R>,
    len: u64,
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> ReadSeekSource<R> {
    /// Wrap a `Read + Seek`, measuring its length by seeking to the end (then
    /// restoring nothing — every `read_at` seeks absolutely anyway).
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
        // Bound-check up front so a request past EOF is a clean error rather
        // than a backend-specific short read.
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > self.len {
            return Err(FormatError::UnexpectedEof {
                // `expected`/`available` are byte counts; report them as the
                // best `usize` we can without truncating on a 32-bit host.
                expected: end.to_usize().unwrap_or(usize::MAX),
                available: self.len.to_usize().unwrap_or(usize::MAX),
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

/// Render an `std::io::Error` to a short owned string for [`FormatError::Source`]
/// (which is `no_std`-friendly and cannot hold the error itself).
#[cfg(feature = "std")]
fn format_io(e: &std::io::Error) -> std::string::String {
    std::format!("{e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

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
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
        // Reading exactly to the end is fine.
        let mut ok = [0u8; 3];
        src.read_at(0, &mut ok).unwrap();
        assert_eq!(ok, [1, 2, 3]);
    }

    #[test]
    fn bytes_source_offset_past_end_is_eof() {
        let src = BytesSource::new(vec![0u8; 8]);
        let mut buf = [0u8; 1];
        assert!(matches!(
            src.read_at(8, &mut buf).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
        // Zero-length read at EOF succeeds.
        src.read_at(8, &mut []).unwrap();
    }

    #[test]
    fn read_exact_at_rejects_oversized_len_without_allocating() {
        // A length far larger than the source must error cleanly rather than
        // attempt to reserve the buffer first. Before the pre-allocation bounds
        // check, this called `vec![0u8; usize::MAX]` and aborted the process.
        let src = BytesSource::new(vec![1u8, 2, 3, 4]);
        assert!(matches!(
            src.read_exact_at(0, usize::MAX).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
        // A read that fits is unaffected.
        assert_eq!(src.read_exact_at(1, 3).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn empty_source() {
        let src = BytesSource::new(Vec::<u8>::new());
        assert_eq!(src.len(), 0);
    }

    #[test]
    fn forwarding_through_reference() {
        let src = BytesSource::new(vec![9u8, 8, 7]);
        let r: &dyn Source = &src;
        let mut buf = [0u8; 2];
        r.read_at(1, &mut buf).unwrap();
        assert_eq!(buf, [8, 7]);
    }

    #[test]
    fn forwarding_through_reference_preserves_metadata_reads() {
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

        // Every read_at against the lazy source matches the in-memory source.
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
        assert!(matches!(
            seek.read_at(2, &mut buf).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
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
    fn eviction_drops_the_least_recently_used_entry_not_the_oldest() {
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
    fn invalidation_takes_every_overlap_and_spares_the_neighbours() {
        let budget = 1024;
        let mut cache = MetadataReadCache::new();
        for offset in [0u64, 10, 20, 30] {
            cache.insert(offset, 10, vec![offset as u8; 10], budget);
        }
        // One long entry starting well before the write below and reaching well
        // past it. Nothing but its own length says it can be reached from there.
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
    fn re_inserting_a_key_replaces_it_rather_than_counting_it_twice() {
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

    /// A hit must not get slower as the cache gets bigger (issue #367).
    ///
    /// The store this replaced walked a `Vec`, so a hit cost O(entries), which
    /// put it above the cost of the positioned read it exists to avoid from
    /// about 3,000 entries on. Measured in release against that read: 9x faster
    /// at 64 entries, 3.2x at 1,024, then 1.2x *slower* at 4,096 and 21.9x
    /// slower at 65,536.
    ///
    /// Across the pair below, the scanning store measured 16.6 in release
    /// (369 ns to 6,134) and 36.8 unoptimized. Indexed, the same pair measures
    /// 1.10 and 1.30. The allowance sits between those two groups with room on
    /// either side: six times what an unoptimized build measures here, and a
    /// fifth of what a return to scanning would.
    #[test]
    fn a_hit_does_not_get_slower_as_the_cache_grows() {
        /// Entry counts either side of the growth, and the factor the cost is
        /// allowed to move across it. Named so the failure message cannot drift
        /// from what was measured.
        const SMALL: usize = 1_024;
        const LARGE: usize = 16_384;
        const ALLOWED_GROWTH: f64 = 8.0;

        fn nanos_per_hit(entries: usize) -> f64 {
            const ENTRY_LEN: usize = 64;
            let budget = entries * ENTRY_LEN * 2;
            let mut cache = MetadataReadCache::new();
            for i in 0..entries {
                cache.insert(
                    (i * ENTRY_LEN) as u64,
                    ENTRY_LEN,
                    vec![7u8; ENTRY_LEN],
                    budget,
                );
            }
            // Warm the caches the machine has, then time a pass that is all hits.
            for i in 0..entries {
                assert!(cache.get((i * ENTRY_LEN) as u64, ENTRY_LEN).is_some());
            }
            let started = std::time::Instant::now();
            for i in 0..entries {
                assert!(cache.get((i * ENTRY_LEN) as u64, ENTRY_LEN).is_some());
            }
            started.elapsed().as_secs_f64() * 1e9 / entries as f64
        }

        let small = nanos_per_hit(SMALL);
        let large = nanos_per_hit(LARGE);
        assert!(
            large < small * ALLOWED_GROWTH,
            "a hit cost {large:.0} ns with {LARGE} entries against {small:.0} ns with \
             {SMALL} -- growing the cache should not move it, and a cost that tracks \
             its size is the shape of a store being searched rather than indexed"
        );
    }

    // -----------------------------------------------------------------------
    // What the cache reports about itself (issue #353)
    // -----------------------------------------------------------------------

    /// A source of `len` bytes that serves every metadata read, so a cache in
    /// front of it is the only thing that can make a read not happen.
    #[cfg(feature = "std")]
    fn ramp(len: usize) -> BytesSource<Vec<u8>> {
        BytesSource::new((0..len).map(|i| i as u8).collect::<Vec<u8>>())
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_read_too_large_to_admit_is_not_charged_as_a_miss() {
        // 64-byte entries are eligible; anything above that is turned away
        // before it reaches the cache.
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
        // The two oversize reads are the caller's to fix by raising
        // `max_entry_bytes`, and folding them in would report the cache at 25%
        // rather than saying which knob is turning them away.
        assert_eq!(stats.hit_rate(), Some(0.5));

        // The other half of the same rule. `with_max_entry_bytes` can name a cap
        // above the whole budget, and a read between the two would pass the entry
        // check only for `insert` to refuse it every time -- a permanent miss
        // reported as an ordinary one. The budget turns it away up front instead.
        let lopsided = MetadataCacheConfig::new(128).with_max_entry_bytes(512);
        let source = MetadataCachingSource::new(ramp(4096), lopsided);
        source.read_metadata_at(0, 256).unwrap();
        source.read_metadata_at(0, 256).unwrap();
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(stats.oversize_reads(), 2);
        assert_eq!(stats.misses(), 0);
        assert_eq!(stats.hit_rate(), None);
    }

    #[cfg(feature = "std")]
    #[test]
    fn no_eligible_read_yet_is_not_a_hit_rate_of_zero() {
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
    fn the_budget_and_a_write_drop_entries_for_different_reasons() {
        // Two 64-byte entries fit; a third forces one out.
        const BUDGET: usize = 128;
        let mut cache = MetadataReadCache::new();
        cache.insert(0, 64, vec![1u8; 64], BUDGET);
        cache.insert(64, 64, vec![2u8; 64], BUDGET);

        // Re-reading a key replaces it. Nothing was dropped for want of room or
        // because the bytes changed, so neither counter moves.
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
        // Occupancy measures the cache rather than tallying its history, so a
        // reset that emptied it would answer a different question than the one
        // `H5Freset_mdc_hit_rate_stats` asks.
        assert_eq!(stats.entries(), 1);
        assert_eq!(stats.bytes(), 64);
        assert!(cache.get(0, 64).is_some(), "the entry is still servable");
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_disabled_cache_reports_nothing_rather_than_zeroes() {
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
    fn a_wrapper_reports_the_cache_it_reads_through() {
        let config = MetadataCacheConfig::new(4096);
        let source = MetadataCachingSource::new(ramp(4096), config);
        // The base-relative view a userblock file reads through forwards its
        // metadata reads to the inner source, so it must forward the account of
        // them too.
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
