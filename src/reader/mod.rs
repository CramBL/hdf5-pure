//! Reading API: File, Dataset, and Group handles for reading HDF5 files.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use hdf5_pure_core::PhysicalAddress;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::MetadataBytes;

#[cfg(test)]
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
#[cfg(test)]
use crate::FixedPointType;
#[cfg(test)]
use crate::UserblockSize;
use crate::VariableLengthType;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::OpenedImageContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_read::ChunkDecoding;
use crate::chunked_read::ChunkRead;
use crate::edit::AppendBuilder;
use crate::edit::AppendGeometry;
use crate::edit::AppendOptions;
use crate::edit::AppendTarget;
use crate::edit::EditBacking;
use crate::edit::MemoryStrategy;
use crate::edit::StagedChild;
use crate::edit::StagedKind;
use crate::edit::StagedMeta;
use crate::edit::StagedObject;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::element::{ElementDecoder, H5Element};
use crate::type_builders::{DatasetBuilder, VL_REF_SIZE};

#[cfg(any(doc, test))]
use crate::FileSpaceStrategy;
use crate::FillTime;
use crate::SuperblockVersion;
use crate::appender::{BufferedAppender, OwnedBufferedAppender};
use crate::attribute;
use crate::attribute::AttributeMessage;
use crate::chunk_cache::{CachePass, ChunkCache, ChunkCacheConfig, ChunkCacheStats};
use crate::compound::CompoundType;
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::data_read;
use crate::data_read::{NumericDecoder, NumericReadTarget};
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::{Datatype, ReferenceType};
use crate::error::{Error, FormatError};
use crate::file_create_properties::FileCreateProperties;
use crate::file_lock::{self, FileLocking, OpenIntent, OpenTarget, WriteMarkPolicy};
use crate::file_space_info::FileSpaceInfo;
use crate::fill_value::{FillPattern, FillValueMessage, FillValueView};
use crate::filter_pipeline::FilterPipeline;
use crate::free_sections::FreeSections;
use crate::free_sections::PersistedFreeSections;
use crate::free_sections::ReusableSpaceStats;
use crate::group_v1;
use crate::group_v1::GroupEntry;
use crate::group_v2::{self, ChildLookup};
use crate::layout_info::{Chunk, ChunkIndex, Filter, FilterId, Layout};
use crate::libver::LibVer;
use crate::message_type::MessageType;
use crate::object_header::HeaderMessage;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::object_path::{LinkName, LinkNameBuf, ObjectPath, ObjectPathBuf};
#[cfg(test)]
use crate::page_publication_failure::PagePublicationFailure;
use crate::read_region::{ReadRegion, RegionRequest};
use crate::read_spec::{RawReadSpec, RawReadStorage};
use crate::shared_message::{self, BufferedResolver, SharedResolver, SourceResolver};
use crate::source::SourceMetadata;
use crate::source::{
    self, BaseOffsetSource, BytesSource, MetadataCacheConfig, MetadataCacheStats,
    MetadataCachingSource, ReadSeekSource, Source, ValidatedSource,
};
use crate::source::{PhysicalSource, StoredBytes, StoredRead};
use crate::superblock;
use crate::superblock::LocatedSuperblock;
use crate::superblock::Superblock;
use crate::superblock::SuperblockSource;
use crate::types::{self, AttrValue, DType};
use crate::vl_data::{self, VlElement, VlenStringReadOptions};
use crate::width::FormatWidths;

#[cfg(test)]
pub(crate) mod test_util;

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    #[cfg(test)]
    mod native_crosschecks;
}

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// Backing store for a [`File`]: either the whole file buffered in memory, or a
/// lazy [`Source`] that reads regions on demand (see [`File::open_streaming`]).
enum Backend {
    InMemory(Vec<u8>),
    Streaming(Box<dyn Source + Send + Sync>),
    /// A read-write file opened with [`File::open_rw`]: a [`WriteEngine`] (exclusive OS lock + staged
    /// edit queues + append geometry cache) behind a lock, so owned handles can
    /// both read and mutate in place. Handle write methods route to the engine,
    /// and `File::commit` applies staged structural edits.
    ///
    /// Either backing — a whole-file mirror, or positioned I/O against the
    /// handle — appears here as the same `WriteEngine`; which one an open
    /// resolved to is the engine's own business rather than the backend's
    /// (issue #198). Reads
    /// borrow the mirror's slice when there is one and go through the image's
    /// `Source` otherwise; see [`with_engine`](FileInner::with_engine). Boxed to
    /// keep the `Backend` enum small (a `WriteEngine` is far larger than the
    /// other variants).
    Edit(Box<Mutex<WriteEngine>>),
}

/// What an operation through a file's write session can do to it, which is what
/// decides how much of an object handle's memo it invalidates — and, with it,
/// which edit surface the operation is using.
///
/// One variant carries both because they coincide: the *staged* surface is the
/// one that commits, and a commit is the only thing that moves an object header,
/// so [`Relocating`](Self::Relocating) is exactly the surface a SWMR writer
/// refuses. An operation off that surface that could relocate would break the
/// pairing and would have to say which it was separately.
#[derive(Clone, Copy)]
enum Change {
    /// Rewrites object headers and can move them: a commit. The staging of one
    /// counts too, though it writes nothing — a pending edit is a header move
    /// this session has not made yet, and an address is worth no more against
    /// one than against the commit that will apply it. Invalidates a handle's
    /// address as well as the header it parsed.
    ///
    /// This is the staged surface, which a SWMR writer refuses.
    Relocating,
    /// Changes bytes without moving any object header: an immediate
    /// [`Dataset::append`], which rewrites the dataset's dimension where its
    /// header stands, and the free-space and status-flag bookkeeping a session's
    /// teardown rewrites where it stands. Invalidates the parsed header alone,
    /// which is what lets a handle reached by object reference — the one kind
    /// with no name to look itself up by — go on appending.
    InPlace,
    /// Changes nothing a handle can observe: a durability barrier over writes
    /// the operations that made them already accounted for, or a question whose
    /// answer the engine caches. Invalidates no memo.
    ///
    /// Distinct from not going through the gate at all, which is what a *read*
    /// does: these still need the write session, and still need the file to be
    /// open for writing and unsealed.
    Nothing,
}

/// Where an object handle's header sits, and what that answer holds as of.
///
/// A handle names its object — by path, or by the address a reference gave it —
/// and this is a memo of what that name last resolved to. See
/// [`FileInner::locate`], which takes one, and the two counters it reads.
#[derive(Clone, Copy)]
struct Resolution {
    content_revision: u64,
    address_revision: u64,
    address: u64,
}

/// The pair of revisions read *before* a resolution is worked out, which the
/// answer is then labelled with.
///
/// Splitting the reads from the address is what keeps the order right at every
/// call site: there is no way to label an address with a revision taken after
/// it, which would claim a freshness the address does not have. See
/// [`FileInner::locate`].
#[derive(Clone, Copy)]
struct Revisions {
    content: u64,
    address: u64,
}

impl Revisions {
    /// Label `address` as worked out at these revisions.
    const fn at(self, address: u64) -> Resolution {
        Resolution {
            content_revision: self.content,
            address_revision: self.address,
            address,
        }
    }
}

/// Where a [`Dataset`] or [`Group`] handle stands with respect to the staged
/// set, worked out by [`FileInner::staged_standing`].
///
/// A handle names its object by path, and a path can mean an object in the
/// file, an object this session has staged, or — for a handle *born* onto a
/// staged creation — nothing at all, once that creation is withdrawn. The third
/// is why the handle carries a mark of its own: without one, a withdrawal is
/// indistinguishable from a commit, and the handle would silently start
/// answering for whatever the file holds at the path, which in the case that
/// produces it is the object the session is deleting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Standing {
    /// Resolve the path against the file, as a handle opened by name does.
    Live,
    /// A creation this session staged owns the path and no commit has written
    /// it: [`Error::NotCommitted`] for anything needing its bytes.
    Pending,
    /// This handle was made onto a staged creation that has since been
    /// withdrawn: [`Error::StagingWithdrawn`].
    Withdrawn,
}

/// A borrowed `Source` view over a [`File`]'s backend, used by the
/// streaming-capable read paths so one call site serves both backends.
pub(crate) enum SourceView<'a> {
    Mem(&'a [u8]),
    Stream(&'a (dyn Source + Send + Sync)),
}

impl Source for SourceView<'_> {
    fn len(&self) -> u64 {
        match self {
            SourceView::Mem(b) => b.len() as u64,
            SourceView::Stream(s) => s.len(),
        }
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        match self {
            SourceView::Mem(b) => BytesSource::new(*b).read_at(offset, buf),
            SourceView::Stream(s) => s.read_at(offset, buf),
        }
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        match self {
            SourceView::Mem(b) => BytesSource::new(*b).read_metadata_at(offset, len),
            SourceView::Stream(s) => s.read_metadata_at(offset, len),
        }
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        match self {
            SourceView::Mem(_) => Ok(0),
            SourceView::Stream(s) => s.metadata_origin(),
        }
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        match self {
            SourceView::Mem(bytes) => {
                let range = FileRange::new(PhysicalAddress::new(offset), len.to_u64())?;
                MetadataBytes::borrowed(range, source::bytes_in(bytes, range)?)
            }
            SourceView::Stream(s) => s.read_metadata(offset, len),
        }
    }

    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        match self {
            // A whole-file buffer is already the cache, and holds no second one.
            SourceView::Mem(_) => None,
            SourceView::Stream(s) => s.metadata_cache_stats(),
        }
    }

    fn reset_metadata_cache_stats(&self) {
        match self {
            SourceView::Mem(_) => {}
            SourceView::Stream(s) => s.reset_metadata_cache_stats(),
        }
    }
}

/// The policy for recovering persisted free space during writable opening.
///
/// Configure this through [`FileAccessProperties::with_persisted_free_space_policy`]. The
/// default is [`Tolerant`](Self::Tolerant). Read-only opening ignores this policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PersistedFreeSpacePolicy {
    /// Rejects persisted managers whose metadata cannot be read or parsed.
    ///
    /// Parses every defined manager before recovering reusable sections. Header and section-list
    /// checksums are verified regardless of the `checksum` feature. Overlapping free sections,
    /// overlapping manager allocations, and extents beyond the logical end of file are rejected.
    Strict,
    /// Continues opening when persisted managers cannot be read or parsed.
    ///
    /// A manager parse failure prevents recovery of reusable sections. Invalid section extents
    /// are skipped.
    #[default]
    Tolerant,
}

/// File-access properties applied when opening an HDF5 file.
///
/// This is the `hdf5-pure` analogue of an HDF5 **file access property list**
/// (`fapl`): one value carrying every access-time setting, built once and passed
/// to whichever open a caller reaches for, exactly as a `fapl` is handed to
/// `H5Fopen`. Every `*_with_options` constructor on [`File`] accepts it, so a
/// read path and a read-write path can share one configuration.
///
/// The `Properties` suffix means this type represents a file access property list. It is a
/// [`Copy`] value with no handle to create or close, runtime property registry, or fallible setter.
///
/// - The metadata cache (`H5Pset_mdc_config`) applies to the streaming and
///   bounded backends; an in-memory open already holds the whole file in one
///   buffer.
/// - The chunk cache (`H5Pset_cache`) is the file-wide default for datasets
///   opened from any backend, overridable per dataset with
///   [`DatasetAccessProperties`].
/// - The locking policy (`H5Pset_file_locking`) applies to the read-write opens.
///   Readers and the SWMR writer take no lock by design, so they ignore it.
/// - The write-mark policy applies to the read-only opens, and has no C
///   counterpart: `H5Fopen` refuses a file marked open for write with no
///   override, where [`with_write_mark_policy`](Self::with_write_mark_policy)
///   can admit a snapshot read of one.
///
/// The persisted-free-space policy applies to writable opening. Configure it through
/// [`with_persisted_free_space_policy`](Self::with_persisted_free_space_policy).
///
/// # Supported properties
///
/// Selected through the [`File`] open-mode constructor, with memory budgets and locking set through
/// [`FileAccessProperties`](Self), [`DatasetAccessProperties`] and [`FileLocking`]. The [status
/// legend](crate::FileCreateProperties#status-legend) explains the status column.
///
/// | HDF5 property / driver | `hdf5-pure` | Status | Behavior |
/// |---|---|---|---|
/// | `H5Fopen(RDONLY)`, default sec2 | [`File::open`](crate::File::open) | **Genuine (read-only)** | Whole-file buffered read, without a lock. |
/// | positioned / on-demand reads | [`File::open_streaming`](crate::File::open_streaming) | **Genuine (read-only, bounded)** | Fetches metadata and chunks on demand, with peak memory near one chunk. |
/// | user-supplied read driver (cf. `H5Pset_driver`) | [`File::from_source`](crate::File::from_source) / [`from_source_with_options`](crate::File::from_source_with_options), over the [`Source`] trait | **Genuine (read-only, bounded)** | The same on-demand reads over bytes that are not a path: an object store addressed by range request, or a sandboxed guest handed byte ranges by its host. A read that comes back short is rejected. |
/// | `H5Pset_fapl_core` / `H5Pset_file_image` | [`File::from_bytes`](crate::File::from_bytes) / [`FileBuilder::finish`](crate::FileBuilder::finish) | **Genuine** | Read an in-memory file image, or build one into a `Vec<u8>`. |
/// | `H5Fopen(RDWR)` | [`File::open_rw`](crate::File::open_rw) / [`open_rw_with_options`](crate::File::open_rw_with_options) | **Genuine (read-write)** | Reads, appends, and staged edits with [`commit`](crate::File::commit). Uses bounded memory for eligible latest-format files. Files with a pre-v2 superblock, a userblock, or non-persistent PAGE allocation require the whole-file mirror. Non-persistent PAGE commits append in fresh typed pages and require a tree proved free of object references. |
/// | `H5Fopen(RDWR)`, bounded memory | [`File::open_rw_with_options`](crate::File::open_rw_with_options) + [`MemoryStrategy::Bounded`] | **Genuine (read-write, bounded)** | [`File::open_rw`](crate::File::open_rw) picks the bounded backing itself. Stating `Bounded` makes it strict: a file the bounded engine cannot edit is rejected, and never mirrored. |
/// | no C counterpart | [`with_memory_strategy`](Self::with_memory_strategy), [`MemoryStrategy`] | **Behavioral** | How much memory a read-write open may spend holding the file, overriding the dispatch above. `Bounded` rejects a file with a pre-v2 superblock, a userblock, or non-persistent PAGE allocation. `Auto`, which is what [`File::open_rw`](crate::File::open_rw) uses unset, falls back to the mirror, and `Mirrored` always mirrors. Non-persistent PAGE files support mirrored staged edits with the restrictions documented by [`File::open_rw`](crate::File::open_rw). [`File::edit_backing`](crate::File::edit_backing) reports which backend an open resolved to, as an [`EditBacking`] (`Bounded` or `Mirrored`), a separate type because `Auto` is a preference between the two and never an outcome. libhdf5 has no analogue: `H5Fopen` picks its own caching with no caller-visible memory contract. |
/// | `H5F_ACC_SWMR_READ` / `H5F_ACC_SWMR_WRITE` | [`File::open_swmr`](crate::File::open_swmr) / [`open_swmr_writer`](crate::File::open_swmr_writer) (`*_with_options`) | **Genuine** | No OS lock. The writer raises the superblock SWMR-write flag (a v3 superblock is required, as in the C library) and appends alone. It always mirrors, so it accepts `MemoryStrategy::Auto` and `Mirrored` and rejects an explicit `Bounded`. |
/// | superblock status flags (`h5clear --status`) | checked by [`File::open`](crate::File::open), [`open_streaming`](crate::File::open_streaming), [`from_source`](crate::File::from_source), [`open_rw`](crate::File::open_rw), [`open_swmr`](crate::File::open_swmr) and [`open_swmr_writer`](crate::File::open_swmr_writer), and cleared by [`File::clear_status_flags`](crate::File::clear_status_flags) | **Genuine** | A file the byte marks as held by a writer is rejected with [`Error::FileMarkedInUse`], as `H5Fopen` rejects it. [`File::open_swmr`](crate::File::open_swmr) follows such a file, and rejects a half-set mark alone. Version-3 superblocks only, which is where the C library checks. [`File::from_bytes`](crate::File::from_bytes) does not consult the byte, and neither, therefore, does [`mat::from_file`](crate::mat::from_file), so a caller holding the bytes can still read a flagged file, at the cost of a copy of it. |
/// | no C counterpart | [`with_write_mark_policy`](Self::with_write_mark_policy), [`WriteMarkPolicy`] | **Behavioral** | Whether a read-only open proceeds past a file marked open for write by a non-SWMR writer, bit 0 alone, which is what a page-buffered session raises and what no SWMR reader can follow, so such a file is otherwise readable only by copying it through [`File::from_bytes`](crate::File::from_bytes). `AllowSnapshot` reads it as it stands, on the caller's assertion that the writer has flushed ([`File::sync`](crate::File::sync), or a clean close), which the byte cannot record and this crate therefore cannot check. It admits nothing else: a SWMR pair belongs to [`File::open_swmr`](crate::File::open_swmr), and [`File::open_rw`](crate::File::open_rw) and [`File::open_swmr_writer`](crate::File::open_swmr_writer) are rejected whatever it says. The default `Refuse` is what `H5Fopen` does, and `H5Fopen` offers no override at all: `h5clear` is its one way through, and it edits the file. |
/// | `H5Pset_file_locking` + `HDF5_USE_FILE_LOCKING` | [`with_locking`](Self::with_locking), [`FileLocking`] | **Genuine** | Exclusive advisory lock on both read-write paths, mirror and bounded, taken non-blocking, so a lock another process holds is [`Error::FileLocked`]. The environment-variable override recognizes the same values as the C library. Readers and the SWMR writer run without a lock by design and ignore the setting. |
/// | `H5Fget_libver_bounds` (read) | [`File::libver_bound`](crate::File::libver_bound) | **Read-only** | Reports the low library-version bound the superblock version implies. |
/// | `H5Pset_cache` (`rdcc`) / `H5Pset_chunk_cache` (`dapl`) | [`with_chunk_cache`](Self::with_chunk_cache), [`DatasetAccessProperties::with_chunk_cache`](crate::DatasetAccessProperties::with_chunk_cache) | **Genuine (all backends)** | A cache of decompressed chunks and parsed indexes, 1 MiB in 16 slots by default. `rdcc_nslots` and `rdcc_nbytes` map directly. There is no write coalescing, since a mutation clears the cache, and no `rdcc_w0`, for the reason below. [`Dataset::chunk_cache_stats`](crate::Dataset::chunk_cache_stats) reports what it did. |
/// | `H5Pset_mdc_config` | [`with_metadata_cache`](Self::with_metadata_cache), [`MetadataCacheConfig`] | **Behavioral (partial)** | A byte budget for a metadata-read LRU on the streaming and bounded backends, off by default. [`File::metadata_cache_stats`](crate::File::metadata_cache_stats) reports what it did (`H5Fget_mdc_hit_rate` and `H5Fget_mdc_size`) and [`File::reset_metadata_cache_stats`](crate::File::reset_metadata_cache_stats) clears the counters (`H5Freset_mdc_hit_rate_stats`). Of `H5AC_cache_config_t` this is `max_size`, and [the metadata cache](#the-metadata-cache-h5pset_mdc_config) below covers what the other 29 fields would mean here. |
/// | no C counterpart | [`with_sync_policy`](Self::with_sync_policy), [`SyncPolicy`], [`File::sync`](crate::File::sync) | **Behavioral** | `Always` requests durability at publication barriers. `OnClose` defers durability until explicit synchronization or session closure. Immediate appends preserve allocation and reference ordering under both policies, including with explicit page buffering. Page-buffered staged commits can retain writes after returning. |
/// | `H5Pset_page_buffer_size` | [`with_page_buffer_size`](Self::with_page_buffer_size) | **Genuine** | A write buffer with a byte budget. Staged commit writes can remain buffered across operations. Immediate appends drain pending writes at allocation and reference publication boundaries. Requires `SyncPolicy::OnClose`, a version-3 superblock, a budget of at least one merge page, and persisted free space on PAGE files. SWMR writers reject it. The editor marks the file for the session lifetime. Off by default. |
/// | `H5Pset_fapl_family` / `split` / `multi` / `mpio` / `direct` / `ros3` / `log` | none | **Unsupported** | Only two implicit drivers exist: an in-memory buffer and `Read + Seek` positioned I/O. A multi-file, parallel, or remote-object file is not opened. |
///
/// ## The metadata cache (`H5Pset_mdc_config`)
///
/// `H5AC_cache_config_t` is 30 fields (version 1, checked against HDF5 2.1.0).
/// [`MetadataCacheConfig`] models one of them, `max_size`, as `max_bytes`, plus a `max_entry_bytes`
/// cap that has no C counterpart. That ratio invites the reading that 29 things are missing. This
/// cache holds byte ranges of a file being read, where the C cache holds parsed entries of a file
/// being written, and each group of fields would mean this here:
///
/// | Group | Fields | Applies here? |
/// |---|---|---|
/// | `max_size` | 1 | **Modeled**, as [`MetadataCacheConfig::max_bytes`](crate::MetadataCacheConfig::max_bytes). |
/// | Adaptive resize: `min_size`, `epoch_length`, `incr_mode` + 4, `flash_incr_mode` + 2, `decr_mode` + 7 | 18 | Not modeled: they are one feature, and not 18 separate settings. See below. |
/// | Dirty-data and parallel: `min_clean_fraction`, `dirty_bytes_threshold`, `metadata_write_strategy` | 3 | **Never.** This cache holds bytes read alone, so there is no dirty entry to keep clean, flush, or write. A read-write session drops the ranges its writes overlap, which [`MetadataCacheStats::invalidations`](crate::MetadataCacheStats::invalidations) counts. |
/// | Diagnostics: `version`, `rpt_fcn_enabled`, `open_trace_file`, `close_trace_file`, `trace_file_name` | 5 | No. `version` is C ABI evolution, and the rest are a trace-file facility this crate has no counterpart to. [`File::metadata_cache_stats`](crate::File::metadata_cache_stats) reports what they were for. |
/// | Initial sizing: `set_initial_size`, `initial_size` | 2 | No effect to have. There is no arena allocated up front: entries are allocated as reads admit them, so the initial size is zero and the maximum is the only bound. |
/// | `evictions_enabled` | 1 | Expressible already, as a budget larger than the metadata a session reads. A separate flag would be a second way to say it, and one that silently uncaps memory. |
///
/// Adaptive resize is the one substantive gap, and leaving it out is deliberate. The algorithm
/// grows the cache when an epoch's hit rate falls below `lower_hr_threshold` and shrinks it when
/// the rate rises above `upper_hr_threshold` or entries age out, so that a caller does not have to
/// pick a size. Picking one is cheap here: the store is indexed and not scanned
/// ([#367](https://github.com/CramBL/hdf5-pure/issues/367)), so a hit costs about the same at
/// 65,000 entries as at 1,000, 161 nanoseconds against 136, measured, and an over-generous budget
/// costs memory and nothing else. The advice the algorithm would arrive at is one line, set the
/// budget generously, and [`File::metadata_cache_stats`](crate::File::metadata_cache_stats)
/// confirms it landed. Note also the direction the algorithm runs in: a low hit rate makes it hold
/// more.
///
/// ## The chunk cache (`rdcc`)
///
/// [`ChunkCacheConfig`] models `rdcc_nslots` and `rdcc_nbytes` directly. The third field,
/// `rdcc_w0`, is not modeled, and the reason is what it discriminates, and not a gap in the
/// eviction code.
///
/// In `H5Dchunk.c`, `w0` is a head start measured in entries: `w[0] = (int)(rdcc->nused *
/// rdcc->w0)`, counted down one per step of `H5D__chunk_cache_prune`. Two methods walk the LRU list
/// from its least-recently-used end. Method 0 preempts the entries that are not partially consumed.
/// Method 1 preempts anything unlocked, and is not introduced until the head start reaches zero. So
/// `w0` sets how far the selective rule gets to run alone, and its discriminator is
/// `H5D_rdcc_ent_t::rd_count` and `wr_count`, per-entry counts of bytes remaining, initialized to
/// the chunk size and decremented by `naccessed` in `H5D__chunk_unlock`. `w0` decides exactly one
/// thing: whether to drop a chunk the caller consumed in full ahead of one it consumed in part.
///
/// Nothing here turns on that decision, for two separate reasons.
///
/// | C field | Applies here? |
/// |---|---|
/// | `wr_count` | **Never.** This cache holds decompressed data read from the file. There is no dirty chunk to write back. A commit in the session drops what it may have made stale, which [`ChunkCacheStats::invalidations`](crate::ChunkCacheStats::invalidations) counts. |
/// | `rd_count` | **Not recorded.** A slot records that it holds a chunk, and never how much of that chunk a caller took. The class `w0` sorts by therefore does not exist here to be sorted. Partial consumption is the common case: an edge chunk of a dataset whose extent is not a multiple of its chunk extent is stored full-size and read in part, which is `rd_count > 0` in C too. |
///
/// The second reason is that on the path where a scan would thrash, there is no prune to order at
/// all. A whole read visits each of its chunks exactly once, so it fills the cache and then stops
/// offering, and it hands back none of the chunks it has already placed for chunks it will never
/// read again. That removes the problem `w0` exists to soften.
///
/// Which figure reports a too-small budget therefore depends on how the dataset is read, and each
/// is structurally zero on the other's path:
///
/// | read | admission | budget signal |
/// |---|---|---|
/// | whole ([`read_f64`](crate::Dataset::read_f64) and its siblings) | fills the cache, then stops offering | `rejections`, with `evictions` at 0 |
/// | row window ([`read_raw_rows`](crate::Dataset::read_raw_rows), `read_*_rows`) | plain LRU, since the chunk its successor needs is the one it finished on | `evictions`, with `rejections` at 0 |
///
/// A row window is the one path that does evict, so it is the only place a `w0` analogue could ever
/// pick a different victim. It would not pick a better one: the entry the next window needs is the
/// one this read finished on, which is the most recently used and the last thing an LRU rule gives
/// up, while the partially-consumed entries `w0` would additionally protect are the leading
/// boundary chunks that a forward sweep never revisits. That is an argument about the access
/// patterns these readers generate, and not a proof for an arbitrary one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[doc(alias = "fapl")]
pub struct FileAccessProperties {
    metadata_cache: MetadataCacheConfig,
    chunk_cache: ChunkCacheConfig,
    locking: FileLocking,
    memory_strategy: Option<MemoryStrategy>,
    libver_bounds: Option<(LibVer, LibVer)>,
    sync_policy: SyncPolicy,
    page_buffer_size: usize,
    write_mark_policy: WriteMarkPolicy,
    persisted_free_space_policy: PersistedFreeSpacePolicy,
}

impl FileAccessProperties {
    /// Creates properties with the crate's default access behavior.
    ///
    /// Persisted free-space recovery defaults to [`PersistedFreeSpacePolicy::Tolerant`].
    pub const fn new() -> Self {
        Self {
            metadata_cache: MetadataCacheConfig::disabled(),
            chunk_cache: ChunkCacheConfig::new(),
            locking: FileLocking::Enabled,
            memory_strategy: None,
            libver_bounds: None,
            sync_policy: SyncPolicy::Always,
            page_buffer_size: 0,
            write_mark_policy: WriteMarkPolicy::Refuse,
            persisted_free_space_policy: PersistedFreeSpacePolicy::Tolerant,
        }
    }

    /// Configure the bounded streaming metadata cache.
    #[doc(alias = "H5Pset_mdc_config")]
    pub const fn with_metadata_cache(mut self, metadata_cache: MetadataCacheConfig) -> Self {
        self.metadata_cache = metadata_cache;
        self
    }

    /// Configure the per-dataset raw chunk cache used by datasets opened from
    /// this file. This is the `H5Pset_cache`-style file-wide default.
    #[doc(alias = "H5Pset_cache")]
    pub const fn with_chunk_cache(mut self, chunk_cache: ChunkCacheConfig) -> Self {
        self.chunk_cache = chunk_cache;
        self
    }

    /// Set the OS advisory file-locking policy for the read-write opens.
    ///
    /// Defaults to [`FileLocking::Enabled`]. Use [`FileLocking::Disabled`] only
    /// when an external mechanism already guarantees single-writer access, or
    /// [`FileLocking::BestEffort`] on a filesystem (such as some network mounts)
    /// where the OS lock is unavailable. Setting `HDF5_USE_FILE_LOCKING` in the
    /// environment overrides this, as in the C library.
    ///
    /// Readers and [`File::open_swmr_writer`] take no lock by design and ignore
    /// this.
    #[doc(alias = "H5Pset_file_locking")]
    pub const fn with_locking(mut self, locking: FileLocking) -> Self {
        self.locking = locking;
        self
    }

    /// Set how much memory a read-write open may use to hold the file.
    ///
    /// Unset by default, which lets the entry point choose:
    /// [`File::open_rw`] uses [`MemoryStrategy::Auto`], preferring the bounded
    /// engine and falling back to the whole-file mirror for a file it cannot
    /// edit. Setting this overrides that default, in either direction, so
    /// [`MemoryStrategy::Bounded`] refuses such a file rather than quietly
    /// spending `O(file size)` memory on a caller who asked not to;
    /// [`MemoryStrategy::Mirrored`] takes
    /// the whole-file mirror unconditionally, as `open_rw` did before it learned
    /// to dispatch.
    ///
    /// The read-only opens ignore this: they build no editing session at all, and
    /// their own names say what memory they spend. [`File::open_swmr_writer`]
    /// does build one, and always mirrors: it accepts
    /// [`MemoryStrategy::Auto`] and [`MemoryStrategy::Mirrored`], both of which
    /// the mirror satisfies, and refuses an explicit [`MemoryStrategy::Bounded`]
    /// with [`Error::EditUnsupported`] rather than quietly not honoring it. Ask a
    /// `File` which backend it resolved to with [`File::edit_backing`].
    pub const fn with_memory_strategy(mut self, memory_strategy: MemoryStrategy) -> Self {
        self.memory_strategy = Some(memory_strategy);
        self
    }

    /// Constrain the on-disk format an editing session may write, mirroring
    /// HDF5's `H5Pset_libver_bounds` — which the C library classes as a *file
    /// access* property for exactly this reason: it governs what a later write
    /// to an existing file is allowed to add.
    ///
    /// Unset by default, which keeps [`File::open_rw`] adding whatever the
    /// content needs. That default is what lets a file the C library wrote under
    /// its own bounds be edited at all, but it means a session can add content
    /// only a newer library can read *without changing the superblock*, and the
    /// caller has no way to see it happen: adding a chunked, filtered, or
    /// resizable dataset to an HDF5 1.8 file needs the version 4 data-layout
    /// message and a 1.10 chunk index, since this crate does not write the
    /// version 1 B-tree index that 1.8 used.
    ///
    /// Setting a `high` below [`LibVer::V110`] refuses that addition with
    /// [`FormatError::LibverTooOldForContent`]
    /// at [`File::commit`] instead — the same refusal
    /// [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds)
    /// gives when writing a whole file, so a `.mat` bounded to 1.8 for MATLAB
    /// stays loadable by MATLAB after an edit.
    ///
    /// `low` only rules formats out — as in the C library it licenses newer
    /// encodings without requiring them — so a lower bound of [`LibVer::V112`],
    /// [`LibVer::V114`] or [`LibVer::LATEST`] leaves the session writing the 1.10
    /// format rather than failing, provided `high` reaches it. An inverted range
    /// such as `V114..=V110` is refused with
    /// [`FormatError::LibverBoundsUnsatisfiable`].
    ///
    /// The read-only opens ignore this: they write nothing. [`File::open_swmr_writer`]
    /// requires a version 3 superblock, so it refuses a `high` below
    /// [`LibVer::V110`] up front rather than accepting a bound it cannot honor.
    #[doc(alias = "H5Pset_libver_bounds")]
    pub const fn with_libver_bounds(mut self, low: LibVer, high: LibVer) -> Self {
        self.libver_bounds = Some((low, high));
        self
    }

    /// Selects the durability policy for a read-write editing session.
    ///
    /// The default is [`SyncPolicy::Always`]. [`SyncPolicy::OnClose`] defers durability until
    /// [`File::sync`] or session closure. Immediate appends issue their writes in publication
    /// order under either policy, including with an explicit page buffer. Page-buffered staged
    /// commits can retain writes after returning.
    ///
    /// Read-only opens ignore this setting.
    pub const fn with_sync_policy(mut self, sync_policy: SyncPolicy) -> Self {
        self.sync_policy = sync_policy;
        self
    }

    /// Sets the byte budget for retaining writes across operations.
    ///
    /// Defaults to `0`, which uses gathering within each operation. A nonzero budget permits staged
    /// commit writes to remain buffered across operations and general ordering barriers.
    /// Immediate appends drain the buffer at allocation and reference publication boundaries,
    /// before publishing their dataspace growth. A successful append has issued its writes to
    /// the operating system, without requesting power-loss durability under [`SyncPolicy::OnClose`].
    ///
    /// Pending ranges within a page are combined when flushed. Reaching the byte budget drains
    /// every pending range, so a smaller budget can increase writes within a long append batch.
    /// The resulting file bytes are independent of the budget.
    ///
    /// The editor marks the superblock for the session's lifetime and clears the mark during
    /// successful closure. An interrupted session can leave staged commit references published
    /// before their content. Opens normally return [`Error::FileMarkedInUse`] while the mark
    /// remains. [`File::clear_status_flags`] permits inspection after the writer stops, without
    /// guaranteeing data integrity.
    ///
    /// # Requirements
    ///
    /// A writable open returns [`Error::EditUnsupported`] for any of these configurations:
    ///
    /// - A budget smaller than the file-space page size for [`FileSpaceStrategy::Page`], or
    ///   smaller than 4 KiB for other strategies
    /// - PAGE storage without persisted free space
    /// - A superblock older than version 3
    /// - [`SyncPolicy::Always`]
    /// - A SWMR writer
    ///
    /// [`File::create_with_options`] rejects an incompatible creation/access pair before
    /// writing the file. Read-only opens ignore this setting.
    #[doc(alias = "H5Pset_page_buffer_size")]
    pub const fn with_page_buffer_size(mut self, bytes: usize) -> Self {
        self.page_buffer_size = bytes;
        self
    }

    /// Let a read-only open proceed past a superblock marked open for write by a
    /// writer that is not a SWMR writer — status-flag bit 0 alone, which is what
    /// [`with_page_buffer_size`](Self::with_page_buffer_size) raises for a
    /// session's whole life.
    ///
    /// Defaults to [`WriteMarkPolicy::Refuse`], which is what `H5Fopen` does with
    /// the same byte: [`File::open`], [`File::open_streaming`] and
    /// [`File::from_source`] all report [`Error::FileMarkedInUse`].
    /// [`WriteMarkPolicy::AllowSnapshot`] reads the file as it stands instead,
    /// through whichever of those opens is passed these properties.
    ///
    /// # What the caller is asserting
    ///
    /// That the writer has flushed: it called [`File::sync`], or it stopped
    /// after a flush and the mark stands only because nothing cleared it (a
    /// clean [`File::close`] takes the mark down, and leaves nothing to opt past).
    /// The mark is durable and says nothing about *when* — a live writer
    /// mid-operation and one that exited without closing carry the same byte —
    /// so this crate cannot check the assertion, and passing this value is how a
    /// caller states it. It is exactly true for a writer under
    /// [`SyncPolicy::OnClose`] that syncs at the
    /// points it wants readable, and it is what the mark exists to guard against
    /// when it is false: page-buffered staged commit references can be written
    /// before their content, so a snapshot taken during flushing can show a
    /// dataset that reads clean and returns fill values, with every checksum
    /// verifying.
    ///
    /// The snapshot is of the bytes on disk at open. A buffered open takes it
    /// whole; a streaming open reads regions on demand, so a writer that carries
    /// on writing can move bytes under it — reach for
    /// [`File::open_with_options`] when the writer may continue, and for
    /// [`File::open_streaming_with_options`] when it has stopped and the file is
    /// too large to buffer.
    ///
    /// # What it does not unlock
    ///
    /// - a **SWMR pair** (both bits): that file has a reader of its own, and
    ///   [`File::open_swmr`] follows it — including across the writer's later
    ///   appends, which a snapshot cannot;
    /// - [`File::open_rw`] and [`File::open_swmr_writer`], which are refused
    ///   whatever this says. A second writer must not join a file a writer
    ///   already holds;
    /// - the OS advisory lock, a separate guard with its own policy
    ///   ([`with_locking`](Self::with_locking)).
    ///
    /// A file left marked by a writer that *crashed* is a different question,
    /// and this is not the answer to it: it reads such a file as willingly as a
    /// flushed one, and leaves the mark standing for the next reader to meet.
    /// [`File::clear_status_flags`] clears the mark after the writer has stopped, without
    /// guaranteeing data integrity.
    ///
    /// libhdf5 offers no counterpart: `H5Fopen` rejects the byte with no
    /// override, and `h5clear` is its only way through. This is the narrower one,
    /// since it changes nothing on disk.
    pub const fn with_write_mark_policy(mut self, policy: WriteMarkPolicy) -> Self {
        self.write_mark_policy = policy;
        self
    }

    /// Sets how writable opening handles persisted free-space metadata.
    ///
    /// Defaults to [`PersistedFreeSpacePolicy::Tolerant`]. With
    /// [`PersistedFreeSpacePolicy::Strict`], a metadata parsing failure leaves the file's bytes,
    /// length, and status flags unchanged, including when a page buffer is configured through
    /// [`with_page_buffer_size`](Self::with_page_buffer_size).
    ///
    /// Read-only opening ignores this setting. The selected policy is available through
    /// [`File::access_properties`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileAccessProperties, PersistedFreeSpacePolicy};
    ///
    /// # let path = test_util::temp::temp_path("strict-open.h5");
    /// # hdf5_pure::FileBuilder::new().write(&path)?;
    /// let properties = FileAccessProperties::new()
    ///     .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict);
    /// let file = File::open_rw_with_options(&path, properties)?;
    /// assert_eq!(
    ///     file.access_properties().persisted_free_space_policy(),
    ///     PersistedFreeSpacePolicy::Strict,
    /// );
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub const fn with_persisted_free_space_policy(
        mut self,
        policy: PersistedFreeSpacePolicy,
    ) -> Self {
        self.persisted_free_space_policy = policy;
        self
    }

    /// Return the configured streaming metadata cache.
    pub const fn metadata_cache(&self) -> MetadataCacheConfig {
        self.metadata_cache
    }

    /// Return the configured library-version bounds, or `None` when an editing
    /// session may write whatever its content needs.
    pub const fn libver_bounds(&self) -> Option<(LibVer, LibVer)> {
        self.libver_bounds
    }

    /// Return the configured per-dataset chunk cache.
    pub const fn chunk_cache(&self) -> ChunkCacheConfig {
        self.chunk_cache
    }

    /// Return the configured file-locking policy.
    pub const fn locking(&self) -> FileLocking {
        self.locking
    }

    /// Return the configured memory strategy, or `None` when none was asked for
    /// and the entry point's own default applies. This is what was *requested*;
    /// for which backend an open resolved to, see [`File::edit_backing`].
    ///
    /// The `Option` distinguishes "no preference stated" from an explicit
    /// [`MemoryStrategy::Auto`], which is what lets an entry point supply its own
    /// default without overriding a caller who asked for one; `None` resolves to
    /// [`MemoryStrategy::Auto`], the only default any entry point now supplies
    /// rather than as a second break on this accessor.
    pub const fn memory_strategy(&self) -> Option<MemoryStrategy> {
        self.memory_strategy
    }

    /// Return the configured `fsync` policy.
    pub const fn sync_policy(&self) -> SyncPolicy {
        self.sync_policy
    }

    /// Return the configured page-buffer budget in bytes; `0` when none was
    /// asked for.
    pub const fn page_buffer_size(&self) -> usize {
        self.page_buffer_size
    }

    /// Return the configured write-mark policy.
    pub const fn write_mark_policy(&self) -> WriteMarkPolicy {
        self.write_mark_policy
    }

    /// Returns the configured policy for persisted free-space recovery during writable opening.
    pub const fn persisted_free_space_policy(&self) -> PersistedFreeSpacePolicy {
        self.persisted_free_space_policy
    }
}

/// Dataset-access properties applied when opening a single dataset.
///
/// This is the `hdf5-pure` analogue of an HDF5 **dataset access property list**
/// (`dapl`). Its chunk cache corresponds to `H5Pset_chunk_cache`: it overrides,
/// for this one dataset, the file-wide chunk-cache default configured with
/// [`FileAccessProperties::with_chunk_cache`] (the `H5Pset_cache` analogue). When
/// left unset, the dataset inherits that file-wide default — matching the `dapl`
/// default sentinels (`H5D_CHUNK_CACHE_*_DEFAULT`), which also mean "use the
/// file's setting".
///
/// The `Properties` suffix means the type stands in for one whole HDF5 property
/// list, so every setting on it has a C counterpart to look up. It is a stand-in
/// and not a port: a plain `Copy` value, with no handle to create or close, no
/// runtime property registry, and no setter that can fail. `dapl` and each
/// `H5Pset_*` it models are doc aliases, so a search for either lands here.
/// The chunk cache is the one `dapl` property modeled, and
/// [Supported properties](#supported-properties) has the rest.
///
/// [`ChunkCacheConfig`] maps `H5Pset_chunk_cache`'s `rdcc_nslots` and
/// `rdcc_nbytes`; its `rdcc_w0` preemption policy is not modeled, for the reason
/// on [`ChunkCacheConfig::from_h5p_cache`].
///
/// Pass it to [`File::dataset_with_options`] or [`Group::dataset_with_options`].
///
/// # Supported properties
///
/// Passed to [`File::dataset_with_options`](crate::File::dataset_with_options) /
/// [`Group::dataset_with_options`](crate::Group::dataset_with_options) as a
/// [`DatasetAccessProperties`](Self), overriding the file-wide access defaults for one dataset. The
/// status column is explained by the [status legend](crate::FileCreateProperties#status-legend).
///
/// | HDF5 property (C API) | `hdf5-pure` | Status | Behavior |
/// |---|---|---|---|
/// | `H5Pset_chunk_cache` | [`with_chunk_cache`](Self::with_chunk_cache) | **Genuine** | Overrides the file-wide chunk cache for this dataset only. Unset means inherit, matching the `H5D_CHUNK_CACHE_*_DEFAULT` sentinels. `rdcc_nslots` and `rdcc_nbytes` map directly, and `rdcc_w0` is not modeled, for the reason under [The chunk cache](crate::FileAccessProperties#the-chunk-cache-rdcc). |
/// | `H5Pset_efile_prefix` / `H5Pset_virtual_prefix` | none | **Unsupported** | External and virtual datasets are not resolved, so there is no prefix to set. |
/// | `H5Pset_virtual_view` / `H5Pset_virtual_printf_gap` | none | **Unsupported** | Virtual datasets are not supported. |
/// | `H5Pset_append_flush` | none | **Unsupported** | There is no append callback and no per-boundary flush, and [`Dataset::append`](crate::Dataset::append) flushes on its own schedule. |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[doc(alias = "dapl")]
pub struct DatasetAccessProperties {
    chunk_cache: Option<ChunkCacheConfig>,
}

impl DatasetAccessProperties {
    /// A value that inherits every file-wide access default.
    pub const fn new() -> Self {
        Self { chunk_cache: None }
    }

    /// Override the raw chunk cache for this one dataset, ignoring the file-wide
    /// default. This is the `H5Pset_chunk_cache` analogue.
    #[doc(alias = "H5Pset_chunk_cache")]
    pub const fn with_chunk_cache(mut self, chunk_cache: ChunkCacheConfig) -> Self {
        self.chunk_cache = Some(chunk_cache);
        self
    }

    /// Return the chunk-cache override, or `None` when the dataset inherits the
    /// file-wide default.
    pub const fn chunk_cache(&self) -> Option<ChunkCacheConfig> {
        self.chunk_cache
    }

    /// Resolve the effective chunk-cache config: the per-dataset override if one
    /// was set, otherwise the file-wide `default`.
    const fn resolved_chunk_cache(&self, default: ChunkCacheConfig) -> ChunkCacheConfig {
        match self.chunk_cache {
            Some(config) => config,
            None => default,
        }
    }
}

/// Returns `Ok(true)` if the file at `path` contains the HDF5 signature at an address where a
/// superblock may start.
///
/// Reads at most 15 bytes at byte 0, byte 512, and each address twice the previous one, up to the
/// signature or the end of the file. Returns `Ok(true)` for a file whose bytes after the signature
/// are truncated or malformed. [`File::open`] parses the superblock and the rest of the file.
/// `H5Fis_accessible` makes the same check.
///
/// # Errors
///
/// Returns an [`io::Error`](std::io::Error) if opening the file, seeking to its end, or reading
/// it fails.
pub fn is_hdf5<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<bool> {
    let handle = std::fs::File::open(path)?;
    let source = ReadSeekSource::new(handle).map_err(std::io::Error::other)?;
    superblock::find_signature(&source)
        .map(|address| address.is_some())
        .map_err(std::io::Error::other)
}

/// Returns `true` if `data` contains the HDF5 signature at an address where a superblock may
/// start.
///
/// This is the in-memory counterpart of [`is_hdf5`].
pub fn is_hdf5_bytes(data: &[u8]) -> bool {
    matches!(superblock::find_signature(data), Ok(Some(_)))
}

/// An open HDF5 file for reading.
struct FileInner {
    backend: Backend,
    /// The mode every object header parse of this file is made under:
    /// [`AccessMode::ReadWrite`] for an editing session and for a repack that parses
    /// its source as a writer does, [`AccessMode::ReadOnly`] for every other open.
    /// Fixed at the open, so the backend a later [`File::refresh`] swaps in cannot
    /// change it.
    access_mode: AccessMode,
    superblock: Superblock,
    /// The address and length widths validated while parsing the superblock.
    widths: FormatWidths,
    /// The effective address base and EOF derived from the physical signature position.
    opened: OpenedImageContext,
    /// Live file handle, retained only when the file was opened with
    /// [`File::open_swmr`] so [`File::refresh`] can re-read appended data.
    handle: Option<std::fs::File>,
    /// File Space Info parsed from the superblock extension, if the file records
    /// one. Best-effort: a malformed or unreadable extension leaves this `None`
    /// rather than failing the open.
    file_space_info: Option<FileSpaceInfo>,
    /// The shared object header message (SOHM) table, if the superblock
    /// extension records one. `None` for almost every file: no common producer
    /// enables shared-message indexes. Best-effort like `file_space_info`, so a
    /// file whose table cannot be read still opens and still reads every object
    /// that shares nothing; what fails is following a reference into the heap,
    /// with [`FormatError::UnsupportedSohmReference`].
    ///
    /// Boxed because it is absent on essentially every file, and this struct is
    /// allocated once per open: one pointer costs less here than the table
    /// inline, and the absent case allocates nothing at all.
    sohm_table: Option<Box<crate::sohm::SohmTable>>,
    access_properties: FileAccessProperties,
    /// Set by [`File::close`] to seal a read-write file: after it, a write
    /// through any surviving [`Dataset`]/[`Group`] handle or [`File`] clone
    /// returns [`Error::FileClosed`]. Reads still work. Only ever set on a
    /// `Backend::Edit` file.
    closed: AtomicBool,
    /// How many times a write session has been given the chance to change this
    /// file's bytes, counted so an owned [`Dataset`] handle can tell that the
    /// object header it parsed no longer says what the file says.
    ///
    /// Advanced by every operation that reaches the write engine; see
    /// [`FileInner::with_engine_mut`], which is the only thing that advances
    /// either counter. Never advances for a read-only or streaming file, whose
    /// bytes cannot move under a handle at all.
    content_revision: AtomicU64,
    /// How many times a write session has been given the chance to *move* an
    /// object header, which is the narrower question of whether an address a
    /// handle is holding still names its object.
    ///
    /// [`Change::InPlace`] leaves this alone: an immediate [`Dataset::append`]
    /// rewrites a dataset's header where it stands, so an address stays good
    /// across one. That is what lets a handle reached by object reference — the
    /// one kind with no name to look itself up by — go on appending, while a
    /// commit ends it.
    address_revision: AtomicU64,
    /// True for a file opened with [`File::open_swmr_writer`]: no OS lock is held,
    /// the superblock's SWMR-write flag is raised, only immediate
    /// [`Dataset::append`] is permitted (the staged surface is refused), and the
    /// flag is cleared on [`File::close`] / `Drop`. `false` for every other file.
    swmr_write: bool,
}

impl Drop for FileInner {
    /// Attempts finalization and synchronization for an unclosed editing session.
    ///
    /// Discards staged edits. Releases writer status flags only after successful synchronization.
    /// For a session rejected with [`Error::AllocationPublicationUncertain`], the editor discards
    /// queued writes and rejects cleanup. During recoverable nonpersistent PAGE publication, it
    /// retains pending writes for verification and recovery.
    fn drop(&mut self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        let Backend::Edit(m) = &self.backend else {
            return;
        };
        let mut session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Both branches write, and this is the last moment anything can order
        // those writes: the handle is gone once this returns, so `File::sync` is
        // not an option the caller still has. The barrier is therefore forced
        // rather than left to the session's `SyncPolicy` — see
        // [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose).
        // SWMR stages nothing and persists no free space, so it skips the
        // re-homing; everything after that is the same teardown for both.
        if !self.swmr_write {
            let _ = session.finalize_persist();
        }
        // Only if the flush actually succeeded. The flags say this session's
        // writes may still be in memory, and a failed `force_sync` is precisely
        // the case where that is still true — taking them down there would
        // publish the file as complete over a drain that did not finish.
        if session.force_sync().is_ok() {
            let _ = session.release_status_flags();
        }
    }
}

impl FileInner {
    /// Opens the HDF5 file at `path` and reads it into memory.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_with_options(path, FileAccessProperties::new())
    }

    /// Open an HDF5 file from a filesystem path with explicit access properties.
    ///
    /// Like [`open`](Self::open), this buffers the whole file in memory. Use
    /// [`open_streaming_with_options`](Self::open_streaming_with_options) when
    /// the metadata cache budget should apply to lazy metadata reads.
    pub fn open_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
        let write_mark = properties.write_mark_policy;
        let inner = Self::from_bytes_with_options(bytes, properties)?;
        // The status-flag check belongs to every open that reads the live file
        // — this one, `open_streaming` and `from_source` — and not to
        // `from_bytes_with_options` (issue #245). The line is snapshot against
        // live, not path against no path: a caller who already holds the bytes
        // has taken its own snapshot, and there is no live file left to
        // coordinate over. This is a deliberate divergence from the C library,
        // which checks under its in-memory core driver too.
        file_lock::check_status_flags(
            &inner.superblock,
            OpenIntent::Read(write_mark),
            OpenTarget::Path(path.as_ref()),
        )?;
        Ok(inner)
    }

    /// Open an HDF5 file for **streaming** reads, fetching regions on demand from
    /// the file instead of buffering it whole.
    ///
    /// This lets a host read a file larger than its address space — the original
    /// motivation being 32-bit targets reading multi-gigabyte files (issue #27).
    /// Metadata and dataset chunks are read through a `ReadSeekSource`, so peak
    /// memory stays close to one chunk plus the metadata being parsed. Chunks
    /// that sit next to each other on disk are fetched together, in reads of at
    /// most 256 KiB (a larger chunk is read on its own), which is what makes a
    /// file written a row at a time — thousands of chunks of a few dozen bytes
    /// — read at a sensible speed. See [`crate::chunk_span`].
    ///
    /// Reads match the buffered [`File::open`]: every storage layout and chunk
    /// index type, both group forms (v2 and v1 symbol-table), and compact,
    /// dense, shared, and variable-length attributes. What differs:
    /// [`as_bytes`](Self::as_bytes) returns an empty slice (there is no
    /// whole-file buffer), a streaming file cannot be the *source* of a
    /// cross-file copy, and chunk decompression is sequential (the `parallel`
    /// feature accelerates only buffered reads).
    pub fn open_streaming<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_streaming_with_options(path, FileAccessProperties::new())
    }

    /// Open an HDF5 file for streaming reads with explicit access properties.
    pub fn open_streaming_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let handle = std::fs::File::open(path.as_ref()).map_err(Error::Io)?;
        let source = ReadSeekSource::new(handle).map_err(Error::Format)?;
        Self::streaming(
            source,
            AccessMode::ReadOnly,
            properties,
            OpenTarget::Path(path.as_ref()),
        )
    }

    /// Opens the file at `path` for streaming reads whose object header parses are
    /// made under `access_mode`. A parse under [`AccessMode::ReadWrite`] rejects a
    /// message of a type this crate cannot name that a decoder with write access
    /// must understand.
    pub(crate) fn open_streaming_with_access_mode(
        path: &std::path::Path,
        access_mode: AccessMode,
    ) -> Result<Self, Error> {
        let handle = std::fs::File::open(path).map_err(Error::Io)?;
        let source = ReadSeekSource::new(handle).map_err(Error::Format)?;
        Self::streaming(
            source,
            access_mode,
            FileAccessProperties::new(),
            OpenTarget::Path(path),
        )
    }

    /// Open an HDF5 file from any [`Source`], reading metadata and chunks on
    /// demand as [`open_streaming`](Self::open_streaming) does.
    pub fn from_source<S: Source + Send + Sync + 'static>(source: S) -> Result<Self, Error> {
        Self::from_source_with_options(source, FileAccessProperties::new())
    }

    /// Open an HDF5 file from any [`Source`] with explicit access properties.
    pub fn from_source_with_options<S: Source + Send + Sync + 'static>(
        source: S,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        // The only source that reaches the parsers from outside the crate, and
        // so the only one whose reads are length-checked: see `ValidatedSource`.
        Self::streaming(
            ValidatedSource::new(source),
            AccessMode::ReadOnly,
            properties,
            OpenTarget::Source,
        )
    }

    /// The body every streaming open shares.
    ///
    /// Wraps the source in whatever metadata cache the properties ask for,
    /// parses the superblock through it, and rejects a file a writer holds.
    /// The three differ in where the source came from, in `access_mode`, and in
    /// `target`, which does nothing but name the file in that rejection. One
    /// body is what keeps the rest of it from drifting apart.
    fn streaming<S: Source + Send + Sync + 'static>(
        source: S,
        access_mode: AccessMode,
        properties: FileAccessProperties,
        target: OpenTarget<'_>,
    ) -> Result<Self, Error> {
        let source: Box<dyn Source + Send + Sync> = if properties.metadata_cache.is_enabled() {
            Box::new(MetadataCachingSource::new(
                source,
                properties.metadata_cache,
            ))
        } else {
            Box::new(source)
        };
        let (superblock, widths, opened) = Self::parse_superblock(source.as_ref())?;
        opened.require_complete(source.len())?;
        file_lock::check_status_flags(
            &superblock,
            OpenIntent::Read(properties.write_mark_policy),
            target,
        )?;
        Ok(Self::from_parts(
            Backend::Streaming(source),
            access_mode,
            superblock,
            widths,
            opened,
            None,
            properties,
        ))
    }

    /// Open an HDF5 file for SWMR (single-writer/multiple-reader) reading.
    ///
    /// Like [`File::open`], but retains a live handle to the file so that
    /// [`File::refresh`] can re-read data appended by a concurrent writer
    /// (whether produced by this crate's append writer, the reference HDF5 C
    /// library, or h5py in SWMR mode). The initial view is a consistent
    /// snapshot; call [`File::refresh`] to advance to a newer one.
    ///
    /// Only the `std` build supports this (it requires a live filesystem
    /// handle); the in-memory [`File::from_bytes`] path cannot refresh.
    pub fn open_swmr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_swmr_with_options(path, FileAccessProperties::new())
    }

    /// Open an HDF5 file for SWMR reading with explicit access properties.
    ///
    /// SWMR reads currently keep an in-memory mirror for refresh semantics, so
    /// only the per-dataset chunk-cache settings affect this backend.
    pub fn open_swmr_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let mut handle = std::fs::File::open(path.as_ref()).map_err(Error::Io)?;
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;
        let (superblock, widths, opened) = Self::parse_superblock(data.as_slice())?;
        if superblock.version() != SWMR_SUPERBLOCK_VERSION {
            opened.require_complete(data.len() as u64)?;
        }
        file_lock::check_status_flags(
            &superblock,
            OpenIntent::SwmrRead,
            OpenTarget::Path(path.as_ref()),
        )?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            AccessMode::ReadOnly,
            superblock,
            widths,
            opened,
            Some(handle),
            properties,
        ))
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Self::from_bytes_with_options(data, FileAccessProperties::new())
    }

    /// Open an HDF5 file from an in-memory byte vector with explicit access properties.
    pub fn from_bytes_with_options(
        data: Vec<u8>,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let (superblock, widths, opened) = Self::parse_superblock(data.as_slice())?;
        opened.require_complete(data.len() as u64)?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            AccessMode::ReadOnly,
            superblock,
            widths,
            opened,
            None,
            properties,
        ))
    }

    /// Open an existing HDF5 file for reading **and** in-place editing, applying
    /// `properties` (its [`FileLocking`] policy governs the OS file lock held for
    /// the file's life, and its chunk cache is the file-wide default).
    fn open_rw<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Self::open_rw_with_default(path, properties, MemoryStrategy::Auto)
    }

    /// Open read-write under the properties' memory strategy, falling back to
    /// `default` when the caller expressed none. The two public read-write entry
    /// [`File::open_rw`] passes [`MemoryStrategy::Auto`]: prefer the bounded
    /// engine, but take the mirror for a file the bounded engine cannot edit
    /// (issue #198, step 4). A caller who states a strategy overrides it.
    fn open_rw_with_default<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
        default: MemoryStrategy,
    ) -> Result<Self, Error> {
        let session = WriteEngine::open_rw_with_strategy(
            path.as_ref(),
            properties.metadata_cache,
            properties.locking,
            properties.memory_strategy.unwrap_or(default),
            properties.persisted_free_space_policy,
        )?;
        Self::from_rw_session(session, properties)
    }

    /// Wrap an opened [`WriteEngine`] as a read-write [`Backend::Edit`] file.
    fn from_rw_session(
        mut session: WriteEngine,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        // The one funnel every read-write session passes through, so the fapl's
        // format bound and `fsync` cadence reach the engine no matter which entry
        // point opened it — the SWMR writer included, which
        // `WriteEngine::open_swmr_writer` says why.
        session.set_libver_bounds(properties.libver_bounds)?;
        session.set_sync_policy(properties.sync_policy);
        // Every page-buffer refusal lives in `set_page_buffer_size`, including the
        // `SyncPolicy::Always` one — which is why `set_sync_policy` must precede
        // this call rather than merely happening to.
        session.set_page_buffer_size(properties.page_buffer_size)?;
        // Reuse the parsed superblock so the image need not provide a whole-file slice.
        let superblock = session.superblock().clone();
        let widths = session.widths();
        let opened = OpenedImageContext::reconcile(
            superblock.base_address,
            superblock.eof_address,
            session.addresses().effective_base().get(),
        )?;
        Ok(Self::from_parts(
            Backend::Edit(Box::new(Mutex::new(session))),
            AccessMode::ReadWrite,
            superblock,
            widths,
            opened,
            None,
            properties,
        ))
    }

    /// Open for SWMR writing: no OS lock, superblock SWMR-write flag raised.
    fn open_swmr_writer<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        // The SWMR writer always mirrors. `Auto` and unset are *satisfied* by
        // that — they ask for the bounded engine where it applies and accept the
        // mirror where it does not — but `Bounded` is a guarantee, and honoring a
        // guarantee by ignoring it is how a caller ends up spending `O(file size)`
        // memory it asked not to. Refusing is also the permissive direction to be
        // wrong in: if this writer ever runs bounded, the refusal stops firing,
        // which breaks nobody.
        if properties.memory_strategy == Some(MemoryStrategy::Bounded) {
            return Err(Error::EditUnsupported(
                "the SWMR writer always holds the file in a whole-file mirror; leave \
                 MemoryStrategy unset, or pass MemoryStrategy::Auto or MemoryStrategy::Mirrored, \
                 to open it",
            ));
        }
        // A library-version bound below 1.10 is the same shape of unhonorable
        // guarantee. SWMR needs a version 3 superblock — neither library reads
        // the SWMR-write flag back on an older one — so a caller asking for the
        // 1.8 format here is asking for a file this writer cannot produce.
        if let Some((low, high)) = properties.libver_bounds
            && LibVer::resolve_writable(Some((low, high))).map_err(Error::Format)? < LibVer::V110
        {
            return Err(Error::EditUnsupported(
                "the SWMR writer requires a version 3 superblock, which is the v1.10 format; \
                 raise the FileAccessProperties library-version bound to open it",
            ));
        }
        // And a page buffer is the third. A SWMR reader follows the writer's
        // ordered phases as they become visible, so coalescing those writes is
        // not a slower or larger file but a reader that sees a state the phases
        // exist to keep it from seeing.
        if properties.page_buffer_size != 0 {
            return Err(Error::EditUnsupported(
                "the SWMR writer cannot buffer its writes: its readers observe the order they \
                 become visible in; leave FileAccessProperties::with_page_buffer_size unset to \
                 open it",
            ));
        }
        let session = WriteEngine::open_swmr_writer(path, properties.sync_policy)?;
        let mut inner = Self::from_rw_session(session, properties)?;
        inner.swmr_write = true;
        Ok(inner)
    }

    /// After the caller has confirmed a [`Backend::Edit`] backend, gate the
    /// mutation: refuse a sealed file with [`Error::FileClosed`], and in
    /// SWMR-writer mode refuse a staged edit (`staged = true`) with
    /// [`Error::SwmrStagedUnsupported`] — only immediate appends are allowed.
    fn check_mutable(&self, staged: bool) -> Result<(), Error> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::FileClosed);
        }
        if staged && self.swmr_write {
            return Err(Error::SwmrStagedUnsupported);
        }
        Ok(())
    }

    /// Gate a staged edit *without* taking the session lock: the backend must
    /// offer the staged surface, and the file must still be mutable.
    ///
    /// This is the same gate the locking helpers apply before locking, split out
    /// so a public method taking a user closure can report a read-only or sealed
    /// file up front, run the closure with no lock held, and take the lock only
    /// to record the result (issue #200).
    fn check_staged_writable(&self) -> Result<(), Error> {
        match &self.backend {
            Backend::Edit(_) => self.check_mutable(true),
            _ => Err(Error::ReadOnly),
        }
    }

    /// Runs an operation against the editing session and updates handle revisions.
    ///
    /// The caller passes the expected effect as `change`. If the engine adopts a different root,
    /// the editor records [`Change::Relocating`] even when the caller passed [`Change::Nothing`].
    /// Handles opened by path then resolve their objects again, and address-only handles become
    /// stale. A relocating or in-place change advances revisions even if `f` returns an error
    /// or only records a staged edit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file has no editing session, or the mutation-gate error
    /// if it cannot be edited. Returns the error from `f` if the operation fails.
    fn with_engine_mut<R>(
        &self,
        change: Change,
        f: impl FnOnce(&mut WriteEngine) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let Backend::Edit(m) = &self.backend else {
            return Err(Error::ReadOnly);
        };
        self.check_mutable(matches!(change, Change::Relocating))?;
        let (out, change) = {
            let mut engine = m.lock().unwrap_or_else(PoisonError::into_inner);
            let root = engine.superblock().root_group_address;
            let out = f(&mut engine);
            let change = if root == engine.superblock().root_group_address {
                change
            } else {
                Change::Relocating
            };
            (out, change)
        };
        self.note(change);
        out
    }

    /// Record that an operation of kind `change` has run against this file.
    ///
    /// The address counter moves *first*, against [`revisions`](Self::revisions)
    /// reading it second. A reader that sees the new content revision has
    /// therefore already synchronized with this release, so it cannot then read
    /// an address revision from before it and conclude that its memoized address
    /// outlived a commit that moved it.
    fn note(&self, change: Change) {
        match change {
            Change::Relocating => {
                self.address_revision.fetch_add(1, Ordering::Release);
                self.content_revision.fetch_add(1, Ordering::Release);
            }
            Change::InPlace => {
                self.content_revision.fetch_add(1, Ordering::Release);
            }
            Change::Nothing => {}
        }
    }

    /// How many times a write session has been given the chance to change this
    /// file's bytes. A [`Dataset`] handle whose header was parsed at this value
    /// still holds what the file holds.
    fn content_revision(&self) -> u64 {
        self.content_revision.load(Ordering::Acquire)
    }

    /// How many times a write session has been given the chance to move an
    /// object header. An address worked out at this value still names its
    /// object.
    fn address_revision(&self) -> u64 {
        self.address_revision.load(Ordering::Acquire)
    }

    /// Work out where a handle's object header sits now, and say what that
    /// answer holds as of.
    ///
    /// One rule, in the order it is written. An address nothing has moved since
    /// still names its object, whichever way the handle names it — which is what
    /// keeps an in-place append anywhere in the file from making every handle
    /// walk its path again. Past that, a handle opened by `path` looks its
    /// object up, which follows it wherever a commit put it, and one reached by
    /// object reference has no name to look up: `memo` held the only address it
    /// will ever have, and the bytes a relocated header vacates still parse as
    /// the object that left them, so there is nothing left to read.
    ///
    /// Both counters are read *before* the resolution, never after: a value
    /// taken afterwards could name a state the address predates, and a handle
    /// that memoized that pairing would go on answering from a header a commit
    /// had already moved. Taken beforehand, a concurrent change makes the
    /// pairing merely stale — which the next use notices, so long as it is the
    /// next *use*. A commit running on another thread between this read and the
    /// bytes it labels is not ordered against either, so a handle shared across
    /// threads can still serve one read from a header a concurrent commit had
    /// moved. What these counters order is a handle against edits already made,
    /// not against one in flight. The callers that classify what they resolved
    /// to — [`Dataset::resolved`], [`Group::header_address`] — parse that header
    /// in the same unordered window, so a commit landing inside it can also make
    /// a live handle report [`Error::NotADataset`] or
    /// [`Error::NotAGroup`] for an object whose kind
    /// never changed. That is the same staleness reporting itself instead of
    /// answering, which is the better half of the trade.
    fn locate(
        &self,
        path: Option<&ObjectPathBuf>,
        memo: Option<Resolution>,
    ) -> Result<Resolution, Error> {
        let revisions = self.revisions();
        // A handle with no memo has never resolved, so its path is walked until it has a physical
        // address to memoize.
        Ok(revisions.at(match (path, memo) {
            (_, Some(memo)) if revisions.address == memo.address_revision => memo.address,
            (Some(path), _) => self.resolve_path(&path.as_path())?,
            (None, _) => return Err(Error::StaleHandle),
        }))
    }

    /// [`locate`](Self::locate), reported as [`Error::NotCommitted`] when this
    /// session has the path *staged* rather than written, and as
    /// [`Error::StagingWithdrawn`] when the staging a handle was born onto is
    /// gone.
    ///
    /// Every handle onto a staged object comes through here, so the distinction
    /// between "there is no such object", "there is one, and `commit` has not
    /// written it yet" and "the one this handle names has been withdrawn" is
    /// made in one place rather than at each caller.
    ///
    /// The staged set is consulted *before* the file for a handle that has never
    /// resolved, because a staged creation can sit on a path the file still
    /// holds: a delete and a create in one commit replace the object there
    /// (issue #305), and until that commit runs the old bytes are still
    /// perfectly readable. Answering from them would hand the caller the object
    /// their handle was explicitly not opened onto.
    ///
    /// `birth` is the handle's own [`Standing`] input — `None` for one opened
    /// onto an object in the file. This question is keyed by path alone, so a
    /// born handle whose creation was withdrawn and replaced by one of the other
    /// kind at the same path reports `NotCommitted` here where
    /// [`staged_dataset_view`](Self::staged_dataset_view) tells it apart; both
    /// refuse, and neither answers from the file.
    fn locate_staged(
        &self,
        path: Option<&ObjectPathBuf>,
        memo: Option<Resolution>,
        birth: Option<u64>,
    ) -> Result<Resolution, Error> {
        if let (None, Some(named)) = (memo, path) {
            match self.staged_standing(named, birth) {
                Standing::Pending => return Err(Error::NotCommitted(named.to_string())),
                Standing::Withdrawn => return Err(Error::StagingWithdrawn(named.to_string())),
                Standing::Live => {}
            }
        }
        match self.locate(path, memo) {
            Err(Error::Format(FormatError::PathNotFound(missing))) => {
                match path.and_then(|named| self.staged_object(named)) {
                    Some(_) => Err(Error::NotCommitted(reported_path(path))),
                    None => Err(Error::Format(FormatError::PathNotFound(missing))),
                }
            }
            other => other,
        }
    }

    /// Run `f` against this file's write session, or `None` when there is none
    /// (a read-only file stages nothing, so it has nothing to be asked about).
    fn query_engine<R>(&self, f: impl FnOnce(&WriteEngine) -> R) -> Option<R> {
        match &self.backend {
            Backend::Edit(m) => Some(f(&m.lock().unwrap_or_else(PoisonError::into_inner))),
            _ => None,
        }
    }

    /// What this session has staged at `path`, if anything. See
    /// [`WriteEngine::staged_object`] for the rule.
    fn staged_object(&self, path: &ObjectPathBuf) -> Option<StagedObject> {
        self.query_engine(|e| e.staged_object(path)).flatten()
    }

    /// Where a handle naming `path` stands: whether the object it addresses is
    /// one the file holds, one this session has staged, or one whose staging has
    /// been withdrawn under it.
    ///
    /// `birth` is the session's [staged generation](WriteEngine::staged_generation)
    /// as of the moment the handle was made onto a staged creation, and `None`
    /// for every handle opened onto an object in the file. It is what separates
    /// the two ways a staged path can stop being staged — the commit published
    /// it, or a [`Group::delete`] withdrew it — which otherwise look identical
    /// from here, and which a handle must not confuse: a withdrawn creation
    /// leaves the file's own object at that path, and it is the object the
    /// session is *deleting*.
    ///
    /// Both halves come from one lock, so a commit cannot land between them and
    /// pair a generation with a staged set from the other side of it.
    ///
    /// A file with no write session stages nothing, so every handle onto one is
    /// live. That includes the `birth`-carrying arm, which such a file cannot
    /// produce: a handle holds its [`FileInner`] alive and a backend never
    /// changes under one, so the case is unreachable rather than merely unlikely.
    fn staged_standing(&self, path: &ObjectPathBuf, birth: Option<u64>) -> Standing {
        let Some((generation, staged)) =
            self.query_engine(|e| (e.staged_generation(), e.staged_object(path).is_some()))
        else {
            return Standing::Live;
        };
        match birth {
            // A commit has taken this session's staged set since the handle was
            // made, so what it names is in the file (or was deleted from it
            // afterwards, which resolving the path reports).
            Some(birth) if birth != generation => Standing::Live,
            Some(_) if !staged => Standing::Withdrawn,
            _ if staged => Standing::Pending,
            _ => Standing::Live,
        }
    }

    /// What a dataset staged at `path` says about itself before it is written,
    /// for a handle whose staged generation at birth was `birth`.
    ///
    /// `Ok(None)` means the handle should read the file: either nothing is
    /// staged there, or a commit has published what was. The
    /// [`Error::StagingWithdrawn`] arm is [`staged_standing`](Self::staged_standing)'s,
    /// decided under the same lock — and told apart by *kind* here, since a
    /// dataset creation withdrawn and replaced by a group at the same path
    /// leaves nothing for this handle to answer from either.
    fn staged_dataset_view(
        &self,
        path: &ObjectPathBuf,
        birth: Option<u64>,
    ) -> Result<Option<StagedMeta>, Error> {
        let Some((generation, meta)) =
            self.query_engine(|e| (e.staged_generation(), e.staged_dataset_meta(path)))
        else {
            return Ok(None);
        };
        match birth {
            Some(birth) if birth != generation => Ok(None),
            Some(_) if meta.is_none() => Err(Error::StagingWithdrawn(path.to_string())),
            _ => Ok(meta),
        }
    }

    /// The session's staged generation now, for a handle being made onto a
    /// creation staged in it. `None` on a file with no write session, which
    /// stages nothing and so makes no such handle.
    fn staged_generation(&self) -> Option<u64> {
        self.query_engine(WriteEngine::staged_generation)
    }

    /// The direct children `parent` gains from this session's staged creations.
    fn staged_children(&self, parent: &ObjectPathBuf) -> StagedChildren {
        self.query_engine(|e| e.staged_children(parent))
            .unwrap_or_default()
    }

    /// The revisions to label a resolution that is about to be worked out with.
    fn revisions(&self) -> Revisions {
        Revisions {
            content: self.content_revision(),
            address: self.address_revision(),
        }
    }

    /// A `Source` view over the backend, for the streaming-capable paths.
    pub(crate) fn source(&self) -> SourceView<'_> {
        match &self.backend {
            Backend::InMemory(v) => SourceView::Mem(v),
            Backend::Streaming(s) => SourceView::Stream(s.as_ref()),
            // A mirror or bounded file's bytes live behind a lock and cannot be
            // lent out as a borrowed view; the read paths that reach every
            // backend go through [`with_source`](Self::with_source) instead.
            Backend::Edit(_) => SourceView::Mem(&[]),
        }
    }

    /// Run `f` with a random-access view of this file's bytes, taking the
    /// write-engine lock when the backend requires one. Unlike
    /// [`source`](Self::source) — which cannot lend a borrowed view out of a
    /// lock and returns an empty view for the mirror and bounded backends —
    /// this serves every backend, so it is the dispatch for read paths (heap
    /// reads for variable-length data, chunk enumeration) that must also work
    /// on a read-write file. `f` must not re-enter this file's backend (the
    /// engine lock is held while it runs).
    pub(crate) fn with_source<R>(&self, f: impl FnOnce(&dyn Source) -> R) -> R {
        match &self.backend {
            Backend::InMemory(v) => f(&BytesSource::new(v.as_slice())),
            Backend::Streaming(s) => f(s.as_ref()),
            Backend::Edit(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                f(core.image())
            }
        }
    }

    /// Run a read against a read-write session's file image, choosing the form
    /// its backing can serve: `on_slice` when the session holds the whole file
    /// in memory, so a slice-walking parser borrows the bytes instead of copying
    /// them, and `on_source` otherwise.
    ///
    /// Both closures must compute the same thing. The pair exists because a
    /// mirror can hand out a whole-file slice and a file-backed image cannot,
    /// not because the two backings answer differently (issue #198).
    fn with_engine<R>(
        engine: &Mutex<WriteEngine>,
        on_slice: impl FnOnce(&[u8]) -> R,
        on_source: impl FnOnce(&dyn Source) -> R,
    ) -> R {
        let core = engine
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match core.image_slice() {
            Some(data) => on_slice(data),
            None => on_source(core.image()),
        }
    }

    /// Parses encoded superblock values and reconciles the opened image's physical positions.
    ///
    /// The returned superblock holds the encoded addresses. The effective base is the physical
    /// signature position, and the reader applies the same relocation to the encoded EOF.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if a read of `file` fails, the signature is absent, `file` ends
    /// inside the superblock, the superblock does not parse, or the relocated EOF is outside the
    /// [`u64`] range.
    fn parse_superblock(
        file: &(impl SuperblockSource + ?Sized),
    ) -> Result<(Superblock, FormatWidths, OpenedImageContext), Error> {
        let LocatedSuperblock {
            address,
            superblock,
            widths,
        } = superblock::read(file)?;
        let opened = OpenedImageContext::reconcile(
            superblock.base_address,
            superblock.eof_address,
            address.get(),
        )?;
        Ok((superblock, widths, opened))
    }

    /// Assemble a [`File`] from parsed parts, then load the File Space Info from
    /// the superblock extension (best-effort, so a bad extension never fails the
    /// open).
    fn from_parts(
        backend: Backend,
        access_mode: AccessMode,
        superblock: Superblock,
        widths: FormatWidths,
        opened: OpenedImageContext,
        handle: Option<std::fs::File>,
        access_properties: FileAccessProperties,
    ) -> Self {
        let mut file = FileInner {
            backend,
            access_mode,
            superblock,
            widths,
            opened,
            handle,
            file_space_info: None,
            sohm_table: None,
            access_properties,
            closed: AtomicBool::new(false),
            content_revision: AtomicU64::new(0),
            address_revision: AtomicU64::new(0),
            swmr_write: false,
        };
        file.file_space_info = file.read_file_space_info();
        file.sohm_table = file.read_sohm_table();
        file
    }

    /// Parse the File Space Info message from the superblock extension, if the
    /// file records one and it can be read. Best-effort: any failure (no
    /// extension, unreadable object header, malformed message) yields `None`.
    fn read_file_space_info(&self) -> Option<FileSpaceInfo> {
        self.parse_file_space_info().ok().flatten()
    }

    /// Parses the File Space Info message from the superblock extension, or returns `None` if the
    /// file has no superblock extension or the extension holds no such message.
    ///
    /// # Errors
    ///
    /// Returns the error of the extension's address, its object header, or the message that does
    /// not parse, such as [`FormatError::InvalidFileSpacePageSize`] for a page size outside
    /// 512 bytes to 1 GiB.
    fn parse_file_space_info(&self) -> Result<Option<FileSpaceInfo>, FormatError> {
        if let Backend::Edit(engine) = &self.backend {
            let session = engine.lock().unwrap_or_else(PoisonError::into_inner);
            return session.parse_file_space_info();
        }
        let Some(rel) = self.superblock.superblock_extension_address else {
            return Ok(None);
        };
        if rel.is_undefined(self.widths.offsets) {
            return Ok(None);
        }
        let abs = self.addresses().resolve(rel)?.get();
        let header = self.parse_header(abs)?;
        let Some(msg) = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILE_SPACE_INFO)
        else {
            return Ok(None);
        };
        hdf5_pure_format::__private::parse_file_space_info(self.widths, &msg.data).map(Some)
    }

    /// Parse the Shared Message Table message from the superblock extension and
    /// read the master table it names, if the file records one.
    ///
    /// Best-effort in the same way and for the same reason as
    /// [`Self::read_file_space_info`]: a file whose shared-message table is
    /// unreadable still opens, and every object in it that shares no message
    /// still reads.
    fn read_sohm_table(&self) -> Option<Box<crate::sohm::SohmTable>> {
        let rel = self.superblock.superblock_extension_address?;
        if rel.is_undefined(self.widths.offsets) {
            return None;
        }
        let abs = self.addresses().resolve(rel).ok()?.get();
        let header = self.parse_header(abs).ok()?;
        let msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SHARED_MESSAGE_TABLE)?;
        let message = crate::sohm::SharedMessageTableMessage::parse_with_offset_width(
            &msg.data,
            self.widths.offsets,
        )
        .ok()?;
        // The table's address, like every address in a header message, is stored
        // relative to the file's base address, so the read is framed the same way
        // every other metadata walk here is.
        let base = self.addresses().effective_base();
        let offset_width = self.widths.offsets;
        match &self.backend {
            Backend::InMemory(v) => crate::sohm::SohmTable::read_with_offset_width(
                source::frame(v, base).ok()?,
                &message,
                offset_width,
            )
            .ok(),
            Backend::Streaming(s) if base.is_zero() => {
                crate::sohm::SohmTable::read_from_source_with_offset_width(
                    &SourceMetadata(s.as_ref()),
                    &message,
                    offset_width,
                )
                .ok()
            }
            Backend::Streaming(s) => crate::sohm::SohmTable::read_from_source_with_offset_width(
                &SourceMetadata(&BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                }),
                &message,
                offset_width,
            )
            .ok(),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| {
                    Ok::<_, Error>(crate::sohm::SohmTable::read_with_offset_width(
                        source::frame(d, base)?,
                        &message,
                        offset_width,
                    )?)
                },
                |s| {
                    if base.is_zero() {
                        Ok(crate::sohm::SohmTable::read_from_source_with_offset_width(
                            &SourceMetadata(s),
                            &message,
                            offset_width,
                        )?)
                    } else {
                        Ok(crate::sohm::SohmTable::read_from_source_with_offset_width(
                            &SourceMetadata(&BaseOffsetSource { inner: s, base }),
                            &message,
                            offset_width,
                        )?)
                    }
                },
            )
            .ok(),
        }
        .map(Box::new)
    }

    /// Re-read the file from disk to pick up data appended by a concurrent
    /// writer, then re-parse the superblock.
    ///
    /// This is the SWMR reader's refresh primitive (analogous to the C library's
    /// `H5Drefresh` / h5py's `Dataset.refresh()`): after it returns, newly
    /// fetched [`Dataset`]/[`Group`] handles observe the writer's appended
    /// chunks and extended dimensions, because they re-parse object headers at
    /// their (stable) addresses against the refreshed bytes. Existing handles
    /// borrow `&self`, so they must be dropped before calling this; re-fetch
    /// them afterward.
    ///
    /// Returns [`Error::SwmrUnsupported`] if the file was not opened with
    /// [`File::open_swmr`]. The superblock is checksum-validated on every
    /// re-read; a transient parse failure (a writer caught mid-flush) is
    /// retried a bounded number of times before being surfaced.
    ///
    /// Cost: each call re-reads the entire file from disk (`O(file size)`).
    /// That keeps the implementation simple and correct, but when following a
    /// large, steadily growing log it is the cost paid per refresh; budget
    /// refresh frequency accordingly.
    pub fn refresh(&mut self) -> Result<(), Error> {
        let handle = self.handle.as_mut().ok_or(Error::SwmrUnsupported)?;

        // A writer only appends (the file grows) and updates a few fixed-size,
        // individually checksummed structures in place (superblock EOF, object
        // header dimensions, array header counts). Re-reading the whole file and
        // re-validating the superblock checksum yields a consistent view; if the
        // superblock is caught mid-update, retry.
        const MAX_ATTEMPTS: u32 = 100;
        let mut last_err = None;
        for attempt in 0..MAX_ATTEMPTS {
            let mut data = Vec::new();
            handle.seek(SeekFrom::Start(0)).map_err(Error::Io)?;
            handle.read_to_end(&mut data).map_err(Error::Io)?;
            match Self::parse_superblock(data.as_slice()) {
                Ok((superblock, widths, opened)) => {
                    self.backend = Backend::InMemory(data);
                    self.superblock = superblock;
                    self.widths = widths;
                    self.opened = opened;
                    self.file_space_info = self.read_file_space_info();
                    self.sohm_table = self.read_sohm_table();
                    // Every byte just moved. `File::refresh` takes `&mut self`
                    // through `Arc::get_mut`, so no handle can be alive to see
                    // it — but the counters are the file's statement about its
                    // own bytes, and leaving them behind here would make that
                    // statement false.
                    *self.content_revision.get_mut() += 1;
                    *self.address_revision.get_mut() += 1;
                    return Ok(());
                }
                Err(e) => {
                    last_err = Some(e);
                    // Brief backoff before re-reading; the writer's in-place
                    // updates are tiny, so a short pause clears the window. Skip
                    // it on the final attempt, where there is no re-read to come.
                    if attempt + 1 < MAX_ATTEMPTS {
                        std::thread::sleep(std::time::Duration::from_micros(
                            50 * (attempt + 1) as u64,
                        ));
                    }
                }
            }
        }
        // The loop always runs at least once and only reaches here via the
        // `Err` arm, so `last_err` is always `Some`; surface the real error.
        Err(last_err.expect("refresh retried at least once before failing"))
    }

    /// Resolves a path to an object-header address, following hard and soft links.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`group_v2::resolve_link_path`] and the backend metadata reads.
    fn resolve_path(&self, path: &ObjectPath<'_>) -> Result<u64, Error> {
        let access = self.access_mode();
        Ok(match &self.backend {
            Backend::InMemory(v) => group_v2::resolve_read_path_any(
                v,
                access,
                &self.superblock,
                self.widths,
                self.addresses(),
                path,
            )?,
            Backend::Streaming(s) => group_v2::resolve_read_path_any_from_source(
                s.as_ref(),
                access,
                &self.superblock,
                self.widths,
                self.addresses(),
                path,
            )?,
            // A staged commit can relocate the object tree's root, so this
            // file's cached superblock may name a stale one; resolve against the
            // session's own superblock, which the commit updates.
            Backend::Edit(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let sb = core.superblock().clone();
                match core.image_slice() {
                    Some(data) => group_v2::resolve_read_path_any(
                        data,
                        access,
                        &sb,
                        core.widths(),
                        core.addresses(),
                        path,
                    )?,
                    None => group_v2::resolve_read_path_any_from_source(
                        core.image(),
                        access,
                        &sb,
                        core.widths(),
                        core.addresses(),
                        path,
                    )?,
                }
            }
        })
    }

    /// Returns the current root's physical file position.
    ///
    /// Reads the editing session's superblock for [`Backend::Edit`], since a commit can relocate
    /// the root. Other backends use this file's cached superblock.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the root address plus the effective base exceeds
    /// [`u64::MAX`].
    fn mirror_root_address(&self) -> Result<u64, FormatError> {
        if let Backend::Edit(m) = &self.backend {
            let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            return core
                .addresses()
                .resolve(core.superblock().root_group_address)
                .map(PhysicalAddress::get);
        }
        self.addresses()
            .resolve(self.superblock.root_group_address)
            .map(PhysicalAddress::get)
    }

    /// Returns the raw file bytes for an in-memory file, or an empty slice for a
    /// streaming file (which has no whole-file buffer).
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backend {
            Backend::InMemory(v) => v,
            // A streaming, mirror, or bounded file has no borrowable whole-file
            // buffer.
            Backend::Streaming(_) | Backend::Edit(_) => &[],
        }
    }

    /// Returns the access properties this file was opened with.
    pub const fn access_properties(&self) -> FileAccessProperties {
        self.access_properties
    }

    /// The backend this file's editing session resolved to, or `None` when there
    /// is no editing session to ask. Always [`EditBacking::Mirrored`] for the
    /// SWMR writer, which builds a session but does not dispatch on the strategy.
    fn edit_backing(&self) -> Option<EditBacking> {
        match &self.backend {
            Backend::Edit(m) => Some(m.lock().unwrap_or_else(|e| e.into_inner()).edit_backing()),
            _ => None,
        }
    }

    /// What this file's metadata cache has done, or `None` where the backend
    /// holds none.
    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.with_source(|source| source.metadata_cache_stats())
    }

    /// Zero those counters, keeping the cached entries.
    fn reset_metadata_cache_stats(&self) {
        self.with_source(|source| source.reset_metadata_cache_stats());
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Returns the address and length widths validated when this file was opened.
    pub(crate) const fn widths(&self) -> FormatWidths {
        self.widths
    }

    /// The whole-file byte image when this file is buffered in memory
    /// ([`open`](Self::open) / [`from_bytes`](Self::from_bytes)); `None` for a
    /// streaming file ([`open_streaming`](Self::open_streaming)). Cross-file
    /// object copy ([`File::copy_from`](crate::File::copy_from)) uses this to read
    /// source objects by absolute address.
    pub(crate) fn in_memory_image(&self) -> Option<&[u8]> {
        match &self.backend {
            Backend::InMemory(data) => Some(data),
            Backend::Streaming(_) | Backend::Edit(_) => None,
        }
    }

    /// Returns the effective base used to interpret stored addresses in this image.
    pub(crate) fn base_address(&self) -> BaseAddress {
        self.addresses().effective_base()
    }

    /// Returns the mapping between stored addresses and physical positions in this image.
    pub(crate) const fn addresses(&self) -> AddressContext {
        self.opened.addresses()
    }

    /// Returns the mode this file reads under.
    pub(crate) fn access_mode(&self) -> AccessMode {
        self.access_mode
    }

    /// Returns the physical byte length of the backing store.
    ///
    /// This includes any userblock prefix and trailing bytes. For a streaming file, the source
    /// reports the length. For an in-memory file, it is the buffer length. The encoded
    /// [`Superblock::eof_address`] may use a different coordinate frame if the image was relocated.
    pub fn file_size(&self) -> u64 {
        match &self.backend {
            Backend::Edit(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                core.image().len()
            }
            _ => self.source().len(),
        }
    }

    /// The minimum library version required to read this file, derived from its
    /// superblock version (the *low bound* of HDF5's `H5Fget_libver_bounds`).
    ///
    /// A version 3 superblock, for example, reports [`LibVer::V110`] because it
    /// was introduced in HDF5 1.10.
    pub fn libver_bound(&self) -> LibVer {
        LibVer::from_superblock_version(self.superblock.version())
    }

    /// Parses the datatype of the Datatype message in `header`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingMessage`] if `header` has no Datatype message, and [`Error::Format`]
    /// if the message does not parse.
    fn header_datatype(&self, header: &ObjectHeader) -> Result<Datatype, Error> {
        let message = find_message(header, MessageType::DATATYPE)?;
        let (datatype, _) =
            hdf5_pure_format::__private::parse_datatype(&self.message_body(message)?)?;
        Ok(datatype)
    }

    fn parse_header(&self, address: u64) -> Result<ObjectHeader, FormatError> {
        let access = self.access_mode();
        match &self.backend {
            Backend::InMemory(v) => ObjectHeader::parse_with_base_and_widths(
                v,
                access,
                address.to_usize()?,
                self.widths,
                self.addresses().effective_base(),
            ),
            Backend::Streaming(s) => ObjectHeader::parse_from_source_with_widths(
                &SourceMetadata(s.as_ref()),
                access,
                address,
                self.widths,
                self.addresses().effective_base(),
            ),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| {
                    ObjectHeader::parse_with_base_and_widths(
                        d,
                        access,
                        address.to_usize()?,
                        self.widths,
                        self.addresses().effective_base(),
                    )
                },
                |s| {
                    ObjectHeader::parse_from_source_with_widths(
                        &SourceMetadata(s),
                        access,
                        address,
                        self.widths,
                        self.addresses().effective_base(),
                    )
                },
            ),
        }
    }

    /// Resolve a base-relative object-header address (the value stored in an
    /// HDF5 `H5R_OBJECT` reference element) to the [`Object`] it points at.
    ///
    /// The reader applies the opened image's effective base to the stored address.
    /// A null (`0`) or undefined (`HADDR_UNDEF`) address, or one whose object header is neither
    /// a dataset nor a group, yields [`FormatError::InvalidObjectReference`].
    fn object_at_relative(
        file: &Arc<FileInner>,
        revisions: Revisions,
        rel_addr: u64,
    ) -> Result<Object, Error> {
        // HADDR_UNDEF and the null address never name a real object. (Relative
        // address 0 is where the superblock sits, not an object header.)
        if rel_addr == u64::MAX || rel_addr == 0 {
            return Err(FormatError::InvalidObjectReference(rel_addr).into());
        }
        let abs = file
            .addresses()
            .resolve(StoredAddress::new(rel_addr))?
            .get();
        let at = revisions.at(abs);
        let hdr = file.parse_header(abs)?;
        if has_message(&hdr, MessageType::DATA_LAYOUT) {
            let chunk_cache = DatasetAccessProperties::new()
                .resolved_chunk_cache(file.access_properties.chunk_cache);
            Ok(Object::Dataset(Box::new(Dataset::new(
                file.clone(),
                at,
                hdr,
                chunk_cache,
                None,
            ))))
        } else if group_v2::is_group(&hdr) {
            Ok(Object::Group(Group::new(file.clone(), at, None)))
        } else {
            Err(FormatError::InvalidObjectReference(rel_addr).into())
        }
    }

    /// Returns the entries of the group `hdr` describes, dispatching on the backend.
    ///
    /// Each entry holds the address the file stores for its child, which
    /// [`entry_address`](Self::entry_address) turns into a position to read at.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the group's own structures do not parse, or if `hdr` describes
    /// no group at all.
    fn group_children(&self, hdr: &ObjectHeader) -> Result<Vec<GroupEntry>, Error> {
        let (widths, base) = (self.widths, self.addresses());
        let k = group_v1::group_leaf_node_k(&self.superblock);
        match &self.backend {
            Backend::InMemory(v) => {
                group_v2::resolve_group_entries(v, self.access_mode, hdr, widths, base, k)
            }
            Backend::Streaming(s) => group_v2::resolve_group_entries_from_source(
                s.as_ref(),
                self.access_mode,
                hdr,
                widths,
                base,
                k,
            ),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| group_v2::resolve_group_entries(d, self.access_mode, hdr, widths, base, k),
                |s| {
                    group_v2::resolve_group_entries_from_source(
                        s,
                        self.access_mode,
                        hdr,
                        widths,
                        base,
                        k,
                    )
                },
            ),
        }
        .map_err(Error::Format)
    }

    /// Returns the absolute file address of the object header `entry` names.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the base address cannot be added to it, which a
    /// crafted entry naming `HADDR_UNDEF` in a file with a userblock arranges.
    fn entry_address(&self, entry: &GroupEntry) -> Result<u64, FormatError> {
        self.addresses()
            .resolve(entry.object_header_address)
            .map(PhysicalAddress::get)
    }

    /// Returns the parsed target of the child named `name`.
    ///
    /// Compact group lookups compare borrowed names and parse matching targets.
    ///
    /// # Errors
    ///
    /// Returns errors from the group metadata and link target parsers.
    fn group_child(
        &self,
        group_address: u64,
        name: LinkName<'_>,
    ) -> Result<ChildLookup, FormatError> {
        let (widths, base) = (self.widths, self.addresses());
        let k = group_v1::group_leaf_node_k(&self.superblock);
        let access = self.access_mode();
        let addr = group_address;
        match &self.backend {
            Backend::InMemory(v) => {
                group_v2::find_child_target(v, access, addr, widths, base, k, name)
            }
            Backend::Streaming(s) => group_v2::find_child_target_from_source(
                s.as_ref(),
                access,
                addr,
                widths,
                base,
                k,
                name,
            ),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| group_v2::find_child_target(d, access, addr, widths, base, k, name),
                |s| group_v2::find_child_target_from_source(s, access, addr, widths, base, k, name),
            ),
        }
    }

    /// Read all attributes attached to an object header, dispatching on the
    /// backend.
    fn attrs_of(&self, hdr: &ObjectHeader) -> Result<HashMap<String, AttrValue>, Error> {
        let (widths, base) = (self.widths, self.addresses());
        let attr_msgs = self.attr_messages_of(hdr)?;
        match &self.backend {
            Backend::Edit(m) => Ok(Self::with_engine(
                m,
                |d| types::attrs_to_map(&attr_msgs, &BytesSource::new(d), widths, base),
                |s| types::attrs_to_map(&attr_msgs, s, widths, base),
            )),
            _ => Ok(types::attrs_to_map(
                &attr_msgs,
                &self.source(),
                widths,
                base,
            )),
        }
    }

    /// The content of a header message, following the reference when the record
    /// marks the message *shared*.
    ///
    /// A shared record's body is not the message: it is an address, and a
    /// committed (`H5Tcommit`) datatype is stored exactly that way. Decoding the
    /// body directly turns a named `H5T_STD_I32LE` into a zero-width time type
    /// with no error anywhere, so every read of a message that HDF5 permits to be
    /// shared — datatype, dataspace, fill value, filter pipeline — goes through
    /// here. The borrowed case allocates nothing, which is every message this
    /// crate writes and nearly every one it reads.
    fn message_body<'m>(
        &self,
        msg: &'m crate::object_header::HeaderMessage,
    ) -> Result<Cow<'m, [u8]>, Error> {
        if !msg.flags.is_shared() {
            return Ok(Cow::Borrowed(&msg.data));
        }
        let (widths, base) = (self.widths, self.addresses().effective_base());
        let access = self.access_mode();
        let sohm = self.sohm_table.as_deref();
        // A shared reference stores its address relative to the base address, so
        // frame the file at `base` exactly as [`Self::attr_messages_of`] does.
        let resolved = match &self.backend {
            Backend::InMemory(v) => BufferedResolver::new(
                source::frame(v, base)?,
                self.addresses(),
                access,
                widths,
                sohm,
            )
            .resolve(&msg.data, msg.msg_type),
            Backend::Streaming(s) if base.is_zero() => {
                SourceResolver::new(s.as_ref(), self.addresses(), access, widths, sohm)
                    .resolve(&msg.data, msg.msg_type)
            }
            Backend::Streaming(s) => SourceResolver::new(
                &BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                },
                self.addresses(),
                access,
                widths,
                sohm,
            )
            .resolve(&msg.data, msg.msg_type),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| {
                    BufferedResolver::new(
                        source::frame(d, base)?,
                        self.addresses(),
                        access,
                        widths,
                        sohm,
                    )
                    .resolve(&msg.data, msg.msg_type)
                },
                |s| {
                    if base.is_zero() {
                        SourceResolver::new(s, self.addresses(), access, widths, sohm)
                            .resolve(&msg.data, msg.msg_type)
                    } else {
                        SourceResolver::new(
                            &BaseOffsetSource { inner: s, base },
                            self.addresses(),
                            access,
                            widths,
                            sohm,
                        )
                        .resolve(&msg.data, msg.msg_type)
                    }
                },
            ),
        }?;
        Ok(Cow::Owned(resolved))
    }

    /// Returns the object-header address a *shared* header message refers to, as the message
    /// stores it, or `None` if the message stores its content inline or refers to the
    /// shared-message heap.
    ///
    /// [`Self::message_body`] answers what the message says; this answers which
    /// object says it. A rewrite needs both: the content to reproduce the type,
    /// and the address to tell which users share one committed object rather than
    /// each naming a type of their own. A heap-stored message has no such object
    /// — it is one anonymous copy rather than a named one — so it answers `None`
    /// and a rewrite spells the message out, which is what the file already
    /// means.
    pub(crate) fn shared_target_address(
        &self,
        msg: &crate::object_header::HeaderMessage,
    ) -> Result<Option<StoredAddress>, Error> {
        if !msg.flags.is_shared() {
            return Ok(None);
        }
        let reference = shared_message::parse_shared_ref_with_widths(&msg.data, self.widths)?;
        match reference.location {
            shared_message::SharedLocation::ObjectHeader(addr) => Ok(Some(addr)),
            shared_message::SharedLocation::SohmHeap(_) => Ok(None),
        }
    }

    /// Extract every attribute message attached to an object header (compact,
    /// shared, and dense storage), dispatching on the backend.
    pub(crate) fn attr_messages_of(
        &self,
        hdr: &ObjectHeader,
    ) -> Result<Vec<crate::attribute::AttributeMessage>, Error> {
        let widths = self.widths;
        // Compact attributes come out of `hdr`, but the two addresses this walk
        // follows are read from message bodies and so are stored relative to the
        // base address: the Attribute Info message's fractal-heap address, and a
        // shared attribute's message address. Frame the file at `base` exactly as
        // [`Self::read_dataset_raw`] does, so both index it directly. For a plain
        // file (`base == 0`) this is the identity; without it, a userblock file's
        // dense attributes are looked for one userblock too early.
        let base = self.addresses().effective_base();
        let access = self.access_mode();
        let sohm = self.sohm_table.as_deref();
        match &self.backend {
            Backend::InMemory(v) => Ok(attribute::extract_attributes_full(
                source::frame(v, base)?,
                access,
                self.addresses(),
                hdr,
                widths,
                sohm,
            )?),
            Backend::Streaming(s) if base.is_zero() => {
                Ok(attribute::extract_attributes_full_from_source(
                    s.as_ref(),
                    self.addresses(),
                    access,
                    hdr,
                    widths,
                    sohm,
                )?)
            }
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                Ok(attribute::extract_attributes_full_from_source(
                    &framed,
                    self.addresses(),
                    access,
                    hdr,
                    widths,
                    sohm,
                )?)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| {
                    Ok(attribute::extract_attributes_full(
                        source::frame(d, base)?,
                        access,
                        self.addresses(),
                        hdr,
                        widths,
                        sohm,
                    )?)
                },
                |s| {
                    if base.is_zero() {
                        Ok(attribute::extract_attributes_full_from_source(
                            s,
                            self.addresses(),
                            access,
                            hdr,
                            widths,
                            sohm,
                        )?)
                    } else {
                        let framed = BaseOffsetSource { inner: s, base };
                        Ok(attribute::extract_attributes_full_from_source(
                            &framed,
                            self.addresses(),
                            access,
                            hdr,
                            widths,
                            sohm,
                        )?)
                    }
                },
            ),
        }
    }

    /// Read a dataset's raw bytes for the given layout, dispatching on the backend.
    fn read_dataset_raw(
        &self,
        spec: RawReadSpec<'_>,
        cache: &ChunkCache,
    ) -> Result<Vec<u8>, FormatError> {
        let widths = self.widths;
        let addresses = self.addresses();
        match &self.backend {
            Backend::InMemory(bytes) => data_read::read_raw_data_cached(
                StoredBytes::new(bytes, addresses)?,
                spec,
                widths,
                cache,
            ),
            Backend::Streaming(physical) => data_read::read_raw_data_cached_from_source(
                &PhysicalSource::new(physical.as_ref()).stored(addresses),
                spec,
                widths,
                cache,
            ),
            Backend::Edit(engine) => Self::with_engine(
                engine,
                |bytes| {
                    data_read::read_raw_data_cached(
                        StoredBytes::new(bytes, addresses)?,
                        spec,
                        widths,
                        cache,
                    )
                },
                |physical| {
                    data_read::read_raw_data_cached_from_source(
                        &PhysicalSource::new(physical).stored(addresses),
                        spec,
                        widths,
                        cache,
                    )
                },
            ),
        }
    }

    fn read_dataset_region(
        &self,
        region: &ReadRegion<'_>,
        cache: &ChunkCache,
        pass: CachePass,
    ) -> Result<Vec<u8>, FormatError> {
        if region.is_whole() {
            return self.read_dataset_raw(region.spec(), cache);
        }
        match &self.backend {
            Backend::InMemory(bytes) => read_region_framed(
                &PhysicalSource::new(&BytesSource::new(bytes)).stored(self.addresses()),
                region,
                self.widths,
                cache,
                pass,
            ),
            Backend::Streaming(source) => read_region_framed(
                &PhysicalSource::new(source.as_ref()).stored(self.addresses()),
                region,
                self.widths,
                cache,
                pass,
            ),
            Backend::Edit(engine) => Self::with_engine(
                engine,
                |bytes| {
                    read_region_framed(
                        &PhysicalSource::new(&BytesSource::new(bytes)).stored(self.addresses()),
                        region,
                        self.widths,
                        cache,
                        pass,
                    )
                },
                |source| {
                    read_region_framed(
                        &PhysicalSource::new(source).stored(self.addresses()),
                        region,
                        self.widths,
                        cache,
                        pass,
                    )
                },
            ),
        }
    }
}

/// Reads a checked region through the opened image's stored-coordinate source view.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedVirtualLayout`] for virtual storage, and
/// [`FormatError::UnexpectedSourceEof`] if a contiguous selection extends past the source's end.
/// Returns errors from reading or placing the selected bytes, applying filters, or reading fill
/// values for unallocated storage.
fn read_region_framed(
    source: &(impl StoredRead + ?Sized),
    region: &ReadRegion<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
    pass: CachePass,
) -> Result<Vec<u8>, FormatError> {
    let spec = region.spec();
    if matches!(spec.storage(), RawReadStorage::Virtual) {
        return Err(FormatError::UnsupportedVirtualLayout);
    }
    if region.is_empty() {
        return Ok(Vec::new());
    }
    if let RawReadStorage::Chunked { .. } = spec.storage() {
        return crate::chunked_read::read_chunked_region_from_source(
            source, region, widths, cache, pass,
        );
    }
    if matches!(spec.storage(), RawReadStorage::ContiguousUnallocated) {
        return spec.fill().buffer(region.byte_len());
    }
    if let RawReadStorage::Contiguous { address } = spec.storage() {
        let end = region.end_offset()?;
        let end = address.checked_add(end)?;
        if end > StoredAddress::new(source.len()) {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end.get(),
                available: source.len(),
            });
        }
    }
    let total = region.byte_len();
    let mut output = Vec::new();
    region.visit_runs(|offset, length| {
        match spec.storage() {
            RawReadStorage::Compact { data } => {
                let end = offset
                    .checked_add(length as u64)
                    .ok_or(FormatError::OffsetOverflow {
                        offset,
                        length: length as u64,
                    })?
                    .to_usize()?;
                let bytes =
                    data.get(offset.to_usize()?..end)
                        .ok_or(FormatError::DataSizeMismatch {
                            expected: end,
                            actual: data.len(),
                        })?;
                if output.is_empty() {
                    output.reserve_exact(total);
                }
                output.extend_from_slice(bytes);
            }
            RawReadStorage::Contiguous { address } => {
                let offset = address.checked_add(offset)?;
                if output.is_empty() {
                    output = source.read_exact_at(offset, length)?;
                    output.reserve_exact(total - length);
                } else {
                    let start = output.len();
                    output.resize(start + length, 0);
                    source.read_at(offset, &mut output[start..])?;
                }
            }
            _ => return Err(FormatError::Internal("region storage dispatch".into())),
        }
        Ok(())
    })?;
    Ok(output)
}

impl std::fmt::Debug for FileInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("File")
            .field("size", &self.file_size())
            .field("superblock_version", &self.superblock.version())
            .finish()
    }
}

/// An open HDF5 file.
///
/// A `File` is an owned, cheaply cloneable handle to an open file: cloning it (or
/// deriving a [`Dataset`]/[`Group`] from it) shares one underlying open file
/// rather than re-reading it. Object handles returned by [`dataset`](Self::dataset),
/// [`group`](Self::group), and [`root`](Self::root) are **owned** — they keep the
/// file open for as long as they live and carry no borrow of the `File`, so they
/// can be stored in a struct, cached, cloned, and moved across threads. They stay
/// usable across a [`commit`](Self::commit), which is what makes caching one
/// worthwhile; see [`commit`](Self::commit) for the two cases that report
/// instead.
#[derive(Clone)]
pub struct File {
    inner: Arc<FileInner>,
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.inner, f)
    }
}

impl File {
    /// Opens the HDF5 file at `path` and reads it into memory.
    ///
    /// [`File::open_streaming`] reads a file larger than memory without buffering it, and
    /// [`File::open_swmr`] follows a file that a concurrent writer is appending to.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TruncatedFile`] through [`Error::Format`] if the physical file is
    /// shorter than the recorded EOF after relocation. Bytes after that position are permitted.
    ///
    /// Returns [`Error::FileMarkedInUse`] if the version-3 superblock's status flags mark
    /// the file as open for writing. `H5Fopen` makes the same check. The flag marks a live writer
    /// or one that exited without closing the file.
    /// [`clear_status_flags`](Self::clear_status_flags) clears a stale flag,
    /// [`open_swmr`](Self::open_swmr) follows a live SWMR writer, and
    /// [`from_bytes`](Self::from_bytes) reads the bytes its caller already holds without the check,
    /// which is the way to read a flagged file on a read-only mount.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open(path)?),
        })
    }

    /// Opens the HDF5 file at `path` with explicit access properties.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`File::open`], subject to `properties`.
    pub fn open_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_with_options(path, properties)?),
        })
    }

    /// Opens the HDF5 file at `path`, fetching metadata and data on demand.
    ///
    /// Metadata and dataset chunks are read through [`ReadSeekSource`], so memory use depends on
    /// the metadata being parsed and the chunks being read. The file may be larger than the
    /// process address space.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TruncatedFile`] through [`Error::Format`] if the physical file is
    /// shorter than the recorded EOF after relocation. Bytes after that position are permitted.
    /// Returns [`Error::FileMarkedInUse`] if the version-3 superblock's status flags mark the file
    /// as open for writing.
    pub fn open_streaming<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming(path)?),
        })
    }

    /// Opens the HDF5 file at `path` for on-demand reads with explicit access properties.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`File::open_streaming`], subject to `properties`.
    pub fn open_streaming_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming_with_options(path, properties)?),
        })
    }

    /// Opens the file at `path` for the streaming reads of a repack, whose object
    /// header parses are made under the `access_mode` the caller's
    /// [`RepackOptions`](crate::RepackOptions) gives.
    pub(crate) fn open_streaming_with_access_mode(
        path: &std::path::Path,
        access_mode: AccessMode,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming_with_access_mode(
                path,
                access_mode,
            )?),
        })
    }

    /// Opens an HDF5 image from a [`Source`], fetching metadata and data on demand.
    ///
    /// This is the streaming open for a caller whose bytes are not a path: an
    /// object store addressed by HTTP range request, a sandboxed guest that
    /// receives byte ranges from its host, a decrypting layer. A [`Source`]
    /// supplies a length and reads at an absolute offset, which is all the
    /// reader asks of a file, so peak memory stays at the metadata being parsed
    /// plus the chunks a read touches — not the file. Wrap a `Read + Seek` in
    /// [`ReadSeekSource`] rather than writing that impl again.
    ///
    /// A file marked as held by a writer is rejected here as it is by
    /// [`open_streaming`](Self::open_streaming); with no path to report, the
    /// error names the source instead. Recovering such a file in place needs a
    /// path, through [`File::clear_status_flags`], so a caller that has none is
    /// left with [`File::from_bytes`](Self::from_bytes) and the whole file in
    /// memory.
    ///
    /// The metadata cache is **off** unless
    /// [`from_source_with_options`](Self::from_source_with_options) turns it
    /// on, which matters more here than it does for a local file: without one,
    /// every read a parser makes is a round trip. See
    /// [`MetadataCacheConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TruncatedFile`] through [`Error::Format`] if [`Source::len`] is less
    /// than the recorded EOF after relocation. Bytes after that position are permitted.
    /// Returns [`Error::FileMarkedInUse`] if the version-3 superblock's status flags mark the image
    /// as open for writing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use hdf5_pure::{File, FormatError, Source};
    ///
    /// // However the bytes actually arrive: a range request, a host call, a
    /// // decrypting layer over a file.
    /// # fn fetch(offset: u64, len: usize) -> Result<Vec<u8>, String> { unimplemented!() }
    ///
    /// struct Remote {
    ///     len: u64,
    /// }
    ///
    /// impl Source for Remote {
    ///     fn len(&self) -> u64 {
    ///         self.len
    ///     }
    ///
    ///     fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
    ///         // Fill the whole request or fail: a short read is an error.
    ///         let bytes = fetch(offset, buf.len()).map_err(FormatError::Source)?;
    ///         buf.copy_from_slice(&bytes);
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let file = File::from_source(Remote { len: 1 << 30 })?;
    /// let rows = file.dataset("frames")?.read_f64_rows(0, 64)?;
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    ///
    /// A `Read + Seek` needs none of that: wrap it in [`ReadSeekSource`].
    /// Reading a *file* that way is [`open_streaming`](Self::open_streaming),
    /// which does the wrapping for you.
    pub fn from_source<S: Source + Send + Sync + 'static>(source: S) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_source(source)?),
        })
    }

    /// Opens an HDF5 image from a [`Source`] with explicit access properties.
    ///
    /// [`MetadataCacheConfig`] controls caching of metadata reads.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`File::from_source`], subject to `properties`.
    pub fn from_source_with_options<S: Source + Send + Sync + 'static>(
        source: S,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_source_with_options(source, properties)?),
        })
    }

    /// Opens the HDF5 file at `path` for single-writer/multiple-reader access.
    ///
    /// Retains a live file handle so [`File::refresh`] can read data appended by a concurrent
    /// writer. With a version-3 superblock, the physical file may be shorter than the recorded
    /// EOF while the writer extends it. Each metadata or data read must have its requested bytes
    /// available.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TruncatedFile`] through [`Error::Format`] if the superblock version
    /// is 0, 1, or 2 and the physical file is shorter than the recorded EOF after relocation.
    /// Returns [`Error::FileMarkedInUse`] if exactly one of the version-3 superblock's write-access
    /// and SWMR-write flags is set. A file with both flags set can be read alongside its SWMR writer.
    #[doc(alias = "H5F_ACC_SWMR_READ")]
    pub fn open_swmr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr(path)?),
        })
    }

    /// Opens the HDF5 file at `path` for SWMR reading with explicit access properties.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`File::open_swmr`].
    pub fn open_swmr_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr_with_options(path, properties)?),
        })
    }

    /// Opens an HDF5 image from an in-memory byte vector.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TruncatedFile`] through [`Error::Format`] if `data` is shorter than
    /// the recorded EOF after relocation. Bytes after that position are permitted.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes(data)?),
        })
    }

    /// Opens an HDF5 image from an in-memory byte vector with explicit access properties.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`File::from_bytes`].
    pub fn from_bytes_with_options(
        data: Vec<u8>,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes_with_options(data, properties)?),
        })
    }

    /// Opens an existing HDF5 file for reading and editing.
    ///
    /// Unlike [`open`](Self::open) (read-only, buffered), this takes an exclusive
    /// OS file lock held for the file's life and lets owned handles modify the
    /// file — immediate [`Dataset::append`]s, plus [`Dataset::write`]/`set_attr`,
    /// [`Group::create_dataset`]/`create_group`/`delete`/`set_attr`, and
    /// [`copy`](Self::copy)/[`copy_from`](Self::copy_from) staged until
    /// [`commit`](Self::commit). The file must use 8-byte offsets and lengths and
    /// keep its superblock at its base address (a canonical userblock, as in a
    /// MATLAB `.mat` file, is supported); anything else is refused with
    /// [`Error::EditUnsupported`].
    ///
    /// The fast immediate [`Dataset::append`] additionally requires a
    /// latest-format (version-2/3) file with no userblock and an
    /// Extensible-Array-indexed dataset; [`Dataset::append_staged`] covers the
    /// general case.
    ///
    /// A [`FileSpaceStrategy::Page`] file with persistence disabled supports staged edits
    /// through the whole-file mirror. Allocations use fresh typed pages after the original
    /// allocated region and physical extent, aligned relative to the effective base.
    /// Freed space remains unavailable, and persistence stays
    /// disabled. Commits require supported attribute storage and a tree proved free of object
    /// references. Immediate append, SWMR writing, and page buffering are unsupported in this mode.
    ///
    /// Two things can turn this open away because another writer holds the file:
    /// the exclusive OS lock, reported as
    /// [`Error::FileLocked`], and the superblock's
    /// status-flags byte, reported as
    /// [`Error::FileMarkedInUse`]. The second
    /// covers what the first cannot — a SWMR writer takes no lock, and a writer
    /// that exited without closing the file leaves the flag behind; recover a
    /// stale one with [`clear_status_flags`](Self::clear_status_flags).
    ///
    /// # Memory
    ///
    /// This picks its backing from the file rather than making the caller pick a
    /// function (issue #198): a latest-format file with no userblock, except a
    /// non-persistent [`FileSpaceStrategy::Page`] file, is edited
    /// **bounded**, holding only the metadata being parsed plus the configured
    /// caches plus what an edit is building, so resident memory does not scale
    /// with the file; anything else falls back to a whole-file in-memory mirror,
    /// which supports pre-v2 superblocks, userblocks, and non-persistent PAGE editing. The two
    /// backings are the same engine over different storage and offer the same
    /// edit surface, differing in one trade: the bounded one applies a large
    /// immediate append in whole-chunk batches, each crash-atomic on its own, so
    /// a crash mid-call leaves a valid shorter dataset rather than none of the
    /// append. Ask a file which it got with
    /// [`edit_backing`](Self::edit_backing), and demand one with
    /// [`FileAccessProperties::with_memory_strategy`] —
    /// [`MemoryStrategy::Mirrored`] restores the unconditional mirror this
    /// entry point used before it learned to dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for unsupported address widths or file layouts.
    /// Returns [`Error::Format`] for malformed File Space Info or invalid address arithmetic,
    /// and [`Error::InvalidSpaceOwnershipExtent`] for invalid allocation extents.
    /// Returns [`Error::FileLocked`] if the file is locked by another process,
    /// [`Error::FileMarkedInUse`] if its status flags indicate a writer, and [`Error::Io`]
    /// if opening or reading fails.
    #[doc(alias = "H5Fopen")]
    pub fn open_rw<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_rw_with_options(path, FileAccessProperties::new())
    }

    /// Opens an existing file for reading and in-place editing with explicit access properties.
    ///
    /// The properties carry the locking policy (the `H5Pset_file_locking` analogue,
    /// [`FileAccessProperties::with_locking`]), the memory strategy
    /// ([`FileAccessProperties::with_memory_strategy`], which overrides the
    /// dispatch described on [`open_rw`](Self::open_rw)), the `fsync` cadence
    /// ([`FileAccessProperties::with_sync_policy`]), the metadata cache used by
    /// the bounded backing, and the file-wide chunk-cache default applied to
    /// datasets opened from this file. Because one [`FileAccessProperties`] value
    /// serves every open, the same configuration can be shared with a read path.
    ///
    /// [`FileAccessProperties::with_persisted_free_space_policy`] controls recovery of persisted
    /// free space. Strict parsing completes before allocator installation or page buffering.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PersistedFreeSpace`] for a manager read or parse failure under
    /// [`PersistedFreeSpacePolicy::Strict`]. The error includes the decoded manager context and
    /// its cause. Other failures are the errors of [`open_rw`](Self::open_rw).
    pub fn open_rw_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_rw(path, properties)?),
        })
    }

    /// Opens a mirrored PAGE crosscheck session with one injected publication error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] unless the file uses non-persistent PAGE allocation.
    /// Returns the opening or configuration error if the session cannot be created.
    #[cfg(test)]
    pub(crate) fn open_page_publication_failure(
        path: &Path,
        failure: PagePublicationFailure,
        sync: SyncPolicy,
        gathered: bool,
    ) -> Result<Self, Error> {
        Ok(Self {
            inner: Arc::new(FileInner::from_rw_session(
                WriteEngine::open_page_publication_failure(path, failure, gathered)?,
                FileAccessProperties::new().with_sync_policy(sync),
            )?),
        })
    }

    /// Reads the complete file through the editing session's retained disk handle.
    ///
    /// Pending writes remain buffered, and the session lock remains held.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file has no editing session, [`Error::Io`] if reading
    /// the length or bytes fails, or [`Error::Format`] if the length exceeds [`usize::MAX`].
    #[cfg(test)]
    pub(crate) fn disk_bytes(&self) -> Result<Vec<u8>, Error> {
        self.inner
            .query_engine(WriteEngine::disk_bytes)
            .ok_or(Error::ReadOnly)?
    }

    /// Reads issued file bytes through the editing session's disk handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file has no editing session, or the snapshot read error.
    #[cfg(test)]
    pub(crate) fn page_publication_snapshot(&self) -> Result<Vec<u8>, Error> {
        self.with_mirror_session(Change::Nothing, |session| {
            session.page_publication_snapshot()
        })
    }

    #[cfg(test)]
    test_util_hdf5::hdf5_1_10_tests! {
    /// Reads the editing session's complete image, including buffered changes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file has no editing session, [`Error::FileClosed`]
    /// if it is closed, or [`Error::Format`] if the image length exceeds [`usize::MAX`] or the
    /// source read fails.
    pub(crate) fn group_inspection_snapshot(&self) -> Result<Vec<u8>, Error> {
        self.with_mirror_session(Change::Nothing, |session| {
            let image = session.image();
            image
                .read_metadata_at(0, image.len().to_usize()?)
                .map_err(Error::from)
        })
    }

    }

    /// Opens a complete file mirror for editing through the source read interface.
    ///
    /// The reader uses [`Source::read_at`] for every read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if `path` cannot be opened for reading and writing. Returns the
    /// locking, format, and configuration errors of [`Self::open_rw`].
    #[cfg(test)]
    pub(crate) fn open_rw_source_only(path: &std::path::Path) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_rw_session(
                WriteEngine::open_source_only(path)?,
                FileAccessProperties::new(),
            )?),
        })
    }

    /// Open an existing file for **SWMR** (single-writer/multiple-reader)
    /// appending: take **no** OS lock (so concurrent readers, and Windows'
    /// mandatory locks, are never blocked) and raise the superblock's SWMR-write
    /// flag so a reader may attach with [`File::open_swmr`], the C library's
    /// `H5F_ACC_SWMR_READ`, or h5py `swmr=True`.
    ///
    /// Only immediate [`Dataset::append`] is permitted, and only over the SWMR
    /// subset — an **unfiltered**, chunk-aligned append, so a concurrent reader
    /// only ever observes a consistent prefix; a filtered or non-chunk-aligned
    /// append returns [`Error::SwmrAppendUnsupported`].
    /// The staged edit surface (`write`/`set_attr`/`create_*`/`delete`/`copy`/
    /// `commit`) returns
    /// [`Error::SwmrStagedUnsupported`].
    /// [`close`](Self::close) clears the SWMR-write flag; a writer that exits
    /// without a clean close leaves it set — recover with
    /// [`clear_status_flags`](Self::clear_status_flags). While the flag stands, this
    /// open is refused with
    /// [`Error::FileMarkedInUse`], which is what keeps a second writer off a
    /// file SWMR gives only one (no OS lock is held to do it).
    ///
    /// Requires a latest-format (version-3 superblock) file with no userblock,
    /// no persisted free space, and a strategy other than [`FileSpaceStrategy::Page`].
    /// Other files are rejected with
    /// [`Error::SwmrAppendUnsupported`]. The version-3 requirement is the C
    /// library's: neither library reads the SWMR-write flag back on an older
    /// superblock, so raising one there would announce the writer to nobody.
    #[doc(alias = "H5F_ACC_SWMR_WRITE")]
    pub fn open_swmr_writer<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_swmr_writer_with_options(path, FileAccessProperties::new())
    }

    /// Open for SWMR appending with explicit access properties — see
    /// [`open_swmr_writer`](Self::open_swmr_writer).
    ///
    /// The properties' chunk cache is the file-wide default for datasets opened
    /// from this file. Its locking policy is ignored, which costs the caller
    /// nothing: SWMR takes no OS lock by design, which is stronger than any
    /// locking a caller could ask for. Its memory strategy is *not* ignored the
    /// same way — this writer always mirrors, so an explicit
    /// [`MemoryStrategy::Bounded`] is a guarantee it cannot meet and is refused
    /// with [`Error::EditUnsupported`]; [`MemoryStrategy::Auto`] and
    /// [`MemoryStrategy::Mirrored`] are both satisfied by the mirror.
    ///
    /// Its [`SyncPolicy`] applies here as to any other
    /// read-write session, the SWMR-write flag included; a reader on this
    /// machine is unaffected either way, since the barriers carry the write
    /// order across power loss rather than across processes.
    pub fn open_swmr_writer_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr_writer(path, properties)?),
        })
    }

    /// Clears superblock status flags left by an interrupted writer.
    ///
    /// Clears the flags used by ordinary and SWMR writers, including the crash mark of a
    /// page-buffered session ([`FileAccessProperties::with_page_buffer_size`]). This is equivalent
    /// to `h5clear --status` (`h5clear -s`). The other `h5clear` operations, such as removing a
    /// metadata cache image or changing the file size, are outside its scope. The superblock
    /// checksum is updated with the flags. A file whose flags are already clear is unchanged,
    /// as is a version 0 or 1 superblock.
    ///
    /// Use this method only after the writer has stopped. With locking enabled, it attempts to
    /// acquire an exclusive OS lock. A live SWMR writer runs without a lock and cannot be excluded
    /// by that check.
    ///
    /// Clearing the flags permits opening a file left marked by an interrupted writer. It does
    /// not repair corruption, validate data integrity, or guarantee that writing can safely
    /// continue. An interrupted page-buffer flush can leave silently incorrect data even when
    /// every checksum verifies, including fill values or bytes from a deleted object.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileLocked`] if another handle holds a conflicting lock and locking is
    /// enabled. Returns [`Error::Io`] if the file cannot be opened for reading and writing, or if
    /// locking, reading, writing, or synchronizing fails. Returns [`Error::Format`] if the
    /// superblock cannot be parsed or serialized.
    pub fn clear_status_flags(path: impl AsRef<Path>) -> Result<(), Error> {
        crate::file_lock::clear_status_flags_at(path.as_ref())
    }

    /// Create a new, empty HDF5 file at `path` and open it for reading and
    /// writing, so its contents can be built entirely through owned handles
    /// ([`Group::create_dataset`]/[`create_group`](Group::create_group), then
    /// [`commit`](Self::commit)).
    ///
    /// Overwrites any existing file at `path`. For an all-at-once write, use
    /// [`FileBuilder`](crate::FileBuilder) instead.
    #[doc(alias = "H5Fcreate")]
    pub fn create<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::create_with_options(
            path,
            FileCreateProperties::new(),
            FileAccessProperties::new(),
        )
    }

    /// Create a new, empty HDF5 file with explicit creation and access properties,
    /// then open it for reading and writing — see [`create`](Self::create).
    ///
    /// Mirrors `H5Fcreate(name, flags, fcpl_id, fapl_id)`: `create` carries the
    /// creation properties recorded in the new file (userblock, file-space
    /// strategy, library-version bounds), and `access` the properties governing
    /// the handle returned (locking policy, `fsync` cadence, chunk cache). Both are values, so a
    /// layout defined once can be reused across every file an application writes.
    ///
    /// A creation property is validated as the file is written, so an invalid
    /// userblock or page size surfaces here rather than when the properties were
    /// built. A file created with [`FileSpaceStrategy::Page`] and persistence enabled supports
    /// immediate [`Dataset::append`] and staged commits. With persistence disabled, staged
    /// edits require the mirror and the restrictions documented by [`File::open_rw`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the creation and access properties cannot be used
    /// together. Returns the build, write, or [`File::open_rw_with_options`] error if file
    /// creation or read-write opening fails.
    pub fn create_with_options<P: AsRef<std::path::Path>>(
        path: P,
        create: FileCreateProperties,
        access: FileAccessProperties,
    ) -> Result<Self, Error> {
        // Refuse a pair the reopen below would refuse, before anything is
        // written: this call promises a file *and* an open handle, and half of
        // that is worse than neither.
        if let Some(reason) = crate::edit::create_would_refuse_reopen(&create, &access) {
            return Err(Error::EditUnsupported(reason));
        }
        let mut builder = crate::writer::FileBuilder::new();
        builder.with_create_properties(create);
        let bytes = builder.finish()?;
        std::fs::write(path.as_ref(), bytes).map_err(Error::Io)?;
        Self::open_rw_with_options(path, access)
    }

    /// Applies staged edits and publishes pending free-space manager changes.
    ///
    /// Staged edits are applied as one transaction. Immediate [`Dataset::append`] calls publish
    /// their elements during the call. With no staged edits, the editor can publish manager changes
    /// pending from retired append allocations, including replacements that leave EOF unchanged.
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`].
    ///
    /// Outstanding [`Dataset`] and [`Group`] handles stay usable: a commit
    /// relocates object headers, and each handle looks its object up again by
    /// path on its first use afterwards, so a long-lived handle answers for the
    /// file the commit left rather than for the copy it moved away from. Two
    /// exceptions, both of which report rather than answer wrongly. A *read*
    /// through a handle onto an object the commit deleted, or replaced with one
    /// of a different kind ([`Error::NotADataset`] or [`Error::NotAGroup`]),
    /// fails the way opening it
    /// would; its write methods still address the file by path, so they stage
    /// and the commit refuses them. And a handle reached by object reference
    /// ([`Dataset::dereference`]) has no path to look up, so it returns
    /// [`Error::StaleHandle`] after staging, a commit, or teardown. Recovery that changes the
    /// root has the same effect. An ordinary [`sync`](Self::sync) preserves these handles when
    /// the root stays unchanged. Dereference again from a fresh read after relocation.
    ///
    /// A handle onto an object this commit *publishes* — one
    /// [`Group::create_group`], [`Group::create_group_with`] or
    /// [`Group::create_dataset`] handed back, or a lookup of a staged name found
    /// — starts reading its object here. Until then it answers
    /// [`Error::NotCommitted`] for anything needing
    /// bytes, and a refused commit leaves it doing so.
    ///
    /// The commit is durable when it returns, under the default
    /// [`SyncPolicy::Always`]; under
    /// [`SyncPolicy::OnClose`] it has reached the operating system and waits for
    /// a [`sync`](Self::sync).
    ///
    /// **A commit refused before it publishes leaves every dataset reading what
    /// it read before.** Almost everything such a commit writes lands where
    /// nothing reaches it until the commit's linearization point; the one edit
    /// that does not is a same-length [`Dataset::write`], which overwrites the
    /// dataset's existing block, and the refusal writes those bytes back on its
    /// way out. A refusal raised before the first write keeps the staged batch
    /// too, so it can be corrected and committed again (issue #316).
    /// Non-persistent PAGE sessions relocate value replacements and retain the original
    /// allocations, using only the superblock for publication.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    /// Close and reopen the file before editing again.
    /// Returns [`Error::EditUnsupported`] in a non-persistent PAGE session if publication bytes
    /// differ from the session image or expected framing after a failed write. Close and reopen
    /// the file before editing again.
    /// A subsequent commit first completes the publication barrier under the selected
    /// synchronization policy and verifies the superblock before consuming staged edits.
    /// The editor can report an I/O error after publishing the complete batch. Session reads
    /// while recovery is pending are insufficient to determine the publication outcome.
    /// Successful recovery during [`sync`](Self::sync) or a subsequent commit verifies and
    /// adopts the issued state. Reopening a valid file also establishes its stored state.
    /// An interrupted file can fail to reopen. Recovery failures leave later staged edits
    /// pending, and recovery that changes the root refreshes handles as described by
    /// [`sync`](Self::sync).
    ///
    /// Two failures do not carry that promise, and both call for **re-reading**
    /// the datasets the batch named rather than for a retry:
    ///
    /// - [`Error::CommitPartiallyApplied`], where the restore itself failed, so
    ///   a dataset may hold either value.
    /// - An error from a step *after* the commit published — repointing the
    ///   object references that named a moved object is the one that can raise
    ///   it. The batch is in the file and stays there; what failed is work the
    ///   commit owed afterwards. The file is valid either way.
    pub fn commit(&self) -> Result<(), Error> {
        self.with_mirror_session(Change::Relocating, |session| session.commit())
    }

    /// Copy the object at `src` to `dst` within this file (the in-file
    /// `H5Ocopy`), staged until [`commit`](Self::commit).
    ///
    /// A dataset whose storage was never allocated is copied as the storage it
    /// has — none — rather than as the fill value reading it answers with, so a
    /// schema-only dataset stays one. A dataset whose elements live in external
    /// files (`H5Pset_external`) carries that same empty storage while holding
    /// data this crate does not read, and is refused with
    /// [`Error::EditUnsupported`]. Copying it would leave its elements behind.
    ///
    /// Soft and external links are copied with their stored targets, so a relative soft link in
    /// the copy is resolved from the copied group.
    ///
    /// Requires a read-write file ([`File::open_rw`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if this file is read-only. The [`commit`](Self::commit) that
    /// applies the copy returns [`Error::EditUnsupported`] if the source reaches one object through
    /// more than one hard link, as an alias or a cycle does, and [`Error::TraversalLimitExceeded`]
    /// if it reaches more objects or hard links than a copy reads.
    pub fn copy(&self, src: &str, dst: &str) -> Result<(), Error> {
        self.with_mirror_session(Change::Relocating, |session| {
            session.copy(&ObjectPathBuf::parse(src), &ObjectPathBuf::parse(dst))
        })
    }

    /// Copy the object at `src` in `source` — a separate, buffered read-only
    /// file — into this file at `dst`: the cross-file `H5Ocopy`, staged until
    /// [`commit`](Self::commit).
    ///
    /// `source` must be a buffered file ([`File::open`] or [`File::from_bytes`],
    /// not [`File::open_streaming`]) that uses 8-byte offsets and has no
    /// userblock; anything else is refused with
    /// [`Error::EditUnsupported`]. The source
    /// subtree is read and validated eagerly, so `source` need not outlive this
    /// call — and so a source this cannot reproduce, external storage included,
    /// is refused by this call. Refusals that concern the *destination* — `dst`
    /// already exists, or its parent group does not — still come from `commit`.
    /// Requires a read-write destination ([`File::open_rw`]).
    ///
    /// Soft and external links are copied with their stored targets, so an absolute soft link in
    /// the copy is resolved from the root group of this file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the source reaches one object through more than one
    /// hard link, as an alias or a cycle does, [`Error::TraversalLimitExceeded`] if it reaches
    /// more objects or hard links than a copy reads, and [`Error::ReadOnly`] if this file is
    /// read-only.
    pub fn copy_from(&self, source: &File, src: &str, dst: &str) -> Result<(), Error> {
        self.with_mirror_session(Change::Relocating, |session| {
            session.copy_from(
                source,
                &ObjectPathBuf::parse(src),
                &ObjectPathBuf::parse(dst),
            )
        })
    }

    /// Report whether this file has structural edits staged but not yet applied
    /// by [`commit`](Self::commit) — [`Dataset::write`]/`set_attr`/`remove_attr`,
    /// [`Dataset::append_staged`], [`Group::create_group`]/`create_dataset`/
    /// `delete`/`set_attr`/`remove_attr`, and [`copy`](Self::copy)/
    /// [`copy_from`](Self::copy_from). Immediate [`Dataset::append`]s are never
    /// staged and do not count. Always `false` for a read-only file.
    ///
    /// A `commit` that refuses puts the staged set back untouched, so this still
    /// answers `true` afterwards and the same batch can be committed again — to
    /// the same refusal, until the session is dropped.
    pub fn has_staged_edits(&self) -> bool {
        match &self.inner.backend {
            Backend::Edit(m) => {
                let session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                session.has_staged_edits()
            }
            _ => false,
        }
    }

    /// Returns a snapshot of this writable session's logical size and reusable free bytes.
    ///
    /// Reports committed changes and immediate appends. Staged edits affect the statistics
    /// when applied by [`commit`](Self::commit). Unused classic aggregate reservations are
    /// included in the reusable total.
    /// See [`ReusableSpaceStats::reusable_free_bytes`] for reservation and persistence semantics.
    ///
    /// Computes the values from cached counters without allocating or enumerating free sections.
    /// The counter work takes constant time with respect to the number of free sections.
    ///
    /// The method acquires the session lock. On some platforms, the first lock initializes mutex
    /// state and can allocate. After initialization, polling does not allocate. Acquiring the
    /// lock can wait for another operation on the same session.
    ///
    /// These values describe session state. They do not predict whether a particular write fits
    /// or how much physical storage is available. For section enumeration, use
    /// [`reusable_free_sections`](Self::reusable_free_sections).
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file was opened read-only. Returns [`Error::Format`]
    /// with [`FormatError::OffsetOverflow`] if the reusable-byte sum exceeds [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure::{File, FileBuilder, MaxExtent};
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("session-stats.h5");
    /// # let mut builder = FileBuilder::new();
    /// # builder.create_dataset("log").with_i32_data(&[1; 4]).with_chunks(&[4])
    /// #     .with_maxshape(&[MaxExtent::Unlimited]);
    /// # builder.write(&path)?;
    /// let file = File::open_rw(&path)?;
    /// file.dataset("log")?.append(&[2; 4])?;
    /// let stats = file.reusable_space_stats()?;
    /// assert_eq!(stats.logical_size(), file.file_size());
    /// assert!(stats.reusable_free_bytes() <= stats.logical_size());
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn reusable_space_stats(&self) -> Result<ReusableSpaceStats, Error> {
        match &self.inner.backend {
            Backend::Edit(engine) => {
                let session = engine
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(session.reusable_space_stats()?)
            }
            _ => Err(Error::ReadOnly),
        }
    }

    /// Issues pending writes and synchronizes file content and metadata to durable storage.
    ///
    /// Synchronization is durable under every [`SyncPolicy`], including [`SyncPolicy::OnClose`].
    /// In a session rejected with [`Error::AllocationPublicationUncertain`], the editor discards
    /// queued storage writes and rejects synchronization. Other pending writes are issued before
    /// synchronization. New staged edits remain pending for [`commit`](Self::commit).
    /// Elements held by a [`BufferedAppender`] or an [`OwnedBufferedAppender`] require the
    /// appender's buffer to be flushed first.
    /// Call [`commit`](Self::commit) first to include pending free-space manager changes from
    /// immediate tail retirement.
    ///
    /// In a non-persistent PAGE session, synchronization also verifies and adopts any pending
    /// publication. The complete earlier batch may become visible even though its commit
    /// reported an I/O error. Session reads while recovery is pending are insufficient to
    /// determine the publication outcome. Successful recovery or reopening a valid file
    /// establishes the committed state. An interrupted file can fail to reopen.
    ///
    /// If recovery changes the root, existing [`Dataset`] and [`Group`] handles opened by path
    /// resolve their objects again on their next read. Address-only handles return
    /// [`Error::StaleHandle`]. Synchronization preserves handle revisions when the root stays
    /// unchanged, including an ordinary sync with no pending recovery.
    ///
    /// [`close`](Self::close) and dropping the last handle perform their own synchronization.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file has no editing session, or [`Error::FileClosed`] if
    /// it has been sealed. Returns [`Error::Io`] if writing or synchronization fails.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    /// Close and reopen the file before editing again.
    /// Returns [`Error::EditUnsupported`] if pending PAGE publication bytes differ from the
    /// accepted image or expected framing. Close and reopen the file before editing again.
    /// Returns [`Error::Format`] if recovery cannot read, parse, or represent the superblock.
    /// A failed recovery leaves later staged edits pending.
    #[doc(alias = "fsync")]
    pub fn sync(&self) -> Result<(), Error> {
        self.with_mirror_session(Change::Nothing, |session| session.force_sync())
    }

    /// Commits any staged edits and seals this file. The exclusive OS lock is
    /// released once the last handle derived from this file is also dropped.
    ///
    /// After `close`, a write through any surviving [`Dataset`]/[`Group`] handle
    /// or [`File`] clone returns [`Error::FileClosed`], and
    /// reads still work. Under [`FileSpaceStrategy::Aggr`] or [`FileSpaceStrategy::FsmAggr`],
    /// an ordinary read-write session releases unused aggregate tails after publishing the shorter
    /// EOA with the current root, provided publication completed successfully. A persistent session
    /// returns unused interior aggregate spare through replacement free-space managers, subject
    /// to its free-section threshold.
    /// A handle reached by [`Dataset::dereference`] reports
    /// [`Error::StaleHandle`] after the commit, while a handle opened by path
    /// re-resolves and keeps reading.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if another handle already closed the file. Returns
    /// [`Error::Io`] if writing, synchronizing, or truncating the file fails. Returns
    /// [`Error::Format`] if metadata geometry or address conversion is invalid. Errors from
    /// applying staged edits are returned as described in [`commit`](Self::commit).
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn close(self) -> Result<(), Error> {
        if matches!(self.inner.backend, Backend::Edit(_)) {
            // SWMR mode stages nothing — the staged surface is refused — so there
            // is nothing to commit, and it persists no free space, so there is
            // nothing to re-home.
            let swmr = self.inner.swmr_write;
            if !swmr {
                self.commit()?;
            }
            // Finalization preserves object addresses while updating free-space managers,
            // aggregate tails, and status flags. Existing handles remain readable.
            self.with_mirror_session(Change::InPlace, |session| {
                if !swmr {
                    session.finalize_persist()?;
                }
                // Forced under every policy, and covering everything above: this
                // call consumes the handle, so it is the last point at which any
                // of these writes can be ordered at all.
                session.force_sync()?;
                // Last, and only after that sync. A session's status flags stand
                // for writes that may still have been in memory — a SWMR writer's
                // pair, a page buffer's crash mark — and this is the point at
                // which none are.
                session.release_status_flags()
            })?;
            self.inner.closed.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Run `f` with the locked write session of a read-write file. `staged`
    /// distinguishes an edit applied by [`commit`](Self::commit) from an immediate
    /// one. Returns [`Error::ReadOnly`] for a read-only file,
    /// [`Error::FileClosed`] once the file is sealed by [`close`](Self::close),
    /// and [`Error::SwmrStagedUnsupported`] for a staged edit on a SWMR-writer
    /// file.
    fn with_mirror_session<R>(
        &self,
        change: Change,
        f: impl FnOnce(&mut WriteEngine) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.inner.with_engine_mut(change, f)
    }

    /// Returns an owned handle to the root group.
    ///
    /// The physical root address is resolved when the handle is used.
    pub fn root(&self) -> Group {
        Group {
            file: self.inner.clone(),
            state: RwLock::new(None),
            path: Some(ObjectPathBuf::root()),
            staged_birth: None,
        }
    }

    /// Resolves a path and returns an owned [`Dataset`] handle.
    ///
    /// Hard and soft links are followed. Relative soft targets start at their containing
    /// group, and absolute targets start at the file root. External links are unsupported.
    ///
    /// The dataset uses the file-wide chunk-cache default (configured with
    /// [`FileAccessProperties::with_chunk_cache`]). To override the cache for this
    /// one dataset, use [`dataset_with_options`](Self::dataset_with_options).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotADataset`] if the path names something that is not a
    /// dataset, and [`Error::NotAGroup`] if a component *along* the path is not
    /// a group: resolving `a/b/c` opens `a` and then `a/b` to look inside them,
    /// so a dataset at `a/b` reports `NotAGroup("a/b")` (issue #365).
    ///
    /// Returns [`FormatError::DanglingSoftLink`] if a soft target has a missing component,
    /// [`FormatError::SoftLinkTraversalLimit`] if resolution would follow more than 64 soft
    /// links, and [`FormatError::ExternalLinkUnsupported`] for an external target.
    pub fn dataset(&self, path: &str) -> Result<Dataset, Error> {
        self.dataset_with_options(path, DatasetAccessProperties::new())
    }

    /// Resolve a path and return an owned [`Dataset`] handle, applying per-dataset
    /// [`DatasetAccessProperties`] that override file-wide access defaults.
    ///
    /// This is the dataset-open-with-access-property-list path (HDF5's `dapl`):
    /// the properties' chunk cache corresponds to `H5Pset_chunk_cache` and takes
    /// precedence, for this dataset only, over the `H5Pset_cache`-style
    /// file-wide default.
    pub fn dataset_with_options(
        &self,
        path: &str,
        properties: DatasetAccessProperties,
    ) -> Result<Dataset, Error> {
        let chunk_cache = properties.resolved_chunk_cache(self.inner.access_properties.chunk_cache);
        let path = ObjectPathBuf::parse(path);
        match self.inner.staged_object(&path).map(|o| o.kind) {
            Some(StagedKind::Dataset) => {
                return Ok(Dataset::pending(self.inner.clone(), chunk_cache, path));
            }
            Some(StagedKind::Group) => return Err(Error::NotADataset(path.to_string())),
            None => {}
        }
        let revisions = self.inner.revisions();
        let addr = self.inner.resolve_path(&path.as_path())?;
        let hdr = self.inner.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DATA_LAYOUT) {
            return Err(Error::NotADataset(path.to_string()));
        }
        Ok(Dataset::new(
            self.inner.clone(),
            revisions.at(addr),
            hdr,
            chunk_cache,
            Some(path),
        ))
    }

    /// Resolves a path and returns an owned [`Group`] handle.
    ///
    /// Hard and soft links are followed. Relative soft targets start at their containing
    /// group, and absolute targets start at the file root. External links are unsupported.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotAGroup`] if the path names an object that is not a
    /// group, the way [`dataset`](Self::dataset) returns
    /// [`Error::NotADataset`] for the mirror case, and
    /// [`FormatError::PathNotFound`] if it names nothing.
    ///
    /// The same error reports a component *along* the path that is not a group,
    /// naming that component's own path rather than the one asked for: `a/b/c`
    /// stopped by a dataset at `a/b` reports `NotAGroup("a/b")` (issue #365).
    ///
    /// Returns [`FormatError::DanglingSoftLink`] if a soft target has a missing component,
    /// [`FormatError::SoftLinkTraversalLimit`] if resolution would follow more than 64 soft
    /// links, and [`FormatError::ExternalLinkUnsupported`] for an external target.
    pub fn group(&self, path: &str) -> Result<Group, Error> {
        let path = ObjectPathBuf::parse(path);
        match self.inner.staged_object(&path).map(|o| o.kind) {
            Some(StagedKind::Group) => {
                return Ok(Group::pending(self.inner.clone(), path));
            }
            Some(StagedKind::Dataset) => return Err(Error::NotAGroup(path.to_string())),
            None => {}
        }
        let revisions = self.inner.revisions();
        let addr = self.inner.resolve_path(&path.as_path())?;
        if !group_v2::is_group(&self.inner.parse_header(addr)?) {
            // Normalized, so that the same object refused here and refused by a
            // live handle below names itself the same way: a handle knows only
            // the normalized path it memoized.
            return Err(Error::NotAGroup(path.to_string()));
        }
        Ok(Group::new(
            self.inner.clone(),
            revisions.at(addr),
            Some(path),
        ))
    }

    /// Reads the file again to observe data appended by a concurrent writer.
    ///
    /// The physical file may be shorter than the recorded EOF while the writer extends it.
    /// Obtain new [`Dataset`] and [`Group`] handles after refreshing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SwmrUnsupported`] if the file was opened without [`File::open_swmr`].
    /// Returns [`Error::HandlesOutstanding`] if an owned object handle or a clone of this file
    /// is alive.
    pub fn refresh(&mut self) -> Result<(), Error> {
        let inner = Arc::get_mut(&mut self.inner).ok_or(Error::HandlesOutstanding)?;
        inner.refresh()
    }

    // --- delegating value getters (forward to the shared inner state) ---

    /// Returns the raw file bytes for an in-memory file, or an empty slice for a
    /// streaming file (which has no whole-file buffer).
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Returns the access properties this file was opened with.
    ///
    /// An open without options, such as [`File::open`], has [`FileAccessProperties::new`].
    pub fn access_properties(&self) -> FileAccessProperties {
        self.inner.access_properties()
    }

    /// Which backend this file's read-write session resolved to:
    /// [`EditBacking::Bounded`] when it reads through a handle, or
    /// [`EditBacking::Mirrored`] when it holds a whole-file image.
    ///
    /// This is how a caller who opened with [`MemoryStrategy::Auto`] finds out
    /// whether the fallback was taken, and so whether memory scales with the
    /// file. A file with no editing session — a read-only open, a streaming open
    /// — reports `None`.
    ///
    /// The answer is an [`EditBacking`] rather than the [`MemoryStrategy`] that
    /// was asked for, because `Auto` is a preference between the two backends and
    /// not an outcome either can report; `.into()` converts back when a later
    /// reopen should be pinned to what this one got.
    pub fn edit_backing(&self) -> Option<EditBacking> {
        self.inner.edit_backing()
    }

    /// What this file's metadata cache has done, and what it is holding.
    ///
    /// [`FileAccessProperties::with_metadata_cache`] sets a byte budget before
    /// any read has happened; this is how a caller finds out whether it was the
    /// right one. See [`MetadataCacheStats`] for which figure answers which
    /// question. Together the two are the `hdf5-pure` counterpart of HDF5's
    /// `H5Fget_mdc_hit_rate` and `H5Fget_mdc_size`.
    ///
    /// `None` where there is no metadata cache to report on: a buffered
    /// [`open`](Self::open) or [`from_bytes`](Self::from_bytes), which already
    /// holds the whole file; a mirrored read-write session, for the same reason;
    /// or a streaming or bounded open left at the default disabled budget.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// use hdf5_pure::{File, FileAccessProperties, MetadataCacheConfig};
    ///
    /// let properties =
    ///     FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(8 << 20));
    /// let file = File::open_streaming_with_options("data.h5", properties)?;
    /// for name in file.root().datasets()? {
    ///     let _ = file.dataset(&name)?.read_raw()?;
    /// }
    ///
    /// let stats = file.metadata_cache_stats().expect("the budget enabled a cache");
    /// println!("{:?} over {} reads, {} evicted", stats.hit_rate(), stats.reads(), stats.evictions());
    /// # Ok(())
    /// # }
    /// ```
    pub fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.inner.metadata_cache_stats()
    }

    /// Zero this file's metadata-cache counters, keeping every cached entry.
    ///
    /// HDF5's `H5Freset_mdc_hit_rate_stats`, for measuring one phase of a
    /// program rather than a whole run: the reads that populate a cache miss by
    /// definition, so a hit rate taken over the run charges the steady state for
    /// the warm-up. Reset after warming to measure the part that repeats.
    ///
    /// It evicts nothing: occupancy, which
    /// [`metadata_cache_stats`](Self::metadata_cache_stats) also reports, is a
    /// measurement of the cache rather than a tally of its history. A file with
    /// no metadata cache ignores the call.
    pub fn reset_metadata_cache_stats(&self) {
        self.inner.reset_metadata_cache_stats();
    }

    /// Returns a reference to the parsed superblock.
    ///
    /// The address fields hold the encoded values. The base and end-of-file
    /// addresses are absolute. The other metadata addresses are base-relative [`StoredAddress`]
    /// values, interpreted through the opened image's address context during navigation.
    /// For a relocated image, the effective base and EOF differ from the encoded values.
    pub fn superblock(&self) -> &Superblock {
        self.inner.superblock()
    }

    /// Resolves a stored address using this file's effective address base.
    ///
    /// For a relocated image, the effective base is the physical signature position. The encoded
    /// [`Superblock::base_address`] may differ. The resolved position need not lie within the
    /// file's current length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `stored` is the undefined marker at
    /// this file's address width, or [`FormatError::OffsetOverflow`] if the sum exceeds
    /// [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, UserblockSize};
    ///
    /// # let mut builder = FileBuilder::new();
    /// # builder.with_userblock(UserblockSize::try_from(512)?);
    /// # let file = File::from_bytes(builder.finish()?)?;
    /// let stored = file.superblock().root_group_address;
    /// let physical = file.resolve_address(stored)?;
    /// assert_eq!(physical.get(), 512 + stored.get());
    /// assert_eq!(file.relative_address(physical)?, stored);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn resolve_address(&self, stored: StoredAddress) -> Result<PhysicalAddress, FormatError> {
        if stored.is_undefined(self.superblock().offset_size) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: stored.get(),
                offset_size: self.superblock().offset_size.get(),
            });
        }
        self.addresses().resolve(stored)
    }

    /// Returns the stored address of a physical position using this file's effective address base.
    ///
    /// Uses the same effective base as [`resolve_address`](Self::resolve_address), including for
    /// relocated images. The physical position need not lie within the file's current length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if `physical` lies below the effective base, or
    /// [`FormatError::DefinedAddressIsUndefined`] if the result is the undefined marker at this
    /// file's address width.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, PhysicalAddress, UserblockSize};
    ///
    /// # let mut builder = FileBuilder::new();
    /// # builder.with_userblock(UserblockSize::try_from(512)?);
    /// # let file = File::from_bytes(builder.finish()?)?;
    /// let origin = PhysicalAddress::new(512);
    /// let stored = file.relative_address(origin)?;
    /// assert_eq!(stored.get(), 0);
    /// assert_eq!(file.resolve_address(stored)?, origin);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    ///
    /// A stored address cannot be passed as a physical position:
    ///
    /// ```compile_fail,E0308
    /// use hdf5_pure::{File, StoredAddress};
    ///
    /// fn convert(file: &File, stored: StoredAddress) {
    ///     let _ = file.relative_address(stored);
    /// }
    /// ```
    pub fn relative_address(
        &self,
        physical: PhysicalAddress,
    ) -> Result<StoredAddress, FormatError> {
        let stored = self.addresses().relative(physical)?;
        if stored.is_undefined(self.superblock().offset_size) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: stored.get(),
                offset_size: self.superblock().offset_size.get(),
            });
        }
        Ok(stored)
    }

    /// Returns the address and length widths validated when this file was opened.
    pub(crate) fn widths(&self) -> FormatWidths {
        self.inner.widths()
    }

    /// Returns the file-space configuration recorded in the superblock extension.
    ///
    /// Returns `Ok(None)` if the file has no File Space Info message. Callers that need the
    /// default configuration can apply [`Option::unwrap_or_default`] to the returned option.
    /// Strategies without free-space managers use the normalized values described by
    /// [`FileSpaceSettings`]. A version 0 message uses [`crate::FileSpacePageSize::DEFAULT`] for its
    /// absent page-size field.
    ///
    /// Works with buffered, streaming, and editing handles. Editing handles read the current
    /// published metadata, excluding staged edits. Use [`Self::reusable_space_stats`] and
    /// [`Self::reusable_free_sections`] to inspect a writable session's reusable space, and
    /// [`Self::file_space_info`] for raw message fields.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the message or its superblock-extension metadata is malformed
    /// or unreadable, including an invalid address, object header, or page size.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileSpaceStrategy};
    /// # use hdf5_pure::{FileBuilder, FileSpaceSettings};
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("settings.h5");
    /// # let mut builder = FileBuilder::new();
    /// # builder.with_file_space_settings(
    /// #     FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
    /// #         .with_persistence(true)
    /// #         .with_threshold(128),
    /// # );
    /// # builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    /// # builder.write(&path)?;
    /// let file = File::open(&path)?;
    /// if let Some(settings) = file.file_space_settings()? {
    ///     assert_eq!(settings.strategy(), FileSpaceStrategy::FsmAggr);
    ///     assert_eq!(settings.threshold(), 128);
    /// }
    /// if let Some(report) = file.persisted_free_sections()? {
    ///     println!("free bytes: {}", report.free_sections().free_bytes());
    ///     for manager in report.managers() {
    ///         println!(
    ///             "slot {}: {:?}, {} sections",
    ///             manager.slot(),
    ///             manager.role(),
    ///             manager.free_sections().section_count()
    ///         );
    ///     }
    /// }
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn file_space_settings(&self) -> Result<Option<FileSpaceSettings>, Error> {
        Ok(self.inner.parse_file_space_info()?.map(|info| {
            let FileSpaceInfo {
                strategy,
                persist,
                threshold,
                page_size,
                ..
            } = info;
            FileSpaceSettings::new(strategy)
                .with_persistence(persist)
                .with_threshold(threshold)
                .with_page_size(page_size)
        }))
    }

    /// Returns the parsed File Space Info message, including its storage-layout fields.
    ///
    /// This advanced metadata view includes the page-end metadata threshold, the end of
    /// allocation before manager placement, and manager-address slots. Use
    /// [`Self::file_space_settings`] for the recorded configuration and
    /// [`Self::persisted_free_sections`] for stored free sections.
    ///
    /// Returns `Ok(None)` if the message is absent. Works with buffered, streaming, and editing
    /// handles. Editing handles read the current published metadata, excluding staged edits.
    /// Parsed version 0 messages retain their six manager slots when persistent and have no
    /// pre-manager end-of-allocation field, as described by [`FileSpaceInfo`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the message or its superblock-extension metadata is malformed
    /// or unreadable, including an invalid address, object header, or page size.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::File;
    /// # use hdf5_pure::{FileBuilder, FileSpaceSettings, FileSpaceStrategy};
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("raw-file-space-info.h5");
    /// # let mut builder = FileBuilder::new();
    /// # builder.with_file_space_settings(
    /// #     FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
    /// #         .with_persistence(true),
    /// # );
    /// # builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    /// # builder.write(&path)?;
    /// let file = File::open(&path)?;
    /// if let Some(info) = file.file_space_info()? {
    ///     println!("EOA before managers: {:?}", info.eoa_pre_fsm);
    ///     println!("manager addresses: {:?}", info.manager_addrs);
    ///     println!("page-end metadata threshold: {}", info.page_end_meta_threshold);
    /// }
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn file_space_info(&self) -> Result<Option<FileSpaceInfo>, Error> {
        Ok(self.inner.parse_file_space_info()?)
    }

    /// Returns the sections recorded in the file's persisted free-space managers.
    ///
    /// Reads managers through buffered or streaming sources without requiring a whole-file
    /// copy. Returns `None` if the File Space Info message is absent or persistence is disabled.
    /// Configured persistence with zero sections returns `Some` with an empty report.
    ///
    /// Section and manager addresses are base-relative [`StoredAddress`] values. For a relocated
    /// source image, the reader resolves stored addresses against the effective base of that image.
    ///
    /// Each defined manager has a report, including empty managers and managers with an unknown
    /// role. The combined count and free-byte total equal the sums of the per-manager values.
    /// [`PersistedFreeSections::manager_metadata_bytes`] reports the file bytes allocated to
    /// manager headers and section lists, including unused list capacity. A manager with zero
    /// free sections can occupy file storage.
    /// Session-reusable space is available through
    /// [`reusable_free_sections`](Self::reusable_free_sections).
    ///
    /// # Errors
    ///
    /// Returns [`Error::FreeSectionInspectionUnsupported`] if this is a writable handle.
    /// Returns [`Error::OverlappingFreeSections`] if free sections overlap within or across
    /// managers, including duplicate extents. Header and section-list checksums are verified
    /// regardless of the `checksum` feature.
    /// Returns [`Error::Format`] if File Space Info records the same manager header address in
    /// multiple slots, File Space Info or manager metadata is malformed or truncated,
    /// manager allocations overlap, an allocation or section exceeds the file allocation, or an
    /// address or byte total overflows. Source read errors are also reported through [`Error::Format`].
    ///
    /// # Examples
    ///
    /// Absence of persistence and configured persistence with zero sections are distinct:
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};
    ///
    /// let file = File::from_bytes(FileBuilder::new().finish()?)?;
    /// assert_eq!(file.persisted_free_sections()?, None);
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_file_space_settings(
    ///     FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
    ///         .with_persistence(true),
    /// );
    /// let file = File::from_bytes(builder.finish()?)?;
    /// let report = file.persisted_free_sections()?.expect("persistence is configured");
    /// assert_eq!(report.free_sections().section_count(), 0);
    /// assert_eq!(report.free_sections().free_bytes(), 0);
    /// assert_eq!(report.manager_metadata_bytes(), 0);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    ///
    /// Reads sections through a streaming handle, prints free bytes and manager metadata bytes,
    /// and compares the combined count with the manager counts:
    ///
    /// ```
    /// # use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};
    /// use hdf5_pure::StoredAddress;
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("free-sections.h5");
    /// # let mut builder = FileBuilder::new();
    /// # builder.with_file_space_settings(
    /// #     FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
    /// #         .with_persistence(true),
    /// # );
    /// # builder.create_dataset("a").with_i32_data(&[1; 100]);
    /// # builder.create_dataset("b").with_i32_data(&[2; 400]);
    /// # builder.create_dataset("c").with_i32_data(&[3; 100]);
    /// # builder.write(&path)?;
    /// # {
    /// #     let file = File::open_rw(&path)?;
    /// #     file.root().delete("b")?;
    /// #     file.commit()?;
    /// # }
    ///
    /// let file = File::open_streaming(&path)?;
    /// let report = file.persisted_free_sections()?.expect("persistence is configured");
    /// let sections = report.free_sections();
    /// println!("Free section bytes: {}", sections.free_bytes());
    /// println!("Persisted manager metadata bytes: {}", report.manager_metadata_bytes());
    /// let section = sections.sections().first().expect("deleted storage is free");
    /// let start: StoredAddress = section.address();
    /// let end: StoredAddress = section.end_address();
    /// assert_eq!(end.get() - start.get(), section.length().get());
    /// assert_eq!(
    ///     report.managers().iter().map(|manager| manager.free_sections().section_count())
    ///         .sum::<usize>(),
    ///     sections.section_count(),
    /// );
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn persisted_free_sections(&self) -> Result<Option<PersistedFreeSections>, Error> {
        if matches!(self.inner.backend, Backend::Edit(_)) {
            return Err(Error::FreeSectionInspectionUnsupported);
        }
        let Some(info) = self.inner.parse_file_space_info()? else {
            return Ok(None);
        };
        if !info.persist {
            return Ok(None);
        }
        let stored_eof = self
            .inner
            .superblock
            .eof_address
            .checked_sub(self.inner.superblock.base_address.get())
            .ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(
                    "the end of file is below the stored base address".into(),
                )
            })?;
        Ok(Some(PersistedFreeSections::read(
            &self.inner.source(),
            self.inner.widths,
            self.inner.addresses().effective_base(),
            stored_eof,
            &info,
        )?))
    }

    /// Returns the free sections currently reusable by this editing session.
    ///
    /// Reports committed changes and immediate in-place appends. Staged edits affect the report after
    /// [`commit`](Self::commit). Sections use [`PhysicalAddress`] offsets from the beginning of
    /// the file, counting any userblock prefix.
    ///
    /// The report can include session aggregate reservations and space absent from persisted
    /// managers. Persisted manager contents are available through
    /// [`persisted_free_sections`](Self::persisted_free_sections) on a read-only handle.
    /// For the same session state, the byte total equals [`ReusableSpaceStats::reusable_free_bytes`].
    /// Adjacent sections can remain separate across page types.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file was opened read-only. Returns [`Error::Format`]
    /// if a section has zero length or its end address or the total free bytes overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};
    /// use hdf5_pure::PhysicalAddress;
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("reusable-sections.h5");
    /// # let mut builder = FileBuilder::new();
    /// # builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    /// # builder.create_dataset("a").with_i32_data(&[1; 100]);
    /// # builder.create_dataset("b").with_i32_data(&[2; 400]);
    /// # builder.create_dataset("c").with_i32_data(&[3; 100]);
    /// # builder.write(&path)?;
    ///
    /// let file = File::open_rw(&path)?;
    /// file.root().delete("b")?;
    /// file.commit()?;
    /// let stats = file.reusable_space_stats()?;
    /// let sections = file.reusable_free_sections()?;
    /// assert!(sections.section_count() > 0);
    /// assert_eq!(stats.reusable_free_bytes(), sections.free_bytes());
    /// for section in sections.sections() {
    ///     let start: PhysicalAddress = section.address();
    ///     let end: PhysicalAddress = section.end_address();
    ///     assert_eq!(end.get() - start.get(), section.length().get());
    /// }
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn reusable_free_sections(&self) -> Result<FreeSections<PhysicalAddress>, Error> {
        match &self.inner.backend {
            Backend::Edit(engine) => {
                let session = engine
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(session.reusable_free_sections()?)
            }
            _ => Err(Error::ReadOnly),
        }
    }

    /// Returns the physical byte length of the backing store.
    ///
    /// This includes any userblock prefix and trailing bytes. The encoded
    /// [`Superblock::eof_address`] may use a different coordinate frame if the image was relocated.
    pub fn file_size(&self) -> u64 {
        self.inner.file_size()
    }

    /// The minimum library version required to read this file, derived from its
    /// superblock version (the *low bound* of HDF5's `H5Fget_libver_bounds`).
    pub fn libver_bound(&self) -> LibVer {
        self.inner.libver_bound()
    }

    /// A `Source` view over the backend, for the streaming-capable paths.
    pub(crate) fn source(&self) -> SourceView<'_> {
        self.inner.source()
    }

    /// The whole-file byte image when this file is buffered in memory; `None`
    /// for a streaming file. Used by cross-file object copy.
    pub(crate) fn in_memory_image(&self) -> Option<&[u8]> {
        self.inner.in_memory_image()
    }

    /// The access mode this file was opened under, which a cross-file copy passes
    /// to every parse it makes of this file's objects.
    pub(crate) fn access_mode(&self) -> AccessMode {
        self.inner.access_mode()
    }

    /// Returns the effective base used to interpret stored metadata addresses in this image.
    pub(crate) fn base_address(&self) -> BaseAddress {
        self.inner.base_address()
    }

    /// Returns the mapping between stored addresses and physical positions in this image.
    pub(crate) fn addresses(&self) -> AddressContext {
        self.inner.addresses()
    }

    /// Returns the class of the object whose header is at `header`, and the hard links of the
    /// object if it is a group.
    ///
    /// The result includes the parsed header for later reuse.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the object header does not parse, or if the object is a group
    /// whose link storage does not parse.
    pub(crate) fn hard_links_at(
        &self,
        header: PhysicalAddress,
    ) -> Result<(ObjectClass, Vec<GroupEntry>, ParsedHeader), Error> {
        let revision = self.inner.content_revision();
        let parsed = self.inner.parse_header(header.get())?;
        let class = ObjectClass::of(&parsed);
        let links = match class {
            ObjectClass::Group => self.inner.group_children(&parsed)?,
            ObjectClass::Dataset | ObjectClass::NamedDatatype | ObjectClass::Unknown => Vec::new(),
        };
        Ok((
            class,
            links,
            ParsedHeader {
                file: Arc::clone(&self.inner),
                address: header,
                revision,
                header: parsed,
            },
        ))
    }

    /// Reads the attributes of the object whose header is at `header`.
    ///
    /// Consumes `parsed` under the source, address, and revision checks of [`ParsedHeader`].
    /// For streaming sources, the reader retains attribute messages and attribute storage
    /// information during a fresh parse.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `parsed` belongs to another
    /// source or address. Propagates object-header and attribute-reading errors.
    pub(crate) fn attr_messages_at(
        &self,
        header: PhysicalAddress,
        parsed: Option<ParsedHeader>,
    ) -> Result<Vec<AttributeMessage>, Error> {
        let parsed = parsed
            .map(|parsed| parsed.reuse(self, header))
            .transpose()?
            .flatten();
        let header = match (parsed, &self.inner.backend) {
            (Some(parsed), _) => parsed,
            (None, Backend::Streaming(source)) => {
                ObjectHeader::parse_from_source_filtered_with_widths(
                    &SourceMetadata(source.as_ref()),
                    self.inner.access_mode(),
                    header.get(),
                    self.inner.widths,
                    self.inner.addresses().effective_base(),
                    MessageFilter::Only(&mut |message_type, _| {
                        matches!(
                            message_type,
                            MessageType::ATTRIBUTE | MessageType::ATTRIBUTE_INFO
                        )
                    }),
                )?
            }
            (None, _) => self.inner.parse_header(header.get())?,
        };
        self.inner.attr_messages_of(&header)
    }

    /// Returns a handle to the group whose object header is at `header`, which `path` identifies
    /// from the root group.
    pub(crate) fn group_at(&self, header: PhysicalAddress, path: ObjectPathBuf) -> Group {
        let at = self.inner.revisions().at(header.get());
        Group::new(Arc::clone(&self.inner), at, Some(path))
    }

    /// Returns a handle to the dataset whose object header is at `header`, which `path`
    /// identifies from the root group.
    ///
    /// The handle has the chunk cache of the file's access properties. Consumes `parsed` under
    /// the source, address, and revision checks of [`ParsedHeader`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the object header does not parse, or with
    /// [`FormatError::Internal`] if `parsed` belongs to another source or address. Returns
    /// [`Error::NotADataset`] if the object has another class.
    pub(crate) fn dataset_at(
        &self,
        header: PhysicalAddress,
        path: ObjectPathBuf,
        parsed: Option<ParsedHeader>,
    ) -> Result<Dataset, Error> {
        let parsed = self.header_at(header, parsed)?;
        if ObjectClass::of(&parsed) != ObjectClass::Dataset {
            return Err(Error::NotADataset(path.to_string()));
        }
        let at = self.inner.revisions().at(header.get());
        let chunk_cache = DatasetAccessProperties::new()
            .resolved_chunk_cache(self.inner.access_properties.chunk_cache);
        Ok(Dataset::new(
            Arc::clone(&self.inner),
            at,
            parsed,
            chunk_cache,
            Some(path),
        ))
    }

    /// Returns the datatype of the committed datatype whose object header is at `header`.
    ///
    /// Consumes `parsed` under the source, address, and revision checks of [`ParsedHeader`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the object header or its Datatype message does not parse,
    /// or with [`FormatError::Internal`] if `parsed` belongs to another source or address. Returns
    /// [`Error::MissingMessage`] if the header has no Datatype message.
    pub(crate) fn named_datatype_at(
        &self,
        header: PhysicalAddress,
        parsed: Option<ParsedHeader>,
    ) -> Result<Datatype, Error> {
        self.inner.header_datatype(&self.header_at(header, parsed)?)
    }

    /// Reuses a current parsed header or reads it at `address`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `parsed` belongs to another
    /// source or address. Propagates object-header parsing errors.
    fn header_at(
        &self,
        address: PhysicalAddress,
        parsed: Option<ParsedHeader>,
    ) -> Result<ObjectHeader, Error> {
        match parsed
            .map(|parsed| parsed.reuse(self, address))
            .transpose()?
            .flatten()
        {
            Some(header) => Ok(header),
            None => Ok(self.inner.parse_header(address.get())?),
        }
    }
}

/// A parsed object header with its source, physical address, and content revision.
///
/// Reuse requires the same shared file state and address. The reader parses again after an
/// in-process content revision change. The caller is responsible for the consistency of external
/// source bytes.
pub(crate) struct ParsedHeader {
    file: Arc<FileInner>,
    address: PhysicalAddress,
    revision: u64,
    header: ObjectHeader,
}

impl ParsedHeader {
    /// Returns the allocated capacities of the message vector and message bodies, in bytes.
    pub(crate) fn allocation_bytes(&self) -> usize {
        self.header.messages.capacity() * core::mem::size_of::<HeaderMessage>()
            + self
                .header
                .messages
                .iter()
                .map(|message| message.data.capacity())
                .sum::<usize>()
    }

    /// Returns the header if its recorded content revision is current.
    ///
    /// Returns [`None`] after an in-process content revision change.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the shared file state or physical address differs.
    fn reuse(
        self,
        file: &File,
        address: PhysicalAddress,
    ) -> Result<Option<ObjectHeader>, FormatError> {
        let Self {
            file: owner,
            address: stored,
            revision,
            header,
        } = self;
        if !Arc::ptr_eq(&owner, &file.inner) || stored != address {
            return Err(FormatError::Internal(
                "a parsed object header belongs to another source or address".into(),
            ));
        }
        Ok((revision == file.inner.content_revision()).then_some(header))
    }
}

// ---------------------------------------------------------------------------
// Object reference target
// ---------------------------------------------------------------------------

/// The resolved target of an HDF5 object reference (`H5R_OBJECT`): either a
/// group or a dataset.
///
/// Produced by [`Dataset::dereference`]. MATLAB `.mat` files use object
/// references pervasively — a cell array stores one reference per element, and
/// the `#subsystem#` machinery references its payloads — so resolving a
/// reference to the group or dataset it names is the foundation for reading
/// those structures.
///
/// The [`Dataset`](Object::Dataset) handle is boxed: it carries a parsed object
/// header and is much larger than a [`Group`](Object::Group) handle, so boxing
/// keeps `Object` (and a `Vec<Object>`) compact without a size disparity. The
/// `Box` derefs transparently, so `&obj_dataset` is usable wherever a
/// `&Dataset` is expected.
///
/// Non-exhaustive: a reference can name an object kind this crate does not yet
/// resolve — a committed (named) datatype is refused with
/// [`FormatError::InvalidObjectReference`]
/// today — so match with a `_` arm.
#[non_exhaustive]
pub enum Object {
    /// The reference points at a group's object header.
    Group(Group),
    /// The reference points at a dataset's object header.
    Dataset(Box<Dataset>),
}

impl std::fmt::Debug for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Object::Group(_) => f.write_str("Object::Group"),
            Object::Dataset(_) => f.write_str("Object::Dataset"),
        }
    }
}

// ---------------------------------------------------------------------------
// Group handle
// ---------------------------------------------------------------------------

/// A group that exists only as a staged edit, handed to
/// [`Group::create_group_with`]'s closure so a whole subtree — attributes,
/// nested groups, datasets — can be described in one call.
///
/// It is a convenience, not the only way in: [`Group::create_group`] and
/// [`Group::create_group_with`] both return a live [`Group`] handle onto the
/// staged group, and everything this offers can be staged through that handle
/// instead. Reach for the closure when the shape of the subtree is known at the
/// call, and for the handle when it is built up by code that takes a `&Group`.
///
/// Every method stages; nothing is written until [`File::commit`].
///
/// The closure holding this records into a buffer rather than into the file's
/// writable session, so nothing is locked while it runs. The recorded operations
/// are applied together when it returns, which is why an object staged here is
/// addressable only once the closure has returned.
pub struct StagedGroup<'a> {
    ops: &'a mut Vec<StagedOp>,
    path: ObjectPathBuf,
}

impl StagedGroup<'_> {
    /// Stage an attribute on this group, applied with its creation on
    /// [`File::commit`].
    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> &mut Self {
        self.ops.push(StagedOp::SetGroupAttr {
            path: self.path.clone(),
            name: name.to_string(),
            value,
        });
        self
    }

    /// Stage an empty subgroup of this group.
    ///
    /// `path` may have several components, which stages a group under this one, or begin with
    /// `/`, which stages it from the root group, both on the terms
    /// [`Group::create_group`] sets out.
    ///
    /// To configure it in the same commit, use
    /// [`create_group_with`](Self::create_group_with). To get a [`Group`] handle
    /// onto it, look it up by name once this closure has returned, or stage it
    /// through [`Group::create_group`] instead, which hands one back.
    pub fn create_group(&mut self, path: &str) -> &mut Self {
        self.create_group_with(path, |_| {})
    }

    /// Stage a subgroup of this group, configured through `build`.
    pub fn create_group_with(
        &mut self,
        path: &str,
        build: impl FnOnce(&mut StagedGroup<'_>),
    ) -> &mut Self {
        let child = self.path.join_path(&ObjectPath::parse(path));
        self.ops.push(StagedOp::CreateGroup(child.clone()));
        let mut staged = StagedGroup {
            ops: &mut *self.ops,
            path: child,
        };
        build(&mut staged);
        self
    }

    /// Stage a dataset in this group, configured through `build`.
    ///
    /// `path` may have several components, as it may for [`create_group`](Self::create_group).
    pub fn create_dataset(
        &mut self,
        path: &str,
        build: impl FnOnce(&mut DatasetBuilder),
    ) -> &mut Self {
        let mut builder = DatasetBuilder::new(path);
        build(&mut builder);
        self.ops.push(StagedOp::CreateDataset {
            path: self.path.join_path(&ObjectPath::parse(path)),
            builder: Box::new(builder),
        });
        self
    }
}

/// One edit recorded by a [`StagedGroup`] closure, replayed onto the writable
/// session after the closure returns.
///
/// The indirection is what keeps user code off the session lock: the closure
/// touches only this buffer, so calling back into the same [`File`] from inside
/// it is at worst wrongly ordered rather than a deadlock (issue #200).
enum StagedOp {
    CreateGroup(ObjectPathBuf),
    SetGroupAttr {
        path: ObjectPathBuf,
        name: String,
        value: AttrValue,
    },
    CreateDataset {
        path: ObjectPathBuf,
        /// Boxed because a `DatasetBuilder` dwarfs the other variants, and a
        /// closure staging many groups would otherwise pay its size per entry.
        builder: Box<DatasetBuilder>,
    },
}

impl StagedOp {
    /// Record this edit on the session. Applied in the order the closure made
    /// the calls, so a group is always staged before its own attributes and
    /// children.
    fn apply(self, session: &mut WriteEngine) -> Result<(), Error> {
        match self {
            StagedOp::CreateGroup(path) => session.create_group(&path),
            StagedOp::SetGroupAttr { path, name, value } => {
                session.set_group_attr(&path, &name, value)
            }
            StagedOp::CreateDataset { path, builder } => {
                session.stage_created_dataset(&path, *builder)
            }
        }
    }
}

/// An owned handle to an HDF5 group.
///
/// The handle names the group by its root-relative path and remembers where that
/// path resolved to. A [`File::commit`] rewrites and relocates object headers,
/// so the memo is worked out again on the first use after any edit and the
/// handle goes on answering for the same group — see [`File::commit`] for the
/// two cases that report instead. Cloning gives a second handle to the same
/// group.
pub struct Group {
    file: Arc<FileInner>,
    /// Where this group's object header sits, as of the file revision it was
    /// resolved at. Re-resolved on first use after an edit could have moved it;
    /// see [`Group::header_address`]. A group carries no parsed header of its
    /// own — it re-reads one per call — so the address is the whole memo, and the
    /// content revision the [`Resolution`] carries beside it names no header this
    /// handle read and is never read back.
    ///
    /// `None` for a group this session has *staged* and not yet committed: there
    /// is no header to name one. Such a handle resolves its path on every use,
    /// and installs a memo the first time a commit gives it something to point
    /// at.
    state: RwLock<Option<Resolution>>,
    /// Root-relative path of this group (the root group's own path has no
    /// components), used to address the group and its children for write
    /// operations on a read-write file, and to find it again after an edit moved
    /// it. `None` for a group reached by object reference
    /// ([`Dataset::dereference`]), which has no resolvable path.
    path: Option<ObjectPathBuf>,
    /// The session's staged generation when this handle was made onto a staged
    /// creation, and `None` for a handle opened onto a group in the file.
    ///
    /// It is what keeps such a handle from being retargeted at a different
    /// object: see [`Standing`] and [`FileInner::staged_standing`].
    staged_birth: Option<u64>,
}

impl Clone for Group {
    /// Clones share nothing but the open file: the clone is a second handle to
    /// the same group, resolved as of the same revision.
    fn clone(&self) -> Self {
        Self {
            file: Arc::clone(&self.file),
            state: RwLock::new(*self.state.read().unwrap_or_else(PoisonError::into_inner)),
            path: self.path.clone(),
            staged_birth: self.staged_birth,
        }
    }
}

impl std::fmt::Debug for Group {
    /// Reports the handle as it stands, without re-resolving: a `Debug` that
    /// read the file could fail, and one that failed would have nothing to
    /// print. `staged` is true for a group this session has created and not yet
    /// committed, which has no address to report.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("Group")
            .field("path", &self.path)
            .field("staged", &(self.staged_birth.is_some() && state.is_none()))
            .finish()
    }
}

impl Group {
    /// Build a handle for a group whose header was found at `at`.
    fn new(file: Arc<FileInner>, at: Resolution, path: Option<ObjectPathBuf>) -> Self {
        Self {
            file,
            state: RwLock::new(Some(at)),
            path,
            staged_birth: None,
        }
    }

    /// Build a handle for a group this session has staged and not yet
    /// committed, which has no header to memoize.
    ///
    /// It addresses the group by name from the moment it is staged: further
    /// creations, deletions and attribute edits under it are staged through it
    /// like any other handle. Anything that has to *read* the group reports
    /// [`Error::NotCommitted`] until [`File::commit`], after which the first use
    /// resolves the path and the handle behaves as if it had been opened by
    /// name.
    ///
    /// The session's staged generation is taken here so that the handle can tell
    /// that commit from a *withdrawal* of the same staging, which leaves the
    /// path meaning whatever the file holds — see [`Standing`].
    fn pending(file: Arc<FileInner>, path: ObjectPathBuf) -> Self {
        Self {
            staged_birth: file.staged_generation(),
            file,
            state: RwLock::new(None),
            path: Some(path),
        }
    }

    /// This group's object-header address (base-adjusted, file-absolute), worked
    /// out again from its path if an edit could have moved it since the memo was
    /// taken. Also what resolves an object reference that points at this group.
    ///
    /// Returns [`Error::StaleHandle`] for a handle
    /// that has no path to re-resolve — one an object reference produced — once
    /// a commit has run under it, and the resolution's own error (a
    /// `PathNotFound`, say, for a group a commit deleted) when the path no
    /// longer names anything. A group this session has staged and not committed
    /// has no header at all, and that is [`Error::NotCommitted`]. A commit that
    /// replaces this group with a dataset of the same name (issue #305) leaves
    /// the path naming something that is not a group, and that is
    /// [`Error::NotAGroup`].
    pub(crate) fn header_address(&self) -> Result<u64, Error> {
        let memo = *self.state.read().unwrap_or_else(PoisonError::into_inner);
        if let Some(memo) = memo {
            if memo.address_revision == self.file.address_revision() {
                return Ok(memo.address);
            }
        }
        let at = self
            .file
            .locate_staged(self.path.as_ref(), memo, self.staged_birth)?;
        // An initial root handle takes its address from the superblock without reading a header.
        // Every other resolution is classified before it is memoized.
        let initial_root =
            memo.is_none() && self.path.as_ref().is_some_and(ObjectPathBuf::is_empty);
        if !initial_root && !group_v2::is_group(&self.file.parse_header(at.address)?) {
            // A path-less handle never reaches here: it failed the short-circuit
            // above, and `locate` answers `StaleHandle` for one whose address
            // memo it cannot reuse.
            return Err(Error::NotAGroup(reported_path(self.path.as_ref())));
        }
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        // Two threads can re-resolve at once. The older answer must not land on
        // top of the newer one, or the newer handle would go on serving an
        // address the file has already moved past. A handle that had no memo
        // takes this one: any address beats naming nothing.
        if state.is_none_or(|memo| at.address_revision >= memo.address_revision) {
            *state = Some(at);
        }
        Ok(at.address)
    }

    /// List the names of datasets in this group.
    ///
    /// To read from the datasets themselves, prefer
    /// [`iter_datasets`](Self::iter_datasets): it hands back opened handles for
    /// the cost of this call, where opening each name separately re-walks the
    /// group once per member.
    ///
    /// A dataset this session has staged and not yet committed is listed too,
    /// once — a name that is both on disk and staged is the replacement of issue
    /// #305, and the staged object is what that name already resolves to.
    pub fn datasets(&self) -> Result<Vec<String>, Error> {
        let (entries, staged) = self.children_and_staged()?;
        let members = staged_members(&staged, StagedKind::Dataset, &entries);
        let mut names = Vec::new();
        for entry in &entries {
            if superseded(&staged, &entry.name) {
                continue;
            }
            let hdr = self.file.parse_header(self.file.entry_address(entry)?)?;
            if has_message(&hdr, MessageType::DATA_LAYOUT) {
                names.push(entry.name.clone());
            }
        }
        names.extend(members);
        Ok(names)
    }

    /// Open every dataset in this group, each paired with its name.
    ///
    /// This is the walk to reach for when the members themselves are what you
    /// want — their attributes, shapes or data — rather than a list of names.
    /// [`datasets`](Self::datasets) already parses every child's object header to
    /// tell a dataset from a group, and then keeps only the name, so following it
    /// with a [`dataset`](Self::dataset) call per entry re-walks the group's link
    /// structure and re-parses that same header. This keeps what it read, and
    /// costs one enumeration of the group rather than one per member.
    ///
    /// **This walk is for taking every member, or nearly every one.** Telling a
    /// dataset from a group means parsing its header, so the whole group is
    /// enumerated and every child's header parsed before the iterator is
    /// returned — breaking out early saves nothing, and reaching one known member
    /// this way costs far more than [`dataset`](Self::dataset) does. Only the
    /// handle construction is deferred to each step, and that is not where the
    /// cost is.
    ///
    /// The headers of the members are held for the length of the walk, since each
    /// one is what its handle is built from. That is bounded by the group being
    /// walked rather than by the file, but it is proportional to the group: a
    /// header carries a compact dataset's data and its compact attributes inline,
    /// so a large group of such datasets is a large allocation.
    ///
    /// Each dataset gets the file-wide chunk-cache default; to override the cache
    /// for one, open it by name with
    /// [`dataset_with_options`](Self::dataset_with_options).
    ///
    /// Members arrive in the order the group's link structure yields them — the
    /// same order [`datasets`](Self::datasets) reports, which is not necessarily
    /// sorted, with this session's staged datasets after them. Each handle is a
    /// snapshot taken when the iterator was built, so a [`File::commit`] that
    /// runs mid-walk is not reflected in the members still to come; re-open the
    /// group to see past it.
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open("runs.h5")?;
    /// for (name, dataset) in file.root().iter_datasets()? {
    ///     println!("{name}: {:?}", dataset.shape());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn iter_datasets(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = (String, Dataset)> + use<>, Error> {
        let revisions = self.file.revisions();
        let (entries, staged) = self.children_and_staged()?;
        // Taken before the entries are consumed below, since which staged
        // members survive depends on the names the file already holds.
        let staged_members = staged_members(&staged, StagedKind::Dataset, &entries);
        // A member is either an on-disk header this walk read, or a staged
        // creation with no header to read yet.
        let mut members: Vec<(String, Option<(u64, ObjectHeader)>)> = Vec::new();
        for entry in entries {
            if superseded(&staged, &entry.name) {
                continue;
            }
            let address = self.file.entry_address(&entry)?;
            let hdr = self.file.parse_header(address)?;
            if has_message(&hdr, MessageType::DATA_LAYOUT) {
                members.push((entry.name, Some((address, hdr))));
            }
        }
        members.extend(staged_members.into_iter().map(|name| (name, None)));
        let file = Arc::clone(&self.file);
        let parent = self.path.clone();
        let chunk_cache = DatasetAccessProperties::new()
            .resolved_chunk_cache(self.file.access_properties.chunk_cache);
        Ok(members.into_iter().map(move |(name, on_disk)| {
            let path = LinkName::new(&name).and_then(|leaf| child_path_of(parent.as_ref(), leaf));
            let dataset = match on_disk {
                Some((address, header)) => Dataset::new(
                    Arc::clone(&file),
                    revisions.at(address),
                    header,
                    chunk_cache,
                    path,
                ),
                // Only a group with a path of its own reports staged children,
                // and a staged name is one path component, so a staged member
                // always has a path of its own.
                None => Dataset::pending(
                    Arc::clone(&file),
                    chunk_cache,
                    path.expect("a staged member is a link name under a parent with a path"),
                ),
            };
            (name, dataset)
        }))
    }

    /// The names of children that are committed (`H5Tcommit`) datatype objects:
    /// an object header carrying a datatype and neither data nor links.
    ///
    /// Such an object is the third kind HDF5 links into a group, and it appears
    /// in neither [`datasets`](Self::datasets) nor [`groups`](Self::groups) — so a
    /// walk that asks only for those two passes over one without noticing. Read
    /// the type itself with [`named_datatype`](Self::named_datatype).
    pub fn named_datatypes(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(self.file.entry_address(entry)?)?;
            if is_named_datatype(&hdr) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// Returns the datatype the committed (`H5Tcommit`) object `path` names holds.
    ///
    /// `path` resolves as it does for [`dataset`](Self::dataset), and its last component is a name
    /// [`named_datatypes`](Self::named_datatypes) reports for the group that holds it.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::PathNotFound`] if a component of `path` reaches nothing,
    /// [`Error::NotAGroup`] if an object along the way is not a group, and
    /// [`Error::NotANamedDatatype`] if `path` identifies an object of another kind, as `H5Topen`
    /// rejects one.
    pub fn named_datatype(&self, path: &str) -> Result<Datatype, Error> {
        self.file
            .header_datatype(&self.named_datatype_header(path)?)
    }

    /// Returns how many things reference the committed (`H5Tcommit`) datatype `path` names:
    /// its hard links, plus every dataset and attribute that names it.
    ///
    /// This is HDF5's own object reference count (`H5Oget_info`'s `rc`), and what
    /// says whether unlinking the name would destroy the type or merely stop it
    /// being reachable through the link. A version 1 object header keeps the count
    /// in its prefix, a version 2 header in an Object Reference Count message,
    /// and a version 2 header without that message has exactly one reference,
    /// which is what the format means by omitting it.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::PathNotFound`], [`Error::NotAGroup`] and
    /// [`Error::NotANamedDatatype`] on the terms [`named_datatype`](Self::named_datatype) sets out,
    /// and
    /// [`FormatError::UnexpectedEof`] for an Object Reference Count message too short to hold its
    /// count.
    pub fn named_datatype_references(&self, path: &str) -> Result<u32, Error> {
        let hdr = self.named_datatype_header(path)?;
        if let Some(count) = hdr.reference_count {
            return Ok(count);
        }
        let Ok(msg) = find_message(&hdr, MessageType::OBJECT_REFERENCE_COUNT) else {
            return Ok(1);
        };
        // version(1) + count(4).
        let body = self.file.message_body(msg)?;
        if body.len() < 5 {
            return Err(Error::Format(FormatError::UnexpectedEof {
                expected: 5,
                available: body.len(),
            }));
        }
        Ok(u32::from_le_bytes([body[1], body[2], body[3], body[4]]))
    }

    /// Returns the object header of the committed datatype `path` identifies.
    ///
    /// The one place the by-name datatype lookups classify what they reached, so
    /// that an object this rejects cannot be one
    /// [`named_datatypes`](Self::named_datatypes) would list. Reached the way
    /// [`group`](Self::group) and [`dataset`](Self::dataset) reach theirs, which
    /// looks each component up by name.
    fn named_datatype_header(&self, path: &str) -> Result<ObjectHeader, Error> {
        let relative = ObjectPath::parse(path);
        let address = self.resolve_path(&relative)?;
        let hdr = self.file.parse_header(address)?;
        if !is_named_datatype(&hdr) {
            return Err(Error::NotANamedDatatype(self.root_relative(&relative)));
        }
        Ok(hdr)
    }

    /// List the names of subgroups in this group.
    ///
    /// To descend into the subgroups themselves, prefer
    /// [`iter_groups`](Self::iter_groups), which hands back opened handles for
    /// the cost of this call.
    ///
    /// A group this session has staged and not yet committed is listed too, on
    /// the terms [`datasets`](Self::datasets) sets out.
    pub fn groups(&self) -> Result<Vec<String>, Error> {
        let (entries, staged) = self.children_and_staged()?;
        let members = staged_members(&staged, StagedKind::Group, &entries);
        let mut names = Vec::new();
        for entry in &entries {
            if superseded(&staged, &entry.name) {
                continue;
            }
            let hdr = self.file.parse_header(self.file.entry_address(entry)?)?;
            if group_v2::is_group(&hdr) {
                names.push(entry.name.clone());
            }
        }
        names.extend(members);
        Ok(names)
    }

    /// Open every subgroup of this group, each paired with its name.
    ///
    /// The counterpart to [`iter_datasets`](Self::iter_datasets), and the way to
    /// recurse without paying a [`group`](Self::group) lookup per child: that
    /// lookup re-walks this group's link structure, which a walk of the whole
    /// tree would otherwise repeat once per subgroup.
    ///
    /// As with [`iter_datasets`](Self::iter_datasets), the whole group is
    /// enumerated and classified before the iterator is returned, so this is the
    /// walk for taking every subgroup rather than for reaching one — breaking out
    /// early saves nothing. A [`Group`] handle carries no parsed header, so
    /// unlike `iter_datasets` this holds none of them.
    ///
    /// Members arrive in the order the group's link structure yields them — the
    /// same order [`groups`](Self::groups) reports, which is not necessarily
    /// sorted, with this session's staged groups after them.
    ///
    /// ```no_run
    /// # use hdf5_pure::{Error, Group};
    /// fn total_datasets(group: &Group) -> Result<usize, Error> {
    ///     let mut n = group.datasets()?.len();
    ///     for (_, child) in group.iter_groups()? {
    ///         n += total_datasets(&child)?;
    ///     }
    ///     Ok(n)
    /// }
    /// ```
    pub fn iter_groups(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = (String, Group)> + use<>, Error> {
        let revisions = self.file.revisions();
        let (entries, staged) = self.children_and_staged()?;
        // Taken before the entries are consumed, as in `iter_datasets`.
        let staged_members = staged_members(&staged, StagedKind::Group, &entries);
        // `None` for a staged member, which has no address to resolve until the
        // commit places its header.
        let mut members: Vec<(String, Option<u64>)> = Vec::new();
        for entry in entries {
            if superseded(&staged, &entry.name) {
                continue;
            }
            // A `Group` handle carries no parsed header, so the header that
            // classified this child is dropped here rather than held for the
            // length of the walk.
            let address = self.file.entry_address(&entry)?;
            if group_v2::is_group(&self.file.parse_header(address)?) {
                members.push((entry.name, Some(address)));
            }
        }
        members.extend(staged_members.into_iter().map(|name| (name, None)));
        let file = Arc::clone(&self.file);
        let parent = self.path.clone();
        Ok(members.into_iter().map(move |(name, address)| {
            let path = LinkName::new(&name).and_then(|leaf| child_path_of(parent.as_ref(), leaf));
            let group = match address {
                Some(address) => Group::new(Arc::clone(&file), revisions.at(address), path),
                // As in `iter_datasets`: a staged member is one path component
                // under a group that has a path of its own.
                None => Group::pending(
                    Arc::clone(&file),
                    path.expect("a staged member is a link name under a parent with a path"),
                ),
            };
            (name, group)
        }))
    }

    /// Read all attributes of this group.
    ///
    /// Each value takes the [`AttrValue`] variant that describes its on-disk
    /// encoding, so the variant reflects the charset, width and dataspace its
    /// writer chose rather than the shape of the data alone: a one-element array
    /// stays an array, an ASCII string does not arrive as a UTF-8
    /// [`String`](AttrValue::String), and a 16-bit integer arrives as
    /// [`I16`](AttrValue::I16) rather than widened, a 32-bit float as
    /// [`F32`](AttrValue::F32). Prefer the accessors —
    /// [`AttrValue::as_str`], [`as_strings`](AttrValue::as_strings),
    /// [`as_i64`](AttrValue::as_i64) and the rest — over matching on the variant,
    /// unless the encoding is the thing you care about. **The variant may become
    /// more specific in a future release** as `AttrValue` grows further ones
    /// (variable-length strings, say), and a `_` arm is required regardless
    /// because the enum is `#[non_exhaustive]`.
    ///
    /// An attribute whose datatype has no `AttrValue` representation is omitted
    /// from the map rather than reported as an error. Read
    /// [`attr_datatypes`](Self::attr_datatypes) to see it.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let hdr = self.file.parse_header(self.header_address()?)?;
        self.file.attrs_of(&hdr)
    }

    /// The exact on-disk [`Datatype`] of every attribute on this group, keyed by
    /// name — including compound field offsets, integer widths and enumeration
    /// members.
    ///
    /// This is the type channel to [`attrs`](Self::attrs)'s value channel, the
    /// pair a dataset already has in [`Dataset::datatype`] and its `read_*`
    /// methods. An [`AttrValue`] is a deliberately lossy view of the value, so an
    /// attribute's byte order, sub-width precision, string padding and
    /// enumeration members are recoverable only from here — its width is not,
    /// since [`attrs`](Self::attrs) keeps that. Its *rank* is not either: that
    /// lives in the
    /// dataspace, which nothing public exposes, so a rank-2 attribute still
    /// reads as a flat `AttrValue` array with no way to recover its shape.
    ///
    /// **Every attribute message is reported, including the ones `attrs` omits**
    /// because no `AttrValue` can carry them, so a name missing from that map can
    /// be told from one the object does not have.
    ///
    /// A **committed** datatype — one created with `H5Tcommit`, what netCDF-4
    /// writes for a user-defined type and what h5py writes for
    /// `f["t"] = np.dtype(...)` — is stored as a reference to the type's own
    /// object header rather than inline, and is resolved to the type it names.
    /// What it does *not* carry is the name: two attributes sharing `/mytype`
    /// report the same [`Datatype`] as one that spells it out inline.
    ///
    /// A boolean attribute is the case that needs both channels. libhdf5
    /// gives `H5T_NATIVE_HBOOL` — what h5py writes for every `np.bool_` — a
    /// [`Datatype::Enumeration`] of `FALSE` and `TRUE` over an 8-bit base, and
    /// `attrs` decodes it through that base, so the value arrives as `0` or `1`
    /// and only the datatype records that it was a bool.
    pub fn attr_datatypes(&self) -> Result<HashMap<String, Datatype>, Error> {
        Ok(self
            .attr_messages()?
            .into_iter()
            .map(|a| (a.name, a.datatype))
            .collect())
    }

    /// Every attribute message on this group as it is encoded on disk, in the
    /// order the header holds them.
    ///
    /// [`attrs`](Self::attrs) decodes each into an [`AttrValue`], which loses the
    /// encoding; this keeps it. Repack copies from here so an attribute survives
    /// a rewrite unchanged, and falls back to the decoded map only where the
    /// bytes are not position-independent.
    pub(crate) fn attr_messages(&self) -> Result<Vec<crate::attribute::AttributeMessage>, Error> {
        let hdr = self.file.parse_header(self.header_address()?)?;
        self.file.attr_messages_of(&hdr)
    }

    /// Returns the dataset `path` names, relative to this group.
    ///
    /// `path` is one link name or several separated by `/`, and a `path` that begins with `/`
    /// resolves from the root group of the file instead. libhdf5 starts an absolute name at
    /// the root the same way (`H5G__traverse_real` in `H5Gtraverse.c`, HDF5 1.14.6). A repeated or
    /// a trailing separator separates nothing, and a `.` component identifies the group reached so
    /// far, so `a/b`, `a//b/` and `./a/b` identify one dataset.
    ///
    /// Hard and soft links are followed. Relative soft targets start at their containing
    /// group, and absolute targets start at the file root. External links are unsupported.
    ///
    /// The dataset uses the file-wide chunk-cache default. To override the cache
    /// for this one dataset, use
    /// [`dataset_with_options`](Self::dataset_with_options).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::PathNotFound`] if a component of `path` reaches nothing,
    /// [`Error::NotADataset`] if `path` identifies an object that is not a dataset, and
    /// [`Error::NotAGroup`] if an object along the way is not a group. Paths in these errors are
    /// relative to the root group, except through an object-reference handle, which has no path
    /// of its own.
    ///
    /// Returns [`FormatError::DanglingSoftLink`] if a soft target has a missing component,
    /// [`FormatError::SoftLinkTraversalLimit`] if resolution would follow more than 64 soft
    /// links, and [`FormatError::ExternalLinkUnsupported`] for an external target.
    pub fn dataset(&self, path: &str) -> Result<Dataset, Error> {
        self.dataset_with_options(path, DatasetAccessProperties::new())
    }

    /// Returns the dataset `path` names, applying per-dataset
    /// [`DatasetAccessProperties`] that override the file-wide access defaults (HDF5's
    /// `dapl`, `H5Pset_chunk_cache`).
    ///
    /// `path` resolves as it does for [`dataset`](Self::dataset).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::PathNotFound`], [`Error::NotADataset`] and [`Error::NotAGroup`] on
    /// the terms [`dataset`](Self::dataset) sets out.
    pub fn dataset_with_options(
        &self,
        path: &str,
        properties: DatasetAccessProperties,
    ) -> Result<Dataset, Error> {
        let chunk_cache = properties.resolved_chunk_cache(self.file.access_properties.chunk_cache);
        let relative = ObjectPath::parse(path);
        let child = self.child_path(&relative);
        if let Some(child) = &child {
            match self.file.staged_object(child).map(|o| o.kind) {
                Some(StagedKind::Dataset) => {
                    return Ok(Dataset::pending(
                        self.file.clone(),
                        chunk_cache,
                        child.clone(),
                    ));
                }
                Some(StagedKind::Group) => return Err(Error::NotADataset(child.to_string())),
                None => {}
            }
        }
        let revisions = self.file.revisions();
        let address = self.resolve_path(&relative)?;
        let hdr = self.file.parse_header(address)?;
        if !has_message(&hdr, MessageType::DATA_LAYOUT) {
            return Err(Error::NotADataset(self.root_relative(&relative)));
        }
        Ok(Dataset::new(
            self.file.clone(),
            revisions.at(address),
            hdr,
            chunk_cache,
            child,
        ))
    }

    /// Returns the subgroup `path` names, relative to this group.
    ///
    /// `path` resolves as it does for [`dataset`](Self::dataset). A relative `path` that spells no
    /// link, `""` or `"."`, identifies this group, and an absolute one, `"/"`, identifies the root
    /// group.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotAGroup`] if `path` identifies an object that is not a group, the way
    /// [`dataset`](Self::dataset) returns [`Error::NotADataset`] for the mirror case, and
    /// [`FormatError::PathNotFound`] if a component of `path` reaches nothing.
    ///
    /// Returns [`FormatError::DanglingSoftLink`] if a soft target has a missing component,
    /// [`FormatError::SoftLinkTraversalLimit`] if resolution would follow more than 64 soft
    /// links, and [`FormatError::ExternalLinkUnsupported`] for an external target.
    pub fn group(&self, path: &str) -> Result<Group, Error> {
        let relative = ObjectPath::parse(path);
        let child = self.child_path(&relative);
        if let Some(child) = &child {
            match self.file.staged_object(child).map(|o| o.kind) {
                Some(StagedKind::Group) => {
                    return Ok(Group::pending(self.file.clone(), child.clone()));
                }
                Some(StagedKind::Dataset) => return Err(Error::NotAGroup(child.to_string())),
                None => {}
            }
        }
        let revisions = self.file.revisions();
        let address = self.resolve_path(&relative)?;
        if !group_v2::is_group(&self.file.parse_header(address)?) {
            return Err(Error::NotAGroup(self.root_relative(&relative)));
        }
        Ok(Group::new(self.file.clone(), revisions.at(address), child))
    }

    /// Returns the address of the object header `path` identifies, walking from this group, or
    /// from the root group where `path` is absolute.
    ///
    /// [`FileInner::group_child`] looks each component up by name, which reads the one Link
    /// message a compact group holds for it and enumerates a version 1 or a dense group
    /// (issue #228). A `path` that spells no link identifies the object the walk starts from.
    ///
    /// The root is the one handle whose object nothing classifies: every other `Group` comes from
    /// a lookup that classified it (`File::group`, `Group::group`, `iter_groups`,
    /// `object_at_relative`) or re-resolves through `header_address`, which classifies again.
    /// `File::root` takes the superblock's word for it, so a root whose header describes some
    /// other object is rejected here, under the empty path a root handle stores.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::PathNotFound`] if a component reaches nothing and
    /// [`Error::NotAGroup`] if an object along the way is not a group. Both report the object
    /// through [`root_relative`](Self::root_relative).
    fn resolve_path(&self, path: &ObjectPath<'_>) -> Result<u64, Error> {
        let own = self.header_address()?;
        Ok(group_v2::resolve_link_path(
            self.file.mirror_root_address()?,
            own,
            self.path.as_ref().unwrap_or(&ObjectPathBuf::root()),
            path,
            |address, name| self.file.group_child(address, name),
        )?)
    }

    /// Returns the root-relative path of the object `path` identifies from this group, or from
    /// the root group where `path` is absolute, and `None` if this group itself has no resolvable
    /// path (reached by object reference).
    fn child_path(&self, path: &ObjectPath<'_>) -> Option<ObjectPathBuf> {
        Some(self.path.as_ref()?.join_path(path))
    }

    /// Returns the path an error reports the object `path` reaches by: its path from the root
    /// group.
    ///
    /// One object is reported one way whether the caller holds a [`File`] or a [`Group`] handle,
    /// and from the root, so that the caller can go and open what the error reported. For a group
    /// reached by object reference, whose own path is `None`, the result is `path` alone, in the
    /// form [`ObjectPath`] displays.
    fn root_relative(&self, path: &ObjectPath<'_>) -> String {
        self.child_path(path)
            .map_or_else(|| path.to_string(), |child| child.to_string())
    }

    /// Creates an empty subgroup at `path` within this group, staged until
    /// [`File::commit`], and returns a handle to it.
    ///
    /// The handle addresses the new group straight away: further groups,
    /// datasets, deletions and attributes can be staged through it, and
    /// [`group`](Self::group) finds it by name from the same session. `path` may have several
    /// components, and every group above the new one has to be in the file already or
    /// staged in this session: `create_group("a/b")` with no `a` anywhere stages it, and the
    /// commit rejects the whole batch (issue #533). Reading it
    /// — its attributes, or a member's data — reports
    /// [`Error::NotCommitted`] until the commit,
    /// after which the same handle answers for the group in the file. Deleting
    /// it before the commit withdraws the staging, and the handle then reports
    /// [`Error::StagingWithdrawn`] without
    /// answering for whatever else the path may name.
    ///
    /// **The handle keeps the file's exclusive OS lock alive**, as every
    /// [`Group`] and [`Dataset`] handle does, so `let g = root.create_group(..)?`
    /// holds what `root.create_group(..)?;` used to drop and a reopen of the
    /// file fails until it goes. [`File::close`] states the rule.
    ///
    /// [`create_group_with`](Self::create_group_with) builds a whole subtree in
    /// one call instead.
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`]. A `path` that begins with `/` stages
    /// the group from the root, as [`dataset`](Self::dataset) resolves an absolute path, and one
    /// that spells no link, `""`, `"."` or `"/"`, is rejected with
    /// [`Error::EditUnsupported`], since a write addresses a link. A name the file already
    /// links to is refused here with
    /// [`Error::EditUnsupported`] unless this
    /// session also deletes it — a [replacement](Self::delete) — since there
    /// would otherwise be no new object for the handle to address. A name this
    /// session already staged a *dataset* at is refused for the same reason —
    /// one name cannot mean two objects, and the handle would answer for
    /// whichever was staged first — while staging the same group twice is
    /// allowed and hands back another handle onto that one group, which is how
    /// attributes and children are added to a group already staged.
    /// [`delete`](Self::delete) withdraws a staged creation, which frees the
    /// name for another.
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("runs.h5")?;
    /// let run = file.root().create_group("run2")?;
    /// run.create_dataset("signal", |b| {
    ///     b.with_f64_data(&[1.0, 2.0, 3.0]);
    /// })?;
    /// file.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_group(&self, path: &str) -> Result<Group, Error> {
        self.create_group_with(path, |_| {})
    }

    /// Creates a subgroup at `path` within this group, configuring it through
    /// `build` (attributes, nested groups and datasets), staged until
    /// [`File::commit`], and returns a handle to it.
    ///
    /// The closure describes a whole subtree in one call, which is what makes it
    /// worth having over [`create_group`](Self::create_group) plus calls on the
    /// handle that returns; either way the new group is addressable by name
    /// before the commit, and the handle this returns is the same one
    /// [`group`](Self::group) would give back.
    ///
    /// The closure records into a buffer rather than into the file itself, and
    /// what it stages is applied together when it returns — so reading the same
    /// [`File`] from inside it sees the file as it was before this call, and
    /// everything it staged is addressable by name from the moment it returns.
    ///
    /// The handle this returns keeps the file's exclusive OS lock alive, as
    /// every [`Group`] and [`Dataset`] handle does; see [`File::close`].
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`]. Name collisions are rejected on
    /// [`create_group`](Self::create_group)'s terms, for the group this creates
    /// and for everything the closure stages under it.
    ///
    /// ```no_run
    /// # use hdf5_pure::{AttrValue, File};
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("runs.h5")?;
    /// file.root().create_group_with("run2", |g| {
    ///     g.set_attr("count", AttrValue::I64(7));
    ///     g.set_attr("label", AttrValue::String("second".into()));
    /// })?;
    /// file.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_group_with(
        &self,
        path: &str,
        build: impl FnOnce(&mut StagedGroup<'_>),
    ) -> Result<Group, Error> {
        let child = self.child_edit_path(path)?;
        let mut ops = vec![StagedOp::CreateGroup(child.clone())];
        build(&mut StagedGroup {
            ops: &mut ops,
            path: child.clone(),
        });
        self.apply_staged(ops)?;
        Ok(Group::pending(self.file.clone(), child))
    }

    /// Creates a dataset at `path` within this group, configuring it through `build`
    /// (shape, data, chunks, filters, …), staged until [`File::commit`], and
    /// returns a handle to it.
    ///
    /// The handle addresses the new dataset straight away, which is what lets a
    /// writer cache one per column while it is still building the schema. It
    /// answers [`shape`](Dataset::shape), [`maxshape`](Dataset::maxshape),
    /// [`dtype`](Dataset::dtype), [`datatype`](Dataset::datatype),
    /// [`is_chunked`](Dataset::is_chunked) and [`filters`](Dataset::filters)
    /// from what was staged, and [`append_staged`](Dataset::append_staged) folds
    /// more elements into the pending creation. Anything that reads the
    /// dataset's bytes — a `read_*`, its attributes, an immediate
    /// [`append`](Dataset::append) — reports
    /// [`Error::NotCommitted`] until the commit, after which the same handle
    /// reads the dataset in the file. Deleting it before the commit withdraws the
    /// staging, and the handle then reports [`Error::StagingWithdrawn`] without
    /// answering for whatever else the path may name.
    ///
    /// **The handle keeps the file's exclusive OS lock alive**, as every
    /// [`Group`] and [`Dataset`] handle does, so `let ds = root.create_dataset(..)?`
    /// holds what `root.create_dataset(..)?;` used to drop and a reopen of the
    /// file fails until it goes. [`File::close`] states the rule.
    ///
    /// As with [`create_group_with`](Self::create_group_with), the closure
    /// configures a builder rather than the file, so it may read the same
    /// [`File`] — it will see the file as it was before this call.
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`]. `path` may have several components or begin with `/`, and one
    /// that spells no link is rejected, all on the terms
    /// [`create_group`](Self::create_group) sets out.
    /// A name the file already
    /// links to is refused here with
    /// [`Error::EditUnsupported`] unless this
    /// session also deletes it — a [replacement](Self::delete) — since there
    /// would otherwise be no new dataset for the handle to address. A name this
    /// session already staged a creation at is refused for the same reason:
    /// two creations at one path are one name for two objects, and the handle
    /// would answer for whichever was staged first. [`delete`](Self::delete)
    /// withdraws a staged creation, which frees the name for another.
    ///
    /// ```no_run
    /// # use hdf5_pure::{File, MaxExtent};
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("runs.h5")?;
    /// let mut col = file.root().create_dataset("col", |b| {
    ///     b.with_f64_data(&[])
    ///         .with_shape(&[0])
    ///         .with_maxshape(&[MaxExtent::Unlimited])
    ///         .with_chunks(&[512]);
    /// })?;
    /// col.append_staged(|a| {
    ///     a.append_f64(&[1.0, 2.0, 3.0]);
    /// })?;
    /// file.commit()?;
    /// assert_eq!(col.read_f64()?, vec![1.0, 2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_dataset(
        &self,
        path: &str,
        build: impl FnOnce(&mut DatasetBuilder),
    ) -> Result<Dataset, Error> {
        let child = self.child_edit_path(path)?;
        let mut builder = DatasetBuilder::new(path);
        build(&mut builder);
        self.apply_staged(vec![StagedOp::CreateDataset {
            path: child.clone(),
            builder: Box::new(builder),
        }])?;
        Ok(Dataset::pending(
            self.file.clone(),
            DatasetAccessProperties::new()
                .resolved_chunk_cache(self.file.access_properties.chunk_cache),
            child,
        ))
    }

    /// Deletes the object at `path` from this group, staged until
    /// [`File::commit`]. See [`create_group`](Self::create_group) for the
    /// file-mode rules.
    ///
    /// Creating a new object at the same path in the same commit *replaces* it:
    /// the removal is applied before the addition and one superblock write
    /// publishes both, so a rotation costs one commit and the path is never
    /// momentarily absent.
    ///
    /// Deleting an object this session **staged** and has not committed
    /// withdraws that staging instead — its attributes, appends and staged
    /// children go with it — since there is no link in the file to unlink. Where
    /// the deletion was part of a replacement, the plain deletion of the file's
    /// own object is what remains. A handle onto the withdrawn creation reports
    /// [`Error::StagingWithdrawn`] from then on:
    /// it names nothing, and the object the file holds at that path is the one
    /// this session is removing.
    ///
    /// Deleting a group carries its whole subtree away, but only a commit that
    /// builds that group *again* can put anything back under it: staging a
    /// creation below a deleted path that nothing recreates is a batch `commit`
    /// refuses, and until it does the file's own children still own their names.
    /// The root itself cannot be deleted.
    ///
    /// A `path` that spells no link, `""`, `"."` or `"/"`, is rejected with
    /// [`Error::EditUnsupported`], since a deletion takes a link. libhdf5 rejects `"."` from
    /// `H5L__delete_cb` (`H5Lint.c`) and an empty name in its argument check
    /// (`H5VL_setup_name_args` in `H5VLint.c`, HDF5 1.14.6). A `path` that begins with `/` deletes
    /// from the root group, as [`create_group`](Self::create_group) stages from there.
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("ring.h5")?;
    /// file.root().delete("t0")?;
    /// file.root().create_dataset("t0", |b| { b.with_i32_data(&[1, 2, 3]); })?;
    /// file.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&self, path: &str) -> Result<(), Error> {
        self.with_child_session(path, |session, child| session.delete(child))
    }

    /// Add or update an attribute on this group, staged until [`File::commit`].
    /// Use [`remove_attr`](Self::remove_attr) to remove one. The
    /// [`root`](File::root) group's attributes are edited the same way.
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`]. An attribute set too large
    /// for the object header — more than eight attributes, or one whose message
    /// the header's 2-byte size field cannot describe — is written to a fractal
    /// heap on `commit`, as it is when the whole file is written, and a group
    /// already storing its attributes in one is rebuilt.
    pub fn set_attr(&self, name: &str, value: AttrValue) -> Result<(), Error> {
        self.with_own_session(|session, path| session.set_group_attr(path, name, value))
    }

    /// Remove an attribute from this group, staged until [`File::commit`].
    /// See [`set_attr`](Self::set_attr) for the file-mode rules.
    pub fn remove_attr(&self, name: &str) -> Result<(), Error> {
        self.with_own_session(|session, path| session.remove_group_attr(path, name))
    }

    /// Runs `f` with the writable session and the root-relative path of the child `path`
    /// identifies.
    ///
    /// # Errors
    ///
    /// Returns what [`child_edit_path`](Self::child_edit_path) returns for `path`, and whatever
    /// `f` returns.
    fn with_child_session<R>(
        &self,
        path: &str,
        f: impl FnOnce(&mut WriteEngine, &ObjectPathBuf) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let child = self.child_edit_path(path)?;
        self.file
            .with_engine_mut(Change::Relocating, |session| f(session, &child))
    }

    /// Refuse an edit staged *through* a handle whose own staged creation has
    /// been withdrawn.
    ///
    /// Every read through such a handle already reports it: they resolve the
    /// path, and [`FileInner::locate_staged`] is where that is decided. The
    /// staging calls resolve nothing — they address the file by path — so this
    /// is where they ask the same question, and without it an edit staged
    /// through a withdrawn group would land under whatever the file holds at its
    /// path, which is the object the session is deleting.
    ///
    /// Costs nothing for a handle opened onto an object in the file: those carry
    /// no birth generation, so the session is never locked to answer.
    fn refuse_if_withdrawn(&self) -> Result<(), Error> {
        let Some(path) = self.path.as_ref().filter(|_| self.staged_birth.is_some()) else {
            return Ok(());
        };
        match self.file.staged_standing(path, self.staged_birth) {
            Standing::Withdrawn => Err(Error::StagingWithdrawn(path.to_string())),
            Standing::Live | Standing::Pending => Ok(()),
        }
    }

    /// Returns the root-relative path of the child `path` identifies, checked as the target of a
    /// staged edit, *without* taking the session lock.
    ///
    /// Paired with [`apply_staged`](Self::apply_staged): the checks run first so
    /// a read-only or sealed file is reported before any user closure runs, the
    /// closure then runs unlocked, and the lock is taken only to record what it
    /// built (issue #200).
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` spells no link, since a write addresses a
    /// link. Returns [`Error::ReadOnly`] if the
    /// file is read-only or
    /// this group has no path of its own, and the failures
    /// [`check_staged_writable`](FileInner::check_staged_writable) and
    /// [`refuse_if_withdrawn`](Self::refuse_if_withdrawn) report.
    fn child_edit_path(&self, path: &str) -> Result<ObjectPathBuf, Error> {
        self.file.check_staged_writable()?;
        self.refuse_if_withdrawn()?;
        let relative = ObjectPath::parse(path);
        if relative.components().is_empty() {
            // A write addresses a link, and a path that parses to no
            // components spells none. libhdf5 rejects the same call from
            // `H5L__delete_cb`, which reports a `"."` as a link it cannot
            // delete (`H5Lint.c`, HDF5 1.14.6).
            return Err(Error::EditUnsupported(NO_LINK_TO_WRITE));
        }
        self.child_path(&relative).ok_or(Error::ReadOnly)
    }

    /// Record already-built edits on the writable session, holding the lock only
    /// for the duration of the replay.
    ///
    /// The file is re-checked here because the closure that produced `ops` ran
    /// unlocked and could have closed the file in the meantime; staging into a
    /// sealed file would otherwise be silently accepted and then dropped.
    fn apply_staged(&self, ops: Vec<StagedOp>) -> Result<(), Error> {
        self.file.with_engine_mut(Change::Relocating, |session| {
            // All or nothing: one call can carry a whole subtree, and each op is
            // validated as it is staged, so a refusal partway must not leave the
            // ops before it recorded.
            session.stage_atomically(|s| {
                for op in ops {
                    op.apply(s)?;
                }
                Ok(())
            })
        })
    }

    /// Run `f` with the writable session and this group's *own* root-relative
    /// path (for attribute edits, which act on the group itself rather than a
    /// child). Returns [`Error::ReadOnly`] if the file is read-only or this
    /// group has no resolvable path, and [`Error::FileClosed`] once the file is
    /// sealed.
    fn with_own_session<R>(
        &self,
        f: impl FnOnce(&mut WriteEngine, &ObjectPathBuf) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.refuse_if_withdrawn()?;
        let path = self.path.clone().ok_or(Error::ReadOnly)?;
        self.file
            .with_engine_mut(Change::Relocating, |session| f(session, &path))
    }

    fn children(&self) -> Result<Vec<GroupEntry>, Error> {
        let hdr = self.file.parse_header(self.header_address()?)?;
        self.file.group_children(&hdr)
    }

    /// This group's on-disk children, paired with the children the session has
    /// staged under it.
    ///
    /// A staged name supersedes an on-disk link of the same name: the two can
    /// coexist only as a replacement (issue #305), where the commit removes the
    /// link before adding the new object, and every by-name lookup already
    /// answers with the staged one. A group that is *itself* staged has no links
    /// on disk to enumerate, and its members are exactly what is staged under
    /// it.
    fn children_and_staged(&self) -> Result<(Vec<GroupEntry>, StagedChildren), Error> {
        let staged = match self.path.as_ref() {
            Some(path) => self.file.staged_children(path),
            // A group reached by object reference cannot name itself, so nothing
            // can have been staged under it by name either.
            None => Vec::new(),
        };
        match self.children() {
            Ok(entries) => Ok((entries, staged)),
            Err(Error::NotCommitted(_)) => Ok((Vec::new(), staged)),
            Err(e) => Err(e),
        }
    }
}

/// The children of one group this session has staged.
type StagedChildren = Vec<StagedChild>;

/// Whether a staged creation takes `name` over from the link the file holds
/// there — which it does only when the same commit removes that link.
///
/// A creation that merely collides with a surviving link is refused where it is
/// staged, so the file's own object is what the name lists as.
fn superseded(staged: &[StagedChild], name: &str) -> bool {
    staged.iter().any(|c| c.name == name && c.replaces_link)
}

/// The staged children of `kind` that own their names, given the on-disk links
/// they sit beside, in the order they were staged.
fn staged_members(staged: &[StagedChild], kind: StagedKind, entries: &[GroupEntry]) -> Vec<String> {
    staged
        .iter()
        .filter(|c| c.kind == kind)
        // Either the file has no link of this name, or the commit removes it.
        .filter(|c| c.replaces_link || !entries.iter().any(|e| e.name == c.name))
        .map(|c| c.name.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Dataset handle
// ---------------------------------------------------------------------------

/// A [`Dataset`] handle's memo: where its object header sits, what that header
/// says, and what both hold as of.
///
/// The handle names the dataset; this is a memo of what that name resolved to.
/// See [`Dataset::resolved`].
struct DatasetState {
    /// Where the header sits — [`Resolution::address`] is base-adjusted and
    /// file-absolute — and the revisions the address and the parse hold as of.
    at: Resolution,
    header: ObjectHeader,
}

/// An owned handle to an HDF5 dataset.
///
/// The handle names the dataset by its root-relative path and remembers where
/// that path resolved to and what the header there said. A [`File::commit`]
/// rewrites and relocates object headers, and an immediate [`append`](Self::append)
/// rewrites one where it stands, so the memo is worked out again on the first
/// use after either and the handle goes on answering for the same dataset — see
/// [`File::commit`] for the two cases that report instead. That covers an edit
/// made through *another* handle to the same dataset as well as through this
/// one. Cloning gives a second handle to the same dataset, sharing its chunk
/// cache.
///
/// Three accessors cannot report a handle that no longer resolves, because they
/// return no `Result`: [`filters`](Self::filters) and
/// [`filter_pipeline`](Self::filter_pipeline) answer empty, the same answer an
/// unfiltered dataset gives, and [`is_chunked`](Self::is_chunked) answers
/// `false`. Every other reader of the header says what went wrong.
///
/// # Reading
///
/// A read returns a flat `Vec<T>` in row-major order. Each typed `read_*` method coerces the
/// stored bytes into its own element type, so a read whose element type differs from the stored
/// datatype can lose precision.
///
/// | Method | Returns |
/// |---|---|
/// | [`read_f64`](Self::read_f64) / [`read_f32`](Self::read_f32) | `Vec<f64>` / `Vec<f32>` |
/// | [`read_i8`](Self::read_i8) / [`read_i16`](Self::read_i16) / [`read_i32`](Self::read_i32) / [`read_i64`](Self::read_i64) | signed-integer vectors |
/// | [`read_u8`](Self::read_u8) / [`read_u16`](Self::read_u16) / [`read_u32`](Self::read_u32) / [`read_u64`](Self::read_u64) | unsigned-integer vectors |
/// | [`read::<T>()`](Self::read) (generic over [`H5Element`]) | `Vec<T>` |
/// | [`read_compound::<T>()`](Self::read_compound) (over [`CompoundType`]) | `Vec<T>` |
/// | [`read_string`](Self::read_string) | `Vec<String>` (fixed- and variable-length string datasets) |
/// | [`read_raw`](Self::read_raw) | `Vec<u8>` (complete unfiltered record bytes) |
/// | [`read_array`](Self::read_array) (`ndarray` feature) | `Array<T, D>` (static rank `D`) |
/// | [`read_array_dyn`](Self::read_array_dyn) (`ndarray` feature) | `ArrayD<T>` (runtime rank) |
///
/// ```rust
/// # let dir = tempfile::tempdir()?;
/// # let path = dir.path().join("output.h5");
/// # let mut builder = hdf5_pure::FileBuilder::new();
/// # builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
/// # builder.write(&path)?;
/// use hdf5_pure::File;
///
/// let file = File::open(&path)?;
/// let ds = file.dataset("temperature")?;
/// let values = ds.read_f64()?;
/// # assert_eq!(values, vec![22.5, 23.1, 21.8]);
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
///
/// [`read::<T>()`](Self::read) requests delivery as `T`, which asserts nothing about the stored
/// datatype, so a lossless read picks the `T` the file stores.
/// [`read_compound::<T>()`](Self::read_compound) decodes each element using its exact on-disk
/// datatype, reading field offsets from the file. [`read_raw`](Self::read_raw) returns the
/// unfiltered record bytes verbatim, which pairs with [`datatype`](Self::datatype) for inspecting
/// or re-emitting an arbitrary type. [`read_array`](Self::read_array) and
/// [`read_array_dyn`](Self::read_array_dyn) require the `ndarray` feature, which the [ndarray
/// guide](crate::_guide::ndarray) covers.
///
/// Two further reads take variable-length strings beyond [`read_string`](Self::read_string):
/// [`read_vlen_strings(options)`](Self::read_vlen_strings) and the streaming
/// [`visit_vlen_strings(options, f)`](Self::visit_vlen_strings), bounded by
/// [`VlenStringReadOptions`], with [`vlen_string_payload_size()`](Self::vlen_string_payload_size)
/// reporting the payload size up front. See the
/// [variable-length strings guide](crate::_guide::vlen_strings).
///
/// # Inspecting a dataset's type
///
/// Two accessors describe an existing dataset's type:
///
/// - [`dtype`](Self::dtype) returns a simplified [`DType`] classification.
/// - [`datatype`](Self::datatype) returns the full low-level [`Datatype`], including exact compound
///   field offsets.
pub struct Dataset {
    file: Arc<FileInner>,
    /// Where this dataset's object header sits and what it says, as of the file
    /// revision they were read at. Re-taken on first use after the file changes;
    /// see [`Dataset::resolved`].
    ///
    /// `None` for a dataset this session has *staged* and not yet committed:
    /// there is no header to read. Such a handle answers the metadata questions
    /// from what was staged and resolves its path on every other use, until a
    /// commit gives it a header to memoize.
    state: RwLock<Option<Arc<DatasetState>>>,
    // Held per-dataset: the chunk index is keyed only by chunk coordinate, so
    // a file-level cache would alias chunk addresses across datasets. Shared
    // between clones, which are the same dataset: a chunk read through one is
    // warm for the other, and an edit through either drops it for both.
    chunk_cache: Arc<ChunkCache>,
    // The effective chunk-cache config for this dataset: the file-wide default
    // or a per-dataset DAPL override. Reported by `chunk_cache_config`.
    chunk_cache_config: ChunkCacheConfig,
    /// Root-relative path of this dataset, used to address it for write
    /// operations on a read-write file, and to find it again after an edit moved
    /// it. `None` for a dataset reached by object reference
    /// ([`Dataset::dereference`]), which has no resolvable path.
    path: Option<ObjectPathBuf>,
    /// The session's staged generation when this handle was made onto a staged
    /// creation, and `None` for a handle opened onto a dataset in the file.
    ///
    /// It is what keeps such a handle from being retargeted at a different
    /// object: see [`Standing`] and [`FileInner::staged_standing`].
    staged_birth: Option<u64>,
}

impl Clone for Dataset {
    /// The clone is a second handle to the same dataset, sharing its chunk cache
    /// and resolved as of the same revision.
    fn clone(&self) -> Self {
        Self {
            file: Arc::clone(&self.file),
            state: RwLock::new(
                self.state
                    .read()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone(),
            ),
            chunk_cache: Arc::clone(&self.chunk_cache),
            chunk_cache_config: self.chunk_cache_config,
            path: self.path.clone(),
            staged_birth: self.staged_birth,
        }
    }
}

impl std::fmt::Debug for Dataset {
    /// Reports the memo as it stands, without re-resolving: a `Debug` that read
    /// the file could fail, and one that failed would have nothing to print.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read().unwrap_or_else(PoisonError::into_inner);
        match state.as_ref() {
            Some(state) => f
                .debug_struct("Dataset")
                .field("messages", &state.header.messages.len())
                .finish(),
            None => f.debug_struct("Dataset").field("staged", &true).finish(),
        }
    }
}

/// How many stored bytes a typed whole-dataset read holds beside its output,
/// before rounding the window up to whole chunk bands.
///
/// The decoded values are the caller's and there is no bound to put on them;
/// what this bounds is the *stored* copy standing next to them, which used to be
/// the whole dataset over again (issue #289). A mebibyte is small against any
/// dataset large enough for that to matter, and large enough that a sweep costs
/// reads in the tens rather than the thousands. A dataset that fits inside one
/// window is read whole, exactly as before.
const TYPED_READ_WINDOW_BYTES: u64 = 1 << 20;

/// How many leading-dimension rows a typed whole-dataset read decodes at a time.
///
/// [`TYPED_READ_WINDOW_BYTES`] of stored bytes, rounded *down* to whole chunk
/// bands so that no chunk is ever decoded for two windows, and never fewer than
/// one row — or one band, when a single band is already over budget, since a
/// window narrower than that would decode the same chunks again.
///
/// A dataset whose rows have no bytes (a zero inner dimension) has no elements
/// to read at all: it answers `NonZeroU64::MAX`, which the caller reads as "one
/// window covers it".
///
/// The answer is a `NonZeroU64` because the sweep advances by it: a window of no
/// rows would leave that loop running forever rather than returning something
/// wrong, and a test cannot report the difference.
fn typed_window_rows(
    dl: &DataLayout,
    ds: &Dataspace,
    elem_size: NonZeroUsize,
) -> Result<NonZeroU64, FormatError> {
    let mut row_bytes = elem_size.get() as u64;
    for &d in ds.dimensions.iter().skip(1) {
        row_bytes = row_bytes
            .checked_mul(d)
            .ok_or(FormatError::OffsetOverflow {
                offset: row_bytes,
                length: d,
            })?;
    }
    if row_bytes == 0 {
        return Ok(NonZeroU64::MAX);
    }

    let mut rows = (TYPED_READ_WINDOW_BYTES / row_bytes).max(1);
    if let DataLayout::Chunked {
        chunk_dimensions, ..
    } = dl
    {
        // A chunked layout message carries rank + 1 dimensions, the last being
        // the element size, so the first is the leading dimension's chunk extent
        // for every layout version this crate parses.
        if let Some(band) = chunk_dimensions.first().copied().filter(|&d| d > 0) {
            rows = if rows < band {
                band
            } else {
                rows - rows % band
            };
        }
    }
    // At least one row always, and at least one whole band when a band applied:
    // that arm runs only when `rows >= band`, so the remainder it subtracts
    // leaves a band standing.
    Ok(NonZeroU64::new(rows).unwrap_or(NonZeroU64::MIN))
}

impl Dataset {
    /// Build a handle for a dataset whose header was found, and read, at `at`.
    fn new(
        file: Arc<FileInner>,
        at: Resolution,
        header: ObjectHeader,
        chunk_cache_config: ChunkCacheConfig,
        path: Option<ObjectPathBuf>,
    ) -> Self {
        Self {
            file,
            state: RwLock::new(Some(Arc::new(DatasetState { at, header }))),
            chunk_cache: Arc::new(ChunkCache::with_config(chunk_cache_config)),
            chunk_cache_config,
            path,
            staged_birth: None,
        }
    }

    /// Build a handle for a dataset this session has staged and not yet
    /// committed, which has no object header to read.
    ///
    /// It answers the questions the staged builder already settles — shape,
    /// maximum shape, datatype, whether the storage is chunked and which filters
    /// it carries — and stages further edits on the pending creation. Everything
    /// that needs bytes reports [`Error::NotCommitted`] until [`File::commit`],
    /// after which the first use resolves the path and the handle behaves as if
    /// it had been opened by name.
    fn pending(
        file: Arc<FileInner>,
        chunk_cache_config: ChunkCacheConfig,
        path: ObjectPathBuf,
    ) -> Self {
        Self {
            staged_birth: file.staged_generation(),
            file,
            state: RwLock::new(None),
            chunk_cache: Arc::new(ChunkCache::with_config(chunk_cache_config)),
            chunk_cache_config,
            path: Some(path),
        }
    }

    /// What this dataset's staged creation says about itself, `Ok(None)` once it
    /// is committed (or when it never was staged), and
    /// [`Error::StagingWithdrawn`] when the creation this handle was made onto
    /// has been withdrawn — the one answer that is neither the staged record nor
    /// the file, because the object at that path is one the session is deleting.
    ///
    /// Only a handle with no memo can be pending: one that has resolved a header
    /// names an object the file already holds, so this costs a lock read rather
    /// than a session lock on the common path.
    fn staged_meta(&self) -> Result<Option<StagedMeta>, Error> {
        if self
            .state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
        {
            return Ok(None);
        }
        let Some(path) = self.path.as_ref() else {
            return Ok(None);
        };
        self.file.staged_dataset_view(path, self.staged_birth)
    }

    /// Refuse an operation that needs this dataset's bytes while it is still
    /// staged.
    fn refuse_if_pending(&self) -> Result<(), Error> {
        match self.staged_meta()? {
            // `staged_meta` answers only for a handle with a path.
            Some(_) => Err(Error::NotCommitted(reported_path(self.path.as_ref()))),
            None => Ok(()),
        }
    }

    /// This dataset's address and parsed header, worked out again if the file
    /// has changed since the memo was taken.
    ///
    /// Returns [`Error::StaleHandle`] for a handle
    /// that has no path to re-resolve — one an object reference produced — once
    /// a commit has run under it, and the resolution's own error (a
    /// `PathNotFound`, say, for a dataset a commit deleted) when the path no
    /// longer names anything. A dataset this session has staged and not
    /// committed has no header at all, and that is
    /// [`Error::NotCommitted`]. A path that now names something other than a
    /// dataset is [`Error::NotADataset`], the error opening it afresh reports.
    fn resolved(&self) -> Result<Arc<DatasetState>, Error> {
        let live = self.file.content_revision();
        let memo = {
            let state = self.state.read().unwrap_or_else(PoisonError::into_inner);
            match state.as_ref() {
                Some(state) if state.at.content_revision == live => {
                    return Ok(Arc::clone(state));
                }
                Some(state) => Some(state.at),
                None => None,
            }
        };
        let at = self
            .file
            .locate_staged(self.path.as_ref(), memo, self.staged_birth)?;
        let header = self.file.parse_header(at.address)?;
        // Checked *before* it is memoized. A header installed and then refused is
        // the answer every later call short-circuits on, so this handle would
        // report `NotADataset` once and then serve the other object's header —
        // which a commit replacing a dataset with a group at the same path
        // (issue #305) makes reachable.
        if !has_message(&header, MessageType::DATA_LAYOUT) {
            // Only a path can reach this: an address memo that survived is the
            // address of the dataset this handle already read there.
            return Err(Error::NotADataset(reported_path(self.path.as_ref())));
        }
        Ok(self.install(at, header))
    }

    /// Memoize `header`, read at `at`, as this handle's resolution.
    ///
    /// Drops the chunk cache: an edit that rewrote this header can have moved
    /// the chunk index and the chunks it names, so what the cache holds belongs
    /// to the copy the edit replaced.
    fn install(&self, at: Resolution, header: ObjectHeader) -> Arc<DatasetState> {
        let fresh = Arc::new(DatasetState { at, header });
        self.chunk_cache.clear();
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        // Two threads can re-resolve at once. The older answer must not land on
        // top of the newer one, or the newer handle would go on serving a header
        // the file has already moved past. A handle that had no memo takes this
        // one: any header beats naming nothing.
        if state
            .as_ref()
            .is_none_or(|memo| at.content_revision >= memo.at.content_revision)
        {
            *state = Some(Arc::clone(&fresh));
        }
        fresh
    }

    /// Returns the address of this dataset's object header, absolute in the file.
    #[cfg(test)]
    pub(crate) fn header_address(&self) -> Result<u64, Error> {
        Ok(self.resolved()?.at.address)
    }

    /// Appends `data` in place along the first, unlimited dimension.
    ///
    /// Every handle onto the dataset reads the new length afterwards.
    ///
    /// Supports numeric scalars, fixed byte arrays, and [`FixedAscii`](crate::FixedAscii)
    /// strings through [`H5Element`]. Numeric datatypes must match exactly. Array dimensions
    /// and base types must match, with byte order ignored for one-byte fixed-point leaves.
    /// Fixed ASCII strings require null-padded ASCII storage of the same width. The writer
    /// preserves the stored datatype and checks compatibility even for empty input.
    ///
    /// The file must have been opened for writing with [`File::open_rw`];
    /// a read-only file returns
    /// [`Error::ReadOnly`]. The target must be a chunked,
    /// rank-1, unlimited, Extensible-Array-indexed dataset: anything else returns
    /// [`Error::AppendInPlaceUnsupported`]. A missing path returns
    /// [`Error::Format`] with [`FormatError::PathNotFound`], and a header message
    /// that does not read or parse returns [`Error::Format`] with the failure the
    /// read or the parse produced.
    ///
    /// An eligible empty dataset may have an undefined index-header address. The first nonempty
    /// write initializes the index with the layout's creation parameters. The writer publishes
    /// ownership and EOF for the complete header and index block before linking the header from
    /// the layout message. Empty appends leave index and element storage unallocated.
    ///
    /// For unfiltered storage and lossless filters, the current and appended lengths may be
    /// unaligned. The writer decodes a partial tail, extends it, and places the replacement
    /// separately from its predecessor. It covers the payload with the superblock's recorded EOF
    /// before publishing the new index record. After confirming the record and its ordering
    /// barrier, the editor retires the predecessor's complete stored allocation. Reuse follows
    /// the file-space strategy, threshold, and PAGE class rules.
    /// Under [`FileSpaceStrategy::Aggr`] and [`FileSpaceStrategy::None`], the editor discards freed
    /// extents.
    ///
    /// A partial tail stored under lossy filters (ZFP or float D-scale scale-offset) cannot
    /// grow through this call, because re-encoding it could change committed values. Use
    /// [`append_with_options`](Self::append_with_options) to permit such a rewrite. Under
    /// [`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), partial tails stay
    /// unfiltered and can grow. With a lossy pipeline, the writer filters a tail when it
    /// becomes complete, after publishing the new extent. The editor retires the original
    /// allocation after confirming its raw replacement, and the intermediate raw allocation after
    /// confirming the final filtered record. An interruption between those updates can leave the
    /// completed chunk unfiltered.
    ///
    /// A trailing chunk absent from the chunk index cannot be read to be grown. Such a dataset
    /// requires a chunk-aligned starting length or [`append_staged`](Self::append_staged).
    ///
    /// With ordinary write buffering, the writer preserves publication order under either
    /// [`SyncPolicy`]. Under the default [`SyncPolicy::Always`], the writer forces writes to
    /// durable storage at each barrier. Under [`SyncPolicy::OnClose`], the writer defers power-loss
    /// durability until [`File::sync`] or clean session closure. The writer also drains explicit
    /// page buffers at allocation and reference publication boundaries. These ordering guarantees
    /// do not cover torn physical metadata writes.
    ///
    /// A **SWMR** writer ([`File::open_swmr_writer`]) keeps the narrower rule it
    /// always had — unfiltered, and chunk-aligned at both ends — because its
    /// readers are concurrent by contract and a rewritten trailing chunk is one
    /// they could be crossing. It allocates at end-of-file for that same reason:
    /// a region this session freed is one a reader may still be inside.
    ///
    /// The editor tries reusable free space before extending the file. In a persistent session,
    /// it writes provisional content before publishing replacement free-space managers that
    /// exclude the complete allocations. It completes ownership publication before linking
    /// references. Unissued intervals remain recorded under the configured threshold policy,
    /// including reusable PAGE tails. An interruption after ownership publication can leave
    /// actual allocations unreferenced.
    ///
    /// Eligible retired storage is reusable during the session. For persistent files, the editor
    /// publishes remaining free-space manager changes during [`File::commit`] or clean session
    /// closure, even when EOF is unchanged. Call [`File::commit`] before [`File::sync`] to include
    /// pending manager updates in an explicit synchronization. The editor retains confirmed
    /// retirements if a later phase fails. After uncertain publication, the editor withholds
    /// possibly referenced allocations and may defer manager cleanup.
    ///
    /// A handle reached by object reference ([`dereference`](Self::dereference))
    /// has no resolvable path, so it names its dataset by the object-header
    /// address it was reached through and can append like any other — until the
    /// session stages or commits an edit. A commit can move that header, and the
    /// bytes it vacates still parse as the dataset they were, so an append
    /// against the old address would land in a header nothing points at. Rather
    /// than do that silently, such an append is refused once edits are staged or
    /// a commit has run; re-open the dataset by path to keep appending. A
    /// path-named handle is unaffected, because the path is resolved afresh every
    /// time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only, or [`Error::AppendInPlaceUnsupported`]
    /// if the dataset or input is unsupported or conflicts with staged edits. Returns
    /// [`Error::SwmrAppendUnsupported`] for filtered or unaligned SWMR appends. Propagates
    /// metadata, allocation, encoding, and I/O errors.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    ///
    /// # Performance
    ///
    /// For byte-array input, the writer borrows the flattened slice and allocates buffers for
    /// chunk encoding and storage. Bounded sessions limit the full-chunk portion of each
    /// batch, allowing at least one chunk and space to complete an existing partial tail.
    /// ASCII and numeric inputs are encoded into a temporary buffer per batch.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, FixedAscii, MaxExtent};
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("fixed.h5");
    /// let mut builder = FileBuilder::new();
    /// builder.create_dataset("bytes")
    ///     .with_data(&[[0u8, 255]])
    ///     .with_maxshape(&[MaxExtent::Unlimited])
    ///     .with_chunks(&[4]);
    /// builder.create_dataset("labels")
    ///     .with_data(&[FixedAscii::<4>::new("ab")?])
    ///     .with_maxshape(&[MaxExtent::Unlimited])
    ///     .with_chunks(&[4]);
    /// builder.write(&path)?;
    ///
    /// let file = File::open_rw(&path)?;
    /// file.dataset("bytes")?.append(&[[128u8, 42]])?;
    /// file.dataset("labels")?.append(&[FixedAscii::<4>::new("xyz")?])?;
    /// assert_eq!(file.dataset("bytes")?.read::<[u8; 2]>()?, [[0, 255], [128, 42]]);
    /// let labels = file.dataset("labels")?.read::<FixedAscii<4>>()?;
    /// assert_eq!(labels, [FixedAscii::new("ab")?, FixedAscii::new("xyz")?]);
    /// drop(file);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn append<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        self.append_with_options(AppendOptions::new(), data)
    }

    /// Appends `data` in place using the policy in `options`.
    ///
    /// The file, dataset, and datatype requirements match [`append`](Self::append).
    /// [`LossyTailReencoding::Allow`](crate::LossyTailReencoding::Allow) permits growing a
    /// partial tail stored under lossy filters, with the additional quantization error
    /// described by [`LossyTailReencoding`](crate::LossyTailReencoding).
    ///
    /// During such a rewrite, the writer publishes the replacement chunk's index record
    /// before the new extent. An interrupted append can therefore leave re-quantized tail
    /// values visible at the old length. Both extents remain readable, but the old values
    /// are not preserved atomically with the length. Other append restrictions apply,
    /// including the rejection of filtered SWMR appends.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only, or
    /// [`Error::AppendInPlaceUnsupported`] if the dataset or input is unsupported, lossy-tail
    /// re-encoding is forbidden, or the append conflicts with staged edits. Returns
    /// [`Error::SwmrAppendUnsupported`] for filtered or unaligned SWMR appends. Propagates
    /// metadata, allocation, encoding, and I/O errors.
    pub fn append_with_options<T: H5Element>(
        &mut self,
        options: AppendOptions,
        data: &[T],
    ) -> Result<(), Error> {
        let g = self.append_geometry()?;
        if let Some(encoded) = T::borrowed_encoding(data) {
            let (bytes, datatype) = encoded?;
            let target = self.append_target()?;
            return self.file.with_engine_mut(Change::InPlace, |engine| {
                engine.append_inplace_bytes(target, options, Some(&datatype), bytes, 4)
            });
        }
        self.append_batches(g, options, data.len() as u64, |b, r| {
            b.append(&data[r]);
        })
    }

    /// Appends raw little-endian element bytes in place.
    ///
    /// Uses [`AppendOptions::new`]. Prefer [`append`](Self::append) when the element type is
    /// known. See [`append_raw_with_options`](Self::append_raw_with_options) for the byte
    /// requirements and errors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only. Other errors are described by
    /// [`append_raw_with_options`](Self::append_raw_with_options).
    pub fn append_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.append_raw_with_options(AppendOptions::new(), bytes)
    }

    /// Appends raw little-endian element bytes using the policy in `options`.
    ///
    /// The byte length must be a multiple of the dataset's element size. Numeric fields must
    /// be little-endian, except one-byte integers, whose byte order is immaterial.
    /// Variable-length and reference datatypes are unsupported. The file requirements,
    /// lossy-tail policy, and interruption behavior match
    /// [`append_with_options`](Self::append_with_options).
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only, or
    /// [`Error::AppendInPlaceUnsupported`] if the bytes, datatype, or dataset are unsupported,
    /// lossy-tail re-encoding is forbidden, or the append conflicts with staged edits. Returns
    /// [`Error::SwmrAppendUnsupported`] for filtered or unaligned SWMR appends. Propagates
    /// metadata, allocation, encoding, and I/O errors.
    pub fn append_raw_with_options(
        &mut self,
        options: AppendOptions,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let g = self.append_geometry()?;
        let es = g.element_size;
        // Whole-element length is checked before any batch applies, so the
        // refusal is atomic (the per-batch validation would only reject the
        // final, short batch after earlier ones had durably committed).
        if bytes.len() % es != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "appended byte length is not a whole number of elements",
            ));
        }
        let total = (bytes.len() / es) as u64;
        self.append_batches(g, options, total, |b, r| {
            b.append_raw(&bytes[r.start * es.get()..r.end * es.get()]);
        })
    }

    /// How an append names this dataset to the session: by path when the handle
    /// has one, so the session can check the target against its own staged
    /// edits, and otherwise by the object-header address the handle was reached
    /// through — which is what lets a handle obtained by object reference append
    /// at all.
    fn append_target(&self) -> Result<AppendTarget<'_>, Error> {
        Ok(match &self.path {
            Some(path) => AppendTarget::Path(path),
            None => AppendTarget::Header(self.resolved()?.at.address),
        })
    }

    /// Creates a buffered appender with [`AppendOptions::new`].
    ///
    /// Elements stay in memory until an automatic write or an explicit flush. The default
    /// rejects an existing filtered lossy partial tail and partial flushes that would leave
    /// one. [`buffered_appender_with_options`](Self::buffered_appender_with_options) permits
    /// explicitly allowing lossy-tail re-encoding.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only. Other errors are described by
    /// [`buffered_appender_with_options`](Self::buffered_appender_with_options).
    pub fn buffered_appender(&mut self) -> Result<BufferedAppender<'_>, Error> {
        self.buffered_appender_with_options(AppendOptions::new())
    }

    /// Creates a buffered appender whose writes use `options` for its lifetime.
    ///
    /// The file and dataset requirements match [`append`](Self::append), and SWMR sessions
    /// are rejected. [`LossyTailReencoding::Allow`](crate::LossyTailReencoding::Allow) permits
    /// growing filtered lossy partial tails and repeated partial flushes. Rewrites can add
    /// quantization error to previously published tail values, with the interruption behavior
    /// described by [`append_with_options`](Self::append_with_options).
    ///
    /// The policy applies to automatic writes, [`flush`](BufferedAppender::flush),
    /// [`finish`](BufferedAppender::finish), and drop. It is not stored in the file or applied
    /// to other append calls, including calls made after reopening the file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only, [`Error::SwmrAppendUnsupported`]
    /// if it is a SWMR session, or [`Error::AppendInPlaceUnsupported`] if the file or dataset
    /// is unsupported or a filtered lossy partial tail requires re-encoding without permission.
    /// Returns [`Error::EditUnsupported`] if another appender conflicts with this one.
    /// Propagates file-state and metadata errors.
    pub fn buffered_appender_with_options(
        &mut self,
        options: AppendOptions,
    ) -> Result<BufferedAppender<'_>, Error> {
        BufferedAppender::new(self, options)
    }

    /// Consumes the dataset and buffers appends with [`AppendOptions::new`].
    ///
    /// The owned dataset retains the editing session after the original [`File`] is dropped.
    /// The default policy rejects an existing filtered lossy partial tail and partial flushes
    /// that would leave one. See [`OwnedBufferedAppender`] for buffering and recovery.
    ///
    /// # Errors
    ///
    /// Returns the errors described by
    /// [`into_buffered_appender_with_options`](Self::into_buffered_appender_with_options).
    pub fn into_buffered_appender(self) -> Result<OwnedBufferedAppender, Error> {
        self.into_buffered_appender_with_options(AppendOptions::new())
    }

    /// Consumes the dataset and buffers appends under a captured policy.
    ///
    /// The file and dataset requirements match
    /// [`buffered_appender_with_options`](Self::buffered_appender_with_options).
    /// The supplied [`AppendOptions`] apply to
    /// automatic writes, [`flush`](OwnedBufferedAppender::flush),
    /// [`finish`](OwnedBufferedAppender::finish), and drop. They are not stored in the file
    /// or applied to other append calls. The owned dataset retains the editing session after
    /// the original [`File`] is dropped. Explicitly closing the file closes that session.
    ///
    /// [`LossyTailReencoding::Allow`](crate::LossyTailReencoding::Allow) permits growing
    /// filtered lossy partial tails and repeated partial flushes, with the quantization and
    /// interruption behavior described by [`append_with_options`](Self::append_with_options).
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only, [`Error::SwmrAppendUnsupported`]
    /// if it is a SWMR session, or [`Error::AppendInPlaceUnsupported`] if the file or dataset
    /// is unsupported or a filtered lossy partial tail requires re-encoding without permission.
    /// Returns [`Error::EditUnsupported`] if another appender conflicts with this one.
    /// Propagates file-state and metadata errors. The dataset is consumed even on error.
    pub fn into_buffered_appender_with_options(
        self,
        options: AppendOptions,
    ) -> Result<OwnedBufferedAppender, Error> {
        OwnedBufferedAppender::new(self, options)
    }

    /// Applies the session's live mutation gate to appender operations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the appender's session is closed, or [`Error::ReadOnly`]
    /// if the file has no editing session. Returns [`Error::AllocationPublicationUncertain`] if
    /// an earlier editing outcome is uncertain, or [`FormatError::Internal`] if an allocation
    /// batch remains provisional.
    pub(crate) fn check_appender_mutable(&self) -> Result<(), Error> {
        self.file.check_mutable(false)?;
        self.file
            .with_engine_mut(Change::Nothing, |engine| engine.check_append_publication())
    }

    /// Register a live `BufferedAppender` on this dataset with the session, so a
    /// staged edit that would stop it from flushing is refused at the call that
    /// creates the conflict rather than in the appender's `Drop`.
    pub(crate) fn claim_for_appender(&self) -> Result<u64, Error> {
        let Backend::Edit(m) = &self.file.backend else {
            return Err(Error::ReadOnly);
        };
        m.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .claim_for_appender(self.path.as_ref())
    }

    /// Release the claim taken by [`claim_for_appender`](Self::claim_for_appender).
    pub(crate) fn release_appender_claim(&self, token: u64) {
        if let Backend::Edit(m) = &self.file.backend {
            m.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .release_appender_claim(token);
        }
    }

    /// Whether this dataset's session is the SWMR writer, whose append rules are
    /// a strict subset of the ordinary ones. `false` for a read-only file, which
    /// has no session to ask.
    pub(crate) fn session_is_swmr(&self) -> bool {
        match &self.file.backend {
            Backend::Edit(m) => m
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_swmr(),
            _ => false,
        }
    }

    /// Immediate in-place append of an already-gathered builder, used by
    /// [`BufferedAppender`], whose bytes are materialized in its buffer before
    /// it decides how many of them to write. `append_batches` exists for the
    /// opposite case — a caller whose bytes are cheaper to build per batch — so
    /// this hands the engine one builder and lets it batch the plan.
    pub(crate) fn append_prebuilt(&mut self, b: &AppendBuilder) -> Result<(), Error> {
        let target = self.append_target()?;
        self.file.with_engine_mut(Change::InPlace, |engine| {
            engine.append_inplace_gathered(target, b, 4)
        })
    }

    /// Fetch (locating on first use) this dataset's append geometry from the
    /// write session, which also applies every refusal that does not depend on
    /// the bytes being appended.
    pub(crate) fn append_geometry(&self) -> Result<AppendGeometry, Error> {
        // An in-place append rewrites bytes where they stand, and a dataset this
        // session has only staged has none. Refused here rather than deep in the
        // engine, so `append`, `append_raw` and `buffered_appender` all say the
        // same thing.
        self.refuse_if_pending()?;
        let target = self.append_target()?;
        self.file
            .with_engine_mut(Change::Nothing, |engine| engine.append_geometry(target))
    }

    /// Immediate in-place append, driven batch by batch. The call is split into
    /// aligned batches — the trailing partial chunk is filled first, then
    /// whole-chunk batches under the session's byte budget — and `fill` builds
    /// each batch's bytes on demand, so a bounded session's peak memory holds
    /// one batch rather than the whole call. A session that keeps the whole file
    /// resident uses one unbounded batch.
    ///
    /// Every predictable refusal (wrong datatype, ineligible dataset) is raised
    /// before the first batch is applied. The cached header and chunk cache are
    /// then refreshed so later reads on this handle observe the new length. A
    /// filtered dataset sitting on a partial trailing chunk is grown by the
    /// first batch, which re-encodes that chunk into a fresh allocation and
    /// leaves every later batch starting on a boundary.
    fn append_batches(
        &mut self,
        AppendGeometry {
            chunk_elems,
            element_size: _,
            current_dim: mut dim,
            lossy_filters: _,
            full_batch_elems,
        }: AppendGeometry,
        options: AppendOptions,
        total_elems: u64,
        fill: impl Fn(&mut AppendBuilder, std::ops::Range<usize>),
    ) -> Result<(), Error> {
        // Worked out once for the whole call: every batch names the same
        // dataset, and an in-place append does not move it.
        let target = self.append_target()?;
        let mut done = 0u64;
        loop {
            // An empty append still runs one (empty) engine call, so datatype
            // validation happens whether or not there are elements.
            let to_boundary = (chunk_elems - dim % chunk_elems) % chunk_elems;
            let take = (total_elems - done).min(to_boundary.saturating_add(full_batch_elems));
            let mut b = AppendBuilder::with_options(options);
            fill(&mut b, done.to_usize()?..(done + take).to_usize()?);
            if let Some(error) = b.take_error() {
                return Err(error.into());
            }
            self.file.with_engine_mut(Change::InPlace, |engine| {
                engine.append_inplace_gathered(target, &b, 4)
            })?;
            dim += take;
            done += take;
            if done >= total_elems {
                break;
            }
        }
        Ok(())
    }

    /// Overwrite this dataset's values, staged until [`File::commit`]. The new
    /// data must match the dataset's existing shape and datatype.
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`]. Unlike [`append`](Self::append) (immediate), this is
    /// a staged edit applied on [`File::commit`].
    pub fn write<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        // Build off the lock, as `write_staged` does: `write_into` is trait
        // code reached with no lock held, keeping both paths identical.
        self.check_staged_edit()?;
        let mut builder = DatasetBuilder::new("");
        T::write_into(&mut builder, data);
        self.with_session_mut(|session, path| session.stage_dataset_write(path, builder))
    }

    /// Overwrite this dataset's values through its full [`DatasetBuilder`],
    /// staged until [`File::commit`] — the builder-level counterpart of
    /// [`write`](Self::write), and the only one of the two that can carry a
    /// **shape**. [`write`](Self::write) sends a flat `&[T]`, so it can overwrite
    /// a one-dimensional dataset only; a multi-dimensional one needs
    /// [`with_shape`](DatasetBuilder::with_shape) and so comes through here, as
    /// do compound, complex, and raw bytes under an explicit datatype.
    ///
    /// The replacement shape and datatype must match the on-disk dataset, except
    /// that byte order is ignored for one-byte fixed-point types, including nested
    /// base and member types. [`File::commit`] returns [`Error::EditUnsupported`]
    /// if the datatype is incompatible or the shape differs.
    ///
    /// This overwrites element bytes and nothing else, so a builder asking for
    /// more than that is refused by **this call**, before anything is staged:
    /// chunking, filters, an extensible shape,
    /// a [`PartialEdgeFiltering`](crate::PartialEdgeFiltering) policy, an attribute, a fill
    /// value, and [`with_path_references`](DatasetBuilder::with_path_references), whose
    /// element bytes are placeholder addresses only a newly created dataset can
    /// resolve. Set those when the dataset is created.
    ///
    /// [`with_vlen_strings`](DatasetBuilder::with_vlen_strings) is **not**
    /// refused: overwriting a variable-length-string dataset places a fresh
    /// global heap collection for the new strings and resolves the staged
    /// element references against it. Overwriting the same dataset again
    /// reclaims the collection the previous overwrite placed, so rotating its
    /// strings in a session does not grow the file without bound. An overwrite
    /// does not reclaim the collections the dataset held when the session
    /// *opened*: a collection can be shared between objects, and only this
    /// session's own placements are known not to be, so
    /// [`repack`](crate::repack()) is what recovers those.
    ///
    /// The reference refusal is on the builder, not on the datatype it produces:
    /// such a dataset can still be overwritten by supplying element bytes that
    /// need no resolving, with
    /// [`with_reference_data`](DatasetBuilder::with_reference_data) or
    /// [`with_raw_data`](DatasetBuilder::with_raw_data). An object reference
    /// supplied that way is screened at `commit` by *address*, against both what
    /// the same commit deletes and what it rewrites elsewhere, so it cannot be
    /// left naming storage the commit is vacating — the answer a target named as
    /// a path already got by name.
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`].
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("counters.h5")?;
    /// let mut ds = file.dataset("ticks")?;
    /// // Keep the dataset's own datatype and supply the replacement bytes it
    /// // describes — three little-endian 16-bit elements here.
    /// let dt = ds.datatype()?;
    /// ds.write_staged(|b| {
    ///     b.with_raw_data(dt, vec![1, 0, 2, 0, 3, 0], 3);
    /// })?;
    /// file.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    /// The closure configures a standalone builder, not the file, so it may read
    /// the same [`File`]; nothing it stages resolves until [`File::commit`].
    pub fn write_staged(&mut self, build: impl FnOnce(&mut DatasetBuilder)) -> Result<(), Error> {
        // Report a read-only, sealed, or unaddressable dataset before running the
        // closure, then run it with no lock held; `stage_dataset_write` names the
        // builder from the dataset's path (issue #200).
        self.check_staged_edit()?;
        let mut builder = DatasetBuilder::new("");
        build(&mut builder);
        self.with_session_mut(|session, path| session.stage_dataset_write(path, builder))
    }

    /// Stages an append to this dataset for publication by [`File::commit`].
    ///
    /// Unlike [`append`](Self::append) (immediate, amortized `O(1)`,
    /// Extensible-Array only), this rebuilds the chunk index on commit and so
    /// also grows datasets whose Extensible-Array index is not yet allocated.
    /// An append that grows a partial tail stored under lossy filters (ZFP or float D-scale
    /// scale-offset) is rejected before any changes are staged. Use
    /// [`append_staged_with_options`](Self::append_staged_with_options) to permit it. Under
    /// [`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), partial tails stay
    /// unfiltered and can grow. With a lossy pipeline, the writer filters a tail when complete.
    /// Configure the appended elements through `build` on the
    /// [`AppendBuilder`]; repeated calls within the builder concatenate in
    /// order. The dataset must be chunked, unlimited along axis 0,
    /// Extensible-Array indexed, rank 1, use a re-encodable filter pipeline, and
    /// have a single hard link, otherwise [`Error::AppendUnsupported`] is
    /// returned on [`File::commit`].
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`].
    /// The closure configures a standalone builder, not the file, so it may read
    /// the same [`File`]; nothing it stages resolves until [`File::commit`].
    ///
    /// The editor retires a replaced tail's complete stored allocation through root publication
    /// at [`File::commit`]. Reuse follows the file-space strategy and retention threshold.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only, or [`Error::AppendUnsupported`]
    /// if growing the partial tail would re-encode lossy values. Propagates errors from
    /// resolving the dataset or checking for conflicting staged edits.
    pub fn append_staged(&mut self, build: impl FnOnce(&mut AppendBuilder)) -> Result<(), Error> {
        self.append_staged_with_options(AppendOptions::new(), build)
    }

    /// Stages an append with the policy in `options` for publication by [`File::commit`].
    ///
    /// The dataset requirements and builder semantics match
    /// [`append_staged`](Self::append_staged). The commit uses the supplied policy, including
    /// permission to re-encode a partial tail stored under lossy filters. Such a rewrite can
    /// change previously published values as described by
    /// [`LossyTailReencoding`](crate::LossyTailReencoding). Complete chunks preceding the
    /// tail retain their storage. The supplied policy does not affect later append calls.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only, or [`Error::AppendUnsupported`]
    /// if growing the partial tail requires lossy re-encoding that `options` rejects.
    /// Propagates errors from resolving the dataset or checking for conflicting staged edits.
    /// [`File::commit`] reports unsupported dataset geometry, datatypes, or filter pipelines.
    pub fn append_staged_with_options(
        &mut self,
        options: AppendOptions,
        build: impl FnOnce(&mut AppendBuilder),
    ) -> Result<(), Error> {
        // A dataset *this handle* names and this session has not written has no
        // bytes to grow, so the elements are folded into the pending creation
        // instead (see the note above). `check_staged_edit` refuses that dataset,
        // so ask it only about one the file already holds; the file-mode and path
        // gates apply either way.
        //
        // The distinction is the handle's, not the path's: a handle onto the
        // object a staged creation *replaces* is not pending, and the session
        // refuses its append rather than growing the replacement under it.
        let pending = self.staged_meta()?.is_some();
        if pending {
            self.file.check_staged_writable()?;
        } else {
            self.check_staged_edit()?;
        }
        let mut builder = AppendBuilder::with_options(options);
        build(&mut builder);
        if let Some(error) = builder.take_error() {
            return Err(error.into());
        }
        self.with_session_mut(|session, path| {
            if pending {
                session.stage_dataset_append_pending(path, builder)
            } else {
                session.stage_dataset_append(path, builder)
            }
        })
    }

    /// Add or update an attribute on this dataset, staged until
    /// [`File::commit`]. Use [`remove_attr`](Self::remove_attr) to remove one.
    /// An attribute set too large for the object header is written to a fractal
    /// heap, exactly as [`Group::set_attr`] does.
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`].
    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> Result<(), Error> {
        self.refuse_if_pending()?;
        self.with_session_mut(|session, path| session.set_dataset_attr(path, name, value))
    }

    /// Remove an attribute from this dataset, staged until [`File::commit`].
    /// See [`set_attr`](Self::set_attr) for the file-mode rules.
    pub fn remove_attr(&mut self, name: &str) -> Result<(), Error> {
        self.refuse_if_pending()?;
        self.with_session_mut(|session, path| session.remove_dataset_attr(path, name))
    }

    /// Gate a staged edit on this dataset *without* taking the session lock:
    /// the file must accept staged edits, this handle must have a resolvable
    /// path, and this dataset's elements must be ones the engine owns.
    ///
    /// The path check belongs here rather than only in
    /// [`with_session_mut`](Self::with_session_mut) so that every reason to
    /// refuse is reported *before* a user closure runs, not after. A handle
    /// reached by object reference ([`dereference`](Self::dereference)) has no
    /// path, and would otherwise have its closure run and its result discarded.
    ///
    /// This is the one gate every data-writing entry point shares — `write`,
    /// `write_staged`, `append_staged`, and the staged rewrite behind
    /// [`BufferedAppender`] — which is why the external-storage refusal is here
    /// and not on any one of them. Attribute edits do not come through here, and
    /// are unaffected: they change the object header, not the elements.
    fn check_staged_edit(&self) -> Result<(), Error> {
        self.file.check_staged_writable()?;
        if self.path.is_none() {
            return Err(Error::ReadOnly);
        }
        // Ahead of the external-storage question, which reads a header this
        // dataset does not have yet. `append_staged` is the one staged edit that
        // *is* supported on a pending dataset, and it re-admits itself below.
        self.refuse_if_pending()?;
        // The elements of an externally stored dataset are not in this file, and
        // the engine writes only this file. Its contiguous layout message records
        // no address, so a write took it for never-allocated storage, appended the
        // new bytes and pointed the layout at them — leaving the file with two
        // contradictory records of where the data lives, and the reference
        // library still reading the external files it had not touched. See
        // [`has_external_storage`](Self::has_external_storage).
        if self.has_external_storage()? {
            return Err(Error::EditUnsupported(
                "this dataset's elements live in external files (H5Pset_external), which this \
engine does not write; writing them into the HDF5 file would leave it disagreeing with \
the external files about where the data is. Delete the dataset and create it again in \
the same commit to replace it",
            ));
        }
        Ok(())
    }

    /// Run `f` with the writable session and this dataset's path, then refresh
    /// the cached header so a later read on this handle reflects any immediate
    /// change (e.g. an append's new dimension). Returns [`Error::ReadOnly`] if
    /// the file is read-only or the handle has no resolvable path (reached by
    /// object reference).
    fn with_session_mut<R>(
        &mut self,
        f: impl FnOnce(&mut WriteEngine, &ObjectPathBuf) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let path = self.path.clone().ok_or(Error::ReadOnly)?;
        self.file
            .with_engine_mut(Change::Relocating, |session| f(session, &path))
    }

    /// The effective raw chunk-cache configuration for this dataset.
    ///
    /// This reflects the per-dataset [`DatasetAccessProperties`] override when one
    /// was supplied to [`File::dataset_with_options`] /
    /// [`Group::dataset_with_options`], otherwise the file-wide default. It is
    /// the read-side analogue of HDF5's `H5Pget_chunk_cache`.
    pub const fn chunk_cache_config(&self) -> ChunkCacheConfig {
        self.chunk_cache_config
    }

    /// A point-in-time snapshot of this dataset handle's chunk-cache occupancy.
    ///
    /// Lets callers confirm a chunk-cache configuration (set with
    /// [`FileAccessProperties::with_chunk_cache`]) is taking effect: after a
    /// chunked read, an enabled cache reports a loaded index and retained
    /// chunks; a disabled one (or one over its budget) reports fewer or none.
    /// The cache is per-handle — though clones of one handle share it — so a
    /// freshly opened [`Dataset`] reports an empty snapshot until its first read.
    pub fn chunk_cache_stats(&self) -> ChunkCacheStats {
        // Notice any edit first, which is what drops the chunks it invalidated.
        // A snapshot taken before that would count chunks no read will ever be
        // served, and this exists to say what the cache is holding *for* a read.
        // An unresolvable handle has no live cache to report, and an empty
        // snapshot is the truthful answer for one.
        let _ = self.resolved();
        self.chunk_cache.stats()
    }

    /// Zero this handle's cumulative chunk-cache counters, leaving the retained
    /// index and chunks in place.
    ///
    /// [`ChunkCacheStats`]'s occupancy figures are unaffected — this resets what
    /// the cache has *done*, not what it is *holding* — so a caller can measure
    /// one read on a cache an earlier read already warmed. The cache is
    /// per-handle, though clones of one handle share it, so this resets the
    /// counters those clones report too.
    pub fn reset_chunk_cache_stats(&self) {
        // Resolve first for the same reason [`Self::chunk_cache_stats`] does: an
        // edit drops the chunks it invalidated, and those belong in the
        // invalidation count of the window being reset, not the next one.
        let _ = self.resolved();
        self.chunk_cache.reset_stats();
    }

    /// Returns the shape (dimensions) of the dataset.
    ///
    /// A dataset this session has staged and not yet committed answers from what
    /// was staged, which includes the elements
    /// [`append_staged`](Self::append_staged) has folded into it.
    pub fn shape(&self) -> Result<Vec<u64>, Error> {
        if let Some(meta) = self.staged_meta()? {
            return Ok(meta.dimensions);
        }
        let ds = self.dataspace()?;
        Ok(ds.dimensions.clone())
    }

    /// Returns the dataset's maximum dimensions, when it is extensible.
    ///
    /// A dimension that grows without bound is [`MaxExtent::Unlimited`].
    /// Returns `Ok(None)` for a fixed-shape dataset: one with no
    /// maximum-dimensions record, and one whose record is the current shape.
    /// A dataset this session has staged and not yet committed reports what was
    /// staged.
    ///
    /// Together with [`is_chunked`](Self::is_chunked) and
    /// [`chunk_shape`](Self::chunk_shape), this lets a caller check up front
    /// whether a dataset is eligible for
    /// [`Dataset::append_staged`](crate::Dataset::append_staged), which needs a
    /// chunked dataset whose first maximum dimension is
    /// [`MaxExtent::Unlimited`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the dataspace message cannot be parsed, and
    /// [`Error::StaleHandle`] for a handle an object reference produced once a
    /// commit has run under it.
    pub fn maxshape(&self) -> Result<Option<Vec<MaxExtent>>, Error> {
        if let Some(meta) = self.staged_meta()? {
            return Ok(meta.maxshape);
        }
        Ok(self
            .dataspace()?
            .extensible_max_dimensions()
            .map(<[MaxExtent]>::to_vec))
    }

    /// Whether the dataset uses chunked storage (as opposed to contiguous or
    /// compact). Filtered datasets are always chunked. Returns `false` for a
    /// dataset with no data-layout message, for a non-chunked layout, and — like
    /// the other accessors that return no `Result` — for a handle that can no
    /// longer be resolved. A dataset this session has staged and not yet
    /// committed answers with the storage its builder selected.
    pub fn is_chunked(&self) -> bool {
        if let Ok(Some(meta)) = self.staged_meta() {
            return meta.chunked;
        }
        matches!(self.data_layout(), Ok(DataLayout::Chunked { .. }))
    }

    /// The dataset's chunk dimensions (one per dataset rank), or `Ok(None)` when
    /// the dataset is not chunked. The element-size dimension the on-disk layout
    /// appends is stripped, so the result lines up with
    /// [`shape`](Self::shape) / [`maxshape`](Self::maxshape).
    pub fn chunk_shape(&self) -> Result<Option<Vec<u64>>, Error> {
        let DataLayout::Chunked {
            chunk_dimensions, ..
        } = self.data_layout()?
        else {
            return Ok(None);
        };
        let rank = self.dataspace()?.dimensions.len();
        if chunk_dimensions.len() <= rank {
            return Ok(None);
        }
        Ok(Some(chunk_dimensions[..rank].to_vec()))
    }

    /// Returns the filter identifiers in application order.
    ///
    /// Returns an empty vector for an unfiltered dataset. For a staged dataset, returns the
    /// identifiers requested by its builder. This reads metadata without decoding chunk data.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the filter message or known filter parameters are invalid.
    /// Propagates errors from resolving the dataset or reading its metadata.
    pub fn filters(&self) -> Result<Vec<FilterId>, Error> {
        if let Some(meta) = self.staged_meta()? {
            return Ok(meta.filters.into_iter().map(|(id, _)| id).collect());
        }
        self.filter_pipeline_parsed()?
            .into_iter()
            .flat_map(|pipeline| pipeline.filters)
            .map(|description| Ok(Filter::from_description(description)?.id()))
            .collect()
    }

    /// How and where this dataset's raw data is stored: compact, contiguous,
    /// chunked, or virtual.
    ///
    /// The structured companion to [`is_chunked`](Self::is_chunked) and
    /// [`chunk_shape`](Self::chunk_shape), which it subsumes: one call that
    /// classifies the layout and, for a [`Layout::Contiguous`] dataset, gives the
    /// absolute address and byte size to seek to, or for a [`Layout::Chunked`]
    /// dataset the chunk shape and [`ChunkIndex`] kind. This parses only the
    /// data-layout message; it never walks the chunk index or reads any data —
    /// use [`chunks`](Self::chunks) for per-chunk locations. The curated analogue
    /// of `H5Pget_layout`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the data-layout message, or the dataspace a
    /// chunked dataset's chunk shape comes from, cannot be parsed.
    pub fn layout(&self) -> Result<Layout, Error> {
        Ok(match self.data_layout()? {
            DataLayout::Compact { data } => Layout::Compact {
                size: data.len() as u64,
            },
            DataLayout::Contiguous { address, size } => Layout::Contiguous {
                address: self.absolute_address(address)?,
                size,
            },
            DataLayout::Chunked { index, .. } => Layout::Chunked {
                // Reuse `chunk_shape` so the two accessors can never disagree on
                // how the element-size dimension is stripped.
                chunk_shape: self.chunk_shape()?.unwrap_or_default(),
                index: ChunkIndex::from_layout(index),
            },
            DataLayout::Virtual => Layout::Virtual,
        })
    }

    /// The [`ChunkIndex`] kind of this chunked dataset, or `Ok(None)` when the
    /// dataset is not chunked.
    ///
    /// A convenience shortcut for the `index` of [`Layout::Chunked`], for the
    /// common up-front append-eligibility check
    /// ([`ChunkIndex::supports_inplace_append`]). Complements
    /// [`maxshape`](Self::maxshape) and [`chunk_shape`](Self::chunk_shape).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the data-layout message cannot be parsed.
    pub fn chunk_index(&self) -> Result<Option<ChunkIndex>, Error> {
        match self.data_layout()? {
            DataLayout::Chunked { index, .. } => Ok(Some(ChunkIndex::from_layout(index))),
            _ => Ok(None),
        }
    }

    /// Returns every allocated chunk in index order, with its originating filter pipeline.
    ///
    /// Reads the chunk index and filter metadata without reading chunk data. Each [`Chunk`]
    /// retains the ordered pipeline and reports raw bypass independently of its recorded mask
    /// through [`Chunk::filters`]. Returns an empty vector for chunked storage without any
    /// allocated chunks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the dataset is not chunked or its layout, chunk index, or
    /// filter metadata is invalid. Propagates errors from resolving the dataset or reading
    /// its metadata.
    pub fn chunks(&self) -> Result<Vec<Chunk>, Error> {
        let dataspace = self.dataspace()?;
        let rank = dataspace.dimensions.len();
        let DataLayout::Chunked {
            flags,
            chunk_dimensions,
            ..
        } = self.data_layout()?
        else {
            return Err(FormatError::ChunkedReadError(
                "chunk enumeration requires a chunked dataset".into(),
            )
            .into());
        };
        let filters: Arc<[Filter]> = self.filter_pipeline()?.into();
        let addresses = self.file.addresses();
        self.raw_chunks()?
            .into_iter()
            .map(|c| {
                let bypassed_raw = ChunkDecoding::at_extent(
                    flags,
                    &chunk_dimensions,
                    &dataspace.dimensions,
                    &c.offsets,
                ) == ChunkDecoding::Verbatim;
                Ok(Chunk::new(
                    Arc::clone(&filters),
                    c.offsets.into_iter().take(rank).collect(),
                    addresses.resolve(c.address)?.get(),
                    c.chunk_size.get(),
                    c.filter_mask,
                    bypassed_raw,
                )?)
            })
            .collect()
    }

    /// Returns the filter pipeline in application order.
    ///
    /// Each [`Filter`] includes its identifier, recorded name, flags, interpreted parameters, and
    /// complete client data. Returns an empty vector for an unfiltered dataset.
    ///
    /// A staged dataset reports [`FilterParameters::Pending`](crate::FilterParameters::Pending)
    /// with the identifier and optional flag requested by its builder. The writer derives the
    /// encoded client data from the dataset when [`File::commit`] serializes it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the filter message or known filter parameters are invalid.
    /// Propagates errors from resolving the dataset or reading its metadata.
    pub fn filter_pipeline(&self) -> Result<Vec<Filter>, Error> {
        if let Some(meta) = self.staged_meta()? {
            return Ok(meta
                .filters
                .into_iter()
                .map(|(id, optional)| Filter::pending(id, optional))
                .collect());
        }
        self.filter_pipeline_parsed()?
            .into_iter()
            .flat_map(|pipeline| pipeline.filters)
            .map(Filter::from_description)
            .collect::<Result<_, _>>()
            .map_err(Error::from)
    }

    /// Returns the absolute file position of `address`, and `None` where a layout message stores
    /// the undefined address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds `u64`.
    fn absolute_address(&self, address: Option<StoredAddress>) -> Result<Option<u64>, Error> {
        match address {
            Some(stored) => Ok(Some(self.file.addresses().resolve(stored)?.get())),
            None => Ok(None),
        }
    }

    /// Returns the simplified datatype of the dataset.
    ///
    /// A dataset this session has staged and not yet committed answers with the
    /// datatype its builder settled on.
    pub fn dtype(&self) -> Result<DType, Error> {
        let dt = self.datatype()?;
        Ok(types::classify_datatype(&dt))
    }

    /// The size in bytes of one on-disk element of this dataset's datatype —
    /// HDF5's datatype storage size (`H5Tget_size`).
    ///
    /// This is the byte width of a single stored element: 8 for `f64`, the
    /// declared length for a fixed-length string, the record size for a compound
    /// type, or the reference/descriptor size for a variable-length type (whose
    /// payload lives separately in the file's global heaps).
    ///
    /// Multiplied by the element count from [`shape`](Self::shape), it is the
    /// exact number of raw bytes a full [`read_raw`](Self::read_raw)
    /// materializes. A caller reading an untrusted file can use it to bound that
    /// allocation up front rather than trusting the file's declared extent: a
    /// dataset can name a small element count yet a per-element size of billions
    /// of bytes, so the product — not the count alone — is what a read allocates.
    pub fn element_size(&self) -> Result<u64, Error> {
        Ok(u64::from(self.datatype()?.type_size()))
    }

    /// Returns the dataset's fill time policy.
    ///
    /// A dataset without a Fill Value message defaults to [`FillTime::IfSet`].
    /// The policy is defined in "The Data Storage - Fill Value Message" of the
    /// [format specification, version 4.0][spec].
    ///
    /// libhdf5 calls this property `H5D_fill_time_t` and sets it with
    /// `H5Pset_fill_time`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the Fill Value message is malformed.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, FillTime};
    ///
    /// let mut fb = FileBuilder::new();
    /// fb.create_dataset("data").with_i32_data(&[1, 2, 3]);
    /// let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    ///
    /// let dataset = file.dataset("data").unwrap();
    /// assert_eq!(dataset.fill_time().unwrap(), FillTime::IfSet);
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fvmessage
    pub fn fill_time(&self) -> Result<FillTime, Error> {
        let state = self.resolved()?;
        let msg = state
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILL_VALUE)
            .or_else(|| {
                state
                    .header
                    .messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FILL_VALUE_OLD)
            });
        let Some(m) = msg else {
            return Ok(FillTime::IfSet);
        };
        let body = self.file.message_body(m)?;
        let view = FillValueView::parse(m.msg_type, &body)?;
        Ok(view.fill_time())
    }

    /// Returns the dataset's parsed Fill Value message, or `None` when absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the Fill Value message is malformed.
    pub(crate) fn fill_value_message(&self) -> Result<Option<FillValueMessage>, Error> {
        let state = self.resolved()?;
        let msg = state
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILL_VALUE)
            .or_else(|| {
                state
                    .header
                    .messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FILL_VALUE_OLD)
            });
        let Some(m) = msg else {
            return Ok(None);
        };
        let body = self.file.message_body(m)?;
        let view = FillValueView::parse(m.msg_type, &body)?;
        Ok(Some(view.to_owned()))
    }

    /// Returns the raw bytes of the dataset's user-defined fill value, or `None`
    /// if no user-defined fill value is set.
    ///
    /// Reads either the current Fill Value message (`0x0005`, versions 1, 2, or
    /// 3) or the legacy Fill Value message (`0x0004`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the Fill Value message is malformed.
    pub(crate) fn defined_fill_bytes(&self) -> Result<Option<Vec<u8>>, Error> {
        let Some(msg) = self.fill_value_message()? else {
            return Ok(None);
        };
        Ok(msg.into_value())
    }

    /// Reads numeric values in row windows with one parsed source encoding.
    ///
    /// Reserves the output for the dataset after reading the first window. The reader keeps
    /// one window of stored bytes beside the output. Datasets that fit in one window are read
    /// whole. The reader checks numeric datatype eligibility even for an empty dataset.
    /// For chunked storage, reuses one parsed index across the windows even when index caching
    /// is disabled.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the numeric encoding is unsupported or the input contains
    /// an incomplete element. Returns [`FormatError::AllocationFailed`] through [`Error::Format`]
    /// if reserving the output for a windowed read fails. Propagates errors from reading the
    /// dataset metadata and stored bytes.
    ///
    /// # Panics
    ///
    /// Panics if decoding a single-window dataset requires more than [`isize::MAX`] bytes
    /// of output capacity.
    fn read_whole_typed<T: NumericReadTarget>(&self) -> Result<Vec<T>, Error> {
        self.read_whole_decoded::<T, NumericDecoder<T>>()
    }

    /// Reads typed elements in row windows with one parsed decoder.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the datatype or input bytes are unsupported for `T`.
    /// Returns [`FormatError::AllocationFailed`] through [`Error::Format`] if reserving
    /// windowed output fails. Propagates errors from reading metadata and stored bytes.
    ///
    /// # Panics
    ///
    /// Panics if the output capacity of a single-window read exceeds [`isize::MAX`] bytes.
    pub(crate) fn read_whole_decoded<T, D: ElementDecoder<T>>(&self) -> Result<Vec<T>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.read_layout()?;
        let pipeline = self.filter_pipeline_parsed()?;
        // See `read_raw`: an unparseable fill value message is carried into the
        // read rather than failing it up front.
        let elem_size = crate::datatype::element_size_usize(&dt)?;
        let fill_msg = self.fill_value_message();
        let fill = match &fill_msg {
            Ok(Some(m)) => FillPattern::from_message(m, elem_size),
            Ok(None) => FillPattern::new(FillTime::IfSet, None, elem_size),
            Err(_) => FillPattern::UNKNOWN,
        };
        let spec = RawReadSpec::parse(&dl, &ds, &dt, pipeline.as_ref(), fill)?;

        // A window is cut by the *stored* element width, while a decoder slices
        // what it is handed by the width of the type it decodes — the base type,
        // for an enumeration. Those are the same width for every valid file, an
        // enumeration's size being its base's. A crafted file where they differ
        // must not get one verdict from a sweep and another from a whole read, so
        // it is read whole.
        let decoded_width = D::decoded_width(&dt);

        let mut out = Vec::new();
        let n0 = ds.dimensions.first().copied().unwrap_or(1);
        let rows = typed_window_rows(&dl, &ds, elem_size)?.get();
        if n0 <= rows || decoded_width != dt.type_size() {
            // No reservation here: the decoder sizes the output from the bytes it
            // was handed, which is exact.
            let raw = self.file.read_dataset_raw(spec, &self.chunk_cache)?;
            D::parse_decoder(&dt)?.append_values(&raw, &mut out)?;
            return Ok(out);
        }

        let values = ds.num_elements().to_usize()?;

        // One pass for the whole sweep, not one per window: the sweep visits each
        // chunk exactly once, so a window that offered its chunks to a cache
        // already full would copy and evict with no later reader for either. The
        // cache ends up holding what a whole read would have left it — the
        // chunks reached first. See [`CachePass`].
        let pass = self.chunk_cache.begin_pass();
        self.file.with_source(|source| {
            let indexes = PhysicalSource::new(source).stored(self.file.addresses());

            let mut chunks = matches!(spec.storage(), RawReadStorage::Chunked { .. })
                .then(|| ChunkRead::new(spec, self.file.widths, &self.chunk_cache, pass));
            let mut read_rows = |start, count| match &mut chunks {
                Some(chunks) => chunks.read_rows(&indexes, start, count),
                None => {
                    let region = ReadRegion::rows(spec, start, count)?;
                    read_region_framed(&indexes, &region, self.file.widths, &self.chunk_cache, pass)
                }
            };
            let decoder = {
                let raw = read_rows(0, rows)?;
                // The reader returns errors from reading the first window before reserving output from
                // file dimensions. It uses `try_reserve` to return allocation errors if that capacity
                // is unavailable.
                out.try_reserve(values)
                    .map_err(|source| FormatError::AllocationFailed { values, source })?;
                let decoder = D::parse_decoder(&dt)?;
                decoder.append_values(&raw, &mut out)?;
                decoder
            };
            let mut start = rows;
            while start < n0 {
                let count = rows.min(n0 - start);
                let raw = read_rows(start, count)?;
                decoder.append_values(&raw, &mut out)?;
                start += count;
            }
            Ok(out)
        })
    }

    /// Read all data as `f64` values.
    ///
    /// This and the other typed whole-dataset readers decode a row window at a
    /// time, so the memory standing beside the returned `Vec` is one window of
    /// stored bytes — on the order of a mebibyte — rather than a second copy of
    /// the dataset. Reading a 4 GiB array costs about 4 GiB, not 8.
    ///
    /// The values are what [`read_raw`](Self::read_raw) returns, decoded: a
    /// dataset stored as a narrower or wider type is converted, so pick the
    /// reader that matches the stored type for a lossless read.
    pub fn read_f64(&self) -> Result<Vec<f64>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `f32` values.
    pub fn read_f32(&self) -> Result<Vec<f32>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `i32` values.
    pub fn read_i32(&self) -> Result<Vec<i32>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `i64` values.
    pub fn read_i64(&self) -> Result<Vec<i64>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `u64` values.
    pub fn read_u64(&self) -> Result<Vec<u64>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `u8` values.
    pub fn read_u8(&self) -> Result<Vec<u8>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `i8` values.
    pub fn read_i8(&self) -> Result<Vec<i8>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `i16` values.
    pub fn read_i16(&self) -> Result<Vec<i16>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `u16` values.
    pub fn read_u16(&self) -> Result<Vec<u16>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `u32` values.
    pub fn read_u32(&self) -> Result<Vec<u32>, Error> {
        self.read_whole_typed()
    }

    /// Read all data as `String` values.
    ///
    /// Fixed-length and variable-length HDF5 string datasets are both
    /// supported. Use [`read_vlen_strings`](Self::read_vlen_strings) when
    /// variable-length allocation limits are required.
    pub fn read_string(&self) -> Result<Vec<String>, Error> {
        let dt = self.datatype()?;
        if vl_data::is_vlen_string_datatype(&dt) {
            self.read_vlen_strings(VlenStringReadOptions::default())
        } else {
            let raw = self.read_raw()?;
            Ok(data_read::read_as_strings(&raw, &dt)?)
        }
    }

    /// Return the total bytes referenced by this VL string dataset.
    ///
    /// This is the payload equivalent of HDF5's `H5Dvlen_get_buf_size`: it
    /// excludes `Vec<String>` and `String` allocation metadata.
    pub fn vlen_string_payload_size(&self) -> Result<u64, Error> {
        let datatype = self.datatype()?;
        if !vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "VariableLength string",
                actual: "non-VariableLength string",
            }
            .into());
        }
        let dataspace = self.dataspace()?;
        let raw = self.read_raw()?;
        Ok(vl_data::vlen_string_payload_size(
            &raw,
            dataspace.num_elements(),
            self.file.widths.offsets,
        )?)
    }

    /// Read a VL string dataset with explicit allocation limits.
    ///
    /// Both limits are checked before any string payload is materialized.
    pub fn read_vlen_strings(&self, options: VlenStringReadOptions) -> Result<Vec<String>, Error> {
        let mut strings = Vec::new();
        self.visit_vlen_strings(options, |string| strings.push(string.to_owned()))?;
        Ok(strings)
    }

    /// Visit a VL string dataset one element at a time.
    ///
    /// The string slice passed to `visitor` is valid only for the duration of
    /// that callback. This avoids retaining all decoded string payloads at once.
    ///
    /// On a read-write file ([`File::open_rw`]) the
    /// visitor runs while the file's engine lock is held, so it must not read
    /// or write through this file (or a clone / handle of it) — doing so
    /// deadlocks. Collect values and act on them after the call instead.
    pub fn visit_vlen_strings<F>(
        &self,
        options: VlenStringReadOptions,
        visitor: F,
    ) -> Result<(), Error>
    where
        F: FnMut(&str),
    {
        let datatype = self.datatype()?;
        if !vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "VariableLength string",
                actual: "non-VariableLength string",
            }
            .into());
        }
        let dataspace = self.dataspace()?;
        if let Some(limit) = options.max_elements()
            && dataspace.num_elements() > limit as u64
        {
            return Err(FormatError::VariableLengthElementLimitExceeded {
                limit,
                actual: dataspace.num_elements(),
            }
            .into());
        }
        let raw = self.read_raw()?;
        self.file.with_source(|source| {
            Ok(vl_data::visit_vl_strings_from_source(
                source,
                &raw,
                dataspace.num_elements(),
                self.file.widths,
                self.file.addresses(),
                options,
                visitor,
            )?)
        })
    }

    /// Read a VL string dataset's exact heap bytes, preserving the
    /// null-vs-empty distinction and never lossily decoding.
    ///
    /// Unlike [`read_vlen_strings`](Self::read_vlen_strings), which returns
    /// `String`s via `from_utf8_lossy` and so cannot reproduce embedded NULs or
    /// non-UTF-8 payloads, this yields each element's raw bytes (or a null
    /// marker). It underpins faithful rewriting (e.g. repack) of VL strings.
    pub(crate) fn read_vlen_string_bytes(
        &self,
        options: VlenStringReadOptions,
    ) -> Result<Vec<vl_data::VlByteObject>, Error> {
        let datatype = self.datatype()?;
        if !vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "VariableLength string",
                actual: "non-VariableLength string",
            }
            .into());
        }
        let dataspace = self.dataspace()?;
        if let Some(limit) = options.max_elements()
            && dataspace.num_elements() > limit as u64
        {
            return Err(FormatError::VariableLengthElementLimitExceeded {
                limit,
                actual: dataspace.num_elements(),
            }
            .into());
        }
        let raw = self.read_raw()?;
        self.file.with_source(|source| {
            Ok(vl_data::read_vl_byte_objects_from_source(
                source,
                &raw,
                dataspace.num_elements(),
                self.file.widths,
                self.file.addresses(),
                1, // VL string references count bytes
                options,
            )?)
        })
    }

    /// Read every element of a *non-string* variable-length (sequence) dataset as
    /// its exact heap bytes, alongside the base-type element size in bytes.
    ///
    /// Each element's heap object holds `length * element_size` bytes, where
    /// `length` is the stored element count and `element_size` is the byte width
    /// of the sequence's base type. Returning the raw bytes (not decoded values)
    /// keeps a faithful rewrite (repack) byte-exact for any base type whose bytes
    /// carry no embedded heap or file addresses. Errors with a
    /// [`TypeMismatch`](crate::FormatError::TypeMismatch) if the datatype is not a
    /// non-string VL datatype.
    pub(crate) fn read_vlen_sequence_bytes(
        &self,
        options: VlenStringReadOptions,
    ) -> Result<(Vec<vl_data::VlByteObject>, usize), Error> {
        let datatype = self.datatype()?;
        let Datatype::VariableLength(VariableLengthType::Sequence {
            element_type: base_type,
        }) = &datatype
        else {
            return Err(FormatError::TypeMismatch {
                expected: "non-string VariableLength",
                actual: "non-VariableLength",
            }
            .into());
        };
        if vl_data::is_vlen_string_datatype(&datatype) {
            return Err(FormatError::TypeMismatch {
                expected: "non-string VariableLength",
                actual: "VariableLength string",
            }
            .into());
        }
        let element_size = base_type.type_size() as usize;
        if element_size == 0 {
            return Err(
                FormatError::VlDataError("non-string VL base type has zero size".into()).into(),
            );
        }
        let dataspace = self.dataspace()?;
        if let Some(limit) = options.max_elements()
            && dataspace.num_elements() > limit as u64
        {
            return Err(FormatError::VariableLengthElementLimitExceeded {
                limit,
                actual: dataspace.num_elements(),
            }
            .into());
        }
        let raw = self.read_raw()?;
        let objects = self.file.with_source(|source| {
            vl_data::read_vl_byte_objects_from_source(
                source,
                &raw,
                dataspace.num_elements(),
                self.file.widths,
                self.file.addresses(),
                element_size,
                options,
            )
        })?;
        Ok((objects, element_size))
    }

    /// Read a dataset whose datatype *contains* variable-length references
    /// without being variable-length itself — a compound with a VL member, or an
    /// array of them (issue #201).
    ///
    /// Returns everything a rewrite needs: the element bytes, where each embedded
    /// reference sits within them, and the heap payload each one names. That lets
    /// the writer re-stage the payloads into a new file's global heap and rewrite
    /// the references in place, which is what keeps a rewrite from carrying the
    /// source file's heap addresses into the destination.
    ///
    /// The references are resolved one slot at a time, so `options`' limits apply
    /// per slot rather than across the whole dataset.
    pub(crate) fn read_embedded_vlen_bytes(
        &self,
        slots: &[vl_data::EmbeddedVlSlot],
        options: VlenStringReadOptions,
    ) -> Result<vl_data::EmbeddedVlData, Error> {
        let stride = self.datatype()?.type_size() as usize;
        let dataspace = self.dataspace()?;
        let n = dataspace.num_elements();
        if let Some(limit) = options.max_elements()
            && n > limit as u64
        {
            return Err(
                FormatError::VariableLengthElementLimitExceeded { limit, actual: n }.into(),
            );
        }

        // A zero-element dataset owns no element bytes, so there is no storage
        // to visit: skip the read rather than open a dataset the C library
        // left unallocated only to receive the same empty buffer back.
        let raw = if n == 0 { Vec::new() } else { self.read_raw()? };
        let n_usize = n.to_usize()?;
        let needed = n_usize
            .checked_mul(stride)
            .ok_or(FormatError::OffsetOverflow {
                offset: n,
                length: stride as u64,
            })?;
        if raw.len() < needed {
            return Err(FormatError::UnexpectedEof {
                expected: needed,
                available: raw.len(),
            }
            .into());
        }

        let mut offsets = Vec::with_capacity(n_usize * slots.len());
        let mut objects = Vec::with_capacity(n_usize * slots.len());
        for slot in slots {
            // Gather this slot's reference from every element into a dense buffer,
            // which is the shape the shared VL reader consumes. Each slot has its
            // own base-type width, so they are resolved a slot at a time rather
            // than in one pass.
            let mut dense = Vec::with_capacity(n_usize * VL_REF_SIZE);
            for e in 0..n_usize {
                let at = e * stride + slot.byte_offset;
                dense.extend_from_slice(&raw[at..at + VL_REF_SIZE]);
                offsets.push(at);
            }
            let resolved = self.file.with_source(|source| {
                vl_data::read_vl_byte_objects_from_source(
                    source,
                    &dense,
                    n,
                    self.file.widths,
                    self.file.addresses(),
                    slot.element_size,
                    options,
                )
            })?;
            objects.extend(resolved);
        }
        Ok(vl_data::EmbeddedVlData {
            raw,
            offsets,
            objects,
        })
    }

    /// Read all attributes of this dataset.
    ///
    /// The variant of each value describes its on-disk encoding; see
    /// [`Group::attrs`] for what that means for matching on it, and prefer the
    /// [`AttrValue`] accessors.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        self.file.attrs_of(&self.resolved()?.header)
    }

    /// The exact on-disk [`Datatype`] of every attribute on this dataset, keyed
    /// by name.
    ///
    /// See [`Group::attr_datatypes`] for what this channel carries that
    /// [`attrs`](Self::attrs) cannot, including how a boolean attribute is
    /// recognized. Note that this describes the *attributes*, not the dataset's
    /// own element type — that is [`datatype`](Self::datatype).
    pub fn attr_datatypes(&self) -> Result<HashMap<String, Datatype>, Error> {
        Ok(self
            .attr_messages()?
            .into_iter()
            .map(|a| (a.name, a.datatype))
            .collect())
    }

    /// Every attribute message on this dataset as it is encoded on disk, in the
    /// order the header holds them.
    ///
    /// See [`Group::attr_messages`] for why repack reads these rather than the
    /// decoded map.
    pub(crate) fn attr_messages(&self) -> Result<Vec<crate::attribute::AttributeMessage>, Error> {
        self.file.attr_messages_of(&self.resolved()?.header)
    }

    /// Returns the exact HDF5 datatype, including compound field offsets and
    /// total record size.
    ///
    /// A committed (`H5Tcommit`) element type — what netCDF-4 writes for a
    /// user-defined type, and what h5py writes for
    /// `create_dataset(..., dtype=f["t"])` — is stored as a reference to the
    /// datatype's own object header and is resolved to the type it names.
    pub fn datatype(&self) -> Result<Datatype, Error> {
        if let Some(meta) = self.staged_meta()? {
            return Ok(meta.datatype);
        }
        let state = self.resolved()?;
        let msg = find_message(&state.header, MessageType::DATATYPE)?;
        let (dt, _) = hdf5_pure_format::__private::parse_datatype(&self.file.message_body(msg)?)?;
        Ok(dt)
    }

    /// Returns the object-header address of this dataset's committed element type, as its
    /// Datatype message stores it, or `None` if the type is stored inline or in the
    /// shared-message heap.
    ///
    /// [`datatype`](Self::datatype) resolves it either way, so this is for
    /// callers that must *reproduce* the dataset: writing the resolved type back
    /// inline loses the link every C-library reader reports by name, and the
    /// address is what says which committed object to name instead.
    pub(crate) fn committed_datatype_address(&self) -> Result<Option<StoredAddress>, Error> {
        let state = self.resolved()?;
        let msg = find_message(&state.header, MessageType::DATATYPE)?;
        self.file.shared_target_address(msg)
    }

    pub(crate) fn dataspace(&self) -> Result<Dataspace, Error> {
        let state = self.resolved()?;
        let msg = find_message(&state.header, MessageType::DATASPACE)?;
        Ok(Dataspace::parse_with_length_width(
            &self.file.message_body(msg)?,
            self.file.widths.lengths,
        )?)
    }

    pub(crate) fn data_layout(&self) -> Result<DataLayout, Error> {
        let state = self.resolved()?;
        let msg = find_message(&state.header, MessageType::DATA_LAYOUT)?;
        Ok(DataLayout::parse_with_widths(&msg.data, self.file.widths)?)
    }

    pub(crate) fn filter_pipeline_parsed(&self) -> Result<Option<FilterPipeline>, Error> {
        let state = self.resolved()?;
        let Some(msg) = state
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILTER_PIPELINE)
        else {
            return Ok(None);
        };
        let body = self.file.message_body(msg)?;
        Ok(Some(crate::filter_pipeline::parse_filter_pipeline(&body)?))
    }

    /// Whether this dataset's element bytes live in files outside this one
    /// (`H5Pset_external`, the External Data Files header message, type 7).
    ///
    /// Such a dataset carries a *contiguous* layout message whose data address
    /// is undefined — the same encoding a never-written dataset uses — so a
    /// caller that reads "no address" as "no storage" would call a dataset full
    /// of data empty. This crate does not follow the external files, so the only
    /// safe answer is to refuse: [`read_layout`](Self::read_layout) refuses a
    /// read of one rather than answering its fill value, and `repack` refuses to
    /// reproduce it without its data.
    pub(crate) fn has_external_storage(&self) -> Result<bool, Error> {
        Ok(self
            .resolved()?
            .header
            .messages
            .iter()
            .any(|m| m.msg_type == MessageType::EXTERNAL_DATA_FILES))
    }

    /// The data layout to read element bytes through, as opposed to the one
    /// [`layout`](Self::layout) reports.
    ///
    /// [`data_layout`](Self::data_layout) answers what the message records;
    /// this answers whether those bytes are reachable at all. The two differ for
    /// exactly one kind of dataset: an externally stored one, whose contiguous layout
    /// with no address is also what a never-written dataset carries, so reading
    /// it would answer the fill value for every element it holds. Introspection
    /// keeps answering — the address-less layout is the evidence a caller needs
    /// — while every path that turns a layout into bytes comes through here and
    /// refuses.
    fn read_layout(&self) -> Result<DataLayout, Error> {
        if self.has_external_storage()? {
            return Err(FormatError::UnsupportedExternalStorage.into());
        }
        self.data_layout()
    }

    /// Returns one [`ChunkInfo`](crate::chunked_read::ChunkInfo) per allocated chunk of this
    /// chunked dataset, the records the chunked reader walks before decompressing.
    ///
    /// Every address is in the frame the chunk index stores it in, so a caller reading the
    /// still-compressed chunk bytes from the file's own source adds the base address first,
    /// as `repack` does when it copies a chunk without decoding it.
    ///
    /// Returns `Err` if the layout is not chunked. Returns `Ok(vec![])` for an
    /// empty / never-allocated chunked dataset (no index address). Covers every
    /// index type the reader supports (v3 B-tree and all five v4 index types, including the
    /// version 2 B-tree).
    pub(crate) fn raw_chunks(&self) -> Result<Vec<crate::chunked_read::ChunkInfo>, Error> {
        let DataLayout::Chunked {
            chunk_dimensions,
            index,
            ..
        } = self.data_layout()?
        else {
            return Err(Error::Format(crate::error::FormatError::ChunkedReadError(
                "chunk enumeration requires a chunked dataset".into(),
            )));
        };
        // An undefined index address means no storage is allocated yet.
        if index.address().is_none() {
            return Ok(Vec::new());
        }
        let dataspace = self.dataspace()?;
        let elem_size = crate::datatype::element_size_usize(&self.datatype()?)?;
        // The chunk index stores base-relative addresses, in its root and in every internal
        // node, so the walk runs over a view of the file framed at the base. Every address it
        // reports stays in that frame.
        self.file.with_source(|source| {
            let indexes = PhysicalSource::new(source).stored(self.file.addresses());
            Ok(crate::chunked_read::collect_chunks_for_layout_from_source(
                &indexes,
                index,
                &chunk_dimensions,
                &dataspace,
                elem_size.get() as u64,
                self.file.widths,
            )?)
        })
    }

    /// The raw `FilterPipeline` message bytes from this dataset's object header,
    /// if it has one. Repack reuses this verbatim so that every filter — including
    /// ones this crate cannot itself apply (ZFP, SZIP, unknown) — is reproduced
    /// byte-for-byte in the repacked file's pipeline message.
    pub(crate) fn filter_pipeline_message_bytes(&self) -> Result<Option<Vec<u8>>, Error> {
        let state = self.resolved()?;
        let Some(msg) = state
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILTER_PIPELINE)
        else {
            return Ok(None);
        };
        // A shared pipeline message's record body is an address, not a pipeline;
        // its resolved content is what a copy must carry. The content itself is
        // position-independent, so copying it verbatim stays faithful.
        Ok(Some(self.file.message_body(msg)?.into_owned()))
    }

    /// Read the dataset's exact unfiltered element bytes.
    ///
    /// For compound datasets this preserves all file padding and uses the
    /// offsets reported by [`datatype`](Self::datatype).
    ///
    /// A dataset whose elements live in external files (`H5Pset_external`) is
    /// refused with `FormatError::UnsupportedExternalStorage` rather than read as
    /// unallocated storage, which is what its layout message alone says. This
    /// applies to every read here; its shape, datatype, and
    /// [`layout`](Self::layout) still read.
    pub fn read_raw(&self) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.read_layout()?;
        // The data layout's on-disk addresses are left base-relative here;
        // `read_dataset_raw` applies the base address centrally (for both
        // contiguous and chunked layouts) by reading from a base-relative view of
        // the file.
        let pipeline = self.filter_pipeline_parsed()?;
        // A fill value message this parser cannot read does not, by itself,
        // make the dataset unreadable: it only decides what *unallocated*
        // storage looks like. Carry the uncertainty into the read and let it
        // fail there, and only there. `Dataset::fill_value` still reports the
        // parse error to a caller asking about the value.
        let elem_size = crate::datatype::element_size_usize(&dt)?;
        let fill_msg = self.fill_value_message();
        let fill = match &fill_msg {
            Ok(Some(m)) => FillPattern::from_message(m, elem_size),
            Ok(None) => FillPattern::new(FillTime::IfSet, None, elem_size),
            Err(_) => FillPattern::UNKNOWN,
        };
        let spec = RawReadSpec::parse(&dl, &ds, &dt, pipeline.as_ref(), fill)?;
        Ok(self.file.read_dataset_raw(spec, &self.chunk_cache)?)
    }

    /// Reads the unfiltered element bytes of complete leading-dimension rows.
    ///
    /// Clamps `start_row` and `num_rows` to the current leading dimension. Each row includes
    /// every element of the inner dimensions, and a scalar counts as one row. A request with
    /// zero rows or a zero inner dimension returns an empty vector.
    ///
    /// Reads a contiguous byte range for compact or contiguous storage and decodes intersecting
    /// chunks for chunked storage. For a selection covering the whole dataset, reads through
    /// [`read_raw`](Self::read_raw). Variable-length string elements contain heap references.
    /// Use [`read_string_rows`](Self::read_string_rows) to read their text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the selection exceeds the platform's buffer limit, byte
    /// arithmetic overflows, or dataset metadata or storage cannot be read. Dataset layout
    /// consistency checks apply to empty selections.
    pub fn read_raw_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u8>, Error> {
        self.read_raw_selection(RegionRequest::Rows {
            start: start_row,
            count: num_rows,
        })
    }

    /// Reads unfiltered element bytes from a unit-stride region in row-major order.
    ///
    /// `start` and `count` contain one coordinate and element count per dataset dimension.
    /// Every range must fit the current shape. Returns an empty vector for a zero count in any
    /// dimension after checking every range. For a scalar dataspace, pass empty slices to read
    /// its single element. A null dataspace has an empty selection.
    ///
    /// The result is packed with the last dimension varying fastest. Contiguous reads fetch the
    /// selected byte runs, and chunked reads decode intersecting chunks. For a region covering the
    /// whole dataset, reads through [`read_raw`](Self::read_raw).
    ///
    /// Variable-length string elements contain heap references. Use
    /// [`read_string_region`](Self::read_string_region) to read their text.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::RegionRankMismatch`] if either slice length differs from the
    /// dataset rank, and [`FormatError::RegionOutOfBounds`] if a range exceeds a dimension's
    /// current extent. Returns [`FormatError::OffsetOverflow`] if byte arithmetic overflows,
    /// and [`FormatError::ValueTooLargeForPlatform`] if the result exceeds the platform's
    /// buffer limit. These format errors are returned as [`Error::Format`].
    ///
    /// Returns errors from reading dataset metadata, storage, or filters. Dataset layout
    /// consistency checks apply to empty selections.
    ///
    /// # Examples
    ///
    /// ```
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("pixels.h5");
    /// # let mut builder = hdf5_pure::FileBuilder::new();
    /// # builder.create_dataset("pixels")
    /// #     .with_u16_data(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    /// #     .with_shape(&[3, 4]);
    /// # builder.write(&path)?;
    /// use hdf5_pure::File;
    ///
    /// let file = File::open(&path)?;
    /// let dataset = file.dataset("pixels")?;
    /// let bytes = dataset.read_raw_region(&[1, 1], &[2, 2])?;
    /// assert_eq!(bytes, [5, 0, 6, 0, 9, 0, 10, 0]);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn read_raw_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<u8>, Error> {
        self.read_raw_selection(RegionRequest::Bounds { start, count })
    }

    /// Reads a unit-stride region as row-major [`f64`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_f64`](Self::read_f64).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    ///
    /// # Examples
    ///
    /// ```
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("volume.h5");
    /// # let values: Vec<_> = (0..24).map(f64::from).collect();
    /// # let mut builder = hdf5_pure::FileBuilder::new();
    /// # builder.create_dataset("volume")
    /// #     .with_f64_data(&values)
    /// #     .with_shape(&[2, 3, 4]);
    /// # builder.write(&path)?;
    /// use hdf5_pure::File;
    ///
    /// let file = File::open(&path)?;
    /// let dataset = file.dataset("volume")?;
    /// let values = dataset.read_f64_region(&[1, 0, 1], &[1, 2, 2])?;
    /// assert_eq!(values, [13.0, 14.0, 17.0, 18.0]);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn read_f64_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<f64>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_f64(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`f32`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_f32`](Self::read_f32). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_f32_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<f32>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_f32(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`i8`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_i8`](Self::read_i8). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_i8_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<i8>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_i8(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`i16`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_i16`](Self::read_i16). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_i16_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<i16>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_i16(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`i32`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_i32`](Self::read_i32). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_i32_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<i32>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_i32(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`i64`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_i64`](Self::read_i64). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_i64_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<i64>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_i64(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`u8`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_u8`](Self::read_u8). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_u8_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<u8>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_u8(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`u16`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_u16`](Self::read_u16). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_u16_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<u16>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_u16(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`u32`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_u32`](Self::read_u32). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_u32_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<u32>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_u32(&raw, &self.datatype()?)?)
    }

    /// Reads a unit-stride region as row-major [`u64`] values.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region) and the numeric
    /// conversions of [`read_u64`](Self::read_u64). See
    /// [`read_f64_region`](Self::read_f64_region) for an example.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the region cannot be read or the datatype cannot be decoded
    /// as numeric values. See [`read_raw_region`](Self::read_raw_region) for selection and
    /// storage errors.
    pub fn read_u64_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<u64>, Error> {
        let raw = self.read_raw_region(start, count)?;
        Ok(data_read::read_as_u64(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_f64`](Self::read_f64) — decodes only the row window.
    pub fn read_f64_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<f64>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_f64(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_f32`](Self::read_f32) — decodes only the row window.
    pub fn read_f32_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<f32>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_f32(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_i8`](Self::read_i8) — decodes only the row window.
    pub fn read_i8_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i8>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_i8(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_i16`](Self::read_i16) — decodes only the row window.
    pub fn read_i16_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i16>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_i16(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_i32`](Self::read_i32) — decodes only the row window.
    pub fn read_i32_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i32>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_i32(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_i64`](Self::read_i64) — decodes only the row window.
    pub fn read_i64_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i64>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_i64(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_u8`](Self::read_u8) — reads only the row window.
    pub fn read_u8_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u8>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_u8(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_u16`](Self::read_u16) — decodes only the row window.
    pub fn read_u16_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u16>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_u16(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_u32`](Self::read_u32) — decodes only the row window.
    pub fn read_u32_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u32>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_u32(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_u64`](Self::read_u64) — decodes only the row window.
    pub fn read_u64_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u64>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_u64(&raw, &self.datatype()?)?)
    }

    /// Windowed [`read_string`](Self::read_string).
    ///
    /// Fixed-length strings decode straight from the window. Variable-length
    /// strings resolve only the window's heap references, so the window memory
    /// bound holds for them too: peak allocation is the window's references,
    /// its text, and the metadata of the heap collections it touches.
    pub fn read_string_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<String>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        self.read_strings_from_raw(&raw, &self.datatype()?)
    }

    /// Reads a unit-stride region of fixed-length or variable-length strings in row-major order.
    ///
    /// Uses the selection rules of [`read_raw_region`](Self::read_raw_region). For variable-length
    /// strings, reads the text referenced by the selected elements.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TypeMismatch`] through [`Error::Format`] if the dataset's datatype
    /// is not a string type. Returns errors from [`read_raw_region`](Self::read_raw_region),
    /// decoding strings, or reading global heap objects.
    ///
    /// # Examples
    ///
    /// ```
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("labels.h5");
    /// # let mut builder = hdf5_pure::FileBuilder::new();
    /// # builder.create_dataset("labels")
    /// #     .with_vlen_strings(&["north", "south", "east", "west", "up", "down"])
    /// #     .with_shape(&[2, 3]);
    /// # builder.write(&path)?;
    /// use hdf5_pure::File;
    ///
    /// let file = File::open_streaming(&path)?;
    /// let dataset = file.dataset("labels")?;
    /// let labels = dataset.read_string_region(&[0, 1], &[2, 2])?;
    /// assert_eq!(labels, ["south", "east", "up", "down"]);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn read_string_region(&self, start: &[u64], count: &[u64]) -> Result<Vec<String>, Error> {
        let raw = self.read_raw_region(start, count)?;
        self.read_strings_from_raw(&raw, &self.datatype()?)
    }

    /// Interpret this dataset as an array of HDF5 object references
    /// (`H5R_OBJECT`) and resolve each, in storage order, to the [`Object`] it
    /// points at.
    ///
    /// MATLAB cell arrays and the `#subsystem#` machinery store their members
    /// this way: the dataset holds one object-header address per element, each
    /// naming an object elsewhere in the file (conventionally under the hidden
    /// `#refs#` group).
    ///
    /// # Errors
    ///
    /// - [`FormatError::TypeMismatch`] if this dataset's datatype is not an
    ///   object reference.
    /// - [`FormatError::InvalidObjectReference`] if an element is a null or
    ///   undefined reference, or does not point at a group or dataset.
    pub fn dereference(&self) -> Result<Vec<Object>, Error> {
        let dt = self.datatype()?;
        if !matches!(
            dt,
            Datatype::Reference {
                ref_type: ReferenceType::Object,
                ..
            }
        ) {
            return Err(FormatError::TypeMismatch {
                expected: "object reference",
                actual: "non-reference datatype",
            }
            .into());
        }
        // An object reference stores an 8-byte object-header address. Refuse a
        // sub-address-width element rather than read a truncated address.
        let elem_size = dt.type_size().to_usize()?;
        if elem_size < 8 {
            return Err(FormatError::TypeMismatch {
                expected: "8-byte object reference",
                actual: "object reference narrower than 8 bytes",
            }
            .into());
        }
        let raw = self.read_raw()?;
        if raw.is_empty() {
            return Ok(Vec::new());
        }
        if !raw.len().is_multiple_of(elem_size) {
            return Err(FormatError::DataSizeMismatch {
                expected: elem_size,
                actual: raw.len(),
            }
            .into());
        }
        // Read after the element bytes the addresses came out of, which is the
        // one place these could be taken too late: a commit landing between that
        // read and this one would move the headers the addresses name, and a
        // handle labelled with the later revisions would call them current.
        let revisions = self.file.revisions();
        let mut out = Vec::with_capacity(raw.len() / elem_size);
        for chunk in raw.chunks_exact(elem_size) {
            let addr = u64::from_le_bytes(chunk[..8].try_into().expect("chunk has >= 8 bytes"));
            out.push(FileInner::object_at_relative(&self.file, revisions, addr)?);
        }
        Ok(out)
    }

    /// Decode all elements of a compound dataset field by field.
    ///
    /// Built-in implementations support numeric tuples with one through twelve
    /// fields. Decoding uses the file's field offsets rather than Rust's tuple
    /// memory layout, so padded compound records are supported safely.
    pub fn read_compound<T: CompoundType>(&self) -> Result<Vec<T>, Error> {
        let datatype = self.datatype()?;
        let element_size = crate::datatype::element_size_usize(&datatype)?;
        if !matches!(datatype, Datatype::Compound { .. }) {
            return Err(FormatError::TypeMismatch {
                expected: "Compound",
                actual: "non-Compound",
            }
            .into());
        }
        let raw = self.read_raw()?;
        if !raw.len().is_multiple_of(element_size.get()) {
            return Err(FormatError::DataSizeMismatch {
                expected: element_size.get(),
                actual: raw.len(),
            }
            .into());
        }
        raw.chunks_exact(element_size.get())
            .map(|bytes| T::decode(&datatype, bytes).map_err(Error::Format))
            .collect()
    }

    /// Verify this dataset against its stored provenance hash.
    ///
    /// Recomputes the SHA-256 of the dataset's raw bytes and compares it with
    /// the `_provenance_sha256` attribute written by
    /// [`DatasetBuilder::with_provenance`](crate::DatasetBuilder::with_provenance).
    /// Returns [`VerifyResult::NoHash`](crate::VerifyResult::NoHash) when the
    /// dataset carries no provenance hash, so a missing hash is distinguishable
    /// from an actual mismatch.
    #[cfg(feature = "provenance")]
    pub fn verify_provenance(&self) -> Result<crate::provenance::VerifyResult, Error> {
        use crate::provenance::{self, ATTR_SHA256, VerifyResult};

        let attrs = self.attrs()?;
        let stored = match attrs.get(ATTR_SHA256).and_then(AttrValue::as_str) {
            Some(s) => s.trim_end_matches('\0').to_string(),
            None => return Ok(VerifyResult::NoHash),
        };

        let computed = provenance::sha256_hex(&self.read_raw()?);
        if computed == stored {
            Ok(VerifyResult::Ok)
        } else {
            Ok(VerifyResult::Mismatch { stored, computed })
        }
    }

    fn read_raw_selection(&self, request: RegionRequest<'_>) -> Result<Vec<u8>, Error> {
        let datatype = self.datatype()?;
        let dataspace = self.dataspace()?;
        let layout = self.read_layout()?;
        let pipeline = self.filter_pipeline_parsed()?;
        let element_size = crate::datatype::element_size_usize(&datatype)?;
        let fill_message = self.fill_value_message();
        let fill = match &fill_message {
            Ok(Some(message)) => FillPattern::from_message(message, element_size),
            Ok(None) => FillPattern::new(FillTime::IfSet, None, element_size),
            Err(_) => FillPattern::UNKNOWN,
        };
        let spec = RawReadSpec::parse(&layout, &dataspace, &datatype, pipeline.as_ref(), fill)?;
        let region = request.parse(spec)?;
        Ok(self
            .file
            .read_dataset_region(&region, &self.chunk_cache, CachePass::LRU)?)
    }

    fn read_strings_from_raw(&self, raw: &[u8], datatype: &Datatype) -> Result<Vec<String>, Error> {
        if !vl_data::is_vlen_string_datatype(datatype) {
            return Ok(data_read::read_as_strings(raw, datatype)?);
        }
        let reference_size = VlElement::serialized_size(self.file.widths.offsets);
        let elements = (raw.len() / reference_size) as u64;
        let mut strings = Vec::new();
        self.file.with_source(|source| -> Result<(), Error> {
            Ok(vl_data::visit_vl_strings_from_source(
                source,
                raw,
                elements,
                self.file.widths,
                self.file.addresses(),
                VlenStringReadOptions::default(),
                |string| strings.push(String::from(string)),
            )?)
        })?;
        Ok(strings)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_message(
    header: &ObjectHeader,
    msg_type: MessageType,
) -> Result<&crate::object_header::HeaderMessage, Error> {
    header
        .messages
        .iter()
        .find(|m| m.msg_type == msg_type)
        .ok_or(Error::MissingMessage(msg_type))
}

fn has_message(header: &ObjectHeader, msg_type: MessageType) -> bool {
    header.messages.iter().any(|m| m.msg_type == msg_type)
}

/// The class of an object, read from the messages of its object header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ObjectClass {
    Dataset,
    Group,
    NamedDatatype,
    /// An object of none of the other classes.
    Unknown,
}

impl ObjectClass {
    // `H5O__obj_class_real` in `H5Oint.c`, HDF5 2.2.0, tests for a group, then a dataset, then a
    // datatype.
    fn of(header: &ObjectHeader) -> Self {
        if group_v2::is_group(header) {
            Self::Group
        } else if has_message(header, MessageType::DATA_LAYOUT) {
            Self::Dataset
        } else if is_named_datatype(header) {
            Self::NamedDatatype
        } else {
            Self::Unknown
        }
    }
}

/// Whether an object header describes a committed (`H5Tcommit`) datatype: it
/// carries a datatype and is neither a dataset nor a group.
///
/// A dataset's header carries a datatype message too — its element type — so
/// "has a datatype message" is not the question, and a lookup that asked only
/// that answered a dataset's element type where it owed a refusal (issue #364).
/// The listing and the by-name lookups share this one predicate so they cannot
/// disagree about the same child.
///
/// The conjunction encodes the precedence the reference library gets from its
/// ordering: `H5O__obj_class_real` walks `H5O_obj_class_g` in reverse, so it
/// asks group, then dataset, then datatype, and that is what `H5Topen` gates on.
/// Two terms are read differently here. It calls a header a dataset for a
/// datatype beside a *dataspace* where this reads a datatype beside a data
/// layout, and a group for a symbol table or link info where this counts a bare
/// link message as well. Every object either library writes carries the messages
/// that make those agree, so the rules part only on a malformed header.
fn is_named_datatype(header: &ObjectHeader) -> bool {
    has_message(header, MessageType::DATATYPE)
        && !has_message(header, MessageType::DATA_LAYOUT)
        && !group_v2::is_group(header)
}

/// The root-relative path of a child named `name` under `parent`, or `None` if
/// the parent has no resolvable path (reached by object reference).
///
/// Free-standing rather than a method on [`Group`] so the member iterators can
/// build child paths from a closure that outlives the borrow of the group they
/// came from.
///
/// A member whose name is no path component, such as the empty name a malformed file can store,
/// has no [`LinkName`] for the iterators to pass, and they return its handle with `None` for a
/// path.
fn child_path_of(parent: Option<&ObjectPathBuf>, name: LinkName<'_>) -> Option<ObjectPathBuf> {
    Some(parent?.join(&LinkNameBuf::from(name)))
}

/// Returns the root-relative path an error reports an object by, and the empty string for a
/// handle reached by object reference, which has no path.
fn reported_path(path: Option<&ObjectPathBuf>) -> String {
    path.map_or_else(String::new, ObjectPathBuf::to_string)
}

/// The reason [`Group::child_edit_path`] gives for a path that spells no link.
const NO_LINK_TO_WRITE: &str = "a write needs a link name, and this path holds none";

// H5F__super_read in src/H5Fsuper.c, HDF5 1.14.6, exempts initial SWMR reads from the EOF
// completeness check for superblock version 3 and later.
const SWMR_SUPERBLOCK_VERSION: SuperblockVersion = SuperblockVersion::Three;

#[cfg(test)]
mod tests;
