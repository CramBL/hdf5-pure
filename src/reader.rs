//! Reading API: File, Dataset, and Group handles for reading HDF5 files.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use crate::address::BaseAddress;
use crate::edit::{
    AppendBuilder, AppendGeometry, AppendTarget, EditBacking, MemoryStrategy, SpaceAccounting,
    StagedChild, StagedKind, StagedMeta, StagedObject, SyncPolicy, WriteEngine,
};
use crate::element::H5Element;
use crate::type_builders::{DatasetBuilder, VL_REF_SIZE};

use crate::appender::BufferedAppender;
use crate::attribute::{extract_attributes_full, extract_attributes_full_from_source};
use crate::chunk_cache::{CachePass, ChunkCache, ChunkCacheConfig, ChunkCacheStats};
use crate::compound::CompoundType;
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::{Datatype, ReferenceType};
use crate::error::{Error, FormatError};
use crate::file_create_properties::FileCreateProperties;
use crate::file_lock::{self, FileLocking, OpenIntent, OpenTarget, WriteMarkPolicy};
use crate::file_space_info::{FileSpaceInfo, FileSpaceStrategy};
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::free_space_manager;
use crate::group_v1::GroupEntry;
use crate::group_v2::{self, ChildLookup, is_group};
use crate::layout_info::{Chunk, ChunkIndex, Filter, Layout};
use crate::libver::LibVer;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::read_spec::RawReadSpec;
use crate::shared_message::{self, BufferedResolver, SharedResolver, SourceResolver};
use crate::signature;
use crate::source::{
    BaseOffsetSource, BytesSource, MetadataCacheConfig, MetadataCacheStats, MetadataCachingSource,
    ReadSeekSource, Source, ValidatedSource, frame,
};
use crate::superblock::Superblock;
use crate::vl_data::{self, VlenStringReadOptions};

use crate::types::{AttrValue, DType, attrs_to_map, classify_datatype};

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

/// File-access properties applied when opening an HDF5 file.
///
/// This is the `hdf5-pure` analogue of an HDF5 **file access property list**
/// (`fapl`): one value carrying every access-time setting, built once and passed
/// to whichever open a caller reaches for, exactly as a `fapl` is handed to
/// `H5Fopen`. Every `*_with_options` constructor on [`File`] accepts it, so a
/// read path and a read-write path can share one configuration.
///
/// The `Properties` suffix means the type stands in for one whole HDF5 property
/// list, so every setting on it has a C counterpart to look up. It is a stand-in
/// and not a port: a plain `Copy` value, with no handle to create or close, no
/// runtime property registry, and no setter that can fail. `fapl` and each
/// `H5Pset_*` it models are doc aliases, so a search for either lands here.
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
/// See the [property-support reference] for the full property-by-property map.
///
/// [property-support reference]: https://github.com/CramBL/hdf5-pure/blob/main/docs/reference/property-support.md
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
}

impl FileAccessProperties {
    /// A value carrying the crate's default access behavior.
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
    /// [`FormatError::LibverTooOldForContent`](crate::FormatError::LibverTooOldForContent)
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
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable).
    ///
    /// The read-only opens ignore this: they write nothing. [`File::open_swmr_writer`]
    /// requires a version 3 superblock, so it refuses a `high` below
    /// [`LibVer::V110`] up front rather than accepting a bound it cannot honor.
    #[doc(alias = "H5Pset_libver_bounds")]
    pub const fn with_libver_bounds(mut self, low: LibVer, high: LibVer) -> Self {
        self.libver_bounds = Some((low, high));
        self
    }

    /// Choose who owns this session's `fsync` cadence — this crate, or the
    /// application through [`File::sync`].
    ///
    /// Defaults to [`SyncPolicy::Always`]: every commit and every immediate
    /// [`Dataset::append`](crate::Dataset::append) forces its writes to durable
    /// storage before returning. [`SyncPolicy::OnClose`] issues no `fsync` at all,
    /// which is what the reference C library does; the writes still reach the
    /// operating system by the time the operation making them returns, so only
    /// power-loss durability moves to the caller. [`with_page_buffer_size`](Self::with_page_buffer_size),
    /// off by default, is the one setting that changes that — and it requires
    /// this policy.
    ///
    /// The read-only opens ignore this: they write nothing.
    pub const fn with_sync_policy(mut self, sync_policy: SyncPolicy) -> Self {
        self.sync_policy = sync_policy;
        self
    }

    /// Let a read-write session's writes accumulate in a page buffer of
    /// `bytes`, so repeated small updates landing in the same page cost one
    /// write rather than one each.
    ///
    /// Defaults to `0`, which is off — as `H5Pset_page_buffer_size` defaults to
    /// off — and leaves the gathering every read-write session already does: one
    /// write per dirty page per *ordering barrier*, so a commit or an append
    /// still reaches the operating system in full before it returns. What this
    /// buys on top is letting a dirty page survive those barriers, which is where
    /// a workload of many small appends into a few pages does most of its
    /// repeating. Measured on a paged file, 32 chunk appends into eight datasets
    /// followed by a commit: **188 writes with the default gathering and 5 with a
    /// page buffer**, of which two are the mark below going up and coming down.
    /// The appends issue nothing at all until the session ends.
    ///
    /// **It pays off over a long session, and costs on a short one.** The crash
    /// mark below is two `fsync`s per session whatever the session then does, so
    /// there is a break-even: measured on an Apple M1 Max (APFS) with 256-byte
    /// appends into eight datasets, 400 appends ran 0.75x — slower — 800 broke
    /// even, and 6,400 ran 1.64x. The ratio climbs with session length, because
    /// the same pages are re-dirtied more often, and narrows to about 1.1x once
    /// 64 KiB payloads rather than metadata churn dominate. One host's numbers,
    /// and the short end is noisy; re-measure on the one that matters with
    /// `cargo bench --bench hot_paths -- page_buffer`, which runs both sides of
    /// the crossing.
    ///
    /// # What it costs, and what pays for it
    ///
    /// Gathered writes go out in address order, and every publish point sits
    /// below the content it reaches, so all of them are issued first. A write
    /// that fails, or a process that dies, mid-flush can therefore leave a file
    /// whose superblock, dataset length or object header names bytes that never
    /// arrived — and two of those read back **clean**, as fill values or as a
    /// deleted object's data, with every checksum verifying. That is what a
    /// write-back page buffer is, rather than a fault in this one:
    /// `H5Pset_page_buffer_size` reorders the same way and makes no
    /// crash-consistency claim either.
    ///
    /// So this session raises superblock status-flag bit 0
    /// (`H5F_SUPER_WRITE_ACCESS`) for its whole life, `fsync`ed once at open and
    /// cleared on a clean [`File::close`] or drop — the mark the reference C
    /// library raises for *any* writer. A session that dies with pages in memory
    /// leaves that byte standing, and a file carrying it is refused by this
    /// crate, by `H5Fopen` and by h5py alike, with
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse). The silent
    /// wrong answer becomes a refusal, and
    /// [`File::clear_swmr_flag`](crate::File::clear_swmr_flag) — the `h5clear -s`
    /// equivalent — is how to look at such a file anyway, knowing what it may
    /// hold. A completed commit's bytes may also still be in this process's
    /// memory when it returns.
    ///
    /// # Refusals
    ///
    /// Four, each refused with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported) rather than
    /// quietly ignored:
    ///
    /// - a budget below the page the session merges within: the file's own
    ///   file-space page size when it was created with
    ///   [`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page), and the
    ///   format's 4 KiB default otherwise. A buffer that cannot hold one page
    ///   drains on every page it touches;
    /// - a **paged** file whose free space is not persisted, which can be neither
    ///   committed to nor appended to, so the buffer would hold nothing while its
    ///   mark blocked every reader;
    /// - a superblock older than version 3, whose status-flags byte no library
    ///   reads back, so the mark above would announce nothing;
    /// - [`SyncPolicy::Always`](crate::SyncPolicy::Always), the default, where
    ///   every barrier is an `fsync` that flushes the buffer on its way out — so
    ///   it would hold nothing while still costing the mark. Pair this with
    ///   [`with_sync_policy(SyncPolicy::OnClose)`](Self::with_sync_policy).
    ///
    /// [`File::create_with_options`] refuses a creation/access pair it could not
    /// then reopen with, rather than writing the file first, and
    /// [`File::open_swmr_writer`] refuses a page buffer outright: its readers
    /// observe the order its writes become visible in, which is exactly what a
    /// buffer coalesces away.
    ///
    /// # Choosing a budget
    ///
    /// Any budget of at least one page is honored. One below 1 MiB is an explicit
    /// request for less resident memory, not a mistake — a writer inside a tight
    /// memory cap can ask for 256 KiB and get it — and what it buys that memory
    /// with is writes: the budget is the point at which everything held is
    /// flushed, so a long contiguous run is flushed and restarted once per
    /// budget's worth of it. Writes issued on a 4 KiB-paged file:
    ///
    /// | workload | unset | 4 KiB | 64 KiB | 1 MiB |
    /// | --- | --- | --- | --- | --- |
    /// | 32 chunk appends into 8 datasets, then a commit | 188 | 25 | 4 | 4 |
    /// | one 4 MiB append | 131 | 1,094 | 74 | 10 |
    ///
    /// On the scattered workload this property exists for, a small budget costs
    /// little; on the long run 64 KiB is seven times the writes of 1 MiB.
    ///
    /// The memory comparison against leaving this unset is not the one the table
    /// suggests. A session that sets nothing already gathers up to 1 MiB of dirty
    /// bytes **per operation**, and releases it at every ordering barrier; a page
    /// buffer holds its budget **across** operations, until the budget is spent,
    /// an `fsync`, or [`File::close`]. So 1 MiB here trades a per-operation peak
    /// for a continuous residency of the same size, and a budget below 1 MiB
    /// lowers both.
    ///
    /// # How this differs from `H5Pset_page_buffer_size`
    ///
    /// **A paged file is not required, where the C library requires one.**
    /// `H5PB_create` refuses an unpaged file because the C page buffer is a page
    /// *cache*, and its `min_meta_perc` / `min_raw_perc` reservations are counted
    /// in pages that the paged allocator keeps segregated by kind. This is a
    /// write gatherer: it merges runs within a page-sized window and flushes
    /// whole, so a window is all it needs, and an unpaged file gets the same
    /// 4 KiB one that every read-write session already gathers under. Since
    /// unpaged is the default strategy, requiring `Page` put this property out of
    /// reach of most files for no reason this implementation had.
    ///
    /// **A small budget costs writes here, where it costs none in C.**
    /// `H5PB_write` sends any I/O of a page or more straight to the driver, so a
    /// small `page_buf_size` there caps memory without throttling a long write.
    /// Nothing bypasses this buffer — the budget is the point at which everything
    /// held is flushed — so a small one turns a single long run into repeated
    /// flushes. Both libraries accept the budget; only this one charges for it.
    /// See [Choosing a budget](#choosing-a-budget) for what it charges.
    ///
    /// **A sub-page budget is refused rather than rounded.** `H5Fopen` rounds it
    /// up to one page silently, and `H5Fcreate` refuses it. A property quietly
    /// ignored is worse than one refused.
    ///
    /// The read-only opens ignore this setting; they write nothing.
    ///
    /// Only the budget of `H5Pset_page_buffer_size` is modeled; its
    /// `min_meta_perc` / `min_raw_perc` reservations are not, since this buffer
    /// does not evict — it flushes whole.
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
    /// [`File::from_source`] all report
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse).
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
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose) that syncs at the
    /// points it wants readable, and it is what the mark exists to guard against
    /// when it is false: a page-buffered session's publish points are written
    /// before the content they name, so a snapshot taken mid-flush can show a
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
    /// [`File::clear_swmr_flag`] — the `h5clear -s` equivalent — is the recovery
    /// there, and it records the decision by clearing the byte.
    ///
    /// The C library offers no counterpart: `H5Fopen` refuses the byte with no
    /// override, and `h5clear` is its only way through. This is the narrower one,
    /// since it changes nothing on disk.
    pub const fn with_write_mark_policy(mut self, policy: WriteMarkPolicy) -> Self {
        self.write_mark_policy = policy;
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
/// The chunk cache is the one `dapl` property modeled; see the
/// [property-support reference] for the rest.
///
/// [`ChunkCacheConfig`] maps `H5Pset_chunk_cache`'s `rdcc_nslots` and
/// `rdcc_nbytes`; its `rdcc_w0` preemption policy is not modeled, for the reason
/// on [`ChunkCacheConfig::from_h5p_cache`].
///
/// Pass it to [`File::dataset_with_options`] or [`Group::dataset_with_options`].
///
/// [property-support reference]: https://github.com/CramBL/hdf5-pure/blob/main/docs/reference/property-support.md
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

/// Test whether a file looks like an HDF5 file, without reading it whole.
///
/// This is the spelling of the C library's `H5Fis_accessible` /
/// `H5Fis_hdf5`: it opens the file and scans only the 8-byte candidate windows
/// where the HDF5 signature is permitted (offsets 0, 512, 1024, 2048, …), so it
/// never buffers the whole file. Returns:
///
/// - `Ok(true)` — the HDF5 signature was found,
/// - `Ok(false)` — the file opened but has no HDF5 signature,
/// - `Err(..)` — the file could not be opened (missing, permissions, …).
///
/// It validates only the signature, not the rest of the format; a truncated or
/// corrupt file past the signature still reports `true`. Use [`File::open`] to
/// fully parse and validate.
pub fn is_hdf5<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<bool> {
    let handle = std::fs::File::open(path)?;
    let source = ReadSeekSource::new(handle).map_err(std::io::Error::other)?;
    match signature::find_signature_in(&source) {
        Ok(_) => Ok(true),
        Err(FormatError::SignatureNotFound) => Ok(false),
        Err(e) => Err(std::io::Error::other(e)),
    }
}

/// Test whether an in-memory buffer begins (at a permitted offset) with the
/// HDF5 signature. The buffer-backed counterpart of [`is_hdf5`].
pub fn is_hdf5_bytes(data: &[u8]) -> bool {
    signature::find_signature(data).is_ok()
}

/// An open HDF5 file for reading.
struct FileInner {
    backend: Backend,
    superblock: Superblock,
    /// Byte offset to add to all relative addresses (= original base_address).
    addr_offset: BaseAddress,
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
    /// Best-effort cleanup for a writer dropped without an explicit
    /// [`File::close`], running only when the last `Arc<FileInner>` clone drops;
    /// a clean `close` already did this work and set `closed`, so this is
    /// idempotent and skipped in that case.
    ///
    /// - A SWMR writer clears the superblock's SWMR-write flag (mirroring
    ///   `File::close`).
    /// - A read-write file that persists its free space rewrites its on-disk
    ///   free-space managers into canonical shape (issue #173), so a
    ///   dropped-without-`close` handle leaves the same file a clean `close`
    ///   would (a no-op unless an immediate append grew the file past them). A
    ///   true crash (`SIGKILL`, power loss) skips `drop` entirely; the appended
    ///   data is still durable, under the default
    ///   [`SyncPolicy::Always`](crate::SyncPolicy).
    ///
    /// Staged edits are *not* committed here: dropping a handle discards them,
    /// which is what `close` exists to distinguish.
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
    /// Open an HDF5 file from a filesystem path.
    ///
    /// Reads the file into memory once. To follow a file that a concurrent
    /// single writer is appending to (SWMR), use [`File::open_swmr`] instead.
    /// To read a file larger than memory (e.g. on a 32-bit host) without
    /// buffering it, use [`File::open_streaming`].
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
    /// whole-file buffer), [`persisted_free_space`](Self::persisted_free_space)
    /// returns no regions, a streaming file cannot be the *source* of a
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
        Self::streaming(source, properties, OpenTarget::Path(path.as_ref()))
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
        Self::streaming(ValidatedSource::new(source), properties, OpenTarget::Source)
    }

    /// The body both streaming opens share.
    ///
    /// Wraps the source in whatever metadata cache the properties ask for,
    /// parses the superblock through it, and refuses a file a writer holds.
    /// The two differ in where the source came from and in `target`, which does
    /// nothing but name the file in that refusal; one body is what keeps the
    /// rest of it from drifting apart.
    fn streaming<S: Source + Send + Sync + 'static>(
        source: S,
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
        let (superblock, addr_offset) = Self::parse_superblock_source(source.as_ref())?;
        file_lock::check_status_flags(
            &superblock,
            OpenIntent::Read(properties.write_mark_policy),
            target,
        )?;
        Ok(Self::from_parts(
            Backend::Streaming(source),
            superblock,
            addr_offset,
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
        let (superblock, addr_offset) = Self::parse_superblock(&data)?;
        file_lock::check_status_flags(
            &superblock,
            OpenIntent::SwmrRead,
            OpenTarget::Path(path.as_ref()),
        )?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            superblock,
            addr_offset,
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
        let (superblock, addr_offset) = Self::parse_superblock(&data)?;
        Ok(Self::from_parts(
            Backend::InMemory(data),
            superblock,
            addr_offset,
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
        // The engine parsed and normalized this at open; take it rather than
        // re-parsing, so the image need not be able to hand out a slice.
        let superblock = session.superblock().clone();
        let addr_offset = superblock.base_address;
        Ok(Self::from_parts(
            Backend::Edit(Box::new(Mutex::new(session))),
            superblock,
            addr_offset,
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

    /// Lock this file's write session for an operation that may change the
    /// file, and record afterwards that it had the chance to.
    ///
    /// Every path that can change the file's bytes — [`File::commit`] and the
    /// copies, every staged and immediate edit a [`Dataset`] or [`Group`] handle
    /// makes, and the session teardown — goes through here, so a new entry point
    /// cannot change the file without classifying what it did. Two write the
    /// file without passing here, both because no handle can be alive to see it:
    /// [`FileInner::drop`], which runs when the last `Arc` goes, and
    /// [`File::refresh`], which takes `&mut self` through `Arc::get_mut` and
    /// advances the counters itself.
    ///
    /// The appender's claim bookkeeping ([`Dataset::claim_for_appender`] and its
    /// pair) locks the engine directly and rightly notes nothing: it records who
    /// is appending, not what the file holds — and it must keep working from a
    /// `Drop` on a sealed file, which this gate refuses.
    ///
    /// The counters advance whether `f` succeeded or not, and for a staged edit
    /// that changes no bytes at all. Both are deliberate: a refused commit can
    /// still have written and rolled back (issues #316 and #344), and the cost
    /// of a revision that did not need advancing is one re-read on the next use
    /// of a handle, where the cost of one that needed advancing and did not is a
    /// wrong answer.
    fn with_engine_mut<R>(
        &self,
        change: Change,
        f: impl FnOnce(&mut WriteEngine) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let Backend::Edit(m) = &self.backend else {
            return Err(Error::ReadOnly);
        };
        self.check_mutable(matches!(change, Change::Relocating))?;
        let out = {
            let mut engine = m.lock().unwrap_or_else(PoisonError::into_inner);
            f(&mut engine)
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
    /// a live handle report [`Error::NotADataset`](crate::Error::NotADataset) or
    /// [`Error::NotAGroup`](crate::Error::NotAGroup) for an object whose kind
    /// never changed. That is the same staleness reporting itself instead of
    /// answering, which is the better half of the trade.
    fn locate(&self, path: Option<&str>, memo: Option<Resolution>) -> Result<Resolution, Error> {
        let revisions = self.revisions();
        // A handle with no memo has never resolved — it names an object this
        // session staged — so there is nothing to short-circuit on, and its path
        // is walked afresh every time until a commit gives it a header.
        Ok(revisions.at(match (path, memo) {
            (_, Some(memo)) if revisions.address == memo.address_revision => memo.address,
            (Some(path), _) => self.resolve_path(path)?,
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
        path: Option<&str>,
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
                let named = path.unwrap_or_default();
                match self.staged_object(named) {
                    Some(_) => Err(Error::NotCommitted(named.to_string())),
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
    fn staged_object(&self, path: &str) -> Option<StagedObject> {
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
    fn staged_standing(&self, path: &str, birth: Option<u64>) -> Standing {
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
        path: &str,
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
    fn staged_children(&self, parent: &str) -> StagedChildren {
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

    /// Parse the superblock from `data`, returning it (with `root_group_address`
    /// normalized to an absolute offset) and the base-address offset.
    fn parse_superblock(data: &[u8]) -> Result<(Superblock, BaseAddress), Error> {
        let sig_offset = signature::find_signature(data)?;
        let mut superblock = Superblock::parse(data, sig_offset)?;
        let addr_offset = superblock.base_address;
        // Normalize root_group_address to absolute so resolve_path_any works.
        superblock.root_group_address = addr_offset.absolute(superblock.root_group_address)?;
        Ok((superblock, addr_offset))
    }

    /// Streaming counterpart of [`parse_superblock`]: locate and parse the
    /// superblock by reading only small windows from the source.
    fn parse_superblock_source<S: Source + ?Sized>(
        source: &S,
    ) -> Result<(Superblock, BaseAddress), Error> {
        let sig_offset = signature::find_signature_in(source)?;
        let mut superblock = Superblock::parse_from_source(source, sig_offset)?;
        let addr_offset = superblock.base_address;
        superblock.root_group_address = addr_offset.absolute(superblock.root_group_address)?;
        Ok((superblock, addr_offset))
    }

    /// Assemble a [`File`] from parsed parts, then load the File Space Info from
    /// the superblock extension (best-effort, so a bad extension never fails the
    /// open).
    fn from_parts(
        backend: Backend,
        superblock: Superblock,
        addr_offset: BaseAddress,
        handle: Option<std::fs::File>,
        access_properties: FileAccessProperties,
    ) -> Self {
        let mut file = FileInner {
            backend,
            superblock,
            addr_offset,
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
        let rel = self.superblock.superblock_extension_address?;
        if rel == u64::MAX {
            return None;
        }
        let abs = self.addr_offset.absolute(rel).ok()?;
        let header = self.parse_header(abs).ok()?;
        let msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FileSpaceInfo)?;
        FileSpaceInfo::parse(
            &msg.data,
            self.superblock.offset_size,
            self.superblock.length_size,
        )
        .ok()
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
        if rel == u64::MAX {
            return None;
        }
        let abs = self.addr_offset.absolute(rel).ok()?;
        let header = self.parse_header(abs).ok()?;
        let msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SharedMessageTable)?;
        let message =
            crate::sohm::SharedMessageTableMessage::parse(&msg.data, self.superblock.offset_size)
                .ok()?;
        // The table's address, like every address in a header message, is stored
        // relative to the file's base address, so the read is framed the same way
        // every other metadata walk here is.
        let base = self.addr_offset;
        let os = self.superblock.offset_size;
        match &self.backend {
            Backend::InMemory(v) => {
                crate::sohm::SohmTable::read(frame(v, base).ok()?, &message, os).ok()
            }
            Backend::Streaming(s) if base.is_zero() => {
                crate::sohm::SohmTable::read_from_source(s.as_ref(), &message, os).ok()
            }
            Backend::Streaming(s) => crate::sohm::SohmTable::read_from_source(
                &BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                },
                &message,
                os,
            )
            .ok(),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| Ok::<_, Error>(crate::sohm::SohmTable::read(frame(d, base)?, &message, os)?),
                |s| {
                    if base.is_zero() {
                        Ok(crate::sohm::SohmTable::read_from_source(s, &message, os)?)
                    } else {
                        Ok(crate::sohm::SohmTable::read_from_source(
                            &BaseOffsetSource { inner: s, base },
                            &message,
                            os,
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
            match Self::parse_superblock(&data) {
                Ok((superblock, addr_offset)) => {
                    self.backend = Backend::InMemory(data);
                    self.superblock = superblock;
                    self.addr_offset = addr_offset;
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

    /// Resolve a path to an object-header address, dispatching on the backend.
    fn resolve_path(&self, path: &str) -> Result<u64, Error> {
        Ok(match &self.backend {
            Backend::InMemory(v) => group_v2::resolve_path_any(v, &self.superblock, path)?,
            Backend::Streaming(s) => {
                group_v2::resolve_path_any_from_source(s.as_ref(), &self.superblock, path)?
            }
            // A staged commit can relocate the object tree's root, so this
            // file's cached superblock may name a stale one; resolve against the
            // session's own superblock, which the commit updates.
            Backend::Edit(m) => {
                let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let sb = core.superblock().clone();
                match core.image_slice() {
                    Some(data) => group_v2::resolve_path_any(data, &sb, path)?,
                    None => group_v2::resolve_path_any_from_source(core.image(), &sb, path)?,
                }
            }
        })
    }

    /// The current root-group address (base-adjusted, absolute). For a read-write
    /// [`Backend::Edit`] file a prior relocating commit can have moved the
    /// root, so take the session's own superblock, which the commit updates;
    /// other backends use this file's cached one.
    fn mirror_root_address(&self) -> u64 {
        if let Backend::Edit(m) = &self.backend {
            let core = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            return core.superblock().root_group_address;
        }
        self.superblock.root_group_address
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

    /// Return the access properties used when opening this file.
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

    /// The base address (`H5F` superblock base address), i.e. the byte offset
    /// added to every stored relative address. Zero for a file with no
    /// userblock.
    pub(crate) fn base_address(&self) -> BaseAddress {
        self.addr_offset
    }

    /// The file-space management strategy this file records in its superblock
    /// extension (set with `H5Pset_file_space_strategy`), or `None` if the file
    /// records none — the default, which the C library also writes as "no
    /// message". See [`file_space_info`](Self::file_space_info) for the full
    /// record (persist flag, threshold, page size).
    pub fn file_space_strategy(&self) -> Option<FileSpaceStrategy> {
        self.file_space_info.as_ref().map(|info| info.strategy)
    }

    /// The full [`FileSpaceInfo`] recorded in this file's superblock extension,
    /// if present and readable.
    pub fn file_space_info(&self) -> Option<&FileSpaceInfo> {
        self.file_space_info.as_ref()
    }

    /// The free regions a file persists on disk in its free-space managers (when
    /// written with `H5Pset_file_space_strategy(..., persist = true)`), as
    /// `(address, length)` pairs sorted by address.
    ///
    /// Empty when the file does not persist free space, or for the streaming
    /// backend (which does not load the manager blocks). The addresses are file
    /// offsets (relative to the base address); reading data is unaffected by the
    /// presence or absence of these managers.
    pub fn persisted_free_space(&self) -> Vec<(u64, u64)> {
        let Some(info) = &self.file_space_info else {
            return Vec::new();
        };
        if !info.persist {
            return Vec::new();
        }
        let Backend::InMemory(data) = &self.backend else {
            return Vec::new();
        };
        let mut sections = free_space_manager::read_persisted_sections(
            data,
            &info.manager_addrs,
            self.addr_offset,
            self.superblock.offset_size,
        )
        .unwrap_or_default();
        // Distinct sections have distinct addresses in any well-formed file, so
        // the tie-break never arises; only a malformed manager can advertise one
        // address twice, and which of the pair is reported first is already
        // unspecified. No `debug_assert` here: this parses untrusted bytes, which
        // must not panic a debug build.
        sections.sort_unstable_by_key(|s| s.addr);
        sections.into_iter().map(|s| (s.addr, s.size)).collect()
    }

    /// The size of the underlying file in bytes (the HDF5 `H5Fget_filesize`).
    ///
    /// This is the total byte length of the backing store — for a streaming
    /// file the length reported by its source, for an in-memory file the length
    /// of its buffer. It includes any userblock prefix and trailing bytes, so it
    /// may exceed the superblock's logical end-of-file address; compare against
    /// `Superblock::eof_address` (reachable via
    /// [`File::superblock`]) to detect appended or unaccounted tail bytes.
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
        LibVer::from_superblock_version(self.superblock.version)
    }

    fn parse_header(&self, address: u64) -> Result<ObjectHeader, FormatError> {
        let os = self.superblock.offset_size;
        let ls = self.superblock.length_size;
        match &self.backend {
            Backend::InMemory(v) => {
                ObjectHeader::parse_with_base(v, address.to_usize()?, os, ls, self.addr_offset)
            }
            Backend::Streaming(s) => {
                ObjectHeader::parse_from_source(s.as_ref(), address, os, ls, self.addr_offset)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| ObjectHeader::parse_with_base(d, address.to_usize()?, os, ls, self.addr_offset),
                |s| ObjectHeader::parse_from_source(s, address, os, ls, self.addr_offset),
            ),
        }
    }

    /// Resolve a base-relative object-header address (the value stored in an
    /// HDF5 `H5R_OBJECT` reference element) to the [`Object`] it points at.
    ///
    /// The stored address is relative to the superblock base address, so any
    /// MAT-file userblock is accounted for here. A null (`0`) or undefined
    /// (`HADDR_UNDEF`) address, or one whose object header is neither a dataset
    /// nor a group, yields [`FormatError::InvalidObjectReference`].
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
            .addr_offset
            .absolute(rel_addr)
            .map_err(|_| FormatError::InvalidObjectReference(rel_addr))?;
        let at = revisions.at(abs);
        let hdr = file.parse_header(abs)?;
        if has_message(&hdr, MessageType::DataLayout) {
            let chunk_cache = DatasetAccessProperties::new()
                .resolved_chunk_cache(file.access_properties.chunk_cache);
            Ok(Object::Dataset(Box::new(Dataset::new(
                file.clone(),
                at,
                hdr,
                chunk_cache,
                None,
            ))))
        } else if is_group(&hdr) {
            Ok(Object::Group(Group::new(file.clone(), at, None)))
        } else {
            Err(FormatError::InvalidObjectReference(rel_addr).into())
        }
    }

    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }

    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }

    /// Resolve the children of a group object header, dispatching on the backend
    /// and converting link addresses to absolute.
    fn group_children(&self, hdr: &ObjectHeader) -> Result<Vec<GroupEntry>, Error> {
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let mut entries = match &self.backend {
            Backend::InMemory(v) => group_v2::resolve_group_entries(v, hdr, os, ls, base),
            Backend::Streaming(s) => {
                group_v2::resolve_group_entries_from_source(s.as_ref(), hdr, os, ls, base)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| group_v2::resolve_group_entries(d, hdr, os, ls, base),
                |s| group_v2::resolve_group_entries_from_source(s, hdr, os, ls, base),
            ),
        }
        .map_err(Error::Format)?;
        for entry in &mut entries {
            // The stored address is relative to the base address; normalize to an
            // absolute file offset. A crafted entry (e.g. the HADDR_UNDEF sentinel)
            // must not wrap or panic.
            entry.object_header_address = base.absolute(entry.object_header_address)?;
        }
        Ok(entries)
    }

    /// The child named `name`, at the absolute file address
    /// [`ChildLookup::Found`] carries.
    ///
    /// The by-name counterpart of [`group_children`](Self::group_children), and
    /// the one to reach for when a single child is wanted: it stops at the match
    /// rather than building an entry, and an owned name, for every other child
    /// of the group (issue #228).
    fn group_child(&self, group_address: u64, name: &str) -> Result<ChildLookup, Error> {
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let addr = group_address;
        match &self.backend {
            Backend::InMemory(v) => group_v2::find_child_address(v, addr, os, ls, base, name),
            Backend::Streaming(s) => {
                group_v2::find_child_address_from_source(s.as_ref(), addr, os, ls, base, name)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| group_v2::find_child_address(d, addr, os, ls, base, name),
                |s| group_v2::find_child_address_from_source(s, addr, os, ls, base, name),
            ),
        }
        .map_err(Error::Format)
    }

    /// Read all attributes attached to an object header, dispatching on the
    /// backend.
    fn attrs_of(&self, hdr: &ObjectHeader) -> Result<HashMap<String, AttrValue>, Error> {
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let attr_msgs = self.attr_messages_of(hdr)?;
        match &self.backend {
            Backend::Edit(m) => Ok(Self::with_engine(
                m,
                |d| attrs_to_map(&attr_msgs, &BytesSource::new(d), os, ls, base),
                |s| attrs_to_map(&attr_msgs, s, os, ls, base),
            )),
            _ => Ok(attrs_to_map(&attr_msgs, &self.source(), os, ls, base)),
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
        if !shared_message::is_shared(msg.flags) {
            return Ok(Cow::Borrowed(&msg.data));
        }
        let (os, ls, base) = (self.offset_size(), self.length_size(), self.addr_offset);
        let sohm = self.sohm_table.as_deref();
        // A shared reference stores its address relative to the base address, so
        // frame the file at `base` exactly as [`Self::attr_messages_of`] does.
        let resolved = match &self.backend {
            Backend::InMemory(v) => BufferedResolver::new(frame(v, base)?, os, ls, sohm)
                .resolve(&msg.data, msg.msg_type),
            Backend::Streaming(s) if base.is_zero() => {
                SourceResolver::new(s.as_ref(), os, ls, sohm).resolve(&msg.data, msg.msg_type)
            }
            Backend::Streaming(s) => SourceResolver::new(
                &BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                },
                os,
                ls,
                sohm,
            )
            .resolve(&msg.data, msg.msg_type),
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| {
                    BufferedResolver::new(frame(d, base)?, os, ls, sohm)
                        .resolve(&msg.data, msg.msg_type)
                },
                |s| {
                    if base.is_zero() {
                        SourceResolver::new(s, os, ls, sohm).resolve(&msg.data, msg.msg_type)
                    } else {
                        SourceResolver::new(&BaseOffsetSource { inner: s, base }, os, ls, sohm)
                            .resolve(&msg.data, msg.msg_type)
                    }
                },
            ),
        }?;
        Ok(Cow::Owned(resolved))
    }

    /// The object-header address a *shared* header message names, or `None` when
    /// the record carries its own content or names the shared-message heap.
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
    ) -> Result<Option<u64>, Error> {
        if !shared_message::is_shared(msg.flags) {
            return Ok(None);
        }
        let reference =
            shared_message::parse_shared_ref(&msg.data, self.offset_size(), self.length_size())?;
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
        let (os, ls) = (self.offset_size(), self.length_size());
        // Compact attributes come out of `hdr`, but the two addresses this walk
        // follows are read from message bodies and so are stored relative to the
        // base address: the Attribute Info message's fractal-heap address, and a
        // shared attribute's message address. Frame the file at `base` exactly as
        // [`Self::read_dataset_raw`] does, so both index it directly. For a plain
        // file (`base == 0`) this is the identity; without it, a userblock file's
        // dense attributes are looked for one userblock too early.
        let base = self.addr_offset;
        let sohm = self.sohm_table.as_deref();
        match &self.backend {
            Backend::InMemory(v) => {
                Ok(extract_attributes_full(frame(v, base)?, hdr, os, ls, sohm)?)
            }
            Backend::Streaming(s) if base.is_zero() => Ok(extract_attributes_full_from_source(
                s.as_ref(),
                hdr,
                os,
                ls,
                sohm,
            )?),
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                Ok(extract_attributes_full_from_source(
                    &framed, hdr, os, ls, sohm,
                )?)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |d| Ok(extract_attributes_full(frame(d, base)?, hdr, os, ls, sohm)?),
                |s| {
                    if base.is_zero() {
                        Ok(extract_attributes_full_from_source(s, hdr, os, ls, sohm)?)
                    } else {
                        let framed = BaseOffsetSource { inner: s, base };
                        Ok(extract_attributes_full_from_source(
                            &framed, hdr, os, ls, sohm,
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
        let (os, ls) = (self.offset_size(), self.length_size());
        // Every on-disk address in `dl` — the contiguous data address, the chunk
        // index root, and (followed deeper in the chunked reader) every B-tree /
        // fixed-array / extensible-array node and chunk-data address — is stored
        // relative to the base address. Present the payload reader a base-relative
        // view of the file so all of them index it directly: slice the in-memory
        // buffer at `base`, or wrap the streaming source to add `base` to each
        // read. For a plain file (`base == 0`) this is the identity.
        let base = self.addr_offset;
        match &self.backend {
            Backend::InMemory(v) => {
                data_read::read_raw_data_cached(frame(v, base)?, spec, os, ls, cache)
            }
            Backend::Streaming(s) if base.is_zero() => {
                data_read::read_raw_data_cached_from_source(s.as_ref(), spec, os, ls, cache)
            }
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                data_read::read_raw_data_cached_from_source(&framed, spec, os, ls, cache)
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |data| {
                    let framed = frame(data, base)?;
                    data_read::read_raw_data_cached(framed, spec, os, ls, cache)
                },
                |s| {
                    let framed = BaseOffsetSource { inner: s, base };
                    data_read::read_raw_data_cached_from_source(&framed, spec, os, ls, cache)
                },
            ),
        }
    }

    /// Windowed counterpart of [`read_dataset_raw`](Self::read_dataset_raw): read
    /// the raw element bytes of the row window `[start_row, start_row + num_rows)`,
    /// touching only the storage it overlaps. Reads through the same base-framed
    /// `Source`, so on-disk addresses resolve the same way. The caller clamps
    /// the window to the dataset.
    fn read_dataset_raw_rows(
        &self,
        spec: RawReadSpec<'_>,
        cache: &ChunkCache,
        pass: CachePass,
        start_row: u64,
        num_rows: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let (os, ls) = (self.offset_size(), self.length_size());
        let (dl, ds, dt) = (spec.layout, spec.dataspace, spec.datatype);
        let elem_size = dt.element_size_usize()?;
        // Elements per row (product of inner dims; 1 when 0-D or 1-D). Checked so
        // a crafted dataspace whose inner dims overflow `usize` errors instead of
        // panicking (debug) or wrapping (release).
        let row_elems: usize = ds.dimensions.iter().skip(1).try_fold(1usize, |acc, &d| {
            acc.checked_mul(d.to_usize()?)
                .ok_or(FormatError::OffsetOverflow {
                    offset: acc as u64,
                    length: d,
                })
        })?;
        let row_bytes =
            row_elems
                .checked_mul(elem_size.get())
                .ok_or(FormatError::OffsetOverflow {
                    offset: row_elems as u64,
                    length: elem_size.get() as u64,
                })?;

        // Compact data is inline in the layout message — no I/O, no framing.
        if let DataLayout::Compact { data } = dl {
            let start = start_row.to_usize()?.checked_mul(row_bytes);
            let len = num_rows.to_usize()?.checked_mul(row_bytes);
            let (Some(start), Some(len)) = (start, len) else {
                return Err(FormatError::OffsetOverflow {
                    offset: start_row,
                    length: row_bytes as u64,
                });
            };
            let end = start.checked_add(len).ok_or(FormatError::OffsetOverflow {
                offset: start as u64,
                length: len as u64,
            })?;
            return data
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or(FormatError::DataSizeMismatch {
                    expected: end,
                    actual: data.len(),
                });
        }

        let base = self.addr_offset;
        match &self.backend {
            Backend::InMemory(v) => {
                let framed = frame(v, base)?;
                read_rows_framed(
                    &BytesSource::new(framed),
                    spec,
                    os,
                    ls,
                    cache,
                    pass,
                    start_row,
                    num_rows,
                    row_bytes,
                )
            }
            Backend::Streaming(s) if base.is_zero() => read_rows_framed(
                s.as_ref(),
                spec,
                os,
                ls,
                cache,
                pass,
                start_row,
                num_rows,
                row_bytes,
            ),
            Backend::Streaming(s) => {
                let framed = BaseOffsetSource {
                    inner: s.as_ref(),
                    base,
                };
                read_rows_framed(
                    &framed, spec, os, ls, cache, pass, start_row, num_rows, row_bytes,
                )
            }
            Backend::Edit(m) => Self::with_engine(
                m,
                |data| {
                    let framed = frame(data, base)?;
                    read_rows_framed(
                        &BytesSource::new(framed),
                        spec,
                        os,
                        ls,
                        cache,
                        pass,
                        start_row,
                        num_rows,
                        row_bytes,
                    )
                },
                |s| {
                    let framed = BaseOffsetSource { inner: s, base };
                    read_rows_framed(
                        &framed, spec, os, ls, cache, pass, start_row, num_rows, row_bytes,
                    )
                },
            ),
        }
    }
}

/// Read a row window through an already base-framed `Source`. Contiguous
/// layouts are one bounded sub-read; chunked layouts use the windowed chunk
/// reader (only the rank-0 crafted-file corner falls back to a whole read
/// plus slice).
fn read_rows_framed<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    os: u8,
    ls: u8,
    cache: &ChunkCache,
    pass: CachePass,
    start_row: u64,
    num_rows: u64,
    row_bytes: usize,
) -> Result<Vec<u8>, FormatError> {
    let (dl, fill) = (spec.layout, spec.fill);
    // A zero-row window reads nothing, uniformly across the *supported* layouts.
    // A `Virtual` layout is unsupported and must still error like `read_raw`
    // does, so it is excluded here and falls through to the match.
    if num_rows == 0 && !matches!(dl, DataLayout::Virtual { .. }) {
        return Ok(Vec::new());
    }
    match dl {
        DataLayout::Compact { .. } => unreachable!("compact is handled before framing"),
        DataLayout::Contiguous { address, size } => {
            // Unallocated storage: the window reads as the fill value, the same
            // answer the whole-dataset readers give for it.
            let Some(addr) = *address else {
                let len = num_rows.to_usize()?.saturating_mul(row_bytes);
                return fill.buffer(len);
            };
            let start =
                start_row
                    .checked_mul(row_bytes as u64)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: start_row,
                        length: row_bytes as u64,
                    })?;
            let len =
                num_rows
                    .to_usize()?
                    .checked_mul(row_bytes)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: num_rows,
                        length: row_bytes as u64,
                    })?;
            // Never read past the dataset's own contiguous storage.
            if start.saturating_add(len as u64) > *size {
                return Err(FormatError::DataSizeMismatch {
                    expected: start.to_usize()?.saturating_add(len),
                    actual: (*size).to_usize()?,
                });
            }
            let off = addr.checked_add(start).ok_or(FormatError::OffsetOverflow {
                offset: addr,
                length: start,
            })?;
            source.read_exact_at(off, len)
        }
        DataLayout::Chunked { .. } => {
            match crate::chunked_read::read_chunked_rows_from_source(
                source, spec, os, ls, cache, pass, start_row, num_rows,
            )? {
                Some(bytes) => Ok(bytes),
                // Rank-0 chunked (a crafted-file corner): fall back to a whole
                // read, then slice.
                None => {
                    let full =
                        data_read::read_raw_data_cached_from_source(source, spec, os, ls, cache)?;
                    let start = start_row.to_usize()? * row_bytes;
                    let len = num_rows.to_usize()? * row_bytes;
                    full.get(start..start + len).map(<[u8]>::to_vec).ok_or(
                        FormatError::DataSizeMismatch {
                            expected: start + len,
                            actual: full.len(),
                        },
                    )
                }
            }
        }
        DataLayout::Virtual { .. } => Err(FormatError::UnsupportedVirtualLayout),
    }
}

impl std::fmt::Debug for FileInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("File")
            .field("size", &self.file_size())
            .field("superblock_version", &self.superblock.version)
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
    /// Open an HDF5 file from a filesystem path.
    ///
    /// Reads the file into memory once. To follow a file that a concurrent
    /// single writer is appending to (SWMR), use [`File::open_swmr`] instead.
    /// To read a file larger than memory (e.g. on a 32-bit host) without
    /// buffering it, use [`File::open_streaming`].
    ///
    /// A file whose superblock marks it as held by a writer is refused with
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse) — the check
    /// `H5Fopen` makes of the same byte. That means a live writer or one that
    /// exited without closing the file; clear a stale flag with
    /// [`clear_swmr_flag`](Self::clear_swmr_flag), and follow a live SWMR writer
    /// with [`open_swmr`](Self::open_swmr). [`from_bytes`](Self::from_bytes) does
    /// not check, since its caller already holds the bytes — which is also the
    /// way to read a flagged file on a read-only mount, where clearing the flag
    /// would need write access.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open(path)?),
        })
    }

    /// Open an HDF5 file from a filesystem path with explicit access properties.
    pub fn open_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_with_options(path, properties)?),
        })
    }

    /// Open an HDF5 file for **streaming** reads, fetching regions on demand from
    /// the file instead of buffering it whole.
    ///
    /// This lets a host read a file larger than its address space. Metadata and
    /// dataset chunks are read through a `ReadSeekSource`, so peak memory stays
    /// close to one chunk plus the metadata being parsed; chunks adjacent on
    /// disk are fetched together, in reads of at most 256 KiB, and a chunk
    /// larger than that is read on its own. Attribute reading and v1
    /// symbol-table groups on the resolved path are not yet supported on this
    /// backend.
    ///
    /// Like [`open`](Self::open), this refuses a file whose superblock marks it
    /// as held by a writer.
    pub fn open_streaming<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming(path)?),
        })
    }

    /// Open an HDF5 file for streaming reads with explicit access properties.
    pub fn open_streaming_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_streaming_with_options(path, properties)?),
        })
    }

    /// Open an HDF5 file from any [`Source`], reading metadata and chunks on
    /// demand exactly as [`open_streaming`](Self::open_streaming) does.
    ///
    /// This is the streaming open for a caller whose bytes are not a path: an
    /// object store addressed by HTTP range request, a sandboxed guest that
    /// receives byte ranges from its host, a decrypting layer. A [`Source`]
    /// supplies a length and reads at an absolute offset, which is all the
    /// reader asks of a file, so peak memory stays at the metadata being parsed
    /// plus the chunks a read touches — not the file. Wrap a `Read + Seek` in
    /// [`ReadSeekSource`] rather than writing that impl again.
    ///
    /// A file marked as held by a writer is refused here as it is by
    /// [`open_streaming`](Self::open_streaming); with no path to report, the
    /// error names the source instead. Recovering such a file in place needs a
    /// path, through [`File::clear_swmr_flag`], so a caller that has none is
    /// left with [`File::from_bytes`](Self::from_bytes) and the whole file in
    /// memory.
    ///
    /// The metadata cache is **off** unless
    /// [`from_source_with_options`](Self::from_source_with_options) turns it
    /// on, which matters more here than it does for a local file: without one,
    /// every read a parser makes is a round trip. See
    /// [`MetadataCacheConfig`].
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

    /// Open an HDF5 file from any [`Source`] with explicit access properties.
    ///
    /// The metadata cache is what a remote source wants tuned: every parser
    /// read becomes a round trip without one. See [`MetadataCacheConfig`].
    pub fn from_source_with_options<S: Source + Send + Sync + 'static>(
        source: S,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_source_with_options(source, properties)?),
        })
    }

    /// Open an HDF5 file for SWMR (single-writer/multiple-reader) reading.
    ///
    /// Like [`File::open`], but retains a live handle to the file so that
    /// [`File::refresh`] can re-read data appended by a concurrent writer.
    ///
    /// This is the open that *follows* a file marked as held by a SWMR writer,
    /// where [`open`](Self::open) refuses one. Only a half-set mark is refused
    /// here, with [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse):
    /// either bit without the other. Write access alone is what a plain
    /// (non-SWMR) writer leaves, and there is no protocol for following a writer
    /// that is not publishing consistent prefixes; the SWMR bit alone is a state
    /// no writer produces. Both bits is the live SWMR writer this exists to
    /// follow, and neither is a quiescent file.
    #[doc(alias = "H5F_ACC_SWMR_READ")]
    pub fn open_swmr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr(path)?),
        })
    }

    /// Open an HDF5 file for SWMR reading with explicit access properties.
    pub fn open_swmr_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_swmr_with_options(path, properties)?),
        })
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes(data)?),
        })
    }

    /// Open an HDF5 file from an in-memory byte vector with explicit access properties.
    pub fn from_bytes_with_options(
        data: Vec<u8>,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::from_bytes_with_options(data, properties)?),
        })
    }

    /// Open an existing HDF5 file for reading **and** in-place editing.
    ///
    /// Unlike [`open`](Self::open) (read-only, buffered), this takes an exclusive
    /// OS file lock held for the file's life and lets owned handles modify the
    /// file — immediate [`Dataset::append`]s, plus [`Dataset::write`]/`set_attr`,
    /// [`Group::create_dataset`]/`create_group`/`delete`/`set_attr`, and
    /// [`copy`](Self::copy)/[`copy_from`](Self::copy_from) staged until
    /// [`commit`](Self::commit). The file must use 8-byte offsets and lengths and
    /// keep its superblock at its base address (a canonical userblock, as in a
    /// MATLAB `.mat` file, is supported); anything else is refused with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported).
    ///
    /// The fast immediate [`Dataset::append`] additionally requires a
    /// latest-format (version-2/3) file with no userblock and an
    /// Extensible-Array-indexed dataset; [`Dataset::append_staged`] covers the
    /// general case.
    ///
    /// Two things can turn this open away because another writer holds the file:
    /// the exclusive OS lock, reported as
    /// [`Error::FileLocked`](crate::Error::FileLocked), and the superblock's
    /// status-flags byte, reported as
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse). The second
    /// covers what the first cannot — a SWMR writer takes no lock, and a writer
    /// that exited without closing the file leaves the flag behind; recover a
    /// stale one with [`clear_swmr_flag`](Self::clear_swmr_flag).
    ///
    /// # Memory
    ///
    /// This picks its backing from the file rather than making the caller pick a
    /// function (issue #198): a latest-format file with no userblock is edited
    /// **bounded**, holding only the metadata being parsed plus the configured
    /// caches plus what an edit is building, so resident memory does not scale
    /// with the file; anything else falls back to a whole-file in-memory mirror,
    /// which is what makes a pre-v2 or userblock file editable at all. The two
    /// backings are the same engine over different storage and offer the same
    /// edit surface, differing in one trade: the bounded one applies a large
    /// immediate append in whole-chunk batches, each crash-atomic on its own, so
    /// a crash mid-call leaves a valid shorter dataset rather than none of the
    /// append. Ask a file which it got with
    /// [`edit_backing`](Self::edit_backing), and demand one with
    /// [`FileAccessProperties::with_memory_strategy`] —
    /// [`MemoryStrategy::Mirrored`] restores the unconditional mirror this
    /// entry point used before it learned to dispatch.
    #[doc(alias = "H5Fopen")]
    pub fn open_rw<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        Self::open_rw_with_options(path, FileAccessProperties::new())
    }

    /// Open an existing file for reading and in-place editing with explicit
    /// access properties — see [`open_rw`](Self::open_rw).
    ///
    /// The properties carry the locking policy (the `H5Pset_file_locking` analogue,
    /// [`FileAccessProperties::with_locking`]), the memory strategy
    /// ([`FileAccessProperties::with_memory_strategy`], which overrides the
    /// dispatch described on [`open_rw`](Self::open_rw)), the `fsync` cadence
    /// ([`FileAccessProperties::with_sync_policy`]), the metadata cache used by
    /// the bounded backing, and the file-wide chunk-cache default applied to
    /// datasets opened from this file. Because one [`FileAccessProperties`] value
    /// serves every open, the same configuration can be shared with a read path.
    pub fn open_rw_with_options<P: AsRef<std::path::Path>>(
        path: P,
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        Ok(File {
            inner: Arc::new(FileInner::open_rw(path, properties)?),
        })
    }

    /// Open exactly as [`open_rw`](Self::open_rw) does, but behind an image that
    /// withholds its whole-file slice, so every read takes the `Source` path
    /// rather than the slice fast path.
    ///
    /// Each read this file serves has two forms (see `with_engine`), and only
    /// the slice form runs in production until a mirrorless backing lands
    /// (issue #198). Opening the same file both ways and comparing is what
    /// holds the other form to the same answers in the meantime.
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
    /// append returns [`Error::SwmrAppendUnsupported`](crate::Error::SwmrAppendUnsupported).
    /// The staged edit surface (`write`/`set_attr`/`create_*`/`delete`/`copy`/
    /// `commit`) returns
    /// [`Error::SwmrStagedUnsupported`](crate::Error::SwmrStagedUnsupported).
    /// [`close`](Self::close) clears the SWMR-write flag; a writer that exits
    /// without a clean close leaves it set — recover with
    /// [`clear_swmr_flag`](Self::clear_swmr_flag). While the flag stands, this
    /// open is refused with
    /// [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse), which is what
    /// keeps a second writer off a file SWMR gives only one (no OS lock is held
    /// to do it).
    ///
    /// Requires a latest-format (version-3 superblock) file with no userblock
    /// and no persisted free-space; other files are refused with
    /// [`Error::SwmrAppendUnsupported`](crate::Error::SwmrAppendUnsupported).
    /// The version-3 requirement is the C library's: neither library reads the
    /// SWMR-write flag back on an older superblock, so raising one there would
    /// announce the writer to nobody.
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
    /// Its [`SyncPolicy`](crate::SyncPolicy) applies here as to any other
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

    /// Clear a stale status flag left in `path` by a writer that exited without a
    /// clean [`close`](Self::close) — the `h5clear -s` equivalent, for recovering
    /// a file that both this crate and the reference C library otherwise refuse
    /// to open ([`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse)). A
    /// no-op if the flag is already clear.
    ///
    /// It takes the exclusive OS lock first, so it cannot clear the flag out
    /// from under a *live* [`open_rw`](Self::open_rw) writer. A live SWMR writer
    /// holds no lock, so make sure it is really gone: clearing the flag under
    /// one leaves its readers with no record that it is publishing.
    ///
    /// It also clears the crash mark a page-buffered session raises
    /// ([`FileAccessProperties::with_page_buffer_size`]), and there the warning is
    /// sharper. That mark stands for pages that were still in memory, so a file
    /// still carrying it was left by a writer that did not finish: clearing it
    /// hands back a file whose datasets may read clean and return fill values or
    /// a deleted object's bytes, with every checksum verifying. Clear it to
    /// salvage what is there, not to resume trusting it. `h5clear` makes the same
    /// trade for the same reason.
    pub fn clear_swmr_flag<P: AsRef<std::path::Path>>(path: P) -> Result<(), Error> {
        crate::file_lock::clear_swmr_flag_at(path.as_ref())
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
    /// built. A file created with [`FileSpaceStrategy::Page`] can be grown
    /// through either editor, by an immediate [`Dataset::append`] or a staged
    /// commit, provided it also persists its free space (issue #198).
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

    /// Apply all staged structural edits made through this file's handles —
    /// [`Dataset::write`]/`set_attr`/`remove_attr` and
    /// [`Group::create_group`]/`delete` — as one transaction. Immediate
    /// [`Dataset::append`]s need no commit.
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    ///
    /// Outstanding [`Dataset`] and [`Group`] handles stay usable: a commit
    /// relocates object headers, and each handle looks its object up again by
    /// path on its first use afterwards, so a long-lived handle answers for the
    /// file the commit left rather than for the copy it moved away from. Two
    /// exceptions, both of which report rather than answer wrongly. A *read*
    /// through a handle onto an object the commit deleted — or replaced with one
    /// of a different kind, which is
    /// [`Error::NotADataset`](crate::Error::NotADataset) or
    /// [`Error::NotAGroup`](crate::Error::NotAGroup) — fails the way opening it
    /// would; its write methods still address the file by path, so they stage
    /// and the commit refuses them. And a handle reached by object reference
    /// ([`Dataset::dereference`]) has no path to look up, so it returns
    /// [`Error::StaleHandle`](crate::Error::StaleHandle) — not only after a
    /// commit but after anything staged, synced or torn down, since only an
    /// immediate [`Dataset::append`] is known to leave every header where it
    /// stands. Dereference again from a fresh read.
    ///
    /// A handle onto an object this commit *publishes* — one
    /// [`Group::create_group`], [`Group::create_group_with`] or
    /// [`Group::create_dataset`] handed back, or a lookup of a staged name found
    /// — starts reading its object here. Until then it answers
    /// [`Error::NotCommitted`](crate::Error::NotCommitted) for anything needing
    /// bytes, and a refused commit leaves it doing so.
    ///
    /// The commit is durable when it returns, under the default
    /// [`SyncPolicy::Always`]; under
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose) it has reached the
    /// operating system and waits for a [`sync`](Self::sync).
    ///
    /// **A commit refused before it publishes leaves every dataset reading what
    /// it read before.** Almost everything such a commit writes lands where
    /// nothing reaches it until the commit's linearization point; the one edit
    /// that does not is a same-length [`Dataset::write`], which overwrites the
    /// dataset's existing block, and the refusal writes those bytes back on its
    /// way out. A refusal raised before the first write keeps the staged batch
    /// too, so it can be corrected and committed again (issue #316).
    ///
    /// # Errors
    ///
    /// Two failures do not carry that promise, and both call for **re-reading**
    /// the datasets the batch named rather than for a retry:
    ///
    /// - [`Error::CommitPartiallyApplied`](crate::Error::CommitPartiallyApplied),
    ///   where the restore itself failed, so a dataset may hold either value.
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
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported) rather than
    /// copied without it.
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn copy(&self, src: &str, dst: &str) -> Result<(), Error> {
        self.with_mirror_session(Change::Relocating, |session| {
            session.copy(&normalize_path(src), &normalize_path(dst))
        })
    }

    /// Copy the object at `src` in `source` — a separate, buffered read-only
    /// file — into this file at `dst`: the cross-file `H5Ocopy`, staged until
    /// [`commit`](Self::commit).
    ///
    /// `source` must be a buffered file ([`File::open`] or [`File::from_bytes`],
    /// not [`File::open_streaming`]) that uses 8-byte offsets and has no
    /// userblock; anything else is refused with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported). The source
    /// subtree is read and validated eagerly, so `source` need not outlive this
    /// call — and so a source this cannot reproduce, external storage included,
    /// is refused by this call. Refusals that concern the *destination* — `dst`
    /// already exists, or its parent group does not — still come from `commit`.
    /// Requires a read-write destination ([`File::open_rw`]); a read-only
    /// one returns [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn copy_from(&self, source: &File, src: &str, dst: &str) -> Result<(), Error> {
        self.with_mirror_session(Change::Relocating, |session| {
            session.copy_from(source, src, dst)
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

    /// Report this read-write file's live space usage as a [`SpaceAccounting`] —
    /// the current logical size, total reusable free bytes, and reusable free
    /// regions. It reflects committed state plus immediate in-place appends, not
    /// edits still staged for [`commit`](Self::commit).
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    pub fn space_accounting(&self) -> Result<SpaceAccounting, Error> {
        match &self.inner.backend {
            Backend::Edit(m) => {
                let session = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                Ok(session.space_accounting())
            }
            _ => Err(Error::ReadOnly),
        }
    }

    /// Force everything written to this file so far to durable storage — the
    /// `fsync` the application issues at its own cadence under
    /// [`SyncPolicy::OnClose`], and a redundant one under the default
    /// [`SyncPolicy::Always`]. A SWMR-writer file syncs the same way.
    ///
    /// This is a durability barrier, not a flush: it writes nothing itself.
    /// Staged edits are not applied ([`commit`](Self::commit) does that, and a
    /// `sync` before one makes only the *previous* state durable), and elements
    /// held by a live [`BufferedAppender`](crate::BufferedAppender) have not
    /// reached the file at all — flush it first.
    ///
    /// There is no need to call it before [`close`](Self::close): `close` — and
    /// dropping the last handle — issues its own barrier under every policy,
    /// because both write and both destroy the handle that would have ordered
    /// those writes. This is the mid-session checkpoint, not the closing one.
    ///
    /// Requires a read-write file ([`File::open_rw`]); a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly), and a sealed one
    /// [`Error::FileClosed`](crate::Error::FileClosed) — a closed file has
    /// already been synced.
    #[doc(alias = "fsync")]
    pub fn sync(&self) -> Result<(), Error> {
        // A barrier over writes the operations that made them already accounted
        // for, so it invalidates no handle. Classing it as a change would end
        // every by-reference handle in the session because the caller asked for
        // an `fsync`.
        self.with_mirror_session(Change::Nothing, |session| session.force_sync())
    }

    /// Commit any staged edits and seal this file. The exclusive OS lock is
    /// released once the last handle derived from this file is also dropped.
    ///
    /// After `close`, a write through any surviving [`Dataset`]/[`Group`] handle
    /// or [`File`] clone returns [`Error::FileClosed`](crate::Error::FileClosed);
    /// reads still work. `close` commits, so the one handle a commit ends ends
    /// here too: one reached by [`Dataset::dereference`] reports
    /// [`Error::StaleHandle`](crate::Error::StaleHandle) afterwards, where a
    /// handle opened by path re-resolves and keeps reading.
    pub fn close(self) -> Result<(), Error> {
        if matches!(self.inner.backend, Backend::Edit(_)) {
            // SWMR mode stages nothing — the staged surface is refused — so there
            // is nothing to commit, and it persists no free space, so there is
            // nothing to re-home.
            let swmr = self.inner.swmr_write;
            if !swmr {
                self.commit()?;
            }
            // Free-space managers and status flags, both rewritten where they
            // stand. No object header moves, so a handle that could read through
            // this file before `close` still can — which is what `close`'s own
            // documentation promises.
            self.with_mirror_session(Change::InPlace, |session| {
                // Immediate appends grow the file past any persisted free-space
                // managers without running a commit tail, so re-home them here. A
                // no-op unless this session left them stale.
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
    /// one. Returns [`Error::ReadOnly`](crate::Error::ReadOnly) for a read-only
    /// file, [`Error::FileClosed`](crate::Error::FileClosed) once the file is
    /// sealed by [`close`](Self::close), and
    /// [`Error::SwmrStagedUnsupported`](crate::Error::SwmrStagedUnsupported) for a
    /// staged edit on a SWMR-writer file.
    fn with_mirror_session<R>(
        &self,
        change: Change,
        f: impl FnOnce(&mut WriteEngine) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.inner.with_engine_mut(change, f)
    }

    /// Returns an owned handle to the root group.
    pub fn root(&self) -> Group {
        let revisions = self.inner.revisions();
        Group::new(
            self.inner.clone(),
            // A relocating commit on a read-write file can move the root, so
            // resolve it from the live mirror rather than the cached superblock.
            revisions.at(self.inner.mirror_root_address()),
            Some(String::new()),
        )
    }

    /// Resolve a path and return an owned [`Dataset`] handle.
    ///
    /// The dataset uses the file-wide chunk-cache default (configured with
    /// [`FileAccessProperties::with_chunk_cache`]). To override the cache for this
    /// one dataset, use [`dataset_with_options`](Self::dataset_with_options).
    ///
    /// Returns [`Error::NotADataset`] if the path names something that is not a
    /// dataset, and [`Error::NotAGroup`] if a component *along* the path is not
    /// a group: resolving `a/b/c` opens `a` and then `a/b` to look inside them,
    /// so a dataset at `a/b` reports `NotAGroup("a/b")` (issue #365).
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
        let normalized = normalize_path(path);
        match self.inner.staged_object(&normalized).map(|o| o.kind) {
            Some(StagedKind::Dataset) => {
                return Ok(Dataset::pending(
                    self.inner.clone(),
                    chunk_cache,
                    normalized,
                ));
            }
            Some(StagedKind::Group) => return Err(Error::NotADataset(normalized)),
            None => {}
        }
        let revisions = self.inner.revisions();
        let addr = self.inner.resolve_path(path)?;
        let hdr = self.inner.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        Ok(Dataset::new(
            self.inner.clone(),
            revisions.at(addr),
            hdr,
            chunk_cache,
            Some(normalized),
        ))
    }

    /// Resolve a path and return an owned [`Group`] handle.
    ///
    /// Returns [`Error::NotAGroup`] if the path names an object that is not a
    /// group, the way [`dataset`](Self::dataset) returns
    /// [`Error::NotADataset`] for the mirror case, and
    /// [`FormatError::PathNotFound`] if it names nothing.
    ///
    /// The same error reports a component *along* the path that is not a group,
    /// naming that component's own path rather than the one asked for: `a/b/c`
    /// stopped by a dataset at `a/b` reports `NotAGroup("a/b")` (issue #365).
    pub fn group(&self, path: &str) -> Result<Group, Error> {
        let normalized = normalize_path(path);
        match self.inner.staged_object(&normalized).map(|o| o.kind) {
            Some(StagedKind::Group) => {
                return Ok(Group::pending(self.inner.clone(), normalized));
            }
            Some(StagedKind::Dataset) => return Err(Error::NotAGroup(normalized)),
            None => {}
        }
        let revisions = self.inner.revisions();
        let addr = self.inner.resolve_path(path)?;
        if !is_group(&self.inner.parse_header(addr)?) {
            // Normalized, so that the same object refused here and refused by a
            // live handle below names itself the same way: a handle knows only
            // the normalized path it memoized.
            return Err(Error::NotAGroup(normalized));
        }
        Ok(Group::new(
            self.inner.clone(),
            revisions.at(addr),
            Some(normalized),
        ))
    }

    /// Re-read the file from disk to pick up data appended by a concurrent
    /// writer, then re-parse the superblock.
    ///
    /// This is the SWMR reader's refresh primitive. Returns
    /// [`Error::SwmrUnsupported`] if the file was not opened with
    /// [`File::open_swmr`], and [`Error::HandlesOutstanding`] if any owned
    /// [`Dataset`]/[`Group`] handle (or a clone of this `File`) is still alive —
    /// drop them before refreshing, then re-fetch them afterward, since they
    /// observe the new bytes only when re-derived from the refreshed file.
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

    /// Return the access properties used when opening this file.
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
    pub fn superblock(&self) -> &Superblock {
        self.inner.superblock()
    }

    /// The file-space management strategy this file records in its superblock
    /// extension, or `None` if it records none.
    pub fn file_space_strategy(&self) -> Option<FileSpaceStrategy> {
        self.inner.file_space_strategy()
    }

    /// The full [`FileSpaceInfo`] recorded in this file's superblock extension,
    /// if present and readable.
    pub fn file_space_info(&self) -> Option<&FileSpaceInfo> {
        self.inner.file_space_info()
    }

    /// The free regions a file persists on disk in its free-space managers, as
    /// `(address, length)` pairs sorted by address.
    pub fn persisted_free_space(&self) -> Vec<(u64, u64)> {
        self.inner.persisted_free_space()
    }

    /// The size of the underlying file in bytes (the HDF5 `H5Fget_filesize`).
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

    /// The base address (superblock base address) added to every stored relative
    /// address. Zero for a file with no userblock.
    pub(crate) fn base_address(&self) -> BaseAddress {
        self.inner.base_address()
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
/// [`FormatError::InvalidObjectReference`](crate::FormatError::InvalidObjectReference)
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
    path: String,
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
    /// To configure it in the same commit, use
    /// [`create_group_with`](Self::create_group_with). To get a [`Group`] handle
    /// onto it, look it up by name once this closure has returned, or stage it
    /// through [`Group::create_group`] instead, which hands one back.
    pub fn create_group(&mut self, name: &str) -> &mut Self {
        self.create_group_with(name, |_| {})
    }

    /// Stage a subgroup of this group, configured through `build`.
    pub fn create_group_with(
        &mut self,
        name: &str,
        build: impl FnOnce(&mut StagedGroup<'_>),
    ) -> &mut Self {
        let child = format!("{}/{}", self.path, name);
        self.ops.push(StagedOp::CreateGroup(child.clone()));
        let mut staged = StagedGroup {
            ops: &mut *self.ops,
            path: child,
        };
        build(&mut staged);
        self
    }

    /// Stage a dataset in this group, configured through `build`.
    pub fn create_dataset(
        &mut self,
        name: &str,
        build: impl FnOnce(&mut DatasetBuilder),
    ) -> &mut Self {
        let mut builder = DatasetBuilder::new(name);
        build(&mut builder);
        self.ops.push(StagedOp::CreateDataset {
            path: format!("{}/{}", self.path, name),
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
    CreateGroup(String),
    SetGroupAttr {
        path: String,
        name: String,
        value: AttrValue,
    },
    CreateDataset {
        path: String,
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
    /// Root-relative path of this group (e.g. `""` for the root, `"a/b"`), used
    /// to address the group and its children for write operations on a
    /// read-write file, and to find it again after an edit moved it. `None` for
    /// a group reached by object reference ([`Dataset::dereference`]), which has
    /// no resolvable path.
    path: Option<String>,
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
            .field("staged", &state.is_none())
            .finish()
    }
}

impl Group {
    /// Build a handle for a group whose header was found at `at`.
    fn new(file: Arc<FileInner>, at: Resolution, path: Option<String>) -> Self {
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
    /// [`Error::NotCommitted`](crate::Error::NotCommitted) until
    /// [`File::commit`], after which the first use resolves the path and the
    /// handle behaves as if it had been opened by name.
    ///
    /// The session's staged generation is taken here so that the handle can tell
    /// that commit from a *withdrawal* of the same staging, which leaves the
    /// path meaning whatever the file holds — see [`Standing`].
    fn pending(file: Arc<FileInner>, path: String) -> Self {
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
    /// Returns [`Error::StaleHandle`](crate::Error::StaleHandle) for a handle
    /// that has no path to re-resolve — one an object reference produced — once
    /// a commit has run under it, and the resolution's own error (a
    /// `PathNotFound`, say, for a group a commit deleted) when the path no
    /// longer names anything. A group this session has staged and not committed
    /// has no header at all, and that is
    /// [`Error::NotCommitted`](crate::Error::NotCommitted). A commit that
    /// replaces this group with a dataset of the same name (issue #305) leaves
    /// the path naming something that is not a group, and that is
    /// [`Error::NotAGroup`](crate::Error::NotAGroup).
    pub(crate) fn header_address(&self) -> Result<u64, Error> {
        let memo = *self.state.read().unwrap_or_else(PoisonError::into_inner);
        if let Some(memo) = memo {
            if memo.address_revision == self.file.address_revision() {
                return Ok(memo.address);
            }
        }
        let at = self
            .file
            .locate_staged(self.path.as_deref(), memo, self.staged_birth)?;
        // Checked before it is memoized. The short-circuit above does not
        // re-check, so an address installed and then refused would be the
        // answer every later call returns without looking at it again.
        if !is_group(&self.file.parse_header(at.address)?) {
            // A path-less handle never reaches here: it failed the short-circuit
            // above, and `locate` answers `StaleHandle` for one whose address
            // memo it cannot reuse. The default stands for the root's own empty
            // path, which is the only empty one a handle holds.
            return Err(Error::NotAGroup(self.path.clone().unwrap_or_default()));
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
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::DataLayout) {
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
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::DataLayout) {
                members.push((entry.name, Some((entry.object_header_address, hdr))));
            }
        }
        members.extend(staged_members.into_iter().map(|name| (name, None)));
        let file = Arc::clone(&self.file);
        let parent = self.path.clone();
        let chunk_cache = DatasetAccessProperties::new()
            .resolved_chunk_cache(self.file.access_properties.chunk_cache);
        Ok(members.into_iter().map(move |(name, on_disk)| {
            let path = child_path_of(parent.as_deref(), &name);
            let dataset = match on_disk {
                Some((address, header)) => Dataset::new(
                    Arc::clone(&file),
                    revisions.at(address),
                    header,
                    chunk_cache,
                    path,
                ),
                // Only a group with a path of its own reports staged children,
                // so a staged member always has one to be named by.
                None => Dataset::pending(
                    Arc::clone(&file),
                    chunk_cache,
                    path.expect("a staged member's parent has a path"),
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
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if is_named_datatype(&hdr) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// The datatype a committed (`H5Tcommit`) child object holds.
    ///
    /// `name` must be one [`named_datatypes`](Self::named_datatypes) returned: a
    /// name that reaches nothing fails with [`FormatError::PathNotFound`], and
    /// one that reaches an object of another kind fails with
    /// [`Error::NotANamedDatatype`], the way `H5Topen` does.
    pub fn named_datatype(&self, name: &str) -> Result<Datatype, Error> {
        Ok(self.named_datatype_at(name)?.0)
    }

    /// How many things reference the committed (`H5Tcommit`) datatype `name`:
    /// its hard links, plus every dataset and attribute that names it.
    ///
    /// This is HDF5's own object reference count (`H5Oget_info`'s `rc`), and what
    /// says whether unlinking the name would destroy the type or merely stop it
    /// being reachable by that name. A header that stores no count has exactly
    /// one reference, which is what the format means by omitting the message.
    ///
    /// A name reaching anything but a committed datatype is
    /// [`Error::NotANamedDatatype`], as for
    /// [`named_datatype`](Self::named_datatype).
    pub fn named_datatype_references(&self, name: &str) -> Result<u32, Error> {
        let (_, hdr) = self.named_datatype_header(name)?;
        let Ok(msg) = find_message(&hdr, MessageType::ObjectReferenceCount) else {
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

    /// The object header of a child that is a committed datatype, and its
    /// address.
    ///
    /// The one place the by-name datatype lookups classify what they reached, so
    /// that a child this refuses cannot be one
    /// [`named_datatypes`](Self::named_datatypes) would list. Reached the way
    /// [`group`](Self::group) and [`dataset`](Self::dataset) reach theirs, which
    /// looks the one name up rather than enumerating the group to find it.
    fn named_datatype_header(&self, name: &str) -> Result<(u64, ObjectHeader), Error> {
        let address = self
            .child_address(name)?
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.parse_header(address)?;
        if !is_named_datatype(&hdr) {
            return Err(Error::NotANamedDatatype(name.to_string()));
        }
        Ok((address, hdr))
    }

    /// The datatype a committed child object holds, and the address of the object
    /// header holding it.
    ///
    /// The address is the identity every user of the type shares: two datasets
    /// naming the same address name one type, and reproducing that requires
    /// matching them up by address rather than by what the type decodes to.
    pub(crate) fn named_datatype_at(&self, name: &str) -> Result<(Datatype, u64), Error> {
        let (address, hdr) = self.named_datatype_header(name)?;
        let msg = find_message(&hdr, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&self.file.message_body(msg)?)?;
        Ok((dt, address))
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
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if is_group(&hdr) {
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
            if is_group(&self.file.parse_header(entry.object_header_address)?) {
                members.push((entry.name, Some(entry.object_header_address)));
            }
        }
        members.extend(staged_members.into_iter().map(|name| (name, None)));
        let file = Arc::clone(&self.file);
        let parent = self.path.clone();
        Ok(members.into_iter().map(move |(name, address)| {
            let path = child_path_of(parent.as_deref(), &name);
            let group = match address {
                Some(address) => Group::new(Arc::clone(&file), revisions.at(address), path),
                // As in `iter_datasets`: staged members exist only under a
                // group that has a path.
                None => Group::pending(
                    Arc::clone(&file),
                    path.expect("a staged member's parent has a path"),
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
    /// A boolean attribute is the case that needs both channels. The C library
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

    /// Get a dataset within this group by name.
    ///
    /// The dataset uses the file-wide chunk-cache default. To override the cache
    /// for this one dataset, use
    /// [`dataset_with_options`](Self::dataset_with_options).
    pub fn dataset(&self, name: &str) -> Result<Dataset, Error> {
        self.dataset_with_options(name, DatasetAccessProperties::new())
    }

    /// Get a dataset within this group by name, applying per-dataset
    /// [`DatasetAccessProperties`] that override file-wide access defaults (HDF5's
    /// `dapl`; see `H5Pset_chunk_cache`).
    pub fn dataset_with_options(
        &self,
        name: &str,
        properties: DatasetAccessProperties,
    ) -> Result<Dataset, Error> {
        let chunk_cache = properties.resolved_chunk_cache(self.file.access_properties.chunk_cache);
        if let Some(child) = self.child_path(name) {
            match self.file.staged_object(&child).map(|o| o.kind) {
                Some(StagedKind::Dataset) => {
                    return Ok(Dataset::pending(self.file.clone(), chunk_cache, child));
                }
                Some(StagedKind::Group) => return Err(Error::NotADataset(name.to_string())),
                None => {}
            }
        }
        let revisions = self.file.revisions();
        let address = self
            .child_address(name)?
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.parse_header(address)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(name.to_string()));
        }
        Ok(Dataset::new(
            self.file.clone(),
            revisions.at(address),
            hdr,
            chunk_cache,
            self.child_path(name),
        ))
    }

    /// Get a subgroup within this group by name.
    ///
    /// Returns [`Error::NotAGroup`] if the child is not a group, the way
    /// [`dataset`](Self::dataset) returns [`Error::NotADataset`] for the mirror
    /// case, and [`FormatError::PathNotFound`] if there is no such child.
    pub fn group(&self, name: &str) -> Result<Group, Error> {
        if let Some(child) = self.child_path(name) {
            match self.file.staged_object(&child).map(|o| o.kind) {
                Some(StagedKind::Group) => {
                    return Ok(Group::pending(self.file.clone(), child));
                }
                Some(StagedKind::Dataset) => return Err(Error::NotAGroup(name.to_string())),
                None => {}
            }
        }
        let revisions = self.file.revisions();
        let address = self
            .child_address(name)?
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        if !is_group(&self.file.parse_header(address)?) {
            return Err(Error::NotAGroup(name.to_string()));
        }
        Ok(Group::new(
            self.file.clone(),
            revisions.at(address),
            self.child_path(name),
        ))
    }

    /// The object-header address of this group's child named `name`.
    ///
    /// The by-name form of [`children`](Self::children): it reads the group's
    /// links without building one entry per child, which is what makes opening
    /// each member of a large group in turn cost the group once rather than once
    /// per member (issue #228).
    fn child_address(&self, name: &str) -> Result<Option<u64>, Error> {
        match self.file.group_child(self.header_address()?, name)? {
            ChildLookup::Found(address) => Ok(Some(address)),
            ChildLookup::Absent => Ok(None),
            // Reached by the one handle whose object is never classified: the
            // root. Every other `Group` comes from a lookup that classified it
            // (`File::group`, `Group::group`, `iter_groups`, `object_at_relative`)
            // or re-resolves through `header_address`, which classifies again;
            // `File::root` takes the superblock's word for it, and nothing checks
            // that the root address names a group. The empty path is the root's
            // own name here, so the refusal names it correctly.
            ChildLookup::NotAGroup => Err(Error::NotAGroup(self.path.clone().unwrap_or_default())),
        }
    }

    /// The root-relative path of a child named `name`, or `None` if this group
    /// itself has no resolvable path (reached by object reference).
    fn child_path(&self, name: &str) -> Option<String> {
        child_path_of(self.path.as_deref(), name)
    }

    /// Create an empty subgroup `name` within this group, staged until
    /// [`File::commit`], and return a handle to it.
    ///
    /// The handle addresses the new group straight away: further groups,
    /// datasets, deletions and attributes can be staged through it, and
    /// [`group`](Self::group) finds it by name from the same session — as does
    /// any group named in this call, but not an intermediate one the commit
    /// fills in (`create_group("a/b")` leaves `a` unaddressable until then).
    /// Reading it
    /// — its attributes, or a member's data — reports
    /// [`Error::NotCommitted`](crate::Error::NotCommitted) until the commit,
    /// after which the same handle answers for the group in the file. Deleting
    /// it before the commit withdraws the staging, and the handle then reports
    /// [`Error::StagingWithdrawn`](crate::Error::StagingWithdrawn) rather than
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
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). A name the file already
    /// links to is refused here with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported) unless this
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
    pub fn create_group(&self, name: &str) -> Result<Group, Error> {
        self.create_group_with(name, |_| {})
    }

    /// Create a subgroup `name` within this group, configuring it through
    /// `build` (attributes, nested groups and datasets), staged until
    /// [`File::commit`], and return a handle to it.
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
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). Name collisions are refused
    /// on [`create_group`](Self::create_group)'s terms, for the group this
    /// creates and for everything the closure stages under it.
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
        name: &str,
        build: impl FnOnce(&mut StagedGroup<'_>),
    ) -> Result<Group, Error> {
        let child = self.child_edit_path(name)?;
        let mut ops = vec![StagedOp::CreateGroup(child.clone())];
        build(&mut StagedGroup {
            ops: &mut ops,
            path: child.clone(),
        });
        self.apply_staged(ops)?;
        Ok(Group::pending(self.file.clone(), child))
    }

    /// Create a dataset `name` within this group, configuring it through `build`
    /// (shape, data, chunks, filters, …), staged until [`File::commit`], and
    /// return a handle to it.
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
    /// [`Error::NotCommitted`](crate::Error::NotCommitted) until the commit,
    /// after which the same handle reads the dataset in the file. Deleting it
    /// before the commit withdraws the staging, and the handle then reports
    /// [`Error::StagingWithdrawn`](crate::Error::StagingWithdrawn) rather than
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
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). A name the file already
    /// links to is refused here with
    /// [`Error::EditUnsupported`](crate::Error::EditUnsupported) unless this
    /// session also deletes it — a [replacement](Self::delete) — since there
    /// would otherwise be no new dataset for the handle to address. A name this
    /// session already staged a creation at is refused for the same reason:
    /// two creations at one path are one name for two objects, and the handle
    /// would answer for whichever was staged first. [`delete`](Self::delete)
    /// withdraws a staged creation, which frees the name for another.
    ///
    /// ```no_run
    /// # use hdf5_pure::File;
    /// # fn main() -> Result<(), hdf5_pure::Error> {
    /// let file = File::open_rw("runs.h5")?;
    /// let mut col = file.root().create_dataset("col", |b| {
    ///     b.with_f64_data(&[])
    ///         .with_shape(&[0])
    ///         .with_maxshape(&[u64::MAX])
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
        name: &str,
        build: impl FnOnce(&mut DatasetBuilder),
    ) -> Result<Dataset, Error> {
        let child = self.child_edit_path(name)?;
        let mut builder = DatasetBuilder::new(name);
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

    /// Delete the object named `name` from this group, staged until
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
    /// [`Error::StagingWithdrawn`](crate::Error::StagingWithdrawn) from then on:
    /// it names nothing, and the object the file holds at that path is the one
    /// this session is removing.
    ///
    /// Deleting a group carries its whole subtree away, but only a commit that
    /// builds that group *again* can put anything back under it: staging a
    /// creation below a deleted path that nothing recreates is a batch `commit`
    /// refuses, and until it does the file's own children still own their names.
    /// The root itself cannot be deleted.
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
    pub fn delete(&self, name: &str) -> Result<(), Error> {
        self.with_child_session(name, |session, child| session.delete(child))
    }

    /// Add or update an attribute on this group, staged until [`File::commit`].
    /// Use [`remove_attr`](Self::remove_attr) to remove one. The
    /// [`root`](File::root) group's attributes are edited the same way.
    ///
    /// Requires a read-write file ([`File::open_rw`]), else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). An attribute set too large
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

    /// Run `f` with the writable session and the root-relative path of child
    /// `name`. Returns [`Error::ReadOnly`](crate::Error::ReadOnly) if the file is
    /// read-only or this group has no resolvable path.
    fn with_child_session<R>(
        &self,
        name: &str,
        f: impl FnOnce(&mut WriteEngine, &str) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.refuse_if_withdrawn()?;
        let child = self.child_path(name).ok_or(Error::ReadOnly)?;
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
        let Some(path) = self.path.as_deref().filter(|_| self.staged_birth.is_some()) else {
            return Ok(());
        };
        match self.file.staged_standing(path, self.staged_birth) {
            Standing::Withdrawn => Err(Error::StagingWithdrawn(path.to_string())),
            Standing::Live | Standing::Pending => Ok(()),
        }
    }

    /// Validate that this group can stage an edit to child `name` and return the
    /// child's root-relative path, *without* taking the session lock.
    ///
    /// Paired with [`apply_staged`](Self::apply_staged): the checks run first so
    /// a read-only or sealed file is reported before any user closure runs, the
    /// closure then runs unlocked, and the lock is taken only to record what it
    /// built (issue #200).
    fn child_edit_path(&self, name: &str) -> Result<String, Error> {
        self.file.check_staged_writable()?;
        self.refuse_if_withdrawn()?;
        self.child_path(name).ok_or(Error::ReadOnly)
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
    /// child). Returns [`Error::ReadOnly`](crate::Error::ReadOnly) if the file is
    /// read-only or this group has no resolvable path, and
    /// [`Error::FileClosed`](crate::Error::FileClosed) once the file is sealed.
    fn with_own_session<R>(
        &self,
        f: impl FnOnce(&mut WriteEngine, &str) -> Result<R, Error>,
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
        let staged = match self.path.as_deref() {
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
    path: Option<String>,
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

/// How many values of the requested type a whole-dataset read produces per
/// stored element, which is what its output buffer is reserved at.
#[derive(Clone, Copy)]
enum OutputSize {
    /// One value per stored element — every numeric decoder.
    PerElement,
    /// One value per stored *byte* — [`Dataset::read_i8`], which reinterprets
    /// bytes rather than decoding elements, and so yields one per byte of a
    /// dataset whose elements are wider than one.
    PerByte,
}

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
        if let Some(band) = chunk_dimensions
            .first()
            .map(|&d| u64::from(d))
            .filter(|&d| d > 0)
        {
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
        path: Option<String>,
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
    /// that needs bytes reports [`Error::NotCommitted`](crate::Error::NotCommitted)
    /// until [`File::commit`], after which the first use resolves the path and
    /// the handle behaves as if it had been opened by name.
    fn pending(file: Arc<FileInner>, chunk_cache_config: ChunkCacheConfig, path: String) -> Self {
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
        let Some(path) = self.path.as_deref() else {
            return Ok(None);
        };
        self.file.staged_dataset_view(path, self.staged_birth)
    }

    /// Refuse an operation that needs this dataset's bytes while it is still
    /// staged.
    fn refuse_if_pending(&self) -> Result<(), Error> {
        match self.staged_meta()? {
            // `staged_meta` answers only for a handle with a path.
            Some(_) => Err(Error::NotCommitted(self.path.clone().unwrap_or_default())),
            None => Ok(()),
        }
    }

    /// This dataset's address and parsed header, worked out again if the file
    /// has changed since the memo was taken.
    ///
    /// Returns [`Error::StaleHandle`](crate::Error::StaleHandle) for a handle
    /// that has no path to re-resolve — one an object reference produced — once
    /// a commit has run under it, and the resolution's own error (a
    /// `PathNotFound`, say, for a dataset a commit deleted) when the path no
    /// longer names anything. A dataset this session has staged and not
    /// committed has no header at all, and that is
    /// [`Error::NotCommitted`](crate::Error::NotCommitted). A path that now
    /// names something other than a dataset is
    /// [`Error::NotADataset`](crate::Error::NotADataset), the same answer
    /// opening it afresh would give.
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
            .locate_staged(self.path.as_deref(), memo, self.staged_birth)?;
        let header = self.file.parse_header(at.address)?;
        // Checked *before* it is memoized. A header installed and then refused is
        // the answer every later call short-circuits on, so this handle would
        // report `NotADataset` once and then serve the other object's header —
        // which a commit replacing a dataset with a group at the same path
        // (issue #305) makes reachable.
        if !has_message(&header, MessageType::DataLayout) {
            // Only a path can reach this: an address memo that survived is the
            // address of the dataset this handle already read there.
            return Err(Error::NotADataset(self.path.clone().unwrap_or_default()));
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

    /// Address of this dataset's object header (base-adjusted, file-absolute).
    /// Used to resolve object references that point at this dataset.
    pub(crate) fn header_address(&self) -> Result<u64, Error> {
        Ok(self.resolved()?.at.address)
    }

    /// Append `data` to this dataset in place, growing it along its first
    /// (unlimited) dimension. Every handle onto the dataset reads the new length
    /// afterwards, this one included.
    ///
    /// The file must have been opened for writing with [`File::open_rw`];
    /// a read-only file returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). The target must be a chunked,
    /// rank-1, unlimited, Extensible-Array-indexed dataset; anything else returns
    /// [`Error::AppendInPlaceUnsupported`](crate::Error::AppendInPlaceUnsupported).
    /// Both the dataset's current length and the appended length are
    /// unconstrained, on a filtered dataset as much as an unfiltered one: a
    /// partial trailing chunk is rewritten into a fresh allocation — decoded,
    /// extended and re-encoded when there is a filter pipeline — and its index
    /// element is repointed once those bytes are on the disk. The bytes the old
    /// chunk occupied are left for [`repack`](crate::repack).
    ///
    /// Two things still require a chunk-aligned starting length. A **lossy**
    /// pipeline (ZFP, or float D-scale scale-offset) is refused, because
    /// re-encoding the trailing chunk would change values that are already
    /// committed rather than reproduce them. And a trailing chunk the chunk index
    /// does not name — one a writer allocated lazily, which the reference C
    /// library does for a chunk it has not written — cannot be read to be grown,
    /// so it is refused too; append whole chunks, or use
    /// [`append_staged`](Self::append_staged).
    ///
    /// The append is immediate and crash-atomic (no `commit` needed) — under the
    /// default [`SyncPolicy::Always`]. Under
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose) the same writes are made
    /// in the same order without the `fsync` barriers between them, so the
    /// append is still immediate and still crash-atomic against *this process*
    /// failing, but ordering it against power loss is the caller's, through
    /// [`File::sync`].
    ///
    /// A **SWMR** writer ([`File::open_swmr_writer`]) keeps the narrower rule it
    /// always had — unfiltered, and chunk-aligned at both ends — because its
    /// readers are concurrent by contract and a rewritten trailing chunk is one
    /// they could be crossing. It allocates at end-of-file for that same reason:
    /// a region this session freed is one a reader may still be inside.
    ///
    /// Where the new chunks land otherwise depends on how the file records its
    /// free space. A default-strategy file spends a hole an earlier commit in
    /// this session left; a file that **persists** its free-space managers
    /// (including every paged file) spends only space the session has first taken
    /// *out* of them, in a rewrite of its own, so no byte an append writes is one
    /// a durable manager advertises — a crash can strand the unspent remainder of
    /// such a batch, but nothing can be handed out twice. A batch gathers every
    /// hole the appended chunk fits in, up to a megabyte of them, so the file
    /// grows only when no hole can hold the chunk.
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
    pub fn append<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        let g = self.append_geometry()?;
        self.append_batches(g, data.len() as u64, |b, r| {
            b.append(&data[r]);
        })
    }

    /// Append raw little-endian element bytes to this dataset in place. Prefer
    /// [`append`](Self::append) when the element type is known; see it for the
    /// file-mode and eligibility rules.
    pub fn append_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
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
        self.append_batches(g, total, |b, r| {
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

    /// A [`BufferedAppender`] over this dataset: appended elements are held in
    /// memory and written a whole chunk at a time, so a caller appending less
    /// than a chunk per call writes to the file once per chunk instead of once
    /// per call, and a filtered dataset is never left mid-chunk for the next
    /// write to re-encode.
    ///
    /// Every eligibility rule [`append`](Self::append) applies is applied here,
    /// so an ineligible dataset is reported now rather than on the first write.
    /// Buffered elements are not in the file until the appender flushes; see
    /// [`BufferedAppender`] for the full bargain.
    pub fn buffered_appender(&mut self) -> Result<BufferedAppender<'_>, Error> {
        BufferedAppender::new(self)
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
            .claim_for_appender(self.path.as_deref())
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
    /// resident reports one unbounded batch, so the call stays a single
    /// crash-atomic apply there.
    ///
    /// Every predictable refusal (wrong datatype, ineligible dataset) is raised
    /// before the first batch is applied. The cached header and chunk cache are
    /// then refreshed so later reads on this handle observe the new length. A
    /// filtered dataset sitting on a partial trailing chunk is grown by the
    /// first batch, which re-encodes that chunk into a fresh allocation and
    /// leaves every later batch starting on a boundary.
    fn append_batches(
        &mut self,
        g: AppendGeometry,
        total_elems: u64,
        fill: impl Fn(&mut AppendBuilder, std::ops::Range<usize>),
    ) -> Result<(), Error> {
        // Worked out once for the whole call: every batch names the same
        // dataset, and an in-place append does not move it.
        let target = self.append_target()?;
        let mut dim = g.current_dim;
        let mut done = 0u64;
        loop {
            // An empty append still runs one (empty) engine call, so datatype
            // validation happens whether or not there are elements.
            let to_boundary = (g.chunk_elems - dim % g.chunk_elems) % g.chunk_elems;
            let take = (total_elems - done).min(to_boundary.saturating_add(g.full_batch_elems));
            let mut b = AppendBuilder::new();
            fill(&mut b, done.to_usize()?..(done + take).to_usize()?);
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
    /// [`Error::ReadOnly`](crate::Error::ReadOnly). Unlike [`append`](Self::append)
    /// (immediate), this is a staged edit applied on [`File::commit`].
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
    /// The replacement must match the on-disk datatype and shape exactly; a
    /// reshape or retype is refused on [`File::commit`].
    ///
    /// This overwrites element bytes and nothing else, so a builder asking for
    /// more than that is refused by **this call**, before anything is staged:
    /// chunking, filters or an extensible shape; an attribute; a fill value; and
    /// [`with_path_references`](DatasetBuilder::with_path_references), whose
    /// element bytes are placeholder addresses only a newly created dataset can
    /// resolve. Set those when the dataset is created.
    ///
    /// [`with_vlen_strings`](DatasetBuilder::with_vlen_strings) is **not**
    /// refused: overwriting a variable-length-string dataset places a fresh
    /// global heap collection for the new strings and resolves the staged
    /// element references against it. Overwriting the same dataset again
    /// reclaims the collection the previous overwrite placed, so rotating its
    /// strings in a session does not grow the file without bound. The
    /// collections it held when the session *opened* are not reclaimed — a
    /// collection can be shared between objects, and only this session's own
    /// placements are known not to be — so [`repack`](crate::repack) is what
    /// recovers those.
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
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
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

    /// Stage an append to this dataset applied on [`File::commit`] — the staged,
    /// index-rebuilding counterpart of the immediate [`append`](Self::append).
    ///
    /// Unlike [`append`](Self::append) (immediate, amortized `O(1)`,
    /// Extensible-Array only), this rebuilds the chunk index on commit and so
    /// also grows datasets whose Extensible-Array index is not yet allocated.
    /// Like it, a dataset under a **lossy** pipeline (ZFP, or float D-scale
    /// scale-offset) whose length is not a whole multiple of its chunk length
    /// is refused: growing that trailing chunk would decode and re-encode
    /// values that are already committed, changing them. That one is refused
    /// *here*, by this call, rather than at the commit — nothing is staged, and
    /// edits staged beside it are unaffected.
    /// Configure the appended elements through `build` on the
    /// [`AppendBuilder`]; repeated calls within the builder concatenate in
    /// order. The dataset must be chunked, unlimited
    /// along axis 0, Extensible-Array indexed, rank 1, use a re-encodable filter
    /// pipeline, and have a single hard link, otherwise
    /// [`Error::AppendUnsupported`](crate::Error::AppendUnsupported) is returned
    /// on [`File::commit`].
    ///
    /// The file must have been opened with [`File::open_rw`], else
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
    /// The closure configures a standalone builder, not the file, so it may read
    /// the same [`File`]; nothing it stages resolves until [`File::commit`].
    pub fn append_staged(&mut self, build: impl FnOnce(&mut AppendBuilder)) -> Result<(), Error> {
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
        let mut builder = AppendBuilder::new();
        build(&mut builder);
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
    /// [`Error::ReadOnly`](crate::Error::ReadOnly).
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
    /// change (e.g. an append's new dimension). Returns
    /// [`Error::ReadOnly`](crate::Error::ReadOnly) if the file is read-only or the
    /// handle has no resolvable path (reached by object reference).
    fn with_session_mut<R>(
        &mut self,
        f: impl FnOnce(&mut WriteEngine, &str) -> Result<R, Error>,
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

    /// The dataset's maximum dimensions, when it is extensible. An unlimited
    /// dimension is reported as `u64::MAX`. Returns `Ok(None)` for a fixed-shape
    /// dataset (no maximum-dimensions record, or one equal to the current shape).
    ///
    /// Together with [`is_chunked`](Self::is_chunked) and
    /// [`chunk_shape`](Self::chunk_shape), this lets a caller check up front
    /// whether a dataset is eligible for
    /// [`Dataset::append_staged`](crate::Dataset::append_staged)
    /// (which requires a chunked dataset whose first maximum dimension is
    /// `u64::MAX`) instead of relying on the append's refusal error.
    pub fn maxshape(&self) -> Result<Option<Vec<u64>>, Error> {
        if let Some(meta) = self.staged_meta()? {
            return Ok(meta.maxshape);
        }
        let ds = self.dataspace()?;
        match &ds.max_dimensions {
            Some(md) if *md != ds.dimensions => Ok(Some(md.clone())),
            _ => Ok(None),
        }
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
        Ok(Some(
            chunk_dimensions[..rank]
                .iter()
                .map(|&c| u64::from(c))
                .collect(),
        ))
    }

    /// The HDF5 filter IDs applied to this dataset's chunks, in pipeline
    /// (application) order, or an empty vector when the dataset is unfiltered.
    /// The IDs are the registered HDF5 filter numbers — e.g. 1 = deflate,
    /// 2 = shuffle, 3 = fletcher32, 6 = scale-offset — so a caller can inspect
    /// the pipeline without decoding a chunk. A dataset this session has staged
    /// and not yet committed reports the pipeline its builder asked for.
    pub fn filters(&self) -> Vec<u16> {
        if let Ok(Some(meta)) = self.staged_meta() {
            return meta.filters.into_iter().map(|(id, _)| id).collect();
        }
        self.filter_pipeline_parsed()
            .map(|p| p.filters.iter().map(|f| f.filter_id).collect())
            .unwrap_or_default()
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
    /// Returns `Err` if the dataset has no data-layout message, if it cannot be
    /// parsed, or if a chunked dataset uses an index kind this crate does not
    /// recognize.
    pub fn layout(&self) -> Result<Layout, Error> {
        Ok(match self.data_layout()? {
            DataLayout::Compact { data } => Layout::Compact {
                size: data.len() as u64,
            },
            DataLayout::Contiguous { address, size } => Layout::Contiguous {
                address: self.absolute_address(address)?,
                size,
            },
            DataLayout::Chunked {
                version,
                chunk_index_type,
                ..
            } => Layout::Chunked {
                // Reuse `chunk_shape` so the two accessors can never disagree on
                // how the element-size dimension is stripped.
                chunk_shape: self.chunk_shape()?.unwrap_or_default(),
                index: ChunkIndex::from_layout(version, chunk_index_type)?,
            },
            DataLayout::Virtual { .. } => Layout::Virtual,
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
    /// Returns `Err` if the data-layout message is missing or cannot be parsed,
    /// or if a chunked dataset uses an index kind this crate does not recognize.
    pub fn chunk_index(&self) -> Result<Option<ChunkIndex>, Error> {
        match self.data_layout()? {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                ..
            } => Ok(Some(ChunkIndex::from_layout(version, chunk_index_type)?)),
            _ => Ok(None),
        }
    }

    /// Enumerate every allocated chunk of this chunked dataset — one [`Chunk`]
    /// (logical offset, absolute file address, on-disk stored size, filter mask)
    /// per chunk, in index order.
    ///
    /// This reads only the chunk index, not the chunk data, so a caller can seek
    /// to and decode chunks one at a time without materializing the whole
    /// dataset. The curated analogue of `H5Dget_num_chunks` + `H5Dget_chunk_info`
    /// (`chunks()?.len()` is the chunk count).
    ///
    /// Returns `Ok(vec![])` for a chunked dataset whose storage has not been
    /// allocated yet (including a not-yet-written dataset that will use a
    /// [`ChunkIndex::BTreeV2`] index). Returns `Err` if the dataset is not chunked
    /// (check [`layout`](Self::layout) or [`is_chunked`](Self::is_chunked) first),
    /// or if its allocated storage is indexed by a [`ChunkIndex::BTreeV2`] index,
    /// which has no enumerator yet.
    pub fn chunks(&self) -> Result<Vec<Chunk>, Error> {
        let rank = self.dataspace()?.dimensions.len();
        Ok(self
            .raw_chunks()?
            .into_iter()
            .map(|c| Chunk {
                offset: c.offsets.into_iter().take(rank).collect(),
                address: c.address,
                storage_size: u64::from(c.chunk_size),
                filter_mask: c.filter_mask,
            })
            .collect())
    }

    /// This dataset's filter pipeline as an ordered list of [`Filter`]s — each
    /// with its identifier, optional name, optional/mandatory flag, and client
    /// data — or an empty vector when the dataset is unfiltered.
    ///
    /// The detailed companion to [`filters`](Self::filters), which returns just
    /// the identifiers. Filters are listed in application (write) order — the
    /// on-disk pipeline order, matching [`filters`](Self::filters); a reader
    /// inverts them in the *reverse* of this order to decode a chunk. The curated
    /// analogue of `H5Pget_nfilters` + `H5Pget_filter2`.
    ///
    /// A dataset this session has staged and not yet committed reports each
    /// filter's identifier and optional flag, and no name or client data: those
    /// are derived from the dataset being written (element size, chunk geometry,
    /// fill value) when [`File::commit`] writes it.
    pub fn filter_pipeline(&self) -> Vec<Filter> {
        if let Ok(Some(meta)) = self.staged_meta() {
            return meta
                .filters
                .into_iter()
                .map(|(id, is_optional)| Filter {
                    id,
                    is_optional,
                    name: None,
                    client_data: Vec::new(),
                })
                .collect();
        }
        self.filter_pipeline_parsed()
            .map(|p| {
                p.filters
                    .into_iter()
                    .map(|f| Filter {
                        id: f.filter_id,
                        is_optional: f.is_optional(),
                        name: f.name,
                        client_data: f.client_data,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Shift a base-relative on-disk address to an absolute file offset using the
    /// superblock base address (`addr_offset`). A no-op for the common
    /// base-zero file. Returns `Ok(None)` for an unallocated (undefined) address.
    fn absolute_address(&self, address: Option<u64>) -> Result<Option<u64>, Error> {
        match address {
            Some(rel) => Ok(Some(self.file.addr_offset.absolute(rel)?)),
            None => Ok(None),
        }
    }

    /// Returns the simplified datatype of the dataset.
    ///
    /// A dataset this session has staged and not yet committed answers with the
    /// datatype its builder settled on.
    pub fn dtype(&self) -> Result<DType, Error> {
        let dt = self.datatype()?;
        Ok(classify_datatype(&dt))
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

    /// The raw bytes of this dataset's user-defined fill value, encoded in its
    /// datatype, or `None` when no user-defined fill value is set (the library
    /// default or an explicitly undefined fill). Reads whichever Fill Value
    /// message the header carries — the current `0x0005` (versions 1/2/3) or the
    /// legacy `0x0004` — so files from this crate, the reference C library, and
    /// h5py are all handled.
    pub(crate) fn defined_fill_bytes(&self) -> Result<Option<Vec<u8>>, Error> {
        let state = self.resolved()?;
        let msg = state
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FillValue)
            .or_else(|| {
                state
                    .header
                    .messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FillValueOld)
            });
        match msg {
            Some(m) => Ok(crate::fill_value::parse_defined_fill_value(
                m.msg_type,
                &self.file.message_body(m)?,
            )?),
            None => Ok(None),
        }
    }

    /// The fill bytes that *unallocated storage reads as* — which is not the
    /// same question [`defined_fill_bytes`](Self::defined_fill_bytes) answers.
    ///
    /// A dataset may declare a fill value and also declare, through the Fill
    /// Value Write Time, that the library never writes it
    /// (`H5D_FILL_TIME_NEVER`). Its unallocated storage then has no defined
    /// contents — the C library leaves the read buffer untouched — so this
    /// returns `None` and the region reads as deterministic zeros rather than as
    /// a value nothing ever put there. `fill_value` still reports the declared
    /// value, because it *is* declared; see [`fill_value_is_written`].
    fn fill_bytes(&self) -> Result<Option<Vec<u8>>, Error> {
        let state = self.resolved()?;
        let msg = state
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FillValue)
            .or_else(|| {
                state
                    .header
                    .messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FillValueOld)
            });
        let Some(m) = msg else {
            return Ok(None);
        };
        let body = self.file.message_body(m)?;
        if !crate::fill_value::fill_value_is_written(m.msg_type, &body)? {
            return Ok(None);
        }
        Ok(crate::fill_value::parse_defined_fill_value(
            m.msg_type, &body,
        )?)
    }

    /// Read the whole dataset with `decode`, sweeping it a row window at a time.
    ///
    /// A typed whole-dataset read used to be [`read_raw`](Self::read_raw)
    /// followed by a decode of the entire buffer, which held the stored bytes
    /// and the decoded values at the same time and so peaked at twice the
    /// dataset — a caller reading a 4 GiB array needed 8 GiB (issue #289).
    /// Decoding a window at a time leaves one window of stored bytes beside the
    /// output instead of a whole second copy of it, and the bytes are identical
    /// either way: a window returns exactly the rows [`read_raw`](Self::read_raw)
    /// would have put there.
    ///
    /// The output buffer is reserved once, at the size the whole dataset decodes
    /// to, so no window reallocates it — a growth step would put a second copy of
    /// the output alongside the first and give back what the windowing saved.
    ///
    /// A dataset that fits in one window — including one with no rows at all — is
    /// read whole. The empty case still runs `decode`, because a decoder is also
    /// what reports a datatype it cannot read, and a zero-element string dataset
    /// must go on failing a numeric read rather than answering with an empty
    /// vector.
    fn read_whole_typed<T, F>(&self, out_size: OutputSize, decode: F) -> Result<Vec<T>, Error>
    where
        F: Fn(&[u8], &Datatype, &mut Vec<T>) -> Result<(), FormatError>,
    {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.read_layout()?;
        let pipeline = self.filter_pipeline_parsed();
        // See `read_raw`: an unparseable fill value message is carried into the
        // read rather than failing it up front.
        let fill_bytes = self.fill_bytes();
        let elem_size = dt.element_size_usize()?;
        let fill = match &fill_bytes {
            Ok(b) => FillPattern::new(b.as_deref(), elem_size),
            Err(_) => FillPattern::UNKNOWN,
        };
        let spec = RawReadSpec {
            layout: &dl,
            dataspace: &ds,
            datatype: &dt,
            pipeline: pipeline.as_ref(),
            fill,
        };

        // What a whole read checks before it reads a byte, and what a sweep would
        // otherwise skip: a compact or contiguous layout whose declared size
        // disagrees with the dataspace is refused. Reading in windows must not
        // turn that into a check that fires only on datasets small enough to be
        // read whole.
        let stored = spec.stored_byte_len()?;

        // A window is cut by the *stored* element width, while a decoder slices
        // what it is handed by the width of the type it decodes — the base type,
        // for an enumeration. Those are the same width for every valid file, an
        // enumeration's size being its base's. A crafted file where they differ
        // must not get one verdict from a sweep and another from a whole read, so
        // it is read whole.
        let decoded_width = data_read::effective_numeric(&dt).type_size();

        let mut out = Vec::new();
        let n0 = ds.dimensions.first().copied().unwrap_or(1);
        let rows = typed_window_rows(&dl, &ds, elem_size)?.get();
        if n0 <= rows || decoded_width != dt.type_size() {
            // No reservation here: `decode` sizes the output from the bytes it
            // was handed, which is exact.
            let raw = self.file.read_dataset_raw(spec, &self.chunk_cache)?;
            decode(&raw, &dt, &mut out)?;
            return Ok(out);
        }

        let values = match out_size {
            OutputSize::PerElement => ds.num_elements().to_usize()?,
            OutputSize::PerByte => stored,
        };

        // One pass for the whole sweep, not one per window: the sweep visits each
        // chunk exactly once, so a window that offered its chunks to a cache
        // already full would copy and evict with no later reader for either. The
        // cache ends up holding what a whole read would have left it — the
        // chunks reached first. See [`CachePass`].
        let pass = self.chunk_cache.begin_pass();
        let mut start = 0;
        while start < n0 {
            let count = rows.min(n0 - start);
            let raw =
                self.file
                    .read_dataset_raw_rows(spec, &self.chunk_cache, pass, start, count)?;
            if start == 0 {
                // Reserved once, and only after a window has come back. This size
                // comes from the file: sizing an allocation from it before
                // reading anything lets a dataspace claiming a terabyte ask for a
                // terabyte over a file that cannot serve one row. `try_reserve`
                // for the same reason — a file-derived capacity that cannot be
                // had is an answer this reader owes its caller, not a panic.
                out.try_reserve(values)
                    .map_err(|_| FormatError::ValueTooLargeForPlatform {
                        value: values as u64,
                        target: "one allocation",
                    })?;
            }
            decode(&raw, &dt, &mut out)?;
            start += count;
        }
        Ok(out)
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
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_f64_into)
    }

    /// Read all data as `f32` values.
    pub fn read_f32(&self) -> Result<Vec<f32>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_f32_into)
    }

    /// Read all data as `i32` values.
    pub fn read_i32(&self) -> Result<Vec<i32>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_i32_into)
    }

    /// Read all data as `i64` values.
    pub fn read_i64(&self) -> Result<Vec<i64>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_i64_into)
    }

    /// Read all data as `u64` values.
    pub fn read_u64(&self) -> Result<Vec<u64>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_u64_into)
    }

    /// Read all data as `u8` values.
    pub fn read_u8(&self) -> Result<Vec<u8>, Error> {
        self.read_raw()
    }

    /// Read all data as `i8` values.
    pub fn read_i8(&self) -> Result<Vec<i8>, Error> {
        self.read_whole_typed(OutputSize::PerByte, |raw, _dt, out| {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "read_i8 reinterprets each stored byte as the signed i8 the caller requested"
            )]
            out.extend(raw.iter().map(|&b| b as i8));
            Ok(())
        })
    }

    /// Read all data as `i16` values.
    pub fn read_i16(&self) -> Result<Vec<i16>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_i16_into)
    }

    /// Read all data as `u16` values.
    pub fn read_u16(&self) -> Result<Vec<u16>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_u16_into)
    }

    /// Read all data as `u32` values.
    pub fn read_u32(&self) -> Result<Vec<u32>, Error> {
        self.read_whole_typed(OutputSize::PerElement, data_read::read_as_u32_into)
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
            self.file.offset_size(),
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
                self.file.offset_size(),
                self.file.length_size(),
                self.file.addr_offset,
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
                self.file.offset_size(),
                self.file.length_size(),
                self.file.addr_offset,
                1, // a VL string's base type is a single byte
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
        let Datatype::VariableLength { base_type, .. } = &datatype else {
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
                self.file.offset_size(),
                self.file.length_size(),
                self.file.addr_offset,
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
                    self.file.offset_size(),
                    self.file.length_size(),
                    self.file.addr_offset,
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
        let msg = find_message(&state.header, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&self.file.message_body(msg)?)?;
        Ok(dt)
    }

    /// The object-header address of this dataset's committed (shared) element
    /// type, or `None` when the type is written in the dataset's own header.
    ///
    /// [`datatype`](Self::datatype) resolves it either way, so this is for
    /// callers that must *reproduce* the dataset: writing the resolved type back
    /// inline loses the link every C-library reader reports by name, and the
    /// address is what says which committed object to name instead.
    pub(crate) fn committed_datatype_address(&self) -> Result<Option<u64>, Error> {
        let state = self.resolved()?;
        let msg = find_message(&state.header, MessageType::Datatype)?;
        self.file.shared_target_address(msg)
    }

    pub(crate) fn dataspace(&self) -> Result<Dataspace, Error> {
        let state = self.resolved()?;
        let msg = find_message(&state.header, MessageType::Dataspace)?;
        Ok(Dataspace::parse(
            &self.file.message_body(msg)?,
            self.file.length_size(),
        )?)
    }

    pub(crate) fn data_layout(&self) -> Result<DataLayout, Error> {
        let state = self.resolved()?;
        let msg = find_message(&state.header, MessageType::DataLayout)?;
        Ok(DataLayout::parse(
            &msg.data,
            self.file.offset_size(),
            self.file.length_size(),
        )?)
    }

    /// A handle that can no longer be resolved reports no pipeline, the same
    /// answer an unfiltered dataset gives: this feeds the two infallible
    /// accessors ([`filters`](Self::filters) and
    /// [`filter_pipeline`](Self::filter_pipeline)), which have no way to say
    /// why. Every caller that *can* say uses a `Result` reader instead.
    pub(crate) fn filter_pipeline_parsed(&self) -> Option<FilterPipeline> {
        let state = self.resolved().ok()?;
        let msg = state
            .header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)?;
        let body = self.file.message_body(msg).ok()?;
        FilterPipeline::parse(&body).ok()
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
            .any(|m| m.msg_type == MessageType::ExternalDataFiles))
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

    /// The raw, still-compressed on-disk bytes of every allocated chunk of this
    /// chunked dataset, with each chunk's `(address, on-disk size, filter mask,
    /// logical offset)` — the same `ChunkInfo`s the chunked reader walks before
    /// decompressing. Used by repack to copy compressed chunks verbatim without
    /// ever decoding them.
    ///
    /// Returns `Err` if the layout is not chunked. Returns `Ok(vec![])` for an
    /// empty / never-allocated chunked dataset (no index address). Covers every
    /// index type the reader supports (v3 B-tree and v4 single-chunk, implicit,
    /// fixed-array, and extensible-array).
    pub(crate) fn raw_chunks(&self) -> Result<Vec<crate::chunked_read::ChunkInfo>, Error> {
        let DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } = self.data_layout()?
        else {
            return Err(Error::Format(crate::error::FormatError::ChunkedReadError(
                "chunk enumeration requires a chunked dataset".into(),
            )));
        };
        // An undefined index address means no storage is allocated yet.
        let Some(addr) = btree_address else {
            return Ok(Vec::new());
        };
        let dataspace = self.dataspace()?;
        let elem_size = self.datatype()?.element_size_usize()?;
        let base = self.file.addr_offset;
        // The chunk index — its root at `addr` and every internal node — stores
        // addresses relative to the base address. Walk it through a base-relative
        // view so those resolve, then shift each returned chunk address back to an
        // absolute file offset, since callers (repack) read the chunk bytes from
        // the full file source.
        self.file.with_source(|source| {
            if base.is_zero() {
                return Ok(crate::chunked_read::collect_chunks_for_layout_from_source(
                    source,
                    version,
                    chunk_index_type,
                    addr,
                    single_chunk_filtered_size,
                    single_chunk_filter_mask,
                    &chunk_dimensions,
                    &dataspace,
                    elem_size,
                    self.file.offset_size(),
                    self.file.length_size(),
                )?);
            }
            let framed = BaseOffsetSource {
                inner: source,
                base,
            };
            let mut chunks = crate::chunked_read::collect_chunks_for_layout_from_source(
                &framed,
                version,
                chunk_index_type,
                addr,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
                &chunk_dimensions,
                &dataspace,
                elem_size,
                self.file.offset_size(),
                self.file.length_size(),
            )?;
            for c in &mut chunks {
                c.address = base.absolute(c.address)?;
            }
            Ok(chunks)
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
            .find(|m| m.msg_type == MessageType::FilterPipeline)
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
        let pipeline = self.filter_pipeline_parsed();
        // A fill value message this parser cannot read does not, by itself,
        // make the dataset unreadable: it only decides what *unallocated*
        // storage looks like. Carry the uncertainty into the read and let it
        // fail there, and only there. `Dataset::fill_value` still reports the
        // parse error to a caller asking about the value.
        let fill_bytes = self.fill_bytes();
        let fill = match &fill_bytes {
            Ok(b) => FillPattern::new(b.as_deref(), dt.element_size_usize()?),
            Err(_) => FillPattern::UNKNOWN,
        };
        let spec = RawReadSpec {
            layout: &dl,
            dataspace: &ds,
            datatype: &dt,
            pipeline: pipeline.as_ref(),
            fill,
        };
        Ok(self.file.read_dataset_raw(spec, &self.chunk_cache)?)
    }

    /// Read the raw element bytes of the row window `[start_row, start_row + num_rows)`
    /// — a range along the first dimension.
    ///
    /// The windowed companion to [`read_raw`](Self::read_raw): only the storage the
    /// window overlaps is read — a bounded sub-read for compact and contiguous
    /// layouts, just the overlapping chunks for chunked layouts — so peak memory
    /// scales with the window, not the dataset. Use it to stream a large dataset a
    /// fixed number of rows at a time.
    ///
    /// Each row keeps its full inner shape, and the bytes match what
    /// [`read_raw`](Self::read_raw) produces for those rows, so the typed
    /// `read_*_rows` helpers decode a window like their whole-dataset forms. The
    /// window is clamped to the first dimension: a read past the end returns only
    /// the rows that exist, and a 0-D scalar is one row. A window covering every
    /// row delegates to [`read_raw`](Self::read_raw), so a full-range window never
    /// costs more than a whole read. Variable-length string
    /// bytes are heap references, not text — use
    /// [`read_string_rows`](Self::read_string_rows).
    pub fn read_raw_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.read_layout()?;

        let n0 = ds.dimensions.first().copied().unwrap_or(1);
        let start = start_row.min(n0);
        let count = num_rows.min(n0 - start);

        // A window covering every row is exactly a whole read: delegate, so it
        // never costs a window-shaped copy on top of one.
        // See `read_raw`: an unparseable fill value message is carried into the
        // read rather than failing it up front.
        let parsed_fill = self.fill_bytes();
        let fill = match &parsed_fill {
            Ok(b) => FillPattern::new(b.as_deref(), dt.element_size_usize()?),
            Err(_) => FillPattern::UNKNOWN,
        };

        let pipeline = self.filter_pipeline_parsed();
        let spec = RawReadSpec {
            layout: &dl,
            dataspace: &ds,
            datatype: &dt,
            pipeline: pipeline.as_ref(),
            fill,
        };

        if start == 0 && count == n0 {
            return Ok(self.file.read_dataset_raw(spec, &self.chunk_cache)?);
        }

        // A lone window's successor is the adjacent one, and the chunk they share
        // is the one this read finishes on; `CachePass::LRU` is what retains it.
        Ok(self.file.read_dataset_raw_rows(
            spec,
            &self.chunk_cache,
            CachePass::LRU,
            start,
            count,
        )?)
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
    #[expect(
        clippy::cast_possible_wrap,
        reason = "read_i8 reinterprets each stored byte as the signed i8 the caller requested"
    )]
    pub fn read_i8_rows(&self, start_row: u64, num_rows: u64) -> Result<Vec<i8>, Error> {
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(raw.iter().map(|&b| b as i8).collect())
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
        self.read_raw_rows(start_row, num_rows)
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
        let dt = self.datatype()?;
        if vl_data::is_vlen_string_datatype(&dt) {
            // The window's heap references, read memory-bounded like any other
            // fixed-size element (4-byte length + collection address + 4-byte
            // object index), one row spanning its inner dimensions. Resolving
            // only those against the global heap keeps the bound — the same
            // resolution `read_string` runs over the whole dataset's references.
            let raw = self.read_raw_rows(start_row, num_rows)?;
            let ref_size = 4 + self.file.offset_size() as usize + 4;
            let num_elements = (raw.len() / ref_size) as u64;
            let mut strings = Vec::new();
            self.file.with_source(|source| -> Result<(), Error> {
                Ok(vl_data::visit_vl_strings_from_source(
                    source,
                    &raw,
                    num_elements,
                    self.file.offset_size(),
                    self.file.length_size(),
                    self.file.addr_offset,
                    VlenStringReadOptions::default(),
                    |string| strings.push(String::from(string)),
                )?)
            })?;
            return Ok(strings);
        }
        let raw = self.read_raw_rows(start_row, num_rows)?;
        Ok(data_read::read_as_strings(&raw, &dt)?)
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
        let element_size = datatype.element_size_usize()?;
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
        use crate::provenance::{ATTR_SHA256, VerifyResult, sha256_hex};

        let attrs = self.attrs()?;
        let stored = match attrs.get(ATTR_SHA256).and_then(AttrValue::as_str) {
            Some(s) => s.trim_end_matches('\0').to_string(),
            None => return Ok(VerifyResult::NoHash),
        };

        let computed = sha256_hex(&self.read_raw()?);
        if computed == stored {
            Ok(VerifyResult::Ok)
        } else {
            Ok(VerifyResult::Mismatch { stored, computed })
        }
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

/// Normalize a user-supplied object path to the root-relative form the write
/// session addresses by: strip any leading/trailing `/` so `"/a/b"` and `"a/b"`
/// name the same object.
fn normalize_path(path: &str) -> String {
    path.trim_matches('/').to_string()
}

fn has_message(header: &ObjectHeader, msg_type: MessageType) -> bool {
    header.messages.iter().any(|m| m.msg_type == msg_type)
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
    has_message(header, MessageType::Datatype)
        && !has_message(header, MessageType::DataLayout)
        && !is_group(header)
}

/// The root-relative path of a child named `name` under `parent`, or `None` if
/// the parent has no resolvable path (reached by object reference).
///
/// Free-standing rather than a method on [`Group`] so the member iterators can
/// build child paths from a closure that outlives the borrow of the group they
/// came from.
fn child_path_of(parent: Option<&str>, name: &str) -> Option<String> {
    parent.map(|p| {
        if p.is_empty() {
            name.to_string()
        } else {
            format!("{p}/{name}")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileBuilder;
    use std::sync::atomic::AtomicUsize;

    // -----------------------------------------------------------------------
    // Reporting the metadata cache (issue #353)
    // -----------------------------------------------------------------------

    /// `SourceView` serves its metadata reads from the streaming backend's
    /// cache, so it has to forward the account of them as well. It is the one
    /// wrapper `File::metadata_cache_stats` does not itself go through (that
    /// dispatch uses `with_source`, which reaches the read-write backend too), so
    /// nothing else would notice the forward going missing.
    #[test]
    fn the_source_view_reports_the_cache_it_reads_through() {
        let backend = MetadataCachingSource::new(
            BytesSource::new((0..=255u8).collect::<Vec<u8>>()),
            MetadataCacheConfig::new(4096),
        );
        let view = SourceView::Stream(&backend);

        assert_eq!(view.metadata_cache_stats().unwrap().reads(), 0);
        view.read_metadata_at(0, 64).unwrap();
        view.read_metadata_at(0, 64).unwrap();
        let stats = view
            .metadata_cache_stats()
            .expect("the backend has a cache, so the view reports it");
        assert_eq!((stats.hits(), stats.misses()), (1, 1));

        view.reset_metadata_cache_stats();
        let cleared = view.metadata_cache_stats().unwrap();
        assert_eq!(cleared.hits(), 0);
        assert_eq!(cleared.entries(), 1, "a reset evicts nothing");

        // A whole-file buffer is the cache; there is no second one to report.
        assert_eq!(SourceView::Mem(&[0u8; 16]).metadata_cache_stats(), None);
    }

    // -----------------------------------------------------------------------
    // Handle re-validation across an edit (issue #351)
    // -----------------------------------------------------------------------

    /// A file with two chunked datasets whose trailing chunk is partial, one
    /// contiguous dataset, and one subgroup.
    fn revalidation_fixture(path: &std::path::Path) {
        let mut b = FileBuilder::new();
        for ds in ["log", "other"] {
            b.create_dataset(ds)
                .with_i32_data(&[0, 1])
                .with_shape(&[2])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[4]);
        }
        b.create_dataset("plain").with_i32_data(&[7, 8, 9]);
        let g = b.create_group("g");
        b.add_group(g.finish());
        b.write(path).unwrap();
    }

    /// The issue itself: a handle taken before a commit goes on answering for
    /// the object after it, rather than for the copy the commit left behind.
    #[test]
    fn a_handle_follows_its_object_across_a_commit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("follow.h5");
        revalidation_fixture(&path);

        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("plain").unwrap();
        let group = file.group("g").unwrap();

        // Both edits relocate an object header: the dataset's own, and — since
        // the new child is linked into it — the group's.
        ds.set_attr("units", AttrValue::AsciiString("m".into()))
            .unwrap();
        group
            .create_dataset("child", |b| {
                b.with_i32_data(&[4, 5]);
            })
            .unwrap();
        file.commit().unwrap();

        assert_eq!(
            sorted(ds.attrs()),
            vec![r#"units=AsciiString("m")"#.to_string()],
            "the dataset handle must report the attribute the commit added"
        );
        assert_eq!(
            group.datasets().unwrap(),
            vec!["child".to_string()],
            "the group handle must report the child the commit added"
        );
        assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]);
        assert_eq!(
            group.dataset("child").unwrap().read_i32().unwrap(),
            vec![4, 5]
        );
        file.close().unwrap();
    }

    /// Two handles on one dataset are two views of one object, not two objects:
    /// an append through either is what the other reads next, chunk cache and
    /// all. The appended elements land in a chunk the reader already holds, so a
    /// retained one would answer with what stood there before.
    #[test]
    fn an_append_through_one_handle_is_what_another_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two_handles.h5");
        revalidation_fixture(&path);

        let file = File::open_rw(&path).unwrap();
        let reader = file.dataset("log").unwrap();
        let mut writer = file.dataset("log").unwrap();

        assert_eq!(reader.read_i32().unwrap(), vec![0, 1]);
        assert!(
            reader.chunk_cache_stats().cached_chunks() > 0,
            "the read must leave the partial trailing chunk cached, or this \
             test cannot tell a retained chunk from a re-read one"
        );

        writer.append(&[2i32, 3]).unwrap();

        assert_eq!(
            reader.chunk_cache_stats().cached_chunks(),
            0,
            "the snapshot must report what the cache holds for a read, and the \
             append left it holding nothing a read will be served"
        );
        assert_eq!(reader.shape().unwrap(), vec![4]);
        assert_eq!(reader.read_i32().unwrap(), vec![0, 1, 2, 3]);
        file.close().unwrap();
    }

    /// A clone is a second handle to the same object, and follows it the same
    /// way. It shares the chunk cache, so the edit that drops one drops both.
    #[test]
    fn a_clone_is_a_second_handle_to_the_same_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clone.h5");
        revalidation_fixture(&path);

        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("log").unwrap();
        let copy = ds.clone();
        let root = file.root();
        let root_copy = root.clone();

        assert_eq!(copy.read_i32().unwrap(), vec![0, 1]);
        assert!(copy.chunk_cache_stats().cached_chunks() > 0);
        assert_eq!(
            ds.chunk_cache_stats().cached_chunks(),
            copy.chunk_cache_stats().cached_chunks(),
            "clones share one cache: a chunk read through either is warm for both"
        );

        ds.append(&[2i32, 3]).unwrap();
        assert_eq!(copy.read_i32().unwrap(), vec![0, 1, 2, 3]);

        root.create_group("later").unwrap();
        file.commit().unwrap();
        assert!(
            root_copy.groups().unwrap().contains(&"later".to_string()),
            "a cloned group handle follows its group across a commit too"
        );
        file.close().unwrap();
    }

    /// A handle to an object a commit deleted has nothing to answer for. The
    /// bytes it vacated still parse as the dataset that left them, so reading
    /// them would answer with data no longer in the file.
    #[test]
    fn a_handle_to_a_deleted_object_refuses_rather_than_reading_what_it_left() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deleted.h5");
        revalidation_fixture(&path);

        let file = File::open_rw(&path).unwrap();
        let ds = file.dataset("plain").unwrap();
        assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]);

        file.root().delete("plain").unwrap();
        file.commit().unwrap();

        assert!(
            matches!(
                ds.read_i32(),
                Err(Error::Format(FormatError::PathNotFound(ref p))) if p == "plain"
            ),
            "reading a deleted dataset must fail the way opening it does, got {:?}",
            ds.read_i32()
        );
        file.close().unwrap();
    }

    /// A handle reached by object reference has no name to look itself up by, so
    /// it pins to the address the reference gave it. An immediate append leaves
    /// that address alone and it keeps reading; a commit can move the header and
    /// it stops.
    #[test]
    fn a_reference_handle_reads_until_a_commit_could_have_moved_its_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("by_ref.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("log")
            .with_i32_data(&[0, 1])
            .with_shape(&[2])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[4]);
        b.create_dataset("refs").with_path_references(&["log"]);
        b.write(&path).unwrap();

        let file = File::open_rw(&path).unwrap();
        let mut by_ref = match file
            .dataset("refs")
            .unwrap()
            .dereference()
            .unwrap()
            .remove(0)
        {
            Object::Dataset(ds) => *ds,
            other => panic!("expected a dataset, got {other:?}"),
        };
        assert_eq!(by_ref.read_i32().unwrap(), vec![0, 1]);

        // An in-place append rewrites the header where it stands, so the address
        // is still the object's and the handle reads its own new elements.
        by_ref.append(&[2i32, 3]).unwrap();
        assert_eq!(by_ref.read_i32().unwrap(), vec![0, 1, 2, 3]);

        // A commit can put the header somewhere else, and nothing on disk marks
        // the bytes it vacated as dead.
        file.root().create_group("g").unwrap();
        file.commit().unwrap();
        assert!(
            matches!(by_ref.read_i32(), Err(Error::StaleHandle)),
            "unexpected: {:?}",
            by_ref.read_i32()
        );

        // And the recovery the error names: dereference again, against the file
        // the commit left.
        let fresh = match file
            .dataset("refs")
            .unwrap()
            .dereference()
            .unwrap()
            .remove(0)
        {
            Object::Dataset(ds) => *ds,
            other => panic!("expected a dataset, got {other:?}"),
        };
        assert_eq!(fresh.read_i32().unwrap(), vec![0, 1, 2, 3]);
        file.close().unwrap();
    }

    /// The counters are what every handle trusts, so every entry point that
    /// reaches the write engine has to declare what it does to them. This is
    /// that declaration, written out entry point by entry point and checked
    /// against the code: a [`Change::Relocating`] advances both counters, an
    /// [`Change::InPlace`] only the content one, and a [`Change::Nothing`]
    /// neither.
    ///
    /// The table is the point. An entry point classified too weakly leaves a
    /// handle memoizing a header it moved; one classified too strongly ends
    /// every by-reference handle in the session for nothing — which is what
    /// `File::sync` did until its line was written here.
    #[test]
    fn every_write_entry_point_declares_what_it_changes() {
        let dir = tempfile::tempdir().unwrap();

        type Step = (&'static str, Change, fn(&File));
        let steps: Vec<Step> = vec![
            ("File::sync", Change::Nothing, |f| {
                f.sync().unwrap();
            }),
            ("Dataset::chunk_cache_stats", Change::Nothing, |f| {
                let _ = f.dataset("log").unwrap().chunk_cache_stats();
            }),
            ("Dataset::reset_chunk_cache_stats", Change::Nothing, |f| {
                f.dataset("log").unwrap().reset_chunk_cache_stats();
            }),
            ("BufferedAppender::new", Change::Nothing, |f| {
                let mut ds = f.dataset("log").unwrap();
                let app = ds.buffered_appender().unwrap();
                app.discard();
            }),
            ("Dataset::append", Change::InPlace, |f| {
                f.dataset("log").unwrap().append(&[9i32]).unwrap();
            }),
            ("Dataset::append_raw", Change::InPlace, |f| {
                f.dataset("log")
                    .unwrap()
                    .append_raw(&7i32.to_le_bytes())
                    .unwrap();
            }),
            ("BufferedAppender::flush", Change::InPlace, |f| {
                let mut ds = f.dataset("log").unwrap();
                let mut app = ds.buffered_appender().unwrap();
                app.append(&[5i32, 6, 7, 8]).unwrap();
                app.flush().unwrap();
            }),
            ("Dataset::write", Change::Relocating, |f| {
                f.dataset("plain").unwrap().write(&[1i32, 2, 3]).unwrap();
            }),
            ("Dataset::set_attr", Change::Relocating, |f| {
                f.dataset("plain")
                    .unwrap()
                    .set_attr("a", AttrValue::I32(1))
                    .unwrap();
            }),
            ("Dataset::remove_attr", Change::Relocating, |f| {
                let mut ds = f.dataset("plain").unwrap();
                ds.set_attr("gone", AttrValue::I32(1)).unwrap();
                f.commit().unwrap();
                ds.remove_attr("gone").unwrap();
            }),
            ("Dataset::write_staged", Change::Relocating, |f| {
                f.dataset("plain")
                    .unwrap()
                    .write_staged(|b| {
                        b.with_i32_data(&[4, 5, 6]);
                    })
                    .unwrap();
            }),
            ("Dataset::append_staged", Change::Relocating, |f| {
                f.dataset("log")
                    .unwrap()
                    .append_staged(|b| {
                        b.append_i32(&[3]);
                    })
                    .unwrap();
            }),
            ("Group::create_group", Change::Relocating, |f| {
                f.root().create_group("fresh").unwrap();
            }),
            ("Group::create_dataset", Change::Relocating, |f| {
                f.root()
                    .create_dataset("made", |b| {
                        b.with_i32_data(&[1]);
                    })
                    .unwrap();
            }),
            ("Group::delete", Change::Relocating, |f| {
                f.root().delete("plain").unwrap();
            }),
            ("Group::set_attr", Change::Relocating, |f| {
                f.root().set_attr("a", AttrValue::I32(1)).unwrap();
            }),
            ("Group::remove_attr", Change::Relocating, |f| {
                f.root().set_attr("gone", AttrValue::I32(1)).unwrap();
                f.commit().unwrap();
                f.root().remove_attr("gone").unwrap();
            }),
            ("Group::create_group_with", Change::Relocating, |f| {
                f.root()
                    .create_group_with("built", |g| {
                        g.set_attr("a", AttrValue::I32(1));
                    })
                    .unwrap();
            }),
            ("File::copy", Change::Relocating, |f| {
                f.copy("plain", "copied").unwrap();
            }),
            ("File::commit", Change::Relocating, |f| {
                f.root().create_group("committed").unwrap();
                f.commit().unwrap();
            }),
        ];

        let revisions = |f: &File| (f.inner.content_revision(), f.inner.address_revision());
        let declared = |c: Change| match c {
            Change::Relocating => "Relocating",
            Change::InPlace => "InPlace",
            Change::Nothing => "Nothing",
        };
        for (name, change, step) in steps {
            let path = dir.path().join(format!("{}.h5", name.replace("::", "_")));
            revalidation_fixture(&path);
            let file = File::open_rw(&path).unwrap();
            let before = revisions(&file);
            step(&file);
            let after = revisions(&file);
            let observed = match (after.0 > before.0, after.1 > before.1) {
                (true, true) => "Relocating",
                (true, false) => "InPlace",
                (false, false) => "Nothing",
                (false, true) => "an address move with no content change",
            };
            assert_eq!(
                observed,
                declared(change),
                "{name} is declared here as one thing and behaves as another \
                 ({before:?} -> {after:?})"
            );
            file.close().unwrap();
        }

        // `close` consumes the file, so it cannot be a row above. It is two
        // operations: the commit it makes, which relocates like any other, and
        // the teardown, which re-homes free space and releases status flags
        // where they stand. Both counters therefore move, and the content one
        // moves twice.
        let path = dir.path().join("File_close.h5");
        revalidation_fixture(&path);
        let file = File::open_rw(&path).unwrap();
        let before = revisions(&file);
        let by_path = file.dataset("plain").unwrap();
        let by_ref = {
            let mut b = FileBuilder::new();
            b.create_dataset("d").with_i32_data(&[1, 2]);
            b.create_dataset("refs").with_path_references(&["d"]);
            let refs_path = dir.path().join("File_close_refs.h5");
            b.write(&refs_path).unwrap();
            let refs = File::open_rw(&refs_path).unwrap();
            let handle = match refs
                .dataset("refs")
                .unwrap()
                .dereference()
                .unwrap()
                .remove(0)
            {
                Object::Dataset(ds) => *ds,
                other => panic!("expected a dataset, got {other:?}"),
            };
            refs.close().unwrap();
            handle
        };
        // `close` consumes its `File`; the counters live on the shared inner
        // state every handle holds, so read them back through one of those.
        let inner = Arc::clone(&file.inner);
        file.close().unwrap();
        let after = (inner.content_revision(), inner.address_revision());
        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            (2, 1),
            "File::close is a Relocating commit and an InPlace teardown"
        );
        assert_eq!(
            by_path.read_i32().unwrap(),
            vec![7, 8, 9],
            "`close` promises reads through surviving handles still work"
        );
        assert!(
            matches!(by_ref.read_i32(), Err(Error::StaleHandle)),
            "the one handle that cannot follow the commit `close` makes: {:?}",
            by_ref.read_i32()
        );
    }

    /// What the *second* counter buys, which nothing else pins: an edit that
    /// moves no object header leaves a by-reference handle — the one kind that
    /// cannot look itself up again — still able to read. Collapse the two
    /// counters into one and every case here becomes `StaleHandle`.
    #[test]
    fn an_edit_that_moves_no_header_leaves_a_reference_handle_reading() {
        let dir = tempfile::tempdir().unwrap();

        // Named for the same reason `Step` above is: a bare tuple of a name and
        // a function pointer reads as noise at the call site.
        type Case = (&'static str, fn(&File));
        let cases: Vec<Case> = vec![
            (
                "an append through another handle to the same dataset",
                |f| {
                    f.dataset("log").unwrap().append(&[9i32]).unwrap();
                },
            ),
            ("an append to a different dataset", |f| {
                f.dataset("other").unwrap().append(&[9i32]).unwrap();
            }),
            ("a durability barrier", |f| {
                f.sync().unwrap();
            }),
        ];

        for (name, edit) in cases {
            let path = dir.path().join(format!("{}.h5", name.replace(' ', "_")));
            let mut b = FileBuilder::new();
            for ds in ["log", "other"] {
                b.create_dataset(ds)
                    .with_i32_data(&[0, 1])
                    .with_shape(&[2])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[4]);
            }
            b.create_dataset("refs").with_path_references(&["log"]);
            b.write(&path).unwrap();

            let file = File::open_rw(&path).unwrap();
            let by_ref = match file
                .dataset("refs")
                .unwrap()
                .dereference()
                .unwrap()
                .remove(0)
            {
                Object::Dataset(ds) => *ds,
                other => panic!("expected a dataset, got {other:?}"),
            };
            assert_eq!(by_ref.read_i32().unwrap(), vec![0, 1], "{name}: before");
            edit(&file);
            assert!(
                by_ref.read_i32().is_ok(),
                "{name} moves no object header, so it must not end a handle that \
                 names its dataset by address: {:?}",
                by_ref.read_i32()
            );
            file.close().unwrap();
        }
    }

    /// A commit can put something else at a handle's path — issue #305 makes a
    /// dataset and a group interchangeable in one commit. The refusal has to
    /// hold on *every* call: a header memoized and then refused is the answer
    /// each later call short-circuits on, and this handle would go on serving
    /// the other object's header without an error.
    #[test]
    fn a_handle_whose_path_becomes_a_group_keeps_refusing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("replaced.h5");
        revalidation_fixture(&path);

        let file = File::open_rw(&path).unwrap();
        let ds = file.dataset("plain").unwrap();
        assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]);

        file.root().delete("plain").unwrap();
        file.root()
            .create_group_with("plain", |g| {
                g.set_attr("i_am_a_group", AttrValue::I32(42));
            })
            .unwrap();
        file.commit().unwrap();

        for call in 1..=3 {
            assert!(
                matches!(ds.attrs(), Err(Error::NotADataset(ref p)) if p == "plain"),
                "call {call} answered {:?}",
                ds.attrs()
            );
            assert!(matches!(ds.read_i32(), Err(Error::NotADataset(_))));
            assert!(matches!(ds.shape(), Err(Error::NotADataset(_))));
        }
        file.close().unwrap();
    }

    /// A read-only file cannot change under a handle, so nothing a reader does
    /// may move the counters — and no handle on one ever pays to re-resolve.
    #[test]
    fn reading_never_moves_the_file_on() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("readonly.h5");
        revalidation_fixture(&path);

        let file = File::open(&path).unwrap();
        let _ = read_everything(&file);
        let ds = file.dataset("log").unwrap();
        let _ = ds.read_i32().unwrap();
        let _ = ds.attrs().unwrap();
        let _ = ds.layout().unwrap();
        let _ = file.root().groups().unwrap();
        assert_eq!(
            (file.inner.content_revision(), file.inner.address_revision()),
            (0, 0),
            "a read must not tell every handle its memo has expired"
        );
    }

    // -----------------------------------------------------------------------
    // Opening an object as the wrong kind (issue #352)
    // -----------------------------------------------------------------------

    /// A dataset at the root and a dataset one level down inside a group.
    ///
    /// Shared by the two sections below: opening one of those datasets *as* a
    /// group is issue #352, and resolving a path *through* one is issue #365, so
    /// the sibling refusals are provably about the same file.
    fn nested_dataset_bytes() -> Vec<u8> {
        let mut b = FileBuilder::new();
        b.create_dataset("plain").with_i32_data(&[1]);
        let mut g = b.create_group("g");
        g.create_dataset("inner").with_i32_data(&[2]);
        b.add_group(g.finish());
        b.finish().unwrap()
    }

    /// The issue: a by-name group lookup took whatever the name resolved to.
    /// `H5Gopen` fails on a non-group, and so must this — at the lookup, where
    /// the caller can act on it, rather than at some later call on a handle that
    /// was never a group.
    ///
    /// The refusal matters most on the calls that did *not* fail: `attrs()`
    /// through such a handle answered with the dataset's attributes, which is a
    /// wrong answer rather than an error.
    #[test]
    fn opening_a_dataset_as_a_group_is_refused() {
        let file = File::from_bytes(nested_dataset_bytes()).unwrap();
        let nested = file.group("g").unwrap();

        // Both by-name forms — from the file by path, and from a group by child
        // name — at the root and one level down. Each is its own lookup, and
        // each was missing the check. The refusal names the object, which is
        // what the old `PathNotFound("object header is not a group")` could not.
        for (label, named, got) in [
            ("File::group", "plain", file.group("plain")),
            ("Group::group", "plain", file.root().group("plain")),
            ("File::group nested", "g/inner", file.group("g/inner")),
            (
                "Group::group from a subgroup",
                "inner",
                nested.group("inner"),
            ),
        ] {
            assert!(
                matches!(&got, Err(Error::NotAGroup(p)) if p == named),
                "{label} answered {:?}",
                got.map(|_| "a group")
            );
        }

        // The name it reports is the normalized one, so the same object refused
        // at a lookup and refused through a live handle names itself the same
        // way — a handle holds only the normalized path.
        assert!(matches!(file.group("/plain/"), Err(Error::NotAGroup(ref p)) if p == "plain"));

        // A name that resolves to nothing stays distinct from one that resolves
        // to the wrong kind: the second reports what is there.
        assert!(matches!(
            file.group("absent"),
            Err(Error::Format(FormatError::PathNotFound(_)))
        ));
        assert!(matches!(
            file.root().group("absent"),
            Err(Error::Format(FormatError::PathNotFound(_)))
        ));

        // And a real group still opens, by either form.
        assert!(file.group("g").is_ok());
        assert!(file.root().group("g").is_ok());
    }

    /// A v1 symbol-table group must keep opening by name.
    ///
    /// The predicate that decides a lookup was, until this change, only a filter
    /// over a listing, where failing to recognise a form merely left a group out.
    /// Gating the lookup on it makes each form it names load-bearing, and the v1
    /// form is the one with no writer here to produce it — the bytes are a
    /// fixture, and a classifier that forgot the symbol table would refuse every
    /// group in every file written before the 1.8 format.
    #[test]
    fn a_symbol_table_group_still_opens_by_name() {
        let file =
            File::from_bytes(include_bytes!("../tests/data/unattributed/two_groups.h5").to_vec())
                .unwrap();

        // Names, not a count: this file holds two one-child groups, so a lookup
        // that classified correctly and then took its sibling's address would
        // pass any count worth asserting.
        for lookup in [file.group("group1"), file.root().group("group1")] {
            assert_eq!(lookup.unwrap().datasets().unwrap(), ["values"]);
        }
    }

    /// A committed datatype is neither a dataset nor a group, so it separates
    /// "is a group" from "is not a dataset" — a check written as the latter
    /// would let this one through.
    #[test]
    fn opening_a_named_datatype_as_a_group_is_refused() {
        let mut b = FileBuilder::new();
        b.commit_datatype("mytype", crate::make_i32_type());
        let file = File::from_bytes(b.finish().unwrap()).unwrap();

        assert_eq!(file.root().named_datatypes().unwrap(), vec!["mytype"]);
        assert!(matches!(file.group("mytype"), Err(Error::NotAGroup(_))));
        assert!(matches!(
            file.root().group("mytype"),
            Err(Error::NotAGroup(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Opening something else as a named datatype (issue #364)
    // -----------------------------------------------------------------------

    /// An object header carrying exactly `types`, and nothing that would make it
    /// parse: the predicate below reads message types and no message body.
    fn header_of(types: &[MessageType]) -> ObjectHeader {
        ObjectHeader {
            version: 2,
            messages: types
                .iter()
                .map(|&msg_type| crate::object_header::HeaderMessage {
                    msg_type,
                    size: 0,
                    flags: 0,
                    creation_order: None,
                    data: Vec::new(),
                })
                .collect(),
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        }
    }

    /// The rule the listing and both by-name lookups now share, stated over the
    /// message combinations rather than over one file's children.
    ///
    /// Two of these cannot be produced by any writer, here or in the reference
    /// library, which is why this is a predicate test and not another fixture. A
    /// header carrying links *and* a datatype is a group to the C library, which
    /// asks whether it is a group before asking whether it is a datatype; a
    /// header carrying neither is no object class at all, and must not become a
    /// datatype by default.
    #[test]
    fn a_committed_datatype_is_a_datatype_that_is_neither_dataset_nor_group() {
        for (label, types, expected) in [
            ("a committed datatype", &[MessageType::Datatype][..], true),
            (
                "a dataset, whose element type is a datatype message too",
                &[MessageType::Datatype, MessageType::DataLayout],
                false,
            ),
            ("a group with a link table", &[MessageType::LinkInfo], false),
            (
                "a group with a symbol table",
                &[MessageType::SymbolTable],
                false,
            ),
            (
                "a group carrying a datatype",
                &[MessageType::LinkInfo, MessageType::Datatype],
                false,
            ),
            (
                "a header with no datatype at all",
                &[MessageType::Dataspace],
                false,
            ),
        ] {
            assert_eq!(is_named_datatype(&header_of(types)), expected, "{label}");
        }
    }

    /// The issue: the by-name datatype lookups asked only whether the child had
    /// a datatype message. Every dataset does — its element type — so a dataset
    /// answered, and the two entry points disagreed with the
    /// `named_datatypes()` listing about the same child.
    #[test]
    fn a_child_that_is_not_a_committed_datatype_is_refused_by_name() {
        let mut b = FileBuilder::new();
        b.commit_datatype("mytype", crate::make_i32_type());
        b.create_dataset("typed")
            .with_i32_data(&[1, 2, 3])
            .with_committed_datatype("mytype");
        b.create_dataset("plain").with_f64_data(&[1.0]);
        let g = b.create_group("g").finish();
        b.add_group(g);
        let file = File::from_bytes(b.finish().unwrap()).unwrap();
        let root = file.root();

        // The listing is the contract both lookups now share, so it is what the
        // refusals below have to agree with.
        assert_eq!(root.named_datatypes().unwrap(), ["mytype"]);

        // A dataset, a dataset carrying that very type, and a group: three kinds
        // that are not a committed datatype, against both entry points. Neither
        // had the check, so each needs its own assertion.
        for name in ["typed", "plain", "g"] {
            let got = root.named_datatype(name);
            assert!(
                matches!(&got, Err(Error::NotANamedDatatype(p)) if p == name),
                "named_datatype({name:?}) answered {got:?}"
            );
            let got = root.named_datatype_references(name);
            assert!(
                matches!(&got, Err(Error::NotANamedDatatype(p)) if p == name),
                "named_datatype_references({name:?}) answered {got:?}"
            );
        }

        // A name that reaches nothing stays distinct from one that reaches the
        // wrong kind, as it is for `group` and `dataset`.
        assert!(matches!(
            root.named_datatype("absent"),
            Err(Error::Format(FormatError::PathNotFound(_)))
        ));
        assert!(matches!(
            root.named_datatype_references("absent"),
            Err(Error::Format(FormatError::PathNotFound(_)))
        ));

        // And the committed type still reads, by both entry points — the value,
        // not merely `is_ok`, since a lookup that refused everything would pass
        // every assertion above.
        assert_eq!(
            root.named_datatype("mytype").unwrap(),
            crate::make_i32_type()
        );
        assert_eq!(root.named_datatype_references("mytype").unwrap(), 2);
    }

    /// The mirror of [`a_handle_whose_path_becomes_a_group_keeps_refusing`]: a
    /// commit can leave a live group handle's path naming a dataset (issue
    /// #305), which is the one way past the lookup check above. The handle
    /// re-resolves, finds the wrong kind, and reports it on every call rather
    /// than serving the dataset's header as a group's.
    #[test]
    fn a_group_handle_whose_path_becomes_a_dataset_keeps_refusing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("replaced_group.h5");
        revalidation_fixture(&path);

        let file = File::open_rw(&path).unwrap();
        let group = file.group("g").unwrap();
        assert!(group.datasets().unwrap().is_empty());

        file.root().delete("g").unwrap();
        file.root()
            .create_dataset("g", |b| {
                b.with_i32_data(&[7]);
            })
            .unwrap();
        file.commit().unwrap();

        // `attrs` every time round: it is the call that answered with the
        // dataset's attributes instead of failing, and a check installed after
        // the memo rather than before it would let the second call through.
        for call in 1..=3 {
            assert!(
                matches!(group.attrs(), Err(Error::NotAGroup(ref p)) if p == "g"),
                "call {call} answered {:?}",
                group.attrs()
            );
        }
        // The rest of the read surface funnels through the same re-resolve, so
        // once each is enough to say the refusal is the group's, not `attrs`'s.
        assert!(matches!(group.datasets(), Err(Error::NotAGroup(_))));
        assert!(matches!(group.groups(), Err(Error::NotAGroup(_))));
        assert!(matches!(
            group.dataset("anything"),
            Err(Error::NotAGroup(_))
        ));

        // The path still names something, and that something still opens as
        // what it now is.
        assert_eq!(file.dataset("g").unwrap().read_i32().unwrap(), vec![7]);
        file.close().unwrap();
    }

    // -----------------------------------------------------------------------
    // Resolving a path *through* something that is not a group (issue #365)
    // -----------------------------------------------------------------------

    /// The issue: resolution opens each component in turn to look the next one
    /// up inside it, and reported a component that is not a group as
    /// `PathNotFound("object header is not a group")` — one string for every
    /// such path, naming no component at all. It read as "this path does not
    /// exist" where the truth was "`plain` is a dataset".
    ///
    /// It is now the same [`Error::NotAGroup`] a *final* component that is not a
    /// group returns (issue #352), so one match covers a path that goes wrong
    /// anywhere along it, and it names the object that stopped the walk rather
    /// than the path that was asked for.
    #[test]
    fn a_path_through_a_non_group_names_the_object_that_stopped_it() {
        let file = File::from_bytes(nested_dataset_bytes()).unwrap();

        // Two entry points, because each resolves the path for itself, and two
        // depths, because the name is the whole prefix walked rather than the
        // one component: `g/inner` is a path the caller can go and open, where a
        // bare `inner` would not say where to find it.
        for (asked, stopper) in [("plain/sub", "plain"), ("g/inner/deeper", "g/inner")] {
            for (entry, got) in [
                ("File::group", file.group(asked).map(|_| ())),
                ("File::dataset", file.dataset(asked).map(|_| ())),
            ] {
                assert!(
                    matches!(&got, Err(Error::NotAGroup(p)) if p == stopper),
                    "{entry}({asked:?}) answered {got:?}, expected the stop at {stopper:?}"
                );
            }
        }

        // A component that names nothing at all stays a `PathNotFound` naming
        // it. The two are different facts, and reading differently is the whole
        // point of the change.
        assert!(matches!(
            file.group("absent/sub"),
            Err(Error::Format(FormatError::PathNotFound(ref p))) if p == "absent"
        ));

        // Empty components are dropped before the walk, so the object is named
        // the same way however the path was spelled.
        assert!(matches!(
            file.group("/plain//sub/"),
            Err(Error::NotAGroup(ref p)) if p == "plain"
        ));

        // The name reaches a caller that only prints the error, too.
        let printed = file
            .group("g/inner/deeper")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(printed.contains("g/inner"), "the message read {printed:?}");

        // And a path that really does run through groups still resolves, so the
        // classification has not turned the walk itself into a refusal.
        assert_eq!(file.dataset("g/inner").unwrap().read_i32().unwrap(), [2]);

        // A committed datatype is neither a dataset nor a group, so it separates
        // "is not a group" from "is a dataset" for an intermediate component the
        // way it does for a final one.
        let mut b = FileBuilder::new();
        b.commit_datatype("mytype", crate::make_i32_type());
        let typed = File::from_bytes(b.finish().unwrap()).unwrap();
        assert!(matches!(
            typed.group("mytype/sub"),
            Err(Error::NotAGroup(ref p)) if p == "mytype"
        ));
    }

    /// The streaming walk is a second copy of the same loop, reading each header
    /// from a `Source`. A fix applied to one and not the other would leave the
    /// backend that exists for files too large to buffer reporting the old
    /// string.
    #[test]
    fn the_streaming_walk_names_the_object_that_stopped_it_too() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested.h5");
        std::fs::write(&path, nested_dataset_bytes()).unwrap();

        let file = File::open_streaming(&path).unwrap();
        for (asked, stopper) in [("plain/sub", "plain"), ("g/inner/deeper", "g/inner")] {
            let got = file.group(asked).map(|_| ());
            assert!(
                matches!(&got, Err(Error::NotAGroup(p)) if p == stopper),
                "group({asked:?}) answered {got:?}"
            );
        }
        assert!(matches!(
            file.group("absent/sub"),
            Err(Error::Format(FormatError::PathNotFound(ref p))) if p == "absent"
        ));
        assert_eq!(file.dataset("g/inner").unwrap().read_i32().unwrap(), [2]);
        // Before the directory goes: an open file blocks its removal on Windows.
        drop(file);
    }

    /// The root is the one group handle nothing classifies: `File::root` takes
    /// the address from the superblock, and no open validates that it names a
    /// group. So a file whose superblock points the root at a dataset reaches
    /// the refusal in `Group::child_address` that every other handle is kept
    /// away from — and names the root by the empty path it carries.
    #[test]
    fn a_root_that_is_not_a_group_refuses_rather_than_being_searched() {
        let mut bytes = nested_dataset_bytes();
        let sig = crate::signature::find_signature(&bytes).unwrap();
        let mut sb = crate::superblock::Superblock::parse(&bytes, sig).unwrap();
        // The fixture has no userblock, so its base address is zero and the
        // absolute address a walk returns is also the stored one the superblock
        // field wants.
        assert_eq!(sb.base_address, BaseAddress::ZERO);
        sb.root_group_address = group_v2::resolve_path_any(&bytes, &sb, "plain").unwrap();
        let rewritten = sb.serialize();
        bytes[sig..sig + rewritten.len()].copy_from_slice(&rewritten);

        let file = File::from_bytes(bytes).unwrap();
        let got = file.root().dataset("anything").map(|_| ());
        assert!(
            matches!(&got, Err(Error::NotAGroup(p)) if p.is_empty()),
            "a root that is not a group must refuse, got {got:?}"
        );
    }

    /// Read everything a read-write file can serve through the paired read
    /// paths, as comparable text.
    ///
    /// Each entry exercises a different `with_engine` call site: path
    /// resolution, object-header parsing, group listing, attribute reads (both
    /// the compact and the dense form), a whole-dataset read, and a row-range
    /// read. Errors are formatted rather than unwrapped so that a *divergence in
    /// which error* is reported also fails the comparison.
    fn read_everything(file: &File) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!("root groups: {:?}", file.root().groups()));
        out.push(format!("root datasets: {:?}", file.root().datasets()));
        out.push(format!("root attrs: {:?}", sorted(file.root().attrs())));

        // `plain/nope` runs the walk through a dataset (issue #365): the two
        // backends must refuse it with the same error as well as agree on the
        // reads that succeed.
        for path in [
            "plain",
            "g/nested",
            "many_attrs",
            "missing",
            "g/missing",
            "plain/nope",
        ] {
            match file.dataset(path) {
                Ok(ds) => {
                    out.push(format!("{path}: shape {:?}", ds.shape()));
                    out.push(format!("{path}: attrs {:?}", sorted(ds.attrs())));
                    out.push(format!("{path}: all {:?}", ds.read_i32()));
                    out.push(format!("{path}: rows {:?}", ds.read_i32_rows(1, 2)));
                    out.push(format!("{path}: raw rows {:?}", ds.read_raw_rows(0, 1)));
                }
                Err(e) => out.push(format!("{path}: error {e}")),
            }
        }
        out
    }

    /// Attribute maps compare only after ordering; `HashMap`'s `Debug` is not
    /// deterministic, and an ordering difference here would be noise rather
    /// than the divergence this is looking for.
    fn sorted(attrs: Result<HashMap<String, AttrValue>, Error>) -> Vec<String> {
        match attrs {
            Ok(map) => {
                let mut v: Vec<String> =
                    map.iter().map(|(k, val)| format!("{k}={val:?}")).collect();
                v.sort();
                v
            }
            Err(e) => vec![format!("error {e}")],
        }
    }

    /// Drive `bytes` down both forms of every read a read-write file serves and
    /// require identical answers.
    ///
    /// Every read has a slice form (walking the whole-file mirror) and a
    /// `Source` form, and until a mirrorless backing lands (issue #198) only the
    /// slice form ever runs. This makes the other form reachable now, so it
    /// cannot quietly drift as its twin is edited.
    fn assert_both_read_paths_agree(bytes: &[u8], what: &str) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("both.h5");
        std::fs::write(&path, bytes).unwrap();

        // One session at a time: `open_rw` takes an exclusive lock, and holding
        // two over one path fails outright where OS locks are mandatory.
        let via_mirror = {
            let f = File::open_rw(&path).unwrap();
            read_everything(&f)
        };
        let via_source = {
            let f = File::open_rw_source_only(&path).unwrap();
            read_everything(&f)
        };

        assert_eq!(
            via_mirror.len(),
            via_source.len(),
            "{what}: the two read paths produced different numbers of results"
        );
        for (m, s) in via_mirror.iter().zip(&via_source) {
            assert_eq!(m, s, "{what}: slice and Source read paths disagree");
        }
        // Guard the guard: a helper that read nothing would make the comparison
        // vacuous, and a file whose datasets all failed to open would too.
        assert!(
            via_mirror.iter().any(|r| r.contains("all Ok(")),
            "{what}: no dataset read succeeded, so this compared nothing"
        );
    }

    /// A file exercising each paired read: a plain dataset, a nested one behind
    /// a group (path resolution), and one carrying enough attributes to force
    /// the dense (fractal-heap) attribute layout rather than compact messages.
    fn both_paths_file_bytes(userblock: Option<u64>) -> Vec<u8> {
        let mut b = FileBuilder::new();
        if let Some(ub) = userblock {
            b.with_userblock(ub);
        }
        b.create_dataset("plain")
            .with_i32_data(&(0..24).collect::<Vec<i32>>())
            .with_shape(&[6, 4])
            .set_attr("units", AttrValue::String("m".into()));
        // Well past the eight-attribute compact limit, so the header converts to
        // the dense layout and the dense extraction path is the one that runs.
        {
            let ds = b
                .create_dataset("many_attrs")
                .with_i32_data(&(0..8).collect::<Vec<i32>>());
            for i in 0..24 {
                ds.set_attr(&format!("attr_{i:02}"), AttrValue::I64(i));
            }
        }
        let mut g = b.create_group("g");
        g.create_dataset("nested")
            .with_i32_data(&(100..112).collect::<Vec<i32>>())
            .with_shape(&[3, 4]);
        b.add_group(g.finish());
        b.finish().unwrap()
    }

    #[test]
    fn both_read_paths_agree() {
        assert_both_read_paths_agree(&both_paths_file_bytes(None), "no userblock");
    }

    /// The userblock case is the one where the two forms are built differently:
    /// the slice form reframes by slicing at the base address, the `Source` form
    /// wraps in a `BaseOffsetSource`. A file with a nonzero base is the only way
    /// to compare them.
    #[test]
    fn both_read_paths_agree_with_a_userblock() {
        assert_both_read_paths_agree(&both_paths_file_bytes(Some(512)), "512-byte userblock");
    }

    /// One 256-element i32 dataset, chunked into 32-element chunks, in memory.
    fn chunked_file_bytes() -> Vec<u8> {
        let data: Vec<i32> = (0..256).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("chunked")
            .with_i32_data(&data)
            .with_shape(&[256])
            .with_chunks(&[32]);
        b.finish().unwrap()
    }

    // The DAPL override must drive the *live* `ChunkCache`, not merely the value
    // reported by `chunk_cache_config()`. These assertions reach the crate's
    // `#[cfg(test)]` cache introspection (unavailable to integration tests), so
    // they fail if the resolved config ever stops flowing into the real cache.

    #[test]
    fn enabled_override_populates_live_cache_over_disabled_file_default() {
        let file = File::from_bytes_with_options(
            chunked_file_bytes(),
            FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
        )
        .unwrap();

        let ds = file
            .dataset_with_options(
                "chunked",
                DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
            )
            .unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

        // The enabled override built the chunk index and retained chunks; the
        // disabled file default would have left both empty.
        assert!(ds.chunk_cache_stats().index_loaded());
        assert!(ds.chunk_cache_stats().cached_chunks() > 0);
    }

    #[test]
    fn disabled_override_suppresses_live_cache_over_enabled_file_default() {
        let file = File::from_bytes_with_options(
            chunked_file_bytes(),
            FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
        )
        .unwrap();

        let ds = file
            .dataset_with_options(
                "chunked",
                DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
            )
            .unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

        // The disabled override suppressed the index and chunk retention; the
        // enabled file default would have populated both.
        assert!(!ds.chunk_cache_stats().index_loaded());
        assert_eq!(ds.chunk_cache_stats().cached_chunks(), 0);
    }

    /// A group child whose stored (base-relative) object-header address overflows
    /// `u64` once the base address is added must be rejected, not wrapped or
    /// panicked on. Reaching this needs a nonzero base address, so the file
    /// carries a userblock; the child link's stored address is then rewritten to
    /// `HADDR_UNDEF` (all ones) so `group_children`'s normalization overflows.
    #[test]
    fn group_child_address_base_overflow_is_rejected() {
        const UB: u64 = 512;
        let mut b = FileBuilder::new();
        b.with_userblock(UB);
        let mut child = b.create_group("child");
        child.create_dataset("inner").with_i32_data(&[1, 2, 3]);
        b.add_group(child.finish());
        let mut bytes = b.finish().unwrap();

        // Baseline: the file reads and the subgroup is listed.
        let file = File::from_bytes(bytes.clone()).unwrap();
        assert_eq!(file.root().groups().unwrap(), vec!["child".to_string()]);

        // Rewrite the child's stored object-header address to HADDR_UNDEF. It is
        // stored base-relative (absolute minus the userblock base) and, for this
        // single-child file, appears exactly once in the bytes. The link lives in
        // the root object header's chunk-0.
        let stored = file
            .root()
            .group("child")
            .unwrap()
            .header_address()
            .unwrap()
            - UB;
        let needle = stored.to_le_bytes();
        let matches: Vec<usize> = bytes
            .windows(8)
            .enumerate()
            .filter(|(_, w)| *w == needle)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "stored child address {stored:#x} was not uniquely locatable: {matches:?}"
        );
        bytes[matches[0]..matches[0] + 8].copy_from_slice(&u64::MAX.to_le_bytes());

        // The v2 object header is checksum-protected, so a real crafted file would
        // carry a matching checksum; recompute the root header's over the edited
        // bytes so parsing reaches the address normalization rather than failing on
        // the checksum first. Mirrors the chunk-0 extent from `parse_v2`.
        #[cfg(feature = "checksum")]
        {
            let root_addr = file.root().header_address().unwrap() as usize;
            assert_eq!(&bytes[root_addr..root_addr + 4], b"OHDR");
            let flags = bytes[root_addr + 5];
            let mut pos = root_addr + 6;
            if flags & 0x20 != 0 {
                pos += 16;
            }
            if flags & 0x10 != 0 {
                pos += 4;
            }
            let width = 1usize << (flags & 0x03);
            let chunk0 = (0..width).fold(0usize, |acc, i| {
                acc | ((bytes[pos + i] as usize) << (8 * i))
            });
            pos += width;
            let chunk0_end = pos + chunk0;
            assert!(
                matches[0] < chunk0_end,
                "patched link address is outside the root header's chunk-0"
            );
            let cs = crate::checksum::jenkins_lookup3(&bytes[root_addr..chunk0_end]);
            bytes[chunk0_end..chunk0_end + 4].copy_from_slice(&cs.to_le_bytes());
        }

        // Iterating the root now normalizes `u64::MAX + base` and must surface the
        // overflow as a format error rather than panicking or wrapping.
        let file = File::from_bytes(bytes).unwrap();
        match file.root().groups() {
            Err(Error::Format(FormatError::OffsetOverflow { offset, length })) => {
                assert_eq!(offset, u64::MAX);
                assert_eq!(length, UB);
            }
            other => panic!("expected group-child address overflow, got {other:?}"),
        }
    }

    /// Recompute a version-2 object header's checksum over its chunk 0, after a
    /// test has edited a message inside it.
    ///
    /// A crafted file a reader must survive carries a *valid* checksum — an
    /// attacker recomputes it — so a test that edits a header and leaves the old
    /// one measures the checksum rather than the thing it meant to.
    #[cfg(feature = "checksum")]
    fn refresh_v2_header_checksum(bytes: &mut [u8], header_addr: usize) -> std::ops::Range<usize> {
        assert_eq!(&bytes[header_addr..header_addr + 4], b"OHDR");
        let flags = bytes[header_addr + 5];
        let mut pos = header_addr + 6;
        if flags & 0x20 != 0 {
            pos += 16;
        }
        if flags & 0x10 != 0 {
            pos += 4;
        }
        let width = 1usize << (flags & 0x03);
        let chunk0 = (0..width).fold(0usize, |acc, i| {
            acc | ((bytes[pos + i] as usize) << (8 * i))
        });
        pos += width;
        let chunk0_end = pos + chunk0;
        let cs = crate::checksum::jenkins_lookup3(&bytes[header_addr..chunk0_end]);
        bytes[chunk0_end..chunk0_end + 4].copy_from_slice(&cs.to_le_bytes());
        header_addr..chunk0_end
    }

    /// A contiguous layout whose declared size disagrees with its dataspace is
    /// refused, whatever the dataset's size and whichever read asks.
    ///
    /// The whole-dataset readers have always refused it. A typed read now takes a
    /// large dataset a row window at a time, and a window only ever checks that
    /// its *own* rows are inside the declared storage — so without the shared
    /// check this refusal would have applied to small datasets, which are still
    /// read whole, and not to large ones. A validation that fires depending on
    /// the size of the input is the kind that surfaces years later as an
    /// inconsistent bug report, which is why both sizes are here.
    #[cfg(feature = "checksum")]
    #[test]
    fn a_layout_size_disagreeing_with_the_dataspace_is_refused_at_every_dataset_size() {
        // One dataset below the typed read's window budget and read whole; one
        // above it and swept.
        for n in [1000usize, 200_000] {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let mut b = crate::writer::FileBuilder::new();
            b.create_dataset("t")
                .with_f64_data(&data)
                .with_shape(&[n as u64]);
            let mut bytes = b.finish().unwrap();
            // Taken before the edit: an edited header fails its checksum, and the
            // address is needed to recompute it.
            let header_addr = {
                let file = File::from_bytes(bytes.clone()).unwrap();
                file.dataset("t").unwrap().header_address().unwrap() as usize
            };

            // The layout message's size field: the dataset's byte length, which
            // appears once in the file. The assertion is the fixture's own guard —
            // patching some other field would test nothing in particular.
            let declared = (n * 8) as u64;
            let needle = declared.to_le_bytes();
            let at: Vec<usize> = bytes
                .windows(8)
                .enumerate()
                .filter(|(_, w)| *w == needle)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                at.len(),
                1,
                "the stored size {declared} was not uniquely locatable in a {n}-element file: {at:?}"
            );
            bytes[at[0]..at[0] + 8].copy_from_slice(&(declared * 2).to_le_bytes());

            let chunk0 = refresh_v2_header_checksum(&mut bytes, header_addr);
            assert!(
                chunk0.contains(&at[0]),
                "the patched size is outside the header chunk whose checksum was refreshed"
            );

            let file = File::from_bytes(bytes).unwrap();
            let ds = file.dataset("t").unwrap();
            let expected = FormatError::DataSizeMismatch {
                expected: n * 8,
                actual: n * 16,
            };
            for (what, err) in [
                ("read_raw", ds.read_raw().unwrap_err()),
                ("read_f64", ds.read_f64().unwrap_err()),
                ("read_i32", ds.read_i32().unwrap_err()),
            ] {
                match err {
                    Error::Format(got) => assert_eq!(
                        format!("{got:?}"),
                        format!("{expected:?}"),
                        "{what} over {n} elements reported the wrong mismatch"
                    ),
                    other => {
                        panic!("{what} over {n} elements: expected a format error, got {other:?}")
                    }
                }
            }
        }
    }

    /// A zero-row window returns `Ok(empty)` uniformly across layouts, including
    /// over unallocated storage. Unallocated storage now reads as the fill value
    /// rather than erroring, so this no longer guards a cross-layout divergence
    /// in the error; what it still pins is that a window of no rows is an *empty*
    /// buffer and not a zero-length fill, which is what a caller iterating past
    /// the end of a dataset sees.
    #[test]
    fn read_rows_framed_zero_row_window_is_ok_even_when_unallocated() {
        let dl = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let ds = Dataspace {
            space_type: crate::dataspace::DataspaceType::Simple,
            rank: 1,
            dimensions: vec![0],
            max_dimensions: None,
        };
        let dt = Datatype::FixedPoint {
            size: 8,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 64,
        };
        let cache = ChunkCache::new();
        let out = read_rows_framed(
            &BytesSource::new(b""),
            RawReadSpec::plain(&dl, &ds, &dt),
            8,
            8,
            &cache,
            CachePass::LRU,
            0,
            0,
            8,
        )
        .expect("a zero-row window must be Ok(empty)");
        assert!(out.is_empty());

        // A Virtual layout is unsupported and must still error for a zero-row
        // window, matching `read_raw`, rather than being swallowed by the early
        // return.
        let virtual_dl = DataLayout::Virtual { version: 4 };
        let err = read_rows_framed(
            &BytesSource::new(b""),
            RawReadSpec::plain(&virtual_dl, &ds, &dt),
            8,
            8,
            &cache,
            CachePass::LRU,
            0,
            0,
            8,
        )
        .expect_err("a virtual layout must error even for a zero-row window");
        assert!(
            matches!(err, FormatError::UnsupportedVirtualLayout),
            "expected UnsupportedVirtualLayout, got {err:?}"
        );
    }

    /// The window a typed whole-dataset read sweeps in is a budget in *stored
    /// bytes* resolved against the dataset's own geometry, and a chunked dataset
    /// adds a second rule on top: whole chunk bands.
    ///
    /// A window that ended mid-band would make the next window decode the band
    /// again — the cost windowing exists to avoid — so the budget is rounded down
    /// to a multiple of the band, and *up* to one whole band when even one band
    /// is over budget. The three ways a window can come out are one rule with
    /// different inputs, which is why they are asserted together.
    #[test]
    fn a_typed_read_windows_in_whole_chunk_bands() {
        const BUDGET: u64 = TYPED_READ_WINDOW_BYTES;
        let ds1 = |dims: &[u64]| Dataspace {
            space_type: crate::dataspace::DataspaceType::Simple,
            rank: dims.len() as u8,
            dimensions: dims.to_vec(),
            max_dimensions: None,
        };
        let contiguous = DataLayout::Contiguous {
            address: Some(0),
            size: 0,
        };
        let chunked = |band: u32| DataLayout::Chunked {
            chunk_dimensions: vec![band, 8],
            btree_address: Some(0),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let elem = NonZeroUsize::new(8).unwrap();

        // Unchunked: the budget, divided by the row width, exactly.
        assert_eq!(
            typed_window_rows(&contiguous, &ds1(&[1 << 20]), elem)
                .unwrap()
                .get(),
            BUDGET / 8
        );

        // A band that divides the budget takes it unchanged; one that does not
        // is rounded down to a whole number of bands, never up.
        assert_eq!(
            typed_window_rows(&chunked(512), &ds1(&[1 << 20]), elem)
                .unwrap()
                .get(),
            BUDGET / 8
        );
        let rows = typed_window_rows(&chunked(300), &ds1(&[1 << 20]), elem)
            .unwrap()
            .get();
        assert_eq!(rows % 300, 0, "a window must end on a chunk band");
        assert!(
            rows <= BUDGET / 8 && rows > BUDGET / 8 - 300,
            "a window must be the largest whole number of bands within the \
             budget, not a smaller one: {rows} rows against {} in budget",
            BUDGET / 8
        );

        // One band over budget: the window is that band, since a narrower one
        // would decode it twice.
        assert_eq!(
            typed_window_rows(&chunked(1 << 20), &ds1(&[1 << 21]), elem)
                .unwrap()
                .get(),
            1 << 20
        );

        // Rows wider than the whole budget: one row, which is the least a window
        // can be and still make progress.
        assert_eq!(
            typed_window_rows(&contiguous, &ds1(&[4, 1 << 20]), elem)
                .unwrap()
                .get(),
            1
        );

        // Rank 0. A chunked layout message carries rank + 1 dimensions, so a
        // scalar's only entry is the element-size trailer and the "band" read
        // out of it is not a band at all. Nothing rests on it: a scalar has one
        // row, so `n0 <= rows` sends it to the whole read whatever this says.
        // Asserted so that a later reading of `first()` as the leading extent
        // has to account for this case rather than discover it.
        let scalar = Dataspace {
            space_type: crate::dataspace::DataspaceType::Scalar,
            rank: 0,
            dimensions: Vec::new(),
            max_dimensions: None,
        };
        assert!(typed_window_rows(&chunked(8), &scalar, elem).unwrap().get() >= 1);

        // A zero inner dimension makes a row zero bytes wide, and the dataset
        // has no elements at all: one window covers it, and nothing divides by
        // zero on the way there.
        assert_eq!(
            typed_window_rows(&contiguous, &ds1(&[4, 0]), elem)
                .unwrap()
                .get(),
            u64::MAX
        );
    }

    /// The two channels a caller has for one object's attributes.
    struct AttrChannels {
        owner: &'static str,
        /// What `attrs` decoded — the values, lossily.
        values: HashMap<String, AttrValue>,
        /// What `attr_datatypes` reported — the encodings, exactly.
        datatypes: HashMap<String, Datatype>,
    }

    /// One written file: its bytes, and both channels read back from each owner.
    struct AttrFile {
        /// The file as written, so a test can assert which storage form it got.
        bytes: Vec<u8>,
        owners: Vec<AttrChannels>,
    }

    /// Put the same attributes on the root group and on a dataset, write the
    /// file, and read both channels back from each owner.
    ///
    /// Both owners, because `Group` and `Dataset` reach their attribute messages
    /// by different routes: one parses an object header by address, the other
    /// already holds one.
    fn attr_channels(
        values: &[(&str, AttrValue)],
        verbatim: &[crate::attribute::AttributeMessage],
    ) -> AttrFile {
        let mut b = FileBuilder::new();
        for (name, value) in values {
            b.set_attr(name, value.clone());
        }
        for message in verbatim {
            b.set_attr_verbatim(message.clone());
        }
        {
            let ds = b.create_dataset("data").with_f64_data(&[1.0]);
            for (name, value) in values {
                ds.set_attr(name, value.clone());
            }
            for message in verbatim {
                ds.set_attr_verbatim(message.clone());
            }
        }
        let bytes = b.finish().unwrap();
        let file = File::from_bytes(bytes.clone()).unwrap();
        let root = file.root();
        let dataset = file.dataset("data").unwrap();
        AttrFile {
            bytes,
            owners: vec![
                AttrChannels {
                    owner: "root group",
                    values: root.attrs().unwrap(),
                    datatypes: root.attr_datatypes().unwrap(),
                },
                AttrChannels {
                    owner: "dataset",
                    values: dataset.attrs().unwrap(),
                    datatypes: dataset.attr_datatypes().unwrap(),
                },
            ],
        }
    }

    /// The two channels on the axes each one carries.
    ///
    /// Both now report an integer's width: the value channel keeps it (#350),
    /// so `count` is `I32` and its datatype is the 4-byte signed type it is
    /// stored as. What the value channel still cannot say is how those bytes are
    /// laid out — `be` holds the same number big-endian and decodes to the same
    /// `I32`, so the datatype channel is the only record that re-encoding from
    /// the value would flip its byte order. `AttrValue` is documented as lossy;
    /// this is where the loss is (#248).
    #[test]
    fn attr_datatypes_reports_the_byte_order_attrs_normalizes() {
        let be = crate::attribute::AttributeMessage {
            name: "be".into(),
            datatype: Datatype::FixedPoint {
                size: 4,
                byte_order: crate::datatype::DatatypeByteOrder::BigEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            },
            dataspace: Dataspace {
                space_type: crate::dataspace::DataspaceType::Scalar,
                rank: 0,
                dimensions: vec![],
                max_dimensions: None,
            },
            raw_data: (-7i32).to_be_bytes().to_vec(),
            datatype_location: crate::shared_message::DatatypeLocation::Inline,
        };

        for c in attr_channels(&[("count", AttrValue::I32(-7))], std::slice::from_ref(&be)).owners {
            assert_eq!(
                c.values.get("count"),
                Some(&AttrValue::I32(-7)),
                "{}: the value channel keeps the width the attribute was written at",
                c.owner
            );
            let Some(Datatype::FixedPoint { size, signed, .. }) = c.datatypes.get("count") else {
                panic!(
                    "{}: expected a fixed-point datatype, got {:?}",
                    c.owner,
                    c.datatypes.get("count")
                );
            };
            assert_eq!(
                (*size, *signed),
                (4, true),
                "{}: the datatype channel must report the width on disk",
                c.owner
            );

            assert_eq!(
                c.values.get("be"),
                Some(&AttrValue::I32(-7)),
                "{}: a big-endian attribute decodes to the value it holds",
                c.owner
            );
            let Some(Datatype::FixedPoint { byte_order, .. }) = c.datatypes.get("be") else {
                panic!(
                    "{}: expected a fixed-point datatype, got {:?}",
                    c.owner,
                    c.datatypes.get("be")
                );
            };
            assert_eq!(
                *byte_order,
                crate::datatype::DatatypeByteOrder::BigEndian,
                "{}: the datatype channel is the only record of the byte order",
                c.owner
            );
        }
    }

    /// Every attribute message is reported, including one `attrs` drops because
    /// no `AttrValue` can carry it.
    ///
    /// That is what lets a caller tell a dropped attribute from an absent one. An
    /// omission with nothing to compare against is invisible, which is how every
    /// `np.bool_` attribute in an h5py file went missing without a trace (#248).
    #[test]
    fn attr_datatypes_reports_an_attribute_attrs_omits() {
        let opaque = Datatype::Opaque {
            size: 3,
            tag: b"rgb".to_vec(),
        };
        let raw = crate::attribute::AttributeMessage {
            name: "raw".into(),
            datatype: opaque.clone(),
            dataspace: Dataspace {
                space_type: crate::dataspace::DataspaceType::Scalar,
                rank: 0,
                dimensions: vec![],
                max_dimensions: None,
            },
            raw_data: vec![1, 2, 3],
            datatype_location: crate::shared_message::DatatypeLocation::Inline,
        };

        for c in attr_channels(&[("count", AttrValue::I32(1))], std::slice::from_ref(&raw)).owners {
            assert!(
                !c.values.contains_key("raw"),
                "{}: an opaque attribute has no `AttrValue`, so `attrs` omits it — \
                 if that changes, this test is measuring the wrong thing",
                c.owner
            );
            assert_eq!(
                c.datatypes.get("raw"),
                Some(&opaque),
                "{}: the datatype channel must report an attribute `attrs` omits",
                c.owner
            );
            // Specific to the one attribute: a channel that reported only the
            // undecodable one, or dropped its neighbour, would pass the above.
            assert!(
                c.values.contains_key("count") && c.datatypes.contains_key("count"),
                "{}: the attribute beside it must appear in both channels",
                c.owner
            );
        }
    }

    /// The refusal reaches a dataset too, not only the attribute the issue was
    /// reported through: the same decoders back `Dataset::read_*`, by both the
    /// whole-dataset and the windowed route.
    #[test]
    fn a_wide_dataset_is_refused_by_the_typed_readers() {
        let dt = Datatype::FixedPoint {
            size: 16,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 128,
        };
        let mut b = FileBuilder::new();
        b.create_dataset("wide")
            .with_raw_data(dt.clone(), vec![0xFF; 32], 2);
        let file = File::from_bytes(b.finish().unwrap()).unwrap();
        let ds = file.dataset("wide").unwrap();

        assert_eq!(
            ds.datatype().unwrap(),
            dt,
            "the datatype still reads: it is the values that have no answer"
        );
        assert!(matches!(
            ds.read_u64(),
            Err(Error::Format(FormatError::NumericElementTooWide {
                size: 16
            }))
        ));
        assert!(matches!(
            ds.read_u64_rows(0, 1),
            Err(Error::Format(FormatError::NumericElementTooWide {
                size: 16
            }))
        ));
        // What is refused is the decode, not the data: the bytes are still
        // there for a caller willing to read the width itself.
        assert_eq!(ds.read_raw().unwrap().len(), 32);
    }

    /// A fixed-point attribute wider than the 64-bit value the readers decode
    /// into joins the attributes `attrs` omits, rather than appearing there
    /// holding part of its value.
    ///
    /// This one used to be *present* in the values channel: nine bytes holding
    /// 2^64 read back as `U64(0)`, a value indistinguishable from an attribute
    /// that really holds zero. Omitting it puts it where the opaque attribute
    /// above already sits — absent from the values, reported in full by the
    /// datatypes — so a caller can see that something was dropped (#361).
    #[test]
    fn an_attribute_too_wide_to_decode_is_omitted_rather_than_truncated() {
        let wide = Datatype::FixedPoint {
            size: 9,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 72,
        };
        // 2^64 exactly, so every one of the low 64 bits is zero.
        let mut raw_data = vec![0u8; 9];
        raw_data[8] = 1;
        let huge = crate::attribute::AttributeMessage {
            name: "huge".into(),
            datatype: wide.clone(),
            dataspace: Dataspace {
                space_type: crate::dataspace::DataspaceType::Scalar,
                rank: 0,
                dimensions: vec![],
                max_dimensions: None,
            },
            raw_data,
            datatype_location: crate::shared_message::DatatypeLocation::Inline,
        };

        for c in attr_channels(&[("count", AttrValue::I32(1))], std::slice::from_ref(&huge)).owners
        {
            assert!(
                !c.values.contains_key("huge"),
                "{}: decoding this attribute would report 0 for a value of 2^64",
                c.owner
            );
            assert_eq!(
                c.datatypes.get("huge"),
                Some(&wide),
                "{}: the datatype channel must still report the width on disk",
                c.owner
            );
            assert!(
                c.values.contains_key("count") && c.datatypes.contains_key("count"),
                "{}: the attribute beside it must still decode",
                c.owner
            );
        }
    }

    /// The channel must cover dense (fractal-heap) attribute storage, not only
    /// the compact form that lives in the object header.
    ///
    /// The two tests above use three attributes, which is well inside the
    /// writer's compact threshold, so on their own they leave every dense path
    /// unpinned: an implementation that walked `hdr.messages` directly and never
    /// touched the heap would pass both and report *nothing* for a real file with
    /// many attributes. Twelve is past the threshold, and the heap signature is
    /// asserted so the fixture cannot quietly revert to compact storage and take
    /// the coverage with it.
    #[test]
    fn attr_datatypes_covers_dense_attribute_storage() {
        let names: Vec<String> = (0..12).map(|i| format!("a{i:02}")).collect();
        let values: Vec<(&str, AttrValue)> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    n.as_str(),
                    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    AttrValue::I32(i as i32),
                )
            })
            .collect();

        let f = attr_channels(&values, &[]);
        assert!(
            f.bytes.windows(4).any(|w| w == b"FRHP"),
            "the fixture must really use dense storage, or this test proves \
             nothing beyond the compact path the other tests already cover"
        );

        for c in f.owners {
            assert_eq!(
                c.datatypes.len(),
                names.len(),
                "{}: every attribute in the heap must be reported, got {:?}",
                c.owner,
                c.datatypes.keys().collect::<Vec<_>>()
            );
            // The two channels must agree on which attributes exist: all of these
            // decode, so neither one has anything to omit here.
            let mut from_values: Vec<&String> = c.values.keys().collect();
            let mut from_types: Vec<&String> = c.datatypes.keys().collect();
            from_values.sort();
            from_types.sort();
            assert_eq!(
                from_values, from_types,
                "{}: the channels disagree",
                c.owner
            );
            for name in &names {
                assert!(
                    matches!(
                        c.datatypes.get(name),
                        Some(Datatype::FixedPoint { size: 4, .. })
                    ),
                    "{}: {name} must keep its 4-byte width through the heap, got {:?}",
                    c.owner,
                    c.datatypes.get(name)
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Coalesced chunk reads (see `crate::chunk_span`)
    // -----------------------------------------------------------------------

    /// A [`Source`] over a file image that counts what the reader asks the file
    /// for, so a test can assert read *volume* and not only the values returned.
    struct CountingSource {
        bytes: Vec<u8>,
        reads: Arc<AtomicUsize>,
        bytes_read: Arc<AtomicUsize>,
    }

    impl Source for CountingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(buf.len(), Ordering::Relaxed);
            BytesSource::new(&self.bytes).read_at(offset, buf)
        }
    }

    /// Counters for one measured read.
    #[derive(Default)]
    struct ReadCounts {
        reads: Arc<AtomicUsize>,
        bytes: Arc<AtomicUsize>,
    }

    impl ReadCounts {
        /// Run `f` and report `(reads, bytes)` it cost.
        fn measure<R>(&self, f: impl FnOnce() -> R) -> (usize, usize) {
            let r0 = self.reads.load(Ordering::Relaxed);
            let b0 = self.bytes.load(Ordering::Relaxed);
            f();
            (
                self.reads.load(Ordering::Relaxed) - r0,
                self.bytes.load(Ordering::Relaxed) - b0,
            )
        }
    }

    /// A streaming [`File`] over `bytes` whose reads are counted, built the way
    /// [`File::open_streaming_with_options`] builds one over a file handle.
    fn counting_streaming_file(
        bytes: Vec<u8>,
        counts: &ReadCounts,
        access: FileAccessProperties,
    ) -> File {
        File::from_source_with_options(
            CountingSource {
                bytes,
                reads: Arc::clone(&counts.reads),
                bytes_read: Arc::clone(&counts.bytes),
            },
            access,
        )
        .expect("open from source")
    }

    /// An `n`-element f64 dataset in chunks of `chunk` elements.
    fn chunked_f64_file(n: usize, chunk: u64) -> (Vec<u8>, Vec<f64>) {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[n as u64])
            .with_chunks(&[chunk]);
        (builder.finish().expect("write file"), data)
    }

    /// One chunk per row is what a writer that appends as data arrives
    /// produces; the rows land next to each other, so the streaming reader must
    /// fetch them in a few spans rather than one read each.
    #[test]
    fn a_streaming_read_coalesces_a_run_of_small_chunks() {
        let n = 1024usize;
        let (bytes, data) = chunked_f64_file(n, 1);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());

        let mut got = Vec::new();
        let (reads, _) = counts.measure(|| got = file.dataset("d").unwrap().read_f64().unwrap());
        assert_eq!(got, data, "the coalesced read must return the same values");

        // Reading each of the 1024 chunks on its own would cost at least that
        // many reads; the whole run plus its metadata fits in far fewer.
        assert!(
            reads < n / 8,
            "expected the {n} chunks to be coalesced into few reads, got {reads}"
        );
    }

    /// A windowed read must coalesce only the chunks its window overlaps: a
    /// plan built over the dataset's whole chunk list would put the rows on
    /// either side of the window inside a span and read them for nothing.
    #[test]
    fn a_windowed_streaming_read_fetches_only_its_own_window() {
        let rows = 4096usize;
        let (bytes, data) = chunked_f64_file(rows, 4);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let ds = file.dataset("d").unwrap();

        // The first window walks (and caches on this handle) the chunk index,
        // so measure the second: what it reads is the window's own chunks.
        assert_eq!(ds.read_f64_rows(0, 8).unwrap(), data[0..8]);
        let (_, window_bytes) =
            counts.measure(|| assert_eq!(ds.read_f64_rows(2048, 8).unwrap(), data[2048..2056]));

        // Eight rows are two 32-byte chunks.
        assert!(
            window_bytes < 256,
            "an 8-row window read {window_bytes} bytes; it needs 64"
        );
    }

    /// The read volume of a whole-dataset read must not depend on which read it
    /// is. The chunk list comes from the handle's cached index on every read
    /// after the first, and that index is a map: it yields no address order at
    /// all. A reader holding one coalesced span re-reads a whole span each time
    /// an unordered walk crosses back, so the second read of a dataset spanning
    /// more than one span cost two orders of magnitude more than the first.
    ///
    /// A regression here is probabilistic rather than certain — the map's order
    /// is seeded per process, and a run that happened to be sorted would pass —
    /// but with hundreds of chunks over two spans the chance of that is nil.
    /// Correct code passes deterministically.
    #[test]
    fn a_second_whole_read_costs_what_the_first_did() {
        // 512 KiB of f64 in 1 KiB chunks: more than one 256 KiB span, so an
        // unordered walk has somewhere to thrash between.
        let n = 64 * 1024usize;
        let dataset_bytes = n * 8;
        let (bytes, data) = chunked_f64_file(n, 128);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let ds = file.dataset("d").unwrap();

        counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));
        let (_, second) = counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));

        assert!(
            second <= dataset_bytes,
            "the second read fetched {second} bytes of a {dataset_bytes}-byte dataset"
        );
    }

    /// The same, for row windows: `docs/guide/streaming.md` walks a dataset in
    /// windows on one handle, so every window but the first takes its chunk
    /// list from the cached index.
    ///
    /// Each window here covers more than one span, which is what it takes to
    /// see the defect: a window whose chunks all fit a single span is served
    /// out of that one buffer whatever order it walks them in.
    #[test]
    fn a_window_loop_costs_the_dataset_once() {
        // 2 MiB of f64 in 1 KiB chunks, read in 512 KiB windows — two spans
        // each.
        let rows = 256 * 1024usize;
        let dataset_bytes = rows * 8;
        let (bytes, data) = chunked_f64_file(rows, 128);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let ds = file.dataset("d").unwrap();

        let window = 64 * 1024;
        let (_, total) = counts.measure(|| {
            for lo in (0..rows).step_by(window) {
                let got = ds.read_f64_rows(lo as u64, window as u64).unwrap();
                assert_eq!(got, data[lo..lo + window]);
            }
        });

        assert!(
            total <= 2 * dataset_bytes,
            "a window loop over a {dataset_bytes}-byte dataset read {total} bytes"
        );
    }

    /// A span must not cover a chunk the chunk cache already holds: the read
    /// skips that chunk, so those bytes would be fetched for nothing.
    ///
    /// The cache here is given room for the whole dataset. At the default
    /// sixteen slots a dataset this size evicts its own warm chunks as it
    /// walks, so every chunk misses on the second read and the plan has nothing
    /// to leave out — a config that cannot show the difference either way.
    #[test]
    fn a_warm_chunk_cache_is_not_re_fetched() {
        // 32 KiB of f64 in 32 chunks of 1 KiB, laid end to end.
        let n = 4096usize;
        let chunk_bytes = 1024usize;
        let (bytes, data) = chunked_f64_file(n, 128);
        let counts = ReadCounts::default();
        let file = counting_streaming_file(
            bytes,
            &counts,
            FileAccessProperties::new()
                .with_chunk_cache(ChunkCacheConfig::new().with_max_slots(64)),
        );
        let ds = file.dataset("d").unwrap();

        // Warm the second half of the dataset: chunks 16..31.
        assert_eq!(ds.read_f64_rows(2048, 2048).unwrap(), data[2048..]);
        assert_eq!(
            ds.chunk_cache_stats().cached_chunks(),
            16,
            "the window's own chunks stay cached"
        );

        let (_, second) = counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));
        assert_eq!(
            second,
            16 * chunk_bytes,
            "only the cold half was needed; a span over the whole dataset would \
             have fetched {} bytes",
            32 * chunk_bytes
        );
    }

    // -----------------------------------------------------------------------
    // Group member iterators (`iter_datasets` / `iter_groups`)
    // -----------------------------------------------------------------------

    /// A root holding `datasets` datasets, `groups` subgroups (each with one
    /// dataset of its own) and one committed datatype, so a member walk has all
    /// three child kinds to sort apart.
    ///
    /// Names are not zero-padded, so lexical order and insertion order disagree
    /// past the tenth member: a walk that silently sorted would show up here.
    fn mixed_member_file(datasets: usize, groups: usize) -> Vec<u8> {
        let mut b = FileBuilder::new();
        b.commit_datatype("a_type", crate::make_i32_type());
        for i in 0..datasets {
            b.create_dataset(&format!("ds{i}"))
                .with_i32_data(&[i as i32, -(i as i32)]);
        }
        for i in 0..groups {
            let mut g = b.create_group(&format!("g{i}"));
            g.create_dataset("inner").with_i32_data(&[i as i32]);
            g.create_dataset("other").with_i32_data(&[-1]);
            b.add_group(g.finish());
        }
        b.finish().expect("write the fixture")
    }

    /// The iterator must report exactly what opening each name reports: the same
    /// members, in the same order, resolving to the same objects. Anything the
    /// two disagree on is a member a caller would see differently for having
    /// chosen the cheaper walk.
    #[test]
    fn iter_datasets_agrees_with_opening_each_name() {
        let file = File::from_bytes(mixed_member_file(12, 3)).unwrap();

        for group in [file.root(), file.group("g1").unwrap()] {
            let names = group.datasets().unwrap();
            assert!(
                !names.is_empty(),
                "the fixture must have members to compare"
            );

            let iterated: Vec<(String, Dataset)> = group.iter_datasets().unwrap().collect();
            assert_eq!(
                iterated.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
                names,
                "the iterator must yield the members `datasets` lists, in that order"
            );

            for (name, ds) in &iterated {
                let opened = group.dataset(name).unwrap();
                assert_eq!(
                    ds.header_address().unwrap(),
                    opened.header_address().unwrap(),
                    "{name}"
                );
                assert_eq!(ds.shape().unwrap(), opened.shape().unwrap(), "{name}");
                assert_eq!(ds.read_i32().unwrap(), opened.read_i32().unwrap(), "{name}");
            }
        }
    }

    /// The subgroup counterpart, including that a handle it yields can be walked
    /// again — recursion through `iter_groups` is the shape it exists for.
    #[test]
    fn iter_groups_agrees_with_opening_each_name() {
        let file = File::from_bytes(mixed_member_file(4, 5)).unwrap();
        let root = file.root();

        let names = root.groups().unwrap();
        let iterated: Vec<(String, Group)> = root.iter_groups().unwrap().collect();
        assert_eq!(
            iterated.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
            names,
            "the iterator must yield the subgroups `groups` lists, in that order"
        );
        assert_eq!(names.len(), 5);

        for (name, group) in &iterated {
            assert_eq!(
                group.header_address().unwrap(),
                root.group(name).unwrap().header_address().unwrap(),
                "{name}"
            );
            let mut inner = group
                .iter_datasets()
                .unwrap()
                .map(|(n, _)| n)
                .collect::<Vec<_>>();
            inner.sort();
            assert_eq!(inner, ["inner", "other"], "{name} must be walkable in turn");
        }
    }

    /// Each iterator must claim only its own kind of child. A committed datatype
    /// is the child that belongs to neither, and the one a walk asking only for
    /// datasets and groups would otherwise be free to mis-sort into either.
    #[test]
    fn the_member_iterators_sort_the_child_kinds_apart() {
        let file = File::from_bytes(mixed_member_file(3, 2)).unwrap();
        let root = file.root();

        let datasets: Vec<String> = root.iter_datasets().unwrap().map(|(n, _)| n).collect();
        let groups: Vec<String> = root.iter_groups().unwrap().map(|(n, _)| n).collect();

        assert_eq!(datasets, ["ds0", "ds1", "ds2"]);
        assert_eq!(groups, ["g0", "g1"]);
        assert_eq!(root.named_datatypes().unwrap(), ["a_type"]);
        assert!(
            !datasets.contains(&"a_type".to_string()) && !groups.contains(&"a_type".to_string()),
            "a committed datatype is neither a dataset nor a group"
        );
    }

    /// The bytes a walk of `n` members reads, by each route: opening every name,
    /// and iterating handles.
    fn member_walk_bytes(n: usize) -> (usize, usize) {
        let mut b = FileBuilder::new();
        for i in 0..n {
            b.create_dataset(&format!("ds{i}"))
                .with_i32_data(&[i as i32]);
        }
        let bytes = b.finish().expect("write the fixture");

        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes.clone(), &counts, FileAccessProperties::new());
        let (_, by_name) = counts.measure(|| {
            let root = file.root();
            for name in root.datasets().unwrap() {
                root.dataset(&name).unwrap();
            }
        });

        let counts = ReadCounts::default();
        let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
        let (_, iterated) =
            counts.measure(|| for (_, _ds) in file.root().iter_datasets().unwrap() {});

        (by_name, iterated)
    }

    /// Opening members by name re-walks the group's link structure once per
    /// member; iterating handles walks it once. Bytes is the metric that
    /// separates them: a group this size keeps its links inline in the object
    /// header, so that header grows with the member count and re-reading it per
    /// member is quadratic, while the *number* of reads stays linear either way
    /// and would show almost nothing.
    ///
    /// Asserted as how the cost scales rather than as one fixture's byte count,
    /// since a fixed number would pass just as well on a walk that stayed
    /// quadratic with a smaller constant.
    #[test]
    fn iterating_members_enumerates_the_group_once() {
        let (_, iterated_16) = member_walk_bytes(16);
        let (by_name_64, iterated_64) = member_walk_bytes(64);

        // The rule this test exists for, and the only assertion here that is
        // about `iter_datasets` itself.
        assert!(
            iterated_64 <= 5 * iterated_16,
            "one enumeration plus one header per member is linear, so four times \
             the members must cost about four times the bytes: {iterated_16} -> \
             {iterated_64}"
        );

        // Contrast, not a property of this code: it holds because `Group::dataset`
        // re-enumerates. If a future change makes that route cheap enough to turn
        // this red, nothing here has regressed — confirm the scaling assertion
        // above still holds and then drop this one.
        assert!(
            iterated_64 < by_name_64,
            "at 64 members the one-enumeration walk should still be the cheaper: \
             {iterated_64} bytes against {by_name_64}"
        );
    }

    /// A yielded handle must carry the chunk-cache configuration the file was
    /// opened with, the same one `dataset` resolves for it.
    ///
    /// Nothing about the values read would show a handle that quietly ignored
    /// it: the cache decides how often the chunk index is re-parsed and how much
    /// decompressed data is retained, not what comes back. So the configuration
    /// has to be asserted directly, or dropping it here is a silent regression.
    #[test]
    fn a_member_handle_carries_the_files_chunk_cache_config() {
        let configured = ChunkCacheConfig::new()
            .with_max_slots(17)
            .with_max_bytes(4096)
            .with_index_cache(false);
        let file = File::from_bytes_with_options(
            mixed_member_file(3, 0),
            FileAccessProperties::new().with_chunk_cache(configured),
        )
        .unwrap();
        let root = file.root();

        let members: Vec<(String, Dataset)> = root.iter_datasets().unwrap().collect();
        assert_eq!(members.len(), 3);
        for (name, ds) in &members {
            assert_eq!(
                ds.chunk_cache_config(),
                root.dataset(name).unwrap().chunk_cache_config(),
                "{name} must resolve its cache the way `dataset` does"
            );
            assert_eq!(
                ds.chunk_cache_config(),
                configured,
                "{name} must carry the configuration the file was opened with"
            );
        }
    }

    /// A file whose root holds `inner` and `sub`, and a group `g0` holding
    /// children of those same names, plus a `refs` dataset pointing at `g0`.
    ///
    /// The duplicated names are the trap: a member handle that wrongly took a
    /// root-relative path would address a real object rather than fail, so the
    /// mistake would look like a successful write.
    fn dereferenced_group_file() -> Vec<u8> {
        let mut b = FileBuilder::new();
        b.create_dataset("inner").with_i32_data(&[0]);
        let mut root_sub = b.create_group("sub");
        root_sub.create_dataset("x").with_i32_data(&[0]);
        b.add_group(root_sub.finish());

        let mut g = b.create_group("g0");
        g.create_dataset("inner").with_i32_data(&[1]);
        let mut nested = g.create_group("sub");
        nested.create_dataset("x").with_i32_data(&[1]);
        g.add_group(nested.finish());
        b.add_group(g.finish());

        b.create_dataset("refs").with_path_references(&["g0"]);
        b.finish().expect("write the fixture")
    }

    /// A group reached by object reference has no resolvable path, so neither can
    /// its members: there is nothing for a write through one to address. The
    /// iterators must carry that `None` across rather than fall back to a
    /// root-relative path, which here would reach a different, real object.
    #[test]
    fn members_of_a_dereferenced_group_have_no_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("refs.h5");
        std::fs::write(&path, dereferenced_group_file()).unwrap();

        let file = File::open_rw(&path).unwrap();
        let mut objects = file.dataset("refs").unwrap().dereference().unwrap();
        let group = match objects.remove(0) {
            Object::Group(g) => g,
            other => panic!("expected a group, got {other:?}"),
        };

        // The file is writable and both names exist at the root, so a refusal
        // here can only come from the handle having no path to address.
        let (name, mut member) = group.iter_datasets().unwrap().next().unwrap();
        assert_eq!(name, "inner");
        assert!(
            matches!(
                member.set_attr("tag", AttrValue::I64(1)),
                Err(Error::ReadOnly)
            ),
            "a member of a path-less group must refuse a write, not address `/inner`"
        );

        let (name, subgroup) = group.iter_groups().unwrap().next().unwrap();
        assert_eq!(name, "sub");
        assert!(
            matches!(
                subgroup.set_attr("tag", AttrValue::I64(1)),
                Err(Error::ReadOnly)
            ),
            "and so must a subgroup of one, not address `/sub`"
        );
    }

    /// A yielded handle must carry the same root-relative path as one opened by
    /// name, or a write through it would address the wrong object — or no object
    /// at all. A member of a *subgroup* is the case that separates them, since
    /// its path has a prefix to get right.
    #[test]
    fn a_member_handle_resolves_to_its_own_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("members.h5");
        std::fs::write(&path, mixed_member_file(2, 2)).unwrap();

        let file = File::open_rw(&path).unwrap();
        let group = file.group("g1").unwrap();
        let (name, mut ds) = group
            .iter_datasets()
            .unwrap()
            .find(|(n, _)| n == "inner")
            .expect("g1/inner");
        assert_eq!(name, "inner");
        ds.set_attr("tag", AttrValue::I64(7)).unwrap();
        file.commit().unwrap();
        // Windows holds the write lock until the session is dropped.
        drop(ds);
        drop(group);
        drop(file);

        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("g1/inner")
                .unwrap()
                .attrs()
                .unwrap()
                .get("tag")
                .and_then(AttrValue::as_i64),
            Some(7),
            "the attribute must land on the member the handle came from"
        );
        assert!(
            !file
                .dataset("g0/inner")
                .unwrap()
                .attrs()
                .unwrap()
                .contains_key("tag"),
            "and on no other group's member of the same name"
        );
    }

    /// Every read-write entry point has to hand the fapl's `fsync` cadence to
    /// the session it opens, and none of them can be checked from outside: a
    /// skipped barrier writes the same bytes as an issued one (issue #263).
    ///
    /// One entry point missing the funnel is the whole failure mode, so this
    /// asserts the property at each of them rather than at the funnel.
    #[test]
    fn every_read_write_open_carries_the_fapl_sync_policy() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let fixture = |name: &str| {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[2]);
            b.write(&path).unwrap();
            path
        };
        let props = || FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose);
        let policy_of = |file: &File| match &file.inner.backend {
            Backend::Edit(m) => m
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .sync_policy(),
            _ => panic!("a read-write open must build an editing session"),
        };

        let opened = File::open_rw_with_options(fixture("open_rw.h5"), props()).unwrap();
        assert_eq!(policy_of(&opened), SyncPolicy::OnClose, "File::open_rw");
        // Windows OS locks are mandatory: release each session before the next
        // open touches the same directory's files.
        drop(opened);

        let created = File::create_with_options(
            dir.path().join("created.h5"),
            crate::FileCreateProperties::new(),
            props(),
        )
        .unwrap();
        assert_eq!(policy_of(&created), SyncPolicy::OnClose, "File::create");
        drop(created);

        let swmr = File::open_swmr_writer_with_options(fixture("swmr.h5"), props()).unwrap();
        assert_eq!(
            policy_of(&swmr),
            SyncPolicy::OnClose,
            "File::open_swmr_writer"
        );
        drop(swmr);

        let bounded = File::open_rw_with_options(
            fixture("bounded.h5"),
            props().with_memory_strategy(MemoryStrategy::Bounded),
        )
        .unwrap();
        assert_eq!(
            policy_of(&bounded),
            SyncPolicy::OnClose,
            "File::open_rw_with_options (bounded)"
        );
    }

    /// The page-buffer property is refused where it cannot be honored, rather
    /// than accepted and ignored (issues #288 and #308).
    ///
    /// Each refusal has a different reason and none stands in for the others: a
    /// budget under one page is a buffer that drains on every page it touches, a
    /// *paged* file that persists no free space can neither commit nor append, a
    /// pre-version-3 superblock carries a status-flags byte no library reads back
    /// so the crash mark would announce nothing, and the SWMR writer's readers
    /// observe the order its writes become visible in. The first is where the C
    /// library refuses `H5Pset_page_buffer_size` too; the rest are this crate's.
    ///
    /// Two shapes are **not** among them, and the accepted cases here are what
    /// pin that. An unpaged file: `H5PB_create` requires the paged allocator
    /// because the C page buffer is a page cache with per-kind reservations, and
    /// this gatherer has neither (issue #357). And a budget below the 1 MiB a
    /// session already gathers under, which is a request for less resident
    /// memory paid for in writes rather than an unhonorable pair (issue #391).
    ///
    /// Every refusal also asserts the file is left byte-identical. Each one
    /// fires with a read-write session already open, and the version-3 one fires
    /// from the same function that raises the mark — so a refusal ordered after
    /// the raise would leave a file marked in use by a session that never
    /// existed, which is a file nothing can open until `clear_swmr_flag`.
    #[test]
    fn a_page_buffer_is_refused_where_it_cannot_be_honored() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let fixture = |name: &str, paged: bool| {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            if paged {
                b.with_file_space_strategy(crate::FileSpaceStrategy::Page, true, 1)
                    .with_file_space_page_size(16 * 1024);
            }
            b.create_dataset("d")
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4]);
            b.write(&path).unwrap();
            path
        };
        let buffered = |bytes| {
            FileAccessProperties::new()
                .with_sync_policy(SyncPolicy::OnClose)
                .with_page_buffer_size(bytes)
        };

        // Each refusal fires after the session is already open read-write, so the
        // file it declined must be left exactly as it was — and still openable.
        let untouched = |label: &str, path: &std::path::Path, before: &[u8]| {
            assert_eq!(
                std::fs::read(path).unwrap(),
                before,
                "{label}: a refused open changed the file"
            );
            assert!(
                File::open(path).is_ok(),
                "{label}: a refused open left the file unopenable"
            );
        };

        // The two that are *accepted*, kept here beside their former siblings so
        // reinstating either refusal fails this test rather than only the
        // behavior tests one module over.
        let unpaged = fixture("unpaged.h5", false);
        let accepted = File::open_rw_with_options(&unpaged, buffered(1 << 20));
        assert!(
            accepted.is_ok(),
            "a page buffer on an unpaged file must be accepted, got {accepted:?}"
        );
        accepted.unwrap().close().unwrap();

        // 256 KiB on a 16 KiB-paged file: a budget the session would not have
        // gathered under, which is the point of asking for it.
        let small_budget = fixture("small_budget.h5", true);
        let accepted = File::open_rw_with_options(&small_budget, buffered(256 * 1024));
        assert!(
            accepted.is_ok(),
            "a page buffer below the byte budget a session gathers under must be \
             accepted, got {accepted:?}"
        );
        accepted.unwrap().close().unwrap();

        let paged = fixture("paged.h5", true);
        let before = std::fs::read(&paged).unwrap();
        let refused = File::open_rw_with_options(&paged, buffered(8192));
        assert!(
            matches!(&refused, Err(Error::EditUnsupported(m)) if m.contains("page size")),
            "a budget below one page must be refused, got {refused:?}"
        );
        untouched("budget under one page", &paged, &before);

        // And the pairing a caller reaches by doing nothing, since Always is the
        // default: every barrier there is an fsync that flushes the buffer.
        let refused = File::open_rw_with_options(
            &paged,
            FileAccessProperties::new().with_page_buffer_size(1 << 20),
        );
        assert!(
            matches!(&refused, Err(Error::EditUnsupported(m)) if m.contains("SyncPolicy::Always")),
            "a page buffer under SyncPolicy::Always must be refused, got {refused:?}"
        );
        untouched("page buffer under Always", &paged, &before);

        // A version-2 superblock. This crate's writer refuses `Page` below the
        // 1.10 format, so the file is built at version 3 and its superblock
        // rewritten — the v2 and v3 layouts are identical apart from the version
        // byte and what the flags byte means, which is exactly the point.
        let old_format = dir.path().join("v2.h5");
        std::fs::copy(&paged, &old_format).unwrap();
        {
            let mut bytes = std::fs::read(&old_format).unwrap();
            let sig = crate::signature::find_signature(&bytes).unwrap();
            let mut sb = crate::superblock::Superblock::parse(&bytes, sig).unwrap();
            assert_eq!(sb.version, 3, "the fixture must start at the newer format");
            sb.version = 2;
            let rewritten = sb.serialize();
            bytes[sig..sig + rewritten.len()].copy_from_slice(&rewritten);
            std::fs::write(&old_format, &bytes).unwrap();
        }
        let before = std::fs::read(&old_format).unwrap();
        assert!(
            File::open(&old_format).is_ok(),
            "the version-2 fixture must be a readable file, or the refusal below \
             could be about anything"
        );
        let refused = File::open_rw_with_options(&old_format, buffered(1 << 20));
        assert!(
            matches!(&refused, Err(Error::EditUnsupported(m)) if m.contains("version-3 superblock")),
            "a page buffer on a pre-v3 superblock must be refused, got {refused:?}"
        );
        untouched("version-2 superblock", &old_format, &before);

        // A paged file whose free space is not persisted, reached two ways: asked
        // for outright, and produced by a userblock, for which persistence is
        // declined however the creation properties were written. Such a session
        // can neither commit nor append, so the buffer would hold nothing while
        // its mark blocked every reader.
        for (label, name, userblock) in [
            ("paged, not persisting", "no_persist.h5", 0u64),
            ("paged with a userblock", "ub_paged.h5", 4096),
        ] {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            if userblock != 0 {
                b.with_userblock(userblock);
            }
            b.with_file_space_strategy(crate::FileSpaceStrategy::Page, userblock != 0, 1)
                .with_file_space_page_size(4096);
            b.create_dataset("d")
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4]);
            b.write(&path).unwrap();
            let before = std::fs::read(&path).unwrap();
            let refused = File::open_rw_with_options(
                &path,
                buffered(1 << 20).with_memory_strategy(MemoryStrategy::Mirrored),
            );
            assert!(
                matches!(&refused, Err(Error::EditUnsupported(m)) if m.contains("persisted")),
                "{label}: a page buffer on a session that cannot write must be refused, \
                 got {refused:?}"
            );
            untouched(label, &path, &before);
        }

        let swmr = fixture("swmr.h5", false);
        let before = std::fs::read(&swmr).unwrap();
        let refused = File::open_swmr_writer_with_options(&swmr, buffered(1 << 20));
        assert!(
            matches!(&refused, Err(Error::EditUnsupported(m)) if m.contains("SWMR")),
            "the SWMR writer must refuse a page buffer, got {refused:?}"
        );
        // The sharpest of the three: `open_swmr_writer` raises the on-disk
        // SWMR-write flag, and a refusal that fired after it would leave every
        // later open reporting `FileMarkedInUse`.
        untouched("swmr", &swmr, &before);

        // And an unset page buffer refuses none of the three.
        for (name, paged) in [("ok_unpaged.h5", false), ("ok_paged.h5", true)] {
            let f = File::open_rw_with_options(fixture(name, paged), FileAccessProperties::new());
            assert!(f.is_ok(), "{name}: an unset page buffer refuses nothing");
        }
        let f = File::open_swmr_writer_with_options(
            fixture("ok_swmr.h5", false),
            FileAccessProperties::new(),
        );
        assert!(f.is_ok(), "swmr: an unset page buffer refuses nothing");
    }

    /// `File::create_with_options` refuses a creation/access pair whose file it
    /// could write but not then open, rather than writing it and failing the open
    /// it promised (issue #288).
    ///
    /// A page buffer needs a budget of at least the file's page size and a
    /// version-3 superblock, and both of those are properties of the file being
    /// *created* — so the refusal belongs before the bytes are written, not in
    /// the reopen. The assertion that matters here is the `!path.exists()`: an
    /// error alone would pass with the file already on disk, which is the defect.
    ///
    /// The version-3 case is the one issue #357 made reachable. While a page
    /// buffer required a paged file it could not be: this crate's builder refuses
    /// `FileSpaceStrategy::Page` below the 1.10 format outright, so no pair got
    /// this far. An unpaged file at `LibVer::V18` is an ordinary buildable file,
    /// and without a check here it would be written and only then refused.
    #[test]
    fn create_with_options_refuses_a_page_buffer_it_could_not_reopen_with() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let paged = |page: u64| {
            crate::FileCreateProperties::new()
                .with_file_space_strategy(crate::FileSpaceStrategy::Page, true, 1)
                .with_file_space_page_size(page)
        };
        // 8192 clears the format's 4096 default and still falls short of this
        // file's 16 KiB page, so it fails only against the page size actually
        // read from the file.
        //
        // The last case is the one a caller reaches by doing nothing: `Always` is
        // the default policy, and every other property in that pair is honorable.
        // It is here because it was missing — the refusal used to sit at the fapl
        // rather than with its siblings, so this function did not restate it and
        // the file was written before the open failed.
        let cases: [(&str, crate::FileCreateProperties, usize, SyncPolicy); 5] = [
            // A page larger than the budget, on a file that does not exist yet:
            // the page size has to come from the creation properties, which is
            // what this pins.
            (
                "page larger than the budget",
                paged(2 << 20),
                1 << 20,
                SyncPolicy::OnClose,
            ),
            (
                "the 1.8 format",
                crate::FileCreateProperties::new()
                    .with_libver_bounds(crate::LibVer::Earliest, crate::LibVer::V18),
                1 << 20,
                SyncPolicy::OnClose,
            ),
            (
                "budget under one page",
                paged(16 * 1024),
                8192,
                SyncPolicy::OnClose,
            ),
            // The unpaged arm of the same check, which decides on its own now
            // that a budget below the session's gather budget is honored (issue
            // #391): an unpaged file has no page size, so it is held to the
            // format's 4 KiB default and 2 KiB falls short of it.
            (
                "unpaged budget under the default page",
                crate::FileCreateProperties::new(),
                2048,
                SyncPolicy::OnClose,
            ),
            (
                "the default sync policy",
                paged(16 * 1024),
                1 << 20,
                SyncPolicy::Always,
            ),
        ];

        for (label, create, budget, policy) in cases {
            let path = dir
                .path()
                .join(std::format!("{}.h5", label.replace(' ', "_")));
            let result = File::create_with_options(
                &path,
                create,
                FileAccessProperties::new()
                    .with_sync_policy(policy)
                    .with_page_buffer_size(budget),
            );
            assert!(
                matches!(result, Err(Error::EditUnsupported(_))),
                "{label}: expected a refusal, got {result:?}"
            );
            assert!(
                !path.exists(),
                "{label}: the file was written and only then refused"
            );
        }

        // The honorable pairs still create — including the unpaged one, which is
        // the default creation properties and so the pair a caller reaches by
        // asking for nothing but the buffer, and the 256 KiB one, which is below
        // the byte budget a session gathers under (issue #391).
        for (label, create, budget) in [
            ("paged", paged(16 * 1024), 1 << 20),
            ("unpaged", crate::FileCreateProperties::new(), 1 << 20),
            ("paged_small_budget", paged(16 * 1024), 256 * 1024),
        ] {
            let ok = File::create_with_options(
                dir.path().join(std::format!("ok_{label}.h5")),
                create,
                FileAccessProperties::new()
                    .with_sync_policy(SyncPolicy::OnClose)
                    .with_page_buffer_size(budget),
            );
            assert!(
                ok.is_ok(),
                "{label}: an honorable budget must create: {ok:?}"
            );
            ok.unwrap().close().unwrap();
        }
    }

    /// `close` and `drop` issue their barrier under *every* policy, on both the
    /// ordinary and the SWMR branch — the four sites where this crate writes
    /// after the last point a caller could have ordered anything (issue #263).
    ///
    /// This is the half of the contract `SyncPolicy` cannot express: the two
    /// `drop` sites are unreachable by any caller discipline at all, since the
    /// handle that would have issued `File::sync` is gone by the time they run.
    /// Asserted through a counting image, because the difference between a
    /// forced barrier and a skipped one is invisible in the bytes.
    #[test]
    fn close_and_drop_force_their_barrier_under_every_policy() {
        use crate::edit::WriteEngine;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        // `swmr` picks the teardown branch; `explicit` picks `close` over `drop`.
        let teardown = |name: &str, swmr: bool, explicit: bool| -> u64 {
            let path = dir.path().join(name);
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..8).collect::<Vec<_>>())
                .with_shape(&[8])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[4]);
            // Persisting, so the ordinary branch has manager re-homing to do:
            // an immediate append below leaves the on-disk managers mid-file,
            // and settling them is the write no earlier sync could cover.
            b.with_file_space_strategy(crate::FileSpaceStrategy::FsmAggr, true, 1);
            b.write(&path).unwrap();

            let syncs = Arc::new(AtomicU64::new(0));
            let session =
                WriteEngine::open_sync_counting(&path, SyncPolicy::OnClose, Arc::clone(&syncs))
                    .unwrap();
            let mut inner = FileInner::from_rw_session(
                session,
                FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap();
            inner.swmr_write = swmr;
            let file = File {
                inner: Arc::new(inner),
            };
            if !swmr {
                // The SWMR branch stages nothing and appends through its own
                // path; give the ordinary branch real work to settle.
                file.dataset("d")
                    .unwrap()
                    .append(&[8i32, 9, 10, 11])
                    .unwrap();
            }
            assert_eq!(
                syncs.load(AtomicOrdering::Relaxed),
                0,
                "nothing before teardown may sync under OnClose ({name})"
            );

            if explicit {
                file.close().unwrap();
            } else {
                drop(file);
            }
            syncs.load(AtomicOrdering::Relaxed)
        };

        for (name, swmr, explicit) in [
            ("close_plain.h5", false, true),
            ("close_swmr.h5", true, true),
            ("drop_plain.h5", false, false),
            ("drop_swmr.h5", true, false),
        ] {
            assert!(
                teardown(name, swmr, explicit) > 0,
                "{name} must force its barrier: the writes it makes are past the \
                 last point a caller could have ordered them"
            );
        }
    }
}
