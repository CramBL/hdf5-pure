//! File-creation properties as one reusable value — the `fcpl` analogue.

use crate::file_space_info::FileSpaceStrategy;
use crate::libver::LibVer;

/// File-creation properties applied when writing a new HDF5 file.
///
/// This is the `hdf5-pure` analogue of an HDF5 **file creation property list**
/// (`fcpl`): one value carrying every creation-time setting, so application code
/// can define a file layout once and reuse it everywhere it writes, instead of
/// repeating a builder call chain and keeping the copies in sync.
///
/// The `Properties` suffix means the type stands in for one whole HDF5 property
/// list, so every setting on it has a C counterpart to look up. It is a stand-in
/// and not a port: a plain `Copy` value, with no handle to create or close, no
/// runtime property registry, and no setter that can fail. `fcpl` and each
/// `H5Pset_*` it models are doc aliases, so a search for either lands here.
///
/// One setting crosses the class line. `H5Pset_libver_bounds` is officially a
/// *file access* property, but this crate checks the bound as the file is
/// written, so [`with_libver_bounds`](Self::with_libver_bounds) lives here with
/// the other write-time settings rather than on
/// [`FileAccessProperties`](crate::FileAccessProperties).
///
/// Pass it to [`FileBuilder::with_create_properties`](crate::FileBuilder::with_create_properties)
/// or [`File::create_with_options`](crate::File::create_with_options). The
/// equivalent [`FileBuilder`](crate::FileBuilder) methods set the same fields one
/// at a time and interoperate freely with this.
///
/// Values are recorded as given and checked when the file is written, not when
/// the properties are built — the value is inert data, so an illegal page size
/// is reported by `finish`/`write` rather than here. Note that a non-paged
/// userblock size is currently **not** validated against HDF5's power-of-two
/// rule. [Supported properties](#supported-properties) has the exact coverage.
///
/// # Supported properties
///
/// HDF5 configures a file through property lists: a file-creation property list (`fcpl`, passed to
/// `H5Fcreate`), a file-access property list (`fapl`, passed to `H5Fcreate` and `H5Fopen`), and a
/// dataset-access property list (`dapl`, passed to `H5Dopen`). The crate exposes no property-list
/// handles, and models all three as plain reusable values:
/// [`FileCreateProperties`](Self), [`FileAccessProperties`](crate::FileAccessProperties), and
/// [`DatasetAccessProperties`](crate::DatasetAccessProperties). Build one once and pass it to
/// [`FileBuilder::with_create_properties`](crate::FileBuilder::with_create_properties) /
/// [`File::create_with_options`](crate::File::create_with_options), to any `*_with_options` open,
/// or to [`File::dataset_with_options`](crate::File::dataset_with_options). The equivalent
/// [`FileBuilder`](crate::FileBuilder) methods set the same creation properties one at a time.
///
/// ## The `Properties` suffix
///
/// A `*Properties` type models one whole HDF5 property list, so every setting on it is looked up
/// here, the few with no C counterpart at all included
/// ([`with_memory_strategy`](crate::FileAccessProperties::with_memory_strategy),
/// [`with_sync_policy`](crate::FileAccessProperties::with_sync_policy)), which the tables mark as
/// such. Three types end in that suffix: [`FileCreateProperties`](Self) (`fcpl`),
/// [`FileAccessProperties`](crate::FileAccessProperties) (`fapl`), and
/// [`DatasetAccessProperties`](crate::DatasetAccessProperties) (`dapl`). Where a setting belongs to
/// a different HDF5 class than the table it appears in, that table labels it with its real class in
/// parentheses.
///
/// Modeling a list is not covering it: each of the three models a subset of its list's properties,
/// and the tables give the whole picture by listing what is unsupported beside what works.
///
/// The suffix is a positive claim only. A `*Options` or `*Config` type is silent either way, and
/// some of them do map to C properties:
///
/// - [`ChunkCacheConfig`](crate::ChunkCacheConfig) holds the `rdcc_*` values that both
///   `H5Pset_cache` (an `fapl` property) and `H5Pset_chunk_cache` (a `dapl` property) take, so it
///   belongs to no single list.
/// - [`MetadataCacheConfig`](crate::MetadataCacheConfig) is the analogue of the
///   `H5AC_cache_config_t` struct that one `fapl` property accepts, and not of a list.
/// - [`RepackOptions`](crate::RepackOptions),
///   [`VlenStringReadOptions`](crate::VlenStringReadOptions) and
///   [`mat::Options`](crate::mat::Options) have no HDF5 counterpart at all.
///
/// The `*_with_options` constructors take a different suffix. It marks the
/// explicit-configuration variant of an open, in the ordinary Rust sense of
/// [`OpenOptions`](std::fs::OpenOptions), and the type it takes is visible in the signature.
///
/// The [file-space strategy guide](crate::_guide::file_space) has the file-space details behind
/// the table below, and [Editing files in place](crate::_guide::editing) and [Streaming large
/// files](crate::_guide::streaming) the read-write access modes.
///
/// ## Status legend
///
/// | Status | Meaning |
/// |---|---|
/// | **Genuine** | The property changes the on-disk result exactly as HDF5 specifies, verified against the reference C library. |
/// | **Recorded** | The value is written to the file and round-trips through the C library, and the layout stays as it was, which is correct for a fresh file, since it has no free space to manage yet. |
/// | **Read-only** | Honored when a file is read, with no write-side effect. |
/// | **Behavioral** | Changes how the crate operates on the file, its memory and its durability, without changing a byte of the result. There is nothing on disk to verify against the C library. |
/// | **Assertion** | Validated on write, and cannot change the emitted format. |
/// | **Unsupported** | No equivalent: the property is absent, or a file requiring it is rejected up front. |
/// | **N/A** | Not meaningful for the on-disk format this crate emits. |
///
/// ## File-creation properties
///
/// Set through [`FileBuilder`](crate::FileBuilder) before [`write`](crate::FileBuilder::write) or
/// [`finish`](crate::FileBuilder::finish), one at a time or all at once with
/// [`FileBuilder::with_create_properties`](crate::FileBuilder::with_create_properties).
/// [`File::create_with_options(path, fcpl, fapl)`](crate::File::create_with_options) applies them
/// on the owned-handle path, mirroring `H5Fcreate(name, flags, fcpl_id, fapl_id)`. It returns an
/// open read-write handle, so a creation and access pair the reopen would reject (a paged file
/// with `persist = false`, or a userblock under
/// [`MemoryStrategy::Bounded`](crate::MemoryStrategy::Bounded)) fails before anything is written.
///
/// | HDF5 property (C API) | `hdf5-pure` | Status | Behavior |
/// |---|---|---|---|
/// | `H5Pset_file_space_strategy(PAGE, …)` | [`with_file_space_strategy(FileSpaceStrategy::Page, …)`](Self::with_file_space_strategy) | **Genuine** | Real page-aligned allocation: metadata and raw data occupy separate pages, and each page's free tail is tracked in a per-page-type `FSHD`/`FSSE` manager. The C library reads it as paged and `H5Fget_freespace` matches the tracked total. |
/// | `H5Pset_file_space_strategy(FSM_AGGR / AGGR / NONE, …)` | `…(FsmAggr / Aggr / None, …)` | **Recorded** | The strategy is stored in the superblock extension, and the layout stays sequential. Freed regions become tracked once a read-write session deletes an object. |
/// | `persist` flag | 2nd argument of [`with_file_space_strategy`](Self::with_file_space_strategy) | **Genuine** (paged) / **Recorded** (non-paged) | Paged: per-page-type managers are written from creation. Non-paged: the flag records intent, and managers appear after a later delete. |
/// | `threshold` | 3rd argument of [`with_file_space_strategy`](Self::with_file_space_strategy) | **Recorded (advisory)** | Round-trips through the C library, and the crate tracks every page tail and freed section whatever its value. |
/// | `H5Pset_file_space_page_size` | [`with_file_space_page_size`](Self::with_file_space_page_size) | **Genuine** (paged) / **Recorded** (non-paged) | Under [`Page`](crate::FileSpaceStrategy::Page) it is the alignment quantum, 4096 by default, and a power of two `>= 512`. Under the other strategies it is recorded and inert. |
/// | `H5Pset_userblock` | [`with_userblock`](Self::with_userblock) | **Genuine** | Reserves a zero-filled prefix, and every address is base-relative. The HDF5 rule of zero, or a power of two `>= 512`, is validated at write time ([`FormatError::InvalidUserblockSize`](crate::FormatError::InvalidUserblockSize)), since the size is the superblock's base address and a reader scans the doubling sequence alone for the signature. Under [`Page`](crate::FileSpaceStrategy::Page) the userblock must also be a whole number of pages. Its contents come from [`FileBuilder::with_userblock_content`](crate::FileBuilder::with_userblock_content), which every output path emits. |
/// | `H5Pset_libver_bounds` (`fapl`) | [`with_libver_bounds`](Self::with_libver_bounds) | **Genuine** (1.8 / 1.10) | A format selector between the two the writer emits: `high` picks the v2 (HDF5 1.8) superblock at `Earliest..=V18` and the v3 (1.10) one at anything reaching 1.10, and a lower bound above 1.10 (`V112`, `V114`, `LATEST`) selects the 1.10 format too. As in the C library, the low bound permits newer encodings and requires none. An upper bound older than 1.8, or below the lower bound, is rejected with [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable). HDF5 classes this as a file-access property, and it sits here because this crate resolves the bound at write time. |
/// | `H5Pset_fill_value` / `H5Pset_fill_time` (`dcpl`) | [`DatasetBuilder::with_fill_value`](crate::DatasetBuilder::with_fill_value) | **Genuine** (per dataset) | Encodes the fill value in a v3 Fill Value message, and [`Dataset::fill_value`](crate::Dataset::fill_value) reads it back, from this crate's files and the C library's. |
/// | `H5Pset_obj_track_times` (`ocpl`) | none | **Unsupported** | Every object is written with times untracked, which is equivalent to `false`, and tracking cannot be enabled. |
/// | `H5Pset_sym_k` / `H5Pset_istore_k` | none | **N/A** | The v3 superblock omits these fields, and every group is new-style, with link messages and a v2 object header. |
/// | `H5Pset_link_phase_change` / `H5Pset_est_link_info` (`gcpl`) | none | **Unsupported** | The Group Info message is written minimal, so the C library's defaults apply (max-compact 8, min-dense 6), and the thresholds are not tunable. |
///
/// ## Compliance and known limits
///
/// The paged and persistent-free-space paths are exercised by C-library crosschecks
/// (`crates/crosscheck/tests/file_space.rs`, `crates/crosscheck/tests/bounded_append.rs`): the
/// reference library recovers the strategy, `H5Fget_freespace` equals the crate's tracked total
/// exactly, and the C library reopens a paged file read-write and re-paginates it. The crate also
/// reads and bounded-mutates genuine C-created paged and persisted files. Compliance here means
/// page alignment and structural validity, and it stops short of reproducing the C allocator's
/// intra-page packing byte for byte.
///
/// The limits that come with it:
///
/// - A paged file must persist its free space to be mutated. Both editors grow a paged `persist =
///   true` file, keeping pages homogeneous and the end of allocation page-aligned. A paged file
///   created without `persist = true`, or one carrying a userblock, has no usable record of which
///   pages hold metadata and which hold raw data, so it cannot be grown at all. Recreate it with
///   `persist = true` and no userblock.
/// - Free space is under-reported and never over-reported. A final metadata-page tail and the old
///   bytes of a relocated partial chunk are left untracked, so `H5Fget_freespace` can read slightly
///   low. The file stays valid.
/// - **`threshold` is advisory** (see the table above).
/// - Only **File Space Info message version 1** is emitted and read.
///
/// The limitations catalog has the full set of deliberate refusals.
///
/// # Examples
///
/// ```no_run
/// use hdf5_pure::{FileCreateProperties, FileSpaceStrategy};
///
/// // Define the layout once...
/// fn paged_layout() -> FileCreateProperties {
///     FileCreateProperties::new()
///         .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
///         .with_file_space_page_size(8192)
/// }
///
/// // ...and reuse it across every write path.
/// let mut builder = hdf5_pure::FileBuilder::new();
/// builder.with_create_properties(paged_layout());
/// builder.create_dataset("data").with_f64_data(&[1.0, 2.0]);
/// builder.write("out.h5").unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(alias = "fcpl")]
pub struct FileCreateProperties {
    userblock: u64,
    libver_bounds: Option<(LibVer, LibVer)>,
    file_space_strategy: Option<(FileSpaceStrategy, bool, u64)>,
    file_space_page_size: Option<u64>,
}

impl FileCreateProperties {
    /// A value carrying the crate's default creation behavior: no userblock,
    /// no library-version bounds, and the writer's default file-space handling.
    pub const fn new() -> Self {
        Self {
            userblock: 0,
            libver_bounds: None,
            file_space_strategy: None,
            file_space_page_size: None,
        }
    }

    /// Reserve a zero-filled userblock of `size` bytes before the superblock.
    ///
    /// HDF5 requires a power of two `>= 512`, or 0 for no userblock; the check
    /// runs when the file is written. See
    /// [`FileBuilder::with_userblock`](crate::FileBuilder::with_userblock) for how
    /// to fill the region afterward.
    #[doc(alias = "H5Pset_userblock")]
    pub const fn with_userblock(mut self, size: u64) -> Self {
        self.userblock = size;
        self
    }

    /// Constrain the on-disk format version to `[low, high]`.
    ///
    /// `high` **selects** the format: `Earliest..=V18` writes the HDF5 1.8 one
    /// and anything reaching 1.10 writes the 1.10 one, so this changes the bytes
    /// of every file the properties are applied to. Content the chosen format
    /// cannot express is refused with
    /// [`FormatError::LibverTooOldForContent`](crate::FormatError::LibverTooOldForContent)
    /// rather than silently upgraded — see
    /// [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds)
    /// for which content that is.
    ///
    /// `low` only rules formats out, licensing newer encodings without requiring
    /// them, so a lower bound of `V112`, `V114` or `LATEST` writes the 1.10
    /// format rather than being refused — provided `high` reaches it. An
    /// inverted range such as `V114..=V110` is refused with
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable).
    ///
    /// HDF5 classes `H5Pset_libver_bounds` as a *file access* property; it sits
    /// here because this crate resolves the bound at write time.
    #[doc(alias = "H5Pset_libver_bounds")]
    pub const fn with_libver_bounds(mut self, low: LibVer, high: LibVer) -> Self {
        self.libver_bounds = Some((low, high));
        self
    }

    /// Set the file-space management strategy, whether free space persists across
    /// close, and the smallest free-space section tracked.
    #[doc(alias = "H5Pset_file_space_strategy")]
    pub const fn with_file_space_strategy(
        mut self,
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
    ) -> Self {
        self.file_space_strategy = Some((strategy, persist, threshold));
        self
    }

    /// Set the file-space page size, the allocation quantum under
    /// [`FileSpaceStrategy::Page`].
    #[doc(alias = "H5Pset_file_space_page_size")]
    pub const fn with_file_space_page_size(mut self, page_size: u64) -> Self {
        self.file_space_page_size = Some(page_size);
        self
    }

    /// Return the configured userblock size in bytes (0 for none).
    pub const fn userblock(&self) -> u64 {
        self.userblock
    }

    /// Return the configured library-version bounds, if any.
    pub const fn libver_bounds(&self) -> Option<(LibVer, LibVer)> {
        self.libver_bounds
    }

    /// Return the configured file-space strategy, persist flag, and threshold, if
    /// any.
    pub const fn file_space_strategy(&self) -> Option<(FileSpaceStrategy, bool, u64)> {
        self.file_space_strategy
    }

    /// Return the configured file-space page size, if any.
    pub const fn file_space_page_size(&self) -> Option<u64> {
        self.file_space_page_size
    }
}

impl Default for FileCreateProperties {
    fn default() -> Self {
        Self::new()
    }
}
