//! File-creation properties as one reusable value — the `fcpl` analogue.

use crate::FileSpaceSettings;
use crate::LinkPhaseChange;
use crate::UserblockSize;
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
/// The root group's link thresholds correspond to the group-creation property
/// `H5Pset_link_phase_change`. Configure them with
/// [`with_root_group_link_phase_change`](Self::with_root_group_link_phase_change).
///
/// `H5Pset_libver_bounds` is officially a
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
/// Userblock and page sizes are checked when [`UserblockSize`] and
/// [`FileSpacePageSize`](crate::FileSpacePageSize) are constructed. Constraints between settings
/// are checked when writing.
///
/// # Supported properties
///
/// HDF5 configures a file through property lists: a file-creation property list (`fcpl`, passed to
/// `H5Fcreate`), a file-access property list (`fapl`, passed to `H5Fcreate` and `H5Fopen`), a
/// dataset-creation property list (`dcpl`, passed to `H5Dcreate`), and a dataset-access property
/// list (`dapl`, passed to `H5Dopen`). The crate exposes no property-list handles, and models all
/// four as plain reusable values: [`FileCreateProperties`](Self),
/// [`FileAccessProperties`](crate::FileAccessProperties),
/// [`DatasetCreateProperties`](crate::DatasetCreateProperties), and
/// [`DatasetAccessProperties`](crate::DatasetAccessProperties). Build one once and pass it to
/// [`FileBuilder::with_create_properties`](crate::FileBuilder::with_create_properties) /
/// [`File::create_with_options`](crate::File::create_with_options), to any `*_with_options` open,
/// to [`DatasetBuilder::with_create_properties`](crate::DatasetBuilder::with_create_properties), or
/// to [`File::dataset_with_options`](crate::File::dataset_with_options). The equivalent
/// [`FileBuilder`](crate::FileBuilder) and [`DatasetBuilder`](crate::DatasetBuilder) methods set
/// the same creation properties one at a time.
///
/// ## The `Properties` suffix
///
/// A `*Properties` type models one whole HDF5 property list, so every setting on it is looked up
/// in the type's table, the few with no C counterpart at all included
/// ([`with_memory_strategy`](crate::FileAccessProperties::with_memory_strategy),
/// [`with_sync_policy`](crate::FileAccessProperties::with_sync_policy)), which the tables mark as
/// such. Four types end in that suffix: [`FileCreateProperties`](Self) (`fcpl`),
/// [`FileAccessProperties`](crate::FileAccessProperties) (`fapl`),
/// [`DatasetCreateProperties`](crate::DatasetCreateProperties) (`dcpl`), and
/// [`DatasetAccessProperties`](crate::DatasetAccessProperties) (`dapl`). Where a setting belongs to
/// a different HDF5 class than the table it appears in, that table labels it with its real class in
/// parentheses.
///
/// Modeling a list is not covering it: each of the four models a subset of its list's properties,
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
/// | **Genuine** | The property changes the on-disk result exactly as HDF5 specifies, verified against libhdf5. |
/// | **Recorded** | Stored in metadata with no effect on storage placement |
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
/// open read-write handle. The constructor rejects
/// [`MemoryStrategy::Bounded`](crate::MemoryStrategy::Bounded) combined with nonpersistent paged
/// settings or a userblock before writing.
///
/// [`FileSpaceSettings`] configures the file-space rows below. Apply the value with
/// [`with_file_space_settings`](Self::with_file_space_settings). The writer uses separate metadata
/// and raw aggregates under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) and
/// [`Aggr`](crate::FileSpaceStrategy::Aggr). With persistence enabled under `FsmAggr`, it records
/// eligible unused aggregate space from creation. Under [`Page`](crate::FileSpaceStrategy::Page),
/// it allocates within pages of the configured size and persists tracked page tails when enabled.
/// Under `Aggr` and [`None`](crate::FileSpaceStrategy::None), persistence is disabled and the
/// threshold is 1. Page size is recorded under every strategy and affects allocation under `Page`.
/// The page-size row distinguishes paged allocation from recording under the other strategies.
///
/// | HDF5 property (C API) | `hdf5-pure` | Status | Behavior |
/// |---|---|---|---|
/// | `H5Pset_file_space_strategy` | [`new`][space-new] | **Genuine** | Allocation strategy |
/// | `persist` flag | [`with_persistence`][space-persist] | **Genuine** | Manager persistence |
/// | `threshold` | [`with_threshold`][space-threshold] | **Genuine** | Section admission |
/// | `H5Pset_file_space_page_size` | [`with_page_size`][page] | **Genuine/Recorded** | Paging |
/// | `H5Pset_userblock` | [`with_userblock`](Self::with_userblock) | **Genuine** | Reserves a zero-filled prefix with a [`UserblockSize`]. Under [`Page`](crate::FileSpaceStrategy::Page), the writer requires a whole number of pages. [`FileBuilder::with_userblock_content`](crate::FileBuilder::with_userblock_content) supplies its contents on every output path. |
/// | `H5Pset_libver_bounds` (`fapl`) | [`with_libver_bounds`](Self::with_libver_bounds) | **Genuine** (1.8 / 1.10) | A format selector between the two the writer emits: `high` picks the v2 (HDF5 1.8) superblock at `Earliest..=V18` and the v3 (1.10) one at anything reaching 1.10, and a lower bound above 1.10 (`V112`, `V114`, `LATEST`) selects the 1.10 format too. As in the C library, the low bound permits newer encodings and requires none. An upper bound older than 1.8, or below the lower bound, is rejected with [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable). HDF5 classes this as a file-access property, and it sits here because this crate resolves the bound at write time. |
/// | `H5Pset_fill_value` / `H5Pset_fill_time` (`dcpl`) | [`DatasetBuilder::with_fill_value`](crate::DatasetBuilder::with_fill_value), [`with_fill_time`](crate::DatasetBuilder::with_fill_time) | **Genuine** (per dataset) | Encodes the fill value and fill time in a v3 Fill Value message, and [`Dataset::fill_value`](crate::Dataset::fill_value) and [`Dataset::fill_time`](crate::Dataset::fill_time) read them back, from this crate's files and the C library's. |
/// | `H5Pset_obj_track_times` (`ocpl`) | none | **Unsupported** | Every object is written with times untracked, which is equivalent to `false`, and tracking cannot be enabled. |
/// | `H5Pset_sym_k` / `H5Pset_istore_k` | none | **N/A** | The v3 superblock omits these fields, and every group is new-style, with link messages and a v2 object header. |
/// | `H5Pset_link_phase_change` (`gcpl`) | [`with_root_group_link_phase_change`](Self::with_root_group_link_phase_change) | **Genuine** | Sets root link thresholds. [`GroupBuilder::with_link_phase_change`](crate::GroupBuilder::with_link_phase_change) sets each nested group's thresholds. |
/// | `H5Pset_est_link_info` (`gcpl`) | none | **Unsupported** | New groups use default entry estimates. |
///
/// [space-new]: crate::FileSpaceSettings::new
/// [space-persist]: crate::FileSpaceSettings::with_persistence
/// [space-threshold]: crate::FileSpaceSettings::with_threshold
/// [page]: crate::FileSpaceSettings::with_page_size
///
/// ## Compliance and known limits
///
/// The paged and persistent-free-space paths are exercised by C-library crosschecks
/// (`crates/crosscheck/tests/main/file_space.rs`, `crates/crosscheck/tests/main/bounded_append.rs`): the
/// reference library recovers the strategy, `H5Fget_freespace` equals the crate's tracked total
/// exactly, and the C library reopens a paged file read-write and re-paginates it. The crate also
/// reads and bounded-mutates genuine C-created paged and persisted files. Compliance here means
/// page alignment and structural validity, and it stops short of reproducing the C allocator's
/// intra-page packing byte for byte.
///
/// The limits that come with it:
///
/// - Persistent paged files support staged commits and immediate appends through mirrored or
///   bounded editing, subject to the access-mode restrictions. Nonpersistent paged files require
///   mirrored editing and support eligible staged commits, with new allocations in fresh pages.
///   Immediate appends are unsupported. The [editing guide](crate::_guide::editing) gives the
///   attribute-storage and object-reference restrictions.
/// - Persisted free sections and session-reusable space have separate inspection methods:
///   [`File::persisted_free_sections`](crate::File::persisted_free_sections) and
///   [`File::reusable_free_sections`](crate::File::reusable_free_sections). Their sections and
///   coordinate systems can differ.
/// - The reader supports File Space Info message versions 0 and 1. The writer emits version 1.
///
/// [Limitations](crate#limitations) has the full set of deliberate rejections.
///
/// # Examples
///
/// ```
/// # let directory = tempfile::tempdir()?;
/// # let path = directory.path().join("out.h5");
/// use hdf5_pure::{
///     FileBuilder, FileCreateProperties, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy,
/// };
///
/// let settings = FileSpaceSettings::new(FileSpaceStrategy::Page)
///     .with_persistence(true)
///     .with_threshold(128)
///     .with_page_size(FileSpacePageSize::try_from(8192)?);
/// let properties = FileCreateProperties::new().with_file_space_settings(settings);
///
/// let mut builder = FileBuilder::new();
/// builder.with_create_properties(properties);
/// builder.create_dataset("data").with_f64_data(&[1.0, 2.0]);
/// builder.write(&path)?;
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(alias = "fcpl")]
pub struct FileCreateProperties {
    userblock: UserblockSize,
    libver_bounds: Option<(LibVer, LibVer)>,
    file_space_settings: Option<FileSpaceSettings>,
    root_group_link_phase_change: LinkPhaseChange,
}

impl FileCreateProperties {
    /// Constructs properties that reserve zero prefix bytes and leave format and file-space
    /// settings unset, with default root link thresholds.
    pub const fn new() -> Self {
        Self {
            userblock: UserblockSize::NONE,
            libver_bounds: None,
            file_space_settings: None,
            root_group_link_phase_change: LinkPhaseChange::DEFAULT,
        }
    }

    /// Sets the size of the reserved prefix before the HDF5 signature.
    ///
    /// Pass [`UserblockSize::NONE`] to reserve zero bytes. The writer fills the prefix with
    /// zeros, or with bytes passed to
    /// [`FileBuilder::with_userblock_content`](crate::FileBuilder::with_userblock_content).
    #[doc(alias = "H5Pset_userblock")]
    pub const fn with_userblock(mut self, size: UserblockSize) -> Self {
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

    /// Sets the file-space configuration recorded in the superblock extension.
    ///
    /// Replaces the configured strategy, persistence flag, threshold, and page size together.
    /// Passing [`FileSpaceSettings::default`] requests a recorded File Space Info message.
    /// The writer omits this message when the settings are left unset by [`Self::new`].
    ///
    /// The writer requires the HDF5 1.10 format when settings are configured. It checks the
    /// format bound and constraints between settings when writing.
    #[doc(alias = "H5Pset_file_space_strategy")]
    #[doc(alias = "H5Pset_file_space_page_size")]
    pub const fn with_file_space_settings(mut self, settings: FileSpaceSettings) -> Self {
        self.file_space_settings = Some(settings);
        self
    }

    /// Sets the root group's compact and dense link storage thresholds.
    ///
    /// The default is [`LinkPhaseChange::default`]. Nested groups have their own policies, set through
    /// [`GroupBuilder::with_link_phase_change`](crate::GroupBuilder::with_link_phase_change).
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, FileCreateProperties, LinkPhaseChange};
    ///
    /// let properties = FileCreateProperties::new()
    ///     .with_root_group_link_phase_change(LinkPhaseChange::AlwaysDense);
    /// let mut builder = FileBuilder::new();
    /// builder.with_create_properties(properties);
    /// builder.create_dataset("values").with_i32_data(&[1, 2]);
    /// let file = File::from_bytes(builder.finish()?)?;
    /// assert_eq!(file.root().link_phase_change()?, LinkPhaseChange::AlwaysDense);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    #[doc(alias = "H5Pset_link_phase_change")]
    pub const fn with_root_group_link_phase_change(mut self, policy: LinkPhaseChange) -> Self {
        self.root_group_link_phase_change = policy;
        self
    }

    /// Returns the root group's configured thresholds, including the default policy when unset.
    pub const fn root_group_link_phase_change(&self) -> LinkPhaseChange {
        self.root_group_link_phase_change
    }

    /// Returns the effective file-space configuration, or `None` if it is unset.
    ///
    /// Explicit default settings are returned as `Some`. The values are normalized as described
    /// by [`FileSpaceSettings`], including under strategies without free-space managers.
    pub const fn file_space_settings(&self) -> Option<FileSpaceSettings> {
        self.file_space_settings
    }

    /// Returns the reserved prefix size, with [`UserblockSize::NONE`] for zero bytes.
    pub const fn userblock(&self) -> UserblockSize {
        self.userblock
    }

    /// Return the configured library-version bounds, if any.
    pub const fn libver_bounds(&self) -> Option<(LibVer, LibVer)> {
        self.libver_bounds
    }
}

impl Default for FileCreateProperties {
    fn default() -> Self {
        Self::new()
    }
}
