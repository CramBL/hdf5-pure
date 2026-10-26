use alloc::vec::Vec;

use crate::DecodedStoredAddress;
use crate::FormatError;

/// The effective file-space management configuration.
///
/// Combines a strategy, persistence flag, free-section threshold, and checked page size.
/// [`FileSpaceStrategy::Aggr`] and [`FileSpaceStrategy::None`] have no free-space managers,
/// so their persistence flag is always `false` and their threshold is always 1.
///
/// libhdf5 configures these properties with `H5Pset_file_space_strategy` and
/// `H5Pset_file_space_page_size`. [`FileSpaceInfo`] provides the raw on-disk message fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileSpaceSettings {
    strategy: FileSpaceStrategy,
    persist: bool,
    threshold: u64,
    page_size: FileSpacePageSize,
}

impl FileSpaceSettings {
    /// Creates settings for `strategy` with persistence disabled, threshold 1, and
    /// [`FileSpacePageSize::DEFAULT`].
    pub const fn new(strategy: FileSpaceStrategy) -> Self {
        Self {
            strategy,
            persist: false,
            threshold: Self::DEFAULT_THRESHOLD,
            page_size: FileSpacePageSize::DEFAULT,
        }
    }

    /// Sets whether free-space managers persist their sections across file close.
    ///
    /// Ignores `persist` for [`FileSpaceStrategy::Aggr`] and [`FileSpaceStrategy::None`].
    pub const fn with_persistence(mut self, persist: bool) -> Self {
        self.persist = match self.strategy {
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => persist,
            FileSpaceStrategy::Aggr | FileSpaceStrategy::None => false,
        };
        self
    }

    /// Sets the minimum free-section size, in bytes, for manager tracking.
    ///
    /// Accepts zero. Ignores `threshold` for [`FileSpaceStrategy::Aggr`] and
    /// [`FileSpaceStrategy::None`].
    pub const fn with_threshold(mut self, threshold: u64) -> Self {
        self.threshold = match self.strategy {
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => threshold,
            FileSpaceStrategy::Aggr | FileSpaceStrategy::None => Self::DEFAULT_THRESHOLD,
        };
        self
    }

    /// Sets the recorded page size, which affects allocation under [`FileSpaceStrategy::Page`].
    ///
    /// Retains `size` under every strategy.
    pub const fn with_page_size(mut self, size: FileSpacePageSize) -> Self {
        self.page_size = size;
        self
    }

    /// Returns the strategy for managing file space.
    pub const fn strategy(&self) -> FileSpaceStrategy {
        self.strategy
    }

    /// Returns whether free-space managers persist their sections across file close.
    pub const fn persist(&self) -> bool {
        self.persist
    }

    /// Returns the minimum free-section size for manager tracking, in bytes.
    pub const fn threshold(&self) -> u64 {
        self.threshold
    }

    /// Returns the recorded page size, including under strategies that allocate without pages.
    pub const fn page_size(&self) -> FileSpacePageSize {
        self.page_size
    }

    /// The threshold for newly constructed settings and strategies without managers.
    const DEFAULT_THRESHOLD: u64 = 1;
}

impl Default for FileSpaceSettings {
    /// Returns [`Self::new`] with [`FileSpaceStrategy::FsmAggr`].
    fn default() -> Self {
        Self::new(FileSpaceStrategy::FsmAggr)
    }
}

/// File-space management strategy, mirroring HDF5's `H5F_fspace_strategy_t`
/// (set with `H5Pset_file_space_strategy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSpaceStrategy {
    /// Free-space managers, aggregators, and the virtual file driver, the
    /// HDF5 default. `H5F_FSPACE_STRATEGY_FSM_AGGR`.
    FsmAggr,
    /// Paged aggregation backed by free-space managers.
    /// `H5F_FSPACE_STRATEGY_PAGE`.
    Page,
    /// Aggregators and the virtual file driver only, no free-space managers.
    /// `H5F_FSPACE_STRATEGY_AGGR`.
    Aggr,
    /// No free-space tracking: allocation only ever appends.
    /// `H5F_FSPACE_STRATEGY_NONE`.
    None,
}

/// The raw fields of a File Space Info message in the superblock extension.
///
/// This format-inspection view includes manager addresses and placement metadata. Use
/// [`FileSpaceSettings`] for the effective configuration. `hdf5_pure::File::file_space_info`
/// parses this message, and `hdf5_pure::File::persisted_free_sections` reads its stored free
/// sections.
///
/// A version 0 message has no page size, page-end metadata threshold, or pre-manager end of
/// allocation. The parser supplies [`FileSpacePageSize::DEFAULT`] and a page-end threshold of 0,
/// and leaves [`eoa_pre_fsm`](Self::eoa_pre_fsm) as `None`. Persistent version 0 messages have six
/// manager slots, while version 1 has twelve. The message layouts are defined in "The File Space
/// Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FileSpaceInfo {
    /// The file-space management strategy.
    pub strategy: FileSpaceStrategy,
    /// Whether free space is persisted to disk across file close (the manager
    /// addresses below are written only when this is set).
    pub persist: bool,
    /// Smallest free-space section size the managers track.
    pub threshold: u64,
    /// The recorded page size, or [`FileSpacePageSize::DEFAULT`] for a version 0 message.
    pub page_size: FileSpacePageSize,
    /// The page-end metadata threshold, or 0 for a version 0 message.
    pub page_end_meta_threshold: u16,
    /// The end-of-allocation address recorded before allocating free-space manager metadata, or
    /// `None` for format version 0, which has no such field.
    pub eoa_pre_fsm: Option<DecodedStoredAddress>,
    /// The addresses of the free-space manager headers, present only if [`persist`] is true. An
    /// unused slot holds [`DecodedStoredAddress::Undefined`].
    /// A persistent version 0 message has six slots, and version 1 has twelve.
    ///
    /// [`persist`]: Self::persist
    pub manager_addrs: Vec<DecodedStoredAddress>,
}

/// A file space page size in bytes, from 512 bytes to 1 GiB.
///
/// A version 1 File Space Info message stores a page size under every strategy, and under
/// [`FileSpaceStrategy::Page`] every allocation is aligned to it. A version 0 message stores none.
/// The field is defined in "The File Space Info Message" of the [format specification, version
/// 4.0][spec].
///
/// libhdf5 sets this value with `H5Pset_file_space_page_size`, which rejects a page size
/// outside the same range (`H5Pfcpl.c`, HDF5 2.2.0).
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::FileSpacePageSize;
/// use hdf5_pure_core::FormatError;
///
/// let page_size = FileSpacePageSize::try_from(8192)?;
/// assert_eq!(page_size.get(), 8192);
/// assert_eq!(
///     FileSpacePageSize::try_from(256),
///     Err(FormatError::InvalidFileSpacePageSize(256))
/// );
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FileSpacePageSize(u64);

impl FileSpacePageSize {
    /// Returns the page size in bytes.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The smallest page size, 512 bytes.
    ///
    /// libhdf5 calls this value `H5F_FILE_SPACE_PAGE_SIZE_MIN`.
    pub const MIN: Self = Self(512);

    /// The largest page size, 1 GiB.
    ///
    /// libhdf5 calls this value `H5F_FILE_SPACE_PAGE_SIZE_MAX`.
    pub const MAX: Self = Self(1 << 30);

    /// The default page size, 4096 bytes.
    ///
    /// libhdf5 calls this value `H5F_FILE_SPACE_PAGE_SIZE_DEF`.
    pub const DEFAULT: Self = Self(4096);
}

impl TryFrom<u64> for FileSpacePageSize {
    type Error = FormatError;

    /// Converts a page size in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFileSpacePageSize`] if `page_size` is less than 512 bytes or
    /// more than 1 GiB.
    fn try_from(page_size: u64) -> Result<Self, FormatError> {
        if (Self::MIN.get()..=Self::MAX.get()).contains(&page_size) {
            Ok(Self(page_size))
        } else {
            Err(FormatError::InvalidFileSpacePageSize(page_size))
        }
    }
}

impl Default for FileSpacePageSize {
    /// Returns [`FileSpacePageSize::DEFAULT`], 4096 bytes.
    fn default() -> Self {
        Self::DEFAULT
    }
}
