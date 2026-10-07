use alloc::vec::Vec;

use crate::FormatError;

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

/// A File Space Info message, the file space management settings a file stores in its superblock
/// extension.
///
/// `hdf5_pure::File::file_space_info` returns the message of a file that has one, and
/// `hdf5_pure::FileBuilder::with_file_space_strategy` sets the strategy of the file the builder
/// writes. The message is defined in "The File Space Info Message" of the [format specification,
/// version 4.0][spec].
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
    /// File-space page size used for paged allocation.
    pub page_size: FileSpacePageSize,
    /// Page-end metadata threshold (paged allocation tuning).
    pub page_end_meta_threshold: u16,
    /// End-of-allocation address recorded before free-space manager metadata
    /// was allocated, or the undefined address, all ones at the file's offset
    /// width, where the file records none.
    pub eoa_pre_fsm: u64,
    /// The addresses of the free-space manager headers, present only if [`persist`] is set. An
    /// unused slot holds the undefined address, all ones at the file's offset width.
    /// `hdf5_pure::File::persisted_free_space` reads the free regions that these managers track.
    ///
    /// [`persist`]: Self::persist
    pub manager_addrs: Vec<u64>,
}

/// A file space page size in bytes, from 512 bytes to 1 GiB.
///
/// A version 1 File Space Info message stores a page size under every strategy, and under
/// [`FileSpaceStrategy::Page`] every allocation is aligned to it. A version 0 message stores none.
/// The field is defined in "The File Space Info Message" of the [format specification, version
/// 4.0][spec].
///
/// The C library sets this value with `H5Pset_file_space_page_size`, which rejects a page size
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
    /// The C library calls this value `H5F_FILE_SPACE_PAGE_SIZE_MIN`.
    pub const MIN: Self = Self(512);

    /// The largest page size, 1 GiB.
    ///
    /// The C library calls this value `H5F_FILE_SPACE_PAGE_SIZE_MAX`.
    pub const MAX: Self = Self(1 << 30);

    /// The default page size, 4096 bytes.
    ///
    /// The C library calls this value `H5F_FILE_SPACE_PAGE_SIZE_DEF`.
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
