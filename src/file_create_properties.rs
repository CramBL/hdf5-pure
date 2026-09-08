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
/// rule; see the property-support reference for the exact coverage.
///
/// See the [property-support reference] for the full property-by-property map.
///
/// [property-support reference]: https://github.com/CramBL/hdf5-pure/blob/main/docs/reference/property-support.md
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
