use crate::address::StoredAddress;

/// One free region of a file: the address the file stores for it and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeSection {
    /// The address of the region.
    pub addr: StoredAddress,
    /// The length of the region in bytes.
    pub size: u64,
}

/// The number of section classes the header of a file free-space manager stores, the three
/// classes the C library registers for the file client (`H5MF.c`, HDF5 2.2.0).
pub(super) const FILE_FSM_NUM_CLASSES: u16 = 3;

/// The section class of the free space of a file that is not paged, `H5MF_FSPACE_SECT_SIMPLE`.
///
/// A section of each of the three classes stores its offset and its class, and no data.
pub const SECTION_CLASS_SIMPLE: u8 = 0;
/// The section class of free space smaller than a page in a paged file, `H5MF_FSPACE_SECT_SMALL`.
pub const SECTION_CLASS_SMALL: u8 = 1;
/// The section class of the manager of free space of a page or more in a paged file,
/// `H5MF_FSPACE_SECT_LARGE`.
///
/// The class identifies the manager, and a section of it may be smaller than a page.
pub const SECTION_CLASS_LARGE: u8 = 2;
