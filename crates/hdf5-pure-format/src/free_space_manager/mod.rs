//! Free-space manager headers (`FSHD`), section lists (`FSSE`), and file-client records.
//!
//! The common framing APIs parse caller-supplied bytes and retain the client ID for
//! client-specific validation. [`FreeSpaceManagerHeader`] and [`parse_section_info`] validate
//! file-client metadata. The complete allocation and used bytes required for parsing are
//! recorded separately in a section-list reference.
//!
//! Version 0 headers and section lists are defined in "Free-space Index" of the [format
//! specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex

mod address_space;
mod file_header;
mod file_section;
mod geometry;
mod header;
mod records;
mod section_list;
#[cfg(test)]
mod test_support;
mod write;

pub use file_header::FreeSpaceManagerHeader;
pub use file_section::{
    FreeSection, SECTION_CLASS_LARGE, SECTION_CLASS_SIMPLE, SECTION_CLASS_SMALL,
};
pub use geometry::{
    SectionInfoCounts, free_space_manager_len, section_info_capacity, section_info_len,
};
pub use header::{
    FreeSpaceManagerHeaderFrame, FreeSpaceManagerHeaderRef, free_space_manager_header_len,
};
pub use records::{parse_section_info, visit_section_info};
pub use section_list::{FreeSpaceManagerSectionListRef, FreeSpaceManagerSectionListView};
pub use write::{
    FreeSpaceManagerEncoding, FreeSpaceManagerHeaderEncoding, FreeSpaceManagerSectionListEncoding,
    PreparedFreeSpaceManager, serialize_free_space_manager,
    serialize_free_space_manager_with_allocation,
};
