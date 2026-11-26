use alloc::format;

use hdf5_pure_core::__private::DecodedStoredAddress;

use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::width::{FormatWidths, LengthWidth};

use super::address_space::AddressSpaceBits;
use super::file_section::FILE_FSM_NUM_CLASSES;
use super::geometry::SectionFields;
use super::header::FreeSpaceManagerHeaderFrame;

/// A free-space manager header, signature `FSHD`, with the fields a reader of its section list
/// needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceManagerHeader {
    /// The number of free bytes the manager tracks, the "Total Space Tracked" field.
    total_space: u64,
    /// The number of sections the manager tracks, the "Total Number of Sections" field.
    total_sections: u64,
    /// The number of bits an address in the section list takes, the "Size of Address Space"
    /// field.
    addr_space_bits: AddressSpaceBits,
    /// The section size limit in bytes recorded in the header.
    max_section_size: u64,
    /// The address of the section list.
    section_list_addr: StoredAddress,
    /// The length of the section list in bytes, the "Size of Serialized Section List Used" field.
    section_list_used: u64,
    /// The allocation length in bytes, the "Allocated Size of Serialized Section List" field.
    section_list_allocated: u64,
}

impl FreeSpaceManagerHeader {
    /// Constructs a file-client header from the common fields of a parsed header.
    ///
    /// For a nonempty manager, checks that the section list has a defined address and enough
    /// used bytes for the declared sections. To decode the section records, use
    /// [`parse_section_info`](super::parse_section_info).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if the client ID is not 1, the section
    /// counts do not describe fully serialized sections, the class count exceeds 3, an empty
    /// manager records nonzero free bytes, or a nonempty manager has an undefined section-list
    /// address or insufficient used bytes.
    pub fn from_frame(frame: &FreeSpaceManagerHeaderFrame) -> Result<Self, FormatError> {
        let FormatWidths { offsets, lengths } = frame.widths();
        let client_id = frame.client_id();
        let total_sections = frame.total_sections();
        let serialized_sections = frame.serialized_sections();
        let unserialized_sections = frame.unserialized_sections();
        let section_classes = frame.section_classes();
        let total_space = frame.total_space();
        let section_list_addr = frame.section_list_addr();
        let section_list_used = frame.section_list_used();
        if client_id != FILE_FSM_CLIENT_ID {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the client ID is {client_id}, not {FILE_FSM_CLIENT_ID}, the ID of the file client"
            )));
        }
        if (serialized_sections, unserialized_sections) != (total_sections, 0) {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header's section count is {total_sections}, its serialized section count \
                 {serialized_sections} and its unserialized section count {unserialized_sections}, \
                 and the file client serializes every section"
            )));
        }
        if section_classes > FILE_FSM_NUM_CLASSES {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header registers {section_classes} section classes, more than the \
                 {FILE_FSM_NUM_CLASSES} of the file client"
            )));
        }
        let header = Self {
            total_space,
            total_sections,
            addr_space_bits: frame.address_space(),
            max_section_size: frame.max_section_size(),
            section_list_addr: match section_list_addr {
                DecodedStoredAddress::Defined(addr) => addr,
                DecodedStoredAddress::Undefined => StoredAddress::undefined(offsets),
            },
            section_list_used,
            section_list_allocated: frame.section_list_allocated(),
        };
        if total_sections == 0 {
            if total_space != 0 {
                return Err(FormatError::InvalidFreeSpaceManager(format!(
                    "the empty manager tracks {total_space} bytes"
                )));
            }
            return Ok(header);
        }
        if section_list_addr == DecodedStoredAddress::Undefined {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header's section count is {total_sections}, and it stores no section list"
            )));
        }
        let max_sections = header
            .section_fields(lengths)
            .max_sections(offsets, section_list_used);
        if total_sections > max_sections {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header's section count is {total_sections}, and a section list of {section_list_used} \
                 bytes holds at most {max_sections}"
            )));
        }
        Ok(header)
    }

    /// Parses the free-space manager header at the start of `data`.
    ///
    /// Reads each length field at the length width of `widths` and the section list address at the
    /// offset width, and accepts version 0 of the header of a file client, client ID 1, alone.
    /// Checks that the used length is at most the allocated length. For a header of one or more
    /// sections, checks that the section list address is defined and the used length holds that
    /// many sections, so a reader
    /// may reserve room for [`total_sections`](Self::total_sections) sections before it reads the
    /// section list.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the header,
    /// [`FormatError::ChecksumMismatch`] if the checksum does not match and the `checksum` feature
    /// is enabled, [`FormatError::UnsupportedVersion`] if the version is not 0, and
    /// [`FormatError::InvalidFreeSpaceManager`] if the signature is not `FSHD`, the client ID
    /// is not 1, the serialized section count differs from the section count, the "Number of
    /// Un-Serialized Sections" field is not 0, the header counts more
    /// section classes than the 3 the C library registers for the file client (`H5MF.c`, HDF5
    /// 2.2.0), the address space is 0 bits or wider than 64 bits, the used length exceeds the
    /// allocated length, an empty manager records nonzero free bytes, or a header of one or
    /// more sections stores the undefined address for its section list or a used length too
    /// short for its sections.
    pub fn parse(widths: FormatWidths, data: &[u8]) -> Result<FreeSpaceManagerHeader, FormatError> {
        Self::from_frame(&FreeSpaceManagerHeaderFrame::parse(widths, data)?)
    }

    /// Returns the number of free bytes the manager tracks, the "Total Space Tracked" field.
    pub fn total_space(&self) -> u64 {
        self.total_space
    }

    /// Returns the number of sections the manager tracks, the "Total Number of Sections" field.
    pub fn total_sections(&self) -> u64 {
        self.total_sections
    }

    /// Returns the number of bits an address in the section list takes, the "Size of Address
    /// Space" field.
    pub fn addr_space_bits(&self) -> u16 {
        self.addr_space_bits.get()
    }

    /// Returns the size limit in bytes recorded in the "Maximum Section Size" field.
    pub fn max_section_size(&self) -> u64 {
        self.max_section_size
    }

    /// Returns the address of the section list, the "Address of Serialized Section List" field.
    pub fn section_list_addr(&self) -> StoredAddress {
        self.section_list_addr
    }

    /// Returns the length of the section list in bytes, the "Size of Serialized Section List
    /// Used" field.
    ///
    /// The parser requires this many bytes, including the checksum.
    pub fn section_list_used(&self) -> u64 {
        self.section_list_used
    }

    /// Returns the allocation length in bytes, the "Allocated Size of Serialized Section List"
    /// field.
    ///
    /// This length is at least [`section_list_used`](Self::section_list_used). Version 0 of the
    /// free-space manager header is defined in "Free-space Index" of the [format specification,
    /// version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
    pub fn section_list_allocated(&self) -> u64 {
        self.section_list_allocated
    }

    /// Returns the widths of the fields of the section list this header describes, in a file with
    /// lengths of `lengths`.
    pub(super) fn section_fields(&self, lengths: LengthWidth) -> SectionFields {
        SectionFields::from_header(
            self.total_sections,
            self.max_section_size,
            self.addr_space_bits,
            lengths,
        )
    }
}

/// The shrink percent the header stores, `H5MF_FSPACE_SHRINK`.
pub(super) const FILE_FSM_SHRINK_PCT: u16 = 80;
/// The expand percent the header stores, `H5MF_FSPACE_EXPAND`.
pub(super) const FILE_FSM_EXPAND_PCT: u16 = 120;
/// The client ID of a file free-space manager, where 0 is a fractal heap's.
pub(super) const FILE_FSM_CLIENT_ID: u8 = 1;

#[cfg(test)]
#[path = "tests/file_header.rs"]
mod tests;
