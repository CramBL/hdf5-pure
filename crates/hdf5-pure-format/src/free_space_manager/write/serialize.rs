use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes::UnexpectedEof;
use crate::codec::Encode;
use crate::error::FormatError;
use crate::free_space_manager::FreeSection;
use crate::width::FormatWidths;

use super::prepared::PreparedFreeSpaceManager;

/// Serializes the header, for `fshd_addr`, and the section list, for `fsse_addr`, of a file
/// free-space manager that holds `sections`.
///
/// Each block stores the address of the other and ends in its own checksum. The writer groups
/// the sections by size and writes them in ascending order of size, then address. It writes
/// `class_id` for every section and uses the shortest section list for those groups.
///
/// Below 8-byte lengths the header stores a maximum section size of all ones less two, and the
/// section list stores each section size at the length width. libhdf5 allocates as many
/// bins as the base 2 logarithm of the maximum section size, rounded down, and files a section of
/// `size` bytes in bin `log2(size)`, so it has a bin only for a section smaller than the largest
/// power of two at or below the maximum section size.
///
/// Returns the header and the section list.
///
/// # Errors
///
/// Returns the errors of [`PreparedFreeSpaceManager::new`] and [`PreparedFreeSpaceManager::place`],
/// or [`FormatError::AllocationFailed`] if the output buffers could not be allocated.
pub fn serialize_free_space_manager(
    widths: FormatWidths,
    sections: &[FreeSection],
    fshd_addr: StoredAddress,
    fsse_addr: StoredAddress,
    class_id: u8,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    serialize_free_space_manager_with_allocation(
        widths, sections, fshd_addr, fsse_addr, class_id, None,
    )
}

/// Serializes a free-space manager with a separately specified section-list allocation.
///
/// Passing `None` serializes the shortest section list for `sections`. Passing `Some` reserves
/// that many bytes in the returned section list, with zero padding between the section records
/// and its trailing checksum. The header stores the complete returned length in both its used
/// and allocated length fields.
///
/// The version 0 header and section list are defined in "Free-space Index" of the [format
/// specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_format::__private as format;
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::FreeSection;
/// use hdf5_pure_format::__private::FreeSpaceManagerHeader;
///
/// let widths = FormatWidths::from_sizes(8, 8)?;
/// let sections = [FreeSection { addr: StoredAddress::new(256), size: 32 }];
/// let fshd_addr = StoredAddress::new(512);
/// let fsse_addr = StoredAddress::new(594);
/// let (header_bytes, section_list) = format::serialize_free_space_manager_with_allocation(
///     widths, &sections, fshd_addr, fsse_addr, format::SECTION_CLASS_SIMPLE, None,
/// )?;
/// let used = FreeSpaceManagerHeader::parse(widths, &header_bytes)?.section_list_used();
/// let allocated = used + 64;
/// let (header_bytes, allocated_section_list) =
///     format::serialize_free_space_manager_with_allocation(
///         widths, &sections, fshd_addr, fsse_addr, format::SECTION_CLASS_SIMPLE, Some(allocated),
///     )?;
/// let header = FreeSpaceManagerHeader::parse(widths, &header_bytes)?;
/// assert_eq!(allocated_section_list.len() as u64, allocated);
/// assert!(allocated_section_list.len() > section_list.len());
/// assert_eq!(header.section_list_used(), allocated);
/// assert_eq!(header.section_list_allocated(), allocated);
/// # Ok::<(), hdf5_pure_format::FormatError>(())
/// ```
///
/// # Errors
///
/// Returns [`FormatError::FreeSpaceSectionListAllocationTooSmall`] if `section_list_allocated`
/// is shorter than the shortest serialized section list. Other failures are the errors of
/// [`serialize_free_space_manager`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
pub fn serialize_free_space_manager_with_allocation(
    widths: FormatWidths,
    sections: &[FreeSection],
    fshd_addr: StoredAddress,
    fsse_addr: StoredAddress,
    class_id: u8,
    section_list_allocated: Option<u64>,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let plan = PreparedFreeSpaceManager::new(widths, sections.iter().copied(), class_id)?;
    let manager = plan.place(fshd_addr, fsse_addr, section_list_allocated)?;
    Ok((
        encode_to_vec(manager.header())?,
        encode_to_vec(manager.section_list())?,
    ))
}

/// Reserves a buffer for the block's complete encoding, then writes the block into it.
///
/// # Errors
///
/// Returns [`FormatError::AllocationFailed`] if the reservation fails. Propagates errors
/// from [`Encode::encode_into`].
fn encode_to_vec(block: impl Encode<Error = UnexpectedEof>) -> Result<Vec<u8>, FormatError> {
    let length = block.encoded_len();
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|source| FormatError::AllocationFailed {
            values: length,
            source,
        })?;
    bytes.resize(length, 0);
    block.encode_into(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
#[path = "../tests/file_writing.rs"]
mod tests;
