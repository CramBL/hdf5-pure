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

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::num::NonZeroU64;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

mod address_space;
mod file_header;
mod header;
mod section_list;
#[cfg(test)]
mod test_support;
mod write;

use address_space::AddressSpaceBits;
use file_header::FILE_FSM_NUM_CLASSES;

pub use file_header::FreeSpaceManagerHeader;
pub use header::{
    FreeSpaceManagerHeaderFrame, FreeSpaceManagerHeaderRef, free_space_manager_header_len,
};
pub use section_list::{FreeSpaceManagerSectionListRef, FreeSpaceManagerSectionListView};
pub use write::{
    FreeSpaceManagerEncoding, FreeSpaceManagerHeaderEncoding, FreeSpaceManagerSectionListEncoding,
    PreparedFreeSpaceManager,
};

/// One free region of a file: the address the file stores for it and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeSection {
    /// The address of the region.
    pub addr: StoredAddress,
    /// The length of the region in bytes.
    pub size: u64,
}

/// Returns the fewest bytes that hold `value`, and at least 1, as `H5VM_limit_enc_size` computes
/// it.
fn enc_size(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    bits.div_ceil(8).max(1)
}

/// Reads `bytes` as a little-endian unsigned integer.
fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

/// The maximum section size libhdf5 gives a file free-space manager under its default driver, sec2,
/// whatever the file's widths: the driver's maximum address, `2^63 - 1` (`H5MF__create_fstype` in
/// `H5MF.c`, HDF5 1.10.11 to 2.2.0), which HDF5 2.2.0 defines as `H5FD_MAXADDR`. The family, multi
/// and MPI-IO drivers give `HADDR_MAX`, `2^64 - 2`, and the core driver `SIZE_MAX - 1`.
const FILE_FSM_MAX_SECTION_SIZE: u64 = (1u64 << 63) - 1;

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

/// Serializes the header, for `fshd_addr`, and the section list, for `fsse_addr`, of a file
/// free-space manager that holds `sections`.
///
/// The header stores the address of the section list, and the section list the address of the
/// header. The section list groups the sections by size, in ascending order of size and then of
/// address, as the C library serializes them (`H5FScache.c`, HDF5 2.2.0), and tags every section
/// with `class_id`, since a manager holds sections of one class. Each block ends in its checksum,
/// and in a file with 8-byte addresses and lengths both match the blocks the C library writes for
/// the same sections.
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
        write::encode_to_vec(manager.header())?,
        write::encode_to_vec(manager.section_list())?,
    ))
}

/// Returns the length in bytes of the header and the section list of a file free-space manager
/// that holds `sections`, or 0 if `sections` is empty.
///
/// A file with no free space stores the undefined address in place of a manager. The length
/// depends on `widths` and on the number and the sizes of the sections, and not on their
/// addresses, so a caller sizes the blocks before it places them.
pub fn free_space_manager_len(widths: FormatWidths, sections: &[FreeSection]) -> u64 {
    if sections.is_empty() {
        return 0;
    }
    let sets = sections
        .iter()
        .map(|section| section.size)
        .collect::<BTreeSet<u64>>()
        .len();
    free_space_manager_header_len(widths) + section_list_len(widths, sets, sections.len())
}

/// Returns the length in bytes of the section list of a manager that holds sections of
/// `section_sizes`.
///
/// The length depends on `widths`, the number of sections, and the number of distinct sizes, and
/// not on the addresses of the sections, and is the length of the section list
/// [`serialize_free_space_manager`] serializes for sections of these sizes.
pub fn section_info_len(widths: FormatWidths, section_sizes: &[u64]) -> u64 {
    let sets = section_sizes.iter().collect::<BTreeSet<&u64>>().len();
    section_list_len(widths, sets, section_sizes.len())
}

/// Consistent section and distinct-size counts for section-list measurement.
///
/// Both counts are zero for an empty list. A nonempty list has at least one size group and
/// at most one group per section. Use [`Self::encoded_len`] to measure the writer's section-list
/// framing at the output field widths.
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::SectionInfoCounts;
///
/// let widths = FormatWidths::from_sizes(8, 8)?;
/// let same_size = SectionInfoCounts::new(3, 1)?;
/// let distinct_sizes = SectionInfoCounts::new(3, 3)?;
/// assert_eq!(same_size.encoded_len(widths)?, 53);
/// assert_eq!(distinct_sizes.encoded_len(widths)?, 71);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectionInfoCounts {
    sections: u64,
    size_groups: u64,
}

impl SectionInfoCounts {
    /// Creates counts for `sections` section records of `size_groups` distinct lengths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceSectionCounts`] if exactly one count is zero,
    /// or if `size_groups` exceeds `sections`.
    pub fn new(sections: u64, size_groups: u64) -> Result<Self, FormatError> {
        if (sections == 0) != (size_groups == 0) || size_groups > sections {
            return Err(FormatError::InvalidFreeSpaceSectionCounts {
                sections,
                size_groups,
            });
        }
        Ok(Self {
            sections,
            size_groups,
        })
    }

    /// Returns the section-list encoding length before allocation padding, in bytes.
    ///
    /// Uses the field widths of [`serialize_free_space_manager`] and includes the section-list
    /// prefix and checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the encoded length exceeds [`u64::MAX`].
    pub fn encoded_len(self, widths: FormatWidths) -> Result<u64, FormatError> {
        let Self {
            sections,
            size_groups,
        } = self;
        let fields = SectionFields::written(widths, sections);
        let fixed = (section_list::section_list_prefix_len(widths.offsets) + CHECKSUM_LEN).to_u64();
        size_groups
            .checked_mul(fields.set_len().to_u64())
            .and_then(|sets| {
                sections
                    .checked_mul(fields.section_len().to_u64())
                    .and_then(|records| sets.checked_add(records))
            })
            .and_then(|payload| fixed.checked_add(payload))
            .ok_or(FormatError::OffsetOverflow {
                offset: fixed,
                length: sections,
            })
    }
    /// Returns the writer's section-count field width in bytes, including for an empty list.
    ///
    /// The width depends on the total section count and is at least one byte.
    pub fn count_width(self) -> usize {
        enc_size(self.sections)
    }
}

/// Returns a section-list capacity sufficient for at most `max_sections` sections.
///
/// Counts each section as a separate size group, so the capacity covers every grouping of
/// section sizes for `widths`. Includes the prefix and checksum even when the
/// section count is zero.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the capacity exceeds [`u64::MAX`].
pub fn section_info_capacity(
    widths: FormatWidths,
    max_sections: usize,
) -> Result<u64, FormatError> {
    SectionInfoCounts::new(max_sections.to_u64(), max_sections.to_u64())?.encoded_len(widths)
}

/// Parses the section list of the manager `header` describes, at the start of `data`, into its free
/// sections.
///
/// The section list occupies the [used length](FreeSpaceManagerHeader::section_list_used) at the
/// start of `data` and ends in its checksum. The file stores `header_addr` relative to the base
/// address. Reads the [declared number](FreeSpaceManagerHeader::total_sections) of sections and
/// ignores unused bytes between the last section and the checksum. Reads each group's count and
/// size together, then acquires all its records before decoding any of them. With the `checksum`
/// feature, verifies the checksum over the used list up to its final four bytes, including
/// unused bytes. libhdf5 skips the unused bytes when reading (`H5FS__cache_sinfo_deserialize` in
/// `H5FScache.c`, HDF5 1.12.3) and registers 3 section classes for the file client (`H5MF.c`,
/// HDF5 2.2.0).
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the used list, its prefix and checksum, or a group
/// is incomplete. For group truncation, the expected end is relative to the start of the list
/// and the available length excludes the checksum.
///
/// Returns [`FormatError::ChecksumMismatch`] if the checksum does not match and the `checksum`
/// feature is enabled, [`FormatError::UnsupportedVersion`] if the version is not 0, or
/// [`FormatError::InvalidFreeSpaceManager`] if the signature is not `FSSE` or the stored header
/// address differs from `header_addr`.
///
/// Returns [`FormatError::InvalidFreeSpaceSectionGroupCount`] if a group declares zero sections or
/// more sections than remain in the header's count, [`FormatError::InvalidFreeSpaceSectionGroupSize`]
/// if a group declares sections of zero bytes, or [`FormatError::InvalidFreeSpaceSectionClass`]
/// if a section's class is unsupported for the file client. Returns
/// [`FormatError::InvalidFreeSpaceSectionSizeSum`] if the decoded section sizes do not sum to the
/// header's total free space.
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if the used length or total section count
/// does not fit a [`usize`].
pub fn parse_section_info(
    widths: FormatWidths,
    data: &[u8],
    header_addr: StoredAddress,
    header: &FreeSpaceManagerHeader,
) -> Result<Vec<FreeSection>, FormatError> {
    SectionInfo::parse(widths, data, header_addr, header)?.collect()
}

/// Calls `visit` for each decoded free section in encoded order.
///
/// Uses the framing and record checks of [`parse_section_info`] without collecting sections.
/// The callback may have been called before a later group, record, or total-space check fails.
/// The caller must check the result before treating the complete section list as valid.
///
/// # Errors
///
/// Returns the parsing errors described by [`parse_section_info`].
pub fn visit_section_info(
    widths: FormatWidths,
    data: &[u8],
    header_addr: StoredAddress,
    header: &FreeSpaceManagerHeader,
    visit: impl FnMut(FreeSection),
) -> Result<(), FormatError> {
    SectionInfo::parse(widths, data, header_addr, header)?.visit(visit)
}

/// A borrowed section list with checked framing and a section count that fits a [`usize`].
///
/// Record counts, classes, and total space require [`Self::visit`] or [`Self::collect`].
struct SectionInfo<'a> {
    /// The cursor over the used list without its checksum, positioned at the first group.
    cursor: bytes::ByteCursor<'a>,
    fields: SectionFields,
    /// The header's declared section count.
    total: usize,
    /// The header's declared total free space, in bytes.
    space: u64,
}

impl<'a> SectionInfo<'a> {
    /// Checks the used length and framing before any record storage is allocated.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the used length or section count
    /// exceeds [`usize::MAX`], [`FormatError::UnexpectedEof`] if `data` is shorter than the used
    /// length, or the framing errors of [`section_list::parse_section_list_prefix`].
    fn parse(
        widths: FormatWidths,
        data: &'a [u8],
        header_addr: StoredAddress,
        header: &FreeSpaceManagerHeader,
    ) -> Result<Self, FormatError> {
        let used = header.section_list_used().to_usize()?;
        let list = data.get(..used).ok_or(FormatError::UnexpectedEof {
            expected: used,
            available: data.len(),
        })?;
        let start = section_list::section_list_prefix_len(widths.offsets);
        section_list::parse_section_list_prefix(widths.offsets, list, header_addr)?;
        Ok(Self {
            cursor: bytes::ByteCursor::with_position(&list[..list.len() - CHECKSUM_LEN], start)?,
            fields: header.section_fields(widths.lengths),
            total: header.total_sections().to_usize()?,
            space: header.total_space(),
        })
    }

    /// Collects checked sections in encoded order with capacity for the declared section count.
    ///
    /// # Errors
    ///
    /// Returns the record and total-space errors of [`Self::visit`].
    fn collect(self) -> Result<Vec<FreeSection>, FormatError> {
        let mut sections = Vec::with_capacity(self.total);
        self.visit(|section| sections.push(section))?;
        Ok(sections)
    }

    /// Checks records and their total space while calling `visit` for each decoded section.
    ///
    /// The callback may have been called before a later check fails.
    ///
    /// # Errors
    ///
    /// Returns the group and record errors of [`SectionFields::visit_sets`], or
    /// [`FormatError::InvalidFreeSpaceSectionSizeSum`] if the sizes do not sum to the header's
    /// total free space.
    fn visit(self, mut visit: impl FnMut(FreeSection)) -> Result<(), FormatError> {
        let Self {
            cursor,
            fields,
            total,
            space: expected,
        } = self;
        let mut space = 0_u128;
        fields.visit_sets(cursor, total, |section| {
            space += u128::from(section.size);
            visit(section);
        })?;
        if space != u128::from(expected) {
            return Err(FormatError::InvalidFreeSpaceSectionSizeSum {
                expected,
                actual: Box::new(space),
            });
        }
        Ok(())
    }
}

/// The widths in bytes of the variable-size fields of a section list: the number of sections in a
/// set, the section size of a set, and the offset of a section.
#[derive(Clone, Copy, Debug)]
struct SectionFields {
    count: usize,
    size: usize,
    offset: usize,
}

impl SectionFields {
    /// Returns the widths of the fields of the section list [`serialize_free_space_manager`]
    /// writes at `widths` for a manager of `sections` sections.
    fn written(widths: FormatWidths, sections: u64) -> Self {
        Self {
            count: enc_size(sections),
            size: enc_size(written_max_section_size(widths.lengths).get()),
            offset: AddressSpaceBits::written(widths.offsets).offset_width(),
        }
    }

    /// Returns the combined count and section-size field width of one size group, in bytes.
    fn set_len(self) -> usize {
        self.count + self.size
    }

    /// Returns the combined address and class field width of one section record, in bytes.
    fn section_len(self) -> usize {
        self.offset + SECTION_CLASS_LEN
    }

    /// Returns the most sections a section list of `list_len` bytes holds at these widths, in a
    /// file with addresses of `offsets`: the sections of one set between the prefix and the
    /// checksum.
    fn max_sections(self, offsets: OffsetWidth, list_len: u64) -> u64 {
        let Self {
            count,
            size,
            offset,
        } = self;
        let fixed =
            (section_list::section_list_prefix_len(offsets) + count + size + CHECKSUM_LEN).to_u64();
        list_len
            .checked_sub(fixed)
            .map_or(0, |payload| payload / (offset + SECTION_CLASS_LEN).to_u64())
    }

    /// Calls `visit` for `remaining` sections from the size groups at `cursor`.
    ///
    /// The caller positions the cursor at the first group in the used list and excludes the
    /// checksum from its input.
    ///
    /// # Errors
    ///
    /// Returns the group and record errors of [`parse_section_info`].
    fn visit_sets(
        self,
        mut cursor: bytes::ByteCursor<'_>,
        mut remaining: usize,
        mut visit: impl FnMut(FreeSection),
    ) -> Result<(), FormatError> {
        let record_len = self.section_len();
        while remaining != 0 {
            let offset = cursor.position();
            let (count, size) = cursor.take(self.set_len())?.split_at(self.count);
            let count = read_uint_le(count);
            if count == 0 || count > remaining.to_u64() {
                return Err(FormatError::InvalidFreeSpaceSectionGroupCount {
                    offset,
                    count,
                    remaining: remaining.to_u64(),
                });
            }
            let size = NonZeroU64::new(read_uint_le(size))
                .ok_or(FormatError::InvalidFreeSpaceSectionGroupSize { offset })?;
            let count = count.to_usize()?;
            // If multiplication overflows, the span exceeds the maximum slice length.
            // The cursor reports its end as `usize::MAX`.
            let records_start = cursor.position();
            let records = cursor.take(count.saturating_mul(record_len))?;
            for (index, record) in records.chunks_exact(record_len).enumerate() {
                let class_id = record[self.offset];
                if u16::from(class_id) >= FILE_FSM_NUM_CLASSES {
                    // Each record starts in the acquired span, within the original input.
                    return Err(FormatError::InvalidFreeSpaceSectionClass {
                        offset: records_start + index * record_len,
                        class_id,
                    });
                }
                visit(FreeSection {
                    addr: StoredAddress::new(read_uint_le(&record[..self.offset])),
                    size: size.get(),
                });
            }
            remaining -= count;
        }
        Ok(())
    }
}

/// Returns the maximum section size the serializer writes at `lengths`.
///
/// libhdf5 checks the bin of a section against the number of bins in an `assert` alone
/// (`H5FS__sect_link_size` in `H5FSsection.c`, HDF5 1.10.11 to 2.2.0). Below 8-byte lengths the
/// maximum is all ones less two, which has the top bit of the field set, so libhdf5 allocates as
/// many bins for it as for the largest value of the field (`H5FS__sinfo_new`). It is below the
/// truncated addresses [`FreeSpaceManagerHeader::section_size_width`] reads as libhdf5's.
fn written_max_section_size(lengths: LengthWidth) -> NonZeroU64 {
    match lengths {
        LengthWidth::Two => {
            const { NonZeroU64::new(width::length_width_max(LengthWidth::Two) - 2).unwrap() }
        }
        LengthWidth::Four => {
            const { NonZeroU64::new(width::length_width_max(LengthWidth::Four) - 2).unwrap() }
        }
        LengthWidth::Eight => const { NonZeroU64::new(FILE_FSM_MAX_SECTION_SIZE).unwrap() },
    }
}

/// Returns the length in bytes of the section list [`serialize_free_space_manager`] writes at
/// `widths` for `sections` sections of `sets` distinct sizes.
fn section_list_len(widths: FormatWidths, sets: usize, sections: usize) -> u64 {
    let fields = SectionFields::written(widths, sections.to_u64());
    (section_list::section_list_prefix_len(widths.offsets)
        + sets * fields.set_len()
        + sections * fields.section_len()
        + CHECKSUM_LEN)
        .to_u64()
}

const CHECKSUM_LEN: usize = 4;
const SECTION_CLASS_LEN: usize = 1;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::free_space;

    #[cfg(feature = "checksum")]
    use crate::checksum;

    use super::*;

    #[rstest]
    #[case::empty(test_support::widths(8, 8), 0, 0, 17)]
    #[case::one(test_support::widths(8, 8), 1, 1, 35)]
    #[case::one_group(test_support::widths(8, 8), 3, 1, 53)]
    #[case::distinct_sizes(test_support::widths(8, 8), 3, 3, 71)]
    #[case::two_byte_fields(test_support::widths(2, 2), 3, 2, 26)]
    #[case::four_byte_fields(test_support::widths(4, 4), 3, 2, 38)]
    #[case::four_byte_lengths(test_support::widths(8, 4), 3, 2, 54)]
    #[case::four_byte_offsets(test_support::widths(4, 8), 3, 2, 46)]
    #[case::two_byte_lengths(test_support::widths(8, 2), 3, 2, 50)]
    #[case::two_byte_offsets(test_support::widths(2, 8), 3, 2, 38)]
    #[case::last_one_byte_count(test_support::widths(8, 8), 255, 1, 2_321)]
    #[case::first_two_byte_count(test_support::widths(8, 8), 256, 1, 2_331)]
    #[case::last_two_byte_count(test_support::widths(8, 8), 65_535, 1, 589_842)]
    #[case::first_three_byte_count(test_support::widths(8, 8), 65_536, 1, 589_852)]
    #[case::last_four_byte_count(test_support::widths(8, 8), (1 << 32) - 1, 1, 38_654_705_684)]
    #[case::first_five_byte_count(test_support::widths(8, 8), 1 << 32, 1, 38_654_705_694)]
    #[case::eight_byte_count(test_support::widths(8, 8), 1 << 56, 1, 648_518_346_341_351_457)]
    fn section_counts_measure_current_wire_fields(
        #[case] widths: FormatWidths,
        #[case] sections: u64,
        #[case] size_groups: u64,
        #[case] expected: u64,
    ) {
        assert_eq!(
            SectionInfoCounts::new(sections, size_groups)
                .unwrap()
                .encoded_len(widths)
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::empty(0, 1)]
    #[case::last_one_byte(255, 1)]
    #[case::first_two_bytes(256, 2)]
    #[case::first_three_bytes(65_536, 3)]
    #[case::first_four_bytes(1 << 24, 4)]
    #[case::first_five_bytes(1 << 32, 5)]
    #[case::first_six_bytes(1 << 40, 6)]
    #[case::first_seven_bytes(1 << 48, 7)]
    #[case::first_eight_bytes(1 << 56, 8)]
    fn section_counts_expose_the_total_count_field_width(
        #[case] sections: u64,
        #[case] expected: usize,
    ) {
        assert_eq!(
            SectionInfoCounts::new(sections, u64::from(sections != 0))
                .unwrap()
                .count_width(),
            expected
        );
    }

    #[rstest]
    #[case::one_section(1, 1, free_space::single_section_info(), 0)]
    #[case::two_sizes(2, 2, free_space::two_section_info(), 0)]
    #[case::allocated_padding(10, 9, free_space::oversized_section_list(), 9)]
    fn section_counts_match_known_native_lengths(
        #[case] sections: u64,
        #[case] size_groups: u64,
        #[case] observed: Vec<u8>,
        #[case] padding: u64,
    ) {
        assert_eq!(
            SectionInfoCounts::new(sections, size_groups)
                .unwrap()
                .encoded_len(test_support::widths(8, 8))
                .unwrap()
                + padding,
            observed.len().to_u64()
        );
    }

    #[rstest]
    #[case::groups_without_sections(0, 1)]
    #[case::sections_without_groups(1, 0)]
    #[case::too_many_groups(3, 4)]
    fn section_counts_reject_impossible_groups(#[case] sections: u64, #[case] size_groups: u64) {
        assert_eq!(
            SectionInfoCounts::new(sections, size_groups).unwrap_err(),
            FormatError::InvalidFreeSpaceSectionCounts {
                sections,
                size_groups
            }
        );
    }

    #[rstest]
    #[case::records(u64::MAX, 1)]
    #[case::groups(u64::MAX, u64::MAX)]
    #[case::sum((u64::MAX - 33) / 9 + 1, 1)]
    fn section_counts_reject_unrepresentable_lengths(
        #[case] sections: u64,
        #[case] size_groups: u64,
    ) {
        assert_eq!(
            SectionInfoCounts::new(sections, size_groups)
                .unwrap()
                .encoded_len(test_support::widths(8, 8))
                .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: 17,
                length: sections
            }
        );
    }

    #[test]
    fn section_counts_measure_the_largest_length_without_materializing_sections() {
        let sections = (u64::MAX - 33) / 9;
        assert_eq!(
            SectionInfoCounts::new(sections, 1)
                .unwrap()
                .encoded_len(test_support::widths(8, 8))
                .unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn section_capacity_preserves_platform_count_limits() {
        let result = section_info_capacity(test_support::widths(8, 8), usize::MAX);
        if usize::BITS == 64 {
            assert_eq!(
                result.unwrap_err(),
                FormatError::OffsetOverflow {
                    offset: 17,
                    length: u64::MAX
                }
            );
        } else {
            assert_eq!(result.unwrap(), 90_194_313_212);
        }
    }

    #[rstest]
    #[case::two(2, 2, 15)]
    #[case::mixed(8, 4, 256)]
    #[case::eight(8, 8, 256)]
    fn section_capacity_bounds_distinct_sizes_and_count_width_changes(
        #[case] offsets: u8,
        #[case] lengths: u8,
        #[case] count: usize,
    ) {
        let widths = test_support::widths(offsets, lengths);
        let sizes: Vec<_> = (1..=count as u64).collect();
        let capacity = section_info_capacity(widths, count).unwrap();
        assert_eq!(section_info_len(widths, &sizes), capacity);
        let fewer: Vec<_> = sizes.into_iter().take(count - 1).collect();
        assert!(section_info_len(widths, &fewer) <= capacity);
        assert!(section_info_len(widths, &alloc::vec![1; count]) <= capacity);
    }

    #[test]
    fn parses_c_library_single_section() {
        // manager @619: one 1600-byte free section at offset 2848, FSSE @701.
        let fshd = free_space::single_section_header();
        let header = FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_space(), 1600);
        assert_eq!(header.total_sections(), 1);
        assert_eq!(header.addr_space_bits(), 63);
        assert_eq!(header.section_list_addr(), StoredAddress::new(701));
        assert_eq!(header.section_list_used(), 35);

        let fsse = free_space::single_section_info();
        let sections = parse_section_info(
            test_support::widths(8, 8),
            &fsse,
            StoredAddress::new(619),
            &header,
        )
        .unwrap();
        assert_eq!(sections, vec![test_support::section(2848, 1600)]);
    }

    #[test]
    fn parses_c_library_two_sections() {
        // manager @736: 16 bytes @871 and 893 bytes @1155, FSSE @818.
        let fshd = free_space::two_section_header();
        let header = FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_space(), 909);
        assert_eq!(header.total_sections(), 2);
        assert_eq!(header.section_list_addr(), StoredAddress::new(818));
        assert_eq!(header.section_list_used(), 53);

        let fsse = free_space::two_section_info();
        let mut sections = parse_section_info(
            test_support::widths(8, 8),
            &fsse,
            StoredAddress::new(736),
            &header,
        )
        .unwrap();
        sections.sort_by_key(|s| s.addr);
        assert_eq!(
            sections,
            vec![
                test_support::section(871, 16),
                test_support::section(1155, 893),
            ]
        );
        // The section sizes sum to the header's tracked total.
        let total: u64 = sections.iter().map(|s| s.size).sum();
        assert_eq!(header.total_space(), total);
    }

    #[test]
    fn serialize_matches_c_library_single_section() {
        // Byte-for-byte reproduction of the FSHD@619 / FSSE@701 fixtures,
        // including the Jenkins checksums the C library verifies on read.
        let fshd_fixture = free_space::single_section_header();
        let fsse_fixture = free_space::single_section_info();
        assert_eq!(
            free_space::raw_manager_eight(619, 701, &[(2848, 1600)]),
            (fshd_fixture.clone(), fsse_fixture.clone())
        );
        let (fshd, fsse) = serialize_free_space_manager(
            test_support::widths(8, 8),
            &[test_support::section(2848, 1600)],
            StoredAddress::new(619),
            StoredAddress::new(701),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
        assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
        // The header length helper agrees with the produced bytes.
        assert_eq!(
            free_space_manager_header_len(test_support::widths(8, 8)),
            fshd.len() as u64
        );
    }

    #[test]
    fn serialize_matches_c_library_two_sections() {
        // FSHD@736 / FSSE@818: two differently-sized sections (16 @871, 893 @1155)
        // emitted as two ascending size-groups.
        let fshd_fixture = free_space::two_section_header();
        let fsse_fixture = free_space::two_section_info();
        assert_eq!(
            free_space::raw_manager_eight(736, 818, &[(1155, 893), (871, 16)]),
            (fshd_fixture.clone(), fsse_fixture.clone())
        );
        let (fshd, fsse) = serialize_free_space_manager(
            test_support::widths(8, 8),
            &[
                test_support::section(1155, 893),
                test_support::section(871, 16),
            ],
            StoredAddress::new(736),
            StoredAddress::new(818),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
        assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
    }

    #[rstest]
    #[case::two_byte_widths(test_support::widths(2, 2))]
    #[case::four_byte_widths(test_support::widths(4, 4))]
    #[case::mixed_widths(test_support::widths(4, 8))]
    #[case::eight_byte_widths(test_support::widths(8, 8))]
    fn section_list_capacity_is_independent_of_its_parse_length(#[case] widths: FormatWidths) {
        let sections = vec![test_support::section(0x1000, 100)];
        let (mut fshd, fsse) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(0x80),
            StoredAddress::new(0x100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
        assert_eq!(header.section_list_allocated(), fsse.len().to_u64());
        let allocated = header.section_list_allocated() + 16;
        let checksum_start = fshd.len() - CHECKSUM_LEN;
        let allocated_start = checksum_start - usize::from(widths.lengths.get());
        let field = &mut fshd[allocated_start..checksum_start];
        field.copy_from_slice(&allocated.to_le_bytes()[..field.len()]);
        fshd.truncate(checksum_start);
        test_util::checksum::append(&mut fshd);
        let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
        assert_eq!(header.section_list_used(), fsse.len().to_u64());
        assert_eq!(header.section_list_allocated(), fsse.len().to_u64() + 16);
        assert_eq!(
            parse_section_info(widths, &fsse, StoredAddress::new(0x80), &header).unwrap(),
            sections
        );
        assert_eq!(
            parse_section_info(
                widths,
                &fsse[..fsse.len() - 1],
                StoredAddress::new(0x80),
                &header
            )
            .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: fsse.len(),
                available: fsse.len() - 1,
            }
        );
    }

    #[rstest]
    #[case::two_byte_widths(test_support::widths(2, 2), 34)]
    #[case::four_byte_widths(test_support::widths(4, 4), 50)]
    #[case::four_byte_lengths(test_support::widths(8, 4), 54)]
    #[case::four_byte_offsets(test_support::widths(4, 8), 78)]
    #[case::eight_byte_widths(test_support::widths(8, 8), 82)]
    fn a_header_holds_one_offset_and_seven_length_fields(
        #[case] widths: FormatWidths,
        #[case] expected: u64,
    ) {
        assert_eq!(free_space_manager_header_len(widths), expected);
    }

    #[rstest]
    fn a_manager_round_trips_at_the_widths_of_its_file(
        #[values(2, 4, 8)] offset_size: u8,
        #[values(2, 4, 8)] length_size: u8,
        #[values(&[][..], &[100], &[100, 100, 100], &[10, 20, 30], &[16384, 512, 512])]
        sizes: &[u64],
    ) {
        let widths = test_support::widths(offset_size, length_size);
        let sections: Vec<FreeSection> = sizes
            .iter()
            .zip(0u64..)
            .map(|(&size, i)| test_support::section(0x1000 + i * 0x4000, size))
            .collect();

        let size_groups = sizes.iter().collect::<BTreeSet<_>>().len().to_u64();
        let measured = SectionInfoCounts::new(sizes.len().to_u64(), size_groups)
            .unwrap()
            .encoded_len(widths)
            .unwrap();
        let (fshd, fsse) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_LARGE,
        )
        .unwrap();

        assert_eq!(fsse.len().to_u64(), measured);
        let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
        assert_eq!(
            (fshd.len() as u64, fsse.len() as u64),
            (
                free_space_manager_header_len(widths),
                section_info_len(widths, sizes)
            )
        );
        assert_eq!(header.section_list_used(), fsse.len() as u64);
        let mut parsed =
            parse_section_info(widths, &fsse, StoredAddress::new(1000), &header).unwrap();
        parsed.sort_by_key(|s| s.addr);
        assert_eq!(parsed, sections);
    }

    // The manager of `four_byte_lengths_header`, with the serializer's maximum section size and
    // each section size at the length width.
    #[test]
    fn a_mixed_width_header_holds_each_field_at_its_width() {
        let (fshd, fsse) = serialize_free_space_manager(
            test_support::widths(8, 4),
            &[test_support::section(0x0E40, 1600)],
            StoredAddress::new(0x0273),
            StoredAddress::new(0x02A9),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();

        assert_eq!(
            fshd[..fshd.len() - 4],
            [
                0x46, 0x53, 0x48, 0x44, 0x00, 0x01, 0x40, 0x06, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x50, 0x00, 0x78, 0x00,
                0x3F, 0x00, 0xFD, 0xFF, 0xFF, 0xFF, 0xA9, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x1F, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00,
            ]
        );
        assert_eq!(
            fsse[..fsse.len() - 4],
            [
                0x46, 0x53, 0x53, 0x45, 0x00, 0x73, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x40, 0x06, 0x00, 0x00, 0x40, 0x0E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]
        );
    }

    #[rstest]
    #[case::total_space(
        test_support::widths(8, 2),
        vec![test_support::section(0x1000, 0x7000), test_support::section(0x9000, 0x7000), test_support::section(0x11000, 0x7000)],
        StoredAddress::new(0x100),
        FormatError::LengthTooLarge { length: 0x1_5000, length_size: 2 },
    )]
    fn a_length_wider_than_its_field_returns_an_error(
        #[case] widths: FormatWidths,
        #[case] sections: Vec<FreeSection>,
        #[case] fsse_addr: StoredAddress,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            serialize_free_space_manager(
                widths,
                &sections,
                StoredAddress::new(0x80),
                fsse_addr,
                SECTION_CLASS_SIMPLE,
            ),
            Err(expected)
        );
    }

    #[rstest]
    #[case::two_byte_lengths(test_support::widths(8, 2), 0x8000)]
    #[case::four_byte_lengths(test_support::widths(8, 4), 0x8000_0000)]
    #[case::eight_byte_lengths(test_support::widths(8, 8), 0x4000_0000_0000_0000)]
    fn a_section_without_a_bin_returns_an_error(#[case] widths: FormatWidths, #[case] limit: u64) {
        let serialize = |size| {
            serialize_free_space_manager(
                widths,
                &[test_support::section(0x100, size)],
                StoredAddress::new(0x80),
                StoredAddress::new(0xC0),
                SECTION_CLASS_SIMPLE,
            )
        };

        let (fshd, _) = serialize(limit - 1).unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
        assert_eq!(1 << header.max_section_size().ilog2(), limit);
        assert_eq!(
            serialize(limit),
            Err(FormatError::InvalidFreeSpaceSectionSize {
                size: limit,
                maximum: limit - 1,
            })
        );
    }

    #[rstest]
    #[case::two_byte_lengths(test_support::widths(8, 2), 0xFFFD)]
    #[case::four_byte_lengths(test_support::widths(8, 4), 0xFFFF_FFFD)]
    fn the_written_maximum_section_size_is_below_the_truncated_driver_addresses(
        #[case] widths: FormatWidths,
        #[case] expected: u64,
    ) {
        let (fshd, _) = serialize_free_space_manager(
            widths,
            &[test_support::section(0x100, 0x10)],
            StoredAddress::new(0x80),
            StoredAddress::new(0xC0),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();

        assert_eq!(
            FreeSpaceManagerHeader::parse(widths, &fshd)
                .unwrap()
                .max_section_size(),
            expected
        );
    }

    // A 63-bit address space is sec2's, and a 32-bit one the core driver's on a 32-bit host.

    #[rstest]
    #[case::zero(0)]
    #[case::nonzero(0xd0)]
    fn an_oversized_c_library_section_list_parses_its_declared_sections(#[case] unused: u8) {
        let header = FreeSpaceManagerHeader::parse(
            test_support::widths(8, 8),
            &free_space::oversized_section_list_header(),
        )
        .unwrap();

        let mut list = free_space::oversized_section_list();
        list[185] = unused;
        let len = list.len();
        test_util::checksum::restamp(&mut list, 0, len);

        let expected = vec![
            test_support::section(842, 10),
            test_support::section(5375, 10),
            test_support::section(474, 24),
            test_support::section(2444, 27),
            test_support::section(1595, 47),
            test_support::section(130, 74),
            test_support::section(2618, 194),
            test_support::section(3793, 254),
            test_support::section(1122, 326),
            test_support::section(4385, 393),
        ];
        assert_eq!(
            parse_section_info(
                test_support::widths(8, 8),
                &list,
                StoredAddress::new(48),
                &header
            ),
            Ok(expected.clone())
        );
        let mut visited = Vec::new();
        visit_section_info(
            test_support::widths(8, 8),
            &list,
            StoredAddress::new(48),
            &header,
            |section| visited.push(section),
        )
        .unwrap();
        assert_eq!(visited, expected);
    }

    #[rstest]
    #[case::byte_184(184)]
    #[case::byte_185(185)]
    #[case::byte_186(186)]
    #[case::byte_187(187)]
    #[case::byte_188(188)]
    #[case::byte_189(189)]
    #[case::byte_190(190)]
    #[case::byte_191(191)]
    #[case::byte_192(192)]
    fn each_unused_section_list_byte_accepts_every_checksummed_value(#[case] at: usize) {
        let widths = test_support::widths(8, 8);
        let header =
            FreeSpaceManagerHeader::parse(widths, &free_space::oversized_section_list_header())
                .unwrap();
        let expected = [
            test_support::section(842, 10),
            test_support::section(5375, 10),
            test_support::section(474, 24),
            test_support::section(2444, 27),
            test_support::section(1595, 47),
            test_support::section(130, 74),
            test_support::section(2618, 194),
            test_support::section(3793, 254),
            test_support::section(1122, 326),
            test_support::section(4385, 393),
        ];
        for value in 0..=u8::MAX {
            let mut list = free_space::oversized_section_list();
            list[at] = value;
            let len = list.len();
            test_util::checksum::restamp(&mut list, 0, len);

            assert_eq!(
                parse_section_info(widths, &list, StoredAddress::new(48), &header).as_deref(),
                Ok(expected.as_slice()),
                "byte {value:#04x} at offset {at}",
            );
        }
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn unused_section_list_bytes_remain_covered_by_the_checksum() {
        let widths = test_support::widths(8, 8);
        let header =
            FreeSpaceManagerHeader::parse(widths, &free_space::oversized_section_list_header())
                .unwrap();
        let mut list = free_space::oversized_section_list();
        list[185] = 0xd0;
        let at = list.len() - 4;
        let expected = u32::from_le_bytes(list[at..].try_into().unwrap());
        let computed = checksum::jenkins_lookup3(&list[..at]);
        assert_ne!(expected, computed);

        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(48), &header),
            Err(FormatError::ChecksumMismatch { expected, computed }),
        );
    }

    #[rstest]
    #[case::a_section_list_of_another_header(
        5,
        &736u64.to_le_bytes(),
        FormatError::InvalidFreeSpaceManager(
            "the section list belongs to the header at 736, not to the header at 619".into()
        ),
        &[]
    )]
    #[case::a_set_of_no_sections(
        13,
        &[0],
        FormatError::InvalidFreeSpaceSectionGroupCount { offset: 13, count: 0, remaining: 1 },
        &[]
    )]
    #[case::a_set_of_empty_sections(
        14,
        &[0; 8],
        FormatError::InvalidFreeSpaceSectionGroupSize { offset: 13 },
        &[]
    )]
    #[case::a_class_the_file_client_does_not_register(
        30,
        &[3],
        FormatError::InvalidFreeSpaceSectionClass { offset: 22, class_id: 3 },
        &[]
    )]
    #[case::sections_that_sum_past_the_total_space(
        14,
        &1601u64.to_le_bytes(),
        FormatError::InvalidFreeSpaceSectionSizeSum { expected: 1600, actual: Box::new(1601) },
        &[test_support::section(2848, 1601)]
    )]
    fn an_invalid_section_list_fails_to_parse(
        #[case] at: usize,
        #[case] value: &[u8],
        #[case] expected: FormatError,
        #[case] expected_visits: &[FreeSection],
    ) {
        let header = FreeSpaceManagerHeader::parse(
            test_support::widths(8, 8),
            &free_space::single_section_header(),
        )
        .unwrap();
        let mut fsse = free_space::single_section_info();
        test_util::bytes::set_slice_at(&mut fsse, at, value);
        let len = fsse.len();
        test_util::checksum::restamp(&mut fsse, 0, len);

        assert_eq!(
            parse_section_info(
                test_support::widths(8, 8),
                &fsse,
                StoredAddress::new(619),
                &header
            ),
            Err(expected.clone())
        );
        let mut visited = Vec::new();
        let error = visit_section_info(
            test_support::widths(8, 8),
            &fsse,
            StoredAddress::new(619),
            &header,
            |section| visited.push(section),
        )
        .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(visited, expected_visits);
    }

    #[test]
    fn a_section_list_shorter_than_its_used_length_fails_to_parse() {
        let list = free_space::single_section_info();
        let header = FreeSpaceManagerHeader::parse(
            test_support::widths(8, 8),
            &free_space::single_section_header(),
        )
        .unwrap();

        assert_eq!(
            parse_section_info(
                test_support::widths(8, 8),
                &list[..list.len() - 1],
                StoredAddress::new(619),
                &header
            ),
            Err(FormatError::UnexpectedEof {
                expected: 35,
                available: 34
            })
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn the_checksum_at_the_end_of_the_used_length_is_verified() {
        // The error reports the flipped checksum, and not the zeros appended past the used length.
        let mut list = free_space::oversized_section_list();
        let header = FreeSpaceManagerHeader::parse(
            test_support::widths(8, 8),
            &free_space::oversized_section_list_header(),
        )
        .unwrap();
        let stored = list.len() - 4;
        list[stored] ^= 0xFF;
        list.extend_from_slice(&[0; 8]);

        assert_eq!(
            parse_section_info(
                test_support::widths(8, 8),
                &list,
                StoredAddress::new(48),
                &header
            ),
            Err(FormatError::ChecksumMismatch {
                expected: 0xAE5C_5CC0,
                computed: 0xAE5C_5C3F,
            })
        );
    }

    #[rstest]
    #[case::default_allocation(None, 0)]
    #[case::exact_allocation(Some(0), 0)]
    #[case::spare_allocation(Some(9), 9)]
    fn serialization_records_the_requested_allocation(
        #[case] extra: Option<u64>,
        #[case] expected_extra: u64,
    ) {
        let widths = test_support::widths(8, 8);
        let sections = [FreeSection {
            addr: StoredAddress::new(4096),
            size: 512,
        }];
        let (_, prefix) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let used = prefix.len() as u64;
        let (header, list) = serialize_free_space_manager_with_allocation(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
            extra.map(|extra| used + extra),
        )
        .unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
        assert_eq!(header.section_list_used(), used + expected_extra);
        assert_eq!(header.section_list_allocated(), used + expected_extra);
        assert_eq!(list.len() as u64, used + expected_extra);
        let records_end = prefix.len() - CHECKSUM_LEN;
        assert_eq!(&list[..records_end], &prefix[..records_end]);
        assert!(
            list[records_end..list.len() - CHECKSUM_LEN]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(1000), &header).unwrap(),
            sections
        );
    }

    #[test]
    fn serialization_rejects_an_allocation_shorter_than_the_section_list() {
        let widths = test_support::widths(8, 8);
        let sections = [FreeSection {
            addr: StoredAddress::new(4096),
            size: 512,
        }];
        let (_, list) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let used = list.len() as u64;
        let allocated = used - 1;
        assert_eq!(
            serialize_free_space_manager_with_allocation(
                widths,
                &sections,
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                SECTION_CLASS_SIMPLE,
                Some(allocated),
            )
            .unwrap_err(),
            FormatError::FreeSpaceSectionListAllocationTooSmall {
                required: used,
                allocated,
            }
        );
    }

    #[rstest]
    #[case::two_byte_lengths(2, 1 << 16)]
    #[case::four_byte_lengths(4, 1 << 32)]
    fn serialization_rejects_an_allocation_outside_the_length_width(
        #[case] length_size: u8,
        #[case] allocated: u64,
    ) {
        assert_eq!(
            serialize_free_space_manager_with_allocation(
                test_support::widths(8, length_size),
                &[FreeSection {
                    addr: StoredAddress::new(4096),
                    size: 512,
                }],
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                SECTION_CLASS_SIMPLE,
                Some(allocated),
            )
            .unwrap_err(),
            FormatError::LengthTooLarge {
                length: allocated,
                length_size,
            }
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn a_header_whose_checksum_does_not_match_fails_to_parse() {
        let mut fshd = free_space::single_section_header();
        let stored = fshd.len() - 4;
        fshd[stored] ^= 0xFF;

        assert_eq!(
            FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd),
            Err(FormatError::ChecksumMismatch {
                expected: 0x1037_1315,
                computed: 0x1037_13EA,
            })
        );
    }

    #[test]
    fn enc_size_matches_reference() {
        assert_eq!(enc_size(0), 1);
        assert_eq!(enc_size(255), 1);
        assert_eq!(enc_size(256), 2);
        assert_eq!(enc_size((1 << 63) - 1), 8);
    }

    #[test]
    fn rejects_bad_signature() {
        let mut fshd = free_space::single_section_header();
        let header = FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd).unwrap();
        let mut fsse = free_space::single_section_info();
        fshd[0] = 0x80;
        fsse[0] = b'X';

        assert_eq!(
            FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd),
            Err(FormatError::InvalidFreeSpaceManager(
                r#"the header signature is b"\x80SHD", not b"FSHD""#.to_owned()
            ))
        );
        assert_eq!(
            parse_section_info(
                test_support::widths(8, 8),
                &fsse,
                StoredAddress::new(619),
                &header
            ),
            Err(FormatError::InvalidFreeSpaceManager(
                r#"the section list signature is b"XSSE", not b"FSSE""#.to_owned()
            ))
        );
    }

    #[rstest]
    #[case::section_list_address(
        test_support::widths(4, 4),
        vec![test_support::section(0x1000, 0x100)],
        StoredAddress::new(0x1_0000_0000),
        FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 },
    )]
    #[case::section_address(
        test_support::widths(2, 4),
        vec![test_support::section(0x1_0000, 0x100)],
        StoredAddress::new(0x100),
        FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 },
    )]
    fn an_address_wider_than_its_field_returns_an_error(
        #[case] widths: FormatWidths,
        #[case] sections: Vec<FreeSection>,
        #[case] fsse_addr: StoredAddress,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            serialize_free_space_manager(
                widths,
                &sections,
                StoredAddress::new(0x80),
                fsse_addr,
                SECTION_CLASS_SIMPLE,
            ),
            Err(expected)
        );
    }

    #[test]
    fn serializing_sections_reports_the_addition_that_overflows_the_total() {
        let sections: Vec<FreeSection> = (0..5)
            .map(|i| test_support::section(0x1000 * (i + 1), (1 << 62) - 1))
            .collect();

        assert_eq!(
            serialize_free_space_manager(
                test_support::widths(8, 8),
                &sections,
                StoredAddress::new(0x80),
                StoredAddress::new(0x100),
                SECTION_CLASS_SIMPLE,
            ),
            Err(FormatError::FreeSpaceSectionSizeOverflow {
                total: u64::MAX - 3,
                size: (1 << 62) - 1,
            })
        );
    }
}
