use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroU64;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;

use super::file_header::FreeSpaceManagerHeader;
use super::file_section::{FILE_FSM_NUM_CLASSES, FreeSection};
use super::geometry::SectionFields;
use super::header::CHECKSUM_LEN;
use super::section_list;

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

impl SectionFields {
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
            let (count, size) = cursor.take(self.set_len())?.split_at(self.count());
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
                let class_id = record[self.offset()];
                if u16::from(class_id) >= FILE_FSM_NUM_CLASSES {
                    // Each record starts in the acquired span, within the original input.
                    return Err(FormatError::InvalidFreeSpaceSectionClass {
                        offset: records_start + index * record_len,
                        class_id,
                    });
                }
                visit(FreeSection {
                    addr: StoredAddress::new(read_uint_le(&record[..self.offset()])),
                    size: size.get(),
                });
            }
            remaining -= count;
        }
        Ok(())
    }
}

/// Reads `bytes` as a little-endian unsigned integer.
fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

#[cfg(test)]
#[path = "tests/file_records.rs"]
mod tests;
