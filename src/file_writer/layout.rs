use alloc::vec::Vec;

use hdf5_pure_format::__private::FreeSection;

use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;

use super::WIDTHS;

/// Reserves section-list capacity for the final metadata page tail.
///
/// `metadata_end` is a stored address before manager placement. Includes the lengths in
/// `other_managers` when testing page alignment. Returns `None` when the metadata ends on a
/// page boundary without this manager and `sections` is empty. When adding the manager would
/// fill a page exactly with no other metadata sections, reserves one extra padding byte so
/// the manager has a page tail to record.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if a manager extent or its padding exceeds [`u64::MAX`].
pub(super) fn paged_metadata_section_allocation(
    page: FileSpacePageSize,
    metadata_end: u64,
    other_managers: impl IntoIterator<Item = u64>,
    sections: &[FreeSection],
) -> Result<Option<u64>, FormatError> {
    let end = other_managers
        .into_iter()
        .try_fold(metadata_end, |offset, length| {
            offset
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow { offset, length })
        })?;
    if sections.is_empty() && end % page.get() == 0 {
        return Ok(None);
    }
    let sizes: Vec<_> = sections
        .iter()
        .map(|section| section.size)
        .chain(core::iter::once(0))
        .collect();
    let allocation = hdf5_pure_format::__private::section_info_len(WIDTHS, &sizes);
    let end = [
        hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS),
        allocation,
    ]
    .into_iter()
    .try_fold(end, |offset, length| {
        offset
            .checked_add(length)
            .ok_or(FormatError::OffsetOverflow { offset, length })
    })?;
    Ok(Some(if sections.is_empty() && end % page.get() == 0 {
        allocation
            .checked_add(1)
            .ok_or(FormatError::OffsetOverflow {
                offset: allocation,
                length: 1,
            })?
    } else {
        allocation
    }))
}
