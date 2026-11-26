use alloc::collections::BTreeSet;
use core::num::NonZeroU64;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::{self, FormatWidths, LengthWidth, OffsetWidth};

use super::address_space::AddressSpaceBits;
use super::file_section::FreeSection;
use super::header::{self, CHECKSUM_LEN};
use super::section_list;

/// Returns the length in bytes of the header and the section list of a file free-space manager
/// that holds `sections`, or 0 if `sections` is empty.
///
/// The zero length assumes the caller omits an empty manager. For a nonempty manager, the
/// length depends only on `widths`, the number of sections, and the number of distinct sizes.
pub fn free_space_manager_len(widths: FormatWidths, sections: &[FreeSection]) -> u64 {
    if sections.is_empty() {
        return 0;
    }
    let sets = sections
        .iter()
        .map(|section| section.size)
        .collect::<BTreeSet<u64>>()
        .len();
    header::free_space_manager_header_len(widths) + section_list_len(widths, sets, sections.len())
}

/// Returns the length in bytes of the section list of a manager that holds sections of
/// `section_sizes`.
///
/// The length depends on `widths`, the number of sections, and the number of distinct sizes, and
/// not on the addresses of the sections, and is the length of the section list
/// [`super::serialize_free_space_manager`] serializes for sections of these sizes.
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
    /// Uses the field widths of [`super::serialize_free_space_manager`] and includes the section-list
    /// prefix and checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the encoded length exceeds [`u64::MAX`].
    pub fn encoded_len(self, widths: FormatWidths) -> Result<u64, FormatError> {
        Ok(self.layout(widths)?.1)
    }

    /// Returns the field widths and the unpadded section-list length in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the encoded length exceeds [`u64::MAX`].
    pub(super) fn layout(self, widths: FormatWidths) -> Result<(SectionFields, u64), FormatError> {
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
            .map(|len| (fields, len))
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

/// The widths in bytes of the variable-size fields of a section list: the number of sections in a
/// set, the section size of a set, and the offset of a section.
#[derive(Clone, Copy, Debug)]
pub(super) struct SectionFields {
    count: usize,
    size: usize,
    offset: usize,
}

impl SectionFields {
    pub(super) fn from_header(
        sections: u64,
        max_section_size: u64,
        address_space: AddressSpaceBits,
        lengths: LengthWidth,
    ) -> Self {
        let offset = address_space.offset_width();
        Self {
            count: enc_size(sections),
            size: Self::section_size_width(max_section_size, lengths, offset),
            offset,
        }
    }

    /// Returns the section size field width in bytes for a list with `offset`-byte offsets.
    ///
    /// The width is the fewest bytes that hold the maximum section size, as "Free-space Index" of
    /// the [format specification, version 4.0][spec] defines it, and `offset` below 8-byte lengths
    /// where the maximum section size is all ones or all ones less one. libhdf5 sizes the section
    /// size fields from the maximum address of its file driver (`H5FS__sinfo_new` in
    /// `H5FSsection.c`), sets the "Size of Address Space" to the number of bits of that address
    /// (`H5MF__create_fstype` in `H5MF.c`), and writes that address into the header truncated to
    /// the length width (`H5FS__cache_hdr_serialize` in `H5FScache.c`), HDF5 1.10.11 to 2.2.0.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
    fn section_size_width(max_section_size: u64, lengths: LengthWidth, offset: usize) -> usize {
        if lengths != LengthWidth::Eight && max_section_size >= width::length_width_max(lengths) - 1
        {
            offset
        } else {
            enc_size(max_section_size)
        }
    }

    /// Returns the widths of the fields of the section list [`super::serialize_free_space_manager`]
    /// writes at `widths` for a manager of `sections` sections.
    fn written(widths: FormatWidths, sections: u64) -> Self {
        Self {
            count: enc_size(sections),
            size: enc_size(written_max_section_size(widths.lengths).get()),
            offset: AddressSpaceBits::written(widths.offsets).offset_width(),
        }
    }

    pub(super) fn count(self) -> usize {
        self.count
    }

    pub(super) fn size(self) -> usize {
        self.size
    }

    pub(super) fn offset(self) -> usize {
        self.offset
    }

    /// Returns the combined count and section-size field width of one size group, in bytes.
    pub(super) fn set_len(self) -> usize {
        self.count + self.size
    }

    /// Returns the combined address and class field width of one section record, in bytes.
    pub(super) fn section_len(self) -> usize {
        self.offset + SECTION_CLASS_LEN
    }

    /// Returns the most sections a section list of `list_len` bytes holds at these widths, in a
    /// file with addresses of `offsets`: the sections of one set between the prefix and the
    /// checksum.
    pub(super) fn max_sections(self, offsets: OffsetWidth, list_len: u64) -> u64 {
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
}

/// Returns the fewest bytes that hold `value`, and at least 1, as `H5VM_limit_enc_size` computes
/// it.
fn enc_size(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    bits.div_ceil(8).max(1)
}

/// Returns the maximum section size the serializer writes at `lengths`.
///
/// libhdf5 checks the bin of a section against the number of bins in an `assert` alone
/// (`H5FS__sect_link_size` in `H5FSsection.c`, HDF5 1.10.11 to 2.2.0). Below 8-byte lengths the
/// maximum is all ones less two, which has the top bit of the field set, so libhdf5 allocates as
/// many bins for it as for the largest value of the field (`H5FS__sinfo_new`). It is below the
/// truncated addresses [`SectionFields::section_size_width`] reads as libhdf5's.
pub(super) fn written_max_section_size(lengths: LengthWidth) -> NonZeroU64 {
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

/// Returns the length in bytes of the section list [`super::serialize_free_space_manager`] writes at
/// `widths` for `sections` sections of `sets` distinct sizes.
fn section_list_len(widths: FormatWidths, sets: usize, sections: usize) -> u64 {
    let fields = SectionFields::written(widths, sections.to_u64());
    (section_list::section_list_prefix_len(widths.offsets)
        + sets * fields.set_len()
        + sections * fields.section_len()
        + CHECKSUM_LEN)
        .to_u64()
}

/// The maximum section size libhdf5 gives a file free-space manager under its default driver, sec2,
/// whatever the file's widths: the driver's maximum address, `2^63 - 1` (`H5MF__create_fstype` in
/// `H5MF.c`, HDF5 1.10.11 to 2.2.0), which HDF5 2.2.0 defines as `H5FD_MAXADDR`. The family, multi
/// and MPI-IO drivers give `HADDR_MAX`, `2^64 - 2`, and the core driver `SIZE_MAX - 1`.
pub(super) const FILE_FSM_MAX_SECTION_SIZE: u64 = (1u64 << 63) - 1;

const SECTION_CLASS_LEN: usize = 1;

#[cfg(test)]
#[path = "tests/geometry.rs"]
mod tests;
