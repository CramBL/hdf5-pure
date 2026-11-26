use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;
use core::num::NonZeroU64;
use core::slice;

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::free_space_manager::file_header::FILE_FSM_CLIENT_ID;
use crate::free_space_manager::file_section::FILE_FSM_NUM_CLASSES;
use crate::free_space_manager::geometry::{self, SectionFields, SectionInfoCounts};
use crate::free_space_manager::{FreeSection, header};
use crate::width;
use crate::width::{FormatWidths, LengthWidth};

/// The checked sections and block lengths of a file free-space manager before placement.
///
/// Groups section addresses by size for reuse across calls to [`Self::place`]. All sections
/// belong to the same file section class.
#[derive(Debug)]
pub struct PreparedFreeSpaceManager {
    widths: FormatWidths,
    class_id: u8,
    groups: BTreeMap<NonZeroU64, SectionAddresses>,
    section_count: u64,
    fields: SectionFields,
    total_space: u64,
    max_section_size: NonZeroU64,
    section_list_len: usize,
}

impl PreparedFreeSpaceManager {
    /// Checks the section records at the file's widths and groups their addresses by size.
    ///
    /// Pass a file section class, [`SECTION_CLASS_SIMPLE`](crate::free_space_manager::SECTION_CLASS_SIMPLE),
    /// [`SECTION_CLASS_SMALL`](crate::free_space_manager::SECTION_CLASS_SMALL), or
    /// [`SECTION_CLASS_LARGE`](crate::free_space_manager::SECTION_CLASS_LARGE).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedSerializedSectionClass`] for any other class, or
    /// [`FormatError::InvalidFreeSpaceSectionSize`] if a section size is zero or exceeds the
    /// maximum for the length width.
    ///
    /// Returns [`FormatError::AddressTooLarge`] or [`FormatError::DefinedAddressIsUndefined`]
    /// if a section address is outside the offset width or equals its undefined address,
    /// respectively. Returns [`FormatError::FreeSpaceSectionSizeOverflow`] if the total size
    /// exceeds [`u64::MAX`].
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the section-list length exceeds [`u64::MAX`],
    /// [`FormatError::LengthTooLarge`] if the total size or section-list length exceeds the
    /// length width, or [`FormatError::ValueTooLargeForPlatform`] if the section list is too
    /// long for a slice on this platform.
    pub fn new(
        widths: FormatWidths,
        sections: impl IntoIterator<Item = FreeSection>,
        class_id: u8,
    ) -> Result<Self, FormatError> {
        // libhdf5 defines the simple, small, and large file classes in `H5MFpkg.h` and sets
        // their extra serialized size to zero in `H5MFsection.c` (HDF5 1.14.6).
        if u16::from(class_id) >= FILE_FSM_NUM_CLASSES {
            return Err(FormatError::UnsupportedSerializedSectionClass {
                client_id: FILE_FSM_CLIENT_ID,
                class_id,
            });
        }
        let max_section_size = geometry::written_max_section_size(widths.lengths);
        let maximum = (1 << max_section_size.ilog2()) - 1;
        let mut groups = BTreeMap::<NonZeroU64, SectionAddresses>::new();
        let mut total_space = 0u64;
        let mut section_count = 0u64;
        for FreeSection { addr, size } in sections {
            let size = NonZeroU64::new(size)
                .filter(|size| size.get() <= maximum)
                .ok_or(FormatError::InvalidFreeSpaceSectionSize { size, maximum })?;
            bytes::require_defined_offset(addr.get(), widths.offsets)?;
            total_space = total_space.checked_add(size.get()).ok_or(
                FormatError::FreeSpaceSectionSizeOverflow {
                    total: total_space,
                    size: size.get(),
                },
            )?;
            // Every section adds at least one byte to the checked sum.
            section_count += 1;
            match groups.entry(size) {
                Entry::Occupied(mut entry) => entry.get_mut().push(addr),
                Entry::Vacant(entry) => {
                    entry.insert(SectionAddresses::Single(addr));
                }
            }
        }
        require_length(total_space, widths.lengths)?;

        for addresses in groups.values_mut() {
            if let SectionAddresses::Multiple(addresses) = addresses {
                addresses.sort_unstable();
            }
        }
        let (fields, section_list_len) =
            SectionInfoCounts::new(section_count, groups.len().to_u64())?.layout(widths)?;
        let section_list_len = slice_len(section_list_len, widths.lengths)?;
        Ok(Self {
            widths,
            class_id,
            groups,
            section_count,
            fields,
            total_space,
            max_section_size,
            section_list_len,
        })
    }

    /// Returns the sections in ascending order of size, then of address within each size.
    pub fn sections(&self) -> impl Iterator<Item = FreeSection> + '_ {
        self.groups.iter().flat_map(|(&size, addresses)| {
            addresses.as_slice().iter().map(move |&addr| FreeSection {
                addr,
                size: size.get(),
            })
        })
    }

    /// Returns the number of size groups in the section list.
    pub fn distinct_size_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns whether any section has a length of `size` bytes.
    pub fn contains_section_size(&self, size: u64) -> bool {
        NonZeroU64::new(size).is_some_and(|size| self.groups.contains_key(&size))
    }

    /// Returns the number of sections, all of which are serialized.
    pub fn section_count(&self) -> u64 {
        self.section_count
    }

    /// Returns the header length in bytes, including its checksum.
    pub fn header_len(&self) -> usize {
        header::header_len(self.widths)
    }

    /// Returns the section-list length in bytes, including its checksum and excluding padding.
    pub fn section_list_len(&self) -> usize {
        self.section_list_len
    }

    pub(super) fn widths(&self) -> FormatWidths {
        self.widths
    }

    pub(super) fn total_space(&self) -> u64 {
        self.total_space
    }

    pub(super) fn max_section_size(&self) -> NonZeroU64 {
        self.max_section_size
    }

    pub(super) fn fields(&self) -> SectionFields {
        self.fields
    }

    pub(super) fn class_id(&self) -> u8 {
        self.class_id
    }

    pub(super) fn groups(&self) -> impl Iterator<Item = (NonZeroU64, &[StoredAddress])> {
        self.groups
            .iter()
            .map(|(&size, addresses)| (size, addresses.as_slice()))
    }
}

/// The addresses of sections of one size, with inline storage for a single address.
#[derive(Debug)]
enum SectionAddresses {
    Multiple(Vec<StoredAddress>),
    Single(StoredAddress),
}

impl SectionAddresses {
    fn push(&mut self, address: StoredAddress) {
        match self {
            Self::Multiple(addresses) => addresses.push(address),
            Self::Single(first) => *self = Self::Multiple(Vec::from([*first, address])),
        }
    }

    fn as_slice(&self) -> &[StoredAddress] {
        match self {
            Self::Multiple(addresses) => addresses,
            Self::Single(address) => slice::from_ref(address),
        }
    }
}

fn require_length(value: u64, width: LengthWidth) -> Result<(), FormatError> {
    if !width::length_width_holds(width, value) {
        return Err(FormatError::LengthTooLarge {
            length: value,
            length_size: width.get(),
        });
    }
    Ok(())
}

/// Converts `value` to a byte length representable by a slice on this platform.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if `value` does not fit the file's length width, or
/// [`FormatError::ValueTooLargeForPlatform`] if it exceeds [`isize::MAX`].
pub(super) fn slice_len(value: u64, width: LengthWidth) -> Result<usize, FormatError> {
    require_length(value, width)?;
    let len = value.to_usize()?;
    if len > isize::MAX.cast_unsigned() {
        return Err(FormatError::ValueTooLargeForPlatform {
            value,
            target: "slice length",
        });
    }
    Ok(len)
}
