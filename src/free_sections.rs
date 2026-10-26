//! Checked free sections and reports of persisted manager contents or session reuse.

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroU64;
use core::ops::Range;

use hdf5_pure_space::__private::ManagerKind;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;
use crate::file_space_info::FileSpaceInfo;
use crate::free_space_manager;
use crate::source::Source;
use crate::width::FormatWidths;

/// A snapshot of a writable session's logical size and reusable free bytes.
///
/// [`File::reusable_space_stats`](crate::File::reusable_space_stats) reads both values under
/// the session lock. They include committed changes and immediate appends. Staged edits affect
/// these values when applied by a commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReusableSpaceStats {
    logical_size: u64,
    reusable_free_bytes: u64,
}

impl ReusableSpaceStats {
    pub(crate) fn new(logical_size: u64, reusable_free_bytes: u64) -> Self {
        Self {
            logical_size,
            reusable_free_bytes,
        }
    }

    /// Returns the session's file-image length in bytes at the time of the snapshot.
    ///
    /// Includes the userblock prefix and any trailing bytes. This equals
    /// [`File::file_size`](crate::File::file_size) for the same session state, and can decrease
    /// when a commit truncates free space at the end of the file.
    ///
    /// Opening a file preserves its recorded end-of-file address. Unaccounted trailing bytes
    /// can therefore make this length exceed
    /// [`Superblock::eof_address`](crate::Superblock::eof_address).
    pub const fn logical_size(&self) -> u64 {
        self.logical_size
    }

    /// Returns the free bytes tracked as reusable when the snapshot was taken.
    ///
    /// Includes tracked regions freed by committed edits and regions recovered from persisted
    /// managers on open. A fresh nonpersistent session does not recover holes from earlier
    /// sessions. Under [`FileSpaceStrategy::None`](crate::FileSpaceStrategy::None), the total is
    /// zero.
    /// On paged files, includes the reusable metadata and raw-data lists and excludes dead or
    /// unclassified space.
    ///
    /// Under [`FileSpaceStrategy::Aggr`](crate::FileSpaceStrategy::Aggr) and
    /// [`FileSpaceStrategy::FsmAggr`](crate::FileSpaceStrategy::FsmAggr), includes unused metadata
    /// and raw aggregate reservations for staged edits and immediate appends, even when the
    /// spare is shorter than the free-section threshold.
    ///
    /// Under [`FileSpaceStrategy::FsmAggr`](crate::FileSpaceStrategy::FsmAggr), the editor admits
    /// abandoned interior spare under the threshold and adjacency rules for freed extents. A
    /// shorter region is tracked only when it adjoins tracked free space. Under
    /// [`FileSpaceStrategy::Aggr`](crate::FileSpaceStrategy::Aggr), the editor discards abandoned
    /// interior spare.
    ///
    /// The total does not guarantee that a particular write fits or predict its growth. A tracked
    /// range can be truncated away when a commit releases free space at the end of the file.
    ///
    /// # Reservations
    ///
    /// Includes unused immediate-append reserves. The editor removes these reserves from the
    /// persisted managers before an append writes into them. A staged commit places its objects
    /// outside the append reserve, then returns unused reserve space when rewriting the managers.
    /// The remaining free space is available to subsequent commits. Clean teardown of a persistent
    /// session also returns unused reservations through a manager rewrite.
    ///
    /// The editor excludes held reservations from persisted managers, so this total can exceed
    /// the persisted section-byte total reported through
    /// [`File::persisted_free_sections`](crate::File::persisted_free_sections).
    pub const fn reusable_free_bytes(&self) -> u64 {
        self.reusable_free_bytes
    }
}

/// The free sections recorded in a file's persisted managers.
///
/// [`File::persisted_free_sections`](crate::File::persisted_free_sections) returns this report
/// when persistence is configured, including when every manager is empty or undefined. Section
/// and manager addresses are relative to the HDF5 base address. The combined sections contain
/// every section from the individual managers, including those with an unknown role.
/// [`manager_metadata_bytes`](Self::manager_metadata_bytes) measures the file storage allocated
/// to the managers themselves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedFreeSections {
    managers: Vec<FreeSpaceManagerSections>,
    free_sections: FreeSections,
    manager_metadata_bytes: u64,
}

impl PersistedFreeSections {
    /// Reads each defined manager slot through the random-access source.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if multiple slots record the same
    /// manager header address, manager allocations overlap, a section has zero length, or a
    /// section or manager allocation exceeds the file allocation. Returns
    /// [`FormatError::OffsetOverflow`] if an address, section end, or byte total overflows.
    /// Other failures are the errors of [`free_space_manager::read_persisted_sections_source`].
    pub(crate) fn read(
        source: &impl Source,
        widths: FormatWidths,
        base: BaseAddress,
        stored_eof: u64,
        info: &FileSpaceInfo,
    ) -> Result<Self, FormatError> {
        let mut managers = Vec::new();
        let mut all_sections = Vec::new();
        let mut allocations = Vec::<Range<u64>>::new();
        for (slot, &address) in info.manager_addrs.iter().enumerate() {
            let DecodedStoredAddress::Defined(address) = address else {
                continue;
            };
            // Allocation requests share a manager by mapping to its slot
            // (`H5MF__alloc_to_fs_type` in `H5MF.c`, HDF5 2.2.0).
            if let Some(previous) = info
                .manager_addrs
                .iter()
                .take(slot)
                .position(|&other| other == DecodedStoredAddress::Defined(address))
            {
                return Err(FormatError::InvalidFreeSpaceManager(format!(
                    "the free-space manager at stored address {} is recorded in slots {previous} and {slot}",
                    address.get()
                )));
            }
            let (sections, blocks) = free_space_manager::read_persisted_sections_source(
                source,
                widths,
                base,
                &[DecodedStoredAddress::Defined(address)],
            )?;
            let physical_eof = base.absolute(StoredAddress::new(stored_eof))?;
            let mut metadata_bytes = 0u64;
            for (address, length) in blocks {
                let end = address
                    .checked_add(length)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: address,
                        length,
                    })?;
                if end > physical_eof {
                    return Err(FormatError::InvalidFreeSpaceManager(format!(
                        "the manager allocation at {address}, {length} bytes long, exceeds the file allocation"
                    )));
                }
                if let Some(previous) = allocations
                    .iter()
                    .find(|previous| address < previous.end && previous.start < end)
                {
                    return Err(FormatError::InvalidFreeSpaceManager(format!(
                        "the manager allocation at {address}, {length} bytes long, overlaps the \
                         manager allocation at {}, {} bytes long",
                        previous.start,
                        previous.end - previous.start,
                    )));
                }
                allocations.push(address..end);
                metadata_bytes =
                    metadata_bytes
                        .checked_add(length)
                        .ok_or(FormatError::OffsetOverflow {
                            offset: metadata_bytes,
                            length,
                        })?;
            }
            let sections = sections
                .into_iter()
                .map(|section| {
                    let checked = FreeSection::new(
                        section.addr.get(),
                        section.size,
                        FreeSectionCoordinates::Stored,
                    )?;
                    let end = base.absolute(StoredAddress::new(checked.end_address()))?;
                    if checked.end_address() > stored_eof || end > source.len() {
                        return Err(FormatError::InvalidFreeSpaceManager(format!(
                            "the free section at {}, {} bytes long, exceeds the file allocation",
                            checked.address(),
                            checked.length()
                        )));
                    }
                    Ok(checked)
                })
                .collect::<Result<Vec<_>, FormatError>>()?;
            all_sections.extend_from_slice(&sections);
            let role = match free_space_manager::manager_kind(info.strategy, slot) {
                Some(ManagerKind::Flat) if slot < NONPAGED_MANAGER_SLOTS => {
                    FreeSpaceManagerRole::FsmAggr
                }
                Some(ManagerKind::Flat) => FreeSpaceManagerRole::Unknown,
                Some(ManagerKind::PagedSmallMetadata) => FreeSpaceManagerRole::PageSmallMetadata,
                Some(ManagerKind::PagedSmallRaw) => FreeSpaceManagerRole::PageSmallRawData,
                Some(ManagerKind::PagedLargeGeneric) => FreeSpaceManagerRole::PageLargeGeneric,
                None => FreeSpaceManagerRole::Unknown,
            };
            managers.push(FreeSpaceManagerSections {
                slot,
                address,
                role,
                free_sections: FreeSections::new(sections)?,
                metadata_bytes,
            });
        }
        let manager_metadata_bytes = managers.iter().try_fold(0u64, |sum, manager| {
            sum.checked_add(manager.metadata_bytes())
                .ok_or(FormatError::OffsetOverflow {
                    offset: sum,
                    length: manager.metadata_bytes(),
                })
        })?;
        Ok(Self {
            managers,
            free_sections: FreeSections::new(all_sections)?,
            manager_metadata_bytes,
        })
    }

    /// Returns the defined managers in ascending slot order.
    ///
    /// Includes defined managers with zero sections and omits undefined manager slots.
    pub fn managers(&self) -> &[FreeSpaceManagerSections] {
        &self.managers
    }

    /// Returns the combined sections of every defined manager.
    ///
    /// The count and byte total equal the sums of the per-manager counts and byte totals.
    pub const fn free_sections(&self) -> &FreeSections {
        &self.free_sections
    }

    /// Returns the file bytes allocated to persisted manager headers and section lists.
    ///
    /// This is the checked sum of [`FreeSpaceManagerSections::metadata_bytes`] for every
    /// defined manager, including managers with an unknown role. It includes unused section-list
    /// capacity and is zero when every manager slot is undefined.
    ///
    /// Excludes the free sections themselves, the File Space Info message and its object header,
    /// userblocks, dataset storage and other metadata, aggregate or append reservations, dead
    /// or unclassified storage, and trailing file slack. It measures persisted manager storage only.
    ///
    /// libhdf5's `H5Fget_info2` reports free-space-manager metadata for an open handle.
    /// `H5FS_size` can use an in-memory section-list size (`H5FS.c`, HDF5 1.14.6), so that
    /// statistic can differ from this allocation count.
    pub const fn manager_metadata_bytes(&self) -> u64 {
        self.manager_metadata_bytes
    }
}

/// The sections recorded in one defined File Space Info manager slot.
///
/// The slot and header address identify the manager recorded in the file. Its
/// [`role`](Self::role) describes the recognized manager category, which can differ from an
/// allocation-request type passed to the C library's `H5Fget_free_sections`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FreeSpaceManagerSections {
    slot: usize,
    address: StoredAddress,
    role: FreeSpaceManagerRole,
    free_sections: FreeSections,
    metadata_bytes: u64,
}

impl FreeSpaceManagerSections {
    /// Returns the zero-based index in the File Space Info manager-address array.
    ///
    /// Version 0 of the message stores six slots, and version 1 stores twelve, as defined in
    /// "The File Space Info Message" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo
    pub const fn slot(&self) -> usize {
        self.slot
    }

    /// Returns the manager header address relative to the HDF5 base address.
    pub const fn address(&self) -> StoredAddress {
        self.address
    }

    /// Returns the manager category recognized from the file strategy and recorded slot.
    pub const fn role(&self) -> FreeSpaceManagerRole {
        self.role
    }

    /// Returns this manager's sections in stored, base-relative coordinates.
    pub const fn free_sections(&self) -> &FreeSections {
        &self.free_sections
    }

    /// Returns the file bytes allocated to this manager's header and section list.
    ///
    /// Includes the width-dependent `FSHD` header size and the allocated `FSSE` section-list
    /// size, including unused capacity. A manager without a section-list allocation occupies
    /// its header size. The reader checks allocation bounds, overlaps, and arithmetic before
    /// constructing the report.
    ///
    /// The version 0 header and section list are defined in "Free-space Index" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
    pub const fn metadata_bytes(&self) -> u64 {
        self.metadata_bytes
    }
}

/// A manager category recognized from the file strategy and recorded manager slot.
///
/// These categories describe persisted managers. Several allocation-request types can share
/// a manager, so a category does not identify every allocation type whose space it contains.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreeSpaceManagerRole {
    /// A manager in one of the six slots used without pages under [`FileSpaceStrategy::FsmAggr`].
    ///
    /// [`FileSpaceStrategy::FsmAggr`]: crate::FileSpaceStrategy::FsmAggr
    FsmAggr,
    /// The generic manager for large sections under
    /// [`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page).
    ///
    /// This manager can contain space from both metadata and raw-data allocations.
    PageLargeGeneric,
    /// The manager for small metadata sections under
    /// [`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page).
    PageSmallMetadata,
    /// The manager for small raw-data sections under
    /// [`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page).
    PageSmallRawData,
    /// A slot whose role is unrecognized for the recorded strategy.
    ///
    /// The report retains the slot, header address, and sections.
    Unknown,
}

/// Checked free sections in ascending address order, with their count and total length.
///
/// [`File::reusable_free_sections`](crate::File::reusable_free_sections) returns absolute
/// physical offsets for an editing session. [`PersistedFreeSections`] contains base-relative
/// sections read from storage. Each section records its [`coordinates`](FreeSection::coordinates).
/// The byte total is the checked sum of the section lengths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FreeSections {
    sections: Vec<FreeSection>,
    free_bytes: u64,
}

impl FreeSections {
    /// Sorts sections by address and computes their total length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum of section lengths exceeds [`u64::MAX`].
    pub(crate) fn new(mut sections: Vec<FreeSection>) -> Result<Self, FormatError> {
        sections.sort_unstable_by_key(FreeSection::address);
        let free_bytes = sections.iter().try_fold(0u64, |sum, section| {
            sum.checked_add(section.length().get())
                .ok_or(FormatError::OffsetOverflow {
                    offset: sum,
                    length: section.length().get(),
                })
        })?;
        Ok(Self {
            sections,
            free_bytes,
        })
    }

    /// Returns the sections in ascending address order.
    pub fn sections(&self) -> &[FreeSection] {
        &self.sections
    }

    /// Returns the number of sections, including sections in managers with an unknown role.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Returns the sum of all section lengths in bytes.
    pub const fn free_bytes(&self) -> u64 {
        self.free_bytes
    }
}

/// A free byte range with a positive length and a representable exclusive end address.
///
/// The reader checks the range before returning it. [`coordinates`](Self::coordinates)
/// determines how to interpret the start and end addresses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreeSection {
    address: u64,
    length: NonZeroU64,
    coordinates: FreeSectionCoordinates,
}

impl FreeSection {
    /// Constructs a section with a positive length and a checked end address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if `length` is zero, or
    /// [`FormatError::OffsetOverflow`] if the end address exceeds [`u64::MAX`].
    pub(crate) fn new(
        address: u64,
        length: u64,
        coordinates: FreeSectionCoordinates,
    ) -> Result<Self, FormatError> {
        let length = NonZeroU64::new(length).ok_or_else(|| {
            FormatError::InvalidFreeSpaceManager("a free section has zero length".into())
        })?;
        address
            .checked_add(length.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: address,
                length: length.get(),
            })?;
        Ok(Self {
            address,
            length,
            coordinates,
        })
    }

    /// Returns the start address in this section's [`coordinates`](Self::coordinates).
    pub const fn address(&self) -> u64 {
        self.address
    }

    /// Returns the positive length in bytes.
    pub const fn length(&self) -> NonZeroU64 {
        self.length
    }

    /// Returns the exclusive end address in this section's [`coordinates`](Self::coordinates).
    ///
    /// This is the start address plus the length, checked when the section is constructed.
    pub const fn end_address(&self) -> u64 {
        self.address + self.length.get()
    }

    /// Returns the coordinate system of the start and end addresses.
    pub const fn coordinates(&self) -> FreeSectionCoordinates {
        self.coordinates
    }
}

/// The coordinate system of a free section's start and end addresses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreeSectionCoordinates {
    /// Physical byte offsets from the beginning of the opened file or source.
    ///
    /// Session-reusable sections use these coordinates. Offsets count the bytes in any
    /// userblock prefix.
    Absolute,
    /// HDF5 addresses relative to the base address.
    ///
    /// Persisted sections use these coordinates. For a relocated source image, the reader
    /// resolves stored addresses against the effective base of that image.
    Stored,
}

/// The number of manager slots used without paged aggregation.
///
/// Both message versions store these slots first, as defined in "The File Space Info Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo
const NONPAGED_MANAGER_SLOTS: usize = 6;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hdf5_pure_core::__private::FileSpaceInfoFields;
    use hdf5_pure_format::__private as format;
    use hdf5_pure_format::__private::FreeSection as StoredSection;
    use rstest::rstest;

    use super::*;
    use crate::file_space_info::FileSpacePageSize;
    use crate::file_space_info::FileSpaceStrategy;
    use crate::source::BytesSource;

    #[test]
    fn section_ends_and_aggregate_sums_are_checked() {
        let zero = FreeSection::new(10, 0, FreeSectionCoordinates::Stored).unwrap_err();
        assert_eq!(
            zero,
            FormatError::InvalidFreeSpaceManager("a free section has zero length".into())
        );
        assert!(matches!(
            FreeSection::new(u64::MAX, 1, FreeSectionCoordinates::Stored).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        ));
        let sections = vec![
            FreeSection::new(0, u64::MAX, FreeSectionCoordinates::Stored).unwrap(),
            FreeSection::new(0, 1, FreeSectionCoordinates::Stored).unwrap(),
        ];
        assert!(matches!(
            FreeSections::new(sections).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        ));
        let section = FreeSection::new(10, 20, FreeSectionCoordinates::Absolute).unwrap();
        assert_eq!(
            (
                section.address(),
                section.length().get(),
                section.end_address()
            ),
            (10, 20, 30)
        );
    }

    #[rstest]
    #[case::flat(FileSpaceStrategy::FsmAggr, 0, FreeSpaceManagerRole::FsmAggr)]
    #[case::flat_other_slot(FileSpaceStrategy::FsmAggr, 5, FreeSpaceManagerRole::FsmAggr)]
    #[case::unsupported_flat_slot(FileSpaceStrategy::FsmAggr, 6, FreeSpaceManagerRole::Unknown)]
    #[case::small_metadata(FileSpaceStrategy::Page, 0, FreeSpaceManagerRole::PageSmallMetadata)]
    #[case::small_raw(FileSpaceStrategy::Page, 2, FreeSpaceManagerRole::PageSmallRawData)]
    #[case::large_mixed(FileSpaceStrategy::Page, 6, FreeSpaceManagerRole::PageLargeGeneric)]
    #[case::unknown(FileSpaceStrategy::Page, 7, FreeSpaceManagerRole::Unknown)]
    fn manager_attribution_preserves_the_recorded_slot(
        #[case] strategy: FileSpaceStrategy,
        #[case] slot: usize,
        #[case] role: FreeSpaceManagerRole,
        #[values(0, 512)] base: u64,
        #[values(2, 4, 8)] offset_width: u8,
        #[values(2, 4, 8)] length_width: u8,
    ) {
        let widths = FormatWidths::from_sizes(offset_width, length_width).unwrap();
        let (bytes, mut info) = manager_file(widths, base, 256, 32);
        info.strategy = strategy;
        info.manager_addrs.swap(0, slot);
        let report = PersistedFreeSections::read(
            &BytesSource::new(&bytes),
            widths,
            BaseAddress::new(base),
            1024,
            &info,
        )
        .unwrap();
        assert_eq!(report.managers().len(), 1);
        let manager = &report.managers()[0];
        assert_eq!(
            (manager.slot(), manager.address().get(), manager.role()),
            (slot, 64, role)
        );
        let header_bytes = 4
            + 1
            + 1
            + 4 * u64::from(length_width)
            + 4 * 2
            + u64::from(length_width)
            + u64::from(offset_width)
            + 2 * u64::from(length_width)
            + 4;
        let list_bytes = 4
            + 1
            + u64::from(offset_width)
            + 1
            + u64::from(length_width)
            + u64::from(offset_width)
            + 1
            + 4;
        assert_eq!(manager.metadata_bytes(), header_bytes + list_bytes);
        assert_eq!(report.manager_metadata_bytes(), header_bytes + list_bytes);
        assert_eq!(manager.free_sections(), report.free_sections());
        assert_eq!(report.free_sections().section_count(), 1);
        assert_eq!(report.free_sections().free_bytes(), 32);
        assert_eq!(
            report.free_sections().sections(),
            [FreeSection::new(256, 32, FreeSectionCoordinates::Stored).unwrap()]
        );
    }

    #[rstest]
    #[case::past_allocation(1000, 32)]
    #[case::overflow(u64::MAX - 15, 32)]
    fn persisted_sections_reject_invalid_ends(#[case] address: u64, #[case] length: u64) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (bytes, info) = manager_file(widths, 0, address, length);
        let error = PersistedFreeSections::read(
            &BytesSource::new(&bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap_err();
        if address == 1000 {
            let FormatError::InvalidFreeSpaceManager(reason) = error else {
                panic!("{error:?}");
            };
            assert_eq!(
                reason,
                "the free section at 1000, 32 bytes long, exceeds the file allocation"
            );
        } else {
            assert!(
                matches!(error, FormatError::OffsetOverflow { offset, length: 32 } if offset == u64::MAX - 15),
                "{error:?}"
            );
        }
    }

    #[rstest]
    #[case::header(70)]
    #[case::section_allocation(160)]
    fn truncated_manager_structures_return_errors(#[case] length: usize) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, info) = manager_file(widths, 0, 256, 32);
        bytes.truncate(length);
        let error = PersistedFreeSections::read(
            &BytesSource::new(&bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap_err();
        if length == 70 {
            assert!(
                matches!(
                    error,
                    FormatError::UnexpectedSourceEof {
                        expected: 146,
                        available: 70
                    }
                ),
                "{error:?}"
            );
        } else {
            assert_eq!(error, FormatError::InvalidFreeSpaceManager(
                "the section list allocation at 146, 35 bytes long, runs past the end of the file, 160 bytes long".into()
            ));
        }
    }

    #[rstest]
    #[case::used_length(false)]
    #[case::list_address(true)]
    fn malformed_manager_fields_return_the_decode_error(#[case] address: bool) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, info) = manager_file(widths, 0, 256, 32);
        let header_length = usize::try_from(format::free_space_manager_header_len(widths)).unwrap();
        let header_end = 64 + header_length;
        let length_bytes = usize::from(widths.lengths.get());
        let used_at = header_end - size_of::<u32>() - 2 * length_bytes;
        let used = u64::from_le_bytes(bytes[used_at..used_at + length_bytes].try_into().unwrap());
        let expected = if address {
            let offset_at = used_at - usize::from(widths.offsets.get());
            bytes[offset_at..offset_at + usize::from(widths.offsets.get())]
                .copy_from_slice(&(u64::MAX - 1).to_le_bytes());
            FormatError::OffsetOverflow {
                offset: u64::MAX - 1,
                length: used,
            }
        } else {
            bytes[used_at..used_at + length_bytes].copy_from_slice(&(used + 1).to_le_bytes());
            FormatError::InvalidFreeSpaceManager(format!(
                "the section list uses {} bytes, more than the {used} bytes allocated to it",
                used + 1
            ))
        };
        crate::checksum::stamp_trailing(&mut bytes, 64, header_length);
        assert_eq!(
            PersistedFreeSections::read(
                &BytesSource::new(bytes),
                widths,
                BaseAddress::ZERO,
                1024,
                &info,
            )
            .unwrap_err(),
            expected
        );
    }

    #[rstest]
    fn an_empty_manager_remains_present_in_the_breakdown(
        #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, mut info) = manager_file(widths, 0, 256, 32);
        info.strategy = strategy;
        let header_address = StoredAddress::new(64);
        let list_address = StoredAddress::new(64 + format::free_space_manager_header_len(widths));
        let (header, list) = format::serialize_free_space_manager(
            widths,
            &[],
            header_address,
            list_address,
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        bytes[64..64 + header.len()].copy_from_slice(&header);
        bytes[64 + header.len()..64 + header.len() + list.len()].copy_from_slice(&list);
        let report = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap();
        assert_eq!(report.managers().len(), 1);
        assert_eq!(report.managers()[0].slot(), 0);
        assert_eq!(report.managers()[0].free_sections().section_count(), 0);
        assert_eq!(report.managers()[0].free_sections().free_bytes(), 0);
        assert_eq!(report.managers()[0].free_sections(), report.free_sections());
        assert_eq!(report.managers()[0].metadata_bytes(), 82 + 17);
        assert_eq!(report.manager_metadata_bytes(), 82 + 17);
    }

    #[rstest]
    fn a_header_without_a_section_list_counts_its_allocation(
        #[values(2, 4, 8)] offset_width: u8,
        #[values(2, 4, 8)] length_width: u8,
    ) {
        let widths = FormatWidths::from_sizes(offset_width, length_width).unwrap();
        let (mut header, _) = format::serialize_free_space_manager(
            widths,
            &[],
            StoredAddress::new(64),
            StoredAddress::undefined(widths.offsets),
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let length_bytes = usize::from(length_width);
        let used_at = header.len() - 4 - 2 * length_bytes;
        header[used_at..used_at + 2 * length_bytes].fill(0);
        let header_len = header.len();
        crate::checksum::stamp_trailing(&mut header, 0, header_len);
        let (mut bytes, info) = manager_file(widths, 0, 256, 32);
        bytes[64..64 + header_len].copy_from_slice(&header);
        let report = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap();
        let expected = 18 + u64::from(offset_width) + 7 * u64::from(length_width);
        assert_eq!(report.managers()[0].metadata_bytes(), expected);
        assert_eq!(report.manager_metadata_bytes(), expected);
        assert_eq!(report.free_sections().free_bytes(), 0);
    }

    #[rstest]
    fn section_list_capacity_counts_beyond_the_used_bytes(
        #[values(2, 4, 8)] offset_width: u8,
        #[values(2, 4, 8)] length_width: u8,
    ) {
        let widths = FormatWidths::from_sizes(offset_width, length_width).unwrap();
        let (mut bytes, info) = manager_file(widths, 0, 256, 32);
        let header_len = usize::try_from(format::free_space_manager_header_len(widths)).unwrap();
        let length_bytes = usize::from(length_width);
        let allocated_at = 64 + header_len - 4 - length_bytes;
        let mut encoded = [0u8; 8];
        encoded[..length_bytes].copy_from_slice(&bytes[allocated_at..allocated_at + length_bytes]);
        let used = u64::from_le_bytes(encoded);
        let allocated = used + 47;
        bytes[allocated_at..allocated_at + length_bytes]
            .copy_from_slice(&allocated.to_le_bytes()[..length_bytes]);
        crate::checksum::stamp_trailing(&mut bytes, 64, header_len);
        let report = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap();
        assert_eq!(
            report.managers()[0].metadata_bytes(),
            u64::try_from(header_len).unwrap() + allocated
        );
        assert_eq!(
            report.manager_metadata_bytes(),
            u64::try_from(header_len).unwrap() + allocated
        );
        assert_eq!(report.free_sections().free_bytes(), 32);
    }

    #[rstest]
    #[case::overlaps_header(
        400,
        Some(
            "the manager allocation at 512, 82 bytes long, overlaps the manager allocation at 146, 400 bytes long"
        )
    )]
    #[case::overlaps_section_list(
        548,
        Some(
            "the manager allocation at 0, 548 bytes long, overlaps the manager allocation at 64, 82 bytes long"
        )
    )]
    #[case::adjacent(366, None)]
    fn manager_allocations_reject_overlaps(#[case] capacity: u64, #[case] reason: Option<&str>) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, mut info) = manager_file(widths, 0, 256, 32);
        let (header, list) = format::serialize_free_space_manager(
            widths,
            &[StoredSection {
                addr: StoredAddress::new(320),
                size: 16,
            }],
            StoredAddress::new(512),
            StoredAddress::new(594),
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        bytes[512..594].copy_from_slice(&header);
        bytes[594..594 + list.len()].copy_from_slice(&list);
        info.manager_addrs[1] = DecodedStoredAddress::Defined(StoredAddress::new(512));
        if capacity == 548 {
            bytes.copy_within(146..181, 0);
            bytes[118..126].copy_from_slice(&0u64.to_le_bytes());
        }
        bytes[134..142].copy_from_slice(&capacity.to_le_bytes());
        crate::checksum::stamp_trailing(&mut bytes, 64, 82);
        let result = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        );
        if let Some(reason) = reason {
            assert_eq!(
                result.unwrap_err(),
                FormatError::InvalidFreeSpaceManager(reason.into())
            );
        } else {
            let report = result.unwrap();
            assert_eq!(report.manager_metadata_bytes(), 82 + capacity + 82 + 35);
        }
    }

    #[test]
    fn manager_allocations_cannot_extend_into_trailing_source_bytes() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (bytes, info) = manager_file(widths, 0, 0, 1);
        let error = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            100,
            &info,
        )
        .unwrap_err();
        let FormatError::InvalidFreeSpaceManager(reason) = error else {
            panic!("{error:?}");
        };
        assert_eq!(
            reason,
            "the manager allocation at 64, 82 bytes long, exceeds the file allocation"
        );
    }

    #[rstest]
    #[case::known_slot(
        1,
        "the free-space manager at stored address 64 is recorded in slots 0 and 1"
    )]
    #[case::unknown_slot(
        7,
        "the free-space manager at stored address 64 is recorded in slots 0 and 7"
    )]
    fn repeated_manager_addresses_are_rejected(#[case] slot: usize, #[case] reason: &str) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (bytes, mut info) = manager_file(widths, 0, 256, 32);
        info.manager_addrs[slot] = info.manager_addrs[0];
        assert_eq!(
            PersistedFreeSections::read(
                &BytesSource::new(bytes),
                widths,
                BaseAddress::ZERO,
                1_024,
                &info,
            )
            .unwrap_err(),
            FormatError::InvalidFreeSpaceManager(reason.into()),
        );
    }

    #[rstest]
    #[case::flat(FileSpaceStrategy::FsmAggr, 1, FreeSpaceManagerRole::FsmAggr)]
    #[case::page_raw(FileSpaceStrategy::Page, 2, FreeSpaceManagerRole::PageSmallRawData)]
    #[case::page_large(FileSpaceStrategy::Page, 6, FreeSpaceManagerRole::PageLargeGeneric)]
    #[case::unknown(FileSpaceStrategy::Page, 7, FreeSpaceManagerRole::Unknown)]
    fn distinct_managers_combine_their_sections(
        #[case] strategy: FileSpaceStrategy,
        #[case] slot: usize,
        #[case] role: FreeSpaceManagerRole,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, mut info) = manager_file(widths, 0, 256, 32);
        let header_address = StoredAddress::new(512);
        let (header, list) = format::serialize_free_space_manager(
            widths,
            &[StoredSection {
                addr: StoredAddress::new(320),
                size: 16,
            }],
            header_address,
            StoredAddress::new(512 + format::free_space_manager_header_len(widths)),
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        bytes[512..512 + header.len()].copy_from_slice(&header);
        bytes[512 + header.len()..512 + header.len() + list.len()].copy_from_slice(&list);
        info.strategy = strategy;
        info.manager_addrs[slot] = DecodedStoredAddress::Defined(header_address);
        let report = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            1_024,
            &info,
        )
        .unwrap();
        let first = FreeSection::new(256, 32, FreeSectionCoordinates::Stored).unwrap();
        let second = FreeSection::new(320, 16, FreeSectionCoordinates::Stored).unwrap();
        assert_eq!(report.free_sections().sections(), [first, second]);
        assert_eq!(report.free_sections().section_count(), 2);
        assert_eq!(report.free_sections().free_bytes(), 48);
        assert_eq!(report.managers().len(), 2);
        assert_eq!(report.managers()[0].metadata_bytes(), 117);
        assert_eq!(report.managers()[1].metadata_bytes(), 117);
        assert_eq!(report.manager_metadata_bytes(), 234);
        assert_eq!(report.managers()[1].role(), role);
        assert_eq!(report.managers()[0].free_sections().sections(), [first]);
        assert_eq!(report.managers()[1].free_sections().sections(), [second]);
        assert_eq!(
            (
                report.managers()[0].slot(),
                report.managers()[0].address().get()
            ),
            (0, 64)
        );
        assert_eq!(
            (
                report.managers()[1].slot(),
                report.managers()[1].address().get()
            ),
            (slot, 512)
        );
    }

    fn manager_file(
        widths: FormatWidths,
        base: u64,
        address: u64,
        length: u64,
    ) -> (Vec<u8>, FileSpaceInfo) {
        let header_address = StoredAddress::new(64);
        let list_address = StoredAddress::new(64 + format::free_space_manager_header_len(widths));
        let (header, list) = format::serialize_free_space_manager(
            widths,
            &[StoredSection {
                addr: StoredAddress::new(address),
                size: length,
            }],
            header_address,
            list_address,
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let mut bytes = vec![0; usize::try_from(base + 1024).unwrap()];
        let at = usize::try_from(base + 64).unwrap();
        bytes[at..at + header.len()].copy_from_slice(&header);
        bytes[at + header.len()..at + header.len() + list.len()].copy_from_slice(&list);
        let mut manager_addrs =
            vec![DecodedStoredAddress::Undefined; format::NUM_FILE_FSM_MANAGERS];
        manager_addrs[0] = DecodedStoredAddress::Defined(header_address);
        let info = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::FsmAggr,
            persist: true,
            threshold: 1,
            page_size: FileSpacePageSize::default(),
            page_end_meta_threshold: 0,
            eoa_pre_fsm: None,
            manager_addrs,
        }
        .build();
        (bytes, info)
    }
}
