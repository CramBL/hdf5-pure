//! Checked free sections and reports of persisted manager contents or session reuse.

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroU64;

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
/// the session lock. They have the same meaning as the scalar fields of
/// [`SpaceAccounting`](crate::SpaceAccounting), including committed changes and immediate
/// appends. Staged edits affect these values when applied by a commit.
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
    pub const fn logical_size(&self) -> u64 {
        self.logical_size
    }

    /// Returns the free bytes tracked as reusable when the snapshot was taken.
    ///
    /// Includes unused append reserves and classic aggregate reservations. On paged files,
    /// includes the reusable metadata and raw-data lists and excludes dead or unclassified
    /// space. The total can differ from persisted manager contents and does not guarantee
    /// that a particular write fits. See [`SpaceAccounting::reusable_free_bytes`]
    /// for allocation and reservation semantics.
    ///
    /// [`SpaceAccounting::reusable_free_bytes`]: crate::SpaceAccounting::reusable_free_bytes
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedFreeSections {
    managers: Vec<FreeSpaceManagerSections>,
    free_sections: FreeSections,
}

impl PersistedFreeSections {
    /// Reads each defined manager slot through the random-access source.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if multiple slots record the same
    /// manager header address, a section has zero length, or a section or manager allocation
    /// exceeds the file allocation. Returns [`FormatError::OffsetOverflow`] if an address,
    /// section end, or byte total overflows.
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
            });
        }
        Ok(Self {
            managers,
            free_sections: FreeSections::new(all_sections)?,
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

    #[test]
    fn an_empty_manager_remains_present_in_the_breakdown() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, info) = manager_file(widths, 0, 256, 32);
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

    #[test]
    fn distinct_managers_combine_their_sections() {
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
        info.manager_addrs[1] = DecodedStoredAddress::Defined(header_address);
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
            (1, 512)
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
