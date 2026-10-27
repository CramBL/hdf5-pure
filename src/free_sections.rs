//! Checked free sections and reports of persisted manager contents or session reuse.

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroU64;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_format::__private::{self as format, FreeSpaceManagerHeaderRef};
use hdf5_pure_space::__private::ManagerKind;

use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
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
    /// Reads every defined manager and checks the combined sections and metadata allocations.
    ///
    /// Verifies header and section-list checksums regardless of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`FreeSectionsError::Overlap`] if free sections overlap within or across managers.
    /// Returns [`FreeSectionsError::Format`] for malformed or unsupported metadata, source
    /// read failures, duplicate manager addresses, overlapping metadata allocations, sections
    /// beyond the logical end of file, or arithmetic overflow.
    pub(crate) fn read(
        source: &impl Source,
        widths: FormatWidths,
        base: BaseAddress,
        stored_eof: u64,
        info: &FileSpaceInfo,
    ) -> Result<Self, FreeSectionsError> {
        Ok(PersistedManagerSet::read(source, widths, base, stored_eof, info)?.into_report())
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

/// A complete parsed manager set with checked sections and metadata allocations.
///
/// Free sections are disjoint, and metadata allocations are checked separately for overlaps.
/// Sections have positive lengths and checked ends within the logical end of file. Metadata
/// allocations include unused section-list capacity and fit within both the logical end of
/// file and the source. Report addresses are base-relative, and allocation ranges are physical.
#[derive(Debug)]
pub(crate) struct PersistedManagerSet {
    report: PersistedFreeSections,
    allocations: Vec<FileRange>,
}

impl PersistedManagerSet {
    /// Reads all defined slots before returning the completed manager set.
    ///
    /// Verifies both checksums, decodes file-client records, and checks their combined geometry.
    /// Callers choose error conversions through the source implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if a manager cannot be read or parsed, manager addresses repeat,
    /// sections or metadata allocations overlap, or an extent exceeds the logical end of file
    /// or the available source. An address, end, or byte total above [`u64::MAX`] is also an error.
    pub(crate) fn read<S: PersistedManagerSource>(
        source: &S,
        widths: FormatWidths,
        base: BaseAddress,
        stored_eof: u64,
        info: &FileSpaceInfo,
    ) -> Result<Self, S::Error> {
        let addresses = AddressContext::from_effective_base(base);
        let logical_eof = addresses.resolve(StoredAddress::new(stored_eof))?;
        let reader = PersistedManagerReader {
            source,
            widths,
            addresses,
            stored_eof,
            logical_eof: logical_eof.get(),
        };
        let mut report = PersistedFreeSections {
            managers: Vec::new(),
            free_sections: FreeSections::new(Vec::new())?,
            manager_metadata_bytes: 0,
        };
        let mut allocations = Vec::new();
        let mut tagged = Vec::new();
        for (slot, &address) in info.manager_addrs.iter().enumerate() {
            let DecodedStoredAddress::Defined(address) = address else {
                continue;
            };
            let mut context = PersistedManagerContext {
                slot,
                header_address: address,
                section_list_address: None,
                section: None,
            };
            if let Some(previous) = info
                .manager_addrs
                .iter()
                .take(slot)
                .position(|&other| other == DecodedStoredAddress::Defined(address))
            {
                return Err(S::contextual_error(context, FormatError::InvalidFreeSpaceManager(format!(
                    "the free-space manager at stored address {} is recorded in slots {previous} and {slot}",
                    address.get()
                )).into()));
            }
            let manager = reader
                .read(&mut context, &mut allocations, &mut tagged, info)
                .map_err(|error| S::contextual_error(context, error))?;
            report.manager_metadata_bytes = report
                .manager_metadata_bytes
                .checked_add(manager.metadata_bytes)
                .ok_or(FormatError::OffsetOverflow {
                    offset: report.manager_metadata_bytes,
                    length: manager.metadata_bytes,
                })?;
            report.managers.push(manager);
        }
        tagged.sort_unstable_by_key(|&(section, _)| section.address());
        for pair in tagged.windows(2) {
            let [(previous, _), (section, context)] = pair else {
                continue;
            };
            if section.address() < previous.end_address() {
                return Err(S::contextual_error(
                    *context,
                    FreeSectionsError::Overlap {
                        address: section.address(),
                        length: section.length().get(),
                        previous_address: previous.address(),
                        previous_length: previous.length().get(),
                    }
                    .into(),
                ));
            }
        }
        report.free_sections =
            FreeSections::new(tagged.into_iter().map(|(section, _)| section).collect())?;
        Ok(Self {
            report,
            allocations,
        })
    }

    /// Returns the base-relative report and complete physical manager allocations.
    pub(crate) fn into_parts(self) -> (PersistedFreeSections, Vec<FileRange>) {
        let Self {
            report,
            allocations,
        } = self;
        (report, allocations)
    }

    fn into_report(self) -> PersistedFreeSections {
        let Self {
            report,
            allocations,
        } = self;
        drop(allocations);
        report
    }

    fn admit_allocation(
        range: FileRange,
        logical_eof: u64,
        allocations: &mut Vec<FileRange>,
    ) -> Result<(), FormatError> {
        let address = range.start().get();
        let length = range.length();
        if range.end().get() > logical_eof {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the manager allocation at {address}, {length} bytes long, exceeds the file allocation"
            )));
        }
        if let Some(previous) = allocations
            .iter()
            .find(|previous| range.start() < previous.end() && previous.start() < range.end())
        {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the manager allocation at {address}, {length} bytes long, overlaps the manager allocation at {}, {} bytes long",
                previous.start().get(),
                previous.length()
            )));
        }
        allocations.push(range);
        Ok(())
    }
}

struct PersistedManagerReader<'a, S> {
    source: &'a S,
    widths: FormatWidths,
    addresses: AddressContext,
    stored_eof: u64,
    logical_eof: u64,
}

impl<S: PersistedManagerSource> PersistedManagerReader<'_, S> {
    fn read(
        &self,
        context: &mut PersistedManagerContext,
        allocations: &mut Vec<FileRange>,
        tagged: &mut Vec<(FreeSection, PersistedManagerContext)>,
        info: &FileSpaceInfo,
    ) -> Result<FreeSpaceManagerSections, S::Error> {
        let Self {
            source,
            widths,
            addresses,
            stored_eof,
            logical_eof,
        } = *self;
        let slot = context.slot;
        let address = context.header_address;
        let header_ref = FreeSpaceManagerHeaderRef::new(widths, addresses.resolve(address)?)?;
        PersistedManagerSet::admit_allocation(header_ref.range(), logical_eof, allocations)?;
        let bytes = source.read(header_ref.parse_range())?;
        let frame = header_ref.parse_checked(FileRegion::new(header_ref.parse_range(), &bytes)?)?;
        context.section_list_address = match frame.section_list_addr() {
            DecodedStoredAddress::Defined(address) => Some(address),
            DecodedStoredAddress::Undefined => None,
        };
        let header = frame.file_header()?;
        let mut metadata_bytes = header_ref.range().length();
        let sections = if let Some(list_address) = context.section_list_address {
            let list_ref = frame.section_list_ref(address, addresses.resolve(list_address)?)?;
            PersistedManagerSet::admit_allocation(list_ref.range(), logical_eof, allocations)?;
            if list_ref.range().end().get() > source.len() {
                return Err(FormatError::UnexpectedSourceEof {
                    expected: list_ref.range().end().get(),
                    available: source.len(),
                }
                .into());
            }
            let bytes = source.read(list_ref.parse_range())?;
            list_ref.parse_checked(FileRegion::new(list_ref.parse_range(), &bytes)?)?;
            metadata_bytes = metadata_bytes
                .checked_add(list_ref.range().length())
                .ok_or(FormatError::OffsetOverflow {
                    offset: metadata_bytes,
                    length: list_ref.range().length(),
                })?;
            format::parse_section_info(widths, &bytes, address, &header)?
        } else {
            Vec::new()
        };
        let sections = sections
            .into_iter()
            .map(|section| {
                context.section = Some((section.addr, section.size));
                let checked = FreeSection::new(
                    section.addr.get(),
                    section.size,
                    FreeSectionCoordinates::Stored,
                )?;
                if checked.end_address() > stored_eof {
                    return Err(FormatError::InvalidFreeSpaceManager(format!(
                        "the free section at {}, {} bytes long, exceeds the file allocation",
                        checked.address(),
                        checked.length()
                    ))
                    .into());
                }
                let end = addresses.resolve(StoredAddress::new(checked.end_address()))?;
                if end.get() > source.len() {
                    return Err(FormatError::UnexpectedSourceEof {
                        expected: end.get(),
                        available: source.len(),
                    }
                    .into());
                }
                tagged.push((checked, *context));
                Ok(checked)
            })
            .collect::<Result<Vec<_>, S::Error>>()?;
        let sections = FreeSections::new(sections).map_err(|error| {
            if let FreeSectionsError::Overlap {
                address, length, ..
            } = error
            {
                context.section = Some((StoredAddress::new(address), length));
            }
            error
        })?;
        context.section = None;
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
        Ok(FreeSpaceManagerSections {
            slot,
            address,
            role,
            free_sections: sections,
            metadata_bytes,
        })
    }
}

/// The decoded manager identity and section associated with a parsing failure.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PersistedManagerContext {
    pub(crate) slot: usize,
    pub(crate) header_address: StoredAddress,
    pub(crate) section_list_address: Option<StoredAddress>,
    pub(crate) section: Option<(StoredAddress, u64)>,
}

/// A source of physical manager bytes with an error type chosen by the caller.
///
/// Implementations can retain acquisition causes and attach decoded manager context.
pub(crate) trait PersistedManagerSource {
    type Error: From<FormatError> + From<FreeSectionsError>;

    fn len(&self) -> u64;
    /// Reads all bytes in `range`.
    ///
    /// # Errors
    ///
    /// Returns an acquisition error if the range is unavailable or the source read fails.
    fn read(&self, range: FileRange) -> Result<Vec<u8>, Self::Error>;
    /// Attaches available manager context to the acquisition or parsing failure.
    fn contextual_error(context: PersistedManagerContext, error: Self::Error) -> Self::Error;
}

impl<S: Source> PersistedManagerSource for S {
    type Error = FreeSectionsError;

    fn len(&self) -> u64 {
        Source::len(self)
    }

    fn read(&self, range: FileRange) -> Result<Vec<u8>, Self::Error> {
        Ok(self.read_exact_at(range.start().get(), range.length().to_usize()?)?)
    }

    fn contextual_error(_context: PersistedManagerContext, error: Self::Error) -> Self::Error {
        error
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
/// Sections are disjoint, and adjacent sections are permitted. The byte total is the checked
/// sum of the section lengths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FreeSections {
    sections: Vec<FreeSection>,
    free_bytes: u64,
}

impl FreeSections {
    /// Sorts disjoint sections by address and computes their total length.
    ///
    /// # Errors
    ///
    /// Returns [`FreeSectionsError::Overlap`] if two sections overlap, including duplicates.
    /// Returns [`FreeSectionsError::Format`] with [`FormatError::OffsetOverflow`] if the sum
    /// of section lengths exceeds [`u64::MAX`].
    pub(crate) fn new(mut sections: Vec<FreeSection>) -> Result<Self, FreeSectionsError> {
        sections.sort_unstable_by_key(FreeSection::address);
        let free_bytes = sections.iter().try_fold(0u64, |sum, section| {
            sum.checked_add(section.length().get())
                .ok_or(FormatError::OffsetOverflow {
                    offset: sum,
                    length: section.length().get(),
                })
        })?;
        for pair in sections.windows(2) {
            let [previous, section] = pair else { continue };
            if section.address() < previous.end_address() {
                return Err(FreeSectionsError::Overlap {
                    address: section.address(),
                    length: section.length().get(),
                    previous_address: previous.address(),
                    previous_length: previous.length().get(),
                });
            }
        }
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

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum FreeSectionsError {
    Format(FormatError),
    Overlap {
        address: u64,
        length: u64,
        previous_address: u64,
        previous_length: u64,
    },
}

impl From<FormatError> for FreeSectionsError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
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
    use crate::address::BaseAddressExt;
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
            FreeSectionsError::Format(FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            })
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
            let FreeSectionsError::Format(FormatError::InvalidFreeSpaceManager(reason)) = error
            else {
                panic!("{error:?}");
            };
            assert_eq!(
                reason,
                "the free section at 1000, 32 bytes long, exceeds the file allocation"
            );
        } else {
            assert!(
                matches!(error, FreeSectionsError::Format(FormatError::OffsetOverflow { offset, length: 32 }) if offset == u64::MAX - 15),
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
                    FreeSectionsError::Format(FormatError::UnexpectedSourceEof {
                        expected: 146,
                        available: 70
                    })
                ),
                "{error:?}"
            );
        } else {
            assert_eq!(
                error,
                FreeSectionsError::Format(FormatError::UnexpectedSourceEof {
                    expected: 181,
                    available: 160
                })
            );
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
            FreeSectionsError::Format(expected)
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
                FreeSectionsError::Format(FormatError::InvalidFreeSpaceManager(reason.into()))
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
        let FreeSectionsError::Format(FormatError::InvalidFreeSpaceManager(reason)) = error else {
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
            FreeSectionsError::Format(FormatError::InvalidFreeSpaceManager(reason.into())),
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

    #[rstest]
    #[case::header(64)]
    #[case::section_list(146)]
    fn checked_manager_parsing_requires_checksums(#[case] address: usize) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, info) = manager_file(widths, 0, 256, 32);
        let checksum_at = if address == 64 { 142 } else { 177 };
        bytes[checksum_at] ^= 1;
        let error = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap_err();
        let FreeSectionsError::Format(FormatError::ChecksumMismatch { expected, computed }) = error
        else {
            panic!("{error:?}");
        };
        assert_eq!(expected ^ computed, 1);
    }

    #[rstest]
    #[case::overlap(272, false)]
    #[case::duplicate(256, false)]
    #[case::adjacent(288, true)]
    fn free_sections_reject_overlaps_within_and_across_managers(
        #[case] address: u64,
        #[case] valid: bool,
        #[values(false, true)] separate: bool,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, mut info) = manager_file(widths, 0, 256, 32);
        let sections = if separate {
            vec![StoredSection {
                addr: StoredAddress::new(address),
                size: 32,
            }]
        } else {
            vec![
                StoredSection {
                    addr: StoredAddress::new(256),
                    size: 32,
                },
                StoredSection {
                    addr: StoredAddress::new(address),
                    size: 32,
                },
            ]
        };
        let header_address = if separate { 512 } else { 64 };
        let list_address = header_address + format::free_space_manager_header_len(widths);
        let (header, list) = format::serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(header_address),
            StoredAddress::new(list_address),
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let at = usize::try_from(header_address).unwrap();
        let list_at = usize::try_from(list_address).unwrap();
        bytes[at..at + header.len()].copy_from_slice(&header);
        bytes[list_at..list_at + list.len()].copy_from_slice(&list);
        if separate {
            info.manager_addrs[7] =
                DecodedStoredAddress::Defined(StoredAddress::new(header_address));
        }
        let result = PersistedFreeSections::read(
            &BytesSource::new(bytes),
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        );
        if valid {
            let report = result.unwrap();
            assert_eq!(report.free_sections().free_bytes(), 64);
            assert_eq!(report.free_sections().section_count(), 2);
        } else {
            assert_eq!(
                result.unwrap_err(),
                FreeSectionsError::Overlap {
                    address,
                    length: 32,
                    previous_address: 256,
                    previous_length: 32,
                }
            );
        }
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::header(64, false)]
    #[case::section_list(146, false)]
    #[case::section_extent(0, true)]
    fn acquisition_preserves_causes_and_manager_context(
        #[case] fail_at: u64,
        #[case] invalid_extent: bool,
    ) {
        use std::error::Error as StdError;
        use std::io;

        use crate::error::Error;

        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (bytes, info) = manager_file(widths, 0, if invalid_extent { 1000 } else { 256 }, 32);
        let error = PersistedManagerSet::read(
            &Acquisition { bytes, fail_at },
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap_err();
        let Error::PersistedFreeSpace {
            manager_slot,
            header_address,
            section_list_address,
            section,
            source,
        } = &error
        else {
            panic!("{error:?}");
        };
        assert_eq!((*manager_slot, header_address.get()), (0, 64));
        assert_eq!(
            *section_list_address,
            (fail_at != 64).then_some(StoredAddress::new(146))
        );
        if invalid_extent {
            assert_eq!(*section, Some((StoredAddress::new(1000), 32)));
            let Error::Format(FormatError::InvalidFreeSpaceManager(reason)) = &**source else {
                panic!("{source:?}");
            };
            assert_eq!(
                reason,
                "the free section at 1000, 32 bytes long, exceeds the file allocation"
            );
        } else {
            assert_eq!(*section, None);
            let cause = error
                .source()
                .unwrap()
                .source()
                .unwrap()
                .downcast_ref::<io::Error>()
                .unwrap();
            assert_eq!(cause.kind(), io::ErrorKind::PermissionDenied);
            assert_eq!(cause.to_string(), "manager read denied");
        }
    }

    #[cfg(feature = "std")]
    struct Acquisition {
        bytes: Vec<u8>,
        fail_at: u64,
    }
    #[cfg(feature = "std")]
    impl PersistedManagerSource for Acquisition {
        type Error = crate::error::Error;
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }
        fn read(&self, range: FileRange) -> Result<Vec<u8>, Self::Error> {
            if range.start().get() == self.fail_at {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "manager read denied",
                )
                .into());
            }
            Ok(BytesSource::new(&self.bytes)
                .read_exact_at(range.start().get(), range.length().to_usize()?)?)
        }
        fn contextual_error(context: PersistedManagerContext, error: Self::Error) -> Self::Error {
            crate::error::Error::PersistedFreeSpace {
                manager_slot: context.slot,
                header_address: context.header_address,
                section_list_address: context.section_list_address,
                section: context.section,
                source: Box::new(error),
            }
        }
    }

    #[rstest]
    #[case::zero_size(
        160,
        0,
        "the set at byte 13 of the section list holds sections of 0 bytes"
    )]
    #[case::invalid_class(
        176,
        3,
        "the class of the section at byte 22 of the section list is 3, not one of the 3 classes of the file client"
    )]
    fn framed_lists_require_file_client_records(
        #[case] at: usize,
        #[case] value: u8,
        #[case] reason: &str,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, info) = manager_file(widths, 0, 256, 32);
        bytes[at] = value;
        crate::checksum::stamp_trailing(&mut bytes, 146, 35);
        assert_eq!(
            PersistedFreeSections::read(
                &BytesSource::new(bytes),
                widths,
                BaseAddress::ZERO,
                1024,
                &info
            )
            .unwrap_err(),
            FreeSectionsError::Format(FormatError::InvalidFreeSpaceManager(reason.into()))
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn an_invalid_later_manager_prevents_a_completed_set() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (mut bytes, mut info) = manager_file(widths, 0, 256, 32);
        let (header, list) = format::serialize_free_space_manager(
            widths,
            &[StoredSection {
                addr: StoredAddress::new(1000),
                size: 32,
            }],
            StoredAddress::new(512),
            StoredAddress::new(594),
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        bytes[512..594].copy_from_slice(&header);
        bytes[594..594 + list.len()].copy_from_slice(&list);
        info.manager_addrs[7] = DecodedStoredAddress::Defined(StoredAddress::new(512));
        let error = PersistedManagerSet::read(
            &Acquisition {
                bytes,
                fail_at: u64::MAX,
            },
            widths,
            BaseAddress::ZERO,
            1024,
            &info,
        )
        .unwrap_err();
        let crate::error::Error::PersistedFreeSpace {
            manager_slot,
            header_address,
            section_list_address,
            section,
            source,
        } = error
        else {
            panic!("{error:?}");
        };
        assert_eq!(
            (
                manager_slot,
                header_address.get(),
                section_list_address,
                section
            ),
            (
                7,
                512,
                Some(StoredAddress::new(594)),
                Some((StoredAddress::new(1000), 32))
            )
        );
        let crate::error::Error::Format(FormatError::InvalidFreeSpaceManager(reason)) = *source
        else {
            panic!("{source:?}");
        };
        assert_eq!(
            reason,
            "the free section at 1000, 32 bytes long, exceeds the file allocation"
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
