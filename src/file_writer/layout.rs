//! Stored-address layout planning for file creation.

use alloc::vec::Vec;

use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ManagerKind;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;
use crate::file_allocation::FileAllocation;
use crate::file_allocation::FileManager;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::free_space_manager;

use super::WIDTHS;

/// A stored file address and the byte length reserved there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Allocation {
    pub(super) address: StoredAddress,
    pub(super) length: u64,
}

impl Allocation {
    /// Reserves a region for the supplied allocation class.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FileAllocation::reserve`].
    fn reserve(
        allocation: &mut FileAllocation,
        length: u64,
        class: AllocationClass,
    ) -> Result<Self, FormatError> {
        let address = allocation.reserve(length, class)?;
        Ok(Self { address, length })
    }
}

/// The byte lengths of an object's header and optional dense storage.
///
/// Dense attribute and link storage include their heaps and indexes.
pub(super) struct ObjectMeasure {
    pub(super) header: u64,
    pub(super) attributes: Option<u64>,
    pub(super) links: Option<u64>,
}

impl ObjectMeasure {
    /// Reserves the header, dense attributes, and dense links in that order.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FileAllocation::reserve`].
    pub(super) fn reserve(
        self,
        allocation: &mut FileAllocation,
    ) -> Result<ObjectAllocation, FormatError> {
        let Self {
            header,
            attributes,
            links,
        } = self;
        let header = Allocation::reserve(allocation, header, AllocationClass::Metadata)?;
        let attributes = attributes
            .map(|length| Allocation::reserve(allocation, length, AllocationClass::Metadata))
            .transpose()?;
        let links = links
            .map(|length| Allocation::reserve(allocation, length, AllocationClass::Metadata))
            .transpose()?;
        Ok(ObjectAllocation {
            header,
            attributes,
            links,
        })
    }
}

/// Stored reservations for one object's header and optional dense storage.
pub(super) struct ObjectAllocation {
    pub(super) header: Allocation,
    pub(super) attributes: Option<Allocation>,
    pub(super) links: Option<Allocation>,
}

/// A dataset region's byte length and storage kind.
///
/// Chunked regions include their chunk index.
#[derive(Clone, Copy)]
pub(super) struct DataMeasure {
    length: u64,
    kind: DataKind,
}

impl DataMeasure {
    /// Records the payload length, including the index for chunked storage.
    pub(super) fn new(length: u64, kind: DataKind) -> Self {
        Self { length, kind }
    }

    /// Reserves raw storage, returning [`None`] for an empty contiguous region.
    ///
    /// A zero-length chunked region has a reservation at the current end of allocation.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FileAllocation::reserve`].
    pub(super) fn reserve(
        self,
        allocation: &mut FileAllocation,
    ) -> Result<Option<Allocation>, FormatError> {
        if self.empty_contiguous() {
            Ok(None)
        } else {
            Allocation::reserve(allocation, self.length, AllocationClass::Raw).map(Some)
        }
    }

    fn empty_contiguous(self) -> bool {
        let Self { length, kind } = self;
        length == 0 && matches!(kind, DataKind::Contiguous)
    }
}

/// The storage layout used to interpret a dataset region's length.
#[derive(Clone, Copy)]
pub(super) enum DataKind {
    Chunked,
    Contiguous,
}

/// The superblock extension, final allocation end, and optional manager for a classic strategy.
pub(super) struct ClassicLayout {
    pub(super) extension: Option<Allocation>,
    pub(super) end: u64,
    pub(super) manager: Option<FileManager>,
}

impl ClassicLayout {
    /// Places the optional extension and releases unused aggregate tails.
    ///
    /// With `persist`, serializes remaining reusable space into a manager at the file's tail.
    ///
    /// # Errors
    ///
    /// Propagates reservation and cleanup errors from [`FileAllocation::reserve`],
    /// [`FileAllocation::finish`], and [`FileAllocation::finish_persisted`].
    pub(super) fn finish(
        mut allocation: FileAllocation,
        extension_len: Option<u64>,
        persist: bool,
    ) -> Result<Self, FormatError> {
        let extension = extension_len
            .map(|length| Allocation::reserve(&mut allocation, length, AllocationClass::Metadata))
            .transpose()?;
        let (end, manager) = if persist {
            allocation.finish_persisted(WIDTHS)?
        } else {
            (allocation.finish()?, None)
        };
        Ok(Self {
            extension,
            end: end.get(),
            manager,
        })
    }
}

/// Free page tails around global-heap collections, grouped by manager kind.
#[derive(Default)]
pub(super) struct CollectionSections {
    pub(super) metadata: Vec<FreeSection>,
    pub(super) raw: Vec<FreeSection>,
    pub(super) large: Vec<FreeSection>,
}

/// Page-aligned global-heap reservations and the free space around them.
pub(super) struct PagedCollections {
    page: FileSpacePageSize,
    /// The first stored address available for following object metadata.
    pub(super) end: u64,
    pub(super) sections: CollectionSections,
}

impl PagedCollections {
    /// Creates an empty collection layout with its cursor at `start`.
    ///
    /// The first reservation is aligned to `page`.
    pub(super) fn new(start: u64, page: FileSpacePageSize) -> Self {
        Self {
            page,
            end: start,
            sections: CollectionSections::default(),
        }
    }

    /// Reserves a collection on a fresh page run and records the surrounding free space.
    ///
    /// Classifies the final page tail as small raw or generic large according to `length`.
    /// Free space before the collection belongs to the metadata manager.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the region or page padding exceeds [`u64::MAX`].
    pub(super) fn reserve(&mut self, length: u64) -> Result<Allocation, FormatError> {
        let Self {
            page,
            end: cursor,
            sections,
        } = self;
        let CollectionSections {
            metadata,
            raw,
            large,
        } = sections;
        let start = align(*cursor, *page)?;
        if start > *cursor {
            metadata.push(FreeSection {
                addr: StoredAddress::new(*cursor),
                size: start - *cursor,
            });
        }
        let end = checked_end(start, length)?;
        *cursor = align(end, *page)?;
        if *cursor > end {
            let section = FreeSection {
                addr: StoredAddress::new(end),
                size: *cursor - end,
            };
            if length >= page.get() {
                large.push(section);
            } else {
                raw.push(section);
            }
        }
        Ok(Allocation {
            address: StoredAddress::new(start),
            length,
        })
    }
}

/// Dataset placements, serialized managers, and region boundaries for PAGE output.
///
/// Addresses and boundaries are stored file coordinates. The caller supplies each dataset's
/// retained value through the placement callback.
pub(super) struct PagedLayout<T> {
    pub(super) extension: Allocation,
    /// Callback results in dataset measurement order.
    pub(super) data: Vec<T>,
    /// Dataset indices in the shared small-raw page run.
    pub(super) small: Vec<usize>,
    /// Dataset indices with separate page-aligned runs.
    pub(super) large: Vec<usize>,
    /// Manager header addresses in File Space Info slot order, undefined for inactive slots.
    pub(super) slots: [u64; NUM_FILE_FSM_MANAGERS],
    /// Metadata, small-raw, and generic-large managers in emission order.
    pub(super) managers: [Option<FileManager>; 3],
    /// The stored address after the extension and manager blocks, before page padding.
    pub(super) metadata_end: u64,
    /// The first stored address of the raw-data region.
    pub(super) raw_start: u64,
    /// The stored address after the small raw regions, before page padding.
    pub(super) small_end: u64,
    /// The page-aligned end of allocation.
    pub(super) end: u64,
}

impl<T> PagedLayout<T> {
    /// Plans the extension, managers, and dataset regions after the object metadata.
    ///
    /// `metadata_end` is the stored address before the extension. `sections` contains free
    /// page tails from collection placement. With `persist`, serializes the active managers.
    ///
    /// Calls `bind` once per dataset in measurement order with its index and allocation,
    /// passing [`None`] for an empty contiguous region. Small regions share a page run, and
    /// each large region starts on a page boundary. Stores callback results in [`Self::data`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a region or its padding exceeds [`u64::MAX`],
    /// or [`FormatError::Internal`] if serialized manager lengths differ from their reservations.
    /// Propagates errors from `bind` and manager serialization.
    pub(super) fn new(
        page: FileSpacePageSize,
        persist: bool,
        metadata_end: u64,
        extension_length: u64,
        measurements: impl ExactSizeIterator<Item = DataMeasure> + Clone,
        sections: CollectionSections,
        mut bind: impl FnMut(usize, Option<Allocation>) -> Result<T, FormatError>,
    ) -> Result<Self, FormatError> {
        let CollectionSections {
            metadata: mut meta_sections,
            raw: mut raw_sections,
            large: mut large_sections,
        } = sections;
        let extension = Allocation {
            address: StoredAddress::new(metadata_end),
            length: extension_length,
        };
        let mut cursor = checked_end(metadata_end, extension_length)?;
        let mut small = Vec::new();
        let mut large = Vec::new();
        let mut small_total = 0;
        let mut large_sizes = Vec::new();
        for (index, measure) in measurements.clone().enumerate() {
            if measure.empty_contiguous() {
                continue;
            }
            if measure.length < page.get() {
                small.push(index);
                small_total = checked_end(small_total, measure.length)?;
            } else {
                large.push(index);
                let padding = align(measure.length, page)? - measure.length;
                if padding > 0 {
                    large_sizes.push(padding);
                }
            }
        }
        large_sizes.extend(large_sections.iter().map(|section| section.size));
        let draw_active = !raw_sections.is_empty() || small_total % page.get() != 0;
        let large_active = !large_sizes.is_empty();
        let mut raw_sizes: Vec<_> = raw_sections.iter().map(|section| section.size).collect();
        raw_sizes.push(0);
        let draw_length = if draw_active {
            checked_end(
                hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS),
                hdf5_pure_format::__private::section_info_len(WIDTHS, &raw_sizes),
            )?
        } else {
            0
        };
        let large_length = if large_active {
            checked_end(
                hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS),
                hdf5_pure_format::__private::section_info_len(WIDTHS, &large_sizes),
            )?
        } else {
            0
        };
        let super_allocation = if persist {
            paged_metadata_section_allocation(
                page,
                cursor,
                [draw_length, large_length],
                &meta_sections,
            )?
        } else {
            None
        };
        let mut slots = [StoredAddress::undefined(WIDTHS.offsets).get(); NUM_FILE_FSM_MANAGERS];
        let mut locations = [None, None, None];
        if persist {
            for (target, kind, length) in [
                (0, ManagerKind::PagedSmallMetadata, super_allocation),
                (
                    1,
                    ManagerKind::PagedSmallRaw,
                    draw_active
                        .then(|| hdf5_pure_format::__private::section_info_len(WIDTHS, &raw_sizes)),
                ),
                (
                    2,
                    ManagerKind::PagedLargeGeneric,
                    large_active.then(|| {
                        hdf5_pure_format::__private::section_info_len(WIDTHS, &large_sizes)
                    }),
                ),
            ] {
                if let Some(length) = length {
                    let header = StoredAddress::new(cursor);
                    cursor = checked_end(
                        cursor,
                        hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS),
                    )?;
                    let section_list = StoredAddress::new(cursor);
                    cursor = checked_end(cursor, length)?;
                    let (slot, _) = free_space_manager::manager_wire(kind);
                    slots[slot] = header.get();
                    locations[target] = Some(ManagerLocation {
                        header,
                        section_list,
                        length,
                        kind,
                    });
                }
            }
        }
        let metadata_end = cursor;
        let raw_start = align(cursor, page)?;
        if raw_start > cursor {
            meta_sections.push(FreeSection {
                addr: StoredAddress::new(cursor),
                size: raw_start - cursor,
            });
        }
        let small_end = checked_end(raw_start, small_total)?;
        let mut small_cursor = raw_start;
        cursor = align(small_end, page)?;
        if small_total > 0 && cursor > small_end {
            raw_sections.push(FreeSection {
                addr: StoredAddress::new(small_end),
                size: cursor - small_end,
            });
        }
        let mut data = Vec::with_capacity(measurements.len());
        for (index, measure) in measurements.enumerate() {
            let placed = if measure.empty_contiguous() {
                None
            } else if measure.length < page.get() {
                let address = StoredAddress::new(small_cursor);
                small_cursor = checked_end(small_cursor, measure.length)?;
                Some(Allocation {
                    address,
                    length: measure.length,
                })
            } else {
                let start = align(cursor, page)?;
                let end = checked_end(start, measure.length)?;
                cursor = align(end, page)?;
                if cursor > end {
                    large_sections.push(FreeSection {
                        addr: StoredAddress::new(end),
                        size: cursor - end,
                    });
                }
                Some(Allocation {
                    address: StoredAddress::new(start),
                    length: measure.length,
                })
            };
            data.push(bind(index, placed)?);
        }
        let mut managers = [None, None, None];
        for ((target, location), sections) in
            locations
                .into_iter()
                .enumerate()
                .zip([meta_sections, raw_sections, large_sections])
        {
            if let Some(location) = location {
                managers[target] = Some(location.serialize(&sections)?);
            }
        }
        Ok(Self {
            extension,
            data,
            small,
            large,
            slots,
            managers,
            metadata_end,
            raw_start,
            small_end,
            end: cursor,
        })
    }
}

/// Stored reservations for a manager's header and section list.
struct ManagerLocation {
    header: StoredAddress,
    section_list: StoredAddress,
    length: u64,
    kind: ManagerKind,
}

impl ManagerLocation {
    /// Serializes the manager at its reserved addresses and checks its lengths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if serialized lengths differ from their reservations.
    /// Propagates manager serialization errors.
    fn serialize(self, sections: &[FreeSection]) -> Result<FileManager, FormatError> {
        let Self {
            header,
            section_list,
            length,
            kind,
        } = self;
        let (_, class) = free_space_manager::manager_wire(kind);
        let (header_bytes, section_bytes) = if kind == ManagerKind::PagedLargeGeneric {
            hdf5_pure_format::__private::serialize_free_space_manager(
                WIDTHS,
                sections,
                header,
                section_list,
                class,
            )?
        } else {
            hdf5_pure_format::__private::serialize_free_space_manager_with_allocation(
                WIDTHS,
                sections,
                header,
                section_list,
                class,
                Some(length),
            )?
        };
        if header_bytes.len() as u64
            != hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS)
            || section_bytes.len() as u64 != length
        {
            return Err(FormatError::Internal(
                "free-space manager bytes differ from their allocation".into(),
            ));
        }
        Ok(FileManager {
            header_address: header,
            section_list_address: section_list,
            header: header_bytes,
            section_list: section_bytes,
        })
    }
}

fn checked_end(offset: u64, length: u64) -> Result<u64, FormatError> {
    offset
        .checked_add(length)
        .ok_or(FormatError::OffsetOverflow { offset, length })
}

fn align(offset: u64, page: FileSpacePageSize) -> Result<u64, FormatError> {
    checked_end(offset, (page.get() - offset % page.get()) % page.get())
}

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

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private as format;
    use hdf5_pure_format::__private::FreeSection;
    use hdf5_pure_format::__private::FreeSpaceManagerHeader;
    use rstest::rstest;

    use super::Allocation;
    use super::CollectionSections;
    use super::DataKind;
    use super::DataMeasure;
    use super::ObjectMeasure;
    use super::PagedCollections;
    use super::PagedLayout;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::error::FormatError;
    use crate::file_allocation::FileAllocation;
    use crate::file_space_info::FileSpacePageSize;
    use crate::file_writer::WIDTHS;

    #[rstest]
    #[case::direct(false)]
    #[case::aggregated(true)]
    fn object_metadata_reservations_keep_header_attribute_and_link_order(#[case] aggregated: bool) {
        let mut allocation = FileAllocation::new(StoredAddress::new(48), aggregated, None);
        let placed = ObjectMeasure {
            header: 80,
            attributes: Some(24),
            links: Some(64),
        }
        .reserve(&mut allocation)
        .unwrap();
        let super::ObjectAllocation {
            header,
            attributes,
            links,
        } = placed;
        assert_eq!(header, span(48, 80));
        assert_eq!(attributes, Some(span(128, 24)));
        assert_eq!(links, Some(span(152, 64)));
    }

    #[rstest]
    #[case::empty_contiguous(DataKind::Contiguous, 0, None, 48)]
    #[case::empty_chunked(DataKind::Chunked, 0, Some((48, 0)), 48)]
    #[case::contiguous(DataKind::Contiguous, 7, Some((48, 7)), 55)]
    #[case::chunked(DataKind::Chunked, 7, Some((48, 7)), 55)]
    fn data_reservations_distinguish_empty_contiguous_storage(
        #[case] kind: DataKind,
        #[case] length: u64,
        #[case] expected: Option<(u64, u64)>,
        #[case] end: u64,
    ) {
        let mut allocation = FileAllocation::new(StoredAddress::new(48), false, None);
        let placed = DataMeasure::new(length, kind)
            .reserve(&mut allocation)
            .unwrap();
        assert_eq!(
            placed,
            expected.map(|(address, length)| span(address, length))
        );
        assert_eq!(allocation.end(), StoredAddress::new(end));
    }

    #[rstest]
    #[case::minimum_page(512, 0)]
    #[case::non_power_of_two(3000, 0)]
    #[case::above_four_gibibytes(512, 5_120_000_000)]
    fn paged_layout_records_small_large_and_empty_data(#[case] page: u64, #[case] offset: u64) {
        let measurements = [
            DataMeasure::new(0, DataKind::Contiguous),
            DataMeasure::new(7, DataKind::Contiguous),
            DataMeasure::new(0, DataKind::Chunked),
            DataMeasure::new(page + 188, DataKind::Contiguous),
        ];
        let plan = PagedLayout::new(
            FileSpacePageSize::try_from(page).unwrap(),
            false,
            offset + 200,
            20,
            measurements.into_iter(),
            CollectionSections::default(),
            |_, allocation| Ok(allocation),
        )
        .unwrap();
        let PagedLayout {
            extension,
            data,
            small,
            large,
            slots,
            managers,
            metadata_end,
            raw_start,
            small_end,
            end,
        } = plan;
        assert_eq!(extension, span(offset + 200, 20));
        assert_eq!(metadata_end, offset + 220);
        assert_eq!(raw_start, offset + page);
        assert_eq!(small_end, offset + page + 7);
        assert_eq!(end, offset + 4 * page);
        assert_eq!(small, [1, 2]);
        assert_eq!(large, [3]);
        assert_eq!(
            data,
            [
                None,
                Some(span(offset + page, 7)),
                Some(span(offset + page + 7, 0)),
                Some(span(offset + 2 * page, page + 188))
            ]
        );
        assert_eq!(slots, [u64::MAX; super::NUM_FILE_FSM_MANAGERS]);
        assert_eq!(managers.map(|manager| manager.is_some()), [false; 3]);
    }

    #[rstest]
    #[case::metadata(PageTail::Metadata, 0, 0)]
    #[case::small_raw(PageTail::Raw, 1, 2)]
    #[case::large(PageTail::Large, 2, 6)]
    fn paged_managers_describe_the_planned_page_tails(
        #[case] kind: PageTail,
        #[case] manager_index: usize,
        #[case] slot: usize,
    ) {
        let page = FileSpacePageSize::try_from(512).unwrap();
        let plan = PagedLayout::new(
            page,
            true,
            200,
            20,
            [
                DataMeasure::new(7, DataKind::Contiguous),
                DataMeasure::new(700, DataKind::Chunked),
            ]
            .into_iter(),
            CollectionSections::default(),
            |_, allocation| Ok(allocation),
        )
        .unwrap();
        let PagedLayout {
            data,
            slots,
            managers,
            metadata_end,
            raw_start,
            small_end,
            end,
            extension: _,
            small: _,
            large: _,
        } = plan;
        let manager = managers[manager_index].as_ref().unwrap();
        assert_eq!(slots[slot], manager.header_address.get());
        let header = FreeSpaceManagerHeader::parse(WIDTHS, &manager.header).unwrap();
        let sections = format::parse_section_info(
            WIDTHS,
            &manager.section_list,
            manager.header_address,
            &header,
        )
        .unwrap();
        let expected = match kind {
            PageTail::Metadata => FreeSection {
                addr: StoredAddress::new(metadata_end),
                size: raw_start - metadata_end,
            },
            PageTail::Raw => FreeSection {
                addr: StoredAddress::new(small_end),
                size: 505,
            },
            PageTail::Large => FreeSection {
                addr: StoredAddress::new(data[1].unwrap().address.get() + 700),
                size: 324,
            },
        };
        assert_eq!(sections, [expected]);
        assert_eq!(end % page.get(), 0);
    }

    #[test]
    fn paged_collections_keep_metadata_and_raw_page_tails_separate() {
        let mut collections = PagedCollections::new(48, FileSpacePageSize::try_from(512).unwrap());
        assert_eq!(collections.end, 48);
        assert_eq!(collections.reserve(3).unwrap(), span(512, 3));
        assert_eq!(collections.reserve(700).unwrap(), span(1024, 700));
        assert_eq!(collections.reserve(512).unwrap(), span(2048, 512));
        let PagedCollections {
            end,
            sections:
                CollectionSections {
                    metadata,
                    raw,
                    large,
                },
            page: _,
        } = collections;
        assert_eq!(end, 2560);
        assert_eq!(
            metadata,
            [FreeSection {
                addr: StoredAddress::new(48),
                size: 464
            }]
        );
        assert_eq!(
            raw,
            [FreeSection {
                addr: StoredAddress::new(515),
                size: 509
            }]
        );
        assert_eq!(
            large,
            [FreeSection {
                addr: StoredAddress::new(1724),
                size: 324
            }]
        );
    }

    #[rstest]
    #[case::extension(u64::MAX, 1, None, u64::MAX, 1)]
    #[case::metadata_padding(u64::MAX - 1, 0, None, u64::MAX - 1, 2)]
    #[case::large_extent_padding(200, 20, Some(u64::MAX), u64::MAX, 1)]
    fn paged_layout_rejects_overflow_before_emission(
        #[case] metadata_end: u64,
        #[case] extension: u64,
        #[case] data_length: Option<u64>,
        #[case] offset: u64,
        #[case] length: u64,
    ) {
        let error = PagedLayout::new(
            FileSpacePageSize::try_from(512).unwrap(),
            false,
            metadata_end,
            extension,
            data_length
                .into_iter()
                .map(|length| DataMeasure::new(length, DataKind::Chunked)),
            CollectionSections::default(),
            |_, allocation| Ok(allocation),
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(error, FormatError::OffsetOverflow { offset, length });
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[rstest]
    #[case::small(128, "page_bound_records", "page_bound_records_control")]
    #[case::large(512, "page_many_bound_records", "page_many_bound_records_control")]
    fn paged_planning_retains_one_record_table(
        #[case] count: usize,
        #[case] region: &'static str,
        #[case] control_region: &'static str,
    ) {
        use core::mem;

        use test_util::allocation;

        let (control, positive) = allocation::measure(control_region, || {
            std::hint::black_box(alloc::vec![0_u8; 1024])
        });
        assert!(positive.blocks > 0, "{positive}");
        assert!(positive.peak_bytes >= 1024, "{positive}");
        drop(control);
        let mut calls = 0;
        let (plan, measured) = allocation::measure(region, || {
            PagedLayout::new(
                FileSpacePageSize::try_from(512).unwrap(),
                false,
                200,
                20,
                (0..count).map(|_| DataMeasure::new(16, DataKind::Contiguous)),
                CollectionSections::default(),
                |index, allocation| {
                    calls += 1;
                    Ok((index, allocation, [0xa5_u8; 256]))
                },
            )
            .unwrap()
        });
        let PagedLayout {
            data,
            small,
            large,
            raw_start,
            extension: _,
            slots: _,
            managers: _,
            metadata_end: _,
            small_end: _,
            end: _,
        } = plan;
        let retained = data.capacity() * mem::size_of_val(data.first().unwrap())
            + (small.capacity() + large.capacity()) * mem::size_of::<usize>();
        assert_eq!(calls, count);
        assert_eq!(data.len(), count);
        assert_eq!(small, (0..count).collect::<alloc::vec::Vec<_>>());
        assert!(large.is_empty());
        for (index, (actual, allocation, bytes)) in data.iter().enumerate() {
            assert_eq!(*actual, index);
            assert_eq!(*allocation, Some(span(raw_start + index as u64 * 16, 16)));
            assert_eq!(*bytes, [0xa5; 256]);
        }
        assert_eq!(measured.live_bytes, retained as u64, "{measured}");
        assert!(measured.peak_bytes >= retained as u64, "{measured}");
        assert!(measured.peak_bytes <= retained as u64 + 1024, "{measured}");
    }

    enum PageTail {
        Large,
        Metadata,
        Raw,
    }

    fn span(address: u64, length: u64) -> Allocation {
        Allocation {
            address: StoredAddress::new(address),
            length,
        }
    }
}
