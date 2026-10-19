//! Stored-address allocation for new files.

use alloc::vec::Vec;
use core::num::NonZeroU64;

use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_format::__private::SECTION_CLASS_SIMPLE;
use hdf5_pure_space::__private::AggregationError;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ClassicAggregators;
use hdf5_pure_space::__private::SessionSpace;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;
use crate::width::FormatWidths;

/// Reserves stored-address ranges by direct append or through metadata and raw aggregates.
pub(crate) struct FileAllocation {
    end: u64,
    aggregates: Option<ClassicAggregators>,
    space: SessionSpace,
}

impl FileAllocation {
    /// Starts allocation at `end`, with separate metadata and raw aggregates if `aggregated`.
    ///
    /// With `threshold` set, tracks abandoned aggregate spare of at least that size for reuse
    /// during construction. Active aggregate spare remains reserved.
    pub(crate) fn new(end: u64, aggregated: bool, threshold: Option<u64>) -> Self {
        Self {
            end,
            aggregates: aggregated.then(ClassicAggregators::new),
            space: SessionSpace::from_strategy(
                threshold.map(|_| FileSpaceStrategy::FsmAggr),
                threshold.unwrap_or_default(),
                FileSpacePageSize::DEFAULT,
            ),
        }
    }

    /// Reserves `length` bytes for `class` and returns their stored address.
    ///
    /// Tries tracked free space before an aggregate or direct append. Returns the current EOA
    /// for a zero-length request. With aggregation enabled, the caller
    /// can receive an address before an earlier allocation of the same class after allocating
    /// a large extent outside an active window.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the reservation exceeds [`u64::MAX`], or
    /// [`FormatError::Internal`] if an active aggregate extends beyond the writer's EOA.
    pub(crate) fn reserve(
        &mut self,
        length: u64,
        class: AllocationClass,
    ) -> Result<StoredAddress, FormatError> {
        let Some(length) = NonZeroU64::new(length) else {
            return Ok(StoredAddress::new(self.end));
        };
        if let Some(extent) = self.space.allocate(length.get(), class) {
            return Ok(StoredAddress::new(extent.start()));
        }
        if let Some(aggregates) = &mut self.aggregates {
            let allocation = aggregates
                .allocate(self.end, length, class)
                .map_err(|error| match error {
                    AggregationError::OffsetOverflow { offset, length } => {
                        FormatError::OffsetOverflow { offset, length }
                    }
                    AggregationError::EndOfAllocationTooSmall { .. } => {
                        FormatError::Internal("aggregate reservation exceeds the writer EOA".into())
                    }
                })?;
            self.end = allocation.end_of_allocation();
            if let Some(end) = self.space.release_freed(self.end, allocation.released()) {
                self.end = end;
            }
            Ok(StoredAddress::new(allocation.extent().start()))
        } else {
            let address = self.end;
            self.end = self
                .end
                .checked_add(length.get())
                .ok_or(FormatError::OffsetOverflow {
                    offset: self.end,
                    length: length.get(),
                })?;
            Ok(StoredAddress::new(address))
        }
    }

    /// Releases unused aggregate tails at EOA and returns the final stored-address end.
    ///
    /// Unused interior space remains allocated. The writer discards the construction free list
    /// without persisting it.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if aggregate cleanup exceeds [`u64::MAX`], or
    /// [`FormatError::Internal`] if an active aggregate extends beyond the writer's EOA.
    pub(crate) fn finish(mut self) -> Result<u64, FormatError> {
        self.settle(false)?;
        Ok(self.end)
    }

    /// Releases unused aggregate reservations and serializes the remaining reusable space.
    ///
    /// Returns the final stored-address EOA and the manager blocks to write. The blocks follow
    /// the file's content and are excluded from the recorded free sections. Returns `None` for
    /// the manager if no sections remain after tail release and threshold admission.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if cleanup or manager placement exceeds
    /// [`u64::MAX`], [`FormatError::AddressTooLarge`] or [`FormatError::LengthTooLarge`] if a
    /// manager field exceeds its width, [`FormatError::InvalidFreeSpaceManager`] if a section
    /// exceeds the manager's size limit, or [`FormatError::Internal`] if an active aggregate
    /// extends beyond the writer's EOA or the free-space total exceeds [`u64::MAX`].
    pub(crate) fn finish_persisted(
        mut self,
        widths: FormatWidths,
    ) -> Result<(u64, Option<FileManager>), FormatError> {
        self.settle(true)?;
        let sections: Vec<_> = self
            .space
            .reusable_sections()
            .map(|extent| FreeSection {
                addr: StoredAddress::new(extent.start()),
                size: extent.len(),
            })
            .collect();
        if sections.is_empty() {
            return Ok((self.end, None));
        }
        let header_address = StoredAddress::new(self.end);
        let length = hdf5_pure_format::__private::free_space_manager_header_len(widths);
        let section_list_address = self
            .end
            .checked_add(length)
            .ok_or(FormatError::OffsetOverflow {
                offset: self.end,
                length,
            })
            .map(StoredAddress::new)?;
        let (header, section_list) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths,
            &sections,
            header_address,
            section_list_address,
            SECTION_CLASS_SIMPLE,
        )?;
        self.end = section_list_address
            .get()
            .checked_add(section_list.len() as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: section_list_address.get(),
                length: section_list.len() as u64,
            })?;
        Ok((
            self.end,
            Some(FileManager {
                header_address,
                section_list_address,
                header,
                section_list,
            }),
        ))
    }

    pub(crate) fn end(&self) -> u64 {
        self.end
    }

    fn settle(&mut self, persist: bool) -> Result<(), FormatError> {
        if let Some(aggregates) = &mut self.aggregates {
            let released = aggregates.reset(self.end).map_err(|error| match error {
                AggregationError::OffsetOverflow { offset, length } => {
                    FormatError::OffsetOverflow { offset, length }
                }
                AggregationError::EndOfAllocationTooSmall { .. } => {
                    FormatError::Internal("aggregate reservation exceeds the writer EOA".into())
                }
            })?;
            self.end = released.end_of_allocation();
            if let Some(end) = self.space.release_freed(
                self.end,
                [AllocationClass::Metadata, AllocationClass::Raw]
                    .into_iter()
                    .filter_map(|class| persist.then(|| released.released(class)).flatten()),
            ) {
                self.end = end;
            }
        }
        Ok(())
    }
}

/// Holds a serialized file free-space manager and its stored placement addresses.
#[derive(Debug)]
pub(crate) struct FileManager {
    pub(crate) header_address: StoredAddress,
    pub(crate) section_list_address: StoredAddress,
    pub(crate) header: Vec<u8>,
    pub(crate) section_list: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::FreeSection;
    use hdf5_pure_format::__private::FreeSpaceManagerHeader;
    use hdf5_pure_space::__private::AllocationClass;
    use rstest::rstest;

    use super::FileAllocation;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::error::FormatError;
    use crate::width::FormatWidths;

    #[rstest]
    #[case::below(1_983, true)]
    #[case::equal(1_984, true)]
    #[case::above(1_985, false)]
    #[case::maximum(u64::MAX, false)]
    fn persistence_admits_settled_spare_without_exposing_manager_storage(
        #[case] threshold: u64,
        #[case] has_manager: bool,
        #[values(2, 4, 8)] width: u8,
    ) {
        let widths = FormatWidths::from_sizes(width, width).unwrap();
        let mut allocation = FileAllocation::new(48, true, Some(threshold));
        allocation.reserve(64, AllocationClass::Metadata).unwrap();
        allocation.reserve(32, AllocationClass::Raw).unwrap();
        assert_eq!(allocation.space.reusable_sections().count(), 0);
        let (end, manager) = allocation.finish_persisted(widths).unwrap();
        assert_eq!(manager.is_some(), has_manager);
        if let Some(manager) = manager {
            let header = FreeSpaceManagerHeader::parse(widths, &manager.header).unwrap();
            assert_eq!(manager.header_address.get(), 2_128);
            assert_eq!(header.section_list_addr(), manager.section_list_address);
            assert_eq!(
                header.section_list_used(),
                manager.section_list.len() as u64
            );
            assert_eq!(
                header.section_list_allocated(),
                manager.section_list.len() as u64
            );
            assert_eq!(
                end,
                manager.section_list_address.get() + header.section_list_allocated()
            );
            let sections = hdf5_pure_format::__private::parse_section_info(
                widths,
                &manager.section_list,
                manager.header_address,
                &header,
            )
            .unwrap();
            assert_eq!(
                sections,
                [FreeSection {
                    addr: StoredAddress::new(112),
                    size: 1_984,
                }]
            );
        } else {
            assert_eq!(end, 2_128);
        }
    }

    #[test]
    fn persistence_keeps_abandoned_and_settled_spare() {
        let mut allocation = FileAllocation::new(48, true, Some(1));
        allocation.reserve(64, AllocationClass::Metadata).unwrap();
        allocation.reserve(64, AllocationClass::Raw).unwrap();
        allocation
            .reserve(2_000, AllocationClass::Metadata)
            .unwrap();
        assert_eq!(
            allocation.reserve(32, AllocationClass::Raw).unwrap().get(),
            112
        );
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let (_, manager) = allocation.finish_persisted(widths).unwrap();
        let manager = manager.unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &manager.header).unwrap();
        let sections = hdf5_pure_format::__private::parse_section_info(
            widths,
            &manager.section_list,
            manager.header_address,
            &header,
        )
        .unwrap();
        assert_eq!(
            sections,
            [
                FreeSection {
                    addr: StoredAddress::new(144),
                    size: 1_952
                },
                FreeSection {
                    addr: StoredAddress::new(2_160),
                    size: 1_984
                },
            ]
        );
        assert_eq!(manager.header_address.get(), 6_144);
    }

    #[test]
    fn an_empty_free_list_emits_no_manager() {
        let mut allocation = FileAllocation::new(48, true, Some(1));
        allocation.reserve(64, AllocationClass::Metadata).unwrap();
        let (end, manager) = allocation
            .finish_persisted(FormatWidths::from_sizes(8, 8).unwrap())
            .unwrap();
        assert_eq!(end, 112);
        assert!(manager.is_none());
    }

    #[rstest]
    #[case::header(u64::MAX, true)]
    #[case::section_list(u64::MAX - 100, false)]
    fn persistent_manager_placement_checks_its_end(#[case] offset: u64, #[case] header: bool) {
        let mut allocation = FileAllocation::new(48, true, Some(1));
        allocation.reserve(64, AllocationClass::Metadata).unwrap();
        allocation.reserve(32, AllocationClass::Raw).unwrap();
        allocation.end = offset;
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let error = allocation.finish_persisted(widths).unwrap_err();
        assert_eq!(
            error,
            FormatError::OffsetOverflow {
                offset: if header {
                    offset
                } else {
                    offset + hdf5_pure_format::__private::free_space_manager_header_len(widths)
                },
                length: if header {
                    hdf5_pure_format::__private::free_space_manager_header_len(widths)
                } else {
                    hdf5_pure_format::__private::section_info_len(widths, &[1_984, 2_016])
                },
            }
        );
    }

    #[rstest]
    #[case::below(1_983, 112)]
    #[case::equal(1_984, 112)]
    #[case::above(1_985, 6_144)]
    fn abandoned_spare_obeys_the_threshold_and_precedes_an_active_window(
        #[case] threshold: u64,
        #[case] expected: u64,
        #[values(AllocationClass::Metadata, AllocationClass::Raw)] class: AllocationClass,
    ) {
        let other = match class {
            AllocationClass::Metadata => AllocationClass::Raw,
            AllocationClass::Raw => AllocationClass::Metadata,
        };
        let mut allocation = FileAllocation::new(48, true, Some(threshold));
        assert_eq!(allocation.reserve(64, class).unwrap().get(), 48);
        assert_eq!(allocation.reserve(64, other).unwrap().get(), 2_096);
        assert_eq!(allocation.reserve(2_000, class).unwrap().get(), 4_144);
        assert_eq!(allocation.reserve(32, class).unwrap().get(), expected);
        assert_eq!(allocation.end(), 6_192);
        assert_eq!(allocation.finish().unwrap(), expected.max(6_112) + 32);
    }

    #[rstest]
    #[case::metadata(AllocationClass::Metadata, AllocationClass::Raw)]
    #[case::raw(AllocationClass::Raw, AllocationClass::Metadata)]
    fn reusable_spare_precedes_a_refill_without_exposing_active_spare(
        #[case] class: AllocationClass,
        #[case] other: AllocationClass,
    ) {
        let mut allocation = FileAllocation::new(48, true, Some(1));
        assert_eq!(allocation.reserve(64, class).unwrap().get(), 48);
        assert_eq!(allocation.reserve(64, other).unwrap().get(), 2_096);
        assert_eq!(allocation.space.reusable_sections().count(), 0);
        assert_eq!(allocation.reserve(2_000, class).unwrap().get(), 4_144);
        assert_eq!(allocation.reserve(1_984, other).unwrap().get(), 112);
        assert_eq!(allocation.reserve(1_984, other).unwrap().get(), 2_160);
        assert_eq!(allocation.reserve(4_096, other).unwrap().get(), 6_192);
        assert_eq!(allocation.reserve(32, class).unwrap().get(), 6_144);
        assert_eq!(allocation.end(), 10_288);
        assert_eq!(allocation.finish().unwrap(), 10_288);
    }

    #[rstest]
    #[case::metadata(AllocationClass::Metadata, AllocationClass::Raw)]
    #[case::raw(AllocationClass::Raw, AllocationClass::Metadata)]
    fn large_bypasses_preserve_an_older_window(
        #[case] class: AllocationClass,
        #[case] other: AllocationClass,
        #[values(None, Some(1))] threshold: Option<u64>,
    ) {
        let mut allocation = FileAllocation::new(48, true, threshold);
        assert_eq!(allocation.reserve(64, class).unwrap().get(), 48);
        assert_eq!(allocation.reserve(64, other).unwrap().get(), 2_096);
        assert_eq!(allocation.reserve(4_096, class).unwrap().get(), 4_144);
        assert_eq!(allocation.reserve(32, class).unwrap().get(), 112);
        assert_eq!(allocation.end(), 8_240);
        assert_eq!(allocation.finish().unwrap(), 8_240);
    }

    #[rstest]
    #[case::aggregated(true, None, 2_096)]
    #[case::reusable(true, Some(1), 2_096)]
    #[case::high_threshold(true, Some(u64::MAX), 2_096)]
    #[case::direct(false, None, 112)]
    fn finishing_releases_only_the_unused_tail(
        #[case] aggregated: bool,
        #[case] threshold: Option<u64>,
        #[case] expected_end: u64,
    ) {
        let mut allocation = FileAllocation::new(48, aggregated, threshold);
        assert_eq!(
            allocation
                .reserve(64, AllocationClass::Metadata)
                .unwrap()
                .get(),
            48
        );
        assert_eq!(
            allocation.reserve(32, AllocationClass::Raw).unwrap().get(),
            expected_end
        );
        assert_eq!(allocation.finish().unwrap(), expected_end + 32);
    }

    #[rstest]
    #[case::aggregated(true, None, hdf5_pure_space::__private::AGGREGATE_BLOCK_SIZE)]
    #[case::reusable(true, Some(1), hdf5_pure_space::__private::AGGREGATE_BLOCK_SIZE)]
    #[case::direct(false, None, 1)]
    fn overflowing_reservations_preserve_the_end(
        #[case] aggregated: bool,
        #[case] threshold: Option<u64>,
        #[case] length: u64,
    ) {
        let mut allocation = FileAllocation::new(u64::MAX, aggregated, threshold);
        assert_eq!(
            allocation.reserve(0, AllocationClass::Raw).unwrap().get(),
            u64::MAX
        );
        assert_eq!(
            allocation.reserve(1, AllocationClass::Raw).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length
            }
        );
        assert_eq!(allocation.end(), u64::MAX);
        assert_eq!(allocation.finish().unwrap(), u64::MAX);
    }
}
