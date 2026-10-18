//! Stored-address allocation for new files.

use core::num::NonZeroU64;

use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_space::__private::AggregationError;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ClassicAggregators;
use hdf5_pure_space::__private::SessionSpace;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;

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
            if let Some(end) = self.space.release_freed(self.end, []) {
                self.end = end;
            }
        }
        Ok(self.end)
    }

    pub(crate) fn end(&self) -> u64 {
        self.end
    }
}

#[cfg(test)]
mod tests {
    use hdf5_pure_space::__private::AllocationClass;
    use rstest::rstest;

    use super::FileAllocation;
    use crate::error::FormatError;

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
