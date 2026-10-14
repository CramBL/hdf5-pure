//! Session allocation policy for separate metadata and raw aggregates.

use core::fmt;
use core::num::NonZeroU64;

use super::allocation::AllocationClass;
use super::extent::Extent;

/// Tracks unused space in separate metadata and raw aggregates.
///
/// For small requests, the reservation size is [`AGGREGATE_BLOCK_SIZE`] bytes. For large
/// requests, allocation extends a trailing aggregate or uses the end of allocation (EOA).
/// `H5MF__aggr_alloc` in `H5MFaggr.c`, HDF5 2.2.0, uses these allocation rules.
///
/// The caller supplies the current EOA in the same address coordinates on every call and
/// performs storage operations from the returned extents and EOA. Copying this value saves
/// the aggregate state for rollback. Storage changes require separate rollback by the caller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClassicAggregators {
    metadata: Option<Aggregate>,
    raw: Option<Aggregate>,
}

impl ClassicAggregators {
    /// Creates empty metadata and raw aggregates.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves space from the selected aggregate or at EOA.
    ///
    /// The caller passes an `eoa` at or beyond the end of every reserved aggregate.
    /// Any unused interior extent returned by [`AggregationAllocation::released`] has the
    /// requested `class` and is no longer reserved by this allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AggregationError::EndOfAllocationTooSmall`] if `eoa` precedes a reserved
    /// aggregate's end, or [`AggregationError::OffsetOverflow`] if a derived extent exceeds
    /// [`u64::MAX`]. The aggregate state is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroU64;
    /// use hdf5_pure_space::__private::{AllocationClass, ClassicAggregators, Extent};
    ///
    /// let mut aggregates = ClassicAggregators::new();
    /// let snapshot = aggregates;
    /// let allocation = aggregates.allocate(
    ///     8_192, NonZeroU64::new(64).unwrap(), AllocationClass::Raw,
    /// )?;
    /// assert_eq!(allocation.extent(), Extent::new(8_192, 64).unwrap());
    /// assert_eq!(allocation.end_of_allocation(), 10_240);
    /// aggregates = snapshot;
    /// assert_eq!(aggregates.unused(AllocationClass::Raw), None);
    /// # Ok::<(), hdf5_pure_space::__private::AggregationError>(())
    /// ```
    pub fn allocate(
        &mut self,
        eoa: u64,
        size: NonZeroU64,
        class: AllocationClass,
    ) -> Result<AggregationAllocation, AggregationError> {
        let minimum = self.minimum_eoa();
        if eoa < minimum {
            return Err(AggregationError::EndOfAllocationTooSmall { eoa, minimum });
        }
        let mut next = *self;
        let allocation = next.reserve(eoa, size, class)?;
        *self = next;
        Ok(allocation)
    }

    /// Releases an unused aggregate tail that ends at EOA.
    ///
    /// Returns the EOA after the release. The caller performs any truncation.
    /// Interior unused extents stay reserved.
    ///
    /// # Errors
    ///
    /// Returns [`AggregationError::EndOfAllocationTooSmall`] if `eoa` precedes a reserved
    /// aggregate's end. The aggregate state is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroU64;
    /// use hdf5_pure_space::__private::{AllocationClass, ClassicAggregators};
    ///
    /// let mut aggregates = ClassicAggregators::new();
    /// let allocation = aggregates.allocate(
    ///     8_192, NonZeroU64::new(64).unwrap(), AllocationClass::Raw,
    /// )?;
    /// let eoa = aggregates.release_tail(allocation.end_of_allocation())?;
    /// assert_eq!(eoa, 8_256);
    /// assert_eq!(aggregates.unused(AllocationClass::Raw), None);
    /// # Ok::<(), hdf5_pure_space::__private::AggregationError>(())
    /// ```
    pub fn release_tail(&mut self, eoa: u64) -> Result<u64, AggregationError> {
        let minimum = self.minimum_eoa();
        if eoa < minimum {
            return Err(AggregationError::EndOfAllocationTooSmall { eoa, minimum });
        }
        let mut eoa = eoa;
        let Self { metadata, raw } = self;
        for window in [metadata, raw] {
            if let Some(unused) = window.and_then(Aggregate::unused)
                && unused.end() == eoa
            {
                eoa = unused.start();
                *window = None;
            }
        }
        Ok(eoa)
    }

    /// Releases unused extents and clears both aggregates.
    ///
    /// Unused tails at EOA reduce the returned EOA. Interior unused extents are returned
    /// separately by allocation class. The caller performs any truncation or free-space
    /// admission.
    ///
    /// # Errors
    ///
    /// Returns [`AggregationError::EndOfAllocationTooSmall`] if `eoa` precedes a reserved
    /// aggregate's end. The aggregate state is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroU64;
    /// use hdf5_pure_space::__private::{AllocationClass, ClassicAggregators, Extent};
    ///
    /// let mut aggregates = ClassicAggregators::new();
    /// let metadata = aggregates.allocate(
    ///     8_192, NonZeroU64::new(64).unwrap(), AllocationClass::Metadata,
    /// )?;
    /// let raw = aggregates.allocate(
    ///     metadata.end_of_allocation(), NonZeroU64::new(64).unwrap(), AllocationClass::Raw,
    /// )?;
    /// let released = aggregates.reset(raw.end_of_allocation())?;
    /// assert_eq!(released.end_of_allocation(), 10_304);
    /// assert_eq!(released.released(AllocationClass::Metadata), Extent::new(8_256, 1_984));
    /// assert_eq!(released.released(AllocationClass::Raw), None);
    /// assert_eq!(aggregates, ClassicAggregators::new());
    /// # Ok::<(), hdf5_pure_space::__private::AggregationError>(())
    /// ```
    pub fn reset(&mut self, eoa: u64) -> Result<AggregationRelease, AggregationError> {
        let mut next = *self;
        let eoa = next.release_tail(eoa)?;
        let released = AggregationRelease {
            metadata: next.unused(AllocationClass::Metadata),
            raw: next.unused(AllocationClass::Raw),
            eoa,
        };
        *self = Self::new();
        Ok(released)
    }

    /// Returns the selected aggregate's unused extent, if any.
    pub fn unused(&self, class: AllocationClass) -> Option<Extent> {
        let Self { metadata, raw } = self;
        match class {
            AllocationClass::Metadata => metadata,
            AllocationClass::Raw => raw,
        }
        .and_then(Aggregate::unused)
    }

    // H5MF__aggr_alloc in H5MFaggr.c, HDF5 2.2.0, extends a trailing aggregate
    // by one block for a small request and by the request size for a large request.
    fn reserve(
        &mut self,
        mut eoa: u64,
        size: NonZeroU64,
        class: AllocationClass,
    ) -> Result<AggregationAllocation, AggregationError> {
        let Self { metadata, raw } = self;
        let (own, other) = match class {
            AllocationClass::Metadata => (metadata, raw),
            AllocationClass::Raw => (raw, metadata),
        };
        let size = size.get();
        if let Some(window) = own {
            if size <= window.range.end() - window.cursor {
                return Ok(AggregationAllocation {
                    extent: window.take(size)?,
                    eoa,
                    released: None,
                });
            }
            if window.range.end() == eoa {
                let growth = if size >= AGGREGATE_BLOCK_SIZE {
                    size
                } else {
                    AGGREGATE_BLOCK_SIZE
                };
                let end = eoa
                    .checked_add(growth)
                    .ok_or(AggregationError::OffsetOverflow {
                        offset: eoa,
                        length: growth,
                    })?;
                window.range = Extent::new(window.range.start(), end - window.range.start())
                    .ok_or(AggregationError::OffsetOverflow {
                        offset: window.range.start(),
                        length: end - window.range.start(),
                    })?;
                return Ok(AggregationAllocation {
                    extent: window.take(size)?,
                    eoa: end,
                    released: None,
                });
            }
        }
        // H5MF__aggr_alloc in H5MFaggr.c, HDF5 2.2.0, releases the other
        // aggregate's EOF tail after it has supplied at least one full block.
        if let Some(window) = other
            && window.range.end() == eoa
            && window.cursor - window.range.start() >= AGGREGATE_BLOCK_SIZE
            && let Some(unused) = window.unused()
        {
            eoa = unused.start();
            *other = None;
        }
        if size >= AGGREGATE_BLOCK_SIZE {
            let extent = Extent::new(eoa, size).ok_or(AggregationError::OffsetOverflow {
                offset: eoa,
                length: size,
            })?;
            return Ok(AggregationAllocation {
                extent,
                eoa: extent.end(),
                released: None,
            });
        }
        let range =
            Extent::new(eoa, AGGREGATE_BLOCK_SIZE).ok_or(AggregationError::OffsetOverflow {
                offset: eoa,
                length: AGGREGATE_BLOCK_SIZE,
            })?;
        let released = own.and_then(Aggregate::unused);
        let mut window = Aggregate { range, cursor: eoa };
        let extent = window.take(size)?;
        *own = Some(window);
        Ok(AggregationAllocation {
            extent,
            eoa: range.end(),
            released,
        })
    }

    fn minimum_eoa(&self) -> u64 {
        let Self { metadata, raw } = self;
        metadata
            .iter()
            .chain(raw)
            .map(|window| window.range.end())
            .max()
            .unwrap_or(0)
    }
}

/// Describes a reserved allocation and any unused extent released during placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AggregationAllocation {
    extent: Extent,
    eoa: u64,
    released: Option<Extent>,
}

impl AggregationAllocation {
    /// Returns the extent reserved for the request.
    pub fn extent(&self) -> Extent {
        self.extent
    }

    /// Returns the EOA required after the allocation.
    pub fn end_of_allocation(&self) -> u64 {
        self.eoa
    }

    /// Returns unused space from a replaced aggregate.
    ///
    /// This extent excludes the allocation and the unused space reserved by either aggregate.
    pub fn released(&self) -> Option<Extent> {
        self.released
    }
}

/// Describes the unused extents released when clearing the aggregates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AggregationRelease {
    metadata: Option<Extent>,
    raw: Option<Extent>,
    eoa: u64,
}

impl AggregationRelease {
    /// Returns the selected class's released interior extent, if any.
    ///
    /// Space removed from EOA is excluded.
    pub fn released(&self, class: AllocationClass) -> Option<Extent> {
        match class {
            AllocationClass::Metadata => self.metadata,
            AllocationClass::Raw => self.raw,
        }
    }

    /// Returns the EOA after releasing unused aggregate tails.
    pub fn end_of_allocation(&self) -> u64 {
        self.eoa
    }
}

/// Reports an invalid EOA or an overflowing aggregate extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AggregationError {
    /// The supplied EOA precedes the end of a reserved aggregate.
    EndOfAllocationTooSmall {
        /// The EOA supplied by the caller.
        eoa: u64,
        /// The greatest end address among the reserved aggregates.
        minimum: u64,
    },
    /// The extent's end exceeds [`u64::MAX`].
    OffsetOverflow { offset: u64, length: u64 },
}

impl core::error::Error for AggregationError {}

impl fmt::Display for AggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndOfAllocationTooSmall { eoa, minimum } => write!(
                f,
                "end of allocation {eoa} is below aggregate end {minimum}"
            ),
            Self::OffsetOverflow { offset, length } => write!(
                f,
                "aggregate range at {offset} with length {length} exceeds u64"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Aggregate {
    range: Extent,
    cursor: u64,
}

impl Aggregate {
    fn take(&mut self, size: u64) -> Result<Extent, AggregationError> {
        let Self { range, cursor } = self;
        let extent = Extent::new(*cursor, size).ok_or(AggregationError::OffsetOverflow {
            offset: *cursor,
            length: size,
        })?;
        debug_assert!(
            extent.end() <= range.end(),
            "allocation exceeds its aggregate"
        );
        *cursor = extent.end();
        Ok(extent)
    }

    fn unused(self) -> Option<Extent> {
        let Self { range, cursor } = self;
        Extent::new(cursor, range.end() - cursor)
    }
}

/// Sets the default aggregate reservation to 2,048 bytes.
///
/// `H5F_META_BLOCK_SIZE_DEF` and `H5F_SDATA_BLOCK_SIZE_DEF` in `H5Fprivate.h`, HDF5 2.2.0,
/// define this default for metadata and raw data.
pub const AGGREGATE_BLOCK_SIZE: u64 = 2048;

#[cfg(test)]
mod tests {
    use core::num::NonZeroU64;

    use alloc::vec::Vec;

    use rstest::rstest;

    use super::AGGREGATE_BLOCK_SIZE;
    use super::AggregationError;
    use super::AllocationClass;
    use super::ClassicAggregators;
    use super::Extent;

    #[rstest]
    #[case(1, 2048)]
    #[case(2047, 2048)]
    #[case(2048, 2048)]
    #[case(2049, 2049)]
    fn initial_reservation_uses_request_size_with_block_minimum(
        #[case] size: u64,
        #[case] reserved: u64,
        #[values(100, 4_294_967_296)] start: u64,
        #[values(AllocationClass::Metadata, AllocationClass::Raw)] class: AllocationClass,
    ) {
        let mut aggregates = ClassicAggregators::new();
        let allocation = aggregates
            .allocate(start, NonZeroU64::new(size).unwrap(), class)
            .unwrap();
        assert_eq!(allocation.extent(), Extent::new(start, size).unwrap());
        assert_eq!(allocation.end_of_allocation(), start + reserved);
        assert_eq!(allocation.released(), None);
        assert_eq!(
            aggregates.unused(class),
            Extent::new(start + size, reserved - size)
        );
    }

    #[test]
    fn exhausted_window_refills_at_eoa() {
        let mut aggregates = ClassicAggregators::new();
        let first = aggregates
            .allocate(100, NonZeroU64::new(64).unwrap(), AllocationClass::Raw)
            .unwrap();
        let second = aggregates
            .allocate(
                first.end_of_allocation(),
                NonZeroU64::new(1984).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        assert_eq!(second.extent(), Extent::new(164, 1984).unwrap());
        assert_eq!(aggregates.unused(AllocationClass::Raw), None);
        let third = aggregates
            .allocate(
                second.end_of_allocation(),
                NonZeroU64::new(1).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        assert_eq!(third.extent(), Extent::new(2148, 1).unwrap());
        assert_eq!(third.end_of_allocation(), 4196);
        assert_eq!(
            aggregates.unused(AllocationClass::Raw),
            Extent::new(2149, 2047)
        );
    }

    #[test]
    fn windows_suballocate_independently() {
        let mut aggregates = ClassicAggregators::new();
        let metadata = aggregates
            .allocate(100, NonZeroU64::new(64).unwrap(), AllocationClass::Metadata)
            .unwrap();
        let raw = aggregates
            .allocate(
                metadata.end_of_allocation(),
                NonZeroU64::new(64).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        let second = aggregates
            .allocate(
                raw.end_of_allocation(),
                NonZeroU64::new(64).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        assert_eq!(raw.extent(), Extent::new(2148, 64).unwrap());
        assert_eq!(second.extent(), Extent::new(164, 64).unwrap());
        assert_eq!(second.end_of_allocation(), 4196);
        assert_eq!(
            aggregates.unused(AllocationClass::Metadata),
            Extent::new(228, 1920)
        );
        assert_eq!(
            aggregates.unused(AllocationClass::Raw),
            Extent::new(2212, 1984)
        );
    }

    #[rstest]
    #[case::trailing(2148, 164, 6148)]
    #[case::interior(3000, 3000, 7000)]
    fn large_request_preserves_spare(#[case] eoa: u64, #[case] start: u64, #[case] end: u64) {
        let mut aggregates = ClassicAggregators::new();
        aggregates
            .allocate(100, NonZeroU64::new(64).unwrap(), AllocationClass::Raw)
            .unwrap();
        let allocation = aggregates
            .allocate(eoa, NonZeroU64::new(4000).unwrap(), AllocationClass::Raw)
            .unwrap();
        assert_eq!(allocation.extent(), Extent::new(start, 4000).unwrap());
        assert_eq!(allocation.end_of_allocation(), end);
        assert_eq!(allocation.released(), None);
        assert_eq!(
            aggregates.unused(AllocationClass::Raw),
            Extent::new(if eoa == 2148 { 4164 } else { 164 }, 1984)
        );
    }

    #[rstest]
    #[case::below_one_block(2047, 2148, 4196, Extent::new(2147, 1))]
    #[case::exactly_one_block(2048, 2148, 4196, None)]
    #[case::above_one_block(2049, 2149, 4197, None)]
    fn other_aggregate_tail_reuse_requires_one_consumed_block(
        #[case] used: u64,
        #[case] start: u64,
        #[case] end: u64,
        #[case] remaining: Option<Extent>,
    ) {
        let mut aggregates = ClassicAggregators::new();
        let first = aggregates
            .allocate(
                100,
                NonZeroU64::new(2047).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        let eoa = if used == 2047 {
            first.end_of_allocation()
        } else {
            aggregates
                .allocate(
                    first.end_of_allocation(),
                    NonZeroU64::new(used - 2047).unwrap(),
                    AllocationClass::Metadata,
                )
                .unwrap()
                .end_of_allocation()
        };
        let raw = aggregates
            .allocate(eoa, NonZeroU64::new(64).unwrap(), AllocationClass::Raw)
            .unwrap();
        assert_eq!(raw.extent(), Extent::new(start, 64).unwrap());
        assert_eq!(raw.end_of_allocation(), end);
        assert_eq!(aggregates.unused(AllocationClass::Metadata), remaining);
    }

    #[test]
    fn refill_returns_only_abandoned_spare() {
        let mut aggregates = ClassicAggregators::new();
        let first = aggregates
            .allocate(
                100,
                NonZeroU64::new(2000).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        let raw = aggregates
            .allocate(
                first.end_of_allocation(),
                NonZeroU64::new(64).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        let second = aggregates
            .allocate(
                raw.end_of_allocation(),
                NonZeroU64::new(64).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        assert_eq!(second.extent(), Extent::new(4196, 64).unwrap());
        assert_eq!(second.released(), Extent::new(2100, 48));
        assert_eq!(second.end_of_allocation(), 6244);
    }

    #[rstest]
    #[case::trailing(4196, 2212, None)]
    #[case::interior(5000, 5000, Extent::new(2212, 1984))]
    fn reset_releases_interior_spare_and_trims_only_eof(
        #[case] eoa: u64,
        #[case] end: u64,
        #[case] raw: Option<Extent>,
    ) {
        let mut aggregates = ClassicAggregators::new();
        let first = aggregates
            .allocate(100, NonZeroU64::new(64).unwrap(), AllocationClass::Metadata)
            .unwrap();
        aggregates
            .allocate(
                first.end_of_allocation(),
                NonZeroU64::new(64).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        let release = aggregates.reset(eoa).unwrap();
        assert_eq!(release.end_of_allocation(), end);
        assert_eq!(
            release.released(AllocationClass::Metadata),
            Extent::new(164, 1984)
        );
        assert_eq!(release.released(AllocationClass::Raw), raw);
        assert_eq!(aggregates, ClassicAggregators::new());
    }

    #[rstest]
    #[case::metadata(AllocationClass::Metadata)]
    #[case::raw(AllocationClass::Raw)]
    fn tail_release_preserves_consumed_prefix(#[case] class: AllocationClass) {
        let mut aggregates = ClassicAggregators::new();
        aggregates
            .allocate(100, NonZeroU64::new(64).unwrap(), class)
            .unwrap();
        assert_eq!(aggregates.release_tail(2148).unwrap(), 164);
        assert_eq!(aggregates, ClassicAggregators::new());
        assert_eq!(aggregates.release_tail(164).unwrap(), 164);
    }

    #[rstest]
    #[case::small(1, 2048)]
    #[case::large(4000, 4000)]
    fn overflow_preserves_state(#[case] size: u64, #[case] growth: u64) {
        let mut aggregates = ClassicAggregators::new();
        let error = aggregates
            .allocate(
                u64::MAX - 100,
                NonZeroU64::new(size).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap_err();
        assert_eq!(
            error,
            AggregationError::OffsetOverflow {
                offset: u64::MAX - 100,
                length: growth
            }
        );
        assert_eq!(aggregates, ClassicAggregators::new());
        let first = aggregates
            .allocate(
                u64::MAX - 2048,
                NonZeroU64::new(2047).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        let snapshot = aggregates;
        let error = aggregates
            .allocate(
                first.end_of_allocation(),
                NonZeroU64::new(size.max(2)).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap_err();
        assert_eq!(
            error,
            AggregationError::OffsetOverflow {
                offset: u64::MAX,
                length: growth
            }
        );
        assert_eq!(aggregates, snapshot);
    }

    #[test]
    fn failed_plan_restores_other_window_tail() {
        let mut aggregates = ClassicAggregators::new();
        let first = aggregates
            .allocate(
                u64::MAX - 4096,
                NonZeroU64::new(2047).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        let second = aggregates
            .allocate(
                first.end_of_allocation(),
                NonZeroU64::new(2).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        let snapshot = aggregates;
        let error = aggregates
            .allocate(
                second.end_of_allocation(),
                NonZeroU64::new(4000).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap_err();
        assert_eq!(
            error,
            AggregationError::OffsetOverflow {
                offset: u64::MAX - 2047,
                length: 4000
            }
        );
        assert_eq!(aggregates, snapshot);
    }

    #[test]
    fn invalid_eoa_preserves_state() {
        let mut aggregates = ClassicAggregators::new();
        aggregates
            .allocate(100, NonZeroU64::new(64).unwrap(), AllocationClass::Metadata)
            .unwrap();
        let snapshot = aggregates;
        let error = AggregationError::EndOfAllocationTooSmall {
            eoa: 2147,
            minimum: 2148,
        };
        assert_eq!(
            aggregates
                .allocate(2147, NonZeroU64::new(1).unwrap(), AllocationClass::Raw)
                .unwrap_err(),
            error
        );
        assert_eq!(aggregates, snapshot);
        assert_eq!(aggregates.release_tail(2147).unwrap_err(), error);
        assert_eq!(aggregates, snapshot);
        assert_eq!(aggregates.reset(2147).unwrap_err(), error);
        assert_eq!(aggregates, snapshot);
    }

    #[test]
    fn snapshot_restores_allocation_sequence() {
        let mut aggregates = ClassicAggregators::new();
        let first = aggregates
            .allocate(100, NonZeroU64::new(64).unwrap(), AllocationClass::Raw)
            .unwrap();
        let snapshot = aggregates;
        let allocation = aggregates
            .allocate(
                first.end_of_allocation(),
                NonZeroU64::new(4000).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        let after = aggregates;
        aggregates = snapshot;
        assert_eq!(
            aggregates
                .allocate(
                    first.end_of_allocation(),
                    NonZeroU64::new(4000).unwrap(),
                    AllocationClass::Raw
                )
                .unwrap(),
            allocation
        );
        assert_eq!(aggregates, after);
    }

    #[test]
    fn mixed_allocations_and_spares_are_disjoint() {
        let mut aggregates = ClassicAggregators::new();
        let mut eoa = 4_294_967_296;
        let mut live = Vec::<Extent>::new();
        for index in 0..1000 {
            let class = if index % 3 == 0 {
                AllocationClass::Metadata
            } else {
                AllocationClass::Raw
            };
            let size = [1, 64, 2000, 2047, 2048, 4000][index % 6];
            let allocation = aggregates
                .allocate(eoa, NonZeroU64::new(size).unwrap(), class)
                .unwrap();
            assert!(allocation.end_of_allocation() >= eoa);
            assert!(
                live.iter()
                    .all(|extent| extent.intersection(allocation.extent()).is_none())
            );
            if let Some(released) = allocation.released() {
                assert!(
                    live.iter()
                        .all(|extent| extent.intersection(released).is_none())
                );
                assert_eq!(released.intersection(allocation.extent()), None);
                assert_eq!(
                    aggregates
                        .unused(AllocationClass::Metadata)
                        .and_then(|extent| extent.intersection(released)),
                    None
                );
                assert_eq!(
                    aggregates
                        .unused(AllocationClass::Raw)
                        .and_then(|extent| extent.intersection(released)),
                    None
                );
            }
            live.push(allocation.extent());
            let metadata = aggregates.unused(AllocationClass::Metadata);
            let raw = aggregates.unused(AllocationClass::Raw);
            assert_eq!(
                metadata.and_then(|extent| raw.and_then(|raw| extent.intersection(raw))),
                None
            );
            assert!(live.iter().all(|extent| {
                metadata
                    .into_iter()
                    .chain(raw)
                    .all(|spare| extent.intersection(spare).is_none())
            }));
            assert!(
                metadata
                    .into_iter()
                    .chain(raw)
                    .all(|extent| extent.len() < AGGREGATE_BLOCK_SIZE)
            );
            eoa = allocation.end_of_allocation();
        }
    }
}
