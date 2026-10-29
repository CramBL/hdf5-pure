//! Independent interval checks for allocation publication tests.
//!
//! Callers supply controlled-hole, persisted-free, and complete reachable coverage, together with
//! independently observed allocation facts. Complete allocation ranges include index storage
//! beyond the parsed bytes and manager padding. Check source bounds and raw-section overlaps
//! before combining overlapping intervals.
//!
//! Publication labels are diagnostic. Establish allocation facts and I/O ordering from separate
//! observations, and keep these helpers outside production allocation measurement regions.

use std::error::Error as StdError;
use std::fmt;

/// A checked half-open interval in a [`u64`] address space.
///
/// Empty intervals are permitted. Both endpoints are representable as [`u64`].
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Interval {
    start: u64,
    end: u64,
}

impl Interval {
    /// Creates an interval beginning at `start` with `length` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ObservationError::RangeOverflow`] if the end exceeds [`u64::MAX`].
    pub fn new(start: u64, length: u64) -> Result<Self, ObservationError> {
        Ok(Self {
            start,
            end: start
                .checked_add(length)
                .ok_or(ObservationError::RangeOverflow { start, length })?,
        })
    }

    /// Returns the inclusive start address.
    pub fn start(self) -> u64 {
        self.start
    }

    /// Returns the exclusive end address.
    pub fn end(self) -> u64 {
        self.end
    }

    /// Returns the number of bytes in the interval.
    pub fn length(self) -> u64 {
        self.end - self.start
    }

    fn contains(self, other: Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "[{}, {})", self.start, self.end)
    }
}

/// A sorted union of disjoint, nonempty intervals.
///
/// Adjacent intervals are combined. The caller checks raw-section overlap before construction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Coverage {
    intervals: Vec<Interval>,
}

impl Coverage {
    /// Combines overlapping and adjacent intervals, discarding empty intervals.
    pub fn new(intervals: impl IntoIterator<Item = Interval>) -> Self {
        let mut intervals: Vec<_> = intervals
            .into_iter()
            .filter(|interval| interval.start != interval.end)
            .collect();
        intervals.sort_unstable();
        let mut union: Vec<Interval> = Vec::with_capacity(intervals.len());
        for interval in intervals {
            if let Some(previous) = union.last_mut()
                && interval.start <= previous.end
            {
                previous.end = previous.end.max(interval.end);
            } else {
                union.push(interval);
            }
        }
        Self { intervals: union }
    }

    /// Returns the normalized intervals in address order.
    pub fn intervals(&self) -> &[Interval] {
        &self.intervals
    }

    /// Returns whether the union covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Returns the bytes covered by `self` and excluded from `other`.
    pub fn difference(&self, other: &Self) -> Self {
        let mut remainder = Vec::new();
        let mut excluded = other.intervals.iter().peekable();
        for interval in &self.intervals {
            let mut start = interval.start;
            while let Some(next) = excluded.peek() {
                if next.end <= start {
                    excluded.next();
                } else if next.start >= interval.end {
                    break;
                } else {
                    if start < next.start {
                        remainder.push(Interval {
                            start,
                            end: next.start,
                        });
                    }
                    start = start.max(next.end).min(interval.end);
                    if start == interval.end {
                        break;
                    }
                    excluded.next();
                }
            }
            if start < interval.end {
                remainder.push(Interval {
                    start,
                    end: interval.end,
                });
            }
        }
        Self {
            intervals: remainder,
        }
    }

    /// Returns the bytes covered by both unions.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut shared = Vec::new();
        let mut left = self.intervals.iter().peekable();
        let mut right = other.intervals.iter().peekable();
        while let (Some(a), Some(b)) = (left.peek(), right.peek()) {
            let start = a.start.max(b.start);
            let end = a.end.min(b.end);
            if start < end {
                shared.push(Interval { start, end });
            }
            if a.end <= b.end {
                left.next();
            } else {
                right.next();
            }
        }
        Self { intervals: shared }
    }
}

impl fmt::Display for Coverage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.intervals)
    }
}

/// The storage represented by an independently observed allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationKind {
    /// Chunk-index metadata, including uninitialized allocated pages.
    Index,
    /// Free-space-manager metadata, including allocated padding.
    Manager,
    /// Stored chunk payload bytes.
    Payload,
}

/// The observed ownership and references to an allocation.
///
/// The caller establishes this state independently of allocator reservations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationState {
    /// Ownership is published and the complete allocation is reachable.
    Linked,
    /// Ownership is published and the allocation is unreferenced.
    Orphaned,
    /// Ownership has not been published.
    Provisional,
    /// The allocation has been released and can be reissued.
    Retired,
    /// Ownership publication has an unconfirmed outcome.
    Uncertain,
}

/// An observed complete allocation and the bytes parsed to inspect it.
///
/// Both ranges are nonempty, and the parsed range lies within the complete allocation. The caller
/// verifies the allocation against independent observations and checks its complete extent, state,
/// and physical source bounds. An unused speculative remainder is not an allocation, including
/// when observed beside an orphaned allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Allocation {
    kind: AllocationKind,
    complete: Interval,
    parsed: Interval,
    state: AllocationState,
}

impl Allocation {
    /// Records a payload whose complete extent is its independently observed stored length.
    ///
    /// # Errors
    ///
    /// Returns [`ObservationError::RangeOverflow`] if the end exceeds [`u64::MAX`], or
    /// [`ObservationError::EmptyAllocation`] if `stored_length` is zero.
    pub fn payload(
        address: u64,
        stored_length: u64,
        state: AllocationState,
    ) -> Result<Self, ObservationError> {
        let range = Interval::new(address, stored_length)?;
        Self::new(AllocationKind::Payload, range, range, state)
    }

    /// Records complete index storage separately from the bytes parsed to inspect it.
    ///
    /// `complete` includes uninitialized allocated pages.
    ///
    /// # Errors
    ///
    /// Returns [`ObservationError::EmptyAllocation`] if either range is empty, or
    /// [`ObservationError::ParseOutsideAllocation`] if `parsed` extends outside `complete`.
    pub fn index(
        complete: Interval,
        parsed: Interval,
        state: AllocationState,
    ) -> Result<Self, ObservationError> {
        Self::new(AllocationKind::Index, complete, parsed, state)
    }

    /// Records complete manager storage separately from the bytes parsed to inspect it.
    ///
    /// `complete` includes unused section-list capacity and allocated padding.
    ///
    /// # Errors
    ///
    /// Returns [`ObservationError::EmptyAllocation`] if either range is empty, or
    /// [`ObservationError::ParseOutsideAllocation`] if `parsed` extends outside `complete`.
    pub fn manager(
        complete: Interval,
        parsed: Interval,
        state: AllocationState,
    ) -> Result<Self, ObservationError> {
        Self::new(AllocationKind::Manager, complete, parsed, state)
    }

    /// Returns the observed storage category.
    pub fn kind(self) -> AllocationKind {
        self.kind
    }

    /// Returns the complete allocation, including storage outside the parsed range.
    pub fn complete(self) -> Interval {
        self.complete
    }

    /// Returns the contained range parsed to inspect the allocation.
    pub fn parsed(self) -> Interval {
        self.parsed
    }

    /// Returns the independently observed ownership state.
    pub fn state(self) -> AllocationState {
        self.state
    }

    fn new(
        kind: AllocationKind,
        complete: Interval,
        parsed: Interval,
        state: AllocationState,
    ) -> Result<Self, ObservationError> {
        if complete.length() == 0 || parsed.length() == 0 {
            return Err(ObservationError::EmptyAllocation { kind });
        }
        if !complete.contains(parsed) {
            return Err(ObservationError::ParseOutsideAllocation { complete, parsed });
        }
        Ok(Self {
            kind,
            complete,
            parsed,
            state,
        })
    }
}

/// A diagnostic label for an inspected publication point.
///
/// The caller establishes synchronization and allocation ownership independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublicationState {
    /// The observation precedes ownership publication.
    BeforeOwnership,
    /// The observation follows ownership publication.
    OwnershipPublished,
    /// The observation follows reference publication.
    ReferencesPublished,
    /// The publication outcome is unconfirmed.
    Uncertain,
}

/// Identifies the case and interruption point in a failed inspection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Context<'a> {
    /// A caller-supplied case name.
    pub case: &'a str,
    /// The caller's interruption-point index.
    pub interruption: usize,
    /// A diagnostic publication label.
    pub publication: PublicationState,
}

/// Checks unused-hole conservation and exclusion of owned storage from free coverage.
///
/// `hole` is the original controlled free coverage H, `persisted_free` is F, and `reachable`
/// is complete reachable allocation coverage L. Complete linked and orphaned allocations form A.
/// The caller establishes that the file-space policy retains every unused byte of `hole`.
/// Requires H − A ⊆ F and L ∩ F = ∅. Provisional, uncertain, and retired allocations do not count
/// toward A. Also checks active allocation overlap, state-specific references, and complete
/// coverage of linked allocations.
///
/// The caller supplies independently observed facts, including complete reachable coverage and
/// actual stored payload lengths or parsed complete metadata allocations. The caller reads the
/// source, checks its length, and observes synchronization independently. Retired storage has been
/// released by a published generation. The caller classifies session-only retirement
/// awaiting publication as owned storage.
///
/// # Examples
///
/// ```
/// use test_util::allocation_oracle::{
///     self, Allocation, AllocationState, Context, Coverage, Interval, PublicationState,
/// };
///
/// let complete = Interval::new(120, 48)?;
/// let parsed = Interval::new(120, 16)?;
/// let index = Allocation::index(complete, parsed, AllocationState::Linked)?;
/// let hole = Coverage::new([Interval::new(100, 100)?]);
/// let free = Coverage::new([Interval::new(100, 20)?, Interval::new(168, 32)?]);
/// allocation_oracle::inspect(
///     Context {
///         case: "complete index",
///         interruption: 3,
///         publication: PublicationState::ReferencesPublished,
///     },
///     &hole,
///     &free,
///     &Coverage::new([complete]),
///     &[index],
/// )?;
/// assert_eq!(hole.difference(&free).intervals(), &[complete]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Failure`] with the first observed [`Violation`] and the supplied context.
pub fn inspect<'a>(
    context: Context<'a>,
    hole: &Coverage,
    persisted_free: &Coverage,
    reachable: &Coverage,
    allocations: &[Allocation],
) -> Result<(), Failure<'a>> {
    if hole.is_empty() {
        return Err(Failure {
            context,
            violation: Violation::EmptyControlledHole,
        });
    }
    for (index, allocation) in allocations.iter().enumerate() {
        let complete = Coverage::new([allocation.complete]);
        if allocation.state != AllocationState::Retired {
            for other in &allocations[..index] {
                if other.state != AllocationState::Retired {
                    let conflicting = complete.intersection(&Coverage::new([other.complete]));
                    if !conflicting.is_empty() {
                        return Err(Failure {
                            context,
                            violation: Violation::OverlappingAllocations {
                                allocations: Box::new([*other, *allocation]),
                                conflicting,
                            },
                        });
                    }
                }
            }
        }
        let linked = complete.intersection(reachable);
        let violation = match allocation.state {
            AllocationState::Provisional | AllocationState::Uncertain if !linked.is_empty() => {
                Some(Violation::ReferenceBeforeOwnership {
                    allocation: *allocation,
                    conflicting: linked,
                })
            }
            AllocationState::Orphaned if !linked.is_empty() => Some(Violation::ReferencedOrphan {
                allocation: *allocation,
                conflicting: linked,
            }),
            AllocationState::Orphaned => {
                let conflicting = complete.intersection(persisted_free);
                (!conflicting.is_empty()).then_some(Violation::OwnedSpaceAdvertisedFree {
                    allocation: *allocation,
                    conflicting,
                })
            }
            AllocationState::Linked => {
                let missing = complete.difference(reachable);
                (!missing.is_empty()).then_some(Violation::IncompleteReachableAllocation {
                    allocation: *allocation,
                    missing,
                })
            }
            _ => None,
        };
        if let Some(violation) = violation {
            return Err(Failure { context, violation });
        }
    }
    let conflicting = reachable.intersection(persisted_free);
    if !conflicting.is_empty() {
        return Err(Failure {
            context,
            violation: Violation::LiveSpaceAdvertisedFree { conflicting },
        });
    }
    let withdrawn = Coverage::new(
        allocations
            .iter()
            .filter(|allocation| {
                matches!(
                    allocation.state,
                    AllocationState::Linked | AllocationState::Orphaned
                )
            })
            .map(|allocation| allocation.complete),
    );
    let missing = hole.difference(&withdrawn).difference(persisted_free);
    if !missing.is_empty() {
        return Err(Failure {
            context,
            violation: Violation::UnusedSpaceMissing { missing },
        });
    }
    Ok(())
}

/// A failed coverage inspection and its case context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Failure<'a> {
    /// The case, interruption point, and diagnostic publication label.
    pub context: Context<'a>,
    /// The first detected violation and its relevant ranges.
    pub violation: Violation,
}

impl fmt::Display for Failure<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "case {:?}, interruption {}, publication {:?}: {:?}",
            self.context.case, self.context.interruption, self.context.publication, self.violation
        )
    }
}

impl StdError for Failure<'_> {}

/// A violated ownership or coverage condition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Violation {
    /// The controlled hole contains zero bytes.
    EmptyControlledHole,
    /// Reachable coverage omits part of a linked allocation.
    IncompleteReachableAllocation {
        /// The linked allocation requiring complete coverage.
        allocation: Allocation,
        /// Allocated bytes missing from reachable coverage.
        missing: Coverage,
    },
    /// Persisted free coverage overlaps reachable storage.
    LiveSpaceAdvertisedFree {
        /// Bytes present in both coverages.
        conflicting: Coverage,
    },
    /// Two active allocation facts overlap.
    OverlappingAllocations {
        /// The overlapping observations, in input order.
        allocations: Box<[Allocation; 2]>,
        /// Bytes shared by the two complete allocations.
        conflicting: Coverage,
    },
    /// Persisted free coverage overlaps an orphaned owned allocation.
    OwnedSpaceAdvertisedFree {
        /// The orphaned allocation.
        allocation: Allocation,
        /// Owned bytes advertised free.
        conflicting: Coverage,
    },
    /// Reachable storage overlaps a provisional or uncertain allocation.
    ReferenceBeforeOwnership {
        /// The allocation without confirmed ownership.
        allocation: Allocation,
        /// Allocated bytes present in reachable coverage.
        conflicting: Coverage,
    },
    /// Reachable storage overlaps an allocation described as orphaned.
    ReferencedOrphan {
        /// The allocation described as unreferenced.
        allocation: Allocation,
        /// Allocated bytes present in reachable coverage.
        conflicting: Coverage,
    },
    /// Persisted free coverage omits unused controlled-hole bytes.
    UnusedSpaceMissing {
        /// Hole bytes excluded from both A and F.
        missing: Coverage,
    },
}

/// An invalid range or allocation observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationError {
    /// A complete allocation or its parsed range is empty.
    EmptyAllocation {
        /// The observed storage category.
        kind: AllocationKind,
    },
    /// A parsed range extends outside the complete allocation.
    ParseOutsideAllocation {
        /// The complete allocation range.
        complete: Interval,
        /// The parsed range requiring containment.
        parsed: Interval,
    },
    /// An interval's exclusive end exceeds the address space.
    RangeOverflow {
        /// The inclusive start.
        start: u64,
        /// The length in bytes.
        length: u64,
    },
}

impl fmt::Display for ObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl StdError for ObservationError {}

#[cfg(test)]
mod tests;
