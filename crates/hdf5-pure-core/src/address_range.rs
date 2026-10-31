//! Checked byte intervals with endpoints in one address coordinate system.

use core::num::NonZeroU64;

use crate::address::Address;
use crate::error::FormatError;

/// A checked half-open interval in one address coordinate system.
///
/// The start is inclusive and the end is exclusive. The length may be zero, and both endpoints
/// fit in [`u64`]. Only [`StoredAddress`](crate::StoredAddress) and
/// [`PhysicalAddress`](crate::PhysicalAddress) are supported as address types.
///
/// Ranges are ordered first by start and then by end. Use [`precedes`](Self::precedes) to compare
/// the positions of disjoint ranges.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{AddressRange, PhysicalAddress};
///
/// let range = AddressRange::new(PhysicalAddress::new(512), 96)?;
/// assert_eq!(range.start(), PhysicalAddress::new(512));
/// assert_eq!(range.end(), PhysicalAddress::new(608));
/// assert_eq!(range.length(), 96);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
///
/// Ranges in different coordinate systems cannot be compared:
///
/// ```compile_fail,E0308
/// use hdf5_pure_core::{AddressRange, PhysicalAddress, StoredAddress};
///
/// fn compare(stored: AddressRange<StoredAddress>, physical: AddressRange<PhysicalAddress>) {
///     let _ = stored == physical;
/// }
/// ```
///
/// Geometric operations also require matching coordinate types:
///
/// ```compile_fail,E0308
/// use hdf5_pure_core::{AddressRange, PhysicalAddress, StoredAddress};
///
/// fn intersect(stored: AddressRange<StoredAddress>, physical: AddressRange<PhysicalAddress>) {
///     let _ = stored.intersection(physical);
/// }
/// ```
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AddressRange<A> {
    start: A,
    end: A,
}

impl<A> AddressRange<A> {
    /// Creates an interval beginning at `start` and spanning `length` bytes.
    ///
    /// The interval is empty when `length` is zero, including at the maximum address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the end exceeds [`u64::MAX`].
    pub fn new(start: A, length: u64) -> Result<Self, FormatError>
    where
        A: Address,
    {
        Ok(Self {
            start,
            end: start.checked_advance(length)?,
        })
    }

    /// Returns the inclusive start address in this interval's coordinate system.
    pub const fn start(self) -> A
    where
        A: Copy,
    {
        self.start
    }

    /// Returns the exclusive end address in this interval's coordinate system.
    pub const fn end(self) -> A
    where
        A: Copy,
    {
        self.end
    }

    /// Returns the number of bytes between the endpoints.
    pub fn length(self) -> u64
    where
        A: Address,
    {
        // Construction establishes that the end is at or after the start.
        self.end
            .distance_from(self.start)
            .expect("a checked range has ordered endpoints")
    }

    /// Returns `true` if the endpoints are equal.
    pub fn is_empty(self) -> bool
    where
        A: Address,
    {
        self.start == self.end
    }

    /// Returns `true` if both endpoints of `required` lie within this interval's endpoints.
    ///
    /// An empty interval at either endpoint is contained.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{AddressRange, PhysicalAddress};
    ///
    /// let range = AddressRange::new(PhysicalAddress::new(512), 96)?;
    /// let at_end = AddressRange::new(range.end(), 0)?;
    /// assert!(range.contains(at_end));
    /// assert!(!range.overlaps(at_end));
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn contains(self, required: Self) -> bool
    where
        A: Address,
    {
        self.start <= required.start && required.end <= self.end
    }

    /// Returns `true` if the intervals have at least one byte in common.
    ///
    /// Empty and adjacent intervals do not overlap.
    pub fn overlaps(self, other: Self) -> bool
    where
        A: Address,
    {
        !self.is_empty() && !other.is_empty() && self.start < other.end && other.start < self.end
    }

    /// Returns the nonempty interval common to both ranges, or `None` if they do not overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{AddressRange, PhysicalAddress};
    ///
    /// let left = AddressRange::new(PhysicalAddress::new(512), 96)?;
    /// let right = AddressRange::new(PhysicalAddress::new(576), 64)?;
    /// let common = left.intersection(right).unwrap();
    /// assert_eq!(common, AddressRange::new(PhysicalAddress::new(576), 32)?);
    /// let adjacent = AddressRange::new(left.end(), 16)?;
    /// assert_eq!(left.intersection(adjacent), None);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn intersection(self, other: Self) -> Option<Self>
    where
        A: Address,
    {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        (start < end).then_some(Self { start, end })
    }

    /// Returns `true` if either interval's end equals the other's start.
    ///
    /// An empty interval adjoins any interval with an endpoint at that position, including
    /// another empty interval.
    pub fn adjoins(self, other: Self) -> bool
    where
        A: Address,
    {
        self.end == other.start || other.end == self.start
    }

    /// Returns `true` if this interval ends at or before `other` starts.
    ///
    /// Adjacent intervals satisfy this relation. Two empty intervals at the same position each
    /// precede the other.
    pub fn precedes(self, other: Self) -> bool
    where
        A: Address,
    {
        self.end <= other.start
    }
}

/// A checked interval with a positive byte length and a representable exclusive end.
///
/// Only [`StoredAddress`](crate::StoredAddress) and [`PhysicalAddress`](crate::PhysicalAddress)
/// are supported as address types. Workspace crates import this type from the hidden `__private`
/// module, outside the supported public API.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NonEmptyRange<A> {
    range: AddressRange<A>,
    length: NonZeroU64,
}

impl<A> NonEmptyRange<A> {
    /// Creates an interval beginning at `start` with the supplied positive byte length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the end exceeds [`u64::MAX`].
    pub fn new(start: A, length: NonZeroU64) -> Result<Self, FormatError>
    where
        A: Address,
    {
        Ok(Self {
            range: AddressRange::new(start, length.get())?,
            length,
        })
    }

    /// Returns the inclusive start address in this interval's coordinate system.
    pub const fn start(self) -> A
    where
        A: Copy,
    {
        self.range.start()
    }

    /// Returns the exclusive end address in this interval's coordinate system.
    pub const fn end(self) -> A
    where
        A: Copy,
    {
        self.range.end()
    }

    /// Returns the positive number of bytes between the endpoints.
    pub const fn length(&self) -> NonZeroU64 {
        self.length
    }

    /// Returns the interval with the same endpoints and coordinate type.
    pub const fn range(self) -> AddressRange<A>
    where
        A: Copy,
    {
        self.range
    }
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;

    use rstest::rstest;

    use super::*;
    use crate::address::{PhysicalAddress, StoredAddress, StoredAddressExt};

    #[rstest]
    #[case::stored(StoredAddress::new)]
    #[case::physical(PhysicalAddress::new)]
    fn nonempty_ranges_preserve_positive_lengths<A: Address + Debug>(
        #[case] address: fn(u64) -> A,
        #[values((512, 96), (u64::MAX - 1, 1), (0, u64::MAX))] bounds: (u64, u64),
    ) {
        let (start, length) = bounds;
        let length = NonZeroU64::new(length).unwrap();
        let range = NonEmptyRange::new(address(start), length).unwrap();
        assert_eq!(range.start(), address(start));
        assert_eq!(range.end(), address(start + length.get()));
        assert_eq!(range.length(), length);
        assert_eq!(
            range.range(),
            AddressRange::new(address(start), length.get()).unwrap()
        );
        assert!(!range.range().is_empty());
    }

    #[rstest]
    #[case::stored(StoredAddress::new)]
    #[case::physical(PhysicalAddress::new)]
    fn nonempty_ranges_reject_end_overflow<A: Address + Debug>(#[case] address: fn(u64) -> A) {
        assert_eq!(
            NonEmptyRange::new(address(u64::MAX), NonZeroU64::new(1).unwrap()).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1,
            }
        );
    }
}
