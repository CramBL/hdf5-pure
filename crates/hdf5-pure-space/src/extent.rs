use core::mem;
use core::num::NonZeroU64;

use hdf5_pure_core::__private::{NonEmptyRange, StoredAddressExt};
use hdf5_pure_core::{AddressRange, StoredAddress};

/// A nonempty interval of stored HDF5 addresses.
///
/// The start is inclusive and the end is exclusive. Both endpoints are [`StoredAddress`] values
/// relative to the file's address base. Construction rejects empty intervals and exclusive ends
/// beyond [`u64::MAX`].
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Extent {
    range: NonEmptyRange<StoredAddress>,
}

impl Extent {
    /// Constructs `[start, start + len)`, or returns `None` for an empty or overflowing range.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_space::__private::Extent;
    ///
    /// let start = StoredAddress::new(512);
    /// let extent = Extent::new(start, 32).unwrap();
    /// assert_eq!(extent.start(), start);
    /// assert_eq!(extent.end(), start.checked_add(32).unwrap());
    /// assert_eq!(extent.len(), 32);
    /// assert_eq!(Extent::new(start, 0), None);
    /// ```
    ///
    /// A physical position cannot be used as a stored address:
    ///
    /// ```compile_fail,E0308
    /// use hdf5_pure_core::PhysicalAddress;
    /// use hdf5_pure_space::__private::Extent;
    ///
    /// let _ = Extent::new(PhysicalAddress::new(512), 32);
    /// ```
    pub fn new(start: StoredAddress, len: u64) -> Option<Self> {
        Some(Self {
            range: NonEmptyRange::new(start, NonZeroU64::new(len)?).ok()?,
        })
    }

    /// Returns the nonempty interval from `start` to `end`, or `None` if `end` is at or before
    /// `start`.
    pub fn between(start: StoredAddress, end: StoredAddress) -> Option<Self> {
        Self::new(start, end.checked_distance_from(start)?)
    }

    /// Returns the first byte of the extent.
    pub const fn start(self) -> StoredAddress {
        self.range.start()
    }

    /// Returns one past the last byte of the extent.
    pub fn end(self) -> StoredAddress {
        self.range.end()
    }

    /// Returns the number of bytes in the extent.
    pub const fn len(self) -> u64 {
        self.range.length().get()
    }

    /// Returns whether the extent contains no bytes.
    ///
    /// Construction rejects empty ranges, so this always returns `false`.
    pub fn is_empty(self) -> bool {
        false
    }

    /// Returns `true` if this extent touches `other` without overlapping it.
    pub fn adjoins(self, other: Self) -> bool {
        self.range().adjoins(other.range())
    }

    /// Returns the overlap of this extent and `other`, or `None` when they are disjoint.
    pub(super) fn intersection(self, other: Self) -> Option<Self> {
        let common = self.range().intersection(other.range())?;
        Self::new(common.start(), common.length())
    }

    /// Returns the checked interval with the same stored endpoints and positive length.
    pub fn range(self) -> AddressRange<StoredAddress> {
        self.range.range()
    }

    /// Returns the whole `align`-sized units inside this extent, or `None` when it contains none.
    ///
    /// This is the part of an extent that is provably in no unit shared with anything live: the
    /// partial edges sit in units whose other bytes may be occupied, so they are left out. Both the
    /// allocation over such interiors ([`crate::__private::FreeList::alloc_whole_units`]) and the
    /// question of how large one is ([`crate::__private::FreeList::largest_whole_units`]) are
    /// defined by this one function, so the two cannot disagree about what counts.
    pub(super) fn aligned_interior(self, align: u64) -> Option<Self> {
        if align == 0 {
            return None;
        }
        let start = StoredAddress::new(self.start().get().checked_next_multiple_of(align)?);
        let end = StoredAddress::new((self.end().get() / align) * align);
        Self::between(start, end)
    }
}

const _: () = {
    assert!(mem::size_of::<Extent>() == 16);
    assert!(mem::size_of::<Option<Extent>>() == 16);
};

#[cfg(test)]
mod tests {
    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::{AddressRange, StoredAddress};
    use rstest::rstest;

    use super::Extent;

    #[test]
    fn extent_rejects_zero_length() {
        assert_eq!(Extent::new(StoredAddress::new(100), 0), None);
    }

    #[test]
    fn extent_rejects_a_range_crossing_u64_max() {
        assert_eq!(Extent::new(StoredAddress::new(u64::MAX - 4), 5), None);
    }

    #[rstest]
    #[case::ordinary(100, 50, 150)]
    #[case::last_byte(u64::MAX - 1, 1, u64::MAX)]
    #[case::whole_address_space(0, u64::MAX, u64::MAX)]
    fn extent_endpoints_preserve_stored_coordinates(
        #[case] start: u64,
        #[case] length: u64,
        #[case] end: u64,
    ) {
        let start = StoredAddress::new(start);
        let end = StoredAddress::new(end);
        let extent = Extent::new(start, length).unwrap();
        let range: AddressRange<StoredAddress> = extent.range();
        assert_eq!(extent.start(), start);
        assert_eq!(extent.end(), end);
        assert_eq!(extent.len(), length);
        assert!(!extent.is_empty());
        assert_eq!(range, AddressRange::new(start, length).unwrap());
        assert_eq!(Extent::between(start, end), Some(extent));
    }

    #[rstest]
    #[case::empty(100, 100)]
    #[case::empty_at_maximum(u64::MAX, u64::MAX)]
    #[case::reversed(101, 100)]
    #[case::reversed_maximum(u64::MAX, 0)]
    fn extent_endpoints_reject_empty_and_reversed_intervals(#[case] start: u64, #[case] end: u64) {
        assert_eq!(
            Extent::between(StoredAddress::new(start), StoredAddress::new(end)),
            None
        );
    }

    #[test]
    fn extents_adjoin_at_an_exact_boundary() {
        assert!(
            Extent::new(StoredAddress::new(100), 50)
                .unwrap()
                .adjoins(Extent::new(StoredAddress::new(150), 25).unwrap())
        );
    }

    #[test]
    fn extents_with_a_one_byte_gap_do_not_adjoin() {
        assert!(
            !Extent::new(StoredAddress::new(100), 50)
                .unwrap()
                .adjoins(Extent::new(StoredAddress::new(151), 25).unwrap())
        );
    }
}
