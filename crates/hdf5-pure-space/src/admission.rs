//! Original allocation boundaries and threshold admission for returned free space.

use alloc::vec;
use alloc::vec::Vec;

use super::extent::Extent;
use super::list::FreeListError;
use super::paged::FreeClass;

/// Nonempty, disjoint allocations in stored-address order, with their free-space classes.
///
/// Adjacent allocations remain separate so threshold admission uses each original length.
/// The caller establishes ownership and eligibility for retirement before returning the extents.
#[derive(Debug, Default)]
pub struct ReturnedExtents {
    entries: Vec<(Extent, FreeClass)>,
}

impl ReturnedExtents {
    /// Creates a batch of ordered, disjoint allocations and their classes.
    ///
    /// # Errors
    ///
    /// Returns [`FreeListError::Unsorted`] if an extent starts before its predecessor, or
    /// [`FreeListError::Overlap`] if an extent overlaps its predecessor in address order.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_space::__private::{Extent, FreeClass, ReturnedExtents};
    ///
    /// let first = Extent::new(StoredAddress::new(100), 32).unwrap();
    /// let second = Extent::new(first.end(), 32).unwrap();
    /// let returned = ReturnedExtents::from_sorted(vec![
    ///     (first, FreeClass::Dead),
    ///     (second, FreeClass::Dead),
    /// ])?;
    /// assert_eq!(returned.entries(), &[(first, FreeClass::Dead), (second, FreeClass::Dead)]);
    /// # Ok::<(), hdf5_pure_space::__private::FreeListError>(())
    /// ```
    pub fn from_sorted(entries: Vec<(Extent, FreeClass)>) -> Result<Self, FreeListError> {
        for pair in entries.windows(2) {
            let previous = pair[0].0;
            let next = pair[1].0;
            if next.start() < previous.start() {
                return Err(FreeListError::Unsorted { previous, next });
            }
            if next.start() < previous.end() {
                return Err(FreeListError::Overlap { previous, next });
            }
        }
        Ok(Self { entries })
    }

    /// Inserts an allocation in address order, preserving adjacent allocation boundaries.
    ///
    /// # Errors
    ///
    /// Returns [`FreeListError::Overlap`] if `extent` overlaps an existing allocation, leaving
    /// the batch unchanged.
    pub fn insert(&mut self, extent: Extent, class: FreeClass) -> Result<(), FreeListError> {
        let index = self
            .entries
            .partition_point(|&(entry, _)| entry.start() < extent.start());
        if let Some(&(previous, _)) = index
            .checked_sub(1)
            .and_then(|index| self.entries.get(index))
            && extent.start() < previous.end()
        {
            return Err(FreeListError::Overlap {
                previous,
                next: extent,
            });
        }
        if let Some(&(next, _)) = self.entries.get(index)
            && next.start() < extent.end()
        {
            return Err(FreeListError::Overlap {
                previous: extent,
                next,
            });
        }
        self.entries.insert(index, (extent, class));
        Ok(())
    }

    /// Removes an allocation and its class by position in address order.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> (Extent, FreeClass) {
        self.entries.remove(index)
    }

    /// Returns the original allocations and classes in stored-address order.
    pub fn entries(&self) -> &[(Extent, FreeClass)] {
        &self.entries
    }

    /// Returns `true` if the batch contains no allocations.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl IntoIterator for ReturnedExtents {
    type Item = (Extent, FreeClass);
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Free space a session tracks, and the threshold rule for which freed extents join it.
///
/// An extent at least as long as the threshold is tracked, and a shorter one only where it merges
/// into tracked space. Each implementation defines where an extent merges.
pub trait TrackedSpace {
    /// What a caller passes beside each extent to choose the list it joins.
    type Class: Copy;

    /// Returns `true` if `extent` merges into tracked space.
    fn merges(&self, extent: Extent) -> bool;

    /// Tracks `extent` as free in the list for `class`.
    fn track(&mut self, extent: Extent, class: Self::Class);

    /// Tracks each of `extents` that is at least `threshold` bytes long or merges into tracked
    /// space, and returns the extents it drops.
    ///
    /// A shorter extent may merge into one this call tracked, whatever the order of `extents`.
    fn release_each(
        &mut self,
        threshold: u64,
        extents: impl IntoIterator<Item = (Extent, Self::Class)>,
    ) -> Vec<(Extent, Self::Class)> {
        // The extents of at least the threshold are tracked first, and the smaller ones are offered
        // again until none merges.
        let (mut pending, eligible): (Vec<_>, Vec<_>) = extents
            .into_iter()
            .partition(|&(extent, _)| extent.len() < threshold);
        for (extent, class) in eligible {
            self.track(extent, class);
        }
        loop {
            let offered = pending.len();
            pending
                .retain(|&(extent, class)| self.release(threshold, extent, class) == Release::Drop);
            if pending.len() == offered {
                break;
            }
        }
        pending
    }

    /// Tracks `extent` if it is at least `threshold` bytes long or merges into tracked space, and
    /// returns which [`Release`] applies.
    fn release(&mut self, threshold: u64, extent: Extent, class: Self::Class) -> Release {
        // `H5MF_xfree` (`H5MF.c`, HDF5 1.14.6) adds a section of at least the threshold and merges a
        // smaller one into an adjoining tracked section through `H5FS_sect_try_merge`
        // (`H5FSsection.c`), dropping it where none adjoins.
        let release = if extent.len() >= threshold {
            Release::Track
        } else if self.merges(extent) {
            Release::Merge
        } else {
            Release::Drop
        };
        if release != Release::Drop {
            self.track(extent, class);
        }
        release
    }
}

/// What [`TrackedSpace::release`] does with a freed extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Release {
    /// The extent is not recorded.
    Drop,
    /// The extent is shorter than the threshold and merges into tracked space.
    Merge,
    /// The extent is at least the threshold and is recorded.
    Track,
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::StoredAddress;
    use rstest::rstest;

    use super::{Release, ReturnedExtents, TrackedSpace};
    use crate::extent::Extent;
    use crate::list::FreeList;
    use crate::list::FreeListError;
    use crate::paged::{FreeClass, PageType};

    #[rstest]
    #[case::empty(&[])]
    #[case::adjacent(&[(100, 20), (120, 30)])]
    #[case::separated(&[(100, 20), (150, 30)])]
    #[case::last_address(&[(u64::MAX - 2, 1), (u64::MAX - 1, 1)])]
    fn returned_extents_preserve_original_boundaries_and_classes(#[case] ranges: &[(u64, u64)]) {
        let entries: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(index, &(start, len))| {
                let class = if index == 0 {
                    FreeClass::Page(PageType::Meta)
                } else {
                    FreeClass::Dead
                };
                (Extent::new(StoredAddress::new(start), len).unwrap(), class)
            })
            .collect();
        let returned = ReturnedExtents::from_sorted(entries.clone()).unwrap();
        assert_eq!(returned.is_empty(), entries.is_empty());
        assert_eq!(returned.entries(), entries);
        assert_eq!(returned.into_iter().collect::<Vec<_>>(), entries);
    }

    #[rstest]
    #[case::out_of_order(120, 10, 100, 10, false)]
    #[case::overlap(100, 30, 120, 20, true)]
    #[case::duplicate(100, 20, 100, 20, true)]
    fn returned_extents_reject_invalid_order_and_overlap(
        #[case] first_start: u64,
        #[case] first_len: u64,
        #[case] next_start: u64,
        #[case] next_len: u64,
        #[case] overlap: bool,
    ) {
        let previous = Extent::new(StoredAddress::new(first_start), first_len).unwrap();
        let next = Extent::new(StoredAddress::new(next_start), next_len).unwrap();
        let error = ReturnedExtents::from_sorted(vec![
            (previous, FreeClass::Dead),
            (next, FreeClass::Dead),
        ])
        .unwrap_err();
        assert_eq!(
            error,
            if overlap {
                FreeListError::Overlap { previous, next }
            } else {
                FreeListError::Unsorted { previous, next }
            }
        );
    }

    #[rstest]
    #[case::before(80, 20, 0)]
    #[case::between(120, 20, 1)]
    #[case::after(160, 20, 2)]
    fn inserted_returns_keep_original_adjoining_allocations(
        #[case] start: u64,
        #[case] len: u64,
        #[case] index: usize,
    ) {
        let first = Extent::new(StoredAddress::new(100), 20).unwrap();
        let second = Extent::new(StoredAddress::new(140), 20).unwrap();
        let inserted = Extent::new(StoredAddress::new(start), len).unwrap();
        let entries = vec![
            (first, FreeClass::Dead),
            (second, FreeClass::Page(PageType::Meta)),
        ];
        let mut returned = ReturnedExtents::from_sorted(entries.clone()).unwrap();
        returned
            .insert(inserted, FreeClass::Page(PageType::Raw))
            .unwrap();
        let mut expected = entries.clone();
        expected.insert(index, (inserted, FreeClass::Page(PageType::Raw)));
        assert_eq!(returned.entries(), expected);
        assert_eq!(
            returned.remove(index),
            (inserted, FreeClass::Page(PageType::Raw))
        );
        assert_eq!(returned.entries(), entries);
    }

    #[rstest]
    #[case::overlap_previous(110, 20, 100, 20, 110, 20)]
    #[case::overlap_next(130, 20, 130, 20, 140, 20)]
    #[case::duplicate(100, 20, 100, 20, 100, 20)]
    fn overlapping_returns_leave_the_batch_unchanged(
        #[case] start: u64,
        #[case] len: u64,
        #[case] previous_start: u64,
        #[case] previous_len: u64,
        #[case] next_start: u64,
        #[case] next_len: u64,
    ) {
        let entries = vec![
            (
                Extent::new(StoredAddress::new(100), 20).unwrap(),
                FreeClass::Dead,
            ),
            (
                Extent::new(StoredAddress::new(140), 20).unwrap(),
                FreeClass::Page(PageType::Meta),
            ),
        ];
        let mut returned = ReturnedExtents::from_sorted(entries.clone()).unwrap();
        let error = returned
            .insert(
                Extent::new(StoredAddress::new(start), len).unwrap(),
                FreeClass::Page(PageType::Raw),
            )
            .unwrap_err();
        assert_eq!(
            error,
            FreeListError::Overlap {
                previous: Extent::new(StoredAddress::new(previous_start), previous_len).unwrap(),
                next: Extent::new(StoredAddress::new(next_start), next_len).unwrap(),
            }
        );
        assert_eq!(returned.entries(), entries);
    }

    /// Exposes the canonical region list as `(addr, len)` pairs for assertions.
    fn regions(fl: &FreeList) -> Vec<(u64, u64)> {
        fl.sections()
    }

    #[rstest]
    #[case::below_the_threshold(THRESHOLD - 1, Release::Drop, &[(0, 100)])]
    #[case::at_the_threshold(THRESHOLD, Release::Track, &[(0, 100), (500, THRESHOLD)])]
    #[case::above_the_threshold(
        THRESHOLD + 1,
        Release::Track,
        &[(0, 100), (500, THRESHOLD + 1)]
    )]
    fn an_isolated_extent_is_tracked_from_the_threshold_up(
        #[case] len: u64,
        #[case] release: Release,
        #[case] expected: &[(u64, u64)],
    ) {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(0), 100).unwrap());

        assert_eq!(
            fl.release(
                THRESHOLD,
                Extent::new(StoredAddress::new(500), len).unwrap(),
                ()
            ),
            release
        );
        assert_eq!(regions(&fl), expected);
    }

    #[rstest]
    #[case::below_it(&[(100, 100)], (50, 50), &[(50, 150)])]
    #[case::above_it(&[(100, 100)], (200, 50), &[(100, 150)])]
    #[case::between_two(&[(100, 100), (250, 100)], (200, 50), &[(100, 250)])]
    fn a_sub_threshold_extent_merges_into_an_adjoining_tracked_section(
        #[case] tracked: &[(u64, u64)],
        #[case] (addr, len): (u64, u64),
        #[case] expected: &[(u64, u64)],
    ) {
        let mut fl = FreeList::new();
        for &(addr, len) in tracked {
            fl.free(Extent::new(StoredAddress::new(addr), len).unwrap());
        }

        assert_eq!(
            fl.release(
                THRESHOLD,
                Extent::new(StoredAddress::new(addr), len).unwrap(),
                ()
            ),
            Release::Merge
        );
        assert_eq!(regions(&fl), expected);
    }

    #[test]
    fn adjoining_sub_threshold_extents_are_both_dropped() {
        let mut fl = FreeList::new();

        assert_eq!(
            [
                fl.release(
                    THRESHOLD,
                    Extent::new(StoredAddress::new(0), THRESHOLD - 1).unwrap(),
                    ()
                ),
                fl.release(
                    THRESHOLD,
                    Extent::new(StoredAddress::new(THRESHOLD - 1), THRESHOLD - 1).unwrap(),
                    ()
                ),
            ],
            [Release::Drop, Release::Drop]
        );
        assert!(fl.is_empty());
    }

    const THRESHOLD: u64 = 64;
}
