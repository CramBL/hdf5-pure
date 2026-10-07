use alloc::vec::Vec;

use super::extent::Extent;

/// Free space a session tracks, and the threshold rule for which freed extents join it.
///
/// An extent at least as long as the threshold is tracked, and a shorter one only where it merges
/// into tracked space. Each implementation defines where an extent merges.
pub(crate) trait TrackedSpace {
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
pub(crate) enum Release {
    /// The extent is not recorded.
    Drop,
    /// The extent is shorter than the threshold and merges into tracked space.
    Merge,
    /// The extent is at least the threshold and is recorded.
    Track,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Release, TrackedSpace};
    use crate::free_space::{Extent, FreeList};

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
        fl.free(Extent::new(0, 100).unwrap());

        assert_eq!(
            fl.release(THRESHOLD, Extent::new(500, len).unwrap(), ()),
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
            fl.free(Extent::new(addr, len).unwrap());
        }

        assert_eq!(
            fl.release(THRESHOLD, Extent::new(addr, len).unwrap(), ()),
            Release::Merge
        );
        assert_eq!(regions(&fl), expected);
    }

    #[test]
    fn adjoining_sub_threshold_extents_are_both_dropped() {
        let mut fl = FreeList::new();

        assert_eq!(
            [
                fl.release(THRESHOLD, Extent::new(0, THRESHOLD - 1).unwrap(), ()),
                fl.release(
                    THRESHOLD,
                    Extent::new(THRESHOLD - 1, THRESHOLD - 1).unwrap(),
                    ()
                ),
            ],
            [Release::Drop, Release::Drop]
        );
        assert!(fl.is_empty());
    }

    #[test]
    fn an_empty_extent_is_dropped_at_admission() {
        let mut fl = FreeList::new();
        let release = Extent::new(100, 0)
            .map(|extent| fl.release(0, extent, ()))
            .unwrap_or(Release::Drop);

        assert_eq!(release, Release::Drop);
        assert!(fl.is_empty());
    }

    const THRESHOLD: u64 = 64;
}
