/// Represents a non-empty byte range in the writable file image.
///
/// The end is exclusive and always greater than the start, so a constructed extent cannot wrap the
/// `u64` address space.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Extent {
    start: u64,
    end: u64,
}

impl Extent {
    /// Constructs `[start, start + len)`, or returns `None` for an empty or overflowing range.
    pub(crate) fn new(start: u64, len: u64) -> Option<Self> {
        let end = start.checked_add(len)?;
        (end > start).then_some(Self { start, end })
    }

    /// Returns the first byte of the extent.
    pub(crate) fn start(self) -> u64 {
        self.start
    }

    /// Returns one past the last byte of the extent.
    pub(crate) fn end(self) -> u64 {
        self.end
    }

    /// Returns the number of bytes in the extent.
    pub(crate) fn len(self) -> u64 {
        self.end - self.start
    }

    /// Returns `true` if this extent touches `other` without overlapping it.
    pub(crate) fn adjoins(self, other: Self) -> bool {
        self.end == other.start || self.start == other.end
    }

    /// Returns the overlap of this extent and `other`, or `None` when they are disjoint.
    pub(super) fn intersection(self, other: Self) -> Option<Self> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        (end > start).then_some(Self { start, end })
    }

    /// Returns the whole `align`-sized units inside this extent, or `None` when it contains none.
    ///
    /// This is the part of an extent that is provably in no unit shared with anything live: the
    /// partial edges sit in units whose other bytes may be occupied, so they are left out. Both the
    /// allocation over such interiors ([`crate::free_space::FreeList::alloc_whole_units`]) and the question of how
    /// large one is ([`crate::free_space::FreeList::largest_whole_units`]) are defined by this one function, so the
    /// two cannot disagree about what counts.
    pub(super) fn aligned_interior(self, align: u64) -> Option<Self> {
        if align == 0 {
            return None;
        }
        let start = self.start.checked_next_multiple_of(align)?;
        let end = (self.end / align) * align;
        (end > start).then_some(Self { start, end })
    }
}

#[cfg(test)]
mod tests {
    use super::Extent;

    #[test]
    fn extent_rejects_zero_length() {
        assert_eq!(Extent::new(100, 0), None);
    }

    #[test]
    fn extent_accepts_a_normal_range() {
        assert_eq!(Extent::new(100, 50), Some(Extent::new(100, 50).unwrap()));
    }

    #[test]
    fn extent_rejects_a_range_crossing_u64_max() {
        assert_eq!(Extent::new(u64::MAX - 4, 5), None);
    }

    #[test]
    fn extent_start_end_and_len_round_trip() {
        let extent = Extent::new(100, 50).unwrap();
        assert_eq!(extent.start(), 100);
        assert_eq!(extent.end(), 150);
        assert_eq!(extent.len(), 50);
    }

    #[test]
    fn extents_adjoin_at_an_exact_boundary() {
        assert!(
            Extent::new(100, 50)
                .unwrap()
                .adjoins(Extent::new(150, 25).unwrap())
        );
    }

    #[test]
    fn extents_with_a_one_byte_gap_do_not_adjoin() {
        assert!(
            !Extent::new(100, 50)
                .unwrap()
                .adjoins(Extent::new(151, 25).unwrap())
        );
    }
}
