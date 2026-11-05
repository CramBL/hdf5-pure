//! Borrowed free-space views, cached totals, and exact retained PAGE descriptors.

use alloc::vec::Vec;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FormatError;

use super::extent::Extent;
use super::list::FreeList;
use super::paged::{FreeClass, PageType};
use super::persistence::PageParts;
use super::session::MergeExtents;

mod retained;

pub use retained::RetainedExtent;
pub use retained::RetainedKind;

/// Admitted free extents borrowed from one allocator state.
///
/// Extent addresses are stored coordinates, and each extent has its free-space class. For PAGE,
/// metadata and raw-data extents are reusable. Unclassified persisted extents are available through
/// [`Self::sections`]. Retired PAGE extents with unknown page type, pending padding, and extents
/// awaiting admission are excluded.
///
/// The allocator remains borrowed while the snapshot or its iterators are used. The caller
/// establishes source consistency separately.
#[derive(Clone, Copy)]
pub struct AdmittedFreeSnapshot<'a> {
    lists: [Option<&'a FreeList>; 3],
    page_size: Option<FileSpacePageSize>,
    enabled: bool,
}

impl<'a> AdmittedFreeSnapshot<'a> {
    pub(crate) fn disabled() -> Self {
        Self {
            lists: [None, None, None],
            page_size: None,
            enabled: false,
        }
    }

    /// Borrows admitted extents and the cached total from one flat free list.
    pub fn flat(free: &'a FreeList) -> Self {
        Self {
            lists: [Some(free), None, None],
            page_size: None,
            enabled: true,
        }
    }

    pub(crate) fn paged(
        page_size: FileSpacePageSize,
        metadata: &'a FreeList,
        raw: &'a FreeList,
        unclassified: &'a FreeList,
    ) -> Self {
        Self {
            lists: [Some(metadata), Some(raw), Some(unclassified)],
            page_size: Some(page_size),
            enabled: true,
        }
    }

    /// Returns whether the allocator maintains reusable free space.
    pub fn is_enabled(self) -> bool {
        self.enabled
    }

    /// Returns the page size for PAGE allocation, or `None` for another strategy.
    pub fn page_size(self) -> Option<FileSpacePageSize> {
        self.page_size
    }

    /// Returns a known PAGE class when every partial edge has matching typed free fragments.
    ///
    /// Each edge's page must contain a typed partial free fragment of the same class. Aligned
    /// whole-page interiors and unclassified free extents are excluded. Returns
    /// [`PageClassEvidence::Conflict`] when an edge page contains both classes or the two edge
    /// pages have different known classes. Otherwise, returns [`PageClassEvidence::Unknown`] for
    /// incomplete edge evidence, non-PAGE snapshots, or extents with no partial edges.
    ///
    /// # Performance
    ///
    /// Uses at most two partial edges and a binary search in each typed free list per edge. The
    /// search takes logarithmic time in each list's length and uses constant auxiliary space.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::{ByteLength, StoredAddressExt};
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{
    ///     Extent, ManagerKind, PageClassEvidence, PageType, SessionSpace,
    /// };
    ///
    /// let fragment = Extent::new(StoredAddress::new(400), ByteLength::new(16)).unwrap();
    /// let space = SessionSpace::from_persisted(
    ///     Some(FileSpaceStrategy::Page),
    ///     1,
    ///     FileSpacePageSize::MIN,
    ///     [(Some(ManagerKind::PagedSmallMetadata), fragment)],
    /// )?;
    /// let allocation = Extent::new(StoredAddress::new(100), ByteLength::new(20)).unwrap();
    /// assert_eq!(
    ///     space.admitted_snapshot().partial_page_class(allocation),
    ///     PageClassEvidence::Known(PageType::Meta),
    /// );
    /// # Ok::<(), hdf5_pure_space::__private::FreeListError>(())
    /// ```
    pub fn partial_page_class(self, extent: Extent) -> PageClassEvidence {
        let Self {
            lists: [metadata, raw, _],
            page_size,
            enabled: _,
        } = self;
        let Some(page_size) = page_size else {
            return PageClassEvidence::Unknown;
        };
        PartialPageClasses {
            metadata: metadata.map_or(&[], FreeList::extents),
            raw: raw.map_or(&[], FreeList::extents),
            page_size,
        }
        .classify(extent, || {})
    }

    /// Returns the sum of the cached reusable free-list byte totals.
    ///
    /// Excludes unclassified PAGE extents. Returns zero when reusable allocation is disabled.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    pub fn reusable_bytes(self) -> Result<ByteLength, FormatError> {
        let first = self.lists[0].map_or(ByteLength::new(0), FreeList::total_bytes);
        let second = self.lists[1].map_or(ByteLength::new(0), FreeList::total_bytes);
        first
            .checked_add(second)
            .ok_or(FormatError::OffsetOverflow {
                offset: first.get(),
                length: second.get(),
            })
    }

    /// Returns the cached byte total of unclassified PAGE extents.
    ///
    /// Returns zero for another strategy.
    pub fn unclassified_bytes(self) -> ByteLength {
        self.lists[2].map_or(ByteLength::new(0), FreeList::total_bytes)
    }

    /// Returns reusable extents in ascending stored-address order.
    ///
    /// Excludes unclassified PAGE extents.
    pub fn reusable_sections(self) -> impl Iterator<Item = Extent> + 'a {
        let (first, second) = self.reusable_lists();
        MergeExtents::new(first.iter().copied(), second.iter().copied())
    }

    /// Returns admitted extents and their classes in ascending stored-address order.
    ///
    /// Flat extents and unclassified PAGE extents have class [`FreeClass::Dead`].
    pub fn sections(self) -> impl Iterator<Item = (Extent, FreeClass)> + 'a {
        let Self {
            lists: [metadata, raw, unclassified],
            page_size,
            enabled: _,
        } = self;
        let metadata = metadata.map_or(&[][..], FreeList::extents);
        let raw = raw.map_or(&[][..], FreeList::extents);
        let first_class = if page_size.is_some() {
            FreeClass::Page(PageType::Meta)
        } else {
            FreeClass::Dead
        };
        let unclassified = unclassified.map_or(&[][..], FreeList::extents);
        let mut lists = [
            (metadata.iter().peekable(), first_class),
            (raw.iter().peekable(), FreeClass::Page(PageType::Raw)),
            (unclassified.iter().peekable(), FreeClass::Dead),
        ];
        core::iter::from_fn(move || {
            let index = lists
                .iter_mut()
                .enumerate()
                .filter_map(|(index, (extents, _))| {
                    extents.peek().map(|extent| (index, extent.start()))
                })
                .min_by_key(|&(_, start)| start)?
                .0;
            let (extents, class) = &mut lists[index];
            extents.next().map(|&extent| (extent, *class))
        })
    }

    /// Returns reusable extents merged with additional raw-data free space.
    ///
    /// Clones the raw-data free list, or the generic free list for flat allocation, and coalesces
    /// `raw_extents` into it. Under PAGE, the caller supplies extents disjoint from metadata free
    /// space. When reusable allocation is disabled, returns an empty iterator without consuming
    /// `raw_extents`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds during iteration if a raw-data extent overlaps metadata free space.
    pub fn reusable_sections_with_raw(
        self,
        raw_extents: impl IntoIterator<Item = Extent>,
    ) -> impl Iterator<Item = Extent> + 'a {
        let Self {
            lists: [first, second, _],
            page_size,
            enabled,
        } = self;
        let (first, second) = if !enabled {
            (&[][..], Vec::new())
        } else {
            let (first, second) = if page_size.is_some() {
                (first.map_or(&[][..], FreeList::extents), second)
            } else {
                (&[][..], first)
            };
            let mut free = second.cloned().unwrap_or_default();
            for extent in raw_extents {
                free.free(extent);
            }
            (first, free.into_extents())
        };
        MergeExtents::new(first.iter().copied(), second.into_iter())
    }

    pub(crate) fn lists(self) -> (&'a [Extent], &'a [Extent], &'a [Extent]) {
        let Self {
            lists: [metadata, raw, unclassified],
            page_size: _,
            enabled: _,
        } = self;
        (
            metadata.map_or(&[][..], FreeList::extents),
            raw.map_or(&[][..], FreeList::extents),
            unclassified.map_or(&[][..], FreeList::extents),
        )
    }

    fn reusable_lists(self) -> (&'a [Extent], &'a [Extent]) {
        (
            self.lists[0].map_or(&[][..], FreeList::extents),
            self.lists[1].map_or(&[][..], FreeList::extents),
        )
    }
}

/// The PAGE class evidence for an extent's partial edges.
///
/// The evidence is limited to typed partial free fragments. The caller establishes allocation
/// ownership, availability, and source consistency separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageClassEvidence {
    /// Typed fragments have conflicting classes within an edge page or between the two edges.
    Conflict,
    /// Every partial edge's page contains typed fragments of this class alone.
    Known(PageType),
    /// The available geometry and typed fragments are insufficient to determine a class.
    Unknown,
}

impl PageClassEvidence {
    fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Conflict, _) | (_, Self::Conflict) => Self::Conflict,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Known(first), Self::Known(second)) if first == second => Self::Known(first),
            (Self::Known(_), Self::Known(_)) => Self::Conflict,
        }
    }
}

#[derive(Clone, Copy)]
struct PartialPageClasses<'a> {
    metadata: &'a [Extent],
    raw: &'a [Extent],
    page_size: FileSpacePageSize,
}

impl PartialPageClasses<'_> {
    fn classify(self, extent: Extent, mut visit: impl FnMut()) -> PageClassEvidence {
        let mut edges = PageParts::new(extent, self.page_size).edges();
        let Some(first) = edges.next() else {
            return PageClassEvidence::Unknown;
        };
        let mut evidence = self.in_page(first.start().get() / self.page_size.get(), &mut visit);
        for edge in edges {
            evidence = evidence
                .combine(self.in_page(edge.start().get() / self.page_size.get(), &mut visit));
        }
        evidence
    }

    fn in_page(self, page: u64, visit: &mut impl FnMut()) -> PageClassEvidence {
        let Self {
            metadata,
            raw,
            page_size,
        } = self;
        match (
            Self::has_fragment(page_size, page, metadata, visit),
            Self::has_fragment(page_size, page, raw, visit),
        ) {
            (true, true) => PageClassEvidence::Conflict,
            (true, false) => PageClassEvidence::Known(PageType::Meta),
            (false, true) => PageClassEvidence::Known(PageType::Raw),
            (false, false) => PageClassEvidence::Unknown,
        }
    }

    fn has_fragment(
        page_size: FileSpacePageSize,
        page: u64,
        extents: &[Extent],
        visit: &mut impl FnMut(),
    ) -> bool {
        let start = page * page_size.get();
        let index = extents.partition_point(|extent| {
            visit();
            extent.end().get() <= start
        });
        let Some(&extent) = extents.get(index) else {
            return false;
        };
        visit();
        PageParts::new(extent, page_size)
            .edges()
            .any(|edge| edge.start().get() / page_size.get() == page)
    }
}

#[cfg(test)]
mod tests;
