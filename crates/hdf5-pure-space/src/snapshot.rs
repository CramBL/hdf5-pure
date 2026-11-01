//! Borrowed views of admitted free-space extents and their cached byte totals.

use alloc::vec::Vec;

use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FormatError;

use super::extent::Extent;
use super::list::FreeList;
use super::paged::{FreeClass, PageType};
use super::session::MergeExtents;

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

    /// Returns the sum of the cached reusable free-list byte totals.
    ///
    /// Excludes unclassified PAGE extents. Returns zero when reusable allocation is disabled.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    pub fn reusable_bytes(self) -> Result<u64, FormatError> {
        let first = self.lists[0].map_or(0, FreeList::total_bytes);
        let second = self.lists[1].map_or(0, FreeList::total_bytes);
        first
            .checked_add(second)
            .ok_or(FormatError::OffsetOverflow {
                offset: first,
                length: second,
            })
    }

    /// Returns the cached byte total of unclassified PAGE extents.
    ///
    /// Returns zero for another strategy.
    pub fn unclassified_bytes(self) -> u64 {
        self.lists[2].map_or(0, FreeList::total_bytes)
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
