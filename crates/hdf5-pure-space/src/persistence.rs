use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::FileSpacePageSize;

use super::extent::Extent;
use super::paged::PagedSections;
use super::snapshot::AdmittedFreeSnapshot;

/// Identifies the free-space manager semantics the writer persists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagerKind {
    /// The generic manager of an unpaged `FSM_AGGR` file.
    Flat,
    /// The generic-large manager of a paged file.
    PagedLargeGeneric,
    /// The small metadata manager of a paged file.
    PagedSmallMetadata,
    /// The small raw-data manager of a paged file.
    PagedSmallRaw,
}

/// Holds the checked semantic extents one free-space manager should persist.
pub struct ManagerSections {
    kind: ManagerKind,
    sections: Vec<Extent>,
}

impl ManagerSections {
    /// Returns one flat manager when `sections` is non-empty.
    pub fn flat(sections: impl IntoIterator<Item = Extent>) -> Vec<Self> {
        let sections: Vec<_> = sections.into_iter().collect();
        if sections.is_empty() {
            Vec::new()
        } else {
            vec![Self {
                kind: ManagerKind::Flat,
                sections,
            }]
        }
    }

    /// Classifies PAGE sections into the managers this writer emits.
    ///
    /// Stores each aligned whole-page interior as one generic-large section. Partial edges remain
    /// within their pages in the typed small manager. Unclassified sections are stored unchanged
    /// in the generic-large manager.
    pub fn paged(sections: &PagedSections, page_size: FileSpacePageSize) -> Vec<Self> {
        Self::paged_lists(
            sections.metadata(),
            sections.raw(),
            sections.unclassified(),
            page_size,
        )
    }

    /// Projects admitted extents into the semantic managers used for persistence.
    ///
    /// Uses the snapshot's page size and classes for PAGE classification. Returns one generic manager
    /// for a nonempty flat snapshot, or no managers for disabled allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_space::__private::{
    ///     AdmittedFreeSnapshot, Extent, FreeList, ManagerKind, ManagerSections,
    /// };
    ///
    /// let extent = Extent::new(StoredAddress::new(100), 32).unwrap();
    /// let free = FreeList::from_sorted([extent])?;
    /// let managers = ManagerSections::from_snapshot(AdmittedFreeSnapshot::flat(&free));
    /// assert_eq!(managers.len(), 1);
    /// assert_eq!(managers[0].kind(), ManagerKind::Flat);
    /// assert_eq!(managers[0].sections(), [extent]);
    /// # Ok::<(), hdf5_pure_space::__private::FreeListError>(())
    /// ```
    pub fn from_snapshot(snapshot: AdmittedFreeSnapshot<'_>) -> Vec<Self> {
        let (metadata, raw, unclassified) = snapshot.lists();
        match snapshot.page_size() {
            Some(page_size) => Self::paged_lists(metadata, raw, unclassified, page_size),
            None => Self::flat(metadata.iter().copied()),
        }
    }

    /// Returns the semantic kind of this manager.
    pub fn kind(&self) -> ManagerKind {
        self.kind
    }

    /// Returns the sections this manager should persist.
    pub fn sections(&self) -> &[Extent] {
        &self.sections
    }

    fn paged_lists(
        metadata_extents: &[Extent],
        raw_extents: &[Extent],
        unclassified: &[Extent],
        page_size: FileSpacePageSize,
    ) -> Vec<Self> {
        let mut metadata = Vec::new();
        let mut raw = Vec::new();
        let mut generic = Vec::new();
        Self::split_typed(metadata_extents, page_size, &mut metadata, &mut generic);
        Self::split_typed(raw_extents, page_size, &mut raw, &mut generic);
        generic.extend_from_slice(unclassified);
        generic.sort_unstable_by_key(|extent| extent.start());
        debug_assert!(
            generic.windows(2).all(|w| w[0].start() < w[1].start()),
            "one address is free in more than one of the metadata, raw, and unclassified lists, \
             so they have stopped being disjoint"
        );

        [
            (ManagerKind::PagedSmallMetadata, metadata),
            (ManagerKind::PagedSmallRaw, raw),
            (ManagerKind::PagedLargeGeneric, generic),
        ]
        .into_iter()
        .filter_map(|(kind, sections)| (!sections.is_empty()).then_some(Self { kind, sections }))
        .collect()
    }

    fn split_typed(
        sections: &[Extent],
        page_size: FileSpacePageSize,
        small: &mut Vec<Extent>,
        generic: &mut Vec<Extent>,
    ) {
        for &extent in sections {
            let PageParts {
                leading,
                interior,
                trailing,
            } = PageParts::new(extent, page_size);
            small.extend(leading);
            generic.extend(interior);
            small.extend(trailing);
        }
    }
}

/// Holds an extent's within-page edges and aligned whole-page interior.
///
/// The nonempty pieces are ordered, disjoint, and cover the original extent. Each edge is
/// contained in one page, and the interior is page aligned at both ends.
pub(super) struct PageParts {
    leading: Option<Extent>,
    interior: Option<Extent>,
    trailing: Option<Extent>,
}

impl PageParts {
    /// Splits `extent` into at most two partial edges and one complete aligned interior.
    pub(super) fn new(extent: Extent, page_size: FileSpacePageSize) -> Self {
        if let Some(interior) = extent.aligned_interior(page_size.get()) {
            return Self {
                leading: Extent::between(extent.start(), interior.start()),
                interior: Some(interior),
                trailing: Extent::between(interior.end(), extent.end()),
            };
        }
        let boundary = extent
            .start()
            .checked_add(page_size.get() - extent.start().get() % page_size.get())
            .unwrap_or(extent.end())
            .min(extent.end());
        Self {
            leading: Extent::between(extent.start(), boundary),
            interior: None,
            trailing: Extent::between(boundary, extent.end()),
        }
    }

    /// Returns the number of nonempty pieces.
    pub(super) fn count(&self) -> usize {
        usize::from(self.leading.is_some())
            + usize::from(self.interior.is_some())
            + usize::from(self.trailing.is_some())
    }
}

/// Specifies the manager-tail lengths retained when releasing trailing free space.
pub const TRAILING_RESERVE_TAILS: u64 = 4;

#[cfg(test)]
mod tests {
    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::StoredAddress;

    use rstest::rstest;

    use super::*;
    use crate::paged::{PageType, PagedEdit};

    fn extent(addr: u64, len: u64) -> Extent {
        Extent::new(StoredAddress::new(addr), len).unwrap()
    }

    #[test]
    fn flat_sections_make_one_generic_manager() {
        let managers = ManagerSections::flat(vec![extent(100, 20), extent(500, 30)]);
        assert_eq!(managers.len(), 1);
        assert_eq!(managers[0].kind(), ManagerKind::Flat);
        assert_eq!(managers[0].sections(), [extent(100, 20), extent(500, 30)]);
    }

    #[test]
    fn empty_flat_sections_make_no_manager() {
        assert!(ManagerSections::flat(Vec::new()).is_empty());
    }

    #[test]
    fn paged_sections_keep_checked_extents_through_manager_classification() {
        let page_size = FileSpacePageSize::DEFAULT;
        let mut paged = PagedEdit::new(page_size);
        paged.seed(extent(3740, 4452), Some(PageType::Meta));
        paged.seed(extent(100, 200), Some(PageType::Meta));
        paged.seed(extent(9000, 200), Some(PageType::Raw));
        paged.seed(extent(13_000, 300), None);
        let sections = paged.sections();
        assert_eq!(sections.metadata(), [extent(100, 200), extent(3740, 4452)]);
        assert_eq!(sections.raw(), [extent(9000, 200)]);
        assert_eq!(sections.unclassified(), [extent(13_000, 300)]);

        let managers = ManagerSections::paged(&sections, page_size);
        assert_eq!(managers.len(), 3);
        assert_eq!(managers[0].kind(), ManagerKind::PagedSmallMetadata);
        assert_eq!(
            managers[0].sections(),
            [extent(100, 200), extent(3740, 356)]
        );
        assert_eq!(managers[1].kind(), ManagerKind::PagedSmallRaw);
        assert_eq!(managers[1].sections(), [extent(9000, 200)]);
        assert_eq!(managers[2].kind(), ManagerKind::PagedLargeGeneric);
        assert_eq!(
            managers[2].sections(),
            [extent(4096, 4096), extent(13_000, 300)]
        );
    }

    #[rstest]
    #[case::one_page(1)]
    #[case::three_pages(3)]
    #[case::many_pages(1 << 32)]
    fn paged_multi_page_runs_remain_one_generic_record(#[case] pages: u64) {
        let page_size = FileSpacePageSize::DEFAULT;
        let run = extent(0, pages * page_size.get());
        let mut paged = PagedEdit::new(page_size);
        paged.seed(run, Some(PageType::Raw));
        let sections = paged.sections();

        let managers = ManagerSections::paged(&sections, page_size);
        assert_eq!(managers.len(), 1);
        assert_eq!(managers[0].kind(), ManagerKind::PagedLargeGeneric);
        assert_eq!(managers[0].sections(), [run]);
    }

    #[rstest]
    #[case::aligned(4_096, 12_288, None, Some((4_096, 12_288)), None)]
    #[case::both_edges(4_000, 8_400, Some((4_000, 96)), Some((4_096, 8_192)), Some((12_288, 112)))]
    #[case::leading_edge(4_000, 8_288, Some((4_000, 96)), Some((4_096, 8_192)), None)]
    #[case::trailing_edge(4_096, 8_304, None, Some((4_096, 8_192)), Some((12_288, 112)))]
    #[case::one_partial_page(100, 100, Some((100, 100)), None, None)]
    #[case::aligned_partial_page(4_096, 100, Some((4_096, 100)), None, None)]
    #[case::crossing_partial_pages(4_000, 200, Some((4_000, 96)), None, Some((4_096, 104)))]
    #[case::last_partial_page(u64::MAX - 4_095, 4_095, Some((u64::MAX - 4_095, 4_095)), None, None)]
    #[case::last_two_partial_pages(u64::MAX - 4_100, 4_100, Some((u64::MAX - 4_100, 5)), None, Some((u64::MAX - 4_095, 4_095)))]
    fn page_parts_preserve_exact_ordered_coverage(
        #[case] start: u64,
        #[case] length: u64,
        #[case] leading: Option<(u64, u64)>,
        #[case] interior: Option<(u64, u64)>,
        #[case] trailing: Option<(u64, u64)>,
    ) {
        let original = extent(start, length);
        let parts = PageParts::new(original, FileSpacePageSize::DEFAULT);
        assert_eq!(
            parts.leading,
            leading.map(|(start, length)| extent(start, length))
        );
        assert_eq!(
            parts.interior,
            interior.map(|(start, length)| extent(start, length))
        );
        assert_eq!(
            parts.trailing,
            trailing.map(|(start, length)| extent(start, length))
        );
        let pieces: Vec<_> = [parts.leading, parts.interior, parts.trailing]
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(parts.count(), pieces.len());
        assert_eq!(pieces.first().unwrap().start(), original.start());
        assert_eq!(pieces.last().unwrap().end(), original.end());
        assert!(
            pieces
                .windows(2)
                .all(|pair| pair[0].end() == pair[1].start())
        );
        assert_eq!(pieces.iter().map(|piece| piece.len()).sum::<u64>(), length);
        for partial in [parts.leading, parts.trailing].into_iter().flatten() {
            assert_eq!(
                partial.start().get() / 4_096,
                (partial.end().get() - 1) / 4_096
            );
        }
    }

    #[rstest]
    #[case::metadata(PageType::Meta, ManagerKind::PagedSmallMetadata)]
    #[case::raw(PageType::Raw, ManagerKind::PagedSmallRaw)]
    fn typed_edges_remain_separate_from_the_generic_interior(
        #[case] ty: PageType,
        #[case] kind: ManagerKind,
    ) {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.seed(extent(4_000, 8_400), Some(ty));
        let managers = ManagerSections::from_snapshot(paged.admitted_snapshot());
        assert_eq!(managers.len(), 2);
        assert_eq!(managers[0].kind(), kind);
        assert_eq!(
            managers[0].sections(),
            [extent(4_000, 96), extent(12_288, 112)]
        );
        assert_eq!(managers[1].kind(), ManagerKind::PagedLargeGeneric);
        assert_eq!(managers[1].sections(), [extent(4_096, 8_192)]);
    }

    #[rstest]
    #[case::minimum(FileSpacePageSize::MIN)]
    #[case::non_power_of_two(FileSpacePageSize::try_from(1_000).unwrap())]
    #[case::default(FileSpacePageSize::DEFAULT)]
    #[case::maximum(FileSpacePageSize::MAX)]
    fn typed_extent_geometry_preserves_arbitrary_supported_page_sizes(
        #[case] page_size: FileSpacePageSize,
    ) {
        let page = page_size.get();
        let mut paged = PagedEdit::new(page_size);
        paged.seed(extent(page - 1, 3 * page + 2), Some(PageType::Raw));
        let managers = ManagerSections::from_snapshot(paged.admitted_snapshot());
        assert_eq!(managers.len(), 2);
        assert_eq!(managers[0].kind(), ManagerKind::PagedSmallRaw);
        assert_eq!(
            managers[0].sections(),
            [extent(page - 1, 1), extent(4 * page, 1)]
        );
        assert_eq!(managers[1].kind(), ManagerKind::PagedLargeGeneric);
        assert_eq!(managers[1].sections(), [extent(page, 3 * page)]);
    }

    #[test]
    fn generic_runs_preserve_live_gaps_and_unclassified_edges() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.seed(extent(0, 8_192), Some(PageType::Meta));
        paged.seed(extent(8_192, 200), None);
        paged.seed(extent(8_400, 100), None);
        paged.seed(extent(12_288, 8_192), Some(PageType::Raw));
        let managers = ManagerSections::from_snapshot(paged.admitted_snapshot());
        assert_eq!(managers.len(), 1);
        assert_eq!(managers[0].kind(), ManagerKind::PagedLargeGeneric);
        assert_eq!(
            managers[0].sections(),
            [
                extent(0, 8_192),
                extent(8_192, 200),
                extent(8_400, 100),
                extent(12_288, 8_192)
            ]
        );
    }
}
