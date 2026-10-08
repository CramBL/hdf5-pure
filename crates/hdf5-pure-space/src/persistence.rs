use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::FileSpacePageSize;

use super::extent::Extent;
use super::paged::PagedSections;

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

/// Holds the semantic sections one free-space manager should persist.
pub struct ManagerSections {
    kind: ManagerKind,
    sections: Vec<Extent>,
}

impl ManagerSections {
    /// Returns one flat manager when `sections` is non-empty.
    pub fn flat(sections: Vec<Extent>) -> Vec<Self> {
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
    /// Metadata and raw-data sections are split at page boundaries. Fragments shorter than one
    /// page remain in their typed small manager, while whole pages enter the generic-large
    /// manager. Unclassified sections stay in the generic-large manager without splitting because
    /// their page type is deliberately unknown.
    pub fn paged(sections: &PagedSections, page_size: FileSpacePageSize) -> Vec<Self> {
        let mut metadata = Vec::new();
        let mut raw = Vec::new();
        let mut generic = Vec::new();
        Self::split_typed(sections.metadata(), page_size, &mut metadata, &mut generic);
        Self::split_typed(sections.raw(), page_size, &mut raw, &mut generic);
        generic.extend(
            sections
                .unclassified()
                .iter()
                .map(|&(addr, len)| Self::section_extent(addr, len)),
        );
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

    /// Returns the semantic kind of this manager.
    pub fn kind(&self) -> ManagerKind {
        self.kind
    }

    /// Returns the sections this manager should persist.
    pub fn sections(&self) -> &[Extent] {
        &self.sections
    }

    fn split_typed(
        sections: &[(u64, u64)],
        page_size: FileSpacePageSize,
        small: &mut Vec<Extent>,
        generic: &mut Vec<Extent>,
    ) {
        let page_size = page_size.get();
        for &(addr, len) in sections {
            let extent = Self::section_extent(addr, len);
            let mut start = extent.start();
            while start < extent.end() {
                let boundary = (start / page_size + 1) * page_size;
                let end = extent.end().min(boundary);
                let piece = Extent::new(start, end - start)
                    .expect("splitting a non-empty free extent leaves a non-empty piece");
                if piece.len() < page_size {
                    small.push(piece);
                } else {
                    generic.push(piece);
                }
                start = end;
            }
        }
    }

    fn section_extent(addr: u64, len: u64) -> Extent {
        Extent::new(addr, len)
            .expect("free-list sections are non-empty and cannot cross the address-space end")
    }
}

/// Specifies the manager-tail lengths retained when releasing trailing free space.
pub const TRAILING_RESERVE_TAILS: u64 = 4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged::{PageType, PagedEdit};

    fn extent(addr: u64, len: u64) -> Extent {
        Extent::new(addr, len).unwrap()
    }

    #[test]
    fn flat_sections_make_one_generic_manager() {
        let managers = ManagerSections::flat(vec![extent(100, 20), extent(500, 30)]);
        assert_eq!(managers.len(), 1);
        assert_eq!(managers[0].kind(), ManagerKind::Flat);
        assert_eq!(
            managers[0].sections().to_vec(),
            vec![extent(100, 20), extent(500, 30)]
        );
    }

    #[test]
    fn empty_flat_sections_make_no_manager() {
        assert!(ManagerSections::flat(Vec::new()).is_empty());
    }

    #[test]
    fn paged_sections_split_at_boundaries_and_keep_unknown_space_generic() {
        let page_size = FileSpacePageSize::DEFAULT;
        let mut paged = PagedEdit::new(page_size);
        paged.seed(extent(3740, 4452), Some(PageType::Meta));
        paged.seed(extent(9000, 200), Some(PageType::Raw));
        paged.seed(extent(13_000, 300), None);
        let sections = paged.sections();

        let managers = ManagerSections::paged(&sections, page_size);
        assert_eq!(managers.len(), 3);
        assert_eq!(managers[0].kind(), ManagerKind::PagedSmallMetadata);
        assert_eq!(managers[0].sections().to_vec(), vec![extent(3740, 356)]);
        assert_eq!(managers[1].kind(), ManagerKind::PagedSmallRaw);
        assert_eq!(managers[1].sections().to_vec(), vec![extent(9000, 200)]);
        assert_eq!(managers[2].kind(), ManagerKind::PagedLargeGeneric);
        assert_eq!(
            managers[2].sections().to_vec(),
            vec![extent(4096, 4096), extent(13_000, 300)]
        );
    }

    #[test]
    fn paged_multi_page_runs_become_whole_generic_pages() {
        let page_size = FileSpacePageSize::DEFAULT;
        let mut paged = PagedEdit::new(page_size);
        paged.seed(extent(0, 3 * page_size.get()), Some(PageType::Raw));
        let sections = paged.sections();

        let managers = ManagerSections::paged(&sections, page_size);
        assert_eq!(managers.len(), 1);
        assert_eq!(managers[0].kind(), ManagerKind::PagedLargeGeneric);
        assert_eq!(
            managers[0].sections().to_vec(),
            vec![
                extent(0, page_size.get()),
                extent(page_size.get(), page_size.get()),
                extent(2 * page_size.get(), page_size.get()),
            ]
        );
    }
}
