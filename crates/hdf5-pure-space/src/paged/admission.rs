//! PAGE threshold admission with class indexes and queued original returns.

use alloc::collections::BTreeMap;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::StoredAddress;

use super::FreeClass;
use super::PageType;
use super::PagedPostFree;
use crate::admission::AdmissionWork;
use crate::admission::OrderedReturns;
use crate::admission::Release;
use crate::admission::TrackedSpace;
use crate::extent::Extent;
use crate::list::FreeList;

impl PagedPostFree {
    /// Admits original returns under PAGE adjacency rules and rebuilds each class in bulk.
    ///
    /// Preserves the class and original boundaries of dropped allocations, returning them in
    /// address order. The caller supplies returns disjoint from each other and tracked space.
    /// Existing tracked space and individually eligible originals seed admission.
    ///
    /// Empty input leaves the lists unchanged. A single return uses [`TrackedSpace::release`].
    /// Multiple returns share class indexes and one queue. Unclassified sections remain borrowed
    /// and unchanged, and can establish adjacency for returns at least one page long.
    ///
    /// # Panics
    ///
    /// Panics if original returns overlap each other. A return overlapping tracked space of its
    /// own class can also panic, including during scalar admission in debug builds.
    pub(super) fn release_batch(
        &mut self,
        threshold: u64,
        extents: impl IntoIterator<Item = (Extent, FreeClass)>,
        work: &mut AdmissionWork,
    ) -> Vec<(Extent, FreeClass)> {
        let mut extents = extents.into_iter();
        let Some(first) = extents.next() else {
            return Vec::new();
        };
        let Some(second) = extents.next() else {
            return if self.release(threshold, first.0, first.1) == Release::Drop {
                vec![first]
            } else {
                Vec::new()
            };
        };
        let mut returned = vec![first, second];
        returned.extend(extents);
        // Returned allocations retain their original lengths and are mutually disjoint.
        let returned =
            OrderedReturns::new(returned, work).expect("returned allocations do not overlap");
        let admission = PageAdmission::new(self, threshold, returned, work);
        let ([meta, raw, dead], dropped) = admission.run(work);
        self.meta = meta;
        self.raw = raw;
        self.dead = dead;
        dropped
    }
}

/// Class indexes and original-return decisions for one PAGE admission batch.
///
/// Returns retain their original extents and classes. The unclassified list is borrowed from
/// the unchanged input state, separately from the metadata, raw, and dead indexes.
struct PageAdmission<'a> {
    page_size: FileSpacePageSize,
    /// Metadata, raw, and dead indexes, in that order.
    classes: [ClassIndex; 3],
    unclassified: &'a FreeList,
    /// Original returns in address order, each with its admission state.
    returned: Vec<PendingReturn>,
    /// Positions of returns scheduled for an adjacency check.
    queue: VecDeque<usize>,
}

impl<'a> PageAdmission<'a> {
    /// Builds class indexes, admits eligible originals, and schedules shorter returns.
    ///
    /// # Panics
    ///
    /// Panics if an eligible return overlaps indexed space of its class.
    fn new(
        post: &'a PagedPostFree,
        threshold: u64,
        returned: OrderedReturns<FreeClass>,
        work: &mut AdmissionWork,
    ) -> Self {
        let PagedPostFree {
            page_size,
            meta,
            raw,
            dead,
            unclassified,
        } = post;
        let mut classes = [meta, raw, dead].map(|list| ClassIndex::new(list, work));
        let mut entries = Vec::new();
        let mut queue = VecDeque::new();
        for (extent, class) in returned {
            work.visit_extent();
            let state = if extent.len() >= ByteLength::new(threshold) {
                classes[ClassIndex::slot(class)].insert(extent, work);
                AdmissionState::Admitted
            } else {
                queue.push_back(entries.len());
                AdmissionState::Queued
            };
            entries.push(PendingReturn {
                extent,
                class,
                state,
            });
        }
        Self {
            page_size: *page_size,
            classes,
            unclassified,
            returned: entries,
            queue,
        }
    }

    /// Resolves the queue and returns canonical class lists and original dropped allocations.
    ///
    /// Considers pending returns at both ends of each changed run, including a far end whose
    /// address stayed unchanged. Dropped allocations are returned in address order.
    ///
    /// # Panics
    ///
    /// Panics if an admitted return overlaps indexed space of its class.
    fn run(self, work: &mut AdmissionWork) -> ([FreeList; 3], Vec<(Extent, FreeClass)>) {
        let Self {
            page_size,
            mut classes,
            unclassified,
            mut returned,
            mut queue,
        } = self;
        while let Some(index) = queue.pop_front() {
            debug_assert!(returned[index].state == AdmissionState::Queued);
            work.visit_queue();
            returned[index].state = AdmissionState::Pending;
            let PendingReturn {
                extent,
                class,
                state: _,
            } = returned[index];
            let unclassified_merge = if extent.len() >= ByteLength::new(page_size.get()) {
                unclassified.adjoins_observed(extent, || work.lookup())
            } else {
                false
            };
            if !unclassified_merge
                && !classes
                    .iter()
                    .any(|list| list.merges(extent, page_size.get(), work))
            {
                continue;
            }
            returned[index].state = AdmissionState::Admitted;
            let run = classes[ClassIndex::slot(class)].insert(extent, work);
            work.lookup();
            let left = returned.binary_search_by_key(&run.start(), |entry| entry.extent.end());
            work.lookup();
            let right = returned.binary_search_by_key(&run.end(), |entry| entry.extent.start());
            for neighbor in [left, right].into_iter().flatten() {
                if returned[neighbor].state == AdmissionState::Pending {
                    returned[neighbor].state = AdmissionState::Queued;
                    queue.push_back(neighbor);
                }
            }
        }
        let dropped = returned
            .into_iter()
            .filter(|entry| entry.state == AdmissionState::Pending)
            .map(|entry| (entry.extent, entry.class))
            .collect();
        (classes.map(|list| list.into_list(work)), dropped)
    }
}

/// An original allocation and class with its current admission decision.
#[derive(Clone, Copy)]
struct PendingReturn {
    extent: Extent,
    class: FreeClass,
    state: AdmissionState,
}

/// Whether a return is indexed, awaiting a decision, or scheduled for examination.
#[derive(Clone, Copy, Eq, PartialEq)]
enum AdmissionState {
    /// Included in its class index.
    Admitted,
    /// Awaiting an admission decision or a neighboring run change.
    Pending,
    /// Scheduled for an adjacency check.
    Queued,
}

/// Canonical extents of one admission class, indexed by stored start address.
struct ClassIndex {
    regions: BTreeMap<StoredAddress, Extent>,
}

impl ClassIndex {
    /// Returns the position of `class` in metadata, raw, and dead order.
    fn slot(class: FreeClass) -> usize {
        match class {
            FreeClass::Dead => 2,
            FreeClass::Page(PageType::Meta) => 0,
            FreeClass::Page(PageType::Raw) => 1,
        }
    }

    /// Copies canonical extents into an address index, observing each input and insertion.
    fn new(list: &FreeList, work: &mut AdmissionWork) -> Self {
        Self {
            regions: list
                .extents()
                .iter()
                .map(|&extent| {
                    work.index_extent();
                    work.insert_extent();
                    (extent.start(), extent)
                })
                .collect(),
        }
    }

    /// Inserts an extent and coalesces its immediate neighbors, returning the complete run.
    ///
    /// # Panics
    ///
    /// Panics if `extent` overlaps an indexed extent.
    fn insert(&mut self, extent: Extent, work: &mut AdmissionWork) -> Extent {
        let mut start = extent.start();
        let mut end = extent.end();
        work.lookup();
        if let Some((&address, &previous)) = self.regions.range(..start).next_back() {
            assert!(
                previous.end() <= start,
                "returned allocation overlaps tracked space"
            );
            if previous.end() == start {
                start = previous.start();
                self.regions.remove(&address);
                work.remove_extent();
            }
        }
        work.lookup();
        if let Some((&address, &next)) = self.regions.range(start..).next() {
            assert!(
                end <= next.start(),
                "returned allocation overlaps tracked space"
            );
            if end == next.start() {
                end = next.end();
                self.regions.remove(&address);
                work.remove_extent();
            }
        }
        // The returned allocation and its adjacent canonical extents have checked enclosing endpoints.
        let run = Extent::between(start, end).expect("adjacent extents have a nonempty union");
        self.regions.insert(start, run);
        work.insert_extent();
        run
    }

    /// Tests within-page or whole-unit adjacency for this class.
    fn merges(&self, extent: Extent, page_size: u64, work: &mut AdmissionWork) -> bool {
        work.lookup();
        let left = self
            .regions
            .range(..extent.start())
            .next_back()
            .map(|(_, &region)| region);
        work.lookup();
        let right = self.regions.get(&extent.end()).copied();
        [left, right].into_iter().flatten().any(|region| {
            if extent.len() < ByteLength::new(page_size) {
                region.adjoins_within(extent, page_size)
            } else {
                region.adjoins_whole_unit(extent, page_size)
            }
        })
    }

    /// Constructs a canonical free list, observing each output extent.
    fn into_list(self, work: &mut AdmissionWork) -> FreeList {
        let Self { regions } = self;
        // Insertion keeps each class's extents sorted, disjoint and coalesced.
        FreeList::from_sorted(regions.into_values().inspect(|_| work.integrate_extent()))
            .expect("indexed free extents are sorted and disjoint")
    }
}
