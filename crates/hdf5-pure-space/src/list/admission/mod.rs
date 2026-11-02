//! Threshold admission for canonical flat free lists.

use alloc::vec;
use alloc::vec::Vec;

use super::FreeList;
use crate::admission::AdmissionWork;
use crate::admission::OrderedReturns;
use crate::admission::Release;
use crate::admission::TrackedSpace;
use crate::extent::Extent;

impl FreeList {
    /// Admits connected runs and rebuilds the canonical free list in bulk.
    ///
    /// A run is admitted if it contains existing free space or an original return at least
    /// `threshold` bytes long. Returns the original allocations of seedless runs in address order.
    /// The caller supplies returns disjoint from each other and existing free space.
    ///
    /// Empty input leaves the list unchanged. A single return uses [`TrackedSpace::release`].
    /// Multi-allocation batches are ordered once when needed, then merged with existing extents
    /// in one sweep before canonical construction.
    ///
    /// # Panics
    ///
    /// Panics if returned allocations overlap each other. A return overlapping existing free
    /// space can also panic, including during scalar admission in debug builds.
    pub(super) fn release_batch(
        &mut self,
        threshold: u64,
        extents: impl IntoIterator<Item = (Extent, ())>,
        work: &mut AdmissionWork,
    ) -> Vec<(Extent, ())> {
        let mut extents = extents.into_iter();
        let Some(first) = extents.next() else {
            return Vec::new();
        };
        let Some(second) = extents.next() else {
            return if self.release(threshold, first.0, ()) == Release::Drop {
                vec![first]
            } else {
                Vec::new()
            };
        };
        let mut returned = vec![first, second];
        returned.extend(extents);
        // Returned allocations are disjoint. Callers retain their original boundaries.
        let mut returned = OrderedReturns::new(returned, work)
            .expect("returned allocations do not overlap")
            .into_iter()
            .peekable();
        let mut tracked = self.extents().iter().copied().peekable();
        let mut admission = FlatAdmission::default();
        while tracked.peek().is_some() || returned.peek().is_some() {
            work.visit_extent();
            if let Some(&extent) = tracked.peek()
                && returned
                    .peek()
                    .is_none_or(|&(next, ())| extent.start() < next.start())
            {
                tracked.next();
                admission.push(extent, true, false);
            } else if let Some((extent, ())) = returned.next() {
                admission.push(extent, extent.len() >= threshold, true);
            }
        }
        admission.finish_run();
        let FlatAdmission {
            admitted,
            dropped,
            run: _,
            seeded: _,
            pending_start: _,
        } = admission;
        // Distinct components have a gap, and each admitted component is one nonempty extent.
        *self = Self::from_sorted(admitted.into_iter().inspect(|_| work.integrate_extent()))
            .expect("admitted runs are sorted and disjoint");
        dropped
    }
}

/// A contiguous run under examination and the completed admission results.
///
/// Existing free space and individually eligible returns are admission seeds. The original
/// allocations of seedless runs remain separate in the dropped result.
#[derive(Default)]
struct FlatAdmission {
    /// Completed admitted runs, sorted by address.
    admitted: Vec<Extent>,
    /// Original returns from completed seedless runs and the current unseeded run.
    dropped: Vec<(Extent, ())>,
    /// The union of the current run's extents.
    run: Option<Extent>,
    /// Whether the current run contains an admission seed.
    seeded: bool,
    /// The length of the dropped result before the current run.
    pending_start: usize,
}

impl FlatAdmission {
    /// Extends the current run, completing it before a gap.
    ///
    /// Buffers original returns until the first admission seed, then discards that run's
    /// buffered originals. Retains the buffer's allocated capacity.
    ///
    /// # Panics
    ///
    /// Panics if `extent` overlaps or precedes the current run.
    fn push(&mut self, extent: Extent, seed: bool, returned: bool) {
        if self.run.is_some_and(|run| run.end() < extent.start()) {
            self.finish_run();
        }
        let Self {
            admitted: _,
            dropped,
            run,
            seeded,
            pending_start,
        } = self;
        *run = Some(match *run {
            Some(run) => {
                assert_eq!(
                    run.end(),
                    extent.start(),
                    "returned space overlaps free space"
                );
                // The sorted adjacent extents have checked endpoints enclosing their union.
                Extent::between(run.start(), extent.end()).expect("adjacent extents have a union")
            }
            None => extent,
        });
        if seed && !*seeded {
            dropped.truncate(*pending_start);
            *seeded = true;
        }
        if returned && !*seeded {
            dropped.push((extent, ()));
        }
    }

    /// Completes the current run, retaining original returns for a seedless run.
    fn finish_run(&mut self) {
        let Self {
            admitted,
            dropped,
            run,
            seeded,
            pending_start,
        } = self;
        if let Some(run) = run.take()
            && *seeded
        {
            admitted.push(run);
            dropped.truncate(*pending_start);
        }
        *seeded = false;
        *pending_start = dropped.len();
    }
}

#[cfg(test)]
mod tests;
