//! Ordering of original returned allocations and observations of admission work.

use alloc::vec::Vec;

use super::Extent;
use super::FreeListError;

/// Disjoint original allocations in stored-address order, with their classes.
///
/// Adjacent allocations retain their original lengths. Allocation ownership and retirement
/// eligibility are separate caller obligations.
pub(crate) struct OrderedReturns<C> {
    entries: Vec<(Extent, C)>,
}

impl<C> OrderedReturns<C> {
    /// Orders allocations and checks for overlap while preserving adjacent boundaries.
    ///
    /// Skips sorting for input already in address order. Retains every allocation's class.
    ///
    /// # Errors
    ///
    /// Returns [`FreeListError::Overlap`] if two allocations overlap in address order.
    pub(crate) fn new(
        mut entries: Vec<(Extent, C)>,
        work: &mut AdmissionWork,
    ) -> Result<Self, FreeListError> {
        if !entries.windows(2).all(|pair| {
            work.visit_order();
            pair[0].0.start() <= pair[1].0.start()
        }) {
            entries.sort_unstable_by(|left, right| {
                work.compare();
                left.0.start().cmp(&right.0.start())
            });
        }
        for pair in entries.windows(2) {
            work.visit_order();
            if pair[0].0.end() > pair[1].0.start() {
                return Err(FreeListError::Overlap {
                    previous: pair[0].0,
                    next: pair[1].0,
                });
            }
        }
        Ok(Self { entries })
    }
}

impl<C> IntoIterator for OrderedReturns<C> {
    type Item = (Extent, C);
    type IntoIter = alloc::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Observations of ordering, traversal, and canonical construction during admission.
///
/// Counter fields are present only when compiling tests.
#[derive(Default)]
pub(crate) struct AdmissionWork {
    /// Adjacent-pair checks before sorting and during overlap checking.
    #[cfg(test)]
    pub(crate) ordered_visits: usize,
    /// Address comparisons performed by sorting unordered input.
    #[cfg(test)]
    pub(crate) sort_comparisons: usize,
    /// Existing extents and original returns consumed by the admission sweep.
    #[cfg(test)]
    pub(crate) extent_visits: usize,
    /// Admitted runs passed to canonical free-list construction.
    #[cfg(test)]
    pub(crate) integrated_extents: usize,
}

impl AdmissionWork {
    fn visit_order(&mut self) {
        #[cfg(test)]
        {
            self.ordered_visits += 1;
        }
    }

    fn compare(&mut self) {
        #[cfg(test)]
        {
            self.sort_comparisons += 1;
        }
    }

    pub(crate) fn visit_extent(&mut self) {
        #[cfg(test)]
        {
            self.extent_visits += 1;
        }
    }

    pub(crate) fn integrate_extent(&mut self) {
        #[cfg(test)]
        {
            self.integrated_extents += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::StoredAddress;
    use rstest::rstest;

    use super::AdmissionWork;
    use super::OrderedReturns;
    use crate::extent::Extent;
    use crate::list::FreeListError;

    #[rstest]
    #[case::sorted(false)]
    #[case::reordered(true)]
    fn ordered_returns_preserve_original_lengths_and_classes(#[case] reverse: bool) {
        let first = Extent::new(StoredAddress::new(100), 8).unwrap();
        let second = Extent::new(first.end(), 8).unwrap();
        let mut entries = vec![(first, 1), (second, 2)];
        if reverse {
            entries.reverse();
        }
        let ordered = OrderedReturns::new(entries, &mut AdmissionWork::default()).unwrap();
        assert_eq!(
            ordered.into_iter().collect::<alloc::vec::Vec<_>>(),
            [(first, 1), (second, 2)]
        );
    }

    #[rstest]
    #[case::sorted(false)]
    #[case::reordered(true)]
    fn ordered_returns_reject_overlapping_allocations(#[case] reverse: bool) {
        let previous = Extent::new(StoredAddress::new(100), 16).unwrap();
        let next = Extent::new(StoredAddress::new(108), 8).unwrap();
        let mut entries = vec![(previous, ()), (next, ())];
        if reverse {
            entries.reverse();
        }
        let error = match OrderedReturns::new(entries, &mut AdmissionWork::default()) {
            Ok(_) => panic!("overlapping returned allocations were accepted"),
            Err(error) => error,
        };
        assert_eq!(error, FreeListError::Overlap { previous, next });
    }
}
