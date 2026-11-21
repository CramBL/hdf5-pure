//! Ordering of original returned allocations.

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
    pub(crate) fn new(mut entries: Vec<(Extent, C)>) -> Result<Self, FreeListError> {
        if !entries
            .windows(2)
            .all(|pair| pair[0].0.start() <= pair[1].0.start())
        {
            entries.sort_unstable_by_key(|(extent, _)| extent.start());
        }
        for pair in entries.windows(2) {
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hdf5_pure_core::__private::ByteLength;
    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::StoredAddress;
    use rstest::rstest;

    use super::OrderedReturns;
    use crate::extent::Extent;
    use crate::list::FreeListError;

    #[rstest]
    #[case::sorted(false)]
    #[case::reordered(true)]
    fn ordered_returns_preserve_original_lengths_and_classes(#[case] reverse: bool) {
        let first = Extent::new(StoredAddress::new(100), ByteLength::new(8)).unwrap();
        let second = Extent::new(first.end(), ByteLength::new(8)).unwrap();
        let mut entries = vec![(first, 1), (second, 2)];
        if reverse {
            entries.reverse();
        }
        let ordered = OrderedReturns::new(entries).unwrap();
        assert_eq!(
            ordered.into_iter().collect::<alloc::vec::Vec<_>>(),
            [(first, 1), (second, 2)]
        );
    }

    #[rstest]
    #[case::sorted(false)]
    #[case::reordered(true)]
    fn ordered_returns_reject_overlapping_allocations(#[case] reverse: bool) {
        let previous = Extent::new(StoredAddress::new(100), ByteLength::new(16)).unwrap();
        let next = Extent::new(StoredAddress::new(108), ByteLength::new(8)).unwrap();
        let mut entries = vec![(previous, ()), (next, ())];
        if reverse {
            entries.reverse();
        }
        let error = match OrderedReturns::new(entries) {
            Ok(_) => panic!("overlapping returned allocations were accepted"),
            Err(error) => error,
        };
        assert_eq!(error, FreeListError::Overlap { previous, next });
    }
}
