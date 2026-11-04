//! Count observations with unknown allocation geometry for owner tests.

use hdf5_pure_format::__private::ObjectReferenceCount;

use super::CountObservation;

/// Returns a test count observation with unknown allocation geometry.
pub(in crate::edit::analysis) fn observation_without_allocation(
    count: ObjectReferenceCount,
) -> CountObservation {
    CountObservation {
        count,
        allocation: None,
    }
}
