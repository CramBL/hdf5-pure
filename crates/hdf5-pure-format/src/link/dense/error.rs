use crate::error::FormatError;
use crate::fractal_heap::write::ManagedHeapPlanError;
use crate::link::storage::LinkStorageError;

/// An error planning or serializing dense group links.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseLinkPlanError {
    /// A field or address cannot be encoded, or serialization differs from the plan.
    Format(FormatError),
    /// Managed-heap planning failed.
    Heap(ManagedHeapPlanError),
    /// The index geometry cannot represent the records.
    IndexGeometry,
    /// Layout arithmetic exceeded the range of its integer type.
    LengthOverflow,
    /// Link encoding or creation-order validation failed.
    Link(LinkStorageError),
}

impl From<FormatError> for DenseLinkPlanError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}
impl From<ManagedHeapPlanError> for DenseLinkPlanError {
    fn from(error: ManagedHeapPlanError) -> Self {
        Self::Heap(error)
    }
}
impl From<LinkStorageError> for DenseLinkPlanError {
    fn from(error: LinkStorageError) -> Self {
        Self::Link(error)
    }
}
