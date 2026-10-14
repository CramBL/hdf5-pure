//! Classification of physical allocation requests.

use super::paged::PageType;

/// The purpose of one physical reservation.
///
/// A reservation has one class, including any payloads embedded in its metadata. PAGE placement
/// defaults to the corresponding [`PageType`], with explicit overrides for array chunk indexes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationClass {
    /// Metadata structures and payloads embedded in the same reservation.
    Metadata,
    /// Dataset data, global heap collections, and separately allocated huge-object payloads.
    Raw,
}

impl From<AllocationClass> for PageType {
    fn from(class: AllocationClass) -> Self {
        match class {
            AllocationClass::Metadata => Self::Meta,
            AllocationClass::Raw => Self::Raw,
        }
    }
}
