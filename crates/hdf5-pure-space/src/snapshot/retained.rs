use crate::extent::Extent;
use crate::paged::PageType;

/// An exact recorded PAGE extent paired with its retained kind.
///
/// Produced by [`SessionSpace::retained_extents`](crate::session::SessionSpace::retained_extents).
/// The caller keeps copied descriptors associated with their original allocator state. Ownership,
/// source freshness, and reuse eligibility require separate evidence, including for other bytes
/// in the same page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainedExtent {
    extent: Extent,
    kind: RetainedKind,
}

impl RetainedExtent {
    pub(crate) fn new(extent: Extent, kind: RetainedKind) -> Self {
        Self { extent, kind }
    }

    /// Returns the exact recorded extent in stored, base-relative coordinates.
    pub fn extent(self) -> Extent {
        self.extent
    }

    /// Returns the state category recorded for this extent.
    pub fn kind(self) -> RetainedKind {
        self.kind
    }
}

/// The PAGE state in which an extent is retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedKind {
    /// Returned space whose page type is not established.
    Dead,
    /// Recorded PAGE padding with its page type.
    Padding(PageType),
    /// Persisted free sections without an established page type.
    Unclassified,
}
