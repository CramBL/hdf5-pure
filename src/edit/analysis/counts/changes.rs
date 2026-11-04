//! Borrowed and owned original facts paired with planned declared counts.

use super::CountEvidence;
use super::referenced::ReferencedCount;

pub(in crate::edit) mod owned;

/// A planned declared count paired with borrowed original facts.
///
/// The original observation or topology-bound evidence is borrowed unchanged. Source freshness,
/// ownership, retirement eligibility, and count-write permission require separate evidence.
pub(in crate::edit) struct CountChange<'plan, 'source> {
    original: OriginalCount<'plan, 'source>,
    remaining: u32,
}

impl<'plan, 'source> CountChange<'plan, 'source> {
    /// Pairs original facts with a planned declared count.
    ///
    /// The caller supplies the final count established by the originating plan.
    pub(in crate::edit::analysis) fn new(
        original: OriginalCount<'plan, 'source>,
        remaining: u32,
    ) -> Self {
        Self {
            original,
            remaining,
        }
    }

    /// Returns the borrowed original facts with their observation class.
    pub(in crate::edit) fn original(&self) -> OriginalCount<'plan, 'source> {
        self.original
    }

    /// Returns the final planned declared count.
    pub(in crate::edit) fn remaining(&self) -> u32 {
        self.remaining
    }
}

/// A borrowed original count in its established observation class.
#[derive(Clone, Copy)]
pub(in crate::edit) enum OriginalCount<'plan, 'source> {
    /// A supported referenced-target observation with optional original-reference minima.
    Referenced(&'plan ReferencedCount<'source>),
    /// Topology-bound source count evidence with its ordinary or retained subject class.
    Topology(&'plan CountEvidence<'source>),
}
