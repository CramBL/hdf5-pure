//! Complete batches of supported original-count observations for supplied copy targets.

use core::num::NonZeroU64;
use std::collections::BTreeMap;

use crate::edit::analysis::AnalysisPhase;
use crate::object_graph::ObjectId;

use super::super::added_users::AddedCopyUsers;
use super::{ReferencedCount, ReferencedCountError};

impl<'source> AnalysisPhase<'source> {
    /// Collects supported original-count observations for every supplied copy target.
    ///
    /// Calls [`Self::referenced_count`] once per target in physical object-header address order.
    /// Returns `None` at the first ambiguous or unsupported count. The accumulated batch is dropped
    /// on incompleteness or error. Existing count-cache entries are preserved.
    ///
    /// Retains one ordered target map with each positive prospective multiplicity. Cache lookup,
    /// target membership, and result insertion are separate ordered-map operations. Complete
    /// topology is optional. Returns an unassociated empty batch for an empty target set, without
    /// source reads, topology discovery, or heap allocation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::referenced_count`].
    pub(in crate::edit) fn referenced_counts(
        &mut self,
        added: &AddedCopyUsers<'source>,
    ) -> Result<Option<ReferencedCounts<'source>>, ReferencedCountError> {
        let mut targets = BTreeMap::new();
        for (object, count) in added.users() {
            let Some(observed) = self.referenced_count(added, object)? else {
                return Ok(None);
            };
            targets.insert(
                object,
                ReferencedTarget {
                    count: observed,
                    added: count,
                },
            );
        }
        Ok(Some(ReferencedCounts { targets }))
    }
}

/// Supported original-count observations for every target in one supplied copy-user union.
///
/// All observations in a nonempty batch share an engine, original root, and phase. Empty batches
/// have no association. The caller establishes completeness and same-file applicability of the
/// actual emission inventory. Original-reference minima and count-containing allocation geometry
/// remain optional in each [`ReferencedCount`]. Source freshness, allocation ownership, retirement
/// eligibility, and count-write permission require separate evidence.
pub(in crate::edit) struct ReferencedCounts<'source> {
    targets: BTreeMap<ObjectId, ReferencedTarget<'source>>,
}

impl<'source> ReferencedCounts<'source> {
    /// Consumes the batch and yields targets in physical object-header address order.
    pub(in crate::edit) fn into_targets(
        self,
    ) -> impl Iterator<Item = (ObjectId, ReferencedTarget<'source>)> {
        let Self { targets } = self;
        targets.into_iter()
    }

    /// Returns whether the batch is empty or its observations match `phase`.
    ///
    /// Uses [`ReferencedCount::matches`] for nonempty batches. Association equality is separate
    /// from source-content validity.
    pub(in crate::edit) fn matches(&self, phase: &AnalysisPhase<'_>) -> bool {
        self.targets
            .first_key_value()
            .is_none_or(|(_, target)| target.count.matches(phase))
    }

    /// Returns the associated original physical root, or `None` for an empty batch.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.targets
            .first_key_value()
            .map(|(_, target)| target.count.root())
    }

    /// Borrows the original observation and prospective multiplicity for a listed `object`.
    pub(in crate::edit) fn get(&self, object: ObjectId) -> Option<&ReferencedTarget<'source>> {
        self.targets.get(&object)
    }

    /// Borrows targets in physical object-header address order.
    pub(in crate::edit) fn targets(
        &self,
    ) -> impl Iterator<Item = (ObjectId, &ReferencedTarget<'source>)> {
        self.targets
            .iter()
            .map(|(&object, target)| (object, target))
    }
}

/// An original count observation paired with a positive prospective multiplicity.
///
/// The prospective multiplicity is the number of supplied copy emissions. Original-user minima
/// remain separate.
pub(in crate::edit) struct ReferencedTarget<'source> {
    count: ReferencedCount<'source>,
    added: NonZeroU64,
}

impl<'source> ReferencedTarget<'source> {
    /// Borrows the original count observation and its optional reference and allocation facts.
    pub(in crate::edit) fn count(&self) -> &ReferencedCount<'source> {
        &self.count
    }

    /// Returns the positive prospective multiplicity from the supplied copy emissions.
    pub(in crate::edit) fn added(&self) -> NonZeroU64 {
        self.added
    }
}
