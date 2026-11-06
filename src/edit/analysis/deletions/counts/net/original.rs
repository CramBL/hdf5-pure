//! Owned count, storage, and eligible-owner facts from one net count plan.

use std::sync::Arc;

use super::{CountStorage, NetCountPlan};
use crate::edit::WriteEngine;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::deletions::counts::ownership::CountHandoffError;
use crate::edit::analysis::deletions::counts::ownership::inventory::CountInventory;
use crate::edit::analysis::physical::claims::PhysicalClaims;
use crate::edit::analysis::physical::claims::domain::SupportedOwnerDomain;
use crate::object_graph::ObjectId;

impl NetCountPlan<'_, '_> {
    /// Collects joint count, storage, and eligible-owner facts from this plan.
    ///
    /// Preserves every represented count, including unchanged, zero, and implicit counts.
    /// Collects supported storage once and records eligible owner identities independently of
    /// allocation availability. The caller may drop the borrowed plan and phase afterward.
    /// An empty plan requires no source reads, topology discovery, or heap allocation.
    ///
    /// # Errors
    ///
    /// Returns [`CountHandoffError::Engine`] if `engine` differs from the plan's borrowed engine.
    pub(in crate::edit) fn original_owners(
        &self,
        engine: &WriteEngine,
    ) -> Result<OriginalOwnerFacts, CountHandoffError> {
        let counts = self.owned_counts(engine)?;
        Ok(OriginalOwnerFacts {
            counts,
            storage: self.collect_storage(),
            eligible_owners: self
                .removals()
                .into_iter()
                .flat_map(|removals| removals.counted_zero_objects())
                .map(|owner| owner.id())
                .collect(),
        })
    }
}

/// Owned count projections, supported storage, and eligible owners from one net count plan.
///
/// Coverage is conditional on the plan's supplied addition and removal inventories. Nonempty
/// plans retain their original phase and physical root. Empty plans retain the allocation session
/// without a phase origin. Source freshness, actual emitted edges, complete ownership, and
/// storage reuse or publication permission require separate evidence.
pub(in crate::edit) struct OriginalOwnerFacts {
    counts: CountInventory,
    storage: CountStorage,
    eligible_owners: Vec<ObjectId>,
}

impl OriginalOwnerFacts {
    /// Associates the joint facts with a matching source and supported object-header domain.
    ///
    /// Checks the allocation session, phase, root, and domain association without source reads
    /// or topology discovery. For an empty plan without a phase origin, checks the supplied source
    /// against the claims and domain. The caller establishes source freshness.
    ///
    /// # Errors
    ///
    /// Returns [`OriginalOwnerError::Session`] for a different allocation session,
    /// [`OriginalOwnerError::Phase`] for a mismatched source or claims phase,
    /// [`OriginalOwnerError::Root`] for a mismatched physical root, or
    /// [`OriginalOwnerError::Domain`] for an incompatible supported domain.
    pub(in crate::edit) fn qualify(
        self,
        engine: &WriteEngine,
        source: &ReconstructionSource,
        claims: &PhysicalClaims,
        domain: SupportedOwnerDomain,
    ) -> Result<OriginalOwnerInventory, OriginalOwnerError> {
        if !self.counts.same_session(engine) {
            return Err(OriginalOwnerError::Session);
        }
        if !claims.matches_phase(source)
            || (self.counts.root().is_some() && !self.counts.shares_phase(source))
        {
            return Err(OriginalOwnerError::Phase);
        }
        if claims.root() != Some(source.object())
            || self
                .counts
                .root()
                .is_some_and(|root| root != source.object())
        {
            return Err(OriginalOwnerError::Root);
        }
        if !domain.matches(engine, claims) {
            return Err(OriginalOwnerError::Domain);
        }
        Ok(OriginalOwnerInventory {
            facts: self,
            source: ReconstructionSource {
                identity: Arc::clone(&source.identity),
                object: source.object(),
            },
            domain,
        })
    }

    /// Borrows every count projection and its original facts in the plan's established order.
    pub(in crate::edit) fn counts(&self) -> &CountInventory {
        &self.counts
    }

    /// Borrows supported original allocations and retained reference roots from the same plan.
    pub(in crate::edit) fn storage(&self) -> &CountStorage {
        &self.storage
    }

    /// Borrows each zero-count ordinary non-root owner once in removal propagation order.
    ///
    /// Includes owners whose storage is only partially known. Eligibility is conditional on
    /// the supplied addition and removal inventories.
    pub(in crate::edit) fn eligible_owners(&self) -> &[ObjectId] {
        &self.eligible_owners
    }
}

/// Joint original-owner facts associated with a supported source object-header domain.
///
/// Retains the facts, reconstruction source, and [`SupportedOwnerDomain`] after the borrowed plan
/// and phase are dropped. Source freshness, actual owner disposition, complete heap-member
/// ownership, and storage reuse or publication permission require separate evidence.
pub(in crate::edit) struct OriginalOwnerInventory {
    facts: OriginalOwnerFacts,
    source: ReconstructionSource,
    domain: SupportedOwnerDomain,
}

impl OriginalOwnerInventory {
    /// Borrows the joint facts collected from one net count plan.
    pub(in crate::edit) fn facts(&self) -> &OriginalOwnerFacts {
        &self.facts
    }

    /// Borrows the original root and shared phase identity checked during qualification.
    pub(in crate::edit) fn source(&self) -> &ReconstructionSource {
        &self.source
    }

    /// Returns whether the allocation session, phase, root, and recorded bounds agree.
    ///
    /// The caller establishes freshness after writes, address reuse, or allocator restoration.
    pub(in crate::edit) fn matches(&self, engine: &WriteEngine, claims: &PhysicalClaims) -> bool {
        self.domain.matches(engine, claims)
    }
}

/// An incompatible association during original-owner qualification.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum OriginalOwnerError {
    /// A supported domain with a different session, phase, root, or recorded bounds.
    Domain,
    /// A source phase different from the claims or nonempty plan's phase.
    Phase,
    /// A physical source root different from the claims or nonempty plan's root.
    Root,
    /// An engine from a different allocation session.
    Session,
}
