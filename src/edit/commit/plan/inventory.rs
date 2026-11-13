//! Original count, ownership, and storage observations paired with prepared tree edits.
//!
//! The editor derives observations from the owned application inputs and retains them with the
//! placed tree only after every placement and value write succeeds.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;

use self::dispositions::{AppliedDispositions, ReconstructionDispositions};
use super::{CommitPlan, Node, OverwritePayloads, PreparedCommit, PreparedTree};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::CollectionCandidates;
use crate::edit::analysis::counts::{CopyUsersError, CopyUsersUnion};
use crate::edit::analysis::heap::qualification::OriginalCollections;
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::analysis::{
    AnalysisPhase, CountHandoffError, NetCountError, OriginalOwnerError, OriginalOwnerFacts,
    OriginalOwnerInventory, ReconstructionSource,
};
use crate::edit::commit::publish::AppliedTree;
use crate::edit::staging::StagedEdits;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

pub(in crate::edit::commit) mod dispositions;
pub(in crate::edit::commit) mod members;
mod outcomes;
pub(in crate::edit::commit) mod prospective;
pub(in crate::edit::commit) mod storage;

/// Owns tree edits and their original count, ownership, and collection observations.
///
/// The editor collects the observations in the same source phase as the tree plan. The caller
/// preserves source freshness until application. The allocation session is checked separately.
/// Original storage exclusions, captured reclamation spans, and owned collection qualification
/// and framing stay paired with these inputs.
pub(in crate::edit::commit) struct PreparedEditInventory {
    prepared: PreparedCommit,
    original: OriginalEdits,
    session: AllocationSession,
    dispositions: ReconstructionDispositions,
    storage: storage::OriginalStorage,
    collections: OriginalCollections,
    reclamation: Result<Vec<(FileRange, FreeClass)>, Error>,
}

impl PreparedEditInventory {
    /// Prepares staged edits and captures original observations when heap candidates are present.
    ///
    /// Returns the original staged set on planning or count-preparation failure. Returns
    /// candidate-free tree and in-place plans without an inventory. The planner prepares counts
    /// for deletions and same-file copies. Retains additional count or ownership qualification
    /// outcomes in [`OriginalEdits`].
    pub(in crate::edit::commit) fn prepare(
        engine: &mut WriteEngine,
        staged: StagedEdits,
    ) -> InventoryPreparation {
        Self::prepare_with_capture(engine, staged, InventoryCapture::Candidates)
    }

    /// Captures original observations for every prepared tree plan.
    ///
    /// Supports evidence tests independently of candidate discovery. Preserves the numeric
    /// in-place early return.
    #[cfg(test)]
    pub(in crate::edit::commit) fn prepare_observed(
        engine: &mut WriteEngine,
        staged: StagedEdits,
    ) -> InventoryPreparation {
        Self::prepare_with_capture(engine, staged, InventoryCapture::Always)
    }

    /// Retains rejected staging and pairs captured observations with their application inputs.
    fn prepare_with_capture(
        engine: &mut WriteEngine,
        staged: StagedEdits,
        capture: InventoryCapture,
    ) -> InventoryPreparation {
        match engine.plan_commit_with_inventory(&staged, capture) {
            Ok((mut plan, original)) => {
                let (staged, writes) = staged.into_apply();
                let mut overwrites = OverwritePayloads { writes };
                plan.release_chunk_inputs(&mut overwrites);
                let prepared = PreparedCommit {
                    plan,
                    staged,
                    overwrites,
                };
                match original {
                    Some(OriginalPreparation {
                        original,
                        dispositions,
                        storage,
                        collections,
                        reclamation,
                    }) => InventoryPreparation::Tree(Self {
                        prepared,
                        original,
                        dispositions,
                        storage,
                        collections,
                        reclamation,
                        session: engine.context.allocations.session(),
                    }),
                    None => InventoryPreparation::Direct(prepared),
                }
            }
            Err(error) => InventoryPreparation::Rejected { staged, error },
        }
    }

    /// Places the owned edits and pairs the result with their original observations.
    ///
    /// Consumes the application inputs on success or failure. The caller handles rollback
    /// after a failure and publication after success. Predecessor classifications and placement
    /// reconciliation outcomes are retained independently of count and ownership qualification.
    /// Consumes captured reclamation spans without repeating their original collection.
    /// Uses an empty reclamation batch for a captured [`Error::EditUnsupported`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] before placement if the allocation session differs
    /// or the prepared plan is not a tree. Returns captured reclamation errors other than
    /// [`Error::EditUnsupported`] before placement. Returns the errors of
    /// [`WriteEngine::apply_tree`].
    pub(in crate::edit::commit) fn apply(
        self,
        engine: &mut WriteEngine,
    ) -> Result<AppliedEditInventory, Error> {
        let Self {
            prepared,
            original,
            session,
            mut dispositions,
            storage,
            collections,
            reclamation,
        } = self;
        if !session.same(&engine.context.allocations.session()) {
            return Err(FormatError::Internal(
                "a prepared edit inventory belongs to another allocation session".into(),
            )
            .into());
        }
        let PreparedCommit {
            plan,
            staged,
            mut overwrites,
        } = prepared;
        let CommitPlan::Tree(tree) = plan else {
            return Err(
                FormatError::Internal("a prepared edit inventory has no tree".into()).into(),
            );
        };
        let reclamation = match reclamation {
            Ok(reclamation) => reclamation,
            Err(Error::EditUnsupported(_)) => Vec::new(),
            Err(error) => return Err(error),
        };
        let tree = engine.apply_tree_recording(
            &mut overwrites,
            tree,
            staged,
            Some(&mut dispositions),
            Some(reclamation),
        )?;
        let dispositions = dispositions.finish(&tree);
        Ok(AppliedEditInventory {
            session,
            tree,
            original,
            dispositions,
            storage,
            collections,
        })
    }

    /// Borrows every original predecessor's classification before placement.
    pub(in crate::edit::commit) fn dispositions(&self) -> &ReconstructionDispositions {
        &self.dispositions
    }

    /// Borrows original candidates, qualification, and framing collected before placement.
    pub(in crate::edit::commit) fn collections(&self) -> &OriginalCollections {
        &self.collections
    }

    /// Borrows the original observations collected from these prepared inputs.
    pub(in crate::edit::commit) fn original(&self) -> &OriginalEdits {
        &self.original
    }
}

/// The planning outcome, with owned application inputs or rejected staging.
#[allow(
    clippy::large_enum_variant,
    reason = "preparation returns its owned inputs without an additional allocation"
)]
pub(in crate::edit::commit) enum InventoryPreparation {
    /// Prepared in-place or tree edits without captured original-owner inventory.
    Direct(PreparedCommit),
    /// The original staged set and the planning error.
    Rejected { staged: StagedEdits, error: Error },
    /// Prepared tree edits paired with their original observations.
    Tree(PreparedEditInventory),
}

/// Selects original-observation capture during commit planning.
#[derive(Clone, Copy, Debug)]
pub(super) enum InventoryCapture {
    /// Captures observations for every tree plan.
    #[cfg(test)]
    Always,
    /// Captures observations when allocation candidates or retained observations are present.
    Candidates,
    /// Omits original-observation capture.
    Omit,
}

impl InventoryCapture {
    pub(super) fn includes(self, candidates: &CollectionCandidates) -> bool {
        match self {
            #[cfg(test)]
            Self::Always => true,
            Self::Candidates => !candidates.is_empty(),
            Self::Omit => false,
        }
    }
}

/// A successfully placed tree paired with its prepared inputs' original observations.
///
/// The editor constructs this pair after [`WriteEngine::apply_tree`] succeeds. Count writes,
/// storage reuse, and publication require separate permission. Original storage exclusions
/// remain paired with the placed tree.
pub(in crate::edit::commit) struct AppliedEditInventory {
    /// The preparation session checked before placement.
    session: AllocationSession,
    tree: AppliedTree,
    original: OriginalEdits,
    dispositions: AppliedDispositions,
    storage: storage::OriginalStorage,
    collections: OriginalCollections,
}

impl AppliedEditInventory {
    /// Borrows original candidates, qualification, and framing retained through application.
    pub(in crate::edit::commit) fn collections(&self) -> &OriginalCollections {
        &self.collections
    }

    /// Borrows original predecessor classifications and the placement reconciliation outcome.
    pub(in crate::edit::commit) fn dispositions(&self) -> &AppliedDispositions {
        &self.dispositions
    }

    /// Borrows the placed root, relocations, and retirement inputs before publication.
    pub(in crate::edit::commit) fn tree(&self) -> &AppliedTree {
        &self.tree
    }

    /// Borrows the original observations retained through successful application.
    pub(in crate::edit::commit) fn original(&self) -> &OriginalEdits {
        &self.original
    }

    /// Returns the placed tree, dropping its paired observations, dispositions, and storage evidence.
    pub(in crate::edit::commit) fn into_tree(self) -> AppliedTree {
        self.tree
    }
}

/// Original observations, storage exclusions, and reclamation spans for one tree plan.
pub(super) struct OriginalPreparation {
    original: OriginalEdits,
    dispositions: ReconstructionDispositions,
    storage: storage::OriginalStorage,
    collections: OriginalCollections,
    reclamation: Result<Vec<(FileRange, FreeClass)>, Error>,
}

impl OriginalPreparation {
    /// Collects original observations and reclamation spans for the prepared tree.
    ///
    /// Moves deletion spans out of the tree into the captured reclamation input. Storage
    /// exclusions are derived from these same observations and reclamation spans. Moves collection
    /// candidates and cached framing into the inventory after storage collection. Retains source,
    /// qualification, framing, and reclamation failures as independent outcomes.
    /// When owner qualification is unavailable, withholds generated retirements through
    /// [`storage::OriginalStorage::withhold_unqualified_reclamation`].
    pub(super) fn collect(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        tree: &mut PreparedTree,
        staged: &StagedEdits,
    ) -> Self {
        let original = OriginalEdits::collect(engine, phase, tree, staged);
        let dispositions = ReconstructionDispositions::collect(engine, phase, tree);
        let mut reclamation =
            engine.prepare_reclamation(&tree.groups, std::mem::take(&mut tree.deleted_free));
        let storage = storage::OriginalStorage::collect(
            engine,
            phase,
            tree,
            &original,
            &dispositions,
            &reclamation,
        );
        storage.withhold_unqualified_reclamation(&original, &mut reclamation);
        let collections = phase.capture_collections(
            dispositions.source().as_ref().ok(),
            std::mem::take(&mut tree.collection_candidates),
        );
        Self {
            original,
            dispositions,
            storage,
            collections,
            reclamation,
        }
    }
}

/// Original count facts and their source and ownership qualification outcomes.
///
/// Complete qualification checks the allocation session, source phase, root, and supported
/// object-header domain. Count facts remain available when domain qualification is unavailable.
/// The caller establishes source freshness and retirement or write permission separately.
#[allow(
    clippy::large_enum_variant,
    reason = "unavailable evidence preserves independent count and source observations"
)]
pub(in crate::edit::commit) enum OriginalEdits {
    /// An incompatible association encountered while qualifying the count facts.
    Association(OriginalOwnerError),
    /// Count and storage facts associated with a supported source object-header domain.
    Complete(OriginalOwnerInventory),
    /// Independent observations retained when complete qualification is unavailable.
    Unavailable {
        /// Count, storage, and eligible-owner facts, or their derivation error.
        counts: Result<OriginalOwnerFacts, EditInputError>,
        /// The original physical root and source-phase identity, or their resolution error.
        source: Result<ReconstructionSource, Error>,
        /// The domain outcome, absent when source resolution failed.
        ///
        /// A physical-claims error is recorded here if domain inspection succeeded.
        domain: Option<Result<OwnerDomain, Error>>,
    },
}

impl OriginalEdits {
    /// Collects original observations using the tree plan's source phase.
    ///
    /// Qualifies successful count facts against the original root and physical claims when
    /// the owner domain is supported. Retains independent outcomes on incomplete qualification.
    pub(super) fn collect(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        tree: &PreparedTree,
        staged: &StagedEdits,
    ) -> Self {
        let counts = Self::count_inputs(engine, phase, &tree.groups, staged);
        let source = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(Error::from)
            .and_then(|root| {
                phase
                    .reconstruction_source(engine, root)
                    .map_err(Error::from)
            });
        let domain = source
            .as_ref()
            .ok()
            .map(|source| phase.supported_owner_domain(source));
        match (counts, source, domain) {
            (Ok(counts), Ok(source), Some(Ok(OwnerDomain::Available(domain)))) => {
                match phase.physical_claims(&source) {
                    Ok(claims) => match counts.qualify(engine, &source, claims, domain) {
                        Ok(inventory) => Self::Complete(inventory),
                        Err(error) => Self::Association(error),
                    },
                    Err(error) => Self::Unavailable {
                        counts: Ok(counts),
                        source: Ok(source),
                        domain: Some(Err(error)),
                    },
                }
            }
            (counts, source, domain) => Self::Unavailable {
                counts,
                source,
                domain,
            },
        }
    }

    /// Derives one net count plan from prepared deletions and each same-file copy emission.
    ///
    /// Counts each physical deletion edge once and preserves each copy emission's multiplicity.
    /// Checks staged foreign copies for an empty committed-datatype inventory at their retained
    /// source widths. Created datasets contribute no original committed-datatype users.
    ///
    /// # Errors
    ///
    /// Returns [`EditInputError`] for failed copy qualification, incomplete inventories, or
    /// errors in count aggregation, planning, or ownership-fact collection.
    pub(super) fn count_inputs(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        groups: &BTreeMap<ObjectPathBuf, Node>,
        staged: &StagedEdits,
    ) -> Result<OriginalOwnerFacts, EditInputError> {
        for (_, copy) in staged.cross_copies() {
            copy.require_foreign_datatype_users_absent()
                .map_err(EditInputError::Copy)?;
        }
        let mut added = CopyUsersUnion::new();
        for (_, copy) in groups.values().flat_map(|node| &node.prepared.copies) {
            let users = phase.copy_users(copy).map_err(EditInputError::Copy)?;
            added = added
                .add(phase, users.as_ref())
                .map_err(EditInputError::Users)?
                .ok_or(EditInputError::IncompleteCopy)?;
        }
        phase
            .net_count_plan(
                groups.values().flat_map(|node| &node.prepared.deletes),
                &added.finish(),
            )
            .map_err(EditInputError::Counts)?
            .ok_or(EditInputError::IncompleteCounts)?
            .original_owners(engine)
            .map_err(EditInputError::Handoff)
    }

    /// Borrows a resolved original source independently of count and domain outcomes.
    ///
    /// Returns [`None`] if source resolution failed or qualification found an incompatible
    /// association.
    pub(in crate::edit::commit) fn source(&self) -> Option<&ReconstructionSource> {
        match self {
            Self::Complete(owners) => Some(owners.source()),
            Self::Unavailable { source, .. } => source.as_ref().ok(),
            Self::Association(_) => None,
        }
    }

    /// Borrows fully qualified facts, or returns [`None`] when qualification failed.
    pub(in crate::edit::commit) fn owners(&self) -> Option<&OriginalOwnerInventory> {
        match self {
            Self::Complete(owners) => Some(owners),
            Self::Association(_) | Self::Unavailable { .. } => None,
        }
    }

    /// Borrows derived count facts independently of domain availability.
    ///
    /// Returns [`None`] for a count-input error or an incompatible qualification association.
    pub(in crate::edit::commit) fn facts(&self) -> Option<&OriginalOwnerFacts> {
        match self {
            Self::Complete(owners) => Some(owners.facts()),
            Self::Unavailable { counts, .. } => counts.as_ref().ok(),
            Self::Association(_) => None,
        }
    }
}

/// A copy-input, count-plan, or owned-fact derivation failure.
#[derive(Debug)]
pub(in crate::edit::commit) enum EditInputError {
    /// A copy could not be inspected or qualified.
    Copy(Error),
    /// Net-count planning failed.
    Counts(NetCountError),
    /// Owned count-fact collection failed.
    Handoff(CountHandoffError),
    /// A same-file copy's committed-datatype inventory is incomplete.
    IncompleteCopy,
    /// Complete count planning is unavailable for the prepared inputs.
    IncompleteCounts,
    /// Copy-user aggregation failed.
    Users(CopyUsersError),
}

#[cfg(test)]
mod tests;
