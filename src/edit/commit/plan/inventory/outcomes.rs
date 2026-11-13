//! Owned application outcomes and retained preparation errors.

use super::EditInputError;
use super::InventoryPreparation;
use super::OriginalEdits;
use crate::edit::WriteEngine;
use crate::edit::commit::CommitOutcome;
use crate::edit::commit::plan::PreparedTreeCounts;
use crate::error::Error;

impl InventoryPreparation {
    /// Applies owned inputs and distinguishes rejected staging from consumed commit outcomes.
    ///
    /// Returns original staging for rejected preparation. Applies [`Self::Direct`] inputs through
    /// [`WriteEngine::apply_commit`]. Applies captured tree inputs with storage exclusions and
    /// checked heap publication. After completed publication, returns
    /// [`CommitOutcome::Published`] with its independent compaction result. Returns
    /// [`CommitOutcome::Applied`] for other consumed outcomes. The finalizer checks the engine's
    /// publication state separately before rollback or completion.
    pub(in crate::edit::commit) fn apply(self, engine: &mut WriteEngine) -> CommitOutcome {
        match self {
            Self::Rejected { staged, error } => CommitOutcome::Rejected { staged, error },
            Self::Direct(prepared) => CommitOutcome::Applied(engine.apply_commit(prepared)),
            Self::Tree(prepared) => match prepared.apply_excluding(engine) {
                Ok(pending) => match pending.publish_compacting_members() {
                    Ok(completion) => completion.into_outcome(),
                    Err(failure) => (*failure).into_outcome(),
                },
                Err(failure) => failure.into_outcome(),
            },
        }
    }
}

impl OriginalEdits {
    /// Extracts an owned source, domain, or count-input error, in that order.
    ///
    /// Returns [`None`] for complete or incompatible qualification. Omits
    /// [`EditInputError::IncompleteCounts`] from unqualified inputs.
    pub(super) fn into_source_error(self) -> Option<Error> {
        match self {
            Self::Unavailable {
                counts,
                source,
                domain,
            } => source
                .err()
                .or_else(|| domain.and_then(Result::err))
                .or_else(|| {
                    counts
                        .err()
                        .filter(|error| !matches!(error, EditInputError::IncompleteCounts))
                        .map(PreparedTreeCounts::input_error)
                }),
            Self::Association(_) | Self::Complete(_) => None,
        }
    }
}
