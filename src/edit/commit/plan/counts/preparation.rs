//! Original count preflight and retirement inputs for prepared tree edits.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;

use super::OwnedCountProjection;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::{CopyUsersError, CountEvidenceError};
use crate::edit::analysis::{
    AnalysisPhase, CountPreflightError, CountProtectionError, NetCountError, OriginalOwnerFacts,
    ProtectedCounts,
};
use crate::edit::commit::plan::Node;
use crate::edit::commit::plan::inventory::{EditInputError, OriginalEdits};
use crate::edit::staging::StagedEdits;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

/// Original count outcomes and storage inputs for a prepared tree.
///
/// The editor applies local count projections before constructing a complete result. Original
/// source fields remain unchanged. Source freshness and publication permission are separate
/// requirements.
pub(in crate::edit::commit) enum PreparedTreeCounts {
    /// Complete numeric count planning with local projections applied.
    Complete {
        /// Checked updates to count fields outside reconstructed objects.
        protected: ProtectedCounts,
        /// Original objects retained for collection-reference analysis.
        retained_roots: Vec<ObjectId>,
    },
    /// Checked updates paired with original facts retained by captured application.
    Captured(ProtectedCounts),
    /// Incomplete count evidence, with deletion allocations and collection candidates withheld.
    Unavailable,
}

impl PreparedTreeCounts {
    pub(in crate::edit::commit) fn input_error(error: EditInputError) -> Error {
        match error {
            EditInputError::Copy(error) => error,
            EditInputError::Users(CopyUsersError::Address(error))
            | EditInputError::Counts(NetCountError::Association(CopyUsersError::Address(error))) => {
                error.into()
            }
            EditInputError::Counts(NetCountError::Evidence(CountEvidenceError::Contradiction(
                error,
            ))) => Error::InconsistentReferenceCount {
                address: error.object.header_address().get(),
                observed: error.observed,
                minimum: error.minimum,
            },
            EditInputError::Counts(NetCountError::Evidence(
                CountEvidenceError::Source(error) | CountEvidenceError::Topology(error),
            )) => error.into(),
            EditInputError::Counts(NetCountError::Arithmetic(_))
            | EditInputError::Users(CopyUsersError::Overflow) => {
                Error::EditUnsupported("the planned reference count does not fit its stored field")
            }
            EditInputError::IncompleteCopy => {
                Error::EditUnsupported("the copy's committed-datatype users cannot be established")
            }
            error => {
                FormatError::Internal(format!("invalid count preparation inputs: {error:?}")).into()
            }
        }
    }
}

/// Joint original facts and source permissions after matching local projection completes.
pub(in crate::edit::commit::plan) enum CountedPreparation {
    /// Complete checked inputs from one source phase and allocation session.
    Complete {
        original: OriginalOwnerFacts,
        protected: ProtectedCounts,
    },
    /// Incomplete count evidence, with count-derived retirement withheld.
    Unavailable,
}

impl CountedPreparation {
    /// Prepares original facts and applies their checked local projections once.
    ///
    /// # Errors
    ///
    /// Returns original contradictions, count overflow, invalid associations, or projection errors.
    pub(in crate::edit::commit::plan) fn prepare(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        groups: &mut BTreeMap<ObjectPathBuf, Node>,
        staged: &StagedEdits,
    ) -> Result<Self, Error> {
        let facts = match OriginalEdits::count_inputs(engine, phase, groups, staged) {
            Ok(facts) => facts,
            Err(EditInputError::IncompleteCounts) => return Ok(Self::Unavailable),
            Err(error) => return Err(PreparedTreeCounts::input_error(error)),
        };
        let preflight = facts
            .preflight(
                engine,
                groups.values().flat_map(|node| {
                    node.prepared
                        .replaces
                        .iter()
                        .map(|replacement| replacement.source())
                        .chain(node.prepared.writes.iter().map(|write| &write.source))
                }),
            )
            .map_err(|error| match error {
                CountPreflightError::Protection(CountProtectionError::Source(error)) => {
                    error.into()
                }
                CountPreflightError::Protection(CountProtectionError::Publication(error)) => error,
                _ => Error::EditUnsupported(
                    "the original reference-count updates cannot be protected",
                ),
            })?;
        let proof = OwnedCountProjection::prepare(
            preflight.counts(),
            groups.values_mut().map(|node| &mut node.prepared),
        )
        .map_err(|error| {
            FormatError::Internal(format!("invalid local count projection: {error:?}"))
        })?
        .apply()
        .map_err(|error| FormatError::Internal(format!("invalid local count update: {error:?}")))?;
        let (facts, protected) = preflight.into_parts(proof).map_err(|error| {
            FormatError::Internal(format!("invalid count projection proof: {error:?}"))
        })?;
        Ok(Self::Complete {
            original: facts,
            protected,
        })
    }

    /// Separates application updates from original facts only after selecting the route.
    pub(in crate::edit::commit::plan) fn into_tree(self, capture: bool) -> TreeCountInputs {
        match self {
            Self::Complete {
                original,
                protected,
            } if capture => {
                let retirement = original.storage().allocations().to_vec();
                TreeCountInputs {
                    counts: PreparedTreeCounts::Captured(protected),
                    original: Some(Ok(original)),
                    retirement,
                }
            }
            Self::Complete {
                original,
                protected,
            } => {
                let storage = original.into_storage();
                let (retirement, retained_roots) = storage.into_parts();
                TreeCountInputs {
                    counts: PreparedTreeCounts::Complete {
                        protected,
                        retained_roots,
                    },
                    original: None,
                    retirement,
                }
            }
            Self::Unavailable => TreeCountInputs {
                counts: PreparedTreeCounts::Unavailable,
                original: capture.then_some(Err(EditInputError::IncompleteCounts)),
                retirement: Vec::new(),
            },
        }
    }
}

/// Count application, captured facts, and complete retirement allocations for one selected route.
pub(in crate::edit::commit::plan) struct TreeCountInputs {
    pub(in crate::edit::commit::plan) counts: PreparedTreeCounts,
    pub(in crate::edit::commit::plan) original: Option<Result<OriginalOwnerFacts, EditInputError>>,
    pub(in crate::edit::commit::plan) retirement: Vec<(FileRange, FreeClass)>,
}
