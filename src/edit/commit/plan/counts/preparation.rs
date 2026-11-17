//! Original count preflight and retirement inputs for prepared tree edits.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;

use super::PreparedCounts;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::CountPreflightError;
use crate::edit::analysis::CountProtectionError;
use crate::edit::analysis::NetCountError;
use crate::edit::analysis::OriginalOwnerFacts;
use crate::edit::analysis::ProtectedCounts;
use crate::edit::analysis::counts::{CopyUsersError, CountEvidenceError};
use crate::edit::commit::plan::Node;
use crate::edit::commit::plan::inventory::{EditInputError, OriginalEdits};
use crate::edit::staging::StagedEdits;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

/// Stores joint owner facts and source updates, or incomplete count evidence.
pub(in crate::edit::commit::plan) enum CountedPreparation {
    /// Complete count preparation with checked local projections applied.
    Complete {
        /// Original count, storage, and eligible-owner facts from the same checked plan.
        original: OriginalOwnerFacts,
        /// Protected count updates for objects outside the reconstruction set.
        protected: ProtectedCounts,
    },
    /// Incomplete count evidence, with count-derived deletion allocations withheld.
    Unavailable,
}

impl CountedPreparation {
    /// Prepares joint owner facts and local projections, or records incomplete count evidence.
    ///
    /// For complete count inputs, uses one count and storage input set for retirement screening
    /// and captured qualification. Checks completion of local count projection against its
    /// originating preflight and retains source updates and reference roots independently of
    /// supported allocation coverage. Returns [`Self::Unavailable`] for incomplete count evidence.
    /// Leaves original fields unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InconsistentReferenceCount`] for a proven original contradiction,
    /// [`Error::EditUnsupported`] for unsupported count or copy preparation, or the source,
    /// publication, or internal projection error.
    pub(in crate::edit::commit::plan) fn prepare(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        groups: &mut BTreeMap<ObjectPathBuf, Node>,
        staged: &StagedEdits,
    ) -> Result<Self, Error> {
        let facts = match OriginalEdits::count_inputs(engine, phase, groups, staged) {
            Ok(facts) => facts,
            Err(EditInputError::IncompleteCounts) => {
                let prepared = Self::Unavailable;

                return Ok(prepared);
            }
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
        let proof = PreparedCounts::owned(
            preflight.counts(),
            groups.values_mut().map(|node| &mut node.prepared),
        )
        .map_err(|error| {
            FormatError::Internal(format!("invalid local count projection: {error:?}"))
        })?
        .apply()
        .map_err(|error| FormatError::Internal(format!("invalid local count update: {error:?}")))?;
        let (original, protected) = preflight.into_parts(proof).map_err(|error| {
            FormatError::Internal(format!("invalid count projection proof: {error:?}"))
        })?;
        let prepared = Self::Complete {
            original,
            protected,
        };

        Ok(prepared)
    }

    /// Separates application, original-capture, and reclamation inputs for the selected tree route.
    ///
    /// For [`Self::Complete`], retains the joint facts and copies their allocation spans for
    /// reclamation if `capture` is true. If it is false, moves storage spans to reclamation and
    /// reference roots to direct application.
    ///
    /// For [`Self::Unavailable`], withholds count-derived retirement and records
    /// [`EditInputError::IncompleteCounts`] when capture is requested.
    pub(in crate::edit::commit::plan) fn into_tree(self, capture: bool) -> TreeCountInputs {
        match self {
            Self::Complete {
                original,
                protected,
            } => {
                let (original, retirement, retained_roots) = if capture {
                    let retirement = original.storage().allocations().to_vec();
                    (Some(Ok(original)), retirement, Vec::new())
                } else {
                    let (_, storage, _) = original.into_parts();
                    let (retirement, retained_roots) = storage.into_parts();
                    (None, retirement, retained_roots)
                };
                TreeCountInputs {
                    counts: PreparedTreeCounts::Complete {
                        protected,
                        retained_roots,
                    },
                    original,
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

/// Stores application, capture, and reclamation inputs from one counted preparation.
pub(in crate::edit::commit::plan) struct TreeCountInputs {
    /// The count application outcome, including any direct-plan reference roots.
    pub(in crate::edit::commit::plan) counts: PreparedTreeCounts,
    /// Captured original facts or incomplete evidence, absent for direct plans.
    pub(in crate::edit::commit::plan) original: Option<Result<OriginalOwnerFacts, EditInputError>>,
    /// Supported original allocation spans and free-space classes for reclamation.
    pub(in crate::edit::commit::plan) retirement: Vec<(FileRange, FreeClass)>,
}

/// Stores source count updates and direct-plan reference roots for tree application.
///
/// Local projections have been checked and applied for [`Self::Complete`]. Complete captured
/// preparations retain original reference roots with their joint owner facts. Source freshness
/// and publication permission require separate evidence.
pub(in crate::edit::commit) enum PreparedTreeCounts {
    /// Complete count preparation with original source fields unchanged.
    Complete {
        /// Checked source count updates outside reconstructed objects.
        protected: ProtectedCounts,
        /// Original reference roots for direct application, empty for captured inventories.
        retained_roots: Vec<ObjectId>,
    },
    /// Unavailable source count updates and retained reference roots.
    Unavailable,
}

impl PreparedTreeCounts {
    /// Converts count-input failures with their original source errors and contradiction fields.
    ///
    /// Reports count overflow and incomplete copy users as [`Error::EditUnsupported`], and
    /// incompatible inputs as [`FormatError::Internal`].
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
            EditInputError::Counts(NetCountError::Arithmetic { .. })
            | EditInputError::Users(CopyUsersError::Overflow { .. }) => {
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
