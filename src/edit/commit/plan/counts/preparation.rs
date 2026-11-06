//! Original count preflight and retirement inputs for prepared tree edits.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;

use super::PreparedCounts;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::{CopyUsersError, CountEvidenceError};
use crate::edit::analysis::{
    AnalysisPhase, CountHandoffError, CountInventory, CountPreflightError, CountProtectionError,
    NetCountError, ProtectedCounts,
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
        /// Original count changes retained after local projection.
        inventory: CountInventory,
        /// Checked updates to count fields outside reconstructed objects.
        protected: ProtectedCounts,
        /// Original objects retained for collection-reference analysis.
        retained_roots: Vec<ObjectId>,
        /// Ordinary non-root objects with zero graph and declared count remainders.
        eligible_owners: Vec<ObjectId>,
        /// Supported original allocations, moved into the tree before reference screening.
        retirement: Vec<(FileRange, FreeClass)>,
    },
    /// Incomplete count evidence, with deletion allocations and collection candidates withheld.
    Unavailable,
}

impl PreparedTreeCounts {
    /// Derives original retirement eligibility and updates checked replacement count records.
    ///
    /// Retains protected source updates and reference roots independently of supported allocation
    /// coverage. Returns [`Self::Unavailable`] for incomplete count evidence. Leaves original source
    /// fields unchanged.
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
            Err(EditInputError::IncompleteCounts) => return Ok(Self::Unavailable),
            Err(error) => return Err(Self::input_error(error)),
        };
        let (counts, storage, eligible_owners) = facts.into_parts();
        let preflight = counts
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
        let (_, proof) = PreparedCounts::owned(
            &preflight,
            groups.values_mut().map(|node| &mut node.prepared),
        )
        .map_err(|error| {
            FormatError::Internal(format!("invalid local count projection: {error:?}"))
        })?
        .apply()
        .map_err(|error| FormatError::Internal(format!("invalid local count update: {error:?}")))?;
        let (inventory, protected) = preflight.into_parts(proof).map_err(|error| {
            FormatError::Internal(format!("invalid count projection proof: {error:?}"))
        })?;
        let (retirement, retained_roots) = storage.into_parts();
        Ok(Self::Complete {
            inventory,
            protected,
            retained_roots,
            eligible_owners,
            retirement,
        })
    }

    fn input_error(error: EditInputError) -> Error {
        match error {
            EditInputError::Copy(error) => error,
            EditInputError::Handoff(CountHandoffError::Source(error))
            | EditInputError::Users(CopyUsersError::Address(error))
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

#[cfg(test)]
mod tests;
