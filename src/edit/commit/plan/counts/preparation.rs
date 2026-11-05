//! Checked object counts and original storage facts for prepared tree commits.

use std::collections::BTreeMap;

use super::super::Node;
use super::super::ReplacedHeader;
use super::CountProjectionError;
use super::PreparedCounts;
use crate::ObjectReferenceCountUpdateIssue;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::CollectionCandidates;
use crate::edit::analysis::CountPreflight;
use crate::edit::analysis::CountStorage;
use crate::edit::analysis::NetCountError;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::edit::analysis::counts::added_users::CopyUsersError;
use crate::edit::analysis::counts::added_users::CopyUsersUnion;
use crate::edit::analysis::counts::balance::CountArithmeticError;
use crate::edit::commit::CountProjectionProof;
use crate::edit::object_header::reference_count::HeaderCountError;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_path::ObjectPathBuf;

/// Prepared count work and the availability of original retirement evidence.
pub(in crate::edit::commit) enum PreparedCountWork {
    /// Checked original facts, owned projections, supported allocations, and collection candidates.
    Complete(Box<PreparedTreeCounts>),
    /// A plan without count changes.
    Empty,
    /// Unavailable count evidence, which excludes general collection retirement.
    Unavailable,
}

impl PreparedCountWork {
    /// Borrows checked work, or returns `None` for empty or unavailable evidence.
    pub(in crate::edit::commit) fn checked(&self) -> Option<&PreparedTreeCounts> {
        match self {
            Self::Complete(counts) => Some(counts),
            Self::Empty | Self::Unavailable => None,
        }
    }

    /// Returns owned checked work, or `None` for empty or unavailable evidence.
    pub(in crate::edit::commit) fn into_checked(self) -> Option<Box<PreparedTreeCounts>> {
        match self {
            Self::Complete(counts) => Some(counts),
            Self::Empty | Self::Unavailable => None,
        }
    }
}

/// Owned count facts and a matching local projection retained through tree publication.
///
/// Changed positive source fields outside the reconstruction set have checked allocation evidence.
/// Supported original storage and retained reference roots remain separate from permission to
/// write, admit, or retire allocations.
pub(in crate::edit::commit) struct PreparedTreeCounts {
    /// Original count facts, reconstruction decisions, and protected source fields.
    pub(in crate::edit::commit) preflight: CountPreflight,
    /// Proof that the matching owned count projections were applied.
    pub(in crate::edit::commit) projection: CountProjectionProof,
    /// Supported original allocations and retained roots from the same count plan.
    pub(in crate::edit::commit) storage: CountStorage,
    /// Collection allocations and origins from eligible objects in the original source phase.
    pub(in crate::edit::commit) collections: CollectionCandidates,
}

impl PreparedTreeCounts {
    /// Prepares declared counts for the tree's same-file copies and deletion records.
    ///
    /// Uses the supplied analysis phase, preserves original count facts, and applies checked
    /// changes to owned replacement records. Retains collection candidates from eligible objects.
    /// Returns [`PreparedCountWork::Empty`] for a plan without count changes, or
    /// [`PreparedCountWork::Unavailable`] when original count planning is unavailable. Boxes
    /// nonempty checked work with its source protection and matching completion proof.
    /// Source writes, reference completeness, admission, and retirement require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for incomplete or unavailable copied datatype-user
    /// counts. Returns count contradictions, arithmetic failures, unsupported updates, source
    /// errors, and inconsistent associations from count planning, preflight, or local projection.
    pub(in crate::edit::commit) fn prepare(
        engine: &WriteEngine,
        nodes: &mut BTreeMap<ObjectPathBuf, Node>,
        analysis: &mut AnalysisPhase<'_>,
    ) -> Result<PreparedCountWork, Error> {
        let mut union = CopyUsersUnion::new();
        for tree in nodes
            .values()
            .flat_map(|node| node.prepared.copies.iter().map(|(_, tree)| tree))
        {
            let users = analysis.copy_users(tree)?;
            let Some(next) = union.add(analysis, users.as_ref())? else {
                return Err(Error::EditUnsupported(
                    "copy datatype-user counts are incomplete",
                ));
            };
            union = next;
        }
        let added = union.finish();
        let Some(plan) = analysis.net_count_plan(
            nodes.values().flat_map(|node| &node.prepared.deletes),
            &added,
        )?
        else {
            if added.users().next().is_some() {
                return Err(Error::EditUnsupported(
                    "copied datatype reference counts are unavailable",
                ));
            }
            return Ok(PreparedCountWork::Unavailable);
        };
        if plan.evidence_changes().next().is_none() {
            return Ok(PreparedCountWork::Empty);
        }
        let mut storage = plan.collect_storage();
        let inventory = plan.owned_counts(engine).map_err(|error| {
            FormatError::Internal(format!("commit count handoff is inconsistent: {error:?}"))
        })?;
        drop(plan);
        let collections = storage.take_collections(analysis)?;
        let preflight = inventory.preflight(
            engine,
            nodes.values().flat_map(|node| {
                node.prepared
                    .replaces
                    .iter()
                    .map(ReplacedHeader::source)
                    .chain(node.prepared.writes.iter().map(|write| &write.source))
            }),
        )?;
        let (disposition, projection) = PreparedCounts::owned(
            &preflight,
            nodes.values_mut().map(|node| &mut node.prepared),
        )?
        .apply()
        .map_err(HeaderCountError::into_application_error)?;
        drop(disposition);
        Ok(PreparedCountWork::Complete(Box::new(Self {
            preflight,
            projection,
            storage,
            collections,
        })))
    }
}

impl From<NetCountError> for Error {
    /// Preserves count operands and classifies source or association failures.
    fn from(error: NetCountError) -> Self {
        match error {
            NetCountError::Arithmetic { object, error } => match error {
                CountArithmeticError::Overflow {
                    original,
                    added,
                    removed,
                } => Error::ObjectReferenceCountOverflow {
                    address: object.header_address().get(),
                    original,
                    added,
                    removed,
                },
                CountArithmeticError::Underflow {
                    original, removed, ..
                } => Error::InvalidObjectReferenceCount {
                    address: object.header_address().get(),
                    observed: original,
                    minimum: removed,
                },
            },
            NetCountError::Association(error) => error.into(),
            NetCountError::Evidence(CountEvidenceError::Contradiction(error)) => {
                Error::InvalidObjectReferenceCount {
                    address: error.object.header_address().get(),
                    observed: error.observed,
                    minimum: error.minimum,
                }
            }
            NetCountError::Evidence(
                CountEvidenceError::Source(error) | CountEvidenceError::Topology(error),
            ) => error.into(),
        }
    }
}

impl From<CopyUsersError> for Error {
    /// Preserves prospective-reference overflow operands and address failures.
    ///
    /// Classifies incompatible engine, phase, or root associations as [`FormatError::Internal`].
    fn from(error: CopyUsersError) -> Self {
        match error {
            CopyUsersError::Address(error) => error.into(),
            CopyUsersError::Overflow {
                object,
                current,
                added,
            } => Error::ObjectReferenceCountAdditionOverflow {
                address: object.header_address().get(),
                current: current.get(),
                added: added.get(),
            },
            CopyUsersError::Engine | CopyUsersError::Phase | CopyUsersError::Root { .. } => {
                FormatError::Internal(
                    "commit copy counts have an inconsistent source association".into(),
                )
                .into()
            }
        }
    }
}

impl From<CountProjectionError> for Error {
    /// Preserves original-count mismatches and classifies unsupported replacement framing.
    ///
    /// Preserves source framing errors and classifies incompatible predecessor associations as
    /// [`FormatError::Internal`].
    fn from(error: CountProjectionError) -> Self {
        match error {
            CountProjectionError::Original {
                object,
                expected,
                actual,
            } => Error::ObjectReferenceCountMismatch {
                address: object.header_address().get(),
                expected,
                actual,
            },
            CountProjectionError::Region { object, error } => match error {
                HeaderCountError::Framing(error) => error.into(),
                HeaderCountError::Ambiguous | HeaderCountError::Unsupported(_) => {
                    Error::ObjectReferenceCountUpdateUnsupported {
                        address: object.header_address().get(),
                        issue: ObjectReferenceCountUpdateIssue::UnsupportedEncoding,
                    }
                }
            },
            CountProjectionError::Phase { .. }
            | CountProjectionError::Preflight
            | CountProjectionError::Reconstruction { .. }
            | CountProjectionError::Repeated { .. } => FormatError::Internal(
                "commit count projection has an inconsistent predecessor association".into(),
            )
            .into(),
        }
    }
}

impl HeaderCountError {
    /// Preserves framing failures and reports inconsistent prepared views as internal errors.
    fn into_application_error(self) -> Error {
        match self {
            HeaderCountError::Framing(error) => error.into(),
            HeaderCountError::Ambiguous | HeaderCountError::Unsupported(_) => {
                FormatError::Internal("a prepared count projection changed its framing".into())
                    .into()
            }
        }
    }
}

#[cfg(test)]
mod tests;
