//! Failure conversions for original-storage application and publication.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::EditStorageConflict;
use hdf5_pure_core::PhysicalAddress;

use super::OriginalStorage;
use super::ParentPreparationError;
use super::PendingEditInventory;
use super::StorageApplyError;
use super::StorageApplyFailure;
use super::StorageConflict;
use super::StorageIntegrityError;
use super::StoragePublicationFailure;
use super::StoragePublicationRejection;
use crate::edit::analysis::CountProtectionError;
use crate::edit::commit::CommitOutcome;
use crate::edit::commit::PublicationError;
use crate::edit::commit::plan::inventory::PreparedEditInventory;
use crate::edit::retirement::RetirementMergeError;
use crate::error::Error;
use crate::error::FormatError;

impl StorageApplyFailure {
    /// Consumes failed inputs and returns [`CommitOutcome::Applied`] with the selected error.
    ///
    /// Prioritizes direct application errors and concrete storage conflicts. For unavailable
    /// capture or failed associations and replacement, extracts retained causes from storage,
    /// predecessor, and original observations in that order. Converts the recorded rejection when
    /// a retained source error is unavailable.
    pub(in crate::edit::commit) fn into_outcome(self) -> CommitOutcome {
        let error = match self {
            Self::Application(error) => error,
            Self::BeforePlacement { prepared, error } => {
                let PreparedEditInventory {
                    prepared: _,
                    original,
                    session: _,
                    dispositions,
                    storage,
                    collections: _,
                    reclamation: _,
                } = *prepared;
                match error {
                    StorageApplyError::Conflict(error)
                        if !matches!(
                            error,
                            StorageConflict::Original(StorageIntegrityError::Unavailable)
                        ) =>
                    {
                        error.into()
                    }
                    error => storage
                        .into_capture_error()
                        .or_else(|| dispositions.into_source_error())
                        .or_else(|| original.into_source_error())
                        .unwrap_or_else(|| match error {
                            StorageApplyError::Unavailable => Error::EditUnsupported(
                                "original predecessor replacement cannot be established",
                            ),
                            StorageApplyError::Conflict(error) => error.into(),
                            error @ (StorageApplyError::Phase
                            | StorageApplyError::Root
                            | StorageApplyError::Session) => FormatError::Internal(format!(
                                "incompatible prepared storage association: {error:?}"
                            ))
                            .into(),
                        }),
                }
            }
        };
        CommitOutcome::Applied(Err(error))
    }
}

impl StoragePublicationFailure<'_> {
    /// Consumes retained publication observations and returns [`CommitOutcome::Applied`].
    ///
    /// Returns direct write errors unchanged. For placement rejection, prioritizes original
    /// predecessor and source errors over unsupported reconciliation. Other rejections use the
    /// recorded exclusion or parent failure.
    pub(in crate::edit::commit) fn into_outcome(self) -> CommitOutcome {
        let error = match self {
            Self::BeforePublication { pending, reason } => {
                let PendingEditInventory {
                    engine: _,
                    session: _,
                    counts: _,
                    original,
                    dispositions,
                    storage,
                    collections: _,
                    prospective: _,
                    pending: _,
                    relocations: _,
                    exclusion,
                } = pending;
                match reason {
                    StoragePublicationRejection::Exclusion => match exclusion {
                        Err(error) => storage_error(storage, error),
                        Ok(()) => FormatError::Internal("a publication exclusion failure has no recorded conflict".into()).into(),
                    },
                    StoragePublicationRejection::Parents(error) => parent_error(storage, *error),
                    StoragePublicationRejection::Placement => dispositions.into_source_error()
                        .or_else(|| original.into_source_error())
                        .unwrap_or(Error::EditUnsupported("prepared placements cannot be reconciled with the original predecessors")),
                }
            }
            Self::Publication {
                original: _,
                dispositions: _,
                storage,
                collections: _,
                prospective: _,
                error,
            } => match error {
                PublicationError::Write(error) => error,
                PublicationError::Storage(error) => storage_error(storage, error),
            },
        };
        CommitOutcome::Applied(Err(error))
    }
}

/// Reports physical conflicts through [`EditStorageConflict`] and returns underlying source errors.
impl From<StorageConflict> for Error {
    fn from(error: StorageConflict) -> Self {
        match error {
            StorageConflict::Source(error) => error,
            StorageConflict::CountAssociation(error) => error.into(),
            StorageConflict::End { proposed, required } => {
                EditStorageConflict::Truncation { proposed, required }.into()
            }
            StorageConflict::Overlap {
                allocation,
                range,
                claim: _,
            } => overlap(allocation.range, range),
            StorageConflict::Original(error) => error.into(),
        }
    }
}

/// Reports claim and protection overlaps and allocation bounds through [`EditStorageConflict`].
///
/// Reports source-length failures through [`FormatError::UnexpectedSourceEof`]. Returns original
/// object-format errors and reports invalid local extents through
/// [`Error::InvalidSpaceOwnershipExtent`]. Reports unavailable observations as
/// [`Error::EditUnsupported`].
impl From<StorageIntegrityError> for Error {
    fn from(error: StorageIntegrityError) -> Self {
        match error {
            StorageIntegrityError::BeyondAllocation { claim, end } => bounds(claim.range(), end),
            StorageIntegrityError::ProtectionBounds { allocation, end } => {
                bounds(allocation.range, end)
            }
            StorageIntegrityError::BeyondSource { claim, end } => {
                FormatError::UnexpectedSourceEof {
                    expected: claim.range().end().get(),
                    available: end.get(),
                }
                .into()
            }
            StorageIntegrityError::TruncatedSource { end, required } => {
                FormatError::UnexpectedSourceEof {
                    expected: required.get(),
                    available: end.get(),
                }
                .into()
            }
            StorageIntegrityError::ConflictingClaims { first, second } => {
                overlap(first.range(), second.range())
            }
            StorageIntegrityError::ConflictingProtection { first, second } => {
                overlap(first.range, second.range)
            }
            StorageIntegrityError::ObjectFormat { object: _, error } => (*error).into(),
            StorageIntegrityError::ObjectExtent {
                object: _,
                address,
                length,
                previous_end,
                logical_size,
            } => Self::InvalidSpaceOwnershipExtent {
                address,
                length,
                previous_end,
                logical_size,
            },
            StorageIntegrityError::Unavailable => {
                Self::EditUnsupported("original storage observations are unavailable")
            }
        }
    }
}

/// Extracts captured errors for unavailable integrity and converts concrete conflicts directly.
fn storage_error(storage: OriginalStorage, error: StorageConflict) -> Error {
    if matches!(
        error,
        StorageConflict::Original(StorageIntegrityError::Unavailable)
    ) {
        return storage
            .into_capture_error()
            .unwrap_or_else(|| Error::from(error));
    }
    error.into()
}

/// Returns underlying address or allocation errors, or physical bounds and overlap diagnostics.
///
/// Reports remaining parent-check failures as [`FormatError::Internal`].
fn parent_error(storage: OriginalStorage, error: ParentPreparationError) -> Error {
    match error {
        ParentPreparationError::Address(error) => error.into(),
        ParentPreparationError::Allocation(error) => error,
        ParentPreparationError::Storage(error) => storage_error(storage, error),
        ParentPreparationError::Bounds { range, end }
        | ParentPreparationError::Batch(RetirementMergeError::Bounds { range, end }) => {
            bounds(range, end)
        }
        ParentPreparationError::Batch(RetirementMergeError::Overlap { first, second }) => {
            overlap(first, second)
        }
        error @ (ParentPreparationError::Batch(
            RetirementMergeError::Class { .. } | RetirementMergeError::Empty(_),
        )
        | ParentPreparationError::OriginalRoot { .. }
        | ParentPreparationError::Protection { .. }
        | ParentPreparationError::RecordedExclusion
        | ParentPreparationError::Root { .. }
        | ParentPreparationError::Session
        | ParentPreparationError::Source { .. }) => {
            FormatError::Internal(format!("invalid original heap publication: {error:?}")).into()
        }
    }
}

/// Reports count-allocation overlaps as physical conflicts and returns source errors unchanged.
///
/// Reports missing allocation geometry as [`Error::EditUnsupported`] and incompatible count
/// associations as [`FormatError::Internal`].
impl From<CountProtectionError> for Error {
    fn from(error: CountProtectionError) -> Self {
        match error {
            CountProtectionError::Source(error) => error.into(),
            CountProtectionError::Publication(error) => error,
            CountProtectionError::Overlap {
                first: (_, first),
                second: (_, second),
            } => overlap(first, second),
            CountProtectionError::Reusable {
                allocation,
                reusable,
                ..
            } => overlap(allocation, reusable),
            CountProtectionError::Retained {
                allocation,
                retained,
                ..
            } => overlap(allocation, retained),
            CountProtectionError::Aggregate {
                allocation, spare, ..
            } => overlap(allocation, spare),
            CountProtectionError::Allocation { .. } => {
                Error::EditUnsupported("the original count allocation is unavailable")
            }
            error @ (CountProtectionError::Phase { .. }
            | CountProtectionError::Repeated { .. }
            | CountProtectionError::Root { .. }
            | CountProtectionError::Session) => FormatError::Internal(format!(
                "incompatible original count association: {error:?}"
            ))
            .into(),
        }
    }
}

fn bounds(range: FileRange, end: PhysicalAddress) -> Error {
    EditStorageConflict::Bounds {
        range: range.range(),
        end,
    }
    .into()
}

fn overlap(first: FileRange, second: FileRange) -> Error {
    EditStorageConflict::Overlap {
        first: first.range(),
        second: second.range(),
    }
    .into()
}

#[cfg(test)]
mod tests;
