//! Failure conversions for original-storage application and publication.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::EditStorageConflict;
use hdf5_pure_core::PhysicalAddress;

use super::super::OriginalEdits;
use super::super::dispositions::ReconstructionDispositions;
use super::OriginalStorage;
use super::ParentPreparationError;
use super::StorageApplyError;
use super::StorageConflict;
use super::StorageIntegrityError;
use crate::edit::analysis::CountProtectionError;
use crate::edit::retirement::RetirementMergeError;
use crate::error::Error;
use crate::error::FormatError;

impl StorageApplyError {
    /// Selects a rejection's retained cause or converts the recorded error.
    ///
    /// Converts concrete storage conflicts directly. For unavailable capture, failed
    /// associations, or unavailable integrity, extracts the first retained cause from storage,
    /// predecessor, and original observations in that order. Converts the recorded rejection
    /// if none of these observations retains a cause.
    pub(super) fn into_error(
        self,
        storage: OriginalStorage,
        dispositions: ReconstructionDispositions,
        original: OriginalEdits,
    ) -> Error {
        match self {
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
pub(super) fn storage_error(storage: OriginalStorage, error: StorageConflict) -> Error {
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
pub(super) fn parent_error(storage: OriginalStorage, error: ParentPreparationError) -> Error {
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
