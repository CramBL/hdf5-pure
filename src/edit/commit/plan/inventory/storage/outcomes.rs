//! Public causes selected at captured application and publication boundaries.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::{EditStorageConflict, PhysicalAddress};

use super::capture::StorageIntegrityError;
use super::parents::ParentPreparationError;
use super::{StorageConflict, StorageUnavailable};
use crate::edit::analysis::CountProtectionError;
use crate::edit::retirement::RetirementMergeError;
use crate::error::{Error, FormatError};

pub(super) fn integrity_error(
    prepared: super::PreparedEditInventory,
    error: StorageIntegrityError,
) -> Error {
    match error {
        StorageIntegrityError::Unavailable => match prepared.storage.unavailable {
            Some(StorageUnavailable::Claims(error) | StorageUnavailable::Coverage(error)) => error,
            _ => prepared
                .dispositions
                .into_source_error()
                .or_else(|| prepared.original.into_source_error())
                .unwrap_or(Error::EditUnsupported(
                    "original storage observations are unavailable",
                )),
        },
        error => error.into(),
    }
}

impl From<StorageConflict> for Error {
    fn from(error: StorageConflict) -> Self {
        match error {
            StorageConflict::Source(error) => error,
            StorageConflict::CountAssociation(error) => error.into(),
            StorageConflict::End { proposed, required } => {
                EditStorageConflict::Truncation { proposed, required }.into()
            }
            StorageConflict::Overlap {
                allocation, range, ..
            } => overlap(allocation.range, range),
        }
    }
}

impl From<StorageIntegrityError> for Error {
    fn from(error: StorageIntegrityError) -> Self {
        match error {
            StorageIntegrityError::BeyondAllocation { range, end } => bounds(range, end),
            StorageIntegrityError::ProtectionBounds { range, end } => bounds(range, end),
            StorageIntegrityError::BeyondSource { range, end } => {
                FormatError::UnexpectedSourceEof {
                    expected: range.end().get(),
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
            StorageIntegrityError::ConflictingClaims { first, second } => overlap(first, second),
            StorageIntegrityError::ConflictingProtection { first, second } => {
                overlap(first, second)
            }
            StorageIntegrityError::ObjectFormat { error } => (*error).into(),
            StorageIntegrityError::ObjectExtent {
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

impl From<CountProtectionError> for Error {
    fn from(error: CountProtectionError) -> Self {
        match error {
            CountProtectionError::Source(error) => error.into(),
            CountProtectionError::Publication(error) => error,
            CountProtectionError::Overlap { first, second } => overlap(first, second),
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
            CountProtectionError::Allocation => {
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

pub(super) fn parent_error(error: ParentPreparationError) -> Error {
    match error {
        ParentPreparationError::Address(error) => error.into(),
        ParentPreparationError::Allocation(error) => error,
        ParentPreparationError::Storage(error) => Error::from(error),
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
