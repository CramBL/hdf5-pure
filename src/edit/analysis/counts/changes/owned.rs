//! Owned original count facts, planned values, and count dispositions.

use core::num::NonZeroU32;
use core::num::NonZeroU64;

use hdf5_pure_format::__private::ObjectReferenceCountField;

use super::{CountChange, OriginalCount};
use crate::edit::analysis::counts::{CountEvidence, HeaderCountAllocation, OrdinaryCountSubject};
use crate::object_graph::ObjectId;

impl CountChange<'_, '_> {
    /// Copies original facts and the planned count without retaining the borrowed plan.
    ///
    /// Preserves the original observation class, optional physical field, and independently
    /// checked allocation geometry. Requires no source reads or heap allocation.
    pub(in crate::edit::analysis) fn detach(&self) -> OwnedCountChange {
        let Self {
            original,
            remaining,
        } = self;
        let (object, original, field, allocation) = match *original {
            OriginalCount::Referenced(observed) => (
                observed.object(),
                OwnedOriginalCount::Referenced {
                    original: observed.original(),
                    incoming: observed.incoming(),
                    minimum: observed.minimum(),
                },
                observed.field(),
                observed.header_allocation(),
            ),
            OriginalCount::Topology(evidence) => {
                let bound = evidence.bound();
                let original = match evidence {
                    CountEvidence::Ordinary(ordinary) => OwnedOriginalCount::Ordinary {
                        original: bound.original,
                        incoming: bound.incoming(),
                        subject: ordinary.subject(),
                    },
                    CountEvidence::Retained(_) => OwnedOriginalCount::Retained {
                        original: bound.original,
                        incoming: bound.incoming(),
                    },
                };
                (
                    bound.object(),
                    original,
                    bound.field(),
                    bound.header_allocation(),
                )
            }
        };
        OwnedCountChange {
            object,
            original,
            remaining: *remaining,
            field,
            allocation,
        }
    }
}

/// Owned original count facts and a planned count for one physical object.
///
/// The original observation class, physical field, and independent allocation facts are preserved.
/// Source associations are retained separately by the enclosing inventory. Source freshness,
/// original-user completeness, ownership, retirement, and write permission require separate evidence.
#[derive(Debug)]
pub(in crate::edit) struct OwnedCountChange {
    object: ObjectId,
    original: OwnedOriginalCount,
    remaining: u32,
    field: Option<ObjectReferenceCountField>,
    allocation: Option<HeaderCountAllocation>,
}

impl OwnedCountChange {
    /// Returns the physical object-header identity of the original target.
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }

    /// Returns the original count facts with their observation class.
    pub(in crate::edit) fn original(&self) -> OwnedOriginalCount {
        self.original
    }

    /// Returns the final planned declared count.
    pub(in crate::edit) fn remaining(&self) -> u32 {
        self.remaining
    }

    /// Returns the original physical count field, or `None` for an implicit count.
    pub(in crate::edit) fn field(&self) -> Option<ObjectReferenceCountField> {
        self.field
    }

    /// Returns the independently checked complete allocation containing the field, if known.
    pub(in crate::edit) fn header_allocation(&self) -> Option<HeaderCountAllocation> {
        self.allocation
    }

    /// Classifies the planned count while borrowing the original entry and any explicit field.
    ///
    /// Checks numeric equality first, including zero or implicit counts. Classifies a changed zero
    /// count outside [`OwnedOriginalCount::Ordinary`] as [`CountDisposition::Retain`]. Classifies
    /// other changed counts as positive with an explicit field, zero, or missing a field. Requires
    /// no source reads or heap allocation.
    pub(in crate::edit) fn disposition(&self) -> CountDisposition<'_> {
        let Self {
            object: _,
            original,
            remaining,
            field,
            allocation: _,
        } = self;
        if original.value() == *remaining {
            return CountDisposition::Unchanged(self);
        }
        if *remaining == 0 && !matches!(original, OwnedOriginalCount::Ordinary { .. }) {
            return CountDisposition::Retain(self);
        }
        match (NonZeroU32::new(*remaining), field.as_ref()) {
            (None, _) => CountDisposition::Unresolved {
                change: self,
                reason: UnresolvedCount::Zero,
            },
            (Some(_), None) => CountDisposition::Unresolved {
                change: self,
                reason: UnresolvedCount::MissingField,
            },
            (Some(count), Some(field)) => CountDisposition::Surviving {
                change: self,
                field,
                count,
            },
        }
    }
}

/// Original count facts in their established topology or referenced-target class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum OwnedOriginalCount {
    /// A positive topology-bound count for a recognized dataset or group header.
    Ordinary {
        /// The original positive declared count, including an implicit one.
        original: NonZeroU32,
        /// The visible hard-link count in the original topology.
        incoming: u32,
        /// The recognized dataset or group message shape.
        subject: OrdinaryCountSubject,
    },
    /// A referenced-target count with any independently established original-reference facts.
    Referenced {
        /// The original declared count, including an implicit one or an unproven zero.
        original: u32,
        /// The visible hard-link count, or `None` without complete topology.
        ///
        /// `Some(0)` records a known absence of visible links.
        incoming: Option<u32>,
        /// The proven positive original-reference minimum, if known.
        minimum: Option<NonZeroU64>,
    },
    /// A positive topology-bound count for the root or an unrecognized header shape.
    Retained {
        /// The original positive declared count, including an implicit one.
        original: NonZeroU32,
        /// The visible hard-link count, excluding the root's superblock reference.
        incoming: u32,
    },
}

impl OwnedOriginalCount {
    /// Returns the original declared count in any observation class.
    pub(in crate::edit) fn value(self) -> u32 {
        match self {
            Self::Ordinary { original, .. } | Self::Retained { original, .. } => original.get(),
            Self::Referenced { original, .. } => original,
        }
    }
}

/// A borrowed classification of a planned count and its original observation class.
///
/// Every variant retains the original entry. Allocation ownership, source freshness, and write or
/// retirement permissions require separate evidence, including for positive explicit counts.
#[derive(Debug)]
pub(in crate::edit) enum CountDisposition<'a> {
    /// A changed zero count outside [`OwnedOriginalCount::Ordinary`].
    Retain(&'a OwnedCountChange),
    /// A changed positive count with an original explicit field.
    Surviving {
        change: &'a OwnedCountChange,
        field: &'a ObjectReferenceCountField,
        count: NonZeroU32,
    },
    /// A planned count equal to the original value, including zero or implicit counts.
    Unchanged(&'a OwnedCountChange),
    /// A changed count excluded from positive explicit preparation, with its reason.
    Unresolved {
        change: &'a OwnedCountChange,
        reason: UnresolvedCount,
    },
}

/// A changed count excluded from positive explicit preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum UnresolvedCount {
    /// A positive planned count without an original explicit field.
    MissingField,
    /// A changed zero count in [`OwnedOriginalCount::Ordinary`], with or without an explicit field.
    Zero,
}
