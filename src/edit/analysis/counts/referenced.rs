//! Original count observations for targets of prospective same-file copies.
//!
//! Complete topology is optional. Source freshness, original-user completeness, allocation
//! ownership, retirement eligibility, and count-write permission require separate evidence.

use core::num::NonZeroU64;
use std::sync::Arc;

use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountField;

use crate::edit::WriteEngine;
use crate::edit::analysis::{AnalysisPhase, CompleteTopology, ReconstructionSource};
use crate::object_graph::{ObjectId, ObjectIndex};

use super::added_users::{AddedCopyUsers, CopyUsersError};
use super::{
    CountContradiction, CountEvidenceError, CountQueries, HeaderCountAllocation, IncomingLinks,
    ROOT_REFERENCE_COUNT,
};

mod batch;

impl<'source> AnalysisPhase<'source> {
    /// Returns an original count for a target of the supplied prospective copy emissions.
    ///
    /// Returns `None` for an empty union, an unlisted target, or an ambiguous or unsupported count.
    /// Reuses the source-count cache and any complete topology already discovered in this phase.
    /// Includes the superblock reference in the root's minimum.
    /// Preserves a zero count when no positive original-reference minimum is known.
    ///
    /// The caller establishes copy-source applicability and preserves the dependent source bytes.
    /// Prospective copy multiplicities are separate from original-user minima.
    ///
    /// # Errors
    ///
    /// Returns [`ReferencedCountError::Association`] if `added` or the cached topology belongs to
    /// another engine, phase, or root, or if original-root resolution fails. Returns
    /// [`ReferencedCountError::Evidence`] for source-count errors, inconsistent topology, or an
    /// observed count below a proven positive original-reference minimum.
    pub(in crate::edit) fn referenced_count(
        &mut self,
        added: &AddedCopyUsers<'source>,
        object: ObjectId,
    ) -> Result<Option<ReferencedCount<'source>>, ReferencedCountError> {
        let Some(origin) = added
            .source_for(self, object)
            .map_err(ReferencedCountError::Association)?
        else {
            return Ok(None);
        };
        let engine = self.engine;
        let topology = self.topology.as_ref().and_then(Option::as_ref);
        let OriginalReferences { incoming, minimum } =
            OriginalReferences::new(engine, &origin, topology, object)?;
        let mut queries = CountQueries::new(engine, &mut self.counts, topology);
        let observation = queries
            .summary(object)
            .map_err(|error| ReferencedCountError::Evidence(CountEvidenceError::Source(error)))?;
        let (count, allocation) = observation.into_parts();
        let (original, field) = match count {
            ObjectReferenceCount::ImplicitOne => (IMPLICIT_REFERENCE_COUNT, None),
            ObjectReferenceCount::Version1(field) | ObjectReferenceCount::Version2(field) => {
                (field.value(), Some(field))
            }
            ObjectReferenceCount::Ambiguous | ObjectReferenceCount::Unsupported(_) => {
                return Ok(None);
            }
        };
        if let Some(minimum) = minimum
            && u64::from(original) < minimum.get()
        {
            return Err(ReferencedCountError::Evidence(
                CountEvidenceError::Contradiction(CountContradiction {
                    object,
                    observed: original,
                    minimum: minimum.get(),
                }),
            ));
        }
        Ok(Some(ReferencedCount {
            engine,
            origin,
            object,
            original,
            incoming,
            minimum,
            field,
            allocation,
        }))
    }
}

/// An original declared count and optional original-reference facts in one analysis phase.
///
/// The observed count satisfies each positive minimum known at construction. A zero count is
/// retained when no positive minimum is known. The original field and independent complete header
/// allocation geometry are optional. The engine, root, and phase are retained for association
/// checks. Source freshness, copy-source applicability, allocation ownership, retirement
/// eligibility, and count-write permission require separate evidence.
pub(in crate::edit) struct ReferencedCount<'source> {
    engine: &'source WriteEngine,
    origin: ReconstructionSource,
    object: ObjectId,
    original: u32,
    incoming: Option<u32>,
    minimum: Option<NonZeroU64>,
    field: Option<ObjectReferenceCountField>,
    allocation: Option<HeaderCountAllocation>,
}

impl ReferencedCount<'_> {
    /// Returns whether the engine, phase identity, and any explicit phase root match.
    ///
    /// Association equality is separate from source-content validity.
    pub(in crate::edit) fn matches(&self, phase: &AnalysisPhase<'_>) -> bool {
        let ReconstructionSource {
            identity,
            object: root,
        } = &self.origin;
        core::ptr::eq(self.engine, phase.engine)
            && phase
                .identity
                .as_ref()
                .is_some_and(|other| Arc::ptr_eq(identity, other))
            && phase.root.is_none_or(|other| other == *root)
    }

    /// Returns the physical object-header identity of the observed target.
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }

    /// Returns the original physical root associated with this observation.
    pub(in crate::edit) fn root(&self) -> ObjectId {
        self.origin.object()
    }

    /// Returns the original declared count, including an implicit one or an unproven zero.
    pub(in crate::edit) fn original(&self) -> u32 {
        self.original
    }

    /// Returns the visible incoming count from complete original topology, when available.
    ///
    /// `Some(0)` indicates a known absence of visible links, including targets outside the
    /// discovered topology. `None` indicates unavailable complete topology. The root's superblock
    /// reference is excluded from this count.
    pub(in crate::edit) fn incoming(&self) -> Option<u32> {
        self.incoming
    }

    /// Returns the proven positive original-reference minimum, when known.
    ///
    /// Includes the root's superblock reference even when complete topology is unavailable.
    pub(in crate::edit) fn minimum(&self) -> Option<NonZeroU64> {
        self.minimum
    }

    /// Returns the original physical count field, or `None` for an implicit one.
    pub(in crate::edit) fn field(&self) -> Option<ObjectReferenceCountField> {
        self.field
    }

    /// Returns the independently checked complete allocation containing an explicit count, if any.
    ///
    /// The count observation is preserved when complete allocation geometry cannot be established.
    pub(in crate::edit) fn header_allocation(&self) -> Option<HeaderCountAllocation> {
        self.allocation
    }
}

/// Visible original links and their positive declared-count minimum, when known.
struct OriginalReferences {
    incoming: Option<u32>,
    minimum: Option<NonZeroU64>,
}

impl OriginalReferences {
    /// Checks original-root association and uses any cached complete topology.
    ///
    /// The root's superblock reference is included independently of topology availability.
    ///
    /// # Errors
    ///
    /// Returns [`ReferencedCountError::Association`] for failed original-root resolution or a
    /// mismatched root, engine, or phase. Returns [`ReferencedCountError::Evidence`] for an
    /// inconsistent visible incoming count in the cached topology.
    fn new(
        engine: &WriteEngine,
        origin: &ReconstructionSource,
        topology: Option<&CompleteTopology<'_>>,
        object: ObjectId,
    ) -> Result<Self, ReferencedCountError> {
        let root = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(|error| ReferencedCountError::Association(CopyUsersError::Address(error)))?;
        if origin.object() != root {
            return Err(ReferencedCountError::Association(CopyUsersError::Root {
                expected: root,
                actual: origin.object(),
            }));
        }
        if let Some(topology) = topology {
            if !core::ptr::eq(engine, topology.engine) {
                return Err(ReferencedCountError::Association(CopyUsersError::Engine));
            }
            if !Arc::ptr_eq(&origin.identity, &topology.identity) {
                return Err(ReferencedCountError::Association(CopyUsersError::Phase));
            }
            let actual = topology.graph.object(ObjectIndex::ROOT).value.id;
            if actual != root {
                return Err(ReferencedCountError::Association(CopyUsersError::Root {
                    expected: root,
                    actual,
                }));
            }
        }
        let incoming = topology
            .map(|topology| {
                topology.indices.get(&object).map_or(Ok(0), |&index| {
                    IncomingLinks::new(index, topology.incoming(object.header_address().get()))
                        .map(IncomingLinks::visible)
                })
            })
            .transpose()
            .map_err(ReferencedCountError::Evidence)?;
        let minimum = NonZeroU64::new(
            u64::from(incoming.unwrap_or(0))
                + if object == root {
                    u64::from(ROOT_REFERENCE_COUNT)
                } else {
                    0
                },
        );
        Ok(Self { incoming, minimum })
    }
}

/// An incompatible phase association or failed original-count evidence.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum ReferencedCountError {
    /// A mismatched engine, phase, or root, or failed root-address resolution.
    Association(CopyUsersError),
    /// A source-count failure, inconsistent topology, or proven count contradiction.
    Evidence(CountEvidenceError),
}

/// The count of a version 2 header without a reference-count message.
///
/// Defined in "The Object Reference Count Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_refcount
const IMPLICIT_REFERENCE_COUNT: u32 = 1;

#[cfg(test)]
mod tests;
