//! Declared counts checked against visible hard links and the root's superblock reference.
//!
//! Header-subject classification and topology identity are retained with each bound. Allocation
//! ownership, reference completeness, source freshness, and publication require separate proofs.

use core::num::NonZeroU32;
use std::collections::BTreeMap;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::MetadataSource;
use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountField;
use hdf5_pure_format::__private::ObjectReferenceCountObserver;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::edit::WriteEngine;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::source::SourceMetadata;
use crate::width::FormatWidths;

use super::AnalysisPhase;
use super::CompleteTopology;
use super::deletions::TopologyIdentity;

mod added_users;

impl<'source> AnalysisPhase<'source> {
    /// Returns a positive declared count consistent with this topology's reference minimum.
    ///
    /// Requires complete topology and an object within it. Returns `None` for missing applicable
    /// topology, outside objects, and ambiguous or unsupported observations. Includes the
    /// superblock reference in the root's minimum and returns [`CountEvidence::Retained`] for
    /// the root. Preserves excess counts and the original physical field, when present.
    /// Header-subject classification remains separate from the numeric bound.
    ///
    /// # Errors
    ///
    /// Returns [`CountEvidenceError::Contradiction`] if the observed count is below the applicable
    /// minimum, [`CountEvidenceError::Topology`] if a discovered non-root object
    /// lacks a positive incoming count, or [`CountEvidenceError::Source`] for errors from
    /// [`Self::object_reference_count`].
    pub(in crate::edit) fn count_evidence(
        &mut self,
        object: ObjectId,
    ) -> Result<Option<CountEvidence<'source>>, CountEvidenceError> {
        self.topology();
        let Self {
            engine,
            identity: _,
            topology,
            counts,
            budget: _,
            root: _,
            links: _,
            references: _,
            liveness: _,
            allocations: _,
            #[cfg(test)]
                candidate_lookups: _,
        } = self;
        CountQueries::new(engine, counts, topology.as_ref().and_then(Option::as_ref))
            .evidence(object)
    }
}

/// Lazy count queries borrowing the observation cache separately from the phase topology.
///
/// The caller can query counts while retaining a bound deletion batch or its removal plan.
/// Observations describe original source counts, independently of graph-relative remainders.
/// Complete header allocation geometry is retained separately from count classification.
pub(in crate::edit) struct CountQueries<'cache, 'source> {
    engine: &'source WriteEngine,
    counts: &'cache mut BTreeMap<ObjectId, Result<CountObservation, FormatError>>,
    topology: Option<&'cache CompleteTopology<'source>>,
}

impl<'cache, 'source> CountQueries<'cache, 'source> {
    /// Borrows a source-phase cache and defers count reads until queried.
    ///
    /// The caller passes the cache for `engine`'s unchanged source phase. Evidence queries check
    /// the engine identity of any applicable `topology`.
    pub(super) fn new(
        engine: &'source WriteEngine,
        counts: &'cache mut BTreeMap<ObjectId, Result<CountObservation, FormatError>>,
        topology: Option<&'cache CompleteTopology<'source>>,
    ) -> Self {
        Self {
            engine,
            counts,
            topology,
        }
    }

    /// Returns a cached raw count observation, with topology optional.
    ///
    /// Uses the complete-parse and failure-caching contract of
    /// [`AnalysisPhase::object_reference_count`].
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`AnalysisPhase::object_reference_count`].
    pub(in crate::edit) fn observation(
        &mut self,
        object: ObjectId,
    ) -> Result<ObjectReferenceCount, FormatError> {
        self.summary(object).map(|summary| summary.count)
    }

    /// Checks a source count against the original reference minimum and subject classification.
    ///
    /// Uses the numeric and applicability requirements of [`AnalysisPhase::count_evidence`].
    /// Engine identity is checked before producing evidence. Source-content validity remains
    /// the caller's responsibility.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AnalysisPhase::count_evidence`], or
    /// [`CountEvidenceError::Topology`] if an applicable topology belongs to another engine.
    pub(in crate::edit) fn evidence(
        &mut self,
        object: ObjectId,
    ) -> Result<Option<CountEvidence<'source>>, CountEvidenceError> {
        let Self {
            engine,
            counts: _,
            topology,
        } = self;
        let query_engine = *engine;
        let Some(topology) = *topology else {
            return Ok(None);
        };
        let CompleteTopology {
            identity,
            engine,
            graph,
            indices,
            incoming,
        } = topology;
        let Some(&index) = indices.get(&object) else {
            return Ok(None);
        };
        let incoming = IncomingLinks::new(index, incoming.get(&object).copied())?;
        let subject = if index == ObjectIndex::ROOT {
            None
        } else {
            graph.object(index).value.count_subject
        };
        let identity = Arc::clone(identity);
        let source = *engine;
        if !std::ptr::eq(query_engine, source) {
            return Err(CountEvidenceError::Topology(FormatError::Internal(
                "count queries use another source engine".into(),
            )));
        }
        let CountObservation {
            count: observed,
            allocation,
        } = self.summary(object).map_err(CountEvidenceError::Source)?;
        let (original, field) = match observed {
            ObjectReferenceCount::ImplicitOne => (1, None),
            ObjectReferenceCount::Version1(field) | ObjectReferenceCount::Version2(field) => {
                (field.value(), Some(field))
            }
            ObjectReferenceCount::Ambiguous | ObjectReferenceCount::Unsupported(_) => {
                return Ok(None);
            }
        };
        let original = NonZeroU32::new(original)
            .filter(|count| u64::from(count.get()) >= incoming.minimum())
            .ok_or(CountEvidenceError::Contradiction(CountContradiction {
                object,
                observed: original,
                minimum: incoming.minimum(),
            }))?;
        let bound = ObjectReferenceBound {
            source,
            topology: identity,
            object,
            original,
            incoming: incoming.visible(),
            field,
            allocation,
        };
        Ok(Some(match subject {
            Some(subject) => CountEvidence::Ordinary(OrdinaryObjectCount { bound, subject }),
            None => CountEvidence::Retained(bound),
        }))
    }

    /// Returns a cached count classification and its independent optional header geometry.
    ///
    /// Reads the complete header on the first query and caches successful summaries and failures.
    ///
    /// # Errors
    ///
    /// Returns the parsing and observation errors of [`CountObservation::read`].
    fn summary(&mut self, object: ObjectId) -> Result<CountObservation, FormatError> {
        let Self {
            engine,
            counts,
            topology: _,
        } = self;
        counts
            .entry(object)
            .or_insert_with(|| CountObservation::read(engine, object))
            .as_ref()
            .copied()
            .map_err(Clone::clone)
    }
}

/// A source-declared count bound with its separate header-subject classification.
pub(in crate::edit) enum CountEvidence<'source> {
    /// A bound paired with a recognized dataset or group header shape.
    Ordinary(OrdinaryObjectCount<'source>),
    /// A bound for the root or a header without a recognized dataset or group shape.
    Retained(ObjectReferenceBound<'source>),
}

impl<'source> CountEvidence<'source> {
    /// Borrows the original declared count and visible incoming count for either subject class.
    pub(in crate::edit) fn bound(&self) -> &ObjectReferenceBound<'source> {
        match self {
            Self::Ordinary(ordinary) => &ordinary.bound,
            Self::Retained(bound) => bound,
        }
    }
}

/// A declared count bound for a header with the recognized dataset or group message shape.
///
/// The classification describes message presence and multiplicity. Allocation geometry and
/// retirement eligibility require separate evidence.
pub(in crate::edit) struct OrdinaryObjectCount<'source> {
    bound: ObjectReferenceBound<'source>,
    subject: OrdinaryCountSubject,
}

impl OrdinaryObjectCount<'_> {
    /// Returns the recognized header shape used to classify this count.
    pub(in crate::edit) fn subject(&self) -> OrdinaryCountSubject {
        self.subject
    }
}

/// A positive declared count at least as large as the reference minimum in one topology.
///
/// The minimum includes visible hard links and, for the root, the superblock reference.
/// Retains the original value and physical field, or an implicit one without a field. Both the
/// engine and topology identities are retained. Optional complete header allocation geometry is
/// checked independently of the count and header-subject classification. References from outside
/// the discovered graph, source consistency, allocation ownership, reference completeness, and
/// publication conditions require separate proofs.
pub(in crate::edit) struct ObjectReferenceBound<'source> {
    source: &'source WriteEngine,
    topology: Arc<TopologyIdentity>,
    object: ObjectId,
    original: NonZeroU32,
    incoming: u32,
    field: Option<ObjectReferenceCountField>,
    allocation: Option<HeaderCountAllocation>,
}

impl ObjectReferenceBound<'_> {
    /// Returns whether both the engine and topology identities match.
    ///
    /// Identity equality is separate from source-content validity.
    pub(in crate::edit) fn matches(&self, topology: &CompleteTopology<'_>) -> bool {
        Arc::ptr_eq(&self.topology, &topology.identity)
            && std::ptr::eq(self.source, topology.engine)
    }

    /// Returns the physical object-header identity associated with the counts.
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }

    /// Returns the original positive declared count, including an implicit one.
    pub(in crate::edit) fn original(&self) -> u32 {
        self.original.get()
    }

    /// Returns the visible incoming count in the matching root-reachable topology.
    ///
    /// The root can have zero visible links. Its superblock reference is excluded from this count.
    pub(in crate::edit) fn incoming(&self) -> u32 {
        self.incoming
    }

    /// Returns the original physical count field, or `None` for an implicit one.
    pub(in crate::edit) fn field(&self) -> Option<ObjectReferenceCountField> {
        self.field
    }

    /// Returns the independently checked complete allocation containing an explicit count, if any.
    ///
    /// The count bound is preserved when allocation geometry cannot be established.
    pub(in crate::edit) fn header_allocation(&self) -> Option<HeaderCountAllocation> {
        self.allocation
    }
}

/// Visible incoming links with a separate declared-count minimum for the root.
#[derive(Clone, Copy)]
enum IncomingLinks {
    Object(NonZeroU32),
    Root(u32),
}

impl IncomingLinks {
    /// Establishes a positive visible count for a non-root object or a possibly zero root count.
    ///
    /// # Errors
    ///
    /// Returns [`CountEvidenceError::Topology`] if a non-root count is missing or zero.
    fn new(index: ObjectIndex, incoming: Option<u32>) -> Result<Self, CountEvidenceError> {
        if index == ObjectIndex::ROOT {
            return Ok(Self::Root(incoming.unwrap_or(0)));
        }
        incoming
            .and_then(NonZeroU32::new)
            .map(Self::Object)
            .ok_or_else(|| {
                CountEvidenceError::Topology(FormatError::Internal(
                    "a discovered non-root object has no positive incoming count".into(),
                ))
            })
    }

    /// Returns the visible hard-link count without the root's superblock reference.
    fn visible(self) -> u32 {
        match self {
            Self::Object(count) => count.get(),
            Self::Root(count) => count,
        }
    }

    /// Returns the declared-count minimum, including the root's superblock reference.
    fn minimum(self) -> u64 {
        match self {
            Self::Object(count) => u64::from(count.get()),
            Self::Root(count) => u64::from(count) + u64::from(ROOT_REFERENCE_COUNT),
        }
    }
}

/// The additional declared reference to the root group from the superblock.
// `H5G_mkroot` adds the superblock reference in `H5Groot.c`, HDF5 1.14.6.
pub(in crate::edit) const ROOT_REFERENCE_COUNT: u32 = 1;

/// A count contradiction, source observation failure, or internal topology inconsistency.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum CountEvidenceError {
    /// An observed count below the visible-link minimum, including the root's superblock reference.
    Contradiction(CountContradiction),
    /// A parsing or source-read failure while observing the declared count.
    Source(FormatError),
    /// An inconsistent topology or count state, or mismatched source-bound evidence.
    Topology(FormatError),
}

impl From<FormatError> for CountEvidenceError {
    /// Classifies a structural planning failure as a topology error.
    fn from(error: FormatError) -> Self {
        Self::Topology(error)
    }
}

/// The physical object and counts that establish an observed undercount.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) struct CountContradiction {
    /// The object whose declared count is insufficient.
    pub(in crate::edit) object: ObjectId,
    /// The original declared count, including an implicit one.
    pub(in crate::edit) observed: u32,
    /// The visible-link minimum, including the root's additional superblock reference.
    pub(in crate::edit) minimum: u64,
}

/// A dataset or group message shape recognized from the already parsed header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum OrdinaryCountSubject {
    Dataset,
    Group,
}

impl OrdinaryCountSubject {
    /// Classifies a group marker without dataset descriptors, or a unique dataset descriptor set.
    ///
    /// Requires exactly one datatype, dataspace, and layout message without a group marker for a
    /// dataset. Returns `None` for all other shapes. Interprets message types without parsing
    /// their bodies again.
    pub(super) fn from_header(header: &ObjectHeader) -> Option<Self> {
        let mut datatype = 0_u8;
        let mut dataspace = 0_u8;
        let mut layout = 0_u8;
        for message in &header.messages {
            let count = match message.msg_type {
                MessageType::DATATYPE => &mut datatype,
                MessageType::DATASPACE => &mut dataspace,
                MessageType::DATA_LAYOUT => &mut layout,
                _ => continue,
            };
            *count = count.saturating_add(1);
        }
        match (
            crate::group::v2::is_group(header),
            datatype,
            dataspace,
            layout,
        ) {
            (true, 0, 0, 0) => Some(Self::Group),
            (false, 1, 1, 1) => Some(Self::Dataset),
            _ => None,
        }
    }
}

/// A count classification and an independently checked optional header allocation.
///
/// The count classification is preserved when complete allocation geometry cannot be established.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CountObservation {
    count: ObjectReferenceCount,
    allocation: Option<HeaderCountAllocation>,
}

impl CountObservation {
    /// Returns the count classification and its independently checked optional allocation.
    pub(super) fn into_parts(self) -> (ObjectReferenceCount, Option<HeaderCountAllocation>) {
        let Self { count, allocation } = self;
        (count, allocation)
    }

    /// Reads the count and allocation summaries from the engine's current image.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Self::read_source`].
    fn read(engine: &WriteEngine, object: ObjectId) -> Result<Self, FormatError> {
        Self::read_source(
            &SourceMetadata(&engine.image()),
            object,
            engine.widths(),
            engine.addresses().effective_base(),
        )
    }

    /// Reads count and allocation summaries through one complete object-header parse.
    ///
    /// Collects declared chunk ranges temporarily and retains only the count-containing allocation
    /// after checking their geometry. Requires the full parse and every count observation to
    /// succeed before returning either summary.
    ///
    /// # Errors
    ///
    /// Returns errors from [`ObjectHeader::parse_from_source_with_chunk_observer`],
    /// [`ObjectReferenceCountObserver::observe`], or [`ObjectReferenceCountObserver::finish`].
    pub(super) fn read_source(
        source: &(impl MetadataSource + ?Sized),
        object: ObjectId,
        widths: FormatWidths,
        base: BaseAddress,
    ) -> Result<Self, FormatError> {
        let observer = ObjectReferenceCountObserver::new(object.header_address());
        let mut observation_error = None;
        let mut allocations = Vec::new();
        let header = ObjectHeader::parse_from_source_with_chunk_observer(
            source,
            AccessMode::ReadWrite,
            object.header_address(),
            widths,
            base,
            MessageFilter::Only(&mut |kind, _| {
                observer.observe_message_type(kind);
                false
            }),
            &mut |chunk| {
                allocations.push(chunk.allocation());
                if observation_error.is_none() {
                    observation_error = observer.observe(chunk).err();
                }
            },
        )?;
        if let Some(error) = observation_error {
            return Err(error);
        }
        let count = observer.finish(&header)?;
        let allocation = match count {
            ObjectReferenceCount::Version1(field) | ObjectReferenceCount::Version2(field) => {
                HeaderCountAllocation::new(field, &mut allocations, source.len())
            }
            ObjectReferenceCount::Ambiguous
            | ObjectReferenceCount::ImplicitOne
            | ObjectReferenceCount::Unsupported(_) => None,
        };
        Ok(Self { count, allocation })
    }
}

/// A complete declared header chunk containing a supported physical count field.
///
/// The object's declared header chunks are nonempty, mutually disjoint, and within the source
/// extent observed at construction. This range exactly matches the field's containing chunk,
/// including framing, padding, and any checksum. Other objects' storage, allocation availability,
/// source freshness, and write permissions require separate evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) struct HeaderCountAllocation {
    range: FileRange,
}

impl HeaderCountAllocation {
    /// Checks declared chunk geometry and selects the complete count-containing allocation.
    ///
    /// The caller passes every complete chunk allocation from one successful header parse.
    /// Sorts `allocations` once. Returns `None` if a range is empty, past `eof`, or overlapping,
    /// or if the field's complete chunk is missing or inconsistent with its field range.
    fn new(
        field: ObjectReferenceCountField,
        allocations: &mut [FileRange],
        eof: u64,
    ) -> Option<Self> {
        allocations.sort_unstable_by_key(|range| range.start());
        if allocations
            .iter()
            .any(|range| range.length() == 0 || range.end() > PhysicalAddress::new(eof))
            || allocations.windows(2).any(|pair| pair[0].overlaps(pair[1]))
        {
            return None;
        }
        let index = allocations
            .binary_search_by_key(&field.chunk().start(), |range| range.start())
            .ok()?;
        let &range = allocations.get(index)?;
        (range == field.chunk() && range.contains(field.range())).then_some(Self { range })
    }

    /// Returns the complete physical allocation of the count-containing header chunk.
    pub(in crate::edit) fn range(self) -> FileRange {
        self.range
    }
}

#[cfg(test)]
mod tests;
