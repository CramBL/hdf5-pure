//! Global heap reference identities and allocated collection members for one source phase.

use std::collections::BTreeSet;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapMembers;
use hdf5_pure_format::__private::GlobalHeapObject;

use super::AnalysisPhase;
use super::ObjectReferences;
use super::ReconstructionSource;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;
use crate::global_heap;
use crate::global_heap::GlobalHeapId;
use crate::message_type::MessageType;

pub(in crate::edit) mod associations;
pub(in crate::edit) mod qualification;

impl AnalysisPhase<'_> {
    /// Returns cached references from the object associated with `source`.
    ///
    /// Collects descriptor identities on the first query without discovering group children.
    /// Retains decoded identities after incomplete interpretation or failure. Source freshness,
    /// topology, collection framing, and ownership require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `source` belongs to another
    /// phase or the facts cannot be retrieved. Source and parsing failures are recorded in
    /// [`ObjectReferences::error`].
    pub(in crate::edit) fn heap_reference_facts(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<&ObjectReferences, Error> {
        self.require_heap_phase(source)?;
        self.read_reference_facts(source.object(), false)
    }

    /// Returns cached parent and allocated-member framing for the collection at `address`.
    ///
    /// Parses the collection on its first query in this phase and records member identities and
    /// stored counts. Logical ownership, external references, and retirement permission require
    /// separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `source` belongs to another phase, or the errors of
    /// [`Self::collection_contents`] if collection parsing fails.
    pub(in crate::edit) fn heap_members(
        &mut self,
        source: &ReconstructionSource,
        address: PhysicalAddress,
    ) -> Result<&CollectionContents, FormatError> {
        self.require_heap_phase(source)?;
        self.collection_contents(address)
    }

    /// Parses collection framing once and reuses its ranges, members, or failure in this phase.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`CollectionContents::read`].
    pub(super) fn collection_contents(
        &mut self,
        address: PhysicalAddress,
    ) -> Result<&CollectionContents, FormatError> {
        self.allocations
            .entry(address)
            .or_insert_with(|| {
                #[cfg(test)]
                super::COLLECTION_ALLOCATIONS.with(|count| count.set(count.get() + 1));
                CollectionContents::read(self.engine, address)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Checks the shared source-phase association before a heap query.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `source` belongs to another phase.
    fn require_heap_phase(&self, source: &ReconstructionSource) -> Result<(), FormatError> {
        if !self
            .identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, &source.identity))
        {
            return Err(FormatError::Internal(
                "heap query belongs to another source phase".into(),
            ));
        }
        Ok(())
    }
}

impl ObjectReferences {
    /// Records interpretation completeness and failure, preserving collected identities.
    pub(in crate::edit) fn record_outcome(&mut self, outcome: Result<bool, HeapReferenceError>) {
        self.complete = matches!(outcome, Ok(true));
        self.error = outcome.err();
    }

    /// Borrows the physical collection addresses found in the object's content.
    pub(in crate::edit) fn collections(&self) -> &BTreeSet<u64> {
        &self.collections
    }

    /// Borrows the checked collection and member identities found in the object's content.
    pub(in crate::edit) fn objects(&self) -> &BTreeSet<GlobalHeapId> {
        &self.objects
    }

    /// Returns whether reference-bearing content was fully interpreted without an error.
    pub(in crate::edit) fn is_complete(&self) -> bool {
        self.complete
    }

    /// Borrows the recorded source or parsing failure, if any.
    ///
    /// Unsupported content can make interpretation incomplete without a recorded error.
    pub(in crate::edit) fn error(&self) -> Option<&HeapReferenceError> {
        self.error.as_ref()
    }
}

/// Errors retained while parsing an object's global heap references.
#[derive(Clone, Debug)]
pub(in crate::edit) enum HeapReferenceError {
    /// Malformed metadata, invalid descriptor geometry, or a source failure.
    Format(FormatError),
    /// A dataset without its required datatype or dataspace message.
    MissingMessage(MessageType),
}

impl From<FormatError> for HeapReferenceError {
    /// Wraps a metadata, descriptor, or source error for the reference cache.
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<HeapReferenceError> for Error {
    /// Converts a cached reference failure to its corresponding public error variant.
    fn from(error: HeapReferenceError) -> Self {
        match error {
            HeapReferenceError::Format(error) => error.into(),
            HeapReferenceError::MissingMessage(message) => Self::MissingMessage(message),
        }
    }
}

/// A complete physical collection allocation and its allocated-member framing.
///
/// Retains member indices, stored counts, allocation ranges, and payload ranges without
/// retaining payload bytes. Logical ownership, external references, and retirement permission
/// require separate evidence.
pub(in crate::edit) struct CollectionContents {
    framing: GlobalHeapMembers,
}

impl CollectionContents {
    /// Parses every object header and records allocated members with the parent range.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`global_heap::collect_members`], including its source-extent check
    /// for the complete allocation.
    fn read(engine: &WriteEngine, address: PhysicalAddress) -> Result<Self, FormatError> {
        let framing =
            global_heap::collect_members(engine.image(), engine.widths().lengths, address)?;
        Ok(Self { framing })
    }

    /// Returns the complete parent allocation in physical coordinates.
    pub(in crate::edit) fn range(&self) -> FileRange {
        self.framing().range()
    }

    /// Borrows the allocated object framing in collection order.
    pub(in crate::edit) fn members(&self) -> &[GlobalHeapObject] {
        self.framing().members()
    }

    /// Borrows the complete collection framing retained by this phase.
    ///
    /// Planning with this metadata requires separate source freshness and removal permission
    /// checks, as described by [`GlobalHeapMembers::compact`].
    pub(in crate::edit) fn framing(&self) -> &GlobalHeapMembers {
        &self.framing
    }
}

/// A destination for checked descriptor identities during reference decoding.
///
/// Implementations retain the collection addresses or the complete identities needed by their
/// callers. Decoding and source reads are independent of that storage choice.
pub(in crate::edit) trait HeapReferenceSink {
    /// Records the collection or member identity from one decoded descriptor.
    fn record(&mut self, id: GlobalHeapId);
}

impl HeapReferenceSink for ObjectReferences {
    fn record(&mut self, id: GlobalHeapId) {
        self.collections.insert(id.collection.get());
        self.objects.insert(id);
    }
}

impl HeapReferenceSink for BTreeSet<u64> {
    fn record(&mut self, id: GlobalHeapId) {
        self.insert(id.collection.get());
    }
}

#[cfg(test)]
mod tests;
