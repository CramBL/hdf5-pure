//! Local allocation geometry for one source phase.

use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

use super::AnalysisPhase;
use super::ReconstructionSource;
use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::source;
use crate::source::SourceMetadata;

impl AnalysisPhase<'_> {
    /// Returns cached local allocation facts for the object associated with `source`.
    ///
    /// Collects the facts on the first matching query without discovering group children.
    /// Caches complete and partial results, including failures, for this source phase. Source
    /// freshness, topology, reference completeness, and retirement permission require separate
    /// evidence.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `source` belongs to another phase. Source, parsing,
    /// and allocation geometry failures are recorded in the returned [`ObjectOwnership`].
    pub(in crate::edit) fn object_ownership(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<&ObjectOwnership, FormatError> {
        if !self
            .identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, &source.identity))
        {
            return Err(FormatError::Internal(
                "ownership query belongs to another source phase".into(),
            ));
        }
        Ok(self
            .ownership
            .entry(source.object())
            .or_insert_with(|| ObjectOwnership::read(self.engine, source.object())))
    }
}

/// Complete physical allocations identified through one object's header.
///
/// The ranges are nonempty, disjoint, and within the observed source extent. Partial or failed
/// collection can retain independently established allocations. The outcome describes local
/// storage support. The caller accounts for referenced global heap collections separately.
/// Topology, reference completeness, allocation availability, and retirement permission require
/// separate evidence.
pub(in crate::edit) struct ObjectOwnership {
    allocations: Vec<ObjectAllocation>,
    outcome: OwnershipOutcome,
}

impl ObjectOwnership {
    /// Collects local allocations and records unsupported storage or failures.
    ///
    /// Returns an empty allocation list if the header parse or its geometry check fails. Preserves
    /// checked header allocations if body geometry is invalid. Retains independently collected
    /// body ranges after other body failures when their geometry is valid.
    fn read(engine: &WriteEngine, object: ObjectId) -> Self {
        #[cfg(test)]
        OWNERSHIP_READS.with(|reads| reads.set(reads.get() + 1));
        let (header, mut allocations) = match ObjectAllocation::read_header(engine, object) {
            Ok(parsed) => parsed,
            Err(error) => {
                return Self {
                    allocations: Vec::new(),
                    outcome: OwnershipOutcome::Failed(error),
                };
            }
        };
        let headers = allocations.len();
        let mut raw = Vec::new();
        let mut metadata = Vec::new();
        let collected = engine.collect_object_owned_spans(&header, &mut raw, &mut metadata, None);
        allocations.extend(metadata.into_iter().map(|range| ObjectAllocation {
            range,
            kind: AllocationKind::Metadata,
        }));
        allocations.extend(raw.into_iter().map(|range| ObjectAllocation {
            range,
            kind: AllocationKind::Data,
        }));
        let outcome = match ObjectAllocation::check_geometry(
            &allocations,
            engine.image().len(),
            engine.superblock().eof_address,
        ) {
            Err(error) => {
                allocations.truncate(headers);
                OwnershipOutcome::Failed(error)
            }
            Ok(()) => match collected {
                Ok(true) => OwnershipOutcome::Complete,
                Ok(false) => OwnershipOutcome::Partial,
                Err(error) => OwnershipOutcome::Failed(error),
            },
        };
        Self {
            allocations,
            outcome,
        }
    }

    /// Borrows the complete physical ranges retained by local collection.
    pub(in crate::edit) fn allocations(&self) -> &[ObjectAllocation] {
        &self.allocations
    }

    /// Returns whether local collection completed without errors or unsupported metadata.
    pub(in crate::edit) fn is_complete(&self) -> bool {
        matches!(self.outcome, OwnershipOutcome::Complete)
    }

    /// Borrows the source, parsing, or geometry failure, if any.
    ///
    /// Unsupported metadata can make collection incomplete without a recorded error.
    pub(in crate::edit) fn error(&self) -> Option<&Error> {
        match &self.outcome {
            OwnershipOutcome::Failed(error) => Some(error),
            OwnershipOutcome::Complete | OwnershipOutcome::Partial => None,
        }
    }
}

/// The local collection outcome, independent of the retained allocation ranges.
enum OwnershipOutcome {
    /// Every local storage collector completed without an error or unsupported metadata.
    Complete,
    /// Parsing or geometry failed, with independent ranges retained separately.
    Failed(Error),
    /// Shared messages, unrecognized messages, or unsupported storage made collection incomplete.
    Partial,
}

/// A complete physical allocation classified by its content.
///
/// [`AllocationKind`] describes the stored content. PAGE admission classes and allocation
/// availability require separate evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) struct ObjectAllocation {
    range: FileRange,
    kind: AllocationKind,
}

impl ObjectAllocation {
    /// Parses an object header and records complete physical chunk allocations.
    ///
    /// # Errors
    ///
    /// Returns parsing and source errors from
    /// [`ObjectHeader::parse_from_source_with_chunk_observer`], or geometry errors from
    /// [`Self::check_geometry`].
    fn read_header(
        engine: &WriteEngine,
        object: ObjectId,
    ) -> Result<(ObjectHeader, Vec<Self>), Error> {
        let mut allocations = Vec::new();
        let header = ObjectHeader::parse_from_source_with_chunk_observer(
            &SourceMetadata(&engine.image()),
            AccessMode::ReadWrite,
            object.header_address(),
            engine.widths(),
            engine.addresses().effective_base(),
            MessageFilter::All,
            &mut |chunk| {
                allocations.push(Self {
                    range: chunk.allocation(),
                    kind: AllocationKind::Header,
                });
            },
        )?;
        Self::check_geometry(
            &allocations,
            engine.image().len(),
            engine.superblock().eof_address,
        )?;
        Ok((header, allocations))
    }

    /// Checks nonempty, disjoint ranges against the physical source extent.
    ///
    /// Uses `logical_size` only in invalid-geometry diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if a range exceeds `length`. Returns
    /// [`Error::InvalidSpaceOwnershipExtent`] if a range is empty or overlaps an earlier range.
    fn check_geometry(allocations: &[Self], length: u64, logical_size: u64) -> Result<(), Error> {
        let mut ordered = allocations.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|allocation| allocation.range.start());
        let mut previous_end = PhysicalAddress::new(0);
        for allocation in ordered {
            let range = allocation.range;
            if range.end() > PhysicalAddress::new(length) {
                return Err(source::unexpected_eof(range, length).into());
            }
            if range.length() == 0 || range.start() < previous_end {
                return Err(Error::InvalidSpaceOwnershipExtent {
                    address: range.start().get(),
                    length: range.length(),
                    previous_end: previous_end.get(),
                    logical_size,
                });
            }
            previous_end = range.end();
        }
        Ok(())
    }

    /// Returns the full allocation range in physical coordinates.
    pub(in crate::edit) fn range(self) -> FileRange {
        self.range
    }

    /// Returns the content category used independently of PAGE classification.
    pub(in crate::edit) fn kind(self) -> AllocationKind {
        self.kind
    }
}

/// The physical content of a collected allocation.
///
/// PAGE admission classes are established separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum AllocationKind {
    /// Dataset storage, including encoded chunks.
    Data,
    /// Object-header storage, including continuation chunks.
    Header,
    /// Indexes, heaps, and other metadata outside the object header.
    Metadata,
}

#[cfg(test)]
std::thread_local! {
    static OWNERSHIP_READS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests;
