//! Count allocation bounds, current header preparation, and writes through the undo journal.
//!
//! Source freshness and count authority require separate evidence.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::MetadataBytesExt;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::MetadataSource;
use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountUpdate;

use crate::address::BaseAddress;
use crate::convert;
use crate::convert::Narrow;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::CountObservation;
use crate::edit::commit::JournalScope;
use crate::edit::commit::OriginalStorage;
use crate::edit::commit::PublicationError;
use crate::edit::commit::PublishedTree;
use crate::edit::retirement::PendingRootFrees;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::source;
use crate::source::Source;
use crate::width::FormatWidths;

use super::CountProtectionError;
use super::ProtectedCount;
use super::ProtectedCounts;

impl ProtectedCounts {
    /// Checks pending retirements against complete original count allocations.
    ///
    /// Rechecks the recorded session, source root, publication availability, and allocator claims.
    /// Preserves the original source association and each retirement range and class. For an empty
    /// count batch, skips engine and retirement inspection.
    ///
    /// Source freshness, confirmed publication, and permission to write require separate evidence.
    ///
    /// # Errors
    ///
    /// Propagates source and allocation-publication errors. Returns [`Error::EditUnsupported`]
    /// if the session or root differs or if allocator claims or pending retirements intersect a
    /// protected allocation.
    /// # Performance
    /// Borrows allocator lists and uses the existing count-allocation index. Uses retained metadata
    /// without source reads or topology discovery.
    pub(in crate::edit) fn before_publication<'a>(
        self,
        engine: &'a mut WriteEngine,
        retirement: PendingRootFrees,
    ) -> Result<CountPublicationPhase<'a>, Error> {
        if let Some(source) = &self.source {
            source
                .reconstructions(engine, [])
                .map_err(protection_error)?;
            self.exclude_space(engine).map_err(protection_error)?;
            if retirement
                .batch
                .entries()
                .iter()
                .any(|&(range, _)| self.overlapping(range).is_some())
            {
                return Err(allocation_conflict());
            }
        }
        Ok(CountPublicationPhase {
            engine,
            counts: self,
            retirement,
        })
    }
}

/// Protected count allocations checked against the engine and pending retirements.
///
/// Borrows the engine exclusively and retains the original retirement batch. Source freshness
/// and write permission require separate evidence.
pub(in crate::edit) struct CountPublicationPhase<'a> {
    engine: &'a mut WriteEngine,
    counts: ProtectedCounts,
    retirement: PendingRootFrees,
}

impl<'a> CountPublicationPhase<'a> {
    /// Checks a physical EOA against every complete protected allocation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `end` precedes a protected allocation's end.
    /// # Performance
    /// Checks the maximal allocation end in constant time without repeating allocator or
    /// retirement checks.
    pub(in crate::edit) fn at_end(
        self,
        end: PhysicalAddress,
    ) -> Result<CountPublicationBounds<'a>, Error> {
        self.counts.require_allocation_end(end)?;
        Ok(CountPublicationBounds { phase: self })
    }
}

/// A publication phase with a physical EOA at or beyond every protected count allocation.
///
/// Retains the exclusive engine loan and original retirement batch. The predecessor root and
/// the proposed root remain distinct.
pub(in crate::edit) struct CountPublicationBounds<'a> {
    phase: CountPublicationPhase<'a>,
}

impl CountPublicationBounds<'_> {
    /// Publishes the tree between source reference count increases and decreases.
    ///
    /// Writes increases through the commit journal before publication. Protects every original
    /// header allocation containing an updated count during publication, then reads current headers
    /// and journals decreases. The caller handles whole-commit
    /// rollback according to publication state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for changed count fields or conflicting original
    /// allocations. Returns [`Error::CommitPartiallyApplied`] if restoring count writes fails.
    /// Other count-preparation, publication, protection, and journal errors propagate.
    pub(in crate::edit) fn with_ordered_counts(
        self,
        relocations: BTreeMap<ObjectId, ObjectId>,
    ) -> Result<(), Error> {
        let mut storage = OriginalStorage::from_counts(&self.phase.counts);
        self.with_ordered_counts_excluding(relocations, &mut storage, |_| Ok(()))
            .map_err(|error| match error {
                PublicationError::Write(error) => error,
                PublicationError::Storage(_) => allocation_conflict(),
            })
    }

    /// Publishes between count increases and decreases with combined storage exclusions and rollback.
    ///
    /// # Errors
    ///
    /// Returns source-association, count-preparation, exclusion, publication or completion errors.
    pub(in crate::edit) fn with_ordered_counts_excluding<T>(
        self,
        relocations: BTreeMap<ObjectId, ObjectId>,
        storage: &mut OriginalStorage,
        complete: impl FnOnce(PublishedTree<'_>) -> Result<T, Error>,
    ) -> Result<T, PublicationError> {
        let Self { phase } = self;
        let CountPublicationPhase {
            engine,
            counts,
            retirement,
        } = phase;
        storage.include_counts(engine, &counts)?;
        if counts.updates().is_empty() {
            return engine.publish_pending_tree_excluding(
                retirement,
                relocations,
                Some(storage),
                |published| complete(published).map_err(Into::into),
            );
        }
        let increases = counts.current_updates(engine, Ordering::Greater)?;
        for update in &increases {
            for (range, bytes) in update.writes() {
                engine.write_inplace_journaled(range.start(), bytes)?;
            }
        }
        let root = retirement.root;
        engine.publish_pending_tree_excluding(retirement, relocations, Some(storage), |published| {
            let requested = published.requested_root();
            let scope = CountWriteScope::after_publication(published, root, &counts)?;
            let decreases = scope.counts.current_updates(scope.engine, Ordering::Less)?;
            let result = JournalScope::run(scope.engine, |journal| {
                for update in &decreases {
                    for (range, bytes) in update.writes() {
                        journal.write(range.start(), bytes)?;
                    }
                }
                Ok(())
            });
            if matches!(&result, Err(Error::CommitPartiallyApplied { .. })) {
                scope.engine.make_allocation_uncertain();
            }
            result?;
            complete(PublishedTree::new(scope.engine, requested)?).map_err(Into::into)
        })
    }
}

fn allocation_conflict() -> Error {
    Error::EditUnsupported("publication conflicts with an original count allocation")
}

fn protection_error(error: CountProtectionError) -> Error {
    match error {
        CountProtectionError::Source(error) => error.into(),
        CountProtectionError::Publication(error) => error,
        _ => allocation_conflict(),
    }
}

impl ProtectedCounts {
    /// Checks the maximal end of the disjoint protected allocations.
    fn require_allocation_end(&self, end: PhysicalAddress) -> Result<(), Error> {
        if self
            .updates()
            .last()
            .is_some_and(|update| end < update.allocation().range().end())
        {
            return Err(allocation_conflict());
        }
        Ok(())
    }

    /// Prepares current count and checksum writes selected by direction of change.
    ///
    /// Checks the allocation session before acquiring headers. Compares the planned remainder
    /// with the original count and prepares selected writes in allocation-address order.
    /// Releases each acquisition before reading the next header. The caller keeps the source
    /// contents stable until applying the writes.
    /// # Errors
    /// Returns [`Error::EditUnsupported`] if the session differs, or the errors from
    /// [`ProtectedCount::prepare_source`].
    fn current_updates(
        &self,
        engine: &WriteEngine,
        order: Ordering,
    ) -> Result<Vec<ObjectReferenceCountUpdate>, Error> {
        if self
            .source
            .as_ref()
            .is_some_and(|source| !source.session.same(&engine.context.allocations.session()))
        {
            return Err(changed_count_field());
        }
        let mut updates = Vec::with_capacity(self.updates().len());
        for original in self
            .updates()
            .iter()
            .filter(|update| update.count().get().cmp(&update.field().value()) == order)
        {
            updates.push(original.prepare_source(
                engine.image(),
                engine.widths(),
                engine.addresses().effective_base(),
            )?);
        }
        Ok(updates)
    }
}

impl ProtectedCount {
    /// Checks current framing and prepares the original field's positive remainder.
    ///
    /// Reads the complete protected allocation and checks the count field and allocation geometry.
    /// The checksum includes all current bytes in the containing chunk.
    /// # Errors
    /// Propagates acquisition and parse errors. Returns [`Error::EditUnsupported`] for metadata
    /// associated with a different range, changed count fields or allocations, or failed encoding.
    fn prepare_source(
        &self,
        source: &(impl Source + ?Sized),
        widths: FormatWidths,
        base: BaseAddress,
    ) -> Result<ObjectReferenceCountUpdate, Error> {
        let expected = self.field();
        let current =
            CurrentMetadata::new(source, self.allocation().range()).map_err(
                |error| match error {
                    CurrentMetadataError::Range => changed_count_field(),
                    CurrentMetadataError::Read(error) => error.into(),
                },
            )?;
        let (observed, allocation) =
            CountObservation::read_source(&current, self.object(), widths, base)?.into_parts();
        match observed {
            ObjectReferenceCount::Version1(field) | ObjectReferenceCount::Version2(field)
                if field.range() == expected.range()
                    && field.chunk() == expected.chunk()
                    && field.flags() == expected.flags() => {}
            _ => return Err(changed_count_field()),
        }
        if allocation != Some(self.allocation()) {
            return Err(changed_count_field());
        }
        expected
            .update(current.region(), self.count().get())
            .map_err(|_error| changed_count_field())
    }
}

fn changed_count_field() -> Error {
    Error::EditUnsupported("an original count field changed during commit")
}

/// An exclusive engine loan with protected count facts checked after tree publication.
struct CountWriteScope<'engine, 'counts> {
    engine: &'engine mut WriteEngine,
    counts: &'counts ProtectedCounts,
}

impl<'engine, 'counts> CountWriteScope<'engine, 'counts> {
    /// Checks the publisher's requested root, the allocation session and claims, and the adopted
    /// EOA. Source freshness requires separate evidence.
    /// # Errors
    /// Returns [`FormatError::Internal`] if publication was requested for a different root and
    /// [`Error::EditUnsupported`] for a different session, conflicting claims, or an EOA below any
    /// protected allocation. Propagates source and allocation-publication errors.
    fn after_publication(
        published: PublishedTree<'engine>,
        root: PhysicalAddress,
        counts: &'counts ProtectedCounts,
    ) -> Result<Self, Error> {
        if published.requested_root() != root {
            return Err(FormatError::Internal(
                "count updates belong to another published tree".into(),
            )
            .into());
        }
        let end = published.end();
        let engine = published.into_engine();
        if let Some(source) = &counts.source {
            if !source.session.same(&engine.context.allocations.session()) {
                return Err(allocation_conflict());
            }
            engine.context.allocations.before_reference()?;
            counts.exclude_space(engine).map_err(protection_error)?;
        }
        counts.require_allocation_end(end)?;
        Ok(Self { engine, counts })
    }
}

/// A source view that reuses one acquired metadata range.
///
/// Reads use physical file positions and the source length recorded at acquisition. The caller
/// must keep the source contents stable while using this view. Object identity, allocation
/// ownership, and permission to write require separate evidence.
///
/// # Performance
///
/// Retains one [`MetadataBytes`] owner. Contained metadata reads borrow or share its backing
/// bytes. For overlapping reads, copies the acquired bytes and reads only uncovered ranges from
/// the source. Metadata reads use [`Source::read_metadata`], and buffer reads use
/// [`Source::read_at`]. Joining an overlapping metadata read allocates a buffer as long as the read,
/// with additional ownership bookkeeping and allocations made by the source.
struct CurrentMetadata<'a, S: Source + ?Sized> {
    source: &'a S,
    bytes: MetadataBytes<'a>,
    length: u64,
}

impl<'a, S: Source + ?Sized> CurrentMetadata<'a, S> {
    /// Acquires `range` from `source` after checking its complete extent.
    ///
    /// # Errors
    ///
    /// Returns [`CurrentMetadataError::Read`] if the range extends past the source length, its
    /// length exceeds [`usize::MAX`], or the source read fails. Returns
    /// [`CurrentMetadataError::Range`] if the returned metadata has a different physical range.
    fn new(source: &'a S, range: FileRange) -> Result<Self, CurrentMetadataError> {
        let length = source.len();
        if range.end().get() > length {
            return Err(CurrentMetadataError::Read(source::unexpected_eof(
                range, length,
            )));
        }
        let bytes = source
            .read_metadata(
                range.start().get(),
                range
                    .length()
                    .to_usize()
                    .map_err(CurrentMetadataError::Read)?,
            )
            .map_err(CurrentMetadataError::Read)?;
        if bytes.range() != range {
            return Err(CurrentMetadataError::Range);
        }
        Ok(Self {
            source,
            bytes,
            length,
        })
    }

    /// Borrows the acquired bytes with their physical range.
    fn region(&self) -> FileRegion<'_> {
        self.bytes.region()
    }

    fn external(&self, range: FileRange) -> Result<MetadataBytes<'_>, FormatError> {
        let bytes = self
            .source
            .read_metadata(range.start().get(), range.length().to_usize()?)?;
        FileRegion::new(range, bytes.as_ref())?;
        bytes.region().subregion(range)?;
        Ok(bytes)
    }

    fn request(&self, offset: u64, length: usize) -> Result<FileRange, FormatError> {
        let range = FileRange::new(PhysicalAddress::new(offset), length.to_u64())?;
        if range.end().get() > self.length {
            return Err(source::unexpected_eof(range, self.length));
        }
        Ok(range)
    }
}

impl<S: Source + ?Sized> MetadataSource for CurrentMetadata<'_, S> {
    type Bytes<'a>
        = MetadataBytes<'a>
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, length: usize) -> Result<Self::Bytes<'_>, FormatError> {
        let request = self.request(offset, length)?;
        let Self {
            source: _,
            bytes,
            length: _,
        } = self;
        if length == 0 {
            return MetadataBytes::borrowed(request, &[]);
        }
        if bytes.range().contains(request) {
            return bytes.subregion(request);
        }
        let Some(parts) = ReadParts::new(request, bytes.range())? else {
            return self.external(request);
        };
        let ReadParts {
            prefix,
            cached,
            suffix,
        } = parts;
        let mut joined = vec![0; length];
        for part in [prefix, suffix] {
            if part.length() > 0 {
                let current = self.external(part)?;
                local_mut(&mut joined, request, part)?.copy_from_slice(current.as_ref());
            }
        }
        local_mut(&mut joined, request, cached)?
            .copy_from_slice(bytes.region().subregion(cached)?.bytes());
        MetadataBytes::owned(request, joined)
    }

    fn len(&self) -> u64 {
        self.length
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        let request = self.request(offset, buffer.len())?;
        let Self {
            source,
            bytes,
            length: _,
        } = self;
        if buffer.is_empty() {
            return Ok(());
        }
        let Some(parts) = ReadParts::new(request, bytes.range())? else {
            return source.read_at(offset, buffer);
        };
        let ReadParts {
            prefix,
            cached,
            suffix,
        } = parts;
        if prefix.length() > 0 {
            source.read_at(prefix.start().get(), local_mut(buffer, request, prefix)?)?;
        }
        local_mut(buffer, request, cached)?
            .copy_from_slice(bytes.region().subregion(cached)?.bytes());
        if suffix.length() > 0 {
            source.read_at(suffix.start().get(), local_mut(buffer, request, suffix)?)?;
        }
        Ok(())
    }

    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.read_metadata(offset, length)?.into_vec())
    }
}

/// A request split into an acquired intersection and uncovered prefix and suffix.
struct ReadParts {
    prefix: FileRange,
    cached: FileRange,
    suffix: FileRange,
}

impl ReadParts {
    fn new(request: FileRange, acquired: FileRange) -> Result<Option<Self>, FormatError> {
        let Some(cached) = request.intersection(acquired) else {
            return Ok(None);
        };
        Ok(Some(Self {
            prefix: FileRange::new(
                request.start(),
                cached.start().get() - request.start().get(),
            )?,
            cached,
            suffix: FileRange::new(cached.end(), request.end().get() - cached.end().get())?,
        }))
    }
}

/// An unsuccessful source read or a metadata range mismatch during acquisition.
#[derive(Debug)]
enum CurrentMetadataError {
    /// The source returned metadata for a different physical range.
    Range,
    /// An extent, local-length conversion, or source-read error.
    Read(FormatError),
}

fn local_mut(
    buffer: &mut [u8],
    whole: FileRange,
    part: FileRange,
) -> Result<&mut [u8], FormatError> {
    FileRegion::new(whole, buffer)?.subregion(part)?;
    let range = convert::slice_range(part.start().get() - whole.start().get(), part.length())?;
    buffer
        .get_mut(range)
        .ok_or(FormatError::InsufficientRegion {
            required_start: part.start().get(),
            required_length: part.length(),
            provided_start: whole.start().get(),
            provided_length: whole.length(),
        })
}

#[cfg(test)]
mod tests;
