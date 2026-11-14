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
use hdf5_pure_format::__private::ObjectReferenceCountField;
use hdf5_pure_format::__private::ObjectReferenceCountUpdate;
use hdf5_pure_format::__private::ObjectReferenceCountUpdateError;
use hdf5_pure_space::__private::FreeClass;

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

#[cfg(test)]
mod test_util;

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
    /// Returns [`CountPublicationError::Protection`] for an inapplicable session or root,
    /// unavailable publication, conflicting allocator claims, or address-conversion errors.
    /// Returns [`CountPublicationError::Retirement`] if a pending retirement intersects a
    /// protected allocation.
    ///
    /// # Performance
    ///
    /// Borrows allocator lists and uses the existing count-allocation index. Uses retained metadata
    /// without source reads or topology discovery.
    pub(in crate::edit) fn before_publication<'a>(
        self,
        engine: &'a mut WriteEngine,
        retirement: PendingRootFrees,
    ) -> Result<CountPublicationPhase<'a>, CountPublicationError> {
        if let Some(source) = &self.source {
            source
                .reconstructions(engine, [])
                .map_err(CountPublicationError::Protection)?;
            self.exclude_space(engine)
                .map_err(CountPublicationError::Protection)?;
            for &(range, class) in retirement.batch.entries() {
                if let Some(update) = self.overlapping(range) {
                    return Err(CountPublicationError::Retirement {
                        object: update.object(),
                        allocation: update.allocation().range(),
                        retired: range,
                        class,
                    });
                }
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
/// The engine is exclusively borrowed, and the original retirement batch is retained. The caller
/// can propose another EOA while retaining those associations. Dropping the phase leaves
/// allocator state unchanged. Source freshness and write permission require separate evidence.
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
    /// Returns [`CountEndFailure`] with [`CountPublicationError::Truncation`] if `end` precedes
    /// a protected allocation's end. The caller can recover the phase and propose another EOA.
    ///
    /// # Performance
    ///
    /// Checks the maximal allocation end in constant time. On success, retains the existing phase
    /// without allocating or repeating allocator and retirement checks. On failure, boxes the
    /// phase and error.
    pub(in crate::edit) fn at_end(
        self,
        end: PhysicalAddress,
    ) -> Result<CountPublicationBounds<'a>, Box<CountEndFailure<'a>>> {
        if let Err(error) = CountAllocationEnd::new(&self.counts, end) {
            return Err(Box::new(CountEndFailure { phase: self, error }));
        }
        Ok(CountPublicationBounds { phase: self, end })
    }
}

/// A physical EOA at or beyond every complete protected count allocation.
///
/// The engine borrow and original retirement batch are retained through [`CountPublicationPhase`].
/// The predecessor source root and prospective publication root remain distinct.
/// The editor consumes the original retirement batch through [`Self::with_ordered_counts`] or
/// [`Self::with_ordered_counts_excluding`].
pub(in crate::edit) struct CountPublicationBounds<'a> {
    phase: CountPublicationPhase<'a>,
    end: PhysicalAddress,
}

impl<'a> CountPublicationBounds<'a> {
    /// Prepares every protected count update after parsing the current complete headers.
    ///
    /// Uses [`CurrentCountUpdates::new`] and borrows these bounds exclusively. Retains the original
    /// retirement batch and EOA. Leaves the bounds and file bytes unchanged on error.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`CurrentCountUpdates::new`].
    #[cfg(test)]
    pub(in crate::edit) fn prepare_counts(
        &mut self,
    ) -> Result<CurrentCountUpdates<'_>, CountPreparationError> {
        let CountPublicationPhase {
            engine,
            counts,
            retirement: _,
        } = &mut self.phase;
        CurrentCountUpdates::new(engine, counts)
    }

    /// Publishes the tree between source reference count increases and decreases.
    ///
    /// Writes increases through the commit journal before publication. Protects every original
    /// header allocation containing an updated count during publication, then reads current headers
    /// and journals decreases before calling `complete`. An empty count batch uses
    /// [`WriteEngine::publish_pending_tree_with_completion`]. The caller handles whole-commit
    /// rollback according to publication state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for changed count fields or conflicting original
    /// allocations. Returns [`Error::CommitPartiallyApplied`] if restoring count writes fails.
    /// Other count-preparation, publication, protection, journal, and completion errors propagate.
    pub(in crate::edit) fn with_ordered_counts<T>(
        self,
        relocations: BTreeMap<ObjectId, ObjectId>,
        complete: impl FnOnce(PublishedTree<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if self.phase.counts.updates().is_empty() {
            let Self { phase, end: _ } = self;
            return phase.engine.publish_pending_tree_with_completion(
                phase.retirement,
                relocations,
                complete,
            );
        }
        let mut storage = OriginalStorage::from_counts(&self.phase.counts);
        self.with_ordered_counts_excluding(relocations, &mut storage, |published, _| {
            complete(published)
        })
        .map_err(|error| match error {
            PublicationError::Write(error) => error,
            PublicationError::Storage(_) => {
                Error::EditUnsupported("publication conflicts with an original count allocation")
            }
        })
    }

    /// Publishes between source-count increases and decreases with `storage` exclusions.
    ///
    /// Adds complete original count-header allocations to `storage`, retaining identical entries.
    /// Checks count-source associations and refines unobserved intervals while preserving their
    /// surrounding portions through [`OriginalStorage::include_counts`].
    /// Journals increases before publication and decreases after publication, before calling
    /// `complete`. The publisher applies the supplied storage exclusions even for an empty count
    /// batch. The callback borrows the combined exclusions after publication and count writes
    /// succeed.
    ///
    /// The caller handles rollback according to publication state. Marks allocation state
    /// uncertain if restoring a post-publication count write fails.
    ///
    /// # Errors
    ///
    /// Returns [`PublicationError::Storage`] if count-header inclusion or publication exclusion
    /// checks fail, including incompatible count-source associations, allocator or retirement
    /// overlaps, and an end below protected storage.
    /// Returns [`PublicationError::Write`] for count-preparation, protection, journal, completion,
    /// or other publication errors.
    pub(in crate::edit) fn with_ordered_counts_excluding<T>(
        self,
        relocations: BTreeMap<ObjectId, ObjectId>,
        storage: &mut OriginalStorage,
        complete: impl FnOnce(PublishedTree<'_>, &OriginalStorage) -> Result<T, Error>,
    ) -> Result<T, PublicationError> {
        let Self { phase, end } = self;
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
                |published| complete(published, storage).map_err(Into::into),
            );
        }
        let increases = CurrentCountUpdates::select(engine, &counts, Some(Ordering::Greater))
            .map_err(Error::from)?;
        for update in &increases {
            for (range, bytes) in update.writes() {
                engine.write_inplace_journaled(range.start(), bytes)?;
            }
        }
        let request = CountCompletion {
            counts: &counts,
            root: retirement.root,
            proposed_end: end,
        };
        engine.publish_pending_tree_excluding(retirement, relocations, Some(storage), |published| {
            let requested = published.requested_root();
            let scope = request
                .into_write_scope(published)
                .map_err(|error| match error {
                    CountCompletionError::Protection(error) => Error::from(error),
                    CountCompletionError::Root { .. } => FormatError::Internal(
                        "count updates belong to another published tree".into(),
                    )
                    .into(),
                })?;
            let decreases =
                CurrentCountUpdates::select(scope.engine, scope.counts, Some(Ordering::Less))
                    .map_err(Error::from)?;
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
            complete(PublishedTree::new(scope.engine, requested)?, storage).map_err(Into::into)
        })
    }

    /// Recovers the checked phase for another EOA proposal.
    #[cfg(test)]
    pub(in crate::edit) fn into_phase(self) -> CountPublicationPhase<'a> {
        self.phase
    }

    /// Returns the predecessor topology root, or `None` for an empty count batch.
    #[cfg(test)]
    pub(in crate::edit) fn original_root(&self) -> Option<ObjectId> {
        self.phase.counts.root()
    }

    /// Returns the prospective root associated with the pending retirements.
    #[cfg(test)]
    pub(in crate::edit) fn root(&self) -> PhysicalAddress {
        self.phase.retirement.root
    }

    /// Returns the checked physical EOA.
    #[cfg(test)]
    pub(in crate::edit) fn end(&self) -> PhysicalAddress {
        self.end
    }

    /// Borrows original count fields and positive remainders in allocation-address order.
    #[cfg(test)]
    pub(in crate::edit) fn updates(&self) -> &[ProtectedCount] {
        self.phase.counts.updates()
    }
}

/// A physical EOA at or beyond every complete protected count allocation.
struct CountAllocationEnd(PhysicalAddress);

impl CountAllocationEnd {
    /// Checks the maximal end of the disjoint, address-ordered protected allocations.
    ///
    /// # Errors
    ///
    /// Returns [`CountPublicationError::Truncation`] if `end` precedes that allocation end.
    fn new(counts: &ProtectedCounts, end: PhysicalAddress) -> Result<Self, CountPublicationError> {
        if let Some(update) = counts.updates().last()
            && end < update.allocation().range().end()
        {
            return Err(CountPublicationError::Truncation {
                object: update.object(),
                allocation: update.allocation().range(),
                end,
            });
        }
        Ok(Self(end))
    }

    fn address(self) -> PhysicalAddress {
        self.0
    }
}

/// A rejected EOA with its unchanged publication phase.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the state retains original observations for failure and qualification diagnostics"
    )
)]
pub(in crate::edit) struct CountEndFailure<'a> {
    /// The original phase, available for another EOA proposal.
    pub(in crate::edit) phase: CountPublicationPhase<'a>,
    /// The reason the proposed EOA was rejected.
    pub(in crate::edit) error: CountPublicationError,
}

/// An inapplicable source, intersecting retirement, or truncated count allocation.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the state retains original observations for failure and qualification diagnostics"
    )
)]
pub(in crate::edit) enum CountPublicationError {
    /// An unsuccessful source, publication, or allocator-claim check.
    Protection(CountProtectionError),
    /// A pending retirement intersecting a complete original count allocation.
    Retirement {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The complete original count allocation.
        allocation: FileRange,
        /// The complete retirement range.
        retired: FileRange,
        /// The retirement allocation's original free-space class.
        class: FreeClass,
    },
    /// A proposed EOA below the end of a complete original count allocation.
    Truncation {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The complete original count allocation.
        allocation: FileRange,
        /// The proposed physical EOA.
        end: PhysicalAddress,
    },
}

impl From<CountPublicationError> for Error {
    fn from(error: CountPublicationError) -> Self {
        match error {
            CountPublicationError::Protection(CountProtectionError::Source(error)) => error.into(),
            CountPublicationError::Protection(CountProtectionError::Publication(error)) => error,
            _ => Self::EditUnsupported("publication conflicts with an original count allocation"),
        }
    }
}

/// Count and checksum writes paired with their original protected fields.
///
/// Borrows the engine exclusively and the original [`ProtectedCounts`] for the same lifetime.
/// Retains encoded updates without retaining acquired metadata bytes. Preparation and dropping
/// this value leave the file and protected counts unchanged.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the owned state retains the checked evidence until its consuming boundary"
    )
)]
pub(in crate::edit) struct CurrentCountUpdates<'source> {
    engine: &'source mut WriteEngine,
    counts: &'source ProtectedCounts,
    updates: Vec<ObjectReferenceCountUpdate>,
}

impl<'source> CurrentCountUpdates<'source> {
    /// Prepares count and checksum writes from the current complete headers.
    ///
    /// Checks the recorded allocation session before allocating or reading a nonempty batch.
    /// Checks each field's physical range, containing chunk, header flags, and complete allocation
    /// against the original observation. Checks the original scalar before preparing the positive
    /// remainder and any checksum with [`ObjectReferenceCountField::update`]. Unrelated current
    /// chunk bytes are included in the checksum calculation.
    ///
    /// Preserves the original source association when the engine has a different root. Source
    /// freshness, current allocation ownership, retirement exclusions, and permission to publish
    /// require separate evidence. The caller must keep the source contents stable during
    /// preparation and until applying the writes.
    ///
    /// # Errors
    ///
    /// Returns [`CountPreparationError::Session`] if the engine's allocation session differs from
    /// the recorded session. Returns [`CountPreparationError::Read`] if acquiring a protected
    /// allocation fails, or [`CountPreparationError::Range`] if the source returns metadata for a
    /// different range. Returns [`CountPreparationError::Parse`] for header parsing or observation
    /// errors, [`CountPreparationError::Field`] for changed count classification or field geometry,
    /// [`CountPreparationError::Allocation`] for changed or invalid complete allocation geometry,
    /// or [`CountPreparationError::Update`] for a changed scalar or count-encoding error.
    ///
    /// # Performance
    ///
    /// Retains compact update descriptors and releases each acquired metadata range before reading
    /// the next object. Reads complete headers without topology discovery. An empty count batch
    /// requires no allocation or source reads.
    #[cfg(test)]
    fn new(
        engine: &'source mut WriteEngine,
        counts: &'source ProtectedCounts,
    ) -> Result<Self, CountPreparationError> {
        let updates = Self::select(engine, counts, None)?;
        Ok(Self {
            engine,
            counts,
            updates,
        })
    }

    /// Prepares writes from current headers selected by the planned count's ordering.
    ///
    /// Compares each planned remainder with its original scalar. [`None`] selects every update.
    /// Reads selected headers and returns encoded writes in ascending allocation address order.
    /// Checks the recorded allocation session before preparing selected fields through
    /// [`ProtectedCount::prepare_source`].
    ///
    /// # Errors
    ///
    /// Returns [`CountPreparationError::Session`] if the allocation session differs, or the errors
    /// of [`ProtectedCount::prepare_source`] for the selected updates.
    fn select(
        engine: &WriteEngine,
        counts: &ProtectedCounts,
        order: Option<Ordering>,
    ) -> Result<Vec<ObjectReferenceCountUpdate>, CountPreparationError> {
        if counts
            .source
            .as_ref()
            .is_some_and(|source| !source.session.same(&engine.context.allocations.session()))
        {
            return Err(CountPreparationError::Session);
        }
        let mut updates = Vec::with_capacity(counts.updates().len());
        for original in counts.updates().iter().filter(|update| {
            order.is_none_or(|order| update.count().get().cmp(&update.field().value()) == order)
        }) {
            updates.push(original.prepare_source(
                engine.image(),
                engine.widths(),
                engine.addresses().effective_base(),
            )?);
        }
        Ok(updates)
    }

    /// Borrows each original field with its prepared writes, in allocation-address order.
    #[cfg(test)]
    pub(in crate::edit) fn updates(
        &self,
    ) -> impl ExactSizeIterator<Item = (&ProtectedCount, &ObjectReferenceCountUpdate)> {
        let Self {
            engine: _,
            counts,
            updates,
        } = self;
        counts.updates().iter().zip(updates)
    }
}

impl ProtectedCount {
    /// Checks current framing and prepares the original field's positive remainder.
    ///
    /// Reads the complete recorded allocation, checks the current field and allocation against
    /// their original observations, and prepares the count and checksum update.
    ///
    /// # Errors
    ///
    /// Returns [`CountPreparationError::Read`] for an extent or read failure,
    /// [`CountPreparationError::Range`] for an unexpected metadata range, or
    /// [`CountPreparationError::Parse`] for header parsing or observation errors. Returns
    /// [`CountPreparationError::Field`] for changed field classification or geometry,
    /// [`CountPreparationError::Allocation`] for changed or invalid complete allocation geometry,
    /// or [`CountPreparationError::Update`] for a changed scalar or encoding failure.
    fn prepare_source(
        &self,
        source: &(impl Source + ?Sized),
        widths: FormatWidths,
        base: BaseAddress,
    ) -> Result<ObjectReferenceCountUpdate, CountPreparationError> {
        let object = self.object();
        let expected = self.field();
        let current =
            CurrentMetadata::new(source, self.allocation().range()).map_err(
                |error| match error {
                    CurrentMetadataError::Range { expected, provided } => {
                        CountPreparationError::Range {
                            object,
                            expected,
                            provided,
                        }
                    }
                    CurrentMetadataError::Read(error) => {
                        CountPreparationError::Read { object, error }
                    }
                },
            )?;
        let (observed, allocation) = CountObservation::read_source(&current, object, widths, base)
            .map_err(|error| CountPreparationError::Parse { object, error })?
            .into_parts();
        match observed {
            ObjectReferenceCount::Version1(field) | ObjectReferenceCount::Version2(field)
                if field.range() == expected.range()
                    && field.chunk() == expected.chunk()
                    && field.flags() == expected.flags() => {}
            actual => {
                return Err(CountPreparationError::Field {
                    object,
                    expected,
                    actual,
                });
            }
        };
        if allocation != Some(self.allocation()) {
            return Err(CountPreparationError::Allocation {
                object,
                expected: self.allocation().range(),
                actual: allocation.map(|allocation| allocation.range()),
            });
        }
        expected
            .update(current.region(), self.count().get())
            .map_err(|error| CountPreparationError::Update { object, error })
    }
}

/// A source, framing, or encoding failure while preparing a protected count update.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the state retains original observations for failure and qualification diagnostics"
    )
)]
pub(in crate::edit) enum CountPreparationError {
    /// Missing, changed, or invalid complete allocation geometry after parsing the current header.
    Allocation {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The original complete protected allocation.
        expected: FileRange,
        /// The current complete allocation, or `None` if its geometry is invalid.
        actual: Option<FileRange>,
    },
    /// A changed count classification, physical field range, containing chunk, or header flags.
    Field {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The original count field.
        expected: ObjectReferenceCountField,
        /// The current count classification.
        actual: ObjectReferenceCount,
    },
    /// A header parsing or count observation error.
    Parse {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The header parsing or count observation error.
        error: FormatError,
    },
    /// Acquired metadata associated with a different physical range.
    Range {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The physical range requested from the source.
        expected: FileRange,
        /// The physical range associated with the acquired metadata.
        provided: FileRange,
    },
    /// An extent, length conversion, or source-read error while acquiring the protected allocation.
    Read {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The allocation acquisition error.
        error: FormatError,
    },
    /// An engine from a different allocation session.
    Session,
    /// A changed scalar or an error from [`ObjectReferenceCountField::update`].
    Update {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The count or checksum preparation error.
        error: ObjectReferenceCountUpdateError,
    },
}

impl From<CountPreparationError> for Error {
    fn from(error: CountPreparationError) -> Self {
        match error {
            CountPreparationError::Parse { error, .. }
            | CountPreparationError::Read { error, .. } => error.into(),
            _ => Self::EditUnsupported("an original count field changed during commit"),
        }
    }
}

/// Borrowed count facts and the root and EOA proposed before publication.
struct CountCompletion<'counts> {
    counts: &'counts ProtectedCounts,
    root: PhysicalAddress,
    proposed_end: PhysicalAddress,
}

impl<'counts> CountCompletion<'counts> {
    /// Checks the publication's requested root and the protected allocations in the adopted state.
    ///
    /// For a nonempty count batch, checks the allocation session, publication availability, and
    /// reusable, retained, and aggregate-spare ranges. Checks the confirmed EOA for every batch.
    /// Source freshness requires separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`CountCompletionError::Root`] if the publisher was called with another root.
    /// Returns [`CountCompletionError::Protection`] for an inapplicable session, unavailable
    /// publication, conflicting allocator claims, address-conversion errors, or an EOA below a
    /// protected allocation's end.
    ///
    /// # Performance
    ///
    /// On success, borrows the protected facts and allocator lists without allocating.
    /// Uses retained metadata without source reads or topology discovery.
    fn into_write_scope<'engine>(
        self,
        published: PublishedTree<'engine>,
    ) -> Result<CountWriteScope<'engine, 'counts>, CountCompletionError> {
        let Self {
            counts,
            root: expected,
            proposed_end,
        } = self;
        let actual = published.requested_root();
        if actual != expected {
            return Err(CountCompletionError::Root { expected, actual });
        }
        let root = published.root();
        let end = published.end();
        let engine = published.into_engine();
        if let Some(source) = &counts.source {
            if !source.session.same(&engine.context.allocations.session()) {
                return Err(CountCompletionError::Protection(
                    CountPublicationError::Protection(CountProtectionError::Session),
                ));
            }
            engine
                .context
                .allocations
                .before_reference()
                .map_err(CountProtectionError::Publication)
                .map_err(CountPublicationError::Protection)
                .map_err(CountCompletionError::Protection)?;
            counts
                .exclude_space(engine)
                .map_err(CountPublicationError::Protection)
                .map_err(CountCompletionError::Protection)?;
        }
        let end = CountAllocationEnd::new(counts, end)
            .map_err(CountCompletionError::Protection)?
            .address();
        Ok(CountWriteScope {
            engine,
            counts,
            root,
            end,
            proposed_end,
        })
    }
}

/// An exclusive engine loan with protected count facts checked after tree publication.
///
/// The publisher consumed the original retirement batch before creating this scope.
/// The producer checks the root passed to the publisher and the confirmed EOA. For a nonempty
/// count batch, it also checks the allocation session, publication availability, and allocator
/// claims. Source freshness requires separate evidence.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the owned state retains the checked evidence until its consuming boundary"
    )
)]
pub(in crate::edit) struct CountWriteScope<'engine, 'counts> {
    engine: &'engine mut WriteEngine,
    counts: &'counts ProtectedCounts,
    /// The final adopted physical root, including any root reclamation.
    root: PhysicalAddress,
    /// The confirmed physical EOA after publication and requested truncation.
    end: PhysicalAddress,
    /// The physical EOA checked before publication.
    proposed_end: PhysicalAddress,
}

/// An incompatible publication root or failed post-publication allocation check.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the state retains original observations for failure and qualification diagnostics"
    )
)]
pub(in crate::edit) enum CountCompletionError {
    /// An inapplicable session, unavailable publication, conflicting claim, or truncated allocation.
    Protection(CountPublicationError),
    /// A requested root different from the original retirement batch's root.
    Root {
        /// The root associated with the original retirement batch.
        expected: PhysicalAddress,
        /// The root requested in the completed publication call.
        actual: PhysicalAddress,
    },
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
            return Err(CurrentMetadataError::Range {
                expected: range,
                provided: bytes.range(),
            });
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

    /// Returns the capacity of the acquired owner's backing allocation, or zero for borrowed bytes.
    ///
    /// Includes capacity outside the acquired range and excludes ownership bookkeeping and
    /// allocations retained separately by the source.
    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        self.bytes.retained_bytes()
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
    Range {
        /// The physical range passed by the caller.
        expected: FileRange,
        /// The physical range of the returned metadata.
        provided: FileRange,
    },
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
