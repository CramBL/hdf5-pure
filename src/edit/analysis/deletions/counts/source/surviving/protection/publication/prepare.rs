//! Count and checksum writes prepared from current object-header framing.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountField;
use hdf5_pure_format::__private::ObjectReferenceCountUpdate;
use hdf5_pure_format::__private::ObjectReferenceCountUpdateError;

use super::CountPublicationBounds;
use super::CountPublicationPhase;
use super::ProtectedCount;
use super::ProtectedCounts;
use super::current::CurrentMetadata;
use super::current::CurrentMetadataError;
use crate::address::BaseAddress;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::CountObservation;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::source::Source;
use crate::width::FormatWidths;

impl CountPublicationBounds<'_> {
    /// Prepares every protected count update after parsing the current complete headers.
    ///
    /// Uses [`CurrentCountUpdates::new`] and borrows these bounds exclusively. Retains the original
    /// retirement batch and EOA. Leaves the bounds and file bytes unchanged on error.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`CurrentCountUpdates::new`].
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
}

/// Count and checksum writes paired with their original protected fields.
///
/// Borrows the engine exclusively and the original [`ProtectedCounts`] for the same lifetime.
/// Retains encoded updates without retaining acquired metadata bytes. Preparation and dropping
/// this value leave the file and protected counts unchanged.
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
    pub(super) fn new(
        engine: &'source mut WriteEngine,
        counts: &'source ProtectedCounts,
    ) -> Result<Self, CountPreparationError> {
        if counts
            .source
            .as_ref()
            .is_some_and(|source| !source.session.same(&engine.context.allocations.session()))
        {
            return Err(CountPreparationError::Session);
        }
        let mut updates = Vec::with_capacity(counts.updates().len());
        for original in counts.updates() {
            updates.push(original.prepare_source(
                engine.image(),
                engine.widths(),
                engine.addresses().effective_base(),
            )?);
        }
        Ok(Self {
            engine,
            counts,
            updates,
        })
    }

    /// Borrows each original field with its prepared writes, in allocation-address order.
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
    /// # Errors
    ///
    /// Returns the errors described by [`CountPublicationBounds::prepare_counts`].
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

#[cfg(test)]
mod tests;
