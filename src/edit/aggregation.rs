//! Classic aggregate reservations for staged edits and immediate appends.

use core::num::NonZeroU64;

use hdf5_pure_space::__private::AggregationError;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ClassicAggregators;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::SessionSpace;

use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::edit::commit::PublicationState;
use crate::error::Error;
use crate::error::FormatError;
use crate::image::FileImage;

use super::FreeClass;
use super::Placement;
use super::WriteEngine;
use super::free_space;

/// Holds the aggregate windows and their stored end of allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AggregateSpace {
    policy: ClassicAggregators,
    eoa: u64,
    /// Whether abandoned aggregate spare requires a persistent manager rewrite.
    manager_dirty: bool,
    /// Whether the cached root and EOA are safe to republish at teardown.
    ///
    /// Teardown preserves the root and EOA after an uncertain publication.
    pub(super) publication_confirmed: bool,
}

impl AggregateSpace {
    /// Writes a reservation at its physical position and pads to its physical EOA.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if a local byte count exceeds `usize`. Returns [`Error::Io`]
    /// if writing the reservation or appending padding fails.
    pub(super) fn place(
        image: &mut dyn FileImage,
        address: u64,
        bytes: &[u8],
        end: u64,
    ) -> Result<(), Error> {
        Self::extend(image, address)?;
        let existing = (image.len() - address).min(bytes.len() as u64).to_usize()?;
        if existing > 0 {
            image.write_at(address, &bytes[..existing])?;
        }
        if existing < bytes.len() {
            image.append(&bytes[existing..])?;
        }
        Self::extend(image, end)
    }

    pub(super) fn new(eoa: u64) -> Self {
        Self {
            policy: ClassicAggregators::new(),
            eoa,
            manager_dirty: false,
            publication_confirmed: true,
        }
    }

    /// Reserves an allocation for `class` and resolves its physical positions.
    ///
    /// The caller writes the reservation through [`place`](Self::place). On error, the
    /// aggregate and reusable-space states are unchanged. In a flat session, tracks abandoned
    /// aggregate spare subject to the free-section threshold.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if aggregate planning or physical address resolution fails.
    pub(super) fn reserve(
        &mut self,
        addresses: AddressContext,
        space: &mut SessionSpace,
        size: NonZeroU64,
        class: AllocationClass,
    ) -> Result<Placement, Error> {
        let len = size.get();
        let mut next = *self;
        let allocation =
            next.policy
                .allocate(next.eoa, size, class)
                .map_err(|error| match error {
                    AggregationError::OffsetOverflow { offset, length } => {
                        FormatError::OffsetOverflow { offset, length }
                    }
                    AggregationError::EndOfAllocationTooSmall { .. } => {
                        FormatError::Internal("aggregate reservation exceeds the editor EOA".into())
                    }
                })?;
        next.eoa = allocation.end_of_allocation();
        let physical_end = addresses.resolve(StoredAddress::new(next.eoa))?.get();
        let placement = Placement::Aggregated {
            addr: addresses
                .resolve(StoredAddress::new(allocation.extent().start()))?
                .get(),
            len,
            end: physical_end,
        };
        // H5MF__aggr_alloc in H5MFaggr.c, HDF5 2.2.0, returns abandoned spare through
        // H5MF_xfree, whose threshold applies to newly tracked free sections.
        if let Some(released) = allocation.released()
            && let Some((post, _)) = space.flat_post_free(next.eoa, [released])
        {
            space.adopt_flat_post_free(post);
            next.manager_dirty = true;
        }
        *self = next;
        Ok(placement)
    }

    /// Tracks freed extents and releases aggregate spare when a free run reaches EOA.
    ///
    /// Clears both aggregate windows before truncation and returns their interior spare to
    /// `space`, subject to its free-section threshold. Returns `None` for a strategy without a
    /// flat free list or when no free run reaches an aggregate tail or EOA.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if aggregate state exceeds EOA or the flat reusable-space
    /// strategy changes during release.
    pub(super) fn release_freed(
        &mut self,
        space: &mut SessionSpace,
        extents: impl IntoIterator<Item = Extent>,
    ) -> Result<Option<u64>, FormatError> {
        let Some((post, eoa)) = space.flat_post_free(self.eoa, extents) else {
            return Ok(None);
        };
        let mut policy = self.policy;
        let tail = policy.release_tail(self.eoa).map_err(|error| {
            FormatError::Internal(format!("cannot release aggregate tails: {error}"))
        })?;
        if eoa == self.eoa
            && !post
                .extents()
                .last()
                .is_some_and(|extent| extent.end() == tail)
        {
            space.adopt_flat_post_free(post);
            return Ok(None);
        }
        // Retained windows prevent truncation below their reserved ends. Clearing both windows
        // returns their interior spare before the free tail is removed.
        let released = policy
            .reset(tail)
            .map_err(|error| FormatError::Internal(format!("cannot reset aggregates: {error}")))?;
        space.adopt_flat_post_free(post);
        let (mut post, eoa) = space
            .flat_post_free(
                eoa.min(released.end_of_allocation()),
                [AllocationClass::Metadata, AllocationClass::Raw]
                    .into_iter()
                    .filter_map(|class| released.released(class)),
            )
            .ok_or_else(|| FormatError::Internal("flat free-space strategy changed".into()))?;
        let eoa = post.take_trailing(eoa).unwrap_or(eoa);
        space.adopt_flat_post_free(post);
        self.policy = policy;
        self.eoa = eoa;
        Ok(Some(eoa))
    }

    /// Returns the total unused bytes in the metadata and raw-data aggregates.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    pub(super) fn unused_bytes(self) -> Result<u64, FormatError> {
        // The aggregate policy stores at most one metadata span and one raw-data span.
        self.unused().try_fold(0u64, |total, extent| {
            total
                .checked_add(extent.len())
                .ok_or(FormatError::OffsetOverflow {
                    offset: total,
                    length: extent.len(),
                })
        })
    }

    pub(super) fn unused(self) -> impl Iterator<Item = Extent> {
        [AllocationClass::Metadata, AllocationClass::Raw]
            .into_iter()
            .filter_map(move |class| self.policy.unused(class))
    }

    pub(super) fn eoa(self) -> u64 {
        self.eoa
    }

    /// Adopts the published EOA and records the manager rewrite as complete.
    ///
    /// Retains windows with unused space and forgets consumed windows beyond `eoa`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if unused aggregate space extends beyond `eoa`.
    pub(super) fn reconcile(mut self, eoa: u64) -> Result<Self, Error> {
        self.policy.reconcile_eoa(eoa).map_err(|error| {
            FormatError::Internal(format!("published EOA excludes aggregate spare: {error}"))
        })?;
        self.eoa = eoa;
        self.manager_dirty = false;
        Ok(self)
    }

    /// Clears both aggregate windows and marks the persistent managers for rewriting.
    ///
    /// Preserves EOA and the publication confirmation state. The caller returns the unused
    /// extents through ordinary threshold admission.
    pub(super) fn without_spare(self) -> Self {
        Self {
            policy: ClassicAggregators::new(),
            manager_dirty: true,
            ..self
        }
    }

    fn extend(image: &mut dyn FileImage, end: u64) -> Result<(), Error> {
        let padding = [0; 4096];
        while image.len() < end {
            let length = (end - image.len()).min(padding.len() as u64).to_usize()?;
            image.append(&padding[..length])?;
        }
        Ok(())
    }
}

impl WriteEngine {
    /// Surrenders aggregate spare when it permits releasing an additional dead suffix.
    ///
    /// The editor retains both windows unless surrender extends the trailing free run by more
    /// than the total spare length. Surrendered spare joins `to_free` for the persistent manager
    /// rewrite, subject to the free-section threshold.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the image length cannot be converted to a stored address.
    pub(super) fn release_aggregates_for_tail(
        &mut self,
        to_free: &mut Vec<(StoredAddress, u64, FreeClass)>,
    ) -> Result<(), Error> {
        let Some(aggregation) = self.aggregation else {
            return Ok(());
        };
        let spare = aggregation.unused().collect::<Vec<_>>();
        if spare.is_empty() || to_free.is_empty() {
            return Ok(());
        }
        let Some(persist) = &self.persist else {
            return Ok(());
        };
        let eoa = self.end_of_allocation()?.get();
        let (mut post, eoa) = self.flat_post_free(eoa, to_free, &persist.old_blocks);
        let retained_start = free_space::trailing_run_start([&post], eoa);
        let released_eoa = post.release_all(persist.threshold, eoa, spare.iter().copied());
        let released_start = free_space::trailing_run_start([&post], released_eoa);
        let spare_len = spare.iter().map(|extent| extent.len()).sum::<u64>();
        if retained_start.saturating_sub(released_start) <= spare_len {
            return Ok(());
        }
        self.aggregation = Some(AggregateSpace {
            policy: ClassicAggregators::new(),
            manager_dirty: true,
            ..aggregation
        });
        to_free.extend(spare.into_iter().map(|extent| {
            (
                StoredAddress::new(extent.start()),
                extent.len(),
                FreeClass::Dead,
            )
        }));
        Ok(())
    }

    /// Returns unused aggregate space and immediate-append reserves through persistent managers.
    ///
    /// Publishes replacement managers with the current root before truncating. A failure before
    /// publication restores the reservation state. An uncertain publication prevents further
    /// aggregate cleanup.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if aggregate geometry, metadata serialization, or address
    /// resolution fails. Returns [`Error::Io`] if writing, synchronizing, or truncating fails.
    pub(super) fn finalize_persistent_aggregates(&mut self) -> Result<(), Error> {
        self.synchronize_aggregates()?;
        let Some(aggregation) = self.aggregation else {
            return Ok(());
        };
        if !aggregation.publication_confirmed {
            return Ok(());
        }
        let spare = aggregation.unused().collect::<Vec<_>>();
        if spare.is_empty()
            && !aggregation.manager_dirty
            && !self.fsm_dirty
            && self.image.len() == self.fsm_len
            && self.reserved.is_empty()
        {
            return Ok(());
        }
        let root = self
            .context
            .addresses()
            .resolve(self.context.superblock().root_group_address)?;
        let snapshot = self.snapshot_free();
        self.aggregation = Some(AggregateSpace::new(aggregation.eoa));
        self.publication = PublicationState::Ready;
        let result = self.commit_persisting_releasing_reserve(
            root,
            spare
                .into_iter()
                .map(|extent| {
                    (
                        StoredAddress::new(extent.start()),
                        extent.len(),
                        FreeClass::Dead,
                    )
                })
                .collect(),
        );
        if result.is_err() && !self.publication.attempted() {
            self.restore_free(snapshot);
        } else if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = result.is_ok();
        }
        result
    }

    /// Advances the stored EOA to include writes outside the aggregate windows.
    ///
    /// Retains aggregate reservations beyond the current image length after an interrupted write.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the image length cannot be converted to a stored address.
    pub(super) fn synchronize_aggregates(&mut self) -> Result<(), Error> {
        let eoa = self.end_of_allocation()?;
        if let Some(aggregation) = &mut self.aggregation {
            aggregation.eoa = aggregation.eoa.max(eoa.get());
        }
        Ok(())
    }

    /// Reserves bytes from the aggregate for `class`, if aggregation is enabled.
    ///
    /// Returns `None` for an empty request or a session without classic aggregation. The caller
    /// writes the reserved bytes through [`AggregateSpace::place`] before making another
    /// reservation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if aggregate planning or physical address resolution fails.
    pub(super) fn reserve_aggregated(
        &mut self,
        len: u64,
        class: AllocationClass,
    ) -> Result<Option<Placement>, Error> {
        let (Some(aggregation), Some(size)) = (self.aggregation.as_mut(), NonZeroU64::new(len))
        else {
            return Ok(None);
        };
        aggregation
            .reserve(self.context.addresses(), &mut self.space, size, class)
            .map(Some)
    }

    /// Releases unused aggregate tails at EOA while retaining the published root.
    ///
    /// In a flat session, tracks interior aggregate spare subject to the free-section threshold.
    /// The finalizer preserves the root and file length after an uncertain publication.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if aggregate planning or address resolution fails. Returns
    /// [`Error::Io`] if publishing the shorter EOA or truncating the image fails.
    pub(super) fn finalize_aggregates(&mut self) -> Result<(), Error> {
        self.synchronize_aggregates()?;
        let Some(mut aggregation) = self.aggregation else {
            return Ok(());
        };
        if !aggregation.publication_confirmed {
            return Ok(());
        }
        let released = aggregation
            .policy
            .reset(aggregation.eoa)
            .map_err(|error| match error {
                AggregationError::OffsetOverflow { offset, length } => {
                    FormatError::OffsetOverflow { offset, length }
                }
                AggregationError::EndOfAllocationTooSmall { .. } => {
                    FormatError::Internal("aggregate reservation exceeds the editor EOA".into())
                }
            })?;
        aggregation.eoa = released.end_of_allocation();
        let post = self.space.flat_post_free(
            aggregation.eoa,
            [AllocationClass::Metadata, AllocationClass::Raw]
                .into_iter()
                .filter_map(|class| released.released(class)),
        );
        let post = post.map(|(mut post, eoa)| {
            aggregation.eoa = post.take_trailing(eoa).unwrap_or(eoa);
            post
        });
        let end = self
            .context
            .addresses()
            .resolve(StoredAddress::new(aggregation.eoa))?;
        if end.get() < self.image.len() {
            let root = self
                .context
                .addresses()
                .resolve(self.context.superblock().root_group_address)?;
            self.publication = PublicationState::Ready;
            let result = self.publish_nonpersisting(root, Some(end));
            if result.is_ok() || self.publication.attempted() {
                aggregation.publication_confirmed = result.is_ok();
                self.aggregation = Some(aggregation);
                if let Some(post) = post {
                    self.space.adopt_flat_post_free(post);
                }
            }
            result?;
        } else {
            self.aggregation = Some(aggregation);
            if let Some(post) = post {
                self.space.adopt_flat_post_free(post);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod immediate_tests;

#[cfg(test)]
mod fsm_tests;

#[cfg(test)]
mod persistent_tests;

#[cfg(test)]
mod tests {
    use std::io::Error as IoError;
    use std::io::ErrorKind;

    use rstest::rstest;

    use crate::FileLocking;
    use crate::FileSpaceSettings;
    use crate::FileSpaceStrategy;
    use crate::MaxExtent;
    use crate::MemoryStrategy;
    use crate::PersistedFreeSpacePolicy;
    use crate::UserblockSize;
    use crate::error::Error;
    use crate::error::FormatError;
    use crate::image::FileImage;
    use crate::image::MirrorImage;
    use crate::image::WriteBuffering;
    use crate::object_path::ObjectPathBuf;
    use crate::reader::File;
    use crate::source::MetadataCacheConfig;
    use crate::source::Source;
    use crate::type_builders::DatasetBuilder;
    use crate::writer::FileBuilder;

    use super::AllocationClass;
    use super::WriteEngine;

    #[rstest]
    #[case::segregation(
        &[(AllocationClass::Metadata, 64), (AllocationClass::Raw, 64), (AllocationClass::Metadata, 64)],
        &[0, 2048, 64], 4096
    )]
    #[case::refill(
        &[(AllocationClass::Raw, 1024), (AllocationClass::Raw, 1024), (AllocationClass::Raw, 64)],
        &[0, 1024, 2048], 4096
    )]
    #[case::large_direct(&[(AllocationClass::Raw, 4096)], &[0], 4096)]
    #[case::large_extension(
        &[(AllocationClass::Raw, 64), (AllocationClass::Raw, 4096)], &[0, 64], 6144
    )]
    fn staged_reservations_use_classic_windows(
        #[case] requests: &[(AllocationClass, usize)],
        #[case] offsets: &[u64],
        #[case] reserved: u64,
        #[values(0, 512, 4096)] userblock: u64,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("windows.h5");
        let mut builder = FileBuilder::new();
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
        builder.write(&path).unwrap();
        let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let start = engine.image.len();
        for ((class, length), offset) in requests.iter().zip(offsets) {
            let bytes = vec![19; *length];
            let placement = engine.reserve(*length as u64, *class).unwrap();
            assert_eq!(placement.address(), start + offset);
            let address = engine.place(placement, &bytes).unwrap();
            assert_eq!(
                engine.image().read_exact_at(address, *length).unwrap(),
                bytes
            );
        }
        assert_eq!(engine.image.len(), start + reserved);
        let state = engine.aggregation.unwrap();
        assert_eq!(state.eoa(), start + reserved - userblock);
        assert_eq!(
            engine.reusable_space_stats().unwrap().reusable_free_bytes(),
            state.unused().map(|extent| extent.len()).sum::<u64>()
        );
    }

    #[rstest]
    #[case::mirror(MemoryStrategy::Mirrored)]
    #[case::bounded(MemoryStrategy::Bounded)]
    fn aggregate_tail_release_preserves_committed_data(#[case] backing: MemoryStrategy) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("close.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
        builder.write(&path).unwrap();
        let mut engine = WriteEngine::open_rw_with_strategy(
            &path,
            MetadataCacheConfig::default(),
            FileLocking::Enabled,
            backing,
            PersistedFreeSpacePolicy::Tolerant,
        )
        .unwrap();
        let mut dataset = DatasetBuilder::new("");
        dataset.with_i32_data(&[11; 16]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("/values"), dataset)
            .unwrap();
        engine.commit().unwrap();
        let before = engine.image.len();
        let tail = engine
            .aggregation
            .unwrap()
            .unused()
            .find(|extent| extent.end() == before)
            .unwrap();
        engine.finalize_persist().unwrap();
        assert_eq!(engine.image.len(), tail.start());
        assert_eq!(engine.context.superblock().eof_address, tail.start());
        assert!(engine.aggregation.unwrap().unused().next().is_none());
        let order = engine.image.issued_write_order();
        assert_eq!(order.last().unwrap().0, engine.context.signature().get());
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("values").unwrap().read_i32().unwrap(),
            [11; 16]
        );
        assert_eq!(file.persisted_free_sections().unwrap(), None);
    }

    #[rstest]
    fn overflowing_aggregate_reservation_keeps_state(
        #[values(AllocationClass::Metadata, AllocationClass::Raw)] class: AllocationClass,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("overflow.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
        builder.write(&path).unwrap();
        let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let state = engine.aggregation;
        let end = engine.image.len();
        let error = engine.reserve(u64::MAX, class).unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::OffsetOverflow { offset, length }) if offset == end && length == u64::MAX),
            "{error:?}"
        );
        assert_eq!(engine.aggregation, state);
        assert_eq!(engine.image.len(), end);
    }

    #[rstest]
    fn failed_staged_write_restores_aggregate_state(
        #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("failed.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(strategy));
        builder.create_dataset("keep").with_i32_data(&[7]);
        builder.write(&path).unwrap();
        let end = std::fs::metadata(&path).unwrap().len();
        let mut engine = WriteEngine::open_torn_writes(&path, end..u64::MAX).unwrap();
        let original = engine.aggregation;
        let mut dataset = DatasetBuilder::new("");
        dataset.with_i32_data(&[11; 16]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("/failed"), dataset)
            .unwrap();
        let error = engine.commit().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected device write failure, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(engine.aggregation, original);
        assert_eq!(engine.context.superblock().eof_address, end);
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(file.root().datasets().unwrap(), ["keep"]);
        assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [7]);
    }

    #[rstest]
    fn failed_staged_write_restores_existing_aggregate_state(
        #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("rollback_reserve.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(strategy));
        builder.write(&path).unwrap();
        let start = std::fs::metadata(&path).unwrap().len();
        let mut engine = WriteEngine::open_torn_writes(&path, start + 64..start + 128).unwrap();
        let mut first = DatasetBuilder::new("");
        first.with_i32_data(&[43; 16]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("/first"), first)
            .unwrap();
        engine.commit().unwrap();
        let state = engine.aggregation;
        let mut second = DatasetBuilder::new("");
        second.with_i32_data(&[47; 16]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("/second"), second)
            .unwrap();
        let error = engine.commit().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected reserved-storage write failure, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert!(!engine.publication.attempted());
        assert_eq!(engine.aggregation, state);
        engine.finalize_persist().unwrap();
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(file.root().datasets().unwrap(), ["first"]);
        assert_eq!(file.dataset("first").unwrap().read_i32().unwrap(), [43; 16]);
    }

    #[rstest]
    fn failed_publish_keeps_consumed_aggregate_space(
        #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("publish.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(strategy));
        builder.write(&path).unwrap();
        let mut engine = WriteEngine::open_torn_writes(&path, 0..48).unwrap();
        let original = engine.aggregation;
        let mut dataset = DatasetBuilder::new("");
        dataset.with_i32_data(&[13; 16]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("/values"), dataset)
            .unwrap();
        let error = engine.commit().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected device write failure, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert!(engine.publication.attempted());
        assert_ne!(engine.aggregation, original);
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("values").unwrap().read_i32().unwrap(),
            [13; 16]
        );
    }

    #[rstest]
    #[case::write_before(PublicationFailure::WriteBefore, false)]
    #[case::write_after(PublicationFailure::WriteAfter, true)]
    #[case::barrier(PublicationFailure::Barrier, true)]
    fn aggregate_teardown_preserves_a_publication_after_transient_failure(
        #[case] failure: PublicationFailure,
        #[case] applied: bool,
        #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transient.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(strategy));
        builder.create_dataset("keep").with_i32_data(&[41]);
        builder.write(&path).unwrap();
        let mut engine =
            WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _length| {
                Ok(Box::new(TransientPublicationImage {
                    inner: WriteEngine::read_mirror(handle)?,
                    failure,
                    publication_written: false,
                    failed: false,
                }))
            })
            .unwrap();
        engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
        let mut dataset = DatasetBuilder::new("");
        dataset.with_i32_data(&[37; 16]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("/values"), dataset)
            .unwrap();
        let error = engine.commit().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected transient publication failure, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert!(engine.publication.attempted());
        let published = File::from_bytes(engine.image.as_slice().unwrap().to_vec()).unwrap();
        let expected = if applied {
            vec!["keep", "values"]
        } else {
            vec!["keep"]
        };
        assert_eq!(published.root().datasets().unwrap(), expected);
        if applied {
            assert_eq!(
                published.dataset("values").unwrap().read_i32().unwrap(),
                [37; 16]
            );
        }
        assert!(!engine.aggregation.unwrap().publication_confirmed);
        let physical_size = engine.image.len();
        engine.commit().unwrap();
        assert!(!engine.aggregation.unwrap().publication_confirmed);
        engine.finalize_persist().unwrap();
        let after_cleanup = engine.image.len();
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(file.root().datasets().unwrap(), expected);
        assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [41]);
        if applied {
            assert_eq!(
                file.dataset("values").unwrap().read_i32().unwrap(),
                [37; 16]
            );
        }
        assert_eq!(after_cleanup, physical_size);
    }

    #[test]
    fn immediate_growth_keeps_previous_reserves_disjoint() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("interleaved.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
        builder
            .create_dataset("append")
            .with_i32_data(&[1, 2, 3, 4])
            .with_shape(&[4])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        builder.write(&path).unwrap();
        let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let mut staged = DatasetBuilder::new("");
        staged.with_i32_data(&[41; 16]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("/first"), staged)
            .unwrap();
        engine.commit().unwrap();
        let before = engine.image.len();
        let policy = engine.aggregation.unwrap();
        let raw = policy.policy.unused(AllocationClass::Raw).unwrap();
        engine
            .append_inplace_i32_phased("append", &[5, 6, 7, 8], 4)
            .unwrap();
        assert_eq!(engine.image.len(), before);
        let after = engine.aggregation.unwrap();
        assert_eq!(
            after.policy.unused(AllocationClass::Metadata),
            policy.policy.unused(AllocationClass::Metadata)
        );
        assert_eq!(
            after.policy.unused(AllocationClass::Raw).unwrap().start(),
            raw.start() + 16
        );
        engine
            .create_group(&ObjectPathBuf::parse("/second"))
            .unwrap();
        engine.commit().unwrap();
        let ownership = engine.space_ownership().unwrap();
        assert!(ownership.analysis_complete);
        assert_eq!(ownership.physical_size, ownership.logical_size);
        engine.finalize_persist().unwrap();
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("append").unwrap().read_i32().unwrap(),
            (1..=8).collect::<Vec<_>>()
        );
        assert_eq!(file.root().groups().unwrap(), ["second"]);
    }

    struct TransientPublicationImage {
        inner: MirrorImage,
        failure: PublicationFailure,
        publication_written: bool,
        failed: bool,
    }

    impl FileImage for TransientPublicationImage {
        fn read_disk_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
            self.inner.read_disk_at(offset, bytes)
        }

        fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
            self.inner.append(bytes)
        }

        fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
            if offset == 0
                && matches!(self.failure, PublicationFailure::WriteBefore)
                && !self.failed
            {
                self.failed = true;
                return Err(Error::Io(IoError::other(
                    "transient publication write failure",
                )));
            }
            self.inner.write_at(offset, bytes)?;
            if offset == 0 {
                self.publication_written = true;
                if matches!(self.failure, PublicationFailure::WriteAfter) && !self.failed {
                    self.failed = true;
                    return Err(Error::Io(IoError::other(
                        "transient publication write failure",
                    )));
                }
            }
            Ok(())
        }

        fn truncate(&mut self, length: u64) -> Result<(), Error> {
            self.inner.truncate(length)
        }

        fn sync_data(&mut self) -> Result<(), Error> {
            self.inner.sync_data()
        }

        fn sync_all(&mut self) -> Result<(), Error> {
            self.inner.sync_all()?;
            if self.publication_written
                && matches!(self.failure, PublicationFailure::Barrier)
                && !self.failed
            {
                self.failed = true;
                return Err(Error::Io(IoError::other(
                    "transient publication barrier failure",
                )));
            }
            Ok(())
        }

        fn ordering_barrier(&mut self) -> Result<(), Error> {
            self.inner.ordering_barrier()
        }

        fn issued_write_order(&self) -> Vec<(u64, u64)> {
            self.inner.issued_write_order()
        }

        fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
            self.inner.set_write_buffering(mode)
        }

        fn as_slice(&self) -> Option<&[u8]> {
            self.inner.as_slice()
        }
    }

    impl Source for TransientPublicationImage {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
            self.inner.read_at(offset, buffer)
        }
    }
    #[derive(Clone, Copy)]
    enum PublicationFailure {
        Barrier,
        WriteAfter,
        WriteBefore,
    }
}
