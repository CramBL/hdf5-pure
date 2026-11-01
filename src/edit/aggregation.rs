//! Classic aggregate reservations for staged edits and immediate appends.

use core::num::NonZeroU64;

use hdf5_pure_space::__private::AggregationError;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ClassicAggregators;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::ReturnedExtents;
use hdf5_pure_space::__private::SessionSpace;

use crate::address::AddressContext;
use crate::address::StoredAddress;
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
    eoa: StoredAddress,
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

    pub(super) fn new(eoa: StoredAddress) -> Self {
        Self {
            policy: ClassicAggregators::new(),
            eoa,
            manager_dirty: false,
            publication_confirmed: true,
        }
    }

    /// Reserves an allocation for `class` and resolves its physical positions.
    ///
    /// Calls `release` with the stored EOA and complete abandoned spare, marking manager state dirty
    /// if the callback returns `true`. The caller applies threshold admission in that callback and
    /// writes the reservation through [`place`](Self::place). On error, aggregate state is unchanged
    /// and the callback has not been called.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if aggregate planning or physical address resolution fails.
    pub(super) fn reserve_with(
        &mut self,
        addresses: AddressContext,
        size: NonZeroU64,
        class: AllocationClass,
        mut release: impl FnMut(StoredAddress, Extent) -> bool,
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
        let physical_end = addresses.resolve(next.eoa)?.get();
        let placement = Placement::Aggregated {
            addr: addresses.resolve(allocation.extent().start())?.get(),
            len,
            end: physical_end,
        };
        // H5MF__aggr_alloc in H5MFaggr.c, HDF5 2.2.0, returns abandoned spare through
        // H5MF_xfree, whose threshold applies to newly tracked free sections.
        if let Some(released) = allocation.released()
            && release(next.eoa, released)
        {
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
    ) -> Result<Option<StoredAddress>, FormatError> {
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

    pub(super) fn eoa(self) -> StoredAddress {
        self.eoa
    }

    /// Adopts the published EOA and records the manager rewrite as complete.
    ///
    /// Retains windows with unused space and forgets consumed windows beyond `eoa`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if unused aggregate space extends beyond `eoa`.
    pub(super) fn reconcile(mut self, eoa: StoredAddress) -> Result<Self, Error> {
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
    /// Returns [`Error::Format`] if the image length cannot be converted to a stored address or
    /// unused aggregate space overlaps another returned allocation.
    pub(super) fn release_aggregates_for_tail(
        &mut self,
        to_free: &mut ReturnedExtents,
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
        let eoa = self.end_of_allocation()?;
        let (mut post, eoa) = self.flat_post_free(eoa, to_free, &persist.old_blocks);
        let retained_start = free_space::trailing_run_start([&post], eoa);
        let released_eoa = post.release_all(persist.threshold, eoa, spare.iter().copied());
        let released_start = free_space::trailing_run_start([&post], released_eoa);
        let spare_len = spare.iter().map(|extent| extent.len()).sum::<u64>();
        if retained_start
            .checked_distance_from(released_start)
            .unwrap_or(0)
            <= spare_len
        {
            return Ok(());
        }
        self.aggregation = Some(AggregateSpace {
            policy: ClassicAggregators::new(),
            manager_dirty: true,
            ..aggregation
        });
        for extent in spare {
            to_free
                .insert(extent, FreeClass::Dead)
                .map_err(|error| FormatError::Internal(error.to_string()))?;
        }
        Ok(())
    }

    /// Returns unused aggregate space through persistent managers.
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
        let mut spare = aggregation.unused().collect::<Vec<_>>();
        spare.sort_unstable_by_key(|extent| extent.start());
        if spare.is_empty()
            && !aggregation.manager_dirty
            && !self.fsm_dirty
            && self.image.len() == self.fsm_len
        {
            return Ok(());
        }
        let root = self
            .context
            .addresses()
            .resolve(self.context.superblock().root_group_address)?;
        let returned = ReturnedExtents::from_sorted(
            spare
                .into_iter()
                .map(|extent| (extent, FreeClass::Dead))
                .collect(),
        )
        .map_err(|error| FormatError::Internal(error.to_string()))?;
        let snapshot = self.snapshot_free();
        self.aggregation = Some(AggregateSpace::new(aggregation.eoa));
        self.publication = PublicationState::Ready;
        let result = self.commit_persisting_releasing_reserve(root, returned);
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
            aggregation.eoa = aggregation.eoa.max(eoa);
        }
        Ok(())
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
        let end = self.context.addresses().resolve(aggregation.eoa)?;
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
mod tests;
