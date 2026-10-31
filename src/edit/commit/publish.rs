//! Root publication, persistent free-space metadata, and PAGE publication recovery.
//!
//! Content is written before the publication barrier. The superblock write and its barrier
//! precede truncation and stored-reference updates. Durability follows [`SyncPolicy`].

use std::collections::BTreeMap;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private as free_space;
#[cfg(doc)]
use hdf5_pure_space::__private::SessionSpace;
use hdf5_pure_space::__private::{
    AllocationClass, Extent as FreeExtent, FreeClass, FreeList, ManagerKind, ManagerSections,
    PageType,
};

use super::{PublicationPurpose, PublicationState};
use crate::SuperblockVersion;
use crate::address::StoredAddress;
use crate::edit::SyncPolicy;
use crate::edit::context::SuperblockPublication;
use crate::edit::object_header;
use crate::edit::retirement::{PendingRootFrees, RootRetirement};
use crate::edit::{MAX_LINK_GRAPH_NODES, TailPlacement, UNDEF, WriteEngine};
use crate::edit::{PersistState, paged_persistence};
use crate::error::{Error, FormatError};
use crate::file_space_info::{self, NUM_FILE_FSM_MANAGERS};
use crate::free_space_manager;
use crate::source::Source;
use crate::width::FormatWidths;

/// Stores the root and address changes after replacement objects have been written.
pub(super) struct AppliedTree {
    /// Replacement root and captured chunk allocations awaiting its publication.
    pub(super) retirement: RootRetirement,
    /// Physical spans eligible for release when the root is published.
    pub(super) freed: Vec<(u64, u64, FreeClass)>,
    /// Absolute old and replacement header addresses for stored-reference updates.
    pub(super) relocations: BTreeMap<u64, u64>,
}

impl WriteEngine {
    /// Publishes a written tree through the session's persistent or non-persistent path.
    ///
    /// Stored-reference updates run after root publication. Non-persistent PAGE commits require
    /// a tree proved free of object references and retain superseded storage.
    /// For non-persistent commits without paging, the publisher restores allocation state saved
    /// after replacement placement if root publication is uncertain, keeping predecessor and
    /// replacement storage unavailable for reuse.
    ///
    /// # Errors
    ///
    /// Returns the publication, reference-update, or root-reclamation error. Errors after
    /// publication leave rollback disabled.
    pub(super) fn publish_tree(&mut self, tree: AppliedTree) -> Result<(), Error> {
        let AppliedTree {
            retirement,
            freed,
            relocations,
        } = tree;
        let PendingRootFrees {
            root: new_root,
            spans: to_free,
        } = retirement.into_pending_frees(freed, self.image.len());
        let mut to_free = self.stored_spans(to_free)?;
        if self.persist.is_some() {
            let root_reclamation = self.plan_root_reclamation(new_root, &mut to_free)?;
            self.commit_persisting_releasing_reserve(new_root, to_free)?;
            if root_reclamation.is_some() {
                self.proved_free_of_references = true;
            }
            self.repoint_stored_references(&relocations)?;
            if let Some(reclamation) = root_reclamation {
                self.reclaim_root_metadata(reclamation)?;
            }
            return Ok(());
        }

        if self.fresh_page_boundary.is_some() {
            self.pad_to_page()?;
            return self.publish_nonpersisting(new_root, None);
        }

        // The publisher allocates no file storage after replacement placement. Restoring this
        // snapshot preserves every allocation a possibly published root can reference.
        let snapshot = self.snapshot_free();
        let mut trunc_to = self.release_freed(to_free)?;
        if let Some(aggregation) = self.aggregation {
            let end = self.context.addresses().resolve(aggregation.eoa())?;
            if end.get() < self.image.len() {
                trunc_to = Some(end);
            }
        }
        let publication = self.publish_nonpersisting(new_root, trunc_to);
        if publication.is_err() && self.publication != PublicationState::Published {
            self.restore_free(snapshot);
        }
        publication?;
        self.repoint_stored_references(&relocations)
    }

    /// Publishes a persistent commit and releases eligible unused aggregate spare.
    ///
    /// Reserves eligible sub-threshold manager storage during the first publication, then
    /// recycles it in a separate publication. Other commits use ordinary tail placement.
    /// Attempts to relocate a trailing manager allocation before truncating.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if aggregate planning, address resolution, or metadata
    /// serialization fails. Returns [`Error::EditUnsupported`] if the extension cannot be
    /// rewritten. Returns [`Error::Io`] if writing, synchronizing, or truncating fails.
    pub(in crate::edit) fn commit_persisting_releasing_reserve(
        &mut self,
        new_root: PhysicalAddress,
        mut to_free: Vec<(StoredAddress, u64, FreeClass)>,
    ) -> Result<(), Error> {
        self.release_aggregates_for_tail(&mut to_free)?;
        if let Some(target) = self.recyclable_flat_tail() {
            self.commit_persisting(new_root, Vec::new(), TailPlacement::Preserve)?;
            self.recycle_flat_tail(new_root, to_free, target)?;
        } else {
            self.commit_persisting(new_root, to_free, TailPlacement::Anywhere)?;
        }
        self.shrink_to_the_trailing_free_run(new_root)
    }

    /// Relocates persistent manager blocks into earlier free space before truncating.
    ///
    /// Runs after the root publication makes the preceding commit's freed regions reusable.
    /// The manager tail must end at EOA or, for a flat file, adjoin a tracked free run reaching
    /// EOA. The combined run below and above the tail must meet the retention threshold in
    /// [`FreeList::release_trailing_run`].
    ///
    /// [`TailPlacement::ReuseOnly`] requires the replacement to fit below the old manager tail.
    /// The writer publishes the replacement with the current root before truncating.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if metadata serialization or address resolution fails. Returns
    /// [`Error::Io`] if writing, synchronizing, or truncating fails.
    pub(in crate::edit) fn shrink_to_the_trailing_free_run(
        &mut self,
        new_root: PhysicalAddress,
    ) -> Result<(), Error> {
        let Some(persist) = self.persist.as_ref() else {
            return Ok(());
        };
        let (Some(tail_start), Some(tail_end)) = (
            persist.old_blocks.iter().map(|extent| extent.start()).min(),
            persist.old_blocks.iter().map(|extent| extent.end()).max(),
        ) else {
            return Ok(());
        };
        // A flat tail may have published reusable space above it. That space contains no live
        // allocations and can be released by the same manager rewrite.
        let eoa = self.end_of_allocation()?;
        if tail_end != eoa
            && (self.space.is_paged() || self.space.trailing_run_start(eoa) != tail_end)
        {
            return Ok(());
        }
        // Apply the retention threshold of `FreeList::release_trailing_run` to the combined
        // free run and old tail before attempting a rewrite.
        let run_start = self.space.trailing_run_start(tail_start);
        let required = tail_end
            .checked_distance_from(tail_start)
            .and_then(|length| length.checked_mul(2 * free_space::TRAILING_RESERVE_TAILS));
        if required
            .zip(eoa.checked_distance_from(run_start))
            .is_none_or(|(required, available)| available < required)
        {
            return Ok(());
        }
        self.commit_persisting(new_root, Vec::new(), TailPlacement::ReuseOnly)
    }

    /// Verifies pending PAGE publication and clears its recovery state on success.
    ///
    /// The caller completes its ordering or synchronization barrier first. Later staged edits
    /// remain pending if verification fails.
    ///
    /// # Errors
    ///
    /// Returns the error from [`recover_page_publication`](Self::recover_page_publication).
    pub(in crate::edit) fn complete_page_publication(&mut self) -> Result<(), Error> {
        if self.fresh_page_boundary.is_some() && self.publication.attempted() {
            self.recover_page_publication()?;
            self.publication = PublicationState::Ready;
        }
        Ok(())
    }

    /// Adopts the published PAGE root and EOA after verifying the accepted superblock bytes.
    ///
    /// Other superblock fields must match the session's framing. Before consuming staged edits
    /// in another commit, the caller drains pending writes through a barrier.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the file bytes differ from the accepted image or
    /// superblock framing. Returns the read, serialization, or parsing error if recovery fails.
    pub(in crate::edit) fn recover_page_publication(&mut self) -> Result<(), Error> {
        let len =
            hdf5_pure_format::__private::serialize_superblock(self.context.superblock())?.len();
        let accepted = self
            .image
            .read_exact_at(self.context.signature().get(), len)?;
        let mut published = vec![0; len];
        self.image
            .read_disk_at(self.context.signature().get(), &mut published)?;
        if published != accepted {
            return Err(Error::EditUnsupported(
                "PAGE publication bytes differ from the session image after a failed write; \
                 close and reopen the file before editing again",
            ));
        }
        let proposed = self.context.recover(&published)?;
        self.context.adopt(proposed)?;
        Ok(())
    }

    /// Publishes the root and EOA after the content barrier, then applies any requested truncation.
    ///
    /// A non-persistent PAGE commit retains its File Space Info message and passes no truncation.
    ///
    /// # Errors
    ///
    /// Returns the address-conversion, serialization, write, synchronization, or truncation error.
    pub(in crate::edit) fn publish_nonpersisting(
        &mut self,
        new_root: PhysicalAddress,
        trunc_to: Option<PhysicalAddress>,
    ) -> Result<(), Error> {
        let base = self.context.addresses();
        let new_eof = trunc_to.unwrap_or_else(|| PhysicalAddress::new(self.image.len()));
        self.barrier()?;
        if matches!(
            self.context.superblock().version(),
            SuperblockVersion::Two | SuperblockVersion::Three
        ) {
            // Adopt the proposed superblock after the publication write and barrier succeed.
            let mut new_sb = self.context.superblock().clone();
            new_sb.root_group_address = base.relative(new_root)?;
            new_sb.eof_address = new_eof.get();
            // Preserve this session's status flags, including an active crash mark.
            new_sb.set_consistency_flags(self.held_status_flags)?;
            self.publish_superblock(self.context.propose(new_sb)?.encode()?)?;
        } else {
            let stored_root = base.relative(new_root)?;
            let mut superblock = self.context.superblock().clone();
            superblock.root_group_address = stored_root;
            superblock.eof_address = new_eof.get();
            let proposed = self.context.propose(superblock)?;
            self.publication = PublicationState::Uncertain;
            self.repoint_v0v1_root(stored_root, new_eof)?;
            self.barrier()?;
            self.context.adopt(proposed)?;
            self.publication = PublicationState::Published;
        }

        // Publish the smaller EOF before truncating, preserving a complete file if interrupted.
        if let Some(cut) = trunc_to {
            self.truncate_published(cut)?;
        }
        Ok(())
    }

    /// Records the regions a non-persisting commit vacated under the active reusable-space
    /// strategy, and returns the file position to truncate the file to when the flat free run
    /// reaches the end of allocation.
    ///
    /// Disabled strategies discard the regions. A non-persistent PAGE commit skips this step
    /// and retains every freed extent without making it reusable.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if the image ends below the base address, and
    /// [`FormatError::OffsetOverflow`] if the truncation point plus the base address exceeds
    /// [`u64::MAX`]. Returns [`FormatError::Internal`] if aggregate state exceeds EOA or the
    /// flat reusable-space strategy changes during release.
    pub(in crate::edit) fn release_freed(
        &mut self,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
    ) -> Result<Option<PhysicalAddress>, FormatError> {
        let eoa = self.end_of_allocation()?;
        if let Some(aggregation) = &mut self.aggregation {
            return aggregation
                .release_freed(
                    &mut self.space,
                    to_free
                        .into_iter()
                        .filter_map(|(addr, len, _)| FreeExtent::new(addr, len)),
                )?
                .map(|cut| self.context.addresses().resolve(cut))
                .transpose();
        }
        self.space
            .release_freed(
                eoa,
                to_free
                    .into_iter()
                    .filter_map(|(addr, len, _)| FreeExtent::new(addr, len)),
            )
            .map(|cut| self.context.addresses().resolve(cut))
            .transpose()
    }

    /// Returns `spans` with each file position converted to a stored address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if a span starts below the base address.
    pub(in crate::edit) fn stored_spans(
        &self,
        spans: Vec<(u64, u64, FreeClass)>,
    ) -> Result<Vec<(StoredAddress, u64, FreeClass)>, FormatError> {
        spans
            .into_iter()
            .map(|(at, len, class)| {
                Ok((
                    self.context
                        .addresses()
                        .relative(PhysicalAddress::new(at))?,
                    len,
                    class,
                ))
            })
            .collect()
    }

    /// Repoints stored references to object headers relocated by this commit.
    ///
    /// The caller publishes the replacement root first so the reference walk reads the
    /// committed tree. The writes end with a barrier under [`SyncPolicy`]. A failure between
    /// publication and that barrier can leave stale references in the committed tree.
    ///
    /// The walk is bounded by [`MAX_LINK_GRAPH_NODES`] and skipped after the tree has been
    /// proved free of object references.
    ///
    /// # Errors
    ///
    /// Returns the reference-planning, write, or barrier error.
    fn repoint_stored_references(&mut self, relocations: &BTreeMap<u64, u64>) -> Result<(), Error> {
        if self.proved_free_of_references {
            return Ok(());
        }
        let plan = crate::reference_patch::plan(
            &self.image(),
            self.context.superblock(),
            self.context.widths(),
            self.context.addresses(),
            relocations,
            MAX_LINK_GRAPH_NODES,
        )?;
        // Cache only a complete proof that the tree contains no object references.
        if plan.proved_free_of_references() {
            self.proved_free_of_references = true;
        }
        if plan.is_empty() {
            return Ok(());
        }
        plan.apply(self)?;
        self.barrier()
    }

    /// Publishes `new_root` with replacement persistent free-space metadata.
    ///
    /// Writes the extension and manager blocks as one contiguous allocation according to
    /// `placement`. [`TailPlacement::Preserve`] excludes the current allocation from returned
    /// free space. [`TailPlacement::Recycle`] can reuse a reserved allocation after an earlier
    /// publication made it unreferenced. Returned extents follow ordinary threshold admission.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address resolution, metadata serialization, or aggregate
    /// reconciliation fails. Returns [`Error::EditUnsupported`] if the extension cannot be
    /// rewritten. Returns [`Error::Io`] if writing, synchronizing, or truncating fails. For PAGE,
    /// returns the errors from [`Self::commit_persisting_paged`].
    ///
    /// # Panics
    ///
    /// Panics if persistence is disabled.
    pub(in crate::edit) fn commit_persisting(
        &mut self,
        new_root: PhysicalAddress,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
        placement: TailPlacement,
    ) -> Result<(), Error> {
        self.publish_persisting(new_root, to_free, placement, PublicationPurpose::Commit)
    }

    /// Publishes replacement persistent managers for a commit or an allocation batch.
    ///
    /// The caller supplies the root and pending frees. During allocation publication, the editor
    /// retains the current root and leaves staged edits pending. The allocator excludes provisional
    /// allocations and published metadata from placement. The writer records complete manager
    /// allocations, including section-list padding.
    ///
    /// [`PublicationPurpose::Allocation`] drains explicit page buffers under
    /// [`SyncPolicy::OnClose`]. Staged commits use their general buffering policy.
    ///
    /// # Errors
    ///
    /// Returns metadata-planning, address-conversion, serialization, write, synchronization,
    /// and truncation errors.
    ///
    /// # Panics
    ///
    /// Panics if persistence is disabled.
    pub(in crate::edit) fn publish_persisting(
        &mut self,
        new_root: PhysicalAddress,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
        placement: TailPlacement,
        purpose: PublicationPurpose,
    ) -> Result<(), Error> {
        // PAGE uses separate managers for each page type.
        if self.space.is_paged() {
            return self.commit_persisting_paged(new_root, to_free, placement, purpose);
        }
        let widths = self.context.widths();
        let offset_width = widths.offsets;
        let (strategy, threshold, page_size, mut old_blocks) = {
            // Only persistent commit paths call this method.
            let ps = self
                .persist
                .as_ref()
                .expect("commit_persisting is only called when persistence is armed");
            (
                ps.strategy,
                ps.threshold,
                ps.page_size,
                ps.old_blocks.clone(),
            )
        };
        if placement == TailPlacement::Preserve {
            old_blocks.clear();
        }
        let base = self.context.addresses();

        // Resolve the stored extension address before reading its object header.
        let old_ext_addr = base
            .resolve(
                self.context
                    .superblock()
                    .superblock_extension_address
                    .filter(|a| !a.is_undefined(offset_width))
                    .ok_or(Error::EditUnsupported(
                        "a persisting file has no superblock extension to update",
                    ))?,
            )?
            .get();

        // File Space Info has a fixed size, so placeholder addresses give the extension's length.
        let placeholder = file_space_info::persistent_single_manager(
            offset_width,
            strategy,
            threshold,
            page_size,
            0,
            0,
        );
        let ext_len = object_header::build_v2_object_header(
            &self.rewrite_extension_region(old_ext_addr, &placeholder)?,
        )?
        .len() as u64;

        // Reserve from previously reusable space. This commit's pending frees remain live
        // until the root publication and cannot supply the tail reservation.
        let current_eoa = self.end_of_allocation()?;
        let limit = (placement == TailPlacement::ReuseOnly)
            .then(|| old_blocks.iter().map(|extent| extent.start()).min())
            .flatten();
        let recycled = if let TailPlacement::Recycle(target) = placement {
            let layout =
                self.recycled_tail_layout(current_eoa, &to_free, &old_blocks, ext_len, target);
            if layout.is_none() {
                old_blocks.push(target);
            }
            layout
        } else {
            None
        };
        let (post, placed_at, tail_len, eoa) = recycled.unwrap_or_else(|| {
            self.flat_tail_layout(
                current_eoa,
                &to_free,
                &old_blocks,
                ext_len,
                widths,
                limit,
                purpose,
            )
        });
        if placed_at.is_none() && placement == TailPlacement::ReuseOnly {
            // The shrink pass requires a reusable allocation below the old tail.
            return Ok(());
        }
        let reused = placed_at.is_some();
        let ext_addr = placed_at.unwrap_or(current_eoa);
        let managers = ManagerSections::flat(post.extents().iter().copied());
        let plan =
            free_space_manager::plan_managers(&managers, ext_addr.checked_add(ext_len)?, widths);
        // For a reused tail, truncate to the planned EOA after publication. The post-commit
        // free list excludes dropped trailing regions and the suffix released by
        // `FreeList::release_trailing_run`. An appended tail ends the file itself.
        let tail_end = ext_addr.checked_add(tail_len)?;
        let final_eoa = if reused { eoa } else { tail_end };
        debug_assert!(
            tail_end <= final_eoa,
            "the tail must end at or below the end-of-allocation it was placed under"
        );
        let final_eof = self.context.addresses().resolve(final_eoa)?;
        let aggregation = self
            .aggregation
            .map(|state| state.reconcile(final_eoa))
            .transpose()?;

        let flat_block = match plan.blocks.as_slice() {
            [] => None,
            [block] if block.kind == ManagerKind::Flat => Some(block),
            [block] => {
                return Err(Error::Format(FormatError::Internal(format!(
                    "flat free-space plan produced a {:?} manager",
                    block.kind
                ))));
            }
            _ => {
                return Err(Error::Format(FormatError::Internal(
                    "flat free-space plan produced more than one manager".into(),
                )));
            }
        };

        // An empty free list is recorded with persistence enabled and undefined manager addresses.
        let ext_oh = if let Some(block) = flat_block {
            // The C library truncates the self-referential suffix above this EOA in
            // `H5MF_tidy_self_referential_fsm_hack` (`H5MF.c`, HDF5 1.10.11).
            // DRAW storage remains below that suffix.
            let eoa_pre_fsm = final_eoa;
            let info = file_space_info::persistent_single_manager(
                offset_width,
                strategy,
                threshold,
                page_size,
                block.fshd_addr.get(),
                eoa_pre_fsm.get(),
            );
            object_header::build_v2_object_header(
                &self.rewrite_extension_region(old_ext_addr, &info)?,
            )?
        } else {
            // With no managers to allocate, libhdf5 records the final end of allocation
            // (`H5MF_settle_meta_data_fsm` and `H5MF__close_aggrfs` in `H5MF.c`, HDF5 1.14.6).
            let info = file_space_info::persistent_empty(
                offset_width,
                strategy,
                threshold,
                page_size,
                final_eoa.get(),
            );
            object_header::build_v2_object_header(
                &self.rewrite_extension_region(old_ext_addr, &info)?,
            )?
        };
        // The reservation uses the same fixed message size as the real extension.
        debug_assert_eq!(
            ext_oh.len() as u64,
            ext_len,
            "extension length must be stable across the placeholder and real messages"
        );

        let region = FreeExtent::new(ext_addr, tail_len).ok_or(Error::Format(
            FormatError::OffsetOverflow {
                offset: ext_addr.get(),
                length: tail_len,
            },
        ))?;
        let new_old_blocks = vec![region];

        // The extension and manager blocks remain unreferenced until superblock publication.
        self.write_tail_block(region, ext_addr, &ext_oh)?;
        if let Some(block) = flat_block {
            let fshd_len = hdf5_pure_format::__private::free_space_manager_header_len(widths);
            let fsse_allocated = tail_len
                .checked_sub(ext_len)
                .and_then(|manager_len| manager_len.checked_sub(fshd_len))
                .ok_or(Error::Format(FormatError::Internal(
                    "flat free-space tail is shorter than its extension and manager header".into(),
                )))?;
            let (fshd, fsse) =
                hdf5_pure_format::__private::serialize_free_space_manager_with_allocation(
                    widths,
                    &block.sections,
                    block.fshd_addr,
                    block.fsse_addr,
                    block.class,
                    Some(fsse_allocated),
                )?;
            self.write_tail_block(region, block.fshd_addr, &fshd)?;
            self.write_tail_block(region, block.fsse_addr, &fsse)?;
        }
        // Complete the content barrier before publishing the root, EOF, and extension.
        self.persistence_barrier(purpose)?;
        let mut new_sb = self.context.superblock().clone();
        new_sb.root_group_address = self.context.addresses().relative(new_root)?;
        new_sb.eof_address = final_eof.get();
        new_sb.superblock_extension_address = Some(ext_addr);
        // Preserve any status flags held by this session.
        new_sb.set_consistency_flags(self.held_status_flags)?;
        let publication = self.context.propose(new_sb)?.encode()?;
        let confirmed = self
            .aggregation
            .as_ref()
            .is_none_or(|state| state.publication_confirmed);
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = false;
        }
        self.publish_persistent_superblock(publication, purpose)?;

        self.space.adopt_flat_post_free(post);
        self.fsm_dirty = false;
        self.persist = Some(PersistState {
            strategy,
            threshold,
            page_size,
            old_blocks: new_old_blocks,
        });
        self.aggregation = aggregation;
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = false;
        }

        // Publish the smaller EOF before truncating. The adopted managers exclude the suffix.
        if final_eof < PhysicalAddress::new(self.image.len()) {
            self.truncate_persistent(final_eof, purpose)?;
        }

        // Record the file length associated with the published managers.
        self.fsm_len = self.image.len();
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = confirmed;
        }
        Ok(())
    }

    /// Returns prospective flat free space and EOA after releasing trailing regions.
    ///
    /// The result includes eligible `to_free` and `old_blocks` spans under the session's
    /// retention threshold. Those spans remain live until publication, so this method computes
    /// a temporary list through [`SessionSpace::flat_post_free`].
    ///
    /// # Panics
    ///
    /// Panics if the session does not use a flat free-space strategy.
    pub(in crate::edit) fn flat_post_free(
        &self,
        eof: StoredAddress,
        to_free: &[(StoredAddress, u64, FreeClass)],
        old_blocks: &[FreeExtent],
    ) -> (FreeList, StoredAddress) {
        let extents = to_free
            .iter()
            .filter_map(|&(addr, len, _)| FreeExtent::new(addr, len))
            .chain(old_blocks.iter().copied());
        // PAGE sessions are dispatched before a flat tail is planned.
        self.space
            .flat_post_free(eof, extents)
            .expect("the flat commit tail requires a flat session strategy")
    }

    /// Reserves a flat manager tail and computes the free space and EOA to publish.
    ///
    /// Returns the prospective free list, reserved stored address, reserved length, and EOA.
    /// The address is [`None`] when the caller must append at `current_eoa`. In that case,
    /// the returned EOA is `current_eoa`, before appending the tail.
    ///
    /// Reservations come from session free space. With a `limit`, only extents ending at or
    /// below it are eligible.
    /// Placement retries with a larger reservation when the serialized managers outgrow it.
    /// After bounded retries, the caller appends. Spare reservation bytes remain owned by
    /// the manager allocation until it is superseded.
    ///
    /// During allocation publication with a nonempty manager set, the writer rounds the complete
    /// extension and manager length to a power of two when representable, through
    /// [`flat_tail_capacity`]. The writer serializes spare bytes as section-list padding.
    ///
    /// The caller restores its allocation snapshot after a failure before publication.
    pub(in crate::edit) fn flat_tail_layout(
        &mut self,
        current_eoa: StoredAddress,
        to_free: &[(StoredAddress, u64, FreeClass)],
        old_blocks: &[FreeExtent],
        ext_len: u64,
        widths: FormatWidths,
        limit: Option<StoredAddress>,
        purpose: PublicationPurpose,
    ) -> (FreeList, Option<StoredAddress>, u64, StoredAddress) {
        /// Limits reservation retries when the serialized manager length increases.
        const ROUNDS: usize = 4;

        // Size the append fallback before trying reservations from reusable space.
        let (probe, _) = self.flat_post_free(current_eoa, to_free, old_blocks);
        let managers = ManagerSections::flat(probe.extents().iter().copied());
        let appended_len = flat_tail_capacity(ext_len, &managers, widths, purpose);
        let mut proposed = appended_len;

        for _ in 0..ROUNDS {
            let allocation = if let Some(limit) = limit {
                self.space.allocate_flat_below(proposed, limit)
            } else {
                self.space.allocate(proposed, AllocationClass::Metadata)
            };
            let Some(reserved) = allocation else {
                break;
            };
            let at = reserved.start();
            let (mut post, eoa) = self.flat_post_free(current_eoa, to_free, old_blocks);
            // Exclude the released suffix before sizing the managers. The tail reservation
            // has already been removed from this list, so the trailing run starts at or
            // above the reservation's end.
            let eoa = post.release_trailing_run(eoa, proposed);
            let managers = ManagerSections::flat(post.extents().iter().copied());
            let len = flat_tail_capacity(ext_len, &managers, widths, purpose);
            if post.is_empty() && len < proposed {
                self.space
                    .return_allocation(reserved, AllocationClass::Metadata);
                break;
            }
            if len <= proposed {
                debug_assert!(
                    reserved.end() <= eoa,
                    "a reused tail at {at:?} of {proposed} bytes runs past the end-of-allocation \
                     {eoa:?} it was placed under"
                );
                return (post, Some(at), proposed, eoa);
            }
            self.space
                .return_allocation(reserved, AllocationClass::Metadata);
            proposed = len;
        }
        // Append beyond the current EOA, retaining the prospective free list below the tail.
        (probe, None, appended_len, current_eoa)
    }

    /// Publishes the root, extension, and final per-class free sections of a persistent PAGE edit.
    ///
    /// Plans the complete metadata reservation before writing. The serialized managers and the
    /// adopted session state include the same page tails and exclude the complete reservation,
    /// including owned section-list padding. Placement uses appended pages or space already free
    /// before this publication. Pending object frees and superseded managers cannot fund it.
    ///
    /// Writes the planned blocks before publishing their addresses in the superblock. Publishes
    /// a smaller EOA before truncating the file, and adopts the planned state after publication
    /// and any truncation succeed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the persistent extension cannot be updated,
    /// a format error if planning or serialization fails, or an I/O error if writing,
    /// synchronization, or truncation fails.
    ///
    /// # Panics
    ///
    /// Panics if persistence is disabled.
    pub(in crate::edit) fn commit_persisting_paged(
        &mut self,
        new_root: PhysicalAddress,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
        placement: TailPlacement,
        purpose: PublicationPurpose,
    ) -> Result<(), Error> {
        let widths = self.context.widths();
        let offset_width = widths.offsets;
        let (strategy, threshold, page_size, old_blocks) = {
            // Only persistent commit paths call this method.
            let ps = self
                .persist
                .as_ref()
                .expect("commit_persisting is only called when persistence is armed");
            (
                ps.strategy,
                ps.threshold,
                ps.page_size,
                ps.old_blocks.clone(),
            )
        };

        let base = self.context.addresses();

        // Record existing tail padding before planning the metadata reservation.
        self.pad_to_page()?;

        // Resolve the stored extension address before reading its object header.
        let old_ext_addr = base
            .resolve(
                self.context
                    .superblock()
                    .superblock_extension_address
                    .filter(|a| !a.is_undefined(offset_width))
                    .ok_or(Error::EditUnsupported(
                        "a persisting file has no superblock extension to update",
                    ))?,
            )?
            .get();

        // Placeholder addresses give the extension's length before placement.
        let placeholder = file_space_info::persistent_managers(
            strategy,
            threshold,
            page_size,
            [UNDEF; NUM_FILE_FSM_MANAGERS],
            0,
        );
        let ext_len = object_header::build_v2_object_header(
            &self.rewrite_extension_region(old_ext_addr, &placeholder)?,
        )?
        .len() as u64;

        let current_eoa = self.end_of_allocation()?;
        let freed: Vec<_> = to_free
            .iter()
            .filter_map(|&(addr, len, class)| {
                FreeExtent::new(addr, len).map(|extent| (extent, class))
            })
            .collect();
        let Some(plan) = paged_persistence::PagedPersistencePlan::new(
            &self.space,
            current_eoa,
            &freed,
            &old_blocks,
            ext_len,
            widths,
            placement == TailPlacement::ReuseOnly,
        )?
        else {
            return Ok(());
        };
        let paged_persistence::PagedPersistencePlan {
            tail,
            slots,
            blocks,
        } = plan;
        let ext_addr = tail.allocation.start();
        let blocks_len = tail.allocation.len();
        let final_eoa = tail.final_eoa;
        let reused = !tail.appended;
        if reused {
            let reserved = self.space.allocate(blocks_len, AllocationClass::Metadata);
            if reserved != Some(tail.allocation) {
                return Err(FormatError::Internal(
                    "a PAGE tail reservation differs from its pure plan".into(),
                )
                .into());
            }
        } else {
            self.begin_page(PageType::Meta)?;
        }
        let final_eof = self.context.addresses().resolve(final_eoa)?;

        let info = if blocks.is_empty() {
            file_space_info::persistent_empty(
                offset_width,
                strategy,
                threshold,
                page_size,
                final_eoa.get(),
            )
        } else {
            file_space_info::persistent_managers(
                strategy,
                threshold,
                page_size,
                slots.map(StoredAddress::get),
                final_eoa.get(),
            )
        };
        let ext_oh = object_header::build_v2_object_header(
            &self.rewrite_extension_region(old_ext_addr, &info)?,
        )?;
        debug_assert_eq!(
            ext_oh.len() as u64,
            ext_len,
            "extension length must be stable across the placeholder and real messages"
        );

        // Include section-list padding in the allocation recorded for future reclamation.
        let region = FreeExtent::new(ext_addr, blocks_len).ok_or(Error::Format(
            FormatError::OffsetOverflow {
                offset: ext_addr.get(),
                length: blocks_len,
            },
        ))?;
        let new_old_blocks = vec![region];

        // The extension and manager blocks remain unreferenced until superblock publication.
        self.write_tail_block(region, ext_addr, &ext_oh)?;
        for block in &blocks {
            self.write_tail_block(region, block.header_address, &block.header)?;
            self.write_tail_block(region, block.section_list_address, &block.section_list)?;
        }
        // Extend an appended reservation to the page-aligned EOF computed by the plan.
        if !reused {
            self.pad_zeros_to(final_eof.get())?;
        }
        // Complete the content barrier before publishing the root, EOF, and extension.
        self.persistence_barrier(purpose)?;
        let mut new_sb = self.context.superblock().clone();
        new_sb.root_group_address = self.context.addresses().relative(new_root)?;
        new_sb.eof_address = final_eof.get();
        new_sb.superblock_extension_address = Some(ext_addr);
        // Preserve the crash mark throughout a page-buffered session.
        new_sb.set_consistency_flags(self.held_status_flags)?;
        self.publish_persistent_superblock(self.context.propose(new_sb)?.encode()?, purpose)?;

        // Complete EOF publication before removing the trailing suffix.
        if final_eof < PhysicalAddress::new(self.image.len()) {
            self.truncate_persistent(final_eof, purpose)?;
        }

        // Adopt prospective free space after publication and any truncation succeed.
        self.space.adopt_paged_post_free(tail.post, !reused, None);
        self.fsm_dirty = false;
        self.persist = Some(PersistState {
            strategy,
            threshold,
            page_size,
            old_blocks: new_old_blocks,
        });
        // The rewritten managers describe persistent free sections at this file length.
        self.fsm_len = self.image.len();
        Ok(())
    }

    /// Publishes a replacement root and EOF in a version 0 or 1 superblock.
    ///
    /// Clears the root symbol-table entry's cached metadata and writes the object-header
    /// address last. These fields are defined in "Format Signature and Superblock" and
    /// ["Symbol Table Entry"][entry] of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns the write error if a field cannot be updated.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
    /// [entry]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
    fn repoint_v0v1_root(
        &mut self,
        new_root: StoredAddress,
        new_eof: PhysicalAddress,
    ) -> Result<(), Error> {
        let os = self.context.superblock().offset_size.get();
        let addr_width = u64::from(os);
        // Field layout after the fixed prefix: base / free-space / EOF / driver
        // addresses, then the root symbol-table entry (link-name offset, object
        // header address, cache type(4), reserved(4), scratch(16)). The prefix is
        // 24 bytes for v0 and 28 for v1 (the latter adds indexed-storage-K).
        let var_start = if self.context.superblock().version() == SuperblockVersion::Zero {
            24
        } else {
            28
        };
        let base = self.context.signature().get() + var_start;
        let eof_off = base + 2 * addr_width;
        let ste = base + 4 * addr_width;
        let oh_addr_off = ste + addr_width;
        let cache_off = ste + 2 * addr_width;
        self.write_at(eof_off, &new_eof.get().to_le_bytes()[..usize::from(os)])?;
        self.write_at(cache_off, &[0u8; 4])?; // cache type = none
        self.write_at(cache_off + 8, &[0u8; 16])?; // clear scratch-pad
        self.write_at(
            oh_addr_off,
            &new_root.get().to_le_bytes()[..usize::from(os)],
        )?;
        Ok(())
    }

    /// Marks publication uncertain before writing, then adopts the superblock after its barrier.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the proposal's format or address domain differs
    /// from the current context. Returns the write or barrier error if publication fails.
    /// Retains [`PublicationState::Uncertain`] on error.
    fn publish_superblock(&mut self, publication: SuperblockPublication) -> Result<(), Error> {
        self.publish_persistent_superblock(publication, PublicationPurpose::Commit)
    }

    /// Adopts the superblock after its write and the selected publication barrier succeed.
    ///
    /// Sets [`PublicationState::Uncertain`] before writing and retains it on failure.
    ///
    /// # Errors
    ///
    /// Returns the write, barrier, or context-adoption error.
    fn publish_persistent_superblock(
        &mut self,
        publication: SuperblockPublication,
        purpose: PublicationPurpose,
    ) -> Result<(), Error> {
        self.publication = PublicationState::Uncertain;
        self.write_at(self.context.signature().get(), publication.bytes())?;
        self.persistence_barrier(purpose)?;
        self.context.adopt(publication.into_proposed())?;
        self.publication = PublicationState::Published;
        Ok(())
    }

    /// Truncates after confirmed EOF publication and applies the session's barrier.
    ///
    /// # Errors
    ///
    /// Returns the truncation or barrier error.
    fn truncate_published(&mut self, cut: PhysicalAddress) -> Result<(), Error> {
        self.truncate_persistent(cut, PublicationPurpose::Commit)
    }

    /// Truncates to the confirmed EOF and applies the selected publication barrier.
    ///
    /// # Errors
    ///
    /// Returns the truncation or barrier error.
    fn truncate_persistent(
        &mut self,
        cut: PhysicalAddress,
        purpose: PublicationPurpose,
    ) -> Result<(), Error> {
        debug_assert_eq!(self.publication, PublicationState::Published);
        self.image.truncate(cut.get())?;
        self.persistence_barrier(purpose)
    }

    /// Drains allocation writes or applies the staged commit's buffering boundary.
    ///
    /// [`SyncPolicy::Always`] synchronizes file content and metadata. Under
    /// [`SyncPolicy::OnClose`], allocation publication drains every buffering mode, while staged
    /// commits use the general ordering barrier.
    ///
    /// # Errors
    ///
    /// Returns the buffered-write or synchronization error.
    fn persistence_barrier(&mut self, purpose: PublicationPurpose) -> Result<(), Error> {
        match (purpose, self.sync_policy) {
            (PublicationPurpose::Allocation, SyncPolicy::OnClose) => self.image.flush(),
            _ => self.barrier(),
        }
    }
}

/// Returns the flat publication allocation length with optional spare capacity.
///
/// For [`PublicationPurpose::Allocation`] with a nonempty manager set, rounds the complete
/// extension and manager length up to a power of two. Returns the serialized length if rounding
/// would overflow or the section-list capacity exceeds the file's length width. For staged
/// commits and empty manager sets, also returns the serialized length.
fn flat_tail_capacity(
    ext_len: u64,
    managers: &[ManagerSections],
    widths: FormatWidths,
    purpose: PublicationPurpose,
) -> u64 {
    let len = ext_len + free_space_manager::manager_blocks_len(managers, widths);
    if managers.is_empty() || matches!(purpose, PublicationPurpose::Commit) {
        return len;
    }
    let Some(capacity) = len.checked_next_power_of_two() else {
        return len;
    };
    let overhead = ext_len + hdf5_pure_format::__private::free_space_manager_header_len(widths);
    let maximum = u64::MAX >> (64 - 8 * widths.lengths.get());
    if capacity - overhead > maximum {
        return len;
    }
    capacity
}

#[cfg(test)]
mod tests {
    use hdf5_pure_space::__private::Extent;
    use hdf5_pure_space::__private::ManagerSections;
    use rstest::rstest;

    use super::PublicationPurpose;
    use crate::address::{StoredAddress, StoredAddressExt};
    use crate::free_space_manager;
    use crate::width::FormatWidths;

    #[rstest]
    #[case::empty(1000, 0, 8, PublicationPurpose::Allocation, None)]
    #[case::staged(1000, 1, 8, PublicationPurpose::Commit, None)]
    #[case::narrow(1000, 1, 2, PublicationPurpose::Allocation, Some(2048))]
    #[case::wide(1000, 1, 8, PublicationPurpose::Allocation, Some(2048))]
    #[case::length_limit(60_000, 1000, 2, PublicationPurpose::Allocation, None)]
    fn flat_tail_capacity_preserves_representable_manager_allocations(
        #[case] ext_len: u64,
        #[case] sections: u64,
        #[case] lengths: u8,
        #[case] purpose: PublicationPurpose,
        #[case] capacity: Option<u64>,
    ) {
        let widths = FormatWidths::from_sizes(8, lengths).unwrap();
        let managers = ManagerSections::flat(
            (0..sections).map(|i| Extent::new(StoredAddress::new(i * 2), 1).unwrap()),
        );
        let minimum = ext_len + free_space_manager::manager_blocks_len(&managers, widths);
        let actual = super::flat_tail_capacity(ext_len, &managers, widths, purpose);
        assert_eq!(actual, capacity.unwrap_or(minimum));
        assert!(actual >= minimum);
        assert!(actual < 2 * minimum);
    }
}
