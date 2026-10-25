use std::collections::{BTreeMap, HashMap};

use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_space::__private as free_space;
use hdf5_pure_space::__private::{
    AllocationClass, Extent as FreeExtent, FreeClass, FreeList, ManagerKind, ManagerSections,
    PageType,
};

#[cfg(doc)]
use hdf5_pure_space::__private::SessionSpace;

#[cfg(doc)]
use super::SyncPolicy;
use crate::SuperblockVersion;
use crate::access_mode::AccessMode;
use crate::address::{AddressContext, StoredAddress, StoredAddressExt};
use crate::datatype;
use crate::error::{Error, FormatError};
use crate::file_space_info::{self, NUM_FILE_FSM_MANAGERS};
use crate::free_space_manager;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::source::{BaseOffsetSource, Source};
use crate::superblock::{self, Superblock};
use crate::type_builders::ObjectRefTarget;
use crate::width::FormatWidths;

use super::attributes::{self, AttrOp, EditedAttrs};
use super::dataset::{FlatDataset, MovingWrite, WritePlan};
use super::group::{self, LinkCreationOrder};
use super::object_header;
use super::staging::{StagedEdits, StagedScope};
use super::{
    FreeSnapshot, InvalidatedAddresses, MAX_LINK_GRAPH_NODES, Node, TailPlacement, UNDEF,
    WriteEngine,
};
use super::{PersistState, copy, paged_persistence};
use super::{
    ensure_ancestors, group_by_parent, retain_disjoint_in_bounds, screen_resolved_references,
};

impl WriteEngine {
    /// Publishes a persistent commit after returning the unspent append reserve.
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
    pub(super) fn commit_persisting_releasing_reserve(
        &mut self,
        new_root: u64,
        mut to_free: Vec<(StoredAddress, u64, FreeClass)>,
    ) -> Result<(), Error> {
        self.release_aggregates_for_tail(&mut to_free)?;
        self.release_reserve();
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
    pub(super) fn shrink_to_the_trailing_free_run(&mut self, new_root: u64) -> Result<(), Error> {
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
        let eoa = self.end_of_allocation()?.get();
        if tail_end != eoa
            && (self.space.is_paged() || self.space.trailing_run_start(eoa) != tail_end)
        {
            return Ok(());
        }
        // Apply the retention threshold of `FreeList::release_trailing_run` to the combined
        // free run and old tail before attempting a rewrite.
        let run_start = self.space.trailing_run_start(tail_start);
        if eoa - run_start < 2 * free_space::TRAILING_RESERVE_TAILS * (tail_end - tail_start) {
            return Ok(());
        }
        self.commit_persisting(new_root, Vec::new(), TailPlacement::ReuseOnly)
    }

    /// Issues pending writes, synchronizes the file, and completes pending PAGE publication.
    ///
    /// The synchronization barrier is durable under every [`SyncPolicy`]. Publication recovery
    /// verifies issued bytes before adopting the superblock and leaves later staged edits pending.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if writing or synchronization fails. Returns the publication recovery
    /// error if the issued superblock cannot be read or verified.
    pub(crate) fn force_sync(&mut self) -> Result<(), Error> {
        self.image.sync_all()?;
        self.complete_page_publication()
    }

    /// Finalizes free-space metadata or classic aggregate tails at session teardown.
    ///
    /// A persisting session rewrites stale free-space managers and returns unspent immediate
    /// append reserves. A persistent session also returns unused aggregate spare through those
    /// managers. A session with classic aggregation releases unused aggregate space at EOA after
    /// publishing the shorter EOA with the current root. An uncertain publication prevents
    /// aggregate cleanup. Non-persisting sessions without classic aggregation require no
    /// finalization.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if metadata serialization, aggregate planning, or address
    /// resolution fails. Returns [`Error::Io`] if writing, syncing, or truncating fails.
    pub(crate) fn finalize_persist(&mut self) -> Result<(), Error> {
        if self.aggregation.is_some() {
            if self.persist.is_some() {
                return self.finalize_persistent_aggregates();
            }
            return self.finalize_aggregates();
        }
        if self.persist.is_none() || (self.image.len() == self.fsm_len && self.reserved.is_empty())
        {
            return Ok(());
        }
        self.commit_persisting_releasing_reserve(
            self.addresses
                .absolute(self.superblock.root_group_address)?,
            Vec::new(),
        )
    }

    /// Applies staged edits using the session's file-space strategy and synchronization policy.
    ///
    /// Tree edits rebuild affected group headers and publish the new root through the
    /// superblock. Dataset value writes may update existing storage directly. On success,
    /// the staged set is cleared.
    ///
    /// Rejection during preflight preserves the staged set for retry. Once application
    /// consumes the staged set, a failed attempt discards that work. Writes into reusable
    /// or newly allocated space may remain unreferenced after a failure.
    ///
    /// On failure before root publication is attempted, the editor restores allocator state and
    /// attempts to restore prior values from the undo journal. Rollback is disabled once
    /// publication is attempted because a failed publication write may already have changed
    /// the file.
    ///
    /// After publication, reference patching or optional root-header reclamation can fail.
    /// The committed tree retains the batch's values, including its in-place overwrites.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if an edit or an encountered file structure cannot
    /// be reproduced. Other preparation and I/O errors propagate from planning, writing,
    /// or synchronizing the file. Returns [`Error::CommitPartiallyApplied`] if restoring prior
    /// values from the undo journal fails, with both the original failure and the restore error.
    pub fn commit(&mut self) -> Result<(), Error> {
        if self.fresh_page_boundary.is_some() && self.publish_attempted {
            self.barrier()?;
            self.complete_page_publication()?;
        }
        self.synchronize_aggregates()?;
        let snapshot = self.snapshot_free();
        self.publish_attempted = false;
        // Cleared at entry, like `superseded_heaps` and for the same reason: a
        // previous attempt that unwound past this function — a panic caught by
        // the session lock, which recovers from poisoning — would otherwise
        // leave entries behind, and this commit's rollback would replay bytes
        // captured before it over whatever lives at those addresses now. The
        // clear on the way out is a separate duty: it stops a journal being
        // held across the idle time between commits.
        self.inplace_undo.clear();
        // The staged set comes out for the duration of the attempt and goes back
        // if the attempt refuses, so a refusal costs the session no staged work
        // (issue #316). What goes back is whatever the attempt did not take: its
        // preflight only reads, and its apply phase takes the whole set in one
        // move at the point of no return, so a preflight refusal restores every
        // edit and a failure past that point restores nothing.
        let mut staged = std::mem::take(&mut self.staged);
        let mut result = self.commit_inner(&mut staged);
        if result.is_err() && !self.publish_attempted {
            // The three legs of the rollback: the file's values, the free
            // lists, and the staged set. Nothing observes the order — the first
            // touches the file and the others only in-memory state, inside one
            // synchronous call — so it is written the way it reads: put the
            // file back, then re-offer what the attempt drew from (issue #344).
            let restored = self.undo_inplace_writes();
            self.restore_free(snapshot);
            self.staged = staged;
            // The set is back, so nothing this session staged has been consumed
            // and every handle onto one of those creations is still pending.
            if let Err(restore) = restored {
                // Both are worth carrying: the refusal is what the caller has to
                // fix, and the write failure is what proves the file changed.
                let refusal = result.expect_err("only a failed attempt is rolled back");
                result = Err(Error::CommitPartiallyApplied {
                    refusal: Box::new(refusal),
                    restore: Box::new(restore),
                });
            }
        } else {
            // The set is gone — published, or in the file as slack a later
            // commit must not re-issue — so a handle onto one of its creations
            // stops being pending and addresses the file from here on.
            self.staged_generation += 1;
        }
        if self.publish_attempted
            && let Some(aggregation) = &mut self.aggregation
        {
            aggregation.publication_confirmed = result.is_ok();
        }
        // For the paths that took no rollback — a commit that succeeded, whose
        // overwrites are now the file's values, and one that failed with the
        // publish already issued, where undoing anything is unsound. A rollback
        // drained the journal on its way through, so it owes nothing here.
        self.inplace_undo.clear();
        if self.fresh_page_boundary.is_some() && self.publish_attempted {
            if result.is_ok() {
                self.publish_attempted = false;
            } else {
                // A failed barrier can leave the new root on disk. Pending writes and uncertain
                // bytes remain owned until a later commit completes the barrier and recovery.
                self.recover_page_publication().ok();
            }
        }
        result
    }

    /// Verifies pending PAGE publication and clears its recovery state on success.
    ///
    /// The caller completes its ordering or synchronization barrier first. Later staged edits
    /// remain pending if verification fails.
    ///
    /// # Errors
    ///
    /// Returns the error from [`recover_page_publication`](Self::recover_page_publication).
    pub(super) fn complete_page_publication(&mut self) -> Result<(), Error> {
        if self.fresh_page_boundary.is_some() && self.publish_attempted {
            self.recover_page_publication()?;
            self.publish_attempted = false;
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
    pub(super) fn recover_page_publication(&mut self) -> Result<(), Error> {
        let len = hdf5_pure_format::__private::serialize_superblock(&self.superblock)?.len();
        let accepted = self.image.read_exact_at(self.sb_sig_off, len)?;
        let mut published = vec![0; len];
        self.image.read_disk_at(self.sb_sig_off, &mut published)?;
        if published != accepted {
            return Err(Error::EditUnsupported(
                "PAGE publication bytes differ from the session image after a failed write; \
                 close and reopen the file before editing again",
            ));
        }
        let recovered = superblock::read(published.as_slice())?.superblock;
        let mut expected = self.superblock.clone();
        expected.root_group_address = recovered.root_group_address;
        expected.eof_address = recovered.eof_address;
        if hdf5_pure_format::__private::serialize_superblock(&expected)? != published {
            return Err(Error::EditUnsupported(
                "PAGE superblock framing changed after a failed publication; \
                 close and reopen the file before editing again",
            ));
        }
        self.superblock = recovered;
        Ok(())
    }

    /// Captures allocator state and heap allocation records for rollback before publication.
    pub(super) fn snapshot_free(&self) -> FreeSnapshot {
        FreeSnapshot {
            space: self.space.allocation_snapshot(),
            aggregation: self.aggregation,
            reserved: self.reserved.clone(),
            vl_overwrite_heaps: self.vl_overwrite_heaps.clone(),
        }
    }

    /// Restores allocator state and heap allocation records after failure before publication.
    ///
    /// The snapshot was captured by [`snapshot_free`](Self::snapshot_free).
    ///
    /// That rests on nothing the live tree reaches naming a span the attempt
    /// allocated, which takes two rules rather than one. Every object a commit
    /// *places* is unreachable until the superblock repoint; and the one write
    /// that does not wait for the repoint — a same-length value overwrite — is
    /// planned only over bytes that carry no allocated address, because
    /// [`prepare_write`](Self::prepare_write) relocates a staged
    /// variable-length overwrite instead of writing it in place, which is
    /// exactly the counterexample (issue #321).
    ///
    /// [`undo_inplace_writes`](Self::undo_inplace_writes) has nevertheless run
    /// by the time this does, so the file's values are back before its regions
    /// are offered again (issue #344).
    pub(super) fn restore_free(&mut self, snapshot: FreeSnapshot) {
        self.vl_overwrite_heaps = snapshot.vl_overwrite_heaps;
        self.space.restore_allocation(snapshot.space);
        self.aggregation = snapshot.aggregation;
        self.reserved = snapshot.reserved;
    }

    /// Overwrites `raw` at `at`, keeping the bytes it replaces so a commit that
    /// fails before its repoint can put them back.
    ///
    /// The read covers exactly the range about to be written, which the write
    /// preflight sized and bounds-checked against the on-disk dataset — a
    /// same-length overwrite by construction.
    pub(super) fn write_inplace_journaled(&mut self, at: u64, raw: &[u8]) -> Result<(), Error> {
        let prior = self.image().read_exact_at(at, raw.len())?;
        self.inplace_undo.push((at, prior));
        self.write_at(at, raw)
    }

    /// Put back the values the refused commit's same-length in-place overwrites
    /// replaced, and drop the journal.
    ///
    /// The third leg of a refused commit's rollback, beside
    /// [`restore_free`](Self::restore_free) and the staged set: those two give
    /// back what the attempt *consumed*, this gives back what it *changed*. It
    /// is the only one of the three that has to touch the file, because a
    /// same-length overwrite is the only edit a commit writes over bytes the
    /// live root already reaches (issue #344).
    ///
    /// # Errors
    ///
    /// Every entry is attempted even after one fails, so the file is restored as
    /// far as it can be, and the first failure met is what comes back. That is the
    /// one outcome leaving values from a batch that was refused, which is why
    /// [`commit`](Self::commit) reports it as
    /// [`Error::CommitPartiallyApplied`] rather than as the refusal.
    pub(super) fn undo_inplace_writes(&mut self) -> Result<(), Error> {
        let journal = core::mem::take(&mut self.inplace_undo);
        if journal.is_empty() {
            return Ok(());
        }
        let mut failed = None;
        // Newest first, and load-bearing rather than conventional. The commit
        // refuses to overwrite the same *path* twice, which is not the same as
        // the same dataset: an object reachable through two hard links has two
        // paths and one data block, and a same-length overwrite through either
        // is allowed precisely because it rewrites the block every link sees.
        // Two such writes in one commit therefore journal the same address
        // twice, the second entry holding what the first write put there.
        // Replaying oldest-first would finish on that value — one from the very
        // batch being rolled back.
        for (at, prior) in journal.into_iter().rev() {
            if let Err(e) = self.write_at(at, &prior) {
                failed.get_or_insert(e);
            }
        }
        match failed {
            Some(e) => Err(e),
            // A restore that is not a barrier can be gathered and reordered
            // past a later write, which is the whole hazard write gathering
            // introduced (issue #288); order it here instead. The data-only
            // barrier is the one that fits, because putting bytes back into
            // blocks that already existed moves no end-of-file — it is a
            // narrower barrier than the `barrier()` the commit itself uses
            // after these writes, deliberately.
            None => self.barrier_data(),
        }
    }

    pub(super) fn commit_inner(&mut self, staged: &mut StagedEdits) -> Result<(), Error> {
        if staged.is_empty() {
            return Ok(());
        }

        // An attempt that failed partway may have recorded superseded collections
        // it never freed; this one must not free them on its behalf, since it is
        // not the commit that replaced what they hold.
        self.superseded_heaps.clear();
        // Drop every heap-collection provenance record this batch could falsify,
        // before any of them is read below.
        self.invalidate_heap_provenance(staged);

        // Invalidate the in-place-append geometry cache before doing any work. A
        // commit that reaches here rewrites and relocates object headers, frees
        // vacated regions into `self.space`, and may truncate the file, any of
        // which can leave a cached `Located` pointing at a moved header or into a
        // now-free-eligible region. Clearing at *entry* (rather than the success
        // tail) means a later failure — including one after the durable root flip,
        // which leaves the session reusable — never strands a stale cache. The
        // no-op fast return above does no such work, so it keeps the cache. The
        // The next in-place append re-locates against the fresh file.
        self.located.clear();
        self.resolved.clear();
        // Past this point the commit may relocate object headers, so every address
        // a caller captured earlier is suspect; see `committed`.
        self.committed = true;

        // On a file with a userblock, stored addresses are relative to this base
        // and the editor converts at every disk boundary (read `stored + base`,
        // write `file_offset - base`). Userblock support covers value overwrites,
        // additions of contiguous and chunked/filtered datasets, in-place and
        // relocating overwrites of every layout (chunked, contiguous, compact) with
        // reclaim, object deletion (with base-aware subtree reclaim), object copy
        // (in-file, and cross-file into a userblock destination), group creation,
        // and compact group attributes. Cross-file copy still requires a base-0
        // *source* (see [`copy_from`](Self::copy_from)).
        let base = self.addresses;

        // --- Preflight value overwrites (`write_dataset`) before any write, under
        // the same all-or-nothing contract as additions. Each is resolved,
        // validated (compatible datatype and matching on-disk shape),
        // and classified: a same-length contiguous overwrite is applied straight
        // in place (no header rewrite, no superblock flip), while a resize or
        // compact rewrite relocates the header and is staged against its parent
        // group so the commit below rebuilds it and patches the link. ---
        let mut inplace_writes: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut moving_writes: Vec<(ObjectPathBuf, LinkNameBuf, u64, MovingWrite)> = Vec::new();
        let mut write_targets: Vec<ObjectPathBuf> = Vec::new();
        // The file-wide hard-link count, computed lazily the first time a commit
        // relocates a header: such a write moves the object's header and patches
        // only the one parent link that names it, so an object reachable through
        // more than one hard link would have its other links left pointing at the
        // stale header. Refuse that rather than silently diverge the aliases (a
        // same-length in-place overwrite is unaffected — it rewrites the shared
        // data block, which every link sees).
        //
        // Read by four places, which is every way a header moves: a relocating
        // overwrite, a staged append, a dataset attribute edit, and — since
        // issue #327 — a rebuilt group, which had the same defect and none of
        // the guard.
        let mut incoming_links: Option<Option<HashMap<u64, u32>>> = None;
        for (full, fd) in staged.writes() {
            // A path named twice in one commit would write it twice (and double-
            // free a resized extent); require separate commits.
            if write_targets.contains(full) {
                return Err(Error::EditUnsupported(
                    "the same dataset is overwritten twice in one commit; use separate commits",
                ));
            }
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &full.as_path(),
            )?;
            match self.prepare_write(addr, fd, full)? {
                WritePlan::InPlace { data_addr, bytes } => {
                    inplace_writes.push((data_addr, bytes));
                }
                // A chunked in-place overwrite never carries staging — a staged
                // one is a `ChunkPayload::Deferred`, which relocates.
                WritePlan::InPlaceChunks { writes } => inplace_writes.extend(writes),
                WritePlan::Moving(mw) => {
                    // A relocating overwrite rewrites the dataset's header and data
                    // address. Every variant is base-aware on a userblock file: the
                    // chunked one rebuilds the chunk blob with stored addresses and
                    // reclaims the old storage base-relative, the contiguous one
                    // stores the relocated data address base-relative (and frees the
                    // old extent at its absolute offset), and the compact one carries
                    // its data inline. The parent link to the rewritten header is
                    // patched base-relative below.
                    //
                    // A relocating overwrite is safe only when this is the
                    // dataset's sole hard link. Compute the link graph once.
                    let counts = incoming_links
                        .get_or_insert_with(|| self.count_incoming_hard_links())
                        .as_ref();
                    match counts.and_then(|c| c.get(&addr)) {
                        Some(&1) => {}
                        _ => {
                            return Err(Error::EditUnsupported(
                                "overwriting a dataset that resizes or relocates its header is \
                                 only supported when it has a single hard link",
                            ));
                        }
                    }
                    let Some((parent, leaf)) = full.split_leaf() else {
                        return Err(Error::EditUnsupported("cannot overwrite the root group"));
                    };
                    moving_writes.push((parent, leaf, addr, mw));
                }
            }
            write_targets.push(full.clone());
        }

        // --- Preflight appends (`append_dataset`) under the same all-or-nothing,
        // single-hard-link contract. Each plans a relocating append — existing
        // chunk data stays in place; the appended (and any rewritten trailing)
        // chunks and a rebuilt Extensible-Array index are staged, and the whole is
        // treated like a relocating overwrite of the dataset's header (staged
        // against its parent group so the commit patches the link). A zero-length
        // append is a no-op and is dropped here. ---
        for (full, ab) in staged.appends() {
            let Some((parent, leaf)) = full.split_leaf() else {
                return Err(Error::AppendUnsupported("cannot append to the root group"));
            };
            if ab.raw().is_empty() {
                continue; // nothing to append
            }
            // A dataset overwritten or appended earlier in this commit would be
            // planned against a stale header and its old storage double-freed;
            // require separate commits.
            if write_targets.contains(full) {
                return Err(Error::AppendUnsupported(
                    "the same dataset is edited more than once in one commit; use separate commits",
                ));
            }
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &full.as_path(),
            )?;
            let mw = Self::prepare_append(&self.image(), addr, ab, base, self.widths)?;
            // A relocating append moves the dataset's object header and patches only
            // the one parent link that names it, so it is safe only when this is the
            // dataset's sole hard link (same rule as a relocating overwrite).
            let counts = incoming_links
                .get_or_insert_with(|| self.count_incoming_hard_links())
                .as_ref();
            match counts.and_then(|c| c.get(&addr)) {
                Some(&1) => {}
                _ => {
                    return Err(Error::AppendUnsupported(
                        "appending relocates the dataset header; only supported when it \
                         has a single hard link",
                    ));
                }
            }
            moving_writes.push((parent, leaf, addr, mw));
            write_targets.push(full.clone());
        }

        // --- Preflight dataset attribute edits (`set_dataset_attr` /
        // `remove_dataset_attr`) under the same all-or-nothing, single-hard-link
        // contract. Each gathers the dataset's verbatim object-header region,
        // applies the compact attribute ops to it, and stages a relocating
        // `AttrEdit` header rewrite against the parent group — like a value
        // overwrite, but the data-layout message (and thus the chunk data and index)
        // is preserved verbatim, so only the header moves. ---
        if !staged.dataset_attrs().is_empty() {
            // Collect the ops per dataset in first-seen path order, so multiple edits
            // to one dataset produce a single relocating header rewrite.
            let mut order: Vec<ObjectPathBuf> = Vec::new();
            let mut ops_by_path: HashMap<&ObjectPathBuf, Vec<&AttrOp>> = HashMap::new();
            for (path, op) in staged.dataset_attrs() {
                if !ops_by_path.contains_key(path) {
                    order.push(path.clone());
                }
                ops_by_path.entry(path).or_default().push(op);
            }
            for full in order {
                let ops = ops_by_path.remove(&full).unwrap();
                let Some((parent, leaf)) = full.split_leaf() else {
                    return Err(Error::EditUnsupported(
                        "cannot set a dataset attribute on the root group; use set_group_attr",
                    ));
                };
                // A dataset already overwritten or appended in this commit would be
                // planned against a stale header; require separate commits.
                if write_targets.contains(&full) {
                    return Err(Error::EditUnsupported(
                        "the same dataset is edited more than once in one commit (an attribute \
                         edit plus another edit); use separate commits",
                    ));
                }
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    AccessMode::ReadWrite,
                    &self.superblock,
                    self.widths,
                    self.addresses,
                    &full.as_path(),
                )?;
                // An attribute edit relocates the dataset's object header and patches
                // only the one naming link, so it is safe only when this is the
                // dataset's sole hard link (same rule as a relocating overwrite).
                let counts = incoming_links
                    .get_or_insert_with(|| self.count_incoming_hard_links())
                    .as_ref();
                match counts.and_then(|c| c.get(&addr)) {
                    Some(&1) => {}
                    _ => {
                        return Err(Error::EditUnsupported(
                            "editing a dataset attribute relocates its header; only supported \
                             when it has a single hard link",
                        ));
                    }
                }
                let region = Self::gather_oh_messages(&self.image(), addr, base)?;
                let edits =
                    attributes::plan_attr_ops(&self.image(), base, Some(addr), &region, &ops)?;

                moving_writes.push((
                    parent,
                    leaf,
                    addr,
                    MovingWrite::AttrEdit {
                        region: edits.region,
                        attrs: edits.attrs,
                        replaces_dense_storage: edits.replaces_dense_storage,
                    },
                ));
                write_targets.push(full);
            }
        }

        // Pre-commit object-header addresses this commit rewrites elsewhere, for
        // `InvalidatedAddresses` below: a reference naming one of these keeps an
        // address the commit is vacating (issue #317). Read off the one list every
        // relocating write is staged on, so a write added later cannot be left out
        // of the screen without failing to compile.
        let mut moved_headers: Vec<u64> =
            moving_writes.iter().map(|&(_, _, addr, _)| addr).collect();

        // Fast path: when the only staged edits are same-length in-place
        // overwrites, apply them straight to their data blocks and return without
        // rebuilding any header or flipping the superblock root. The commit's
        // linearization point is the synced data write — there is no tree to
        // repoint, so each overwrite stands alone. (A persisting file takes the
        // same path: no free-space change occurs.)
        //
        // Because this path never rewrites the superblock, it deliberately leaves
        // it untouched — including a pre-existing stale consistency flag (e.g. one
        // left by a crashed SWMR writer). A lone same-length value overwrite does
        // not introduce any inconsistency, so it does not clear one either; an edit
        // that takes the full path below (any header/root change) clears the flag
        // as usual.
        //
        // A staged variable-length overwrite cannot reach here at all: it
        // relocates, so `moving_writes` is non-empty for it. That is enforced
        // where the plan is chosen (`prepare_write`) rather than re-tested here,
        // and it has to hold for a second reason besides the one that made it
        // relocate — such an overwrite places a global heap collection, and an
        // append moves end-of-file, which only the superblock this path leaves
        // untouched records (issue #321).
        if moving_writes.is_empty() && staged.scope() == StagedScope::Datasets {
            // This path's own point of no return, and it takes the staged set
            // for the same reason the main one below does (issue #316): every
            // refusal that applies to these overwrites has already run, and a
            // batch some of which has been written is not one a later `commit`
            // may re-issue. It is also what this path already did, back when the
            // write preflight drained the staged set on its way here.
            //
            // Leaving the set in place would make a half-written batch
            // retryable — these overwrites are same-length writes to fixed
            // addresses, so repeating them is idempotent — but that reads the
            // rule the other way round for one path, on an argument nothing
            // enforces. What the file holds is settled separately and the same
            // way on both paths: a `write_at` that fails partway is undone
            // through the journal below, so the batch is dropped but the
            // dataset's values are the ones it had (issue #344).
            drop(std::mem::take(staged));
            for (data_addr, bytes) in &inplace_writes {
                self.write_inplace_journaled(*data_addr, bytes)?;
            }
            self.barrier()?;
            return Ok(());
        }

        // --- Plan: build the tree of "dirty" groups (root plus every group on a
        // path to an addition or deletion), validating every target before any
        // write. `add_targets` records the full paths created this commit, used
        // to reject a deletion that overlaps an addition. ---
        let mut nodes: BTreeMap<ObjectPathBuf, Node> = BTreeMap::new();
        nodes.entry(ObjectPathBuf::root()).or_default(); // root is always dirty
        let mut add_targets: Vec<ObjectPathBuf> = Vec::new();
        // Where each in-file `copy` reads from. A copy takes its bytes from the
        // *pre-commit* file, so a source this same commit replaces would copy the
        // object being removed while the replacement lands at the same path — see
        // the delete-staging loop, which refuses that. Cross-file copies are not
        // tracked: their source is in another file, so no path here can name it.
        let mut copy_sources: Vec<ObjectPathBuf> = Vec::new();
        let mut attr_targets: Vec<ObjectPathBuf> = Vec::new();

        // Mark explicitly-created new groups, ensuring their ancestor chain.
        for path in staged.groups() {
            if path.is_empty() {
                return Err(Error::EditUnsupported("cannot create the root group"));
            }
            ensure_ancestors(&mut nodes, path);
            nodes.entry(path.clone()).or_default().is_new = true;
            add_targets.push(path.clone());
        }

        // Make each staged dataset's parent group a node (with its ancestor
        // chain). The datasets themselves stay in the staged set, which the
        // preflight reads but must not empty (issue #316); `datasets_by_group`
        // below is the grouped view both the preflight and the apply loop use.
        for (parent, fd) in staged.datasets() {
            add_targets.push(parent.join(&fd.name));
            ensure_ancestors(&mut nodes, parent);
        }

        // Attach relocating value overwrites (resized contiguous or compact) to
        // their parent group nodes: the new header is written below and the
        // parent's existing link patched to it, like an existing child group.
        for (parent, leaf, old_oh, mw) in moving_writes {
            ensure_ancestors(&mut nodes, &parent);
            nodes
                .entry(parent)
                .or_default()
                .writes
                .push((leaf, old_oh, mw));
        }

        // Stage group attribute edits against their target groups. A target may
        // be a newly-created group from this same commit, but not a copied
        // destination or a dataset being added in the same commit. The ops
        // themselves stay in the staged set and are looked up by path where they
        // are applied, so a refusal below still gives them back (issue #316).
        for (path, _) in staged.group_attrs() {
            ensure_ancestors(&mut nodes, path);
            attr_targets.push(path.clone());
        }

        // Stage copies: validate the source subtree is copyable (read-only),
        // then treat the destination like an addition to its parent group.
        for (src, dst) in staged.copies() {
            if src.is_empty() {
                return Err(Error::EditUnsupported("cannot copy the root group"));
            }
            let Some((parent, leaf)) = dst.split_leaf() else {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            };
            if dst.starts_with(src) {
                return Err(Error::EditUnsupported(
                    "cannot copy an object into itself or its own subtree",
                ));
            }
            let src_addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &src.as_path(),
            )?;
            // Read the source subtree from this file's own mirror (`cross_file`
            // false: same address space, so verbatim addresses stay valid). On a
            // userblock file the stored addresses are base-relative, so pass this
            // session's base for the read to absolutize them.
            let tree = Self::read_copy_subtree(
                &self.image(),
                AccessMode::ReadWrite,
                src_addr,
                0,
                false,
                base,
                self.widths,
            )?;
            copy_sources.push(src.clone());
            add_targets.push(dst.clone());
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().copies.push((leaf, tree));
        }

        // Stage cross-file copies: their subtrees were already read out of the
        // source file (with foreign-address screening) when `copy_from` was
        // called, so here they are simply linked into the destination parent like
        // any other addition.
        for (dst, _) in staged.cross_copies() {
            let Some((parent, leaf)) = dst.split_leaf() else {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            };
            add_targets.push(dst.clone());
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().cross_copies.push(leaf);
        }

        // Stage deletions: each must exist, must not overlap any other staged
        // change *unless* this commit replaces what it removes, and is recorded
        // against its parent group (which becomes dirty). `deleted_addrs` keeps
        // each removed object's header address so its owned blocks can be
        // reclaimed after the commit lands (issue #21).
        let delete_targets = staged.deletes();
        let mut deleted_addrs: Vec<u64> = Vec::new();
        for (i, d) in delete_targets.iter().enumerate() {
            let Some((parent, leaf)) = d.split_leaf() else {
                return Err(Error::EditUnsupported("cannot delete the root group"));
            };
            let del_addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &d.as_path(),
            )?;
            deleted_addrs.push(del_addr);
            // A deletion may overlap other staged work when this commit
            // *replaces* what it removes: an addition names exactly `d`, so the
            // removal and the new object at the same path are one rotation
            // rather than an edit of something being deleted (issue #305).
            let recreated = add_targets.iter().any(|t| t == d);
            // A replacement also requires that every group node at or below `d`
            // is one this commit builds fresh. A node that is *not* new is
            // rebuilt from the old object's on-disk header — the object being
            // replaced — and both ways that lands are wrong, in the two shapes
            // this guard was measured against:
            //
            // * At `d` itself (a group attribute set on a path this commit
            //   replaces with a *dataset*), the node and the replacement share a
            //   `path_addr` key. The replacement's address overwrites the group's,
            //   the parent's link is then patched to the address it already had,
            //   and the rebuilt group header is left orphaned — a commit that
            //   returns `Ok` having **silently discarded** the staged attribute.
            // * Strictly below `d`, the parent is a freshly built region with no
            //   existing link to patch, so `patch_link_target` reports a missing
            //   child link partway through the apply. That leaves a valid file
            //   (the superblock is never repointed) but appends dead bytes and
            //   reports the wrong thing.
            //
            // Refusing here keeps both in the preflight, where the commit is
            // all-or-nothing and the message can name the actual conflict.
            if recreated
                && !nodes
                    .iter()
                    .all(|(key, node)| !key.starts_with(d) || node.is_new)
            {
                return Err(Error::EditUnsupported(
                    "a staged edit names a group at or under a replaced path that this \
                     commit does not itself create; create it in the same commit, or use \
                     separate commits",
                ));
            }
            // A copy reading from a path this commit replaces is the same conflict
            // seen from the other side: `read_copy_subtree` already took its bytes
            // from the pre-commit file, so the commit would place the *original*
            // at the copy's destination while placing something else at `d` — two
            // different objects from one path, in one commit, with no error. A
            // source that is merely deleted and not replaced is unambiguous (it is
            // a move) and stays allowed.
            if recreated {
                for t in &copy_sources {
                    if t.starts_with(d) {
                        return Err(Error::EditUnsupported(
                            "a copy in this commit reads from a path the same commit \
                             replaces; use separate commits",
                        ));
                    }
                }
            }
            // Past that return `recreated` means the whole touched subtree is
            // fresh, so an addition at or below `d` lands in the replacement. The
            // apply loop already removes a group's deleted links before appending
            // any new one (`remove_link_from_region` runs first), so a
            // replacement needs no further sequencing here.
            for t in &add_targets {
                if recreated && t.starts_with(d) {
                    continue;
                }
                if t.starts_with(d) || d.starts_with(t) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps an addition in the same commit; \
                         replace the path instead, or use separate commits",
                    ));
                }
            }
            for t in &attr_targets {
                if recreated && t.starts_with(d) {
                    continue;
                }
                if t.starts_with(d) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a group-attribute edit in the same commit; use separate commits",
                    ));
                }
            }
            for t in &write_targets {
                if t.starts_with(d) {
                    // `write_targets` holds three kinds — a value overwrite, a
                    // dataset-attribute edit, and a staged append — so the
                    // message names what they have in common rather than only
                    // the first of them.
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a staged edit to a dataset in the same \
                         commit; use separate commits",
                    ));
                }
            }
            for (j, d2) in delete_targets.iter().enumerate() {
                if i != j && d2.starts_with(d) {
                    return Err(Error::EditUnsupported(
                        "overlapping deletions in one commit; delete the common parent only",
                    ));
                }
            }
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().deletes.push(leaf);
        }

        // Resolve / validate each node's base object-header region up front.
        // Every existing dirty group is rewritten to a freshly-appended header,
        // so its old header becomes dead bytes once the superblock is repointed;
        // `superseded_addrs` records those old headers for reclamation (#21),
        // paired with the path whose rebuilt header replaces each — the two
        // halves of one relocation, which is what an object reference stored
        // elsewhere in the file has to be repointed across (issue #324).
        let keys: Vec<ObjectPathBuf> = nodes.keys().cloned().collect();
        let mut superseded_addrs: Vec<(ObjectPathBuf, u64)> = Vec::new();
        for key in &keys {
            let is_new = nodes[key].is_new;
            if is_new {
                nodes.get_mut(key).unwrap().base_region = group::fresh_group_region();
            } else {
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    AccessMode::ReadWrite,
                    &self.superblock,
                    self.widths,
                    self.addresses,
                    &key.as_path(),
                )?;
                // Rebuilding this group moves its header and patches only the
                // link this commit resolved it through, so every other hard link
                // to it would be left naming the old header — which this commit
                // then frees. The aliases show the pre-commit group until
                // something reuses the span, and are unreadable after (issue
                // #327). Same rule, and the same lazily-computed count, as the
                // three relocating dataset writes above.
                //
                // The root group is exempt because it is named by the superblock
                // rather than by a link, so it has no entry in the count and
                // nothing to strand. Every other group was resolved by path and
                // therefore has at least one.
                if !key.is_empty() {
                    let counts = incoming_links
                        .get_or_insert_with(|| self.count_incoming_hard_links())
                        .as_ref();
                    match counts.and_then(|c| c.get(&addr).copied()) {
                        Some(1) => {}
                        // Known, and more than one: the aliases are the problem.
                        Some(_) => {
                            return Err(Error::EditUnsupported(
                                "editing a group relocates its object header; only supported \
                                 when it has a single hard link",
                            ));
                        }
                        // Not known: the file-wide walk that counts links gave up
                        // — an object header it could not parse, a group it could
                        // not enumerate, or a link graph past its bound. A
                        // different refusal because it is a different problem, and
                        // "it has more than one hard link" would send the reader
                        // looking for a second link that may not exist.
                        None => {
                            return Err(Error::EditUnsupported(
                                "editing a group relocates its object header, and this file's \
                                 links could not be walked to establish that nothing else \
                                 names it",
                            ));
                        }
                    }
                }
                let info = self.inspect_group(addr)?;
                superseded_addrs.push((key.clone(), addr));
                let node = nodes.get_mut(key).unwrap();
                node.base_region = info.region;
                node.existing_links = info.link_names;
            }
        }

        // A rebuilt group's old header is vacated just as a relocated dataset's is,
        // so the two join one list for the screen below.
        moved_headers.extend(superseded_addrs.iter().map(|&(_, a)| a));

        // Apply and validate group attribute edits before any writes. This keeps
        // unsupported attribute edits under the same all-or-nothing preflight
        // contract as unsupported dataset additions. A variable-length attribute
        // is not fully resolved here — its global heap collection is built (it
        // is self-contained, no address needed yet) but placed and patched into
        // `base_region` only in the apply loop below, once its address is known.
        let attrs_by_group = group_by_parent(staged.group_attrs().iter().map(|(p, op)| (p, op)));
        // The pre-commit header address of each group this commit rewrites. An
        // edit that sends a group's attributes to a fractal heap reads the set it
        // rebuilds back out of the file, which needs the address; a group this
        // commit *creates* has neither an address nor any stored attribute, and
        // its `None` says so.
        let existing_group_addrs: HashMap<ObjectPathBuf, u64> =
            superseded_addrs.iter().cloned().collect();
        for key in &keys {
            if let Some(ops) = attrs_by_group.get(key) {
                let region = std::mem::take(&mut nodes.get_mut(key).unwrap().base_region);
                let edits = attributes::plan_attr_ops(
                    &self.image(),
                    base,
                    existing_group_addrs.get(key).copied(),
                    &region,
                    ops,
                )?;
                let node = nodes.get_mut(key).unwrap();
                node.base_region = edits.region;
                node.attrs = edits.attrs;
                node.replaces_dense_storage = edits.replaces_dense_storage;
            }
        }

        // Map each node to its direct child group nodes, each with the link name
        // it has in that parent (for link wiring). The root group has no name in
        // a parent, so `split_leaf` gives `None` for it and the loop skips it.
        let mut children: BTreeMap<ObjectPathBuf, Vec<(ObjectPathBuf, LinkNameBuf)>> =
            BTreeMap::new();
        for key in &keys {
            if let Some((parent, leaf)) = key.split_leaf() {
                children
                    .entry(parent)
                    .or_default()
                    .push((key.clone(), leaf));
            }
        }

        // Each group's added datasets, in the order the apply loop will place
        // them, for the guards below to read. Borrowed from the staged set:
        // every one of those guards may still refuse, and a refusal gives the
        // caller back every edit it was holding (issue #316). The apply loop
        // rebuilds the same grouping, by the same rule, once it owns the set.
        let datasets_by_group = group_by_parent(staged.datasets().iter().map(|(p, fd)| (p, fd)));

        // A group that tracks *link* creation order takes an addition — the
        // apply loop numbers each added link from the running maximum its Link
        // Info message records — up to the point the addition would send its
        // links dense, which is refused before anything is written. The count
        // this screens is the one the group would hold afterwards: the links it
        // has now, less the ones this commit deletes, plus the ones it adds.
        for key in &keys {
            let node = &nodes[key];
            let added = node.copies.len()
                + node.cross_copies.len()
                + datasets_by_group.get(key).map_or(0, Vec::len)
                + children.get(key).map_or(0, |kids| {
                    kids.iter().filter(|(child, _)| nodes[child].is_new).count()
                });
            if added > 0 {
                let kept = node
                    .existing_links
                    .iter()
                    .filter(|name| !node.deletes.iter().any(|d| d.as_str() == *name))
                    .count();
                group::reject_dense_link_creation_order(&node.base_region, kept + added)?;
            }
        }

        // Validate names: no addition may collide with an existing link or with
        // another addition under the same parent. A link this same commit
        // deletes is not one of the existing ones — the apply loop removes it
        // from the region before any addition is appended, so replacing an
        // object at its own path is a rotation, not a collision (issue #305).
        for key in &keys {
            let node = &nodes[key];
            let mut adding: Vec<&str> = Vec::new();
            for fd in datasets_by_group.get(key).into_iter().flatten() {
                adding.push(fd.name.as_str());
            }
            for (child, leaf) in children.get(key).into_iter().flatten() {
                if nodes[child].is_new {
                    adding.push(leaf.as_str());
                }
            }
            for (leaf, _) in &node.copies {
                adding.push(leaf.as_str());
            }
            for leaf in &node.cross_copies {
                adding.push(leaf.as_str());
            }
            for (i, name) in adding.iter().enumerate() {
                let survives = node.existing_links.iter().any(|n| n == name)
                    && !node.deletes.iter().any(|n| n.as_str() == *name);
                if survives || adding[..i].contains(name) {
                    return Err(Error::EditUnsupported(
                        "a link with this name already exists in the target group",
                    ));
                }
            }
        }

        // Content the caller's library-version bound cannot carry is refused
        // here, before any write, so a rejected addition leaves the commit
        // unapplied.
        self.check_libver_admits(staged.datasets().iter().map(|(_, fd)| fd))?;

        // Enumerate what this commit's deletions reclaim, from the current
        // on-disk layout and before any byte moves. It is read here, ahead of
        // every remaining refusal, because two of them screen against it: an
        // object reference this commit writes must not name space the same
        // commit frees (issue #317). The spans are carried to `to_free` below
        // rather than walked a second time, so the screen and the allocator
        // always mean the same thing by "removed".
        //
        // An object's storage is reclaimed only when the link being removed is
        // its LAST hard link: HDF5 objects can have several hard links, and one
        // reachable through a surviving link is still live (freeing it would
        // corrupt the survivor). Count every hard link in the pre-commit file
        // and reclaim a deleted object only when its count is exactly 1.
        // `deleted_addrs` is de-duplicated first so two delete paths that are
        // hard links to the same object are not visited (and freed) twice. If
        // the link graph cannot be walked in full, no deleted object is
        // reclaimed (a safe leak) — and none is screened either, which is sound
        // in the same direction: nothing is freed, so no reference dangles.
        // Superseded group headers and relocated dataset headers are vacated too,
        // and are screened separately: `moved_headers` holds them, and they are
        // addresses rather than spans.
        let mut deleted_free: Vec<(u64, u64, FreeClass)> = Vec::new();
        deleted_addrs.sort_unstable();
        deleted_addrs.dedup();
        if !deleted_addrs.is_empty() {
            if let Some(incoming) = self.count_incoming_hard_links() {
                for &a in &deleted_addrs {
                    self.collect_free_spans(a, 0, &incoming, &mut deleted_free);
                }
            }
        }
        let deleted_collections = self.deleted_collections(&deleted_addrs);
        // An in-file copy re-emits its source's element bytes verbatim, so a
        // copied object reference keeps naming whatever it named in the source —
        // including an object this same commit is removing (issue #317). The page
        // types the walk records are for the allocator, not for this.
        //
        // Removals only, and this is the one place the two halves part company. A
        // *supplied* address is a fresh claim about the file, and a commit that
        // moves the object falsifies it. A *copied* one repeats a claim the file
        // already made, and since issue #324 a move no longer falsifies either
        // one: the commit's last act walks the tree it published and repoints
        // every reachable stored address, the copy's among them
        // ([`crate::reference_patch`], and `a_copy_made_in_the_same_commit_that_
        // moves_its_target_is_repointed`). A removal is the case that stays,
        // because there is no new address to point at.
        //
        // Refusing on `moved` here would also cost what it cannot buy: every
        // commit that reaches here rebuilds its root group, so `moved` is never
        // empty, and a chunked object-reference dataset is refused on a non-empty
        // screen alone rather than on a matching address — so every such copy
        // would be refused, in every commit.
        let for_copied = InvalidatedAddresses {
            removed: deleted_free
                .iter()
                .map(|&(off, len, _)| (off, len))
                .collect(),
            moved: Vec::new(),
            base: self.addresses,
        };
        {
            // Framed at the base address: a shared-message address is stored
            // relative to it, and `SourceResolver` reads its references as
            // absolute within the view it is given.
            let image = self.image();
            let framed = BaseOffsetSource {
                inner: image,
                base: self.addresses.effective_base(),
            };
            for key in &keys {
                for (_, tree) in &nodes[key].copies {
                    copy::screen_copied_references(
                        tree,
                        self.addresses,
                        self.widths,
                        &for_copied,
                        &framed,
                    )?;
                }
            }
        }

        // The same screen the copies just got, plus the half a supplied address
        // earns and a copied one does not.
        let for_supplied = InvalidatedAddresses {
            moved: moved_headers,
            ..for_copied
        };

        // A shared-message index record can hold an object-header address too,
        // and it is the one stored address the commit's closing repoint walk
        // cannot reach. Screen it here, against the same two lists.
        self.screen_shared_message_index(&for_supplied)?;

        // References a builder already resolved to addresses never reach
        // `resolve_reference_target`, so they are screened out of the element
        // bytes instead — additions and value overwrites alike, since neither
        // form's bytes are touched again before they are written. A slot still
        // holding a placeholder screens as the zero it is, and is screened for
        // real by `resolve_reference_target` when it resolves.
        //
        // Cross-file copies need no screen for the opposite reason:
        // `reject_foreign_addresses` refuses a reference datatype outright on
        // that path.
        for (_, fd) in staged.datasets().iter().chain(staged.writes()) {
            screen_resolved_references(&fd.dt, &fd.raw, &for_supplied)?;
        }

        // Anything this commit adds could be a reference container the last walk
        // did not see, so it retires that walk's "no references in this file"
        // finding (issue #324). See `proved_free_of_references` for which of
        // these four clauses is load-bearing and which are belt-and-braces, and
        // for why the other five staged collections need no clause at all.
        //
        // A staged dataset's *attributes* are screened beside its element
        // datatype. Neither can carry a reference today — a committed one is
        // refused by `flatten_dataset`, and `AttrValue` has no reference variant
        // — but that is a refusal in another function rather than a property of
        // this collection, and the check that means what it says is the one that
        // looks.
        if staged
            .datasets()
            .iter()
            .chain(staged.writes())
            .any(|(_, fd)| {
                datatype::datatype_holds_object_address(&fd.dt)
                    || fd
                        .attrs
                        .iter()
                        .any(|a| datatype::datatype_holds_object_address(&a.datatype))
            })
            || !staged.copies().is_empty()
            || !staged.cross_copies().is_empty()
        {
            self.proved_free_of_references = false;
        }

        // Prove every object-reference target resolves before any write (see
        // `preflight_reference_targets`'s doc comment): otherwise a reference
        // resolution failure discovered mid-apply-loop would leave every
        // earlier-processed group's real writes (headers, data, copied
        // subtrees) orphaned in the file despite `commit()` returning `Err`.
        Self::preflight_reference_targets(
            &keys,
            &datasets_by_group,
            &nodes,
            &add_targets,
            &write_targets,
            delete_targets,
            &for_supplied,
            &self.image(),
            &self.superblock,
            self.widths,
            self.addresses,
        )?;

        if self.fresh_page_boundary.is_some() {
            self.preflight_fresh_pages(staged, &nodes)?;
        }

        // --- The point of no return. Every refusal is behind us, so the staged
        // set is taken here rather than at the top of this function: a refusal
        // above restores *every* edit the caller staged, and a failure below —
        // an I/O error mid-apply, which leaves the file valid because the
        // superblock is repointed last — restores none of them, so no later
        // `commit` can finish a batch this one abandoned (issue #316). Nothing
        // above this line may consume from `staged`. ---
        let mut taken = std::mem::take(staged).into_apply();
        // The same paths as the borrow above, owned now that the set has moved.
        let delete_targets = std::mem::take(&mut taken.deletes);
        // The same grouping as `datasets_by_group` above, by the same rule and
        // so in the same order — which is what keeps the apply loop's placement
        // order the one `preflight_reference_targets` proved.
        let mut flat = group_by_parent(taken.datasets.drain(..));
        // A cross-file copy's subtree can move now, so it joins its destination
        // group's in-file copies. The parent node exists: the preflight made one
        // for every destination.
        for (dst, tree) in taken.cross_copies.drain(..) {
            // A destination that identifies no link is rejected above.
            let (parent, leaf) = dst.split_leaf().expect("a copy destination names a link");
            nodes.get_mut(&parent).unwrap().copies.push((leaf, tree));
        }

        // Gather the regions this commit will vacate, read from the current
        // on-disk layout before any byte moves: every deleted object's owned
        // blocks plus every superseded group header. They stay out of the free
        // list until every append this commit makes is behind it, so none of
        // them is reused while it is still live. (On the non-persisting tail
        // that point is a few lines *before* the superblock write rather than
        // after it — the apply loop is where the appends happen, and it is long
        // done by then.) Enumeration is
        // best-effort — `collect_free_spans` simply omits anything it cannot
        // account for exhaustively, so the worst case is unreclaimed dead bytes,
        // never a freed-but-live region.
        // It starts as the deleted objects' blocks, enumerated above the preflight
        // because a refusal there screens against them.
        let mut to_free: Vec<(u64, u64, FreeClass)> = deleted_free;

        // A superseded group header is dead once the root is repointed. Its chunk
        // spans are enumerated base-aware (`oh_chunk_spans` shifts continuation
        // addresses by the userblock base and returns absolute file offsets), as is
        // the delete path (`collect_free_spans`), so all of this reclamation works
        // on userblock files too.
        for (key, a) in &superseded_addrs {
            let a = *a;
            if let Ok(spans) = self.oh_chunk_spans(a) {
                if nodes[key].replaces_dense_storage
                    && let Ok(region) = Self::gather_oh_messages(&self.image(), a, self.addresses)
                    && let Some(storage) = self.dense_attribute_storage_spans(&region, &spans)
                {
                    to_free.extend(storage);
                }
                to_free.extend(self.object_header_free_spans(&spans)?);
            }
        }

        // A relocating overwrite (`write_dataset` resize, or any compact rewrite)
        // vacates the dataset's old object header, and a resized contiguous one
        // also vacates its old data block: both become dead once the parent's
        // relinked header lands. `superseded_addrs` covers only the rebuilt group
        // headers, not the relocated dataset's own header, so record that here too.
        // The pre-commit dataset-header address rides on the write plan (`old_oh`),
        // recorded where the plan was made; its chunks and old data extent are
        // freed only after the superblock repoint.

        // The single-hard-link guard in the write preflight makes freeing the old
        // header safe (no surviving link still points at it).
        for key in &keys {
            for (_leaf, old_oh, mw) in &nodes[key].writes {
                match mw {
                    MovingWrite::Contiguous {
                        old_extent: Some(extent),
                        ..
                    } => to_free.push((extent.0, extent.1, FreeClass::Page(PageType::Raw))),
                    // A relocated chunked dataset vacates its old chunk index and
                    // chunk data blocks. `chunked_storage_spans` returns `None` for
                    // anything it cannot enumerate exhaustively (leaving dead bytes
                    // rather than freeing a region still in use); the old header
                    // chunks are freed generically below.
                    MovingWrite::Chunked { old_addr, .. } => {
                        if let Some(spans) = self.chunked_storage_spans(*old_addr) {
                            to_free.extend(spans);
                        }
                    }
                    // A relocating append keeps the existing chunk *data* in place
                    // (shared by both indexes during the commit), so only the old
                    // index structure and the relocated old trailing chunk are dead.
                    // The old header chunks are freed by the generic path below.
                    MovingWrite::AppendedChunks {
                        old_addr,
                        old_tail_extent,
                        kept_chunks,
                        ..
                    } => {
                        if let Some(spans) = self.chunked_index_spans(*old_addr) {
                            // The old index is reclaimed as raw only where it
                            // provably sits in a raw page, and recorded as dead
                            // otherwise. See `index_is_provably_raw`. The
                            // dataset's old chunk data is the kept chunks
                            // (base-relative) plus the trailing partial chunk
                            // this append relocated, both already in hand, so
                            // the proof needs no second walk of the index.
                            let data: Vec<(u64, u64)> = kept_chunks
                                .iter()
                                .filter_map(|c| {
                                    Some((base.absolute(c.address).ok()?, c.stored_size))
                                })
                                .chain(*old_tail_extent)
                                .collect();
                            let class = if self.index_is_provably_raw(&data, &spans) {
                                FreeClass::Page(PageType::Raw)
                            } else {
                                FreeClass::Dead
                            };
                            to_free.extend(spans.into_iter().map(|(a, l)| (a, l, class)));
                        }
                        if let Some(ext) = old_tail_extent {
                            // The relocated old trailing chunk is raw data.
                            to_free.push((ext.0, ext.1, FreeClass::Page(PageType::Raw)));
                        }
                    }
                    _ => {}
                }
                // The relocated dataset's old header chunks are dead too. An attribute edit of
                // an already-dense set also replaces the set's version 2 B-tree indexes and its
                // fractal heap.
                if let Ok(spans) = self.oh_chunk_spans(*old_oh) {
                    if matches!(
                        mw,
                        MovingWrite::AttrEdit {
                            replaces_dense_storage: true,
                            ..
                        }
                    ) && let Ok(region) =
                        Self::gather_oh_messages(&self.image(), *old_oh, self.addresses)
                        && let Some(storage) = self.dense_attribute_storage_spans(&region, &spans)
                    {
                        to_free.extend(storage);
                    }
                    to_free.extend(self.object_header_free_spans(&spans)?);
                }
            }
        }

        // Defense in depth: never hand the free list an out-of-bounds or
        // overlapping span. The last-link guard plus the per-object checks
        // should already make the accumulated spans disjoint; this enforces it
        // as a whole-commit invariant against the pre-commit end-of-file. Any
        // dropped span (which should not occur for a well-formed file) only
        // leaks, never corrupts.
        retain_disjoint_in_bounds(&mut to_free, self.image.len());

        // --- Apply: process deepest groups first so each parent sees its
        // children's new addresses, then repoint the superblock last.
        // `path_addr` accumulates every group's and dataset's address as it is
        // placed — read by `resolve_reference_target` to resolve a same-commit
        // object-reference target (see the dataset-placement loop below for the
        // group/dataset key convention: a group's own path, or a dataset's
        // full parent+name path). ---
        let mut path_addr: BTreeMap<ObjectPathBuf, u64> = BTreeMap::new();
        // Where each object-header address this commit vacates has been rewritten
        // to, for repointing the object references the rest of the file already
        // stores (issue #324). Filled from the two places a header moves: a
        // relocating value overwrite, recorded in the loop below as its plan is
        // applied, and a rebuilt group, joined from `superseded_addrs` once every
        // group's new address is known.
        let mut relocations: BTreeMap<u64, u64> = BTreeMap::new();
        let mut by_depth = keys.clone();
        // Stable on purpose: `keys` comes from a `BTreeMap`, so equal-depth
        // groups arrive in path order, and this order is the order they are
        // placed in — which fixes every address the commit writes. An unstable
        // sort may permute a depth group, and the file's layout with it.
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len())); // deepest first
        for key in &by_depth {
            let (mut region, deletes, copies, writes, attrs) = {
                let node = nodes.get_mut(key).unwrap();
                (
                    std::mem::take(&mut node.base_region),
                    std::mem::take(&mut node.deletes),
                    std::mem::take(&mut node.copies),
                    std::mem::take(&mut node.writes),
                    std::mem::take(&mut node.attrs),
                )
            };

            // Remove deleted links first (verbatim-preserving the rest).
            //
            // First is a correctness requirement rather than a convention, and
            // has been since a replacement became one commit (issue #305):
            // `remove_link_from_region` matches by *name*, so a removal running
            // after the additions below would take the replacement's link with
            // the original's and leave the path gone. Every link this loop could
            // collide with — a copy's, a dataset's, a new child group's — is
            // appended after it, which is what keeps that unreachable. (A
            // relocating write and an existing child group *patch* a link rather
            // than appending one, so neither can be a replacement's.)
            for name in &deletes {
                region = group::remove_link_from_region(&region, name.as_str())?;
            }

            // Every link appended below takes the next creation index from this
            // group's running maximum, on the groups that track link creation
            // order and on no others; `link_order.record` writes the bumped
            // maximum back once the last of them is placed. Read after the
            // deletions above because a deletion leaves the maximum alone — the
            // gap it opens is never handed out again.
            let mut link_order = LinkCreationOrder::for_region(&region)?;

            // Write each staged source subtree and link its root into this group.
            for (leaf, tree) in copies {
                let root = self.write_copy_subtree(&tree)?;
                region.push_link(leaf.as_str(), base.relative(root)?, link_order.take()?);
            }

            // Datasets directly under this group. Appended addresses are absolute
            // file offsets, and the contiguous data-layout address is stored
            // relative to the base address. Placed
            // non-reference datasets first (recording each into `path_addr`), then
            // reference datasets — a reference to a *non-reference* sibling added
            // in the same group's batch resolves regardless of staging
            // order (`Vec::sort_by_key` is stable, so within each of the two
            // groups the original order is preserved). Two reference datasets that
            // target each other in the same batch are still call-order-dependent —
            // whichever is placed first resolves the other, and the reverse
            // direction is safely refused as "still writing" (never corrupted),
            // caught up front by `preflight_reference_targets`.
            let mut group_datasets: Vec<FlatDataset> =
                flat.remove(key).into_iter().flatten().collect();
            // Stable on purpose: this is a partition, and the relative order
            // within each half is what the paragraph above promises.
            group_datasets.sort_by_key(|fd| fd.reference_targets.is_some());
            for mut fd in group_datasets {
                self.resolve_dataset_attributes(&mut fd)?;
                fd.resolve_references(|target| {
                    Self::resolve_reference_target(
                        target,
                        &path_addr,
                        &nodes,
                        &add_targets,
                        &write_targets,
                        &delete_targets,
                        &for_supplied,
                        &self.image(),
                        &self.superblock,
                        self.widths,
                        self.addresses,
                    )
                })?;
                let oh_addr = self.write_dataset(&mut fd)?;
                region.push_link(
                    fd.name.as_str(),
                    base.relative(oh_addr)?,
                    link_order.take()?,
                );
                path_addr.insert(key.join(&fd.name), oh_addr);
            }

            // Relocating value overwrites under this group: write the new data and
            // rewritten header, then patch this group's existing link to it. On a
            // userblock file only the chunked variant reaches here (contiguous and
            // compact resizes are refused in the write preflight).
            for (leaf, old_oh, mw) in &writes {
                let new_oh = self.write_moving(mw)?;
                group::patch_link_target(&mut region, leaf.as_str(), base.relative(new_oh)?)?;
                relocations.insert(*old_oh, new_oh);
            }

            // Wire links to dirty child groups (new → add a link; existing →
            // patch the existing link to the child's new address).
            for (child, leaf) in children.get(key).into_iter().flatten() {
                let child_name = leaf.as_str();
                let child_addr = base.relative(path_addr[child])?;
                if nodes[child].is_new {
                    region.push_link(child_name, child_addr, link_order.take()?);
                } else {
                    group::patch_link_target(&mut region, child_name, child_addr)?;
                }
            }

            // Every addition to this group is placed, so its Link Info message
            // can record the maximum creation index it has now assigned. A group
            // that gained no link, or that does not track the order, is left
            // byte-identical.
            link_order.record(&mut region)?;

            // Whatever this group's attribute edits left to place: a
            // variable-length attribute's heap collection, or a whole dense set.
            self.place_edited_attrs(&mut region, attrs)?;

            let oh = object_header::build_v2_object_header(&region)?;
            let addr = self.alloc_or_append_typed(&oh, AllocationClass::Metadata)?;
            path_addr.insert(key.clone(), addr);
        }

        // Prepared in-place bytes require no heap allocation. Journal existing data before writing
        // it.
        for (data_addr, bytes) in &inplace_writes {
            self.write_inplace_journaled(*data_addr, bytes)?;
        }

        // Every group's new header address is known now that the apply loop has
        // run, so the rebuilt groups can join the relocation map. A group whose
        // key is missing would be one the loop did not place, which cannot happen:
        // `superseded_addrs` is filled from `keys`, and the loop visits all of it.
        for (key, old) in &superseded_addrs {
            if let Some(&new) = path_addr.get(key) {
                relocations.insert(*old, new);
            }
        }

        // Repoint the superblock at the new root last: this is the commit's
        // linearization point. Until it lands, the file on disk still points at
        // the old root (the appended objects are merely unreferenced trailing
        // bytes), so a failure here leaves a valid file.
        //
        // That ordering is only crash-safe if the appended objects are durable
        // before the root pointer is flipped; otherwise a power loss could
        // persist the flip ahead of the data it references, leaving the root
        // pointing at bytes that never reached disk. `flush` on a plain `File`
        // does not force a write-back, so sync the appended bytes to disk first
        // (the barrier), then flip the pointer, then sync the flip.
        let new_root = path_addr[&ObjectPathBuf::root()];
        to_free.extend(self.unreferenced_collection_spans(new_root, deleted_collections));

        // Keep superseded collections unavailable until the new root is durable. These collections
        // were placed as raw allocations by this session, so their PAGE free-space type is known.
        to_free.extend(
            self.superseded_heaps
                .drain(..)
                .map(|(addr, len)| (addr, len, FreeClass::Page(PageType::Raw))),
        );
        // Re-run the whole-commit invariant over the set these just joined, since
        // the pass above ran before the in-place writes resolved and so never saw
        // them. This is defensive and nothing reaches it: no test fails when it
        // is deleted, and no sequence has been found that puts a superseded
        // collection out of bounds or across another freed span. It is kept
        // because a heap collection is freed on the strength of a *record*
        // rather than of a structure read back out of the file, which is the one
        // span where a stale answer would not be caught by anything else.
        retain_disjoint_in_bounds(&mut to_free, self.image.len());
        let mut to_free = self.stored_spans(to_free)?;

        // A persisting file keeps its freed space recorded on disk rather than
        // truncating it away, so its commit takes a different tail: one that
        // rewrites the on-disk managers rather than only repointing the superblock.
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

        let mut trunc_to = self.release_freed(to_free)?;
        if let Some(aggregation) = self.aggregation {
            let end = self
                .addresses
                .resolve(StoredAddress::new(aggregation.eoa()))?;
            if end.get() < self.image.len() {
                trunc_to = Some(end);
            }
        }
        self.publish_nonpersisting(new_root, trunc_to)?;
        self.repoint_stored_references(&relocations)
    }

    /// Proves that a non-persistent PAGE commit can publish without patching stored references.
    ///
    /// Checks the existing tree and prospective attribute storage before consuming staged edits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for object references, unsupported attribute storage,
    /// or an incomplete reference inspection. Returns address-resolution and object-header record
    /// errors from [`prove_free_of_references`](crate::reference_patch::prove_free_of_references).
    pub(super) fn preflight_fresh_pages(
        &self,
        staged: &StagedEdits,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
    ) -> Result<(), Error> {
        let unsupported_dataset =
            staged
                .datasets()
                .iter()
                .chain(staged.writes())
                .any(|(_, dataset)| {
                    dataset.attrs_are_dense
                        || datatype::datatype_holds_object_address(&dataset.dt)
                        || dataset
                            .attrs
                            .iter()
                            .any(|attr| datatype::datatype_holds_object_address(&attr.datatype))
                });
        // Same-file copies come from the tree checked below. Cross-file copies reject file
        // addresses when staged, so their datatypes cannot add object references.
        if unsupported_dataset
            || staged
                .cross_copies()
                .iter()
                .any(|(_, tree)| tree.has_dense_attributes())
            || nodes.values().any(|node| {
                matches!(node.attrs, EditedAttrs::Dense(_))
                    || node.writes.iter().any(|(_, _, write)| {
                        matches!(
                            write,
                            MovingWrite::AttrEdit {
                                attrs: EditedAttrs::Dense(_),
                                ..
                            }
                        )
                    })
            })
            || !crate::reference_patch::prove_free_of_references(
                &self.image(),
                &self.superblock,
                self.widths,
                self.addresses,
                MAX_LINK_GRAPH_NODES,
            )?
        {
            return Err(Error::EditUnsupported(
                "staged edits of PAGE files without persisted free space require a complete \
                 object-reference-free tree; repack the file to edit reference-bearing or \
                 unsupported attribute storage",
            ));
        }
        Ok(())
    }

    /// Publishes the root and EOA after the content barrier, then applies any requested truncation.
    ///
    /// A non-persistent PAGE commit retains its File Space Info message and passes no truncation.
    ///
    /// # Errors
    ///
    /// Returns the address-conversion, serialization, write, synchronization, or truncation error.
    pub(super) fn publish_nonpersisting(
        &mut self,
        new_root: u64,
        trunc_to: Option<PhysicalAddress>,
    ) -> Result<(), Error> {
        let base = self.addresses;
        let new_eof = trunc_to.map_or(self.image.len(), PhysicalAddress::get);
        self.barrier()?;
        if matches!(
            self.superblock.version,
            SuperblockVersion::Two | SuperblockVersion::Three
        ) {
            // Build the new superblock off a clone and adopt it only once the
            // write succeeds, so a failed write does not desync the in-memory
            // state. The v2/v3 superblock carries its own checksum.
            let mut new_sb = self.superblock.clone();
            new_sb.root_group_address = base.relative(new_root)?;
            new_sb.eof_address = new_eof;
            // Publish the flags this session *raised*, so a flag the source file
            // arrived carrying (left set by a crashed SWMR writer, say) is
            // scrubbed and this clean commit leaves the file properly closed for
            // the C library (issue #73). serialize() recomputes the v2/v3
            // checksum.
            //
            // Non-persistent PAGE sessions reject page buffering and use zero status flags.
            new_sb.consistency_flags = self.held_status_flags;
            let sb_bytes = hdf5_pure_format::__private::serialize_superblock(&new_sb)?;
            self.publish_attempted = true;
            self.write_at(self.sb_sig_off, &sb_bytes)?;
            self.barrier()?;
            self.superblock = new_sb;
        } else {
            self.publish_attempted = true;
            self.repoint_v0v1_root(base.relative(new_root)?, new_eof)?;
            self.barrier()?;
            self.superblock.root_group_address = self.addresses.relative(new_root)?;
            self.superblock.eof_address = new_eof;
        }

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable. A crash between the two leaves a file
        // whose superblock end-of-file is correct and whose trailing bytes are
        // mere unreferenced slack, which the next open ignores; the reverse order
        // could advertise an end-of-file past the actual file length.
        if let Some(cut) = trunc_to {
            self.image.truncate(cut.get())?;
            self.barrier()?;
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
    pub(super) fn release_freed(
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
                        .filter_map(|(addr, len, _)| FreeExtent::new(addr.get(), len)),
                )?
                .map(|cut| self.addresses.resolve(StoredAddress::new(cut)))
                .transpose();
        }
        self.space
            .release_freed(
                eoa.get(),
                to_free
                    .into_iter()
                    .filter_map(|(addr, len, _)| FreeExtent::new(addr.get(), len)),
            )
            .map(|cut| self.addresses.resolve(StoredAddress::new(cut)))
            .transpose()
    }

    /// Returns `spans` with each file position converted to a stored address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if a span starts below the base address.
    pub(super) fn stored_spans(
        &self,
        spans: Vec<(u64, u64, FreeClass)>,
    ) -> Result<Vec<(StoredAddress, u64, FreeClass)>, FormatError> {
        spans
            .into_iter()
            .map(|(at, len, class)| Ok((self.addresses.relative(at)?, len, class)))
            .collect()
    }

    /// Repoint every object reference the file already stores at a header this
    /// commit moved (issue #324).
    ///
    /// Called at the end of each commit tail, *after* the superblock repoint,
    /// which is where it belongs on both counts. The commit is atomic at the
    /// repoint and this is a fixup derived from it, so it must not run earlier:
    /// a crash before the repoint has to leave the pre-commit file, its stored
    /// references included. And the tree it walks has to be the committed one —
    /// a rebuilt group's header is a fresh copy of the pre-commit one and
    /// carries the old address of anything it referenced, so patching the
    /// superseded header instead would correct bytes that are already dead.
    ///
    /// The writes are followed by a [`barrier`](Self::barrier), so a commit that
    /// returns under [`SyncPolicy::Always`] has put its references on disk and
    /// not merely its tree. That rests on `barrier`'s own contract rather than
    /// on a crash test — nothing here sweeps the window — and it costs nothing
    /// on the files that do not need it: a plan with nothing in it returns
    /// first, and on a file holding no object reference there is never anything
    /// in it.
    ///
    /// A crash between the repoint and that barrier leaves the commit standing
    /// with some references still stale, which is exactly the state every commit
    /// left before this existed.
    ///
    /// Bounded by [`MAX_LINK_GRAPH_NODES`], the same budget
    /// [`count_incoming_hard_links`](Self::count_incoming_hard_links) walks the
    /// link graph under.
    pub(super) fn repoint_stored_references(
        &mut self,
        relocations: &BTreeMap<u64, u64>,
    ) -> Result<(), Error> {
        if self.proved_free_of_references {
            return Ok(());
        }
        let plan = crate::reference_patch::plan(
            &self.image(),
            &self.superblock,
            self.widths,
            self.addresses,
            relocations,
            MAX_LINK_GRAPH_NODES,
        )?;
        // Record the walk's verdict before applying it, and only ever the
        // positive proof: a walk that found a reference, or could not read some
        // object, leaves the question open rather than answering "yes there are
        // references", because the next commit's answer could differ.
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
    pub(super) fn commit_persisting(
        &mut self,
        new_root: u64,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
        placement: TailPlacement,
    ) -> Result<(), Error> {
        // A paged file records its free space in per-page-type managers and keeps
        // its allocation page-aligned, so it takes its own tail (issue #198).
        if self.space.is_paged() {
            return self.commit_persisting_paged(new_root, to_free, placement);
        }
        let widths = self.widths;
        let offset_width = widths.offsets;
        let (strategy, threshold, page_size, mut old_blocks) = {
            // Copy what we need so no borrow of `self.persist` is held across the
            // `&mut self` writes below; the old state stays in place so a failure
            // leaves the session reusable.
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
        let base = self.addresses;

        // The extension address is stored relative to the base address, and every
        // reader of the extension below indexes the image, so it is shifted once
        // here for all of them.
        let old_ext_addr = base.absolute(
            self.superblock
                .superblock_extension_address
                .filter(|a| !a.is_undefined(offset_width))
                .ok_or(Error::EditUnsupported(
                    "a persisting file has no superblock extension to update",
                ))?,
        )?;

        // The persist File Space Info message is fixed-size, so the rewritten
        // extension's length is independent of the addresses it will carry: size
        // it with a placeholder to place the FSM blocks that follow it.
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

        // Place the tail — the rewritten extension and the manager blocks — in a
        // hole an *earlier* commit freed where one fits, and at end-of-file where
        // none does. `self.space` holds only durable reusable space. This commit's own frees stay
        // in `to_free` until the repoint, so a reused span holds bytes
        // already unreachable from the on-disk root, which is the guarantee
        // [`reserve`](Self::reserve) rests on too.
        let current_eoa = self.end_of_allocation()?.get();
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
            self.flat_tail_layout(current_eoa, &to_free, &old_blocks, ext_len, widths, limit)
        });
        if placed_at.is_none() && placement == TailPlacement::ReuseOnly {
            // The shrink pass appends nothing: growing the file by a tail is the
            // opposite of what it was called for. Nothing has been written and the
            // layout handed its reservation back, so the session is exactly as the
            // commit before it left it.
            return Ok(());
        }
        let reused = placed_at.is_some();
        let ext_addr = placed_at.unwrap_or(current_eoa);
        let managers = ManagerSections::flat(post.extents().iter().copied());
        let plan = free_space_manager::plan_managers(
            &managers,
            StoredAddress::new(ext_addr).offset(ext_len),
            widths,
        );
        // For a reused tail, truncate to the planned EOA after publication. The post-commit
        // free list excludes dropped trailing regions and the suffix released by
        // `FreeList::release_trailing_run`. An appended tail ends the file itself.
        let final_eoa = if reused { eoa } else { ext_addr + tail_len };
        debug_assert!(
            ext_addr + tail_len <= final_eoa,
            "the tail must end at or below the end-of-allocation it was placed under"
        );
        let final_eof = self.addresses.resolve(StoredAddress::new(final_eoa))?;
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

        // Build the real extension and the FSM blocks. With no free space to
        // record we still refresh the extension (persist on, managers undefined).
        let ext_oh = if let Some(block) = flat_block {
            // `eoa_pre_fsm` is the end-of-allocation before the free-space-manager
            // section blocks (`FSHD`/`FSSE`) were allocated: a consumer may shrink
            // back to here and rebuild them. For an appended tail that is the FSHD,
            // not the extension — the extension sits below it and persists, so
            // shrinking leaves the superblock and its extension pointer valid (only
            // the manager blocks, which are rewritten every commit, are discarded).
            // A tail placed *inside* the file gave the allocation back nothing to
            // shrink, so it records the end-of-allocation itself. Either way it is
            // defined, which a persisting file's message must be or an
            // assertion-enabled libhdf5 aborts on open (issue #178), and is the
            // value `H5Fget_freespace` accounts for correctly (verified in the
            // crosscheck).
            let eoa_pre_fsm = if reused {
                final_eoa
            } else {
                block.fshd_addr.get()
            };
            let info = file_space_info::persistent_single_manager(
                offset_width,
                strategy,
                threshold,
                page_size,
                block.fshd_addr.get(),
                eoa_pre_fsm,
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
                final_eoa,
            );
            object_header::build_v2_object_header(
                &self.rewrite_extension_region(old_ext_addr, &info)?,
            )?
        };
        // Both forms of the message carry the same twelve manager slots, so either
        // one measures what the placeholder did. The tail was reserved for that
        // length, and the manager blocks are placed after it.
        debug_assert_eq!(
            ext_oh.len() as u64,
            ext_len,
            "extension length must be stable across the placeholder and real messages"
        );

        let new_old_blocks = vec![FreeExtent::new(ext_addr, tail_len).ok_or(Error::Format(
            FormatError::OffsetOverflow {
                offset: ext_addr,
                length: tail_len,
            },
        ))?];

        // Write the extension, then the FSM blocks. Both forms are safe against a
        // crash here: an appended tail is past everything live, a reused one sits
        // in space an earlier commit freed, and neither is referenced until the
        // repoint below.
        let region = (ext_addr, tail_len);
        self.write_tail_block(region, StoredAddress::new(ext_addr), &ext_oh)?;
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
        // Barrier, then repoint the superblock (root, eof, and the new extension)
        // — the linearization point — and sync it.
        self.barrier()?;
        let mut new_sb = self.superblock.clone();
        new_sb.root_group_address = self.addresses.relative(new_root)?;
        new_sb.eof_address = final_eof.get();
        new_sb.superblock_extension_address = Some(StoredAddress::new(ext_addr));
        // As above, and zero for the same reason: this is the *unpaged* persisting
        // tail, which no page-buffered session reaches (issue #73, issue #308).
        new_sb.consistency_flags = self.held_status_flags;
        let sb_bytes = hdf5_pure_format::__private::serialize_superblock(&new_sb)?;
        self.publish_attempted = true;
        let confirmed = self
            .aggregation
            .as_ref()
            .is_none_or(|state| state.publication_confirmed);
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = false;
        }
        self.write_at(self.sb_sig_off, &sb_bytes)?;
        self.barrier()?;
        self.superblock = new_sb;

        self.space.adopt_flat_post_free(post);
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

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable, exactly as the non-persisting tail
        // does: a crash between the two leaves trailing bytes the next open
        // ignores, where the reverse order would advertise an end-of-file past
        // the file's actual length. The managers just written describe `post`,
        // which no longer names anything above this cut.
        if final_eof < PhysicalAddress::new(self.image.len()) {
            self.image.truncate(final_eof.get())?;
            self.barrier()?;
        }

        // The repoint is durable: the prior free list plus this commit's vacated
        // regions are now genuinely free, and the freshly written blocks become
        // the ones a future commit will supersede.
        // The rewritten managers describe persistent free sections at this file length.
        self.fsm_len = self.image.len();
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = confirmed;
        }
        Ok(())
    }

    /// Returns the free space a flat persisting commit is about to record, and
    /// `eof` less the dropped regions that end it.
    ///
    /// The list is the session's durable list plus each of the regions this commit
    /// vacated and of the superseded extension and manager blocks that is at least
    /// `threshold` bytes long or adjoins a region of the list ([`FreeList::release_all`]).
    /// Those regions and blocks are dead once the superblock is repointed.
    ///
    /// Returned as a temporary rather than folded into the session, for the reason
    /// [`SessionSpace::paged_post_free`] does the same: every region
    /// gathered here is still *live* until that repoint, and everything in between
    /// can fail, so a session that believed them free after a failed commit would
    /// hand them out over the objects still occupying them.
    ///
    /// Called once to *size* the tail and again for each round its placement takes
    /// to settle, since what remains free depends on the space the tail has drawn
    /// for itself ([`flat_tail_layout`](Self::flat_tail_layout)).
    pub(super) fn flat_post_free(
        &self,
        eof: u64,
        to_free: &[(StoredAddress, u64, FreeClass)],
        old_blocks: &[FreeExtent],
    ) -> (FreeList, u64) {
        let extents = to_free
            .iter()
            .filter_map(|&(addr, len, _)| FreeExtent::new(addr.get(), len))
            .chain(old_blocks.iter().copied());
        self.space
            .flat_post_free(eof, extents)
            .expect("the flat commit tail requires a flat session strategy")
    }

    /// Reserves free space for a flat persisting commit's tail, and returns the free list the tail
    /// leaves behind, its address, its length, and the end-of-allocation the commit publishes.
    ///
    /// `current_eoa` and every address in the result are stored addresses. The address is `None`
    /// when nothing reusable fits, and the caller then appends the tail at `current_eoa`. The
    /// length is the tail's in either case. The end-of-allocation of a reused tail is `current_eoa`
    /// less the dropped regions that reach it and what [`FreeList::release_trailing_run`] releases
    /// of the free space below them, and that of an appended tail is `current_eoa`.
    ///
    /// Reserves a proposed length before measuring the managers for the remaining free sections.
    /// Accepts a plan that fits within the reservation and retries with a larger length otherwise.
    /// Spare capacity belongs to the manager allocation and is reclaimed when it is superseded.
    ///
    /// The writer records spare reservation bytes in the section list's allocated length and
    /// fills them with zeros. On reopen, the reader recovers the complete allocation for
    /// reclamation.
    ///
    /// Iteration is capped rather than trusted to converge: the length is a step
    /// function of the section set, so a proposal can in principle keep growing.
    /// Appending is always available and always correct, so giving up costs space
    /// rather than a guarantee — and the cap bounds the work, since each round
    /// clones the free list and sizes the section list for real.
    ///
    /// Draws from [`space`](Self::space) before writing. [`commit`](Self::commit) and
    /// [`finalize_persistent_aggregates`](Self::finalize_persistent_aggregates) restore their
    /// snapshots after a failure before publication.
    ///
    /// When [`space`](Self::space) has no hole for it, the tail may take one out of
    /// [`reserved`](Self::reserved) instead. Only the rewrite that publishes an
    /// append's draw ([`reserve_for_immediate_append`](Self::reserve_for_immediate_append))
    /// reaches this with anything there — every other caller releases the reserve
    /// first — and for that rewrite the fallback is what keeps the draw from
    /// growing the file: a draw takes every hole the append fits in, and on a
    /// flat file that is the one list the tail is placed from, so without it the
    /// rewrite appended its tail at end-of-file every time and the file grew by a
    /// tail per draw (issue #413). Space in the reserve is exactly what this
    /// rewrite publishes as *not* free, so a tail placed there is live in bytes
    /// the managers do not advertise, which is the same standing a tail placed
    /// from [`space`](Self::space) has once the repoint lands. Session space is tried first, so an
    /// ordinary hole is spent before the append's own space. Best fit within
    /// the reserve then takes its smallest run, leaving the largest for the
    /// append that drew it. The paged tail
    /// has no such fallback: it opens a page when nothing fits, and
    /// `paged_group_churn_with_populated_datasets_reaches_a_steady_size` holds it
    /// to not doing so once per draw.
    ///
    /// With a `limit`, restricts reservations to extents ending at or below it and uses only
    /// [`space`](Self::space). This permits relocating managers below the old tail.
    pub(super) fn flat_tail_layout(
        &mut self,
        current_eoa: u64,
        to_free: &[(StoredAddress, u64, FreeClass)],
        old_blocks: &[FreeExtent],
        ext_len: u64,
        widths: FormatWidths,
        limit: Option<u64>,
    ) -> (FreeList, Option<u64>, u64, u64) {
        /// Enough rounds for the section set to settle after a reservation shrinks
        /// it, without letting a proposal that keeps growing spin.
        const ROUNDS: usize = 4;

        // The first proposal: what the tail would measure with nothing reserved.
        // A manager block's length depends only on the sections it records, never
        // on where it sits. It is also the answer for a tail that ends up appended,
        // since every round hands its reservation back before this returns.
        let (probe, _) = self.flat_post_free(current_eoa, to_free, old_blocks);
        let managers = ManagerSections::flat(probe.extents().iter().copied());
        let appended_len = ext_len + free_space_manager::manager_blocks_len(&managers, widths);
        let mut proposed = appended_len;

        for _ in 0..ROUNDS {
            let allocation = if let Some(limit) = limit {
                self.space.allocate_flat_below(proposed, limit)
            } else {
                self.space.allocate(proposed, AllocationClass::Metadata)
            };
            let (reserved, from_reserve) = match allocation {
                Some(extent) => (extent, false),
                None if limit.is_none() => match self.reserved.alloc(proposed) {
                    Some(extent) => (extent, true),
                    None => break,
                },
                None => break,
            };
            let at = reserved.start();
            let (mut post, eoa) = self.flat_post_free(current_eoa, to_free, old_blocks);
            // Exclude the released suffix before sizing the managers. The tail reservation
            // has already been removed from this list, so the trailing run starts at or
            // above the reservation's end.
            let eoa = post.release_trailing_run(eoa, proposed);
            let managers = ManagerSections::flat(post.extents().iter().copied());
            let len = ext_len + free_space_manager::manager_blocks_len(&managers, widths);
            if len <= proposed {
                debug_assert!(
                    at + proposed <= eoa,
                    "a reused tail at {at} of {proposed} bytes runs past the end-of-allocation \
                     {eoa} it was placed under"
                );
                return (post, Some(at), proposed, eoa);
            }
            if from_reserve {
                self.reserved.free(reserved);
            } else {
                self.space
                    .return_allocation(reserved, AllocationClass::Metadata);
            }
            proposed = len;
        }
        // Nothing reusable fits, so the tail is appended past end-of-file and no
        // trailing run is released: whatever tracked free space reaches end-of-file
        // was too small for the tail and stays recorded below it, and a dropped
        // extent that ended the allocation stays below it untracked.
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
    pub(super) fn commit_persisting_paged(
        &mut self,
        new_root: u64,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
        placement: TailPlacement,
    ) -> Result<(), Error> {
        let widths = self.widths;
        let offset_width = widths.offsets;
        let (strategy, threshold, page_size, old_blocks) = {
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

        let base = self.addresses;

        // Page-align the file before anything else, so the rewritten extension and
        // the manager blocks begin on a page boundary and stay in metadata pages.
        // The padded tail becomes free space of whatever type that page held.
        self.pad_to_page()?;

        // As on the flat path: the extension address is stored relative to the base
        // address, and every reader of the extension below indexes the image.
        let old_ext_addr = base.absolute(
            self.superblock
                .superblock_extension_address
                .filter(|a| !a.is_undefined(offset_width))
                .ok_or(Error::EditUnsupported(
                    "a persisting file has no superblock extension to update",
                ))?,
        )?;

        // The 12-slot persist message is fixed-size, so a placeholder sizes the
        // rewritten extension before its manager addresses are known — and before
        // its address is, which is what lets the tail be sized before it is placed.
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

        let current_eoa = self.end_of_allocation()?.get();
        let freed: Vec<_> = to_free
            .iter()
            .filter_map(|&(addr, len, class)| {
                FreeExtent::new(addr.get(), len).map(|extent| (extent, class))
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
        let final_eof = self.addresses.resolve(StoredAddress::new(final_eoa))?;

        let info = if blocks.is_empty() {
            file_space_info::persistent_empty(
                offset_width,
                strategy,
                threshold,
                page_size,
                final_eoa,
            )
        } else {
            file_space_info::persistent_managers(
                strategy,
                threshold,
                page_size,
                slots.map(StoredAddress::get),
                final_eoa,
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

        // Exactly the bytes written, contiguous from the extension. A session that reopens this
        // file records the same extents from the message and manager headers, so nothing depends
        // on remembering this across a close.
        let new_old_blocks = vec![FreeExtent::new(ext_addr, blocks_len).ok_or(Error::Format(
            FormatError::OffsetOverflow {
                offset: ext_addr,
                length: blocks_len,
            },
        ))?];

        // Write the extension, then every manager block. Both forms are safe against
        // a crash here: an appended tail is past everything live, and a reused one
        // sits in space an earlier commit freed. Neither is referenced until the
        // repoint below.
        let region = (ext_addr, blocks_len);
        self.write_tail_block(region, StoredAddress::new(ext_addr), &ext_oh)?;
        for block in &blocks {
            self.write_tail_block(region, block.header_address, &block.header)?;
            self.write_tail_block(region, block.section_list_address, &block.section_list)?;
        }
        // An appended tail ends mid-page; pad it out, so the end-of-allocation stays
        // a whole number of pages. A reused tail is already inside the file, and
        // matched its reservation exactly, so there is nothing to pad — and where
        // it released a trailing run its end-of-allocation is *below* the current
        // end-of-file, which the truncation after the repoint gives back.
        if !reused {
            self.pad_zeros_to(final_eof.get())?;
        }
        // Barrier, then repoint the superblock (root, eof, and the new extension)
        // — the linearization point — and sync it.
        self.barrier()?;
        let mut new_sb = self.superblock.clone();
        new_sb.root_group_address = self.addresses.relative(new_root)?;
        new_sb.eof_address = final_eof.get();
        new_sb.superblock_extension_address = Some(StoredAddress::new(ext_addr));
        // The one publish site a page-buffered session actually reaches, and so
        // the one where this must not be a literal zero: a crash mark has to
        // outlive every commit the session makes (issue #308).
        new_sb.consistency_flags = self.held_status_flags;
        let sb_bytes = hdf5_pure_format::__private::serialize_superblock(&new_sb)?;
        self.publish_attempted = true;
        self.write_at(self.sb_sig_off, &sb_bytes)?;
        self.barrier()?;
        self.superblock = new_sb;

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable, for the reason the flat tail gives:
        // a crash between the two leaves trailing bytes the next open ignores,
        // where the reverse order would advertise an end-of-file past the file's
        // actual length.
        if final_eof < PhysicalAddress::new(self.image.len()) {
            self.image.truncate(final_eof.get())?;
            self.barrier()?;
        }

        // The repoint is durable. Only now are this commit's vacated regions
        // genuinely free, so adopt the lists built above and drop the padding tails
        // they already account for. The blocks just written become the ones the next
        // commit supersedes.
        self.space.adopt_paged_post_free(tail.post, !reused, None);
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

    /// Repoint a version 0/1 superblock at the rebuilt (now v2) root group and
    /// update its end-of-file field, patching the raw bytes in place — these
    /// superblocks carry no checksum. The root symbol-table entry is switched to
    /// cache type 0 (its scratch-pad B-tree / local-heap addresses, which
    /// describe the old symbol-table group, no longer apply). The
    /// object-header-address write is done last so it is the linearization point.
    pub(super) fn repoint_v0v1_root(
        &mut self,
        new_root: StoredAddress,
        new_eof: u64,
    ) -> Result<(), Error> {
        let os = self.superblock.offset_size.get();
        let addr_width = u64::from(os);
        // Field layout after the fixed prefix: base / free-space / EOF / driver
        // addresses, then the root symbol-table entry (link-name offset, object
        // header address, cache type(4), reserved(4), scratch(16)). The prefix is
        // 24 bytes for v0 and 28 for v1 (the latter adds indexed-storage-K).
        let var_start = if self.superblock.version == SuperblockVersion::Zero {
            24
        } else {
            28
        };
        let base = self.sb_sig_off + var_start;
        let eof_off = base + 2 * addr_width;
        let ste = base + 4 * addr_width;
        let oh_addr_off = ste + addr_width;
        let cache_off = ste + 2 * addr_width;
        self.write_at(eof_off, &new_eof.to_le_bytes()[..usize::from(os)])?;
        self.write_at(cache_off, &[0u8; 4])?; // cache type = none
        self.write_at(cache_off + 8, &[0u8; 16])?; // clear scratch-pad
        self.write_at(
            oh_addr_off,
            &new_root.get().to_le_bytes()[..usize::from(os)],
        )?;
        Ok(())
    }

    /// Resolves one object-reference element's target to the address stored on
    /// disk. [`ObjectRefTarget::Raw`] is written back verbatim: a null or an
    /// undefined reference points at nothing, and the whole-file writer treats
    /// it the same way. [`ObjectRefTarget::Path`] resolves, in order:
    ///
    /// 1. Against `path_addr` — every group and dataset this commit has
    ///    already placed (a sibling dataset placed earlier in the same
    ///    group's batch — see the apply loop's non-reference-first ordering —
    ///    or a descendant subtree fully processed earlier in the deepest-first
    ///    walk).
    /// 2. Against the pre-commit on-disk file
    ///    ([`resolve_path_any`](crate::group_v2::resolve_path_any)), but only
    ///    when the path is untouched by this commit, so its pre-commit
    ///    address is guaranteed to still be valid post-commit. "Touched"
    ///    means: a dirty group (`nodes`, new or merely rewritten because an
    ///    addition lives under it — its own address changes either way); a
    ///    path this commit adds, or that lies under a subtree this commit
    ///    copies in (`add_targets`, checked by prefix so a copy's interior is
    ///    covered even though only its root is enumerated there); or a
    ///    `write_dataset` target (`write_targets`) — conservatively refused
    ///    even for a same-length overwrite that does not actually relocate,
    ///    since resolving that distinction here is not worth the complexity.
    /// 3. If the path resolves nowhere at all (neither this commit nor the
    ///    pre-commit file has ever heard of it), as an undefined reference
    ///    (`HADDR_UNDEF`) — mirroring [`ObjectRefTarget::Path`]'s existing
    ///    whole-file-writer resolution convention for the same builder type.
    ///
    /// A path that step 1 misses but step 2 identifies as commit-touched is
    /// refused with a clear [`Error::EditUnsupported`] rather than resolved to
    /// a stale or wrong address — the one case this engine cannot resolve
    /// without the whole-file writer's two-pass dummy/real-address scheme.
    ///
    /// A path this same commit *deletes* (`delete_targets`) is refused for a
    /// different reason, and so carries its own message: the address step 2
    /// would resolve is not stale but doomed, and the next commit to reuse the
    /// span leaves a reference that dereferenced cleanly reading whatever
    /// landed there. The test is by **prefix**, because a deletion takes the
    /// whole subtree with it (`collect_free_spans` walks it): a reference to a
    /// child of a deleted group dangles exactly as a reference to the group
    /// itself does (issue #314).
    ///
    /// It is a conservative test in the same sense `write_targets` is. A child
    /// whose object survives the delete through another hard link keeps its
    /// address — `collect_free_spans` reclaims nothing when the incoming count
    /// is not 1 — and is refused here anyway, so the message states what the
    /// commit does rather than what the allocator will conclude.
    ///
    /// The delete test runs *after* the other three so that a replacement this
    /// commit has merely not placed yet — a sibling group later in the same
    /// depth band — is reported as an ordering problem rather than as a
    /// deletion. That ordering is a better default, not a partition: because
    /// `add_targets` claims a replaced path's whole subtree by prefix, a child
    /// the replacement does *not* recreate is genuinely doomed and still
    /// reports "still writing". Distinguishing it would mean enumerating what
    /// a replacement actually rebuilds, which neither list holds. A path the
    /// commit puts back is resolved by step 1 before either test is reached.
    pub(super) fn resolve_reference_target(
        target: &ObjectRefTarget,
        path_addr: &BTreeMap<ObjectPathBuf, u64>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl Source + ?Sized),
        superblock: &Superblock,
        widths: FormatWidths,
        addresses: AddressContext,
    ) -> Result<StoredAddress, Error> {
        let path = match target {
            // An address carries no name to test, so the delete check the path
            // arm makes by prefix is made here on the address itself.
            //
            // It fires on nothing today: every `Raw` this crate stages is a
            // sentinel, because `repack`'s faithful re-emit resolves a real
            // target to a `Path` and leaves only the null and undefined
            // references raw. It is here because the variant carries an
            // arbitrary address, and a producer that staged a real one would
            // otherwise write it past the screen its `Path` twin gets — which
            // is the shape of this bug in the first place (issue #317).
            ObjectRefTarget::Raw(addr) => {
                if let Some(refusal) = invalidated.refusal(*addr) {
                    return Err(Error::EditUnsupported(refusal));
                }
                return Ok(*addr);
            }
            ObjectRefTarget::Path(path) => path,
        };
        let base = addresses;
        if let Some(&addr) = path_addr.get(path) {
            return Ok(base.relative(addr)?);
        }
        if nodes.contains_key(path)
            || add_targets.iter().any(|t| path.starts_with(t))
            || write_targets.contains(path)
        {
            return Err(Error::EditUnsupported(
                "an object-reference dataset targets a path this commit is still writing; \
                 use separate commits",
            ));
        }
        // After the three above; see this function's doc for why, and for what
        // that ordering does and does not buy.
        if delete_targets.iter().any(|d| path.starts_with(d)) {
            return Err(Error::EditUnsupported(
                "an object-reference dataset targets an object this commit deletes, or one \
                 under it; the reference would be left pointing at storage the delete can \
                 reclaim",
            ));
        }
        Ok(
            match crate::group_v2::resolve_path_any_from_source(
                src,
                AccessMode::ReadWrite,
                superblock,
                widths,
                addresses,
                &path.as_path(),
            ) {
                Ok(addr) => base.relative(addr)?,
                Err(_) => StoredAddress::new(UNDEF),
            },
        )
    }

    /// Prove, before any byte of this commit is written, that every
    /// object-reference target across every staged dataset will resolve
    /// successfully — either against a pre-existing untouched object or
    /// against something this same commit places. [`resolve_reference_target`]
    /// classifies a target purely from *whether* an [`ObjectPathBuf`] has been placed
    /// yet (`path_addr.get`), never from the address *value*, so replaying the
    /// apply loop's placement order here with a placeholder address standing in
    /// for "already placed" reproduces the exact same verdict the apply loop's
    /// own calls will reach later, without writing anything.
    ///
    /// The placeholder is the file's **base address**, not zero. Any value
    /// reproduces the verdict, but it is also handed to the same `addr - base`
    /// that converts a real address into its stored form: a zero placeholder
    /// underflows that on every file with a userblock, panicking in a debug
    /// build on an otherwise legal edit. At the base address it converts to a
    /// stored `0`, which is exactly the "placed, real address not known yet"
    /// this stands for. If
    /// this preflight pass returns `Ok`, none of the apply loop's own
    /// `resolve_reference_target` calls can fail, so a reference-resolution
    /// error can no longer leave earlier-processed groups' real writes
    /// orphaned in the file (the failure surfaces here instead, before the
    /// apply loop's first `place`/`write_at`).
    ///
    /// [`resolve_reference_target`]: Self::resolve_reference_target
    pub(super) fn preflight_reference_targets(
        keys: &[ObjectPathBuf],
        flat: &BTreeMap<&ObjectPathBuf, Vec<&FlatDataset>>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl Source + ?Sized),
        superblock: &Superblock,
        widths: FormatWidths,
        addresses: AddressContext,
    ) -> Result<(), Error> {
        let mut by_depth = keys.to_vec();
        // Stable on purpose, for the same reason as the apply loop's copy: this
        // simulation is only faithful if it walks the groups in that same order.
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let mut sim_addr: BTreeMap<ObjectPathBuf, u64> = BTreeMap::new();
        for key in &by_depth {
            if let Some(datasets) = flat.get(key) {
                // Mirrors the apply loop's `group_datasets.sort_by_key(|fd|
                // fd.reference_targets.is_some())`: non-reference datasets are
                // placed (and so become resolvable) before any reference
                // dataset in the same group.
                let mut ordered: Vec<&FlatDataset> = datasets.to_vec();
                // Stable on purpose, like the loop it mirrors.
                ordered.sort_by_key(|fd| fd.reference_targets.is_some());
                for fd in ordered {
                    if let Some(patches) = &fd.reference_targets {
                        for patch in patches {
                            Self::resolve_reference_target(
                                &patch.target,
                                &sim_addr,
                                nodes,
                                add_targets,
                                write_targets,
                                delete_targets,
                                invalidated,
                                src,
                                superblock,
                                widths,
                                addresses,
                            )?;
                        }
                    }
                    sim_addr.insert(key.join(&fd.name), addresses.effective_base().get());
                }
            }
            sim_addr.insert(key.clone(), addresses.effective_base().get());
        }
        Ok(())
    }
}
