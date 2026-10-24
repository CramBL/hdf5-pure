//! Pending editor operations, path indexes, and batch rollback.

use std::collections::{HashMap, HashSet};

use hdf5_pure_format::__private::ChunkedGeometry;

use super::{AppendBuilder, AttrOp, CopyTree, FlatDataset, LOSSY_TAIL_REFUSAL, WriteEngine};
use crate::access_mode::AccessMode;
use crate::address::BaseAddressExt;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::error::Error;
use crate::file_writer::{LENGTH_WIDTH, OFFSET_WIDTH};
use crate::layout_info::FilterId;
use crate::message_type::MessageType;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::source::BytesSource;
use crate::type_builders::{AttrValue, DatasetBuilder};

/// Every edit a session has staged and not yet committed, as one value.
///
/// [`WriteEngine::commit`] takes the whole set out for the duration of an
/// attempt and puts it back if that attempt rejects, so a rejected commit costs
/// the session no staged work - the same guarantee [`super::FreeSnapshot`] gives the
/// free lists, for the same reason (issue #316). Keeping the vectors together
/// makes that total: a
/// staged kind added later participates by construction, where a tenth field
/// would have to be remembered in a hand-written restore.
#[derive(Default)]
pub(super) struct StagedEdits {
    /// Datasets staged by `create_dataset`, as (parent group path, dataset).
    ///
    /// Flattened at staging: [`super::flatten_dataset`] is the
    /// one step of the commit's preflight that *consumes* what it validates, so
    /// running it here is what lets the preflight read the staged set without
    /// destroying it (issue #316). It is a pure function of the builder, so the
    /// guards it raises - a missing shape, data that does not match it, a
    /// feature this engine cannot reproduce - are reported at the call that
    /// stages the dataset, where the caller still has the context to fix them.
    pub(super) datasets: Vec<(ObjectPathBuf, FlatDataset)>,
    /// Value overwrites staged by `write_dataset`, as (full dataset path,
    /// dataset). Each replaces an existing dataset's values in place. The new
    /// shape must match the on-disk shape, and the datatype must be compatible
    /// under [`super::overwrite_datatypes_match`]. Applied on the next `commit`.
    /// Flattened at staging, for the reason given on [`datasets`](Self::datasets).
    /// the match against the on-disk dataset is a commit-time check and stays
    /// one, since it reads the file.
    pub(super) writes: Vec<(ObjectPathBuf, FlatDataset)>,
    /// Appends staged by `append_dataset`, as (full dataset path, builder). Each
    /// grows an existing chunked, unlimited, Extensible-Array-indexed dataset
    /// along axis 0 by keeping its existing chunk data in place and rebuilding the
    /// index over the kept plus newly-appended (and any rewritten trailing) chunks.
    /// Applied on the next `commit`.
    pub(super) appends: Vec<(ObjectPathBuf, AppendBuilder)>,
    /// New groups staged by `create_group`, as full paths.
    pub(super) groups: Vec<ObjectPathBuf>,
    /// Group attribute edits staged as (group path, operation). The path may be
    /// a group created in this same session.
    pub(super) group_attrs: Vec<(ObjectPathBuf, AttrOp)>,
    /// Dataset attribute edits staged as (full dataset path, operation), applied
    /// on the next `commit`. Each relocates the dataset's object header (like a
    /// relocating overwrite): the header is rebuilt with the compact-attribute
    /// change, its single naming link is patched, and the old header freed - the
    /// dataset's data and chunk index stay in place. The target must be an existing,
    /// single-hard-link dataset using compact (not dense fractal-heap) attributes.
    pub(super) dataset_attrs: Vec<(ObjectPathBuf, AttrOp)>,
    /// Links staged for removal by `delete`, as full paths.
    pub(super) deletes: Vec<ObjectPathBuf>,
    /// Object copies staged by `copy`, as (source path, destination full path).
    pub(super) copies: Vec<(ObjectPathBuf, ObjectPathBuf)>,
    /// Cross-file object copies staged by `copy_from`, as (destination full path,
    /// the source subtree already read out of the other file). The subtree is read
    /// and foreign-address-screened eagerly in `copy_from` (the source file is
    /// borrowed only for that call), then linked in at the next `commit`.
    pub(super) cross_copies: Vec<(ObjectPathBuf, CopyTree)>,
    /// Where the creation at each full path sits in [`datasets`](Self::datasets)
    /// / [`groups`](Self::groups), so a by-name question about the staged set
    /// uses a hash lookup.
    ///
    /// The scan is what made a writer that stages one dataset per column and
    /// then binds a handle to each quadratic in the number of columns (issue
    /// #392). Both maps are derived from the vectors beside them and are kept in
    /// step at the three places those change: the push that stages a creation,
    /// the truncation [`rewind`](Self::rewind) makes, and the rebuild
    /// [`withdraw_at`](Self::withdraw_at) makes after removing from the middle.
    /// A path staged twice keeps its *first* position, which is the entry a scan
    /// would have found. The commit rejects such a pair anyway.
    pub(super) dataset_at: HashMap<ObjectPathBuf, usize>,
    pub(super) group_at: HashMap<ObjectPathBuf, usize>,
}

/// What a session has staged at a path, for a handle addressing an object the
/// commit has not written yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StagedKind {
    Group,
    Dataset,
}

/// A creation this session has staged at a path, and whether it takes that path
/// over from the file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StagedObject {
    pub(crate) kind: StagedKind,
    /// Whether this same commit removes the link the file holds at that path.
    ///
    /// True makes the creation a *replacement* (issue #305). False means the
    /// path is available for creation - a creation colliding with one that
    /// survives is rejected where it is staged, and would not be reported here
    /// either, since the file's own object is what the name means.
    pub(crate) replaces_link: bool,
}

/// One direct child a group gains from this session's staged creations.
pub(crate) struct StagedChild {
    pub(crate) name: String,
    pub(crate) kind: StagedKind,
    /// Whether a staged deletion removes the link this name has in the file, so
    /// the creation supersedes it. See
    /// [`StagedObject::replaces_link`].
    pub(crate) replaces_link: bool,
}

/// What a staged dataset can say about itself before it is written.
///
/// Everything here is settled when the dataset is staged - [`super::flatten_dataset`]
/// has already validated the builder - so a handle reports from it without
/// reading the file, and the reports are what the commit goes on to write.
pub(crate) struct StagedMeta {
    pub(crate) datatype: Datatype,
    pub(crate) dimensions: Vec<u64>,
    /// `None` for a fixed-shape dataset, matching `Dataset::maxshape`.
    pub(crate) maxshape: Option<Vec<MaxExtent>>,
    pub(crate) chunked: bool,
    /// Each staged filter's registered id and its `H5Z_FLAG_OPTIONAL` flag, in
    /// pipeline order.
    pub(crate) filters: Vec<(FilterId, bool)>,
}

/// Where a [`StagedEdits`] stood before a batch of staging calls, so a batch
/// that fails partway can be undone (see [`StagedEdits::rewind`]).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct StagedMark {
    pub(super) datasets: usize,
    pub(super) writes: usize,
    pub(super) appends: usize,
    pub(super) groups: usize,
    pub(super) group_attrs: usize,
    pub(super) dataset_attrs: usize,
    pub(super) deletes: usize,
    pub(super) copies: usize,
    pub(super) cross_copies: usize,
}

impl StagedEdits {
    /// The length of every staged vector right now, for
    /// [`rewind`](Self::rewind).
    ///
    /// Destructured here and in
    /// [`rewind`](Self::rewind), so that the "a staged kind added later
    /// participates by construction" claim on [`StagedEdits`] is enforced:
    /// a struct pattern naming fewer fields than the struct has
    /// does not compile, and the unused binding a tenth kind would leave in
    /// `rewind` is an error under the crate's `-D warnings`.
    pub(super) fn mark(&self) -> StagedMark {
        let Self {
            datasets,
            writes,
            appends,
            groups,
            group_attrs,
            dataset_attrs,
            deletes,
            copies,
            cross_copies,
            // Derived from `datasets` and `groups`, and restored with them.
            dataset_at: _,
            group_at: _,
        } = self;
        StagedMark {
            datasets: datasets.len(),
            writes: writes.len(),
            appends: appends.len(),
            groups: groups.len(),
            group_attrs: group_attrs.len(),
            dataset_attrs: dataset_attrs.len(),
            deletes: deletes.len(),
            copies: copies.len(),
            cross_copies: cross_copies.len(),
        }
    }

    /// Drop everything staged since `mark`.
    ///
    /// Staging inside a batch only ever appends - every `stage_*` entry point
    /// validates and then pushes - so truncating to the recorded lengths is an
    /// exact undo of the calls made in between, and leaves anything staged
    /// before them alone. The two operations that instead *change* what is
    /// already staged - the fold
    /// [`WriteEngine::stage_dataset_append_pending`] makes into a staged
    /// creation, and the withdrawal [`withdraw_at`](Self::withdraw_at) makes for
    /// [`WriteEngine::delete`] - would leave that untrue, so
    /// [`WriteEngine::refuse_mid_batch`] rejects them while a batch is open.
    ///
    /// The by-path index is truncated with the vectors it describes, so a
    /// rewound creation stops reporting as well as stops existing.
    pub(super) fn rewind(&mut self, mark: StagedMark) {
        for i in mark.datasets..self.datasets.len() {
            let key = {
                let (parent, fd) = &self.datasets[i];
                parent.join(&fd.name)
            };
            // Only where it still points at the entry being dropped: a second
            // creation at one path never displaced the first.
            if self.dataset_at.get(&key) == Some(&i) {
                self.dataset_at.remove(&key);
            }
        }
        for i in mark.groups..self.groups.len() {
            let key = self.groups[i].clone();
            if self.group_at.get(&key) == Some(&i) {
                self.group_at.remove(&key);
            }
        }
        let StagedMark {
            datasets,
            writes,
            appends,
            groups,
            group_attrs,
            dataset_attrs,
            deletes,
            copies,
            cross_copies,
        } = mark;
        self.datasets.truncate(datasets);
        self.writes.truncate(writes);
        self.appends.truncate(appends);
        self.groups.truncate(groups);
        self.group_attrs.truncate(group_attrs);
        self.dataset_attrs.truncate(dataset_attrs);
        self.deletes.truncate(deletes);
        self.copies.truncate(copies);
        self.cross_copies.truncate(cross_copies);
    }

    /// Stage a dataset creation, indexing it by its full path.
    pub(super) fn push_dataset(&mut self, parent: ObjectPathBuf, fd: FlatDataset) {
        self.dataset_at
            .entry(parent.join(&fd.name))
            .or_insert(self.datasets.len());
        self.datasets.push((parent, fd));
    }

    /// Stage a group creation, indexing it by its path.
    pub(super) fn push_group(&mut self, path: ObjectPathBuf) {
        self.group_at
            .entry(path.clone())
            .or_insert(self.groups.len());
        self.groups.push(path);
    }

    /// Where the staged dataset at `path` sits, if there is one.
    ///
    /// The position the index gives is checked against the entry it refers to.
    /// An index that fell behind the vector produces a miss.
    pub(super) fn dataset_position(&self, path: &ObjectPathBuf) -> Option<usize> {
        let i = *self.dataset_at.get(path)?;
        let (parent, fd) = self.datasets.get(i)?;
        path.covers_child(parent, &fd.name).then_some(i)
    }

    /// The staged dataset at `path`, if there is one.
    pub(super) fn dataset_at(&self, path: &ObjectPathBuf) -> Option<&FlatDataset> {
        let i = self.dataset_position(path)?;
        self.datasets.get(i).map(|(_, fd)| fd)
    }

    /// The staged dataset at `path`, mutably.
    pub(super) fn dataset_at_mut(&mut self, path: &ObjectPathBuf) -> Option<&mut FlatDataset> {
        let i = self.dataset_position(path)?;
        self.datasets.get_mut(i).map(|(_, fd)| fd)
    }

    /// Whether a group creation is staged at `path`. Checked against the entry
    /// the index names, for the reason [`dataset_position`](Self::dataset_position)
    /// gives.
    pub(super) fn has_group_at(&self, path: &ObjectPathBuf) -> bool {
        match self.group_at.get(path) {
            Some(&i) => self.groups.get(i).is_some_and(|p| p == path),
            None => false,
        }
    }

    /// Whether a staged deletion removes the link at `path`, or the link to an
    /// ancestor that carries it away.
    pub(super) fn deletes_cover(&self, path: &ObjectPathBuf) -> bool {
        self.deletes.iter().any(|d| path.starts_with(d))
    }

    /// Whether a staged deletion hands `path` over to a creation staged there  -
    /// which makes the pair a *replacement*
    /// (issue #305).
    ///
    /// Stricter than [`deletes_cover`](Self::deletes_cover), and deliberately:
    /// a deletion of an *ancestor* only carries the name over when this same
    /// session builds that ancestor again, which is exactly what the commit
    /// requires of everything at or under a replaced path. Deleting `g` and
    /// creating `g/x` without recreating `g` is a batch the commit rejects, so
    /// the file's own `g/x` is still what the path refers to - reporting it as a
    /// replacement would make a live, readable dataset report
    /// [`Error::NotCommitted`] and vanish from its group's listing until the
    /// rejection.
    ///
    /// The exact deletion is the empty case of the same rule: with `d == path`
    /// there is no ancestor between them to have been recreated.
    pub(super) fn deletes_hand_over(&self, path: &ObjectPathBuf) -> bool {
        self.deletes.iter().any(|d| {
            path.starts_with(d) && (d.len()..path.len()).all(|n| self.has_group_at(&path.prefix(n)))
        })
    }

    /// Withdraw every edit staged at or under `path`, as though the calls that
    /// staged them had never been made, and report whether a *creation* was
    /// among them.
    ///
    /// This is what lets [`WriteEngine::delete`] cancel an object this session
    /// staged and has not written: there is nothing in the file to unlink, so a
    /// deletion of it is a withdrawal. Deletions are deliberately left alone  -
    /// one under a withdrawn creation still refers to a link the file holds, and two
    /// overlapping deletions are rejected by the commit.
    ///
    /// Destructured for the reason [`mark`](Self::mark) is: a staged kind added
    /// later has to be classified here.
    pub(super) fn withdraw_at(&mut self, path: &ObjectPathBuf) -> bool {
        let before = {
            let Self {
                datasets,
                writes,
                appends,
                groups,
                group_attrs,
                dataset_attrs,
                deletes,
                copies,
                cross_copies,
                dataset_at: _,
                group_at: _,
            } = self;
            let before = datasets.len() + groups.len();
            datasets.retain(|(parent, fd)| !path.covers_child(parent, &fd.name));
            writes.retain(|(p, _)| !p.starts_with(path));
            appends.retain(|(p, _)| !p.starts_with(path));
            groups.retain(|p| !p.starts_with(path));
            group_attrs.retain(|(p, _)| !p.starts_with(path));
            dataset_attrs.retain(|(p, _)| !p.starts_with(path));
            copies.retain(|(_, dst)| !dst.starts_with(path));
            cross_copies.retain(|(dst, _)| !dst.starts_with(path));
            let _ = &deletes;
            before
        };
        let withdrew = self.datasets.len() + self.groups.len() < before;
        if withdrew {
            self.reindex();
        }
        withdrew
    }

    /// Rebuild the by-path index from the vectors, after a removal from the
    /// middle shifted every position after it.
    pub(super) fn reindex(&mut self) {
        self.dataset_at.clear();
        self.group_at.clear();
        for (i, (parent, fd)) in self.datasets.iter().enumerate() {
            self.dataset_at.entry(parent.join(&fd.name)).or_insert(i);
        }
        for (i, path) in self.groups.iter().enumerate() {
            self.group_at.entry(path.clone()).or_insert(i);
        }
    }

    /// Whether nothing at all is staged.
    ///
    /// Requested as "is the mark the zero mark", which is exact - nothing is staged
    /// exactly when every vector has length zero - and leaves [`mark`](Self::mark)
    /// as the single place that lists every vector. The two callers,
    /// [`WriteEngine::has_staged_edits`] and the commit's own no-op return, then
    /// cannot come to disagree, and a staged kind added later reaches both. (The
    /// commit's *fast path* requests a narrower question and spells its own subset
    /// out.)
    pub(super) fn is_empty(&self) -> bool {
        self.mark() == StagedMark::default()
    }
}

impl WriteEngine {
    /// Stage a new dataset, added on the next [`commit`](Self::commit).
    ///
    /// `path` is the dataset's full path. Everything before the last component
    /// identifies the parent group, which must exist (or be created in this session).
    /// `builder` is the same [`DatasetBuilder`] [`FileBuilder`](crate::FileBuilder)
    /// uses, configured by the caller. Its name is taken from `path`, so the two
    /// cannot disagree.
    ///
    /// The builder is passed in finished. That is deliberate: a borrow into the engine forces the
    /// caller to hold it - and, above this layer, its lock - for as long as the
    /// builder is being configured, which is what let a user closure deadlock
    /// against the same file it was reading (issue #200).
    ///
    /// The dataset may be contiguous or chunked, and chunked datasets may be
    /// filtered (`with_deflate`, `with_shuffle`, `with_fletcher32`,
    /// `with_scale_offset`, `with_zfp`) and/or extensible (`with_maxshape`). An
    /// empty (zero-element) dataset is supported under either storage, a
    /// dataset with `with_provenance` is supported, and a
    /// contiguous dataset may carry variable-length attributes, a
    /// variable-length-string payload (`with_vlen_strings`), or path-resolved
    /// object-reference elements (`with_path_references`. Chunking any of
    /// these is not supported). An attribute set too large for the object header
    /// is written to a fractal heap, as the whole-file writer does.
    pub(crate) fn stage_created_dataset(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        let Some((parent, leaf)) = path.split_leaf() else {
            return Err(Error::EditUnsupported("dataset path has an empty name"));
        };
        self.refuse_creation_collision(path, StagedKind::Dataset)?;
        self.staged
            .push_dataset(parent, super::flatten_dataset(builder, leaf)?);
        Ok(())
    }

    /// Stage an in-place overwrite of an **existing** dataset's values (the HDF5
    /// `H5Dwrite` whole-dataset write), applied on the next
    /// [`commit`](Self::commit).
    ///
    /// `path` is the full path of a dataset that must already exist. `builder`
    /// supplies the replacement data and is named from `path`, as in
    /// [`stage_created_dataset`](Self::stage_created_dataset).
    ///
    /// The replacement shape must match the on-disk shape, and the datatype must
    /// be compatible under [`super::overwrite_datatypes_match`], or [`Self::commit`]
    /// returns [`Error::EditUnsupported`]. Byte order is ignored for one-byte
    /// fixed-point types, including nested base and member types. Contiguous, compact,
    /// and chunked (including filtered) datasets are all supported. The dataset's
    /// existing chunk geometry, filter pipeline, and chunk index are taken from the
    /// on-disk header. A chunk index this engine cannot enumerate (a version-2
    /// B-tree) is rejected. Partial / sub-region writes are out of scope - the
    /// whole dataset is replaced.
    ///
    /// What the builder alone can rule out  -
    /// [`refuse_unsupported_overwrite`](Self::refuse_unsupported_overwrite)  -
    /// is rejected *here*.
    ///
    /// When the new data is the same length as the existing contiguous data block
    /// (the common case), the bytes are written straight into that block: no
    /// object header is rewritten and the superblock root is not flipped, so the
    /// commit's linearization point is the synced data write itself. A chunked
    /// dataset is handled the same way when every (re-encoded) chunk is the same
    /// byte length as the slot it replaces - an unfiltered overwrite (chunk sizes
    /// are fixed by the unchanged shape) or a filtered one whose re-encoded chunks
    /// match - so it too writes straight into the existing chunk slots. When the
    /// length differs (a resized contiguous block, or a filtered chunk that no
    /// longer fits), the dataset's storage is rebuilt at end-of-file (or in
    /// reusable freed space), the old extent is freed, the data-layout message is
    /// repointed, the object header is rewritten, and the parent group's link is
    /// patched - exactly like an addition relocates the path up to the root. A
    /// relocating overwrite moves the object header, so it is rejected unless the
    /// dataset has a single hard link.
    pub(crate) fn stage_dataset_write(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        // The root group has no link to overwrite: the superblock holds its address.
        let Some((_, leaf)) = path.split_leaf() else {
            return Err(Error::EditUnsupported("cannot overwrite the root group"));
        };
        let fd = super::flatten_dataset(builder, leaf)?;
        Self::refuse_unsupported_overwrite(&fd)?;
        self.staged.writes.push((path.clone(), fd));
        Ok(())
    }

    /// Stage an append of new elements to an **existing** chunked, unlimited
    /// dataset, applied on the next [`commit`](Self::commit).
    ///
    /// `path` identifies a dataset that must already exist. `builder` supplies the
    /// elements to add via its typed / generic / raw `append_*` methods.
    ///
    /// Unlike [`stage_dataset_write`](Self::stage_dataset_write) (a value
    /// overwrite that forbids any shape change) this **grows** the dataset along
    /// its first (axis-0) dimension. It works on **filtered** datasets: the
    /// appended chunks are compressed through the dataset's own on-disk filter
    /// pipeline (deflate / shuffle / fletcher32 / scale-offset / LZF, and ZFP
    /// with the `zfp` feature), and the pipeline, datatype, fill value, and attributes are
    /// preserved verbatim. Appends of any length are supported - when the
    /// dataset's current length is not a whole multiple of the chunk length, the
    /// single trailing partial chunk is read, extended, and re-encoded. Every
    /// other existing chunk is carried by metadata alone, so the existing data is
    /// not rewritten and the file does not grow by the whole dataset per append.
    ///
    /// This does **not** use SWMR and sets no consistency flag. Like every other
    /// staged edit it commits by appending the new chunks and a rebuilt
    /// index at end-of-file and repointing the superblock last (under the
    /// session's exclusive lock), so a crash leaves either the original dataset or
    /// the fully-grown one, never a torn state.
    ///
    /// The first release supports the Extensible-Array chunk index (the index the
    /// reference C library and h5py select for a single unlimited dimension under
    /// the latest format, and the one this crate writes for every unlimited
    /// dataset), rank-1 datasets, and datasets with a single hard link. A dataset
    /// that is not chunked, not unlimited along axis 0, not Extensible-Array
    /// indexed, higher than rank 1, uses a filter this engine cannot re-encode,
    /// has a sparse chunk grid, or (for [`append_raw`](AppendBuilder::append_raw))
    /// has a big-endian element datatype is rejected with
    /// [`Error::AppendUnsupported`]. Use [`Dataset::is_chunked`](crate::Dataset::is_chunked),
    /// [`maxshape`](crate::Dataset::maxshape), and [`filters`](crate::Dataset::filters)
    /// to check eligibility up front.
    pub(crate) fn stage_dataset_append(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_some() {
            // This call came through a handle onto the object *in the file*: a
            // handle onto the staged creation goes to
            // [`stage_dataset_append_pending`] instead. So the elements could
            // only be meant for the object the same commit replaces, which is
            // never what the caller wants and is what `commit` rejects two calls
            // later. Said here, where the handle that requested is still in hand.
            return Err(Error::EditUnsupported(
                "a dataset is staged at this path in the same commit, so an append through a \
                 handle onto the object the file holds there could only grow the object being \
                 replaced; append through the handle that staged the creation, or commit first",
            ));
        }
        self.refuse_lossy_partial_tail(path, &builder)?;
        self.staged.appends.push((path.clone(), builder));
        Ok(())
    }

    /// Reject, at the call that stages an append, a dataset whose lossy pipeline
    /// sits on a partial trailing chunk - the rejection
    /// [`prepare_append`](Self::prepare_append) raises during the commit's
    /// preflight, brought forward to where the caller can act on it (issue #407).
    ///
    /// Waiting for the commit makes that rejection permanent for the session and
    /// costs every edit staged beside it: a preflight failure restores the staged
    /// set, so the offending append is still there at the next
    /// [`commit`](crate::File::commit), and [`File::close`](crate::File::close)
    /// runs one and returns its `Err` - discarding the whole staged set on the way
    /// out. `BufferedAppender::new` rejects the same dataset eagerly for the same
    /// reason. This puts the staged path on the same footing.
    ///
    /// **Positive proof only.** Every reason the geometry cannot be read here - a
    /// path that does not resolve, a header this engine does not parse, a layout
    /// that is not chunked, a rank this release does not append to - leaves the
    /// result to the preflight, which reports each of them. Nothing about the bytes
    /// being appended enters into it: the on-disk length is what decides whether a
    /// trailing chunk has to be re-encoded, and staged appends never change that
    /// before the commit reads it. The commit-time check stays as the backstop,
    /// and is the only one that runs for an append staged through some other
    /// entry point.
    fn refuse_lossy_partial_tail(
        &self,
        path: &ObjectPathBuf,
        builder: &AppendBuilder,
    ) -> Result<(), Error> {
        // A zero-length append is dropped by the preflight before it ever reaches
        // `prepare_append`, so it stays a no-op here.
        if builder.raw().is_empty() {
            return Ok(());
        }
        if self.appends_onto_a_lossy_partial_tail(path) {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
        Ok(())
    }

    /// The geometry question behind [`refuse_lossy_partial_tail`](Self::refuse_lossy_partial_tail):
    /// does the file hold, at `path`, a rank-1 chunked dataset under a lossy
    /// filter pipeline whose length is not a whole multiple of its chunk length?
    /// `false` whenever that cannot be established, for whatever reason.
    fn appends_onto_a_lossy_partial_tail(&self, path: &ObjectPathBuf) -> bool {
        let Ok(addr) = crate::group_v2::resolve_path_any_from_source(
            &self.image(),
            AccessMode::ReadWrite,
            &self.superblock,
            self.widths,
            self.addresses,
            &path.as_path(),
        ) else {
            return false;
        };
        let Ok(region) = Self::gather_oh_messages(&self.image(), addr, self.addresses) else {
            return false;
        };
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        let mut p = 0;
        while let Ok(Some((msg_type, body, body_end))) = region.next_message(p) {
            match msg_type {
                MessageType::DATATYPE => datatype = Some((body, body_end)),
                MessageType::DATASPACE => dataspace = Some((body, body_end)),
                MessageType::DATA_LAYOUT => layout = Some((body, body_end)),
                MessageType::FILTER_PIPELINE => filter = Some((body, body_end)),
                _ => {}
            }
            p = body_end;
        }
        // No pipeline at all is nothing to re-encode, and an unparseable one is
        // rejected by the preflight on its own terms.
        let (Some((fb, fe)), Some((dt_b, dt_e)), Some((ds_b, ds_e)), Some((lb, le))) =
            (filter, datatype, dataspace, layout)
        else {
            return false;
        };
        let Ok(pipeline) = crate::filter_pipeline::parse_filter_pipeline(&region[fb..fe]) else {
            return false;
        };
        if super::pipeline_lossless(&pipeline) {
            return false;
        }
        let Ok((disk_dt, _)) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])
        else {
            return false;
        };
        let Ok(disk_ds) =
            Dataspace::parse_with_length_width(&region[ds_b..ds_e], self.widths.lengths)
        else {
            return false;
        };
        let Ok(dl) = DataLayout::parse_with_widths(&region[lb..le], self.widths) else {
            return false;
        };
        // Rank 1 is the shape the preflight's `current_dim0 % chunk_elems`
        // arithmetic is about. Anything else it rejects before reaching that test.
        if disk_ds.dimensions.len() != 1 {
            return false;
        }
        let Ok(geometry) = ChunkedGeometry::new(&disk_dt, &disk_ds, &dl) else {
            return false;
        };
        match (geometry.spatial.first(), disk_ds.dimensions.first()) {
            (Some(&chunk_elems), Some(&dim0)) if chunk_elems != 0 => dim0 % chunk_elems != 0,
            _ => false,
        }
    }

    /// Stage an append made through a handle onto a dataset **this session
    /// staged and has not committed**, folding the elements into the pending
    /// creation.
    ///
    /// The caller establishes that its handle is the pending one.
    /// [`stage_dataset_append`](Self::stage_dataset_append) is the entry point
    /// for a handle onto an object the file already holds, and rejects the same
    /// path. Between the caller's check and this lock another thread can commit,
    /// which leaves the dataset in the file and nothing staged at its path - so
    /// the append becomes an ordinary staged one.
    pub(crate) fn stage_dataset_append_pending(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_none() {
            // Nothing staged here after all, so these elements grow the object the
            // file holds - including its trailing chunk, which the same eager
            // rejection applies to. The fold below operates on the staged bytes.
            self.refuse_lossy_partial_tail(path, &builder)?;
            self.staged.appends.push((path.clone(), builder));
            return Ok(());
        }
        self.refuse_mid_batch()?;
        self.extend_staged_dataset(path, &builder)
    }

    /// Reject an edit that changes or withdraws something already staged while a
    /// [`stage_atomically`](Self::stage_atomically) batch is open.
    ///
    /// [`StagedEdits::rewind`] undoes a batch by truncating the staged vectors,
    /// which is exact only while staging appends to them. The two operations
    /// that do otherwise - the fold
    /// [`stage_dataset_append_pending`](Self::stage_dataset_append_pending)
    /// makes, and the withdrawal [`delete`](Self::delete) makes - would leave a
    /// rejected batch holding half of one. No caller reaches this today: a batch
    /// replays [`StagedOp`](crate::StagedGroup)s, which only ever create. It is
    /// here so the claim `rewind` rests on is enforced.
    fn refuse_mid_batch(&self) -> Result<(), Error> {
        if self.staging_batch {
            return Err(Error::EditUnsupported(
                "an edit that changes or withdraws an already-staged object cannot be made \
                 inside an atomic staging batch, whose undo drops additions only",
            ));
        }
        Ok(())
    }

    /// Fold an append onto a dataset this same session staged and has not
    /// committed into the pending creation: the elements are added to the
    /// bytes the commit will write and the leading dimension grows to match.
    ///
    /// There is no dataset in the file to grow, so this is the same edit as
    /// having handed the elements to the builder in the first place - which is
    /// why it requires neither an unlimited dimension nor a chunked layout,
    /// where an append onto a *committed* dataset needs both. Anything the
    /// staged creation must resolve at commit time per element - a
    /// variable-length string payload, an object reference - is rejected: those
    /// carry a side table indexed by element that this cannot extend.
    ///
    /// A dataset with `with_provenance` is grown:
    /// its attributes are derived from the bytes, so they are rebuilt from the
    /// grown ones ([`FlatDataset::rebuild_provenance`]) and the digest the
    /// commit writes is the digest of what it writes.
    fn extend_staged_dataset(
        &mut self,
        path: &ObjectPathBuf,
        builder: &AppendBuilder,
    ) -> Result<(), Error> {
        if builder.dt_conflict() {
            return Err(Error::AppendUnsupported(
                "this append mixes element datatypes; use one element type per append",
            ));
        }
        let fd = self
            .staged
            .dataset_at_mut(path)
            .expect("caller checked a dataset is staged at this path");
        if fd.vl_string_staging.is_some() || fd.reference_targets.is_some() {
            return Err(Error::AppendUnsupported(
                "a staged variable-length-string or object-reference dataset cannot be appended \
                 to before it is committed: its per-element side table is built with the \
                 dataset. Supply every element through the builder that creates it",
            ));
        }
        if let Some(dt) = builder.elem_dt() {
            if *dt != fd.dt {
                return Err(Error::AppendUnsupported(
                    "appended element datatype does not match the staged dataset's datatype",
                ));
            }
        }
        let Some((lead, inner)) = fd.ds.dimensions.split_first() else {
            return Err(Error::AppendUnsupported(
                "a scalar dataset has no dimension to append along",
            ));
        };
        // One row of the leading dimension, in bytes: the element size times
        // every trailing extent. A rank-1 dataset's row is one element.
        let mut row_bytes = u64::from(fd.dt.type_size());
        for &d in inner {
            row_bytes = row_bytes.checked_mul(d).ok_or(Error::AppendUnsupported(
                "staged dataset row size overflows",
            ))?;
        }
        if row_bytes == 0 {
            return Err(Error::AppendUnsupported(
                "a staged dataset whose rows hold no bytes cannot be appended to",
            ));
        }
        let bytes = builder.raw();
        if bytes.len() as u64 % row_bytes != 0 {
            return Err(Error::AppendUnsupported(
                "appended byte length is not a whole number of rows of the staged dataset",
            ));
        }
        let grown = lead
            .checked_add(bytes.len() as u64 / row_bytes)
            .ok_or(Error::AppendUnsupported("staged dataset length overflows"))?;
        // A finite maximum is a promise the commit would reject to break.
        if let Some(max) = fd
            .ds
            .max_dimensions
            .as_ref()
            .and_then(|m| m.first().copied())
            && matches!(max, MaxExtent::Fixed(limit) if limit < grown)
        {
            return Err(Error::AppendUnsupported(
                "appending would grow the staged dataset past its maximum shape",
            ));
        }
        fd.raw.extend_from_slice(bytes);
        fd.ds.dimensions[0] = grown;
        // The digest covers the bytes, so it is recomputed over the grown ones. Doing
        // it here, on the record the commit writes, is what keeps
        // `verify_provenance` passing on the file this produces.
        #[cfg(feature = "provenance")]
        fd.rebuild_provenance();
        Ok(())
    }

    /// Whether any staged tree edit is still uncommitted. In-place appends
    /// ([`append_inplace_gathered`](Self::append_inplace_gathered)) are applied immediately and are
    /// never staged, so they never affect this. It reflects only edits awaiting
    /// [`commit`](Self::commit) - `create_group`, `create_dataset`,
    /// `write_dataset`, `append_dataset`, group and dataset attribute edits,
    /// `delete`, `copy`, and `copy_from`. Dropping the session silently discards
    /// any staged edits.
    ///
    /// A [`commit`](Self::commit) that rejects leaves this reporting `true`: the
    /// staged set it declined to apply is put back exactly as it was.
    pub fn has_staged_edits(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Run `f`, and if it fails, drop whatever it managed to stage.
    ///
    /// One caller-facing call can stage many edits - `create_group_with` stages
    /// a group, its attributes and its whole subtree - and each is validated as
    /// it is staged, so the fifth dataset can be rejected after four have been
    /// recorded. Without this, such a call would return `Err` having changed the
    /// session anyway, which is the half of issue #316 that lives one level
    /// above `commit`: a rejected operation must cost the session nothing.
    pub(crate) fn stage_atomically<R>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let mark = self.staged.mark();
        // Saved and restored, so a nested batch does
        // not lift the outer one's guard on its way out.
        let outer = std::mem::replace(&mut self.staging_batch, true);
        let result = f(self);
        self.staging_batch = outer;
        if result.is_err() {
            self.staged.rewind(mark);
        }
        result
    }

    /// Whether `target` (an [`append_inplace_gathered`](Self::append_inplace_gathered) dataset path)
    /// or any of its ancestors is named by a staged edit that a later
    /// [`commit`](Self::commit) would relocate, replace, or delete. `create_group`
    /// and group-attribute edits are excluded: they rewrite a group header without
    /// moving a descendant dataset's header or freeing its storage, so they cannot
    /// stale the append geometry cache.
    pub(super) fn append_conflicts_with_pending(&self, target: &ObjectPathBuf) -> bool {
        let hits = |p: &ObjectPathBuf| target.overlaps(p);
        self.staged.writes.iter().any(|(p, _)| hits(p))
            || self.staged.appends.iter().any(|(p, _)| hits(p))
            || self.staged.deletes.iter().any(&hits)
            || self.staged.copies.iter().any(|(_, dst)| hits(dst))
            || self.staged.cross_copies.iter().any(|(dst, _)| hits(dst))
            || self.staged.dataset_attrs.iter().any(|(p, _)| hits(p))
            || self
                .staged
                .datasets
                .iter()
                .any(|(parent, fd)| target.overlaps(&parent.join(&fd.name)))
    }

    /// What this session has staged at `path`, for a handle that can
    /// address a not-yet-committed object by name.
    ///
    /// Only a *creation* reports, and only when it owns the path - when the path
    /// is available, or when this same commit removes the one it does.
    /// Both halves matter:
    ///
    /// - A staged **deletion** on its own hides nothing. The object is in the
    ///   file and still readable, and a lookup that pretended otherwise would
    ///   reject a read the file can serve.
    /// - A creation **colliding** with a link the commit does not remove is one
    ///   [`refuse_creation_collision`](Self::refuse_creation_collision) turns
    ///   away where it is staged, so this arm is the same rule requested a second
    ///   time. Reporting such a creation
    ///   would shadow a live, readable object with a handle that never becomes
    ///   valid, so the file's own object stays the meaning of the name.
    /// - The two together are a **replacement** (issue #305), and the creation
    ///   is what the path names from the moment it is staged, before the commit
    ///   as after. "The two together" is
    ///   [`StagedEdits::deletes_hand_over`]'s rule: the deletion refers to this path,
    ///   or an ancestor this session builds again.
    ///
    /// Only a path this session staged is reported, and every group above a
    /// staged object must be one the file already holds or one this session
    /// stages at its own path. [`commit`](Self::commit) rejects the whole batch
    /// with [`Error::EditUnsupported`] when a group above a creation is neither,
    /// the state `a/b` is in when [`stage_created_dataset`](Self::stage_created_dataset) stages
    /// `a/b/col` alone.
    pub(crate) fn staged_object(&self, path: &ObjectPathBuf) -> Option<StagedObject> {
        if self.stages_no_creations() {
            return None;
        }
        if path.is_empty() {
            // The root always exists. Nothing can stage it.
            return None;
        }
        let kind = if self.staged.dataset_at(path).is_some() {
            StagedKind::Dataset
        } else if self.staged.has_group_at(path) {
            StagedKind::Group
        } else {
            return None;
        };
        let replaces_link = self.staged.deletes_hand_over(path);
        // The file is requested only about a path a creation actually names, so an
        // ordinary open of an object nothing is staged at never pays for this.
        if !replaces_link && self.path_in_file(path) {
            return None;
        }
        Some(StagedObject {
            kind,
            replaces_link,
        })
    }

    /// Reject a creation at a path something already owns - a link the file
    /// holds and this session does not remove, or a creation this session
    /// already staged - at the call that stages it.
    ///
    /// The commit rejects the same batch (`a link with this name already
    /// exists`), and the file arm raises that rejection's own error so the two
    /// cannot disagree about what a collision is. Rejecting here is what lets
    /// [`Group::create_dataset`](crate::Group::create_dataset) and its two
    /// siblings hand back a handle onto the object they stage: a creation that
    /// merely collides is not what its path names - the file's object still is
    /// so a handle onto it would have addressed the *old* object while
    /// claiming to address the new one.
    ///
    /// A creation this session already stages at the path is rejected for the
    /// same reason, one the commit's own rejection cannot replace. The staged
    /// set is indexed by path and the index keeps the *first* record at each one
    /// (`or_insert`), so a second creation is staged behind the first and every
    /// handle onto that path - the one this call would hand back included  -
    /// reports `shape`, `dtype`, `filters` and `append_staged` from the record
    /// the caller did not just describe. No committed object is misaddressed,
    /// since the commit rejects the batch, but the window between the two is one
    /// where a handle reports another object's geometry as its own.
    ///
    /// Two *group* creations at one path are the exception and stay allowed:
    /// they refer to one node, the commit builds one group from them, and a handle
    /// onto either addresses that group. Re-staging a group is how a caller adds
    /// attributes or children to one it already staged.
    ///
    /// A [`copy`](Self::copy) destination is still the commit's collision to
    /// reject: it hands back no handle, so nothing can be misaddressed by it.
    fn refuse_creation_collision(
        &self,
        path: &ObjectPathBuf,
        kind: StagedKind,
    ) -> Result<(), Error> {
        // An empty path identifies the root group, which no link reaches and
        // nothing creates. Both callers reject it with a message of their own
        // before this is reached, so this returns `Ok` and leaves the collision
        // below to a path that identifies a link.
        if path.is_empty() {
            return Ok(());
        }
        // Requested before the file, so that staging twice over a link this session
        // deletes - a replacement, which the file arm below lets through - is
        // caught by the same rule as staging twice over nothing.
        if self.staged.dataset_at(path).is_some()
            || (kind == StagedKind::Dataset && self.staged.has_group_at(path))
        {
            return Err(Error::EditUnsupported(
                "this session already stages an object with this name in the target group; \
                 delete it to withdraw that staging before staging another in its place",
            ));
        }
        if self.staged.deletes_hand_over(path) || !self.path_in_file(path) {
            return Ok(());
        }
        Err(Error::EditUnsupported(
            "a link with this name already exists in the target group",
        ))
    }

    /// Whether the file this session is editing holds a link at `path`, as of
    /// its committed state.
    ///
    /// Resolved against a borrowed slice where the backing can lend one, as the
    /// reader's own path resolution does: this is requested once per by-name lookup
    /// of a path a creation is staged at, so the copy a `Source` read would make
    /// per link block is worth avoiding.
    fn path_in_file(&self, path: &ObjectPathBuf) -> bool {
        match self.image.as_slice() {
            Some(data) => crate::group_v2::resolve_path_any(
                data,
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &path.as_path(),
            )
            .is_ok(),
            None => crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &path.as_path(),
            )
            .is_ok(),
        }
    }

    /// Whether this session has staged no creations at all, which is the common
    /// case on the by-name lookup path. Checked before the path is owned, so a
    /// lookup of an object already in the file returns on this flag alone.
    fn stages_no_creations(&self) -> bool {
        self.staged.groups.is_empty() && self.staged.datasets.is_empty()
    }

    /// How many times a commit has consumed this session's staged set. See the
    /// [`staged_generation`](Self::staged_generation) field for the rule, and
    /// [`crate::Error::StagingWithdrawn`] for what a handle does with it.
    pub(crate) fn staged_generation(&self) -> u64 {
        self.staged_generation
    }

    /// What a staged dataset at `path` says about itself, for the introspection
    /// a handle onto it can report before the commit writes any bytes.
    ///
    /// Cloned out of the staged [`FlatDataset`], which is the whole record of
    /// what the commit will write, so these reports cannot drift from the
    /// dataset that lands. Reports only for a creation that owns its path, on
    /// the terms [`staged_object`](Self::staged_object) sets out.
    pub(crate) fn staged_dataset_meta(&self, path: &ObjectPathBuf) -> Option<StagedMeta> {
        if self.staged_object(path)?.kind != StagedKind::Dataset {
            return None;
        }
        let fd = self.staged.dataset_at(path)?;
        Some(StagedMeta {
            datatype: fd.dt.clone(),
            dimensions: fd.ds.dimensions.clone(),
            // Reported the way `Dataset::maxshape` reports an on-disk one: a
            // maximum equal to the current shape is a fixed-shape dataset.
            maxshape: fd.ds.extensible_max_dimensions().map(<[MaxExtent]>::to_vec),
            // The rule the commit applies: chunk options or an extensible shape
            // select chunked storage, anything else contiguous.
            chunked: fd.chunk_options.is_chunked() || fd.maxshape.is_some(),
            filters: fd
                .chunk_options
                .filters
                .iter()
                .map(|f| (f.kind.filter_id(), f.optional))
                .collect(),
        })
    }

    /// The direct children `parent` gains from this session's staged creations:
    /// the groups first and then the datasets, each in the order it was staged.
    ///
    /// `parent` may itself be staged, which is what lets a listing on a
    /// not-yet-committed group report its members. Only children the session
    /// created are reported, for the reason [`staged_object`](Self::staged_object)
    /// gives.
    ///
    /// Unlike [`staged_object`](Self::staged_object) this does not read the file
    /// whether each name is already taken: the caller is enumerating a group and
    /// already holds its on-disk links, so `replaces_link` is all it needs to
    /// apply the same rule without a path resolution per child.
    pub(crate) fn staged_children(&self, parent: &ObjectPathBuf) -> Vec<StagedChild> {
        if self.stages_no_creations() {
            return Vec::new();
        }
        // Whether this group is itself a replacement staged here - its on-disk
        // links go with the object being removed, so every creation under it
        // owns its name - and which links directly under it are removed by
        // name: the two ways a creation here is a replacement. A deletion of the group that this session does *not* build
        // again is neither: the commit rejects that batch, so the file's own
        // children still own their names (see [`StagedEdits::deletes_hand_over`]).
        let base_deleted =
            self.staged.deletes_hand_over(parent) && self.staged.has_group_at(parent);
        let deleted_here: HashSet<&str> = self
            .staged
            .deletes
            .iter()
            .filter_map(|d| d.name_under(parent))
            .map(LinkNameBuf::as_str)
            .collect();

        let mut seen: HashSet<&str> = HashSet::new();
        let mut out: Vec<StagedChild> = Vec::new();
        for path in &self.staged.groups {
            if let Some(name) = path.name_under(parent) {
                let name = name.as_str();
                if seen.insert(name) {
                    out.push(StagedChild {
                        name: name.to_string(),
                        kind: StagedKind::Group,
                        replaces_link: base_deleted || deleted_here.contains(name),
                    });
                }
            }
        }
        for (p, fd) in &self.staged.datasets {
            if p == parent && seen.insert(fd.name.as_str()) {
                out.push(StagedChild {
                    name: fd.name.as_str().to_string(),
                    kind: StagedKind::Dataset,
                    replaces_link: base_deleted || deleted_here.contains(fd.name.as_str()),
                });
            }
        }
        out
    }

    /// Stage a new (empty) group at `path`, created on the next
    /// [`commit`](Self::commit). The parent must already exist or be created in
    /// the same session. Populate the group with datasets via
    /// [`stage_created_dataset`](Self::stage_created_dataset) using a path under it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is [`ObjectPathBuf::root`], which no link
    /// reaches, if a live buffered appender covers `path`, or if the name is taken on the terms
    /// [`refuse_creation_collision`](Self::refuse_creation_collision) sets out.
    pub fn create_group(&mut self, path: &ObjectPathBuf) -> Result<(), Error> {
        // Nothing links to the root group, and every file has one, so there is
        // no link here to create. Rejected here, as a deletion of the root is:
        // a creation staged at the root would mark it a node this commit builds
        // fresh, until the commit rejected the batch.
        if path.is_empty() {
            return Err(Error::EditUnsupported("cannot create the root group"));
        }
        self.refuse_if_claimed(path)?;
        self.refuse_creation_collision(path, StagedKind::Group)?;
        self.staged.push_group(path.clone());
        Ok(())
    }

    /// Stage an attribute add or replacement on a group, applied on the next
    /// [`commit`](Self::commit).
    ///
    /// `path` identifies the group to edit, and [`ObjectPathBuf::root`] the root group. The
    /// group may already exist or may be created earlier in the same session
    /// with [`create_group`](Self::create_group). Attributes - fixed-size or
    /// variable-length - are stored compactly in the rebuilt group header while
    /// they fit it, and in a fractal heap past that, on the terms
    /// [`super::plan_attr_ops`] sets out.
    pub fn set_group_attr(
        &mut self,
        path: &ObjectPathBuf,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.group_attrs.push((
            path.clone(),
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stage removal of a compact attribute from a group, applied on the next
    /// [`commit`](Self::commit).
    ///
    /// `path` identifies the group to edit, and [`ObjectPathBuf::root`] the root group. The
    /// named attribute must exist in the committed group state after any earlier
    /// staged attribute operations for the same group have been applied.
    pub fn remove_group_attr(&mut self, path: &ObjectPathBuf, name: &str) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.group_attrs.push((
            path.clone(),
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
    }

    /// Stage an attribute add or replacement on an **existing dataset**, applied on
    /// the next [`commit`](Self::commit).
    ///
    /// `path` identifies the dataset to edit. Attributes - fixed-size or
    /// variable-length - are stored in the rebuilt dataset header
    /// while they fit it, and in a fractal heap past that, on the terms
    /// [`super::plan_attr_ops`] sets out. Applying it relocates the dataset's object
    /// header (the header is rewritten and its single naming link repointed. The
    /// dataset's data and chunk index stay in place), so it is supported only when
    /// the dataset has a **single hard link**. To set attributes on a dataset being *created* in this
    /// session, use the builder's [`set_attr`](crate::DatasetBuilder::set_attr)
    /// instead.
    pub fn set_dataset_attr(
        &mut self,
        path: &ObjectPathBuf,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.dataset_attrs.push((
            path.clone(),
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stage removal of a compact attribute from an **existing dataset**, applied on
    /// the next [`commit`](Self::commit).
    ///
    /// `path` identifies the dataset to edit. The named attribute must exist in the
    /// committed dataset state after any earlier staged attribute operations for the
    /// same dataset have been applied. Like [`set_dataset_attr`](Self::set_dataset_attr)
    /// it relocates the dataset header and requires a single hard link.
    pub fn remove_dataset_attr(&mut self, path: &ObjectPathBuf, name: &str) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.dataset_attrs.push((
            path.clone(),
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
    }

    /// Stage removal of the link at `path` (the HDF5 `H5Ldelete`), applied on the
    /// next [`commit`](Self::commit). The link's object - and, for a group, its
    /// whole subtree - becomes unreachable. The bytes it occupied are returned to
    /// this session's free list (issue #21): a later commit reuses them for new
    /// objects, and if a freed run reaches
    /// end-of-file the file is truncated. Contiguous and chunked datasets (their
    /// chunk index and chunk data blocks) and whole group subtrees are all
    /// reclaimed. Dense attribute and link storage is reclaimed with the object, its fractal heap
    /// where the engine proves the heap's ownership. A global heap collection is reclaimed when
    /// only the deleted objects refer to it. Reclaim is best-effort. An object whose blocks this
    /// engine cannot enumerate exhaustively is left as dead bytes to avoid freeing a region that
    /// may still be in use, and every collection stays allocated when the engine cannot read each
    /// object that can refer to one.
    /// Freed space is reused within the open session. For a file created with
    /// `H5Pset_file_space_strategy(persist = true)` it is also recorded on disk so
    /// it survives reopen, otherwise it is forgotten
    /// on close. After reuse, an object reference to a deleted object may resolve
    /// to an unrelated object (deleting a referenced object is undefined in HDF5).
    ///
    /// The path must exist. The link's parent group must itself be editable in
    /// place (compact links, single-chunk header). The target being removed has
    /// no such restriction.
    ///
    /// # Replacing an object
    ///
    /// Deleting a path and creating a new object at that same path in one commit
    /// is a *replacement*, and is accepted (issue #305): the removal is applied
    /// before the addition, and the commit's single superblock write publishes
    /// both, so the path is occupied at every instant a crash could interrupt.
    /// The new object need not resemble the old one - a dataset may replace a
    /// group, or the reverse - and replacing a group discards its whole subtree.
    ///
    /// Everything this commit adds *below* a replaced path lands in the
    /// replacement, whatever order the calls were made in: staging
    /// `create_dataset("g/x")` and then replacing `g` puts `x` in the new group,
    /// not the one being removed. A staged edit that could only mean the
    /// *original* is rejected instead - a group under the replaced path that this
    /// commit does not itself create, an attribute set on the object being
    /// removed, a value overwrite inside it, or a [`copy`](crate::File::copy)
    /// reading from it.
    ///
    /// See [`Group::delete`](crate::Group::delete) for the public form and a
    /// worked example.
    ///
    /// A deletion may not overlap a staged change at a *different* path
    /// (deleting `/a` while adding `/a/b`, unless `/a` is itself replaced in the
    /// same commit), nor an edit to the object being removed (an attribute set
    /// on it, or a value overwrite of something inside it). Split those into
    /// separate commits.
    /// Deleting something this session **staged** and has not committed
    /// withdraws that staging instead: there is no link in the file to unlink,
    /// so the session is left as though the creation had never been made - its
    /// attributes, appends and staged children go with it. When the file *also*
    /// holds a link at the path (the creation was replacing it), the withdrawal
    /// leaves the plain deletion of the file's own object behind.
    ///
    /// The root group, [`ObjectPathBuf::root`], is rejected with [`Error::EditUnsupported`]:
    /// no link reaches it, so there is no link to remove.
    pub fn delete(&mut self, path: &ObjectPathBuf) -> Result<(), Error> {
        // The root is not linked from anywhere, so there is no link to remove  -
        // and an empty path is a prefix of every other, so a deletion staged
        // here would make every staged creation in the session look like a
        // replacement of a file object until the commit rejected the batch.
        // Rejected by name, as creating the root is.
        if path.is_empty() {
            return Err(Error::EditUnsupported(
                "cannot delete the root group; delete its members instead",
            ));
        }
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_some() || self.staged.has_group_at(path) {
            self.refuse_mid_batch()?;
            self.staged.withdraw_at(path);
            // Withdrawing is the whole deletion unless the file holds a link
            // here too - and if it does, a deletion of it may already be staged,
            // which is what made this a replacement in the first place. A second
            // one would be an overlapping deletion the commit rejects.
            if self.staged.deletes_cover(path) || !self.path_in_file(path) {
                return Ok(());
            }
        }
        self.staged.deletes.push(path.clone());
        Ok(())
    }

    /// Stage a deep copy of the object at `src` to a new link at `dst` (the HDF5
    /// `H5Ocopy`), applied on the next [`commit`](Self::commit). The source - a
    /// dataset or a whole group subtree - is duplicated: fresh copies of every
    /// object's data and header are written, internal links and the contiguous
    /// data address are repointed to the copies, and a link named by `dst`'s last
    /// component is added to `dst`'s parent group. The original is untouched.
    ///
    /// The copy reflects the file's on-disk state at commit time. `src` must
    /// exist and `dst` must not (and may not lie inside `src`). A chunked (and
    /// filtered) dataset is copied with its chunk payloads and filter pipeline
    /// preserved byte-for-byte (the index is rebuilt at the new location, so a
    /// source using a B-tree-v1 or implicit index is reproduced with an equivalent
    /// v4 index). The source subtree must otherwise be copyable in place: compact
    /// links and attributes, single-chunk headers, and a chunk index this engine
    /// can rebuild. A version 2 B-tree or a sparse/unallocated chunk grid is unsupported, so
    /// `commit` reports [`Error::EditUnsupported`].
    ///
    /// A *contiguous* dataset whose storage was never allocated is copied as the
    /// unallocated storage,
    /// which is what keeps a schema-only source from arriving materialized at the
    /// size its shape declares. A dataset storing its elements in external files
    /// (`H5Pset_external`) carries that same empty storage over data this crate
    /// does not read, and is rejected by name so the two report distinct results
    /// (issue #336).
    pub fn copy(&mut self, src: &ObjectPathBuf, dst: &ObjectPathBuf) -> Result<(), Error> {
        // Both ends matter: the source is read and the destination is written,
        // and a commit relocates headers along either path.
        self.refuse_if_claimed(src)?;
        self.refuse_if_claimed(dst)?;
        self.staged.copies.push((src.clone(), dst.clone()));
        Ok(())
    }

    /// Stage a deep copy of the object at `src` in another open file `source` to a
    /// new link at `dst` in this file - a *cross-file* HDF5 `H5Ocopy` - applied on
    /// the next [`commit`](Self::commit). Like [`copy`](Self::copy) but the source
    /// lives in a separate, independently-opened [`File`](crate::File) reader
    /// outside the file being edited.
    ///
    /// The source - a dataset or a whole group subtree - is duplicated faithfully:
    /// fresh, byte-identical copies of every object's header and data are appended
    /// to this file, internal links repointed, and a link named by `dst`'s last
    /// component added to `dst`'s parent group (which must already exist or be
    /// created earlier in this session). Both files are left otherwise untouched.
    /// the destination only changes on `commit`.
    ///
    /// Unlike the same-file [`copy`](Self::copy), the source is read **eagerly**
    /// here (the `source` borrow need not outlive the call), so this returns
    /// `Result`: the source subtree is resolved, validated, and read out before
    /// returning, and only an already-validated copy is queued for `commit`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the copy cannot be reproduced exactly
    /// in another file. Because the copy is byte-for-byte verbatim, anything that
    /// embeds a *source-file* absolute address is rejected (it would dangle here):
    /// **variable-length** or **reference** datasets and attributes (including a
    /// chunked dataset whose elements are variable-length or references, whose
    /// chunk payloads embed such addresses), and any **shared header message** (a
    /// committed datatype, or an SOHM-shared dataspace, fill value, or filter
    /// pipeline). As with [`copy`](Self::copy) a chunked/filtered source is copied
    /// with its chunk payloads and pipeline preserved (index rebuilt at the new
    /// location). The source must use compact links and attributes, single-chunk
    /// version-2 headers, and a chunk index this engine can enumerate (a
    /// version-2 B-tree, or a sparse chunk grid, is rejected). The
    /// `source` must be a buffered file ([`File::open`](crate::File::open) or
    /// [`File::from_bytes`](crate::File::from_bytes), not
    /// [`open_streaming`](crate::File::open_streaming)) using 8-byte offsets and no
    /// userblock, and `src` must exist in it and not be the root group.
    ///
    /// # Note
    ///
    /// The source is read under the mode its own file is open under, read-only for
    /// every source this copy accepts, so an object header message of a type this
    /// crate cannot name that a decoder with write access must understand is read
    /// past. The copy reproduces an object's header records verbatim, so such a
    /// message on `src`'s own header reaches the destination. `H5Ocopy` copies a
    /// message of an unknown type into the destination header the same way, since a
    /// message without a `copy_file` callback is copied raw (`H5Ocopy.c`,
    /// HDF5 2.2.0).
    pub fn copy_from(
        &mut self,
        source: &crate::reader::File,
        src: &ObjectPathBuf,
        dst: &ObjectPathBuf,
    ) -> Result<(), Error> {
        // The source bytes must be addressable: a streaming file is rejected.
        let src_data = source.in_memory_image().ok_or(Error::EditUnsupported(
            "cross-file copy requires a buffered source file (File::open or File::from_bytes), not a streaming one",
        ))?;
        let src_sb = source.superblock();
        if src_sb.offset_size != OFFSET_WIDTH || src_sb.length_size != LENGTH_WIDTH {
            return Err(Error::EditUnsupported(
                "cross-file copy requires the source file to use 8-byte offsets and lengths",
            ));
        }
        if !source.base_address().is_zero() {
            return Err(Error::EditUnsupported(
                "cross-file copy requires the source file to have no userblock (base address 0)",
            ));
        }

        if src.is_empty() {
            return Err(Error::EditUnsupported("cannot copy the root group"));
        }
        if dst.is_empty() {
            return Err(Error::EditUnsupported("copy destination path is empty"));
        }

        let src_addr = crate::group_v2::resolve_path_any(
            src_data,
            source.access_mode(),
            src_sb,
            source.widths(),
            source.addresses(),
            &src.as_path(),
        )?;
        // Read (and foreign-address-screen) the whole subtree now, while `source`
        // is borrowed. The owned tree carries every byte the commit will write. The
        // source is gated to base 0 above, so its stored addresses are absolute.
        let tree = Self::read_copy_subtree(
            &BytesSource::new(src_data),
            source.access_mode(),
            src_addr,
            0,
            true,
            source.addresses(),
            source.widths(),
        )?;
        self.refuse_if_claimed(dst)?;
        self.staged.cross_copies.push((dst.clone(), tree));
        Ok(())
    }
}
