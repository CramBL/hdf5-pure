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

impl WriteEngine {
    /// Stages a prepared dataset for creation at `path` on the next [`commit`](Self::commit).
    ///
    /// The dataset name is taken from the last component of `path`. The parent group must
    /// exist in the file or be created in the same commit. [`super::flatten_dataset`] prepares
    /// and checks the builder before it is stored in the staged set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root, collides with an existing
    /// object, or overlaps a live buffered appender. Returns the preparation error if
    /// [`super::flatten_dataset`] cannot prepare `builder`.
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

    /// Stages replacement values for an existing dataset on the next [`commit`](Self::commit).
    ///
    /// The replacement is prepared by [`super::flatten_dataset`]. Shape and datatype
    /// compatibility with the committed dataset are checked during the commit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root or overlaps a live buffered
    /// appender, or if `builder` specifies chunking, attributes, fill settings, or
    /// unresolved reference targets. Returns the preparation error if
    /// [`super::flatten_dataset`] cannot prepare `builder`.
    pub(crate) fn stage_dataset_write(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        let Some((_, leaf)) = path.split_leaf() else {
            return Err(Error::EditUnsupported("cannot overwrite the root group"));
        };
        let fd = super::flatten_dataset(builder, leaf)?;
        Self::refuse_unsupported_overwrite(&fd)?;
        self.staged.push(StagedOp::WriteDataset(path.clone(), fd));
        Ok(())
    }

    /// Stages an append to a committed dataset for the next [`commit`](Self::commit).
    ///
    /// The commit checks the dataset's append geometry and storage. A lossy partial trailing
    /// chunk is rejected at staging when its geometry can be read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a dataset creation is staged at `path` or a live buffered
    /// appender overlaps it. Returns [`Error::AppendUnsupported`] if a nonempty append would
    /// re-encode a partial trailing chunk through a lossy pipeline.
    pub(crate) fn stage_dataset_append(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_some() {
            // This handle addresses the committed object being replaced at this path.
            return Err(Error::EditUnsupported(
                "a dataset is staged at this path in the same commit, so an append through a \
                 handle onto the object the file holds there could only grow the object being \
                 replaced; append through the handle that staged the creation, or commit first",
            ));
        }
        self.refuse_lossy_partial_tail(path, &builder)?;
        self.staged
            .push(StagedOp::AppendDataset(path.clone(), builder));
        Ok(())
    }

    /// Extends a staged dataset creation with the appended elements.
    ///
    /// The caller checks that its handle addresses a pending creation. If another thread
    /// commits that creation before this call, the append is staged against the committed
    /// dataset instead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`, or if
    /// an atomic batch is open while a creation is staged there. Returns the error from
    /// [`extend_staged_dataset`](Self::extend_staged_dataset) for an incompatible pending
    /// append, or from [`refuse_lossy_partial_tail`](Self::refuse_lossy_partial_tail) for a
    /// committed dataset.
    pub(crate) fn stage_dataset_append_pending(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_none() {
            self.refuse_lossy_partial_tail(path, &builder)?;
            self.staged
                .push(StagedOp::AppendDataset(path.clone(), builder));
            return Ok(());
        }
        self.refuse_mid_batch()?;
        self.extend_staged_dataset(path, &builder)
    }

    /// Rolls back the additions made by `f` if it returns an error.
    ///
    /// Nested batches preserve the outer batch's restriction on modifying or withdrawing
    /// operations already staged. Rollback restores the operation counts and creation indexes.
    ///
    /// # Errors
    ///
    /// Returns the error from `f` after restoring the staged set to its previous mark.
    pub(crate) fn stage_atomically<R>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let mark = self.staged.mark();
        let outer = std::mem::replace(&mut self.staging_batch, true);
        let result = f(self);
        self.staging_batch = outer;
        if result.is_err() {
            self.staged.rewind(mark);
        }
        result
    }

    /// Stages an empty group for creation on the next [`commit`](Self::commit).
    ///
    /// The parent group must exist in the file or be created in the same commit. Repeated
    /// creation of the same staged group is allowed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root, overlaps a live buffered
    /// appender, or collides with a committed object or staged dataset.
    pub fn create_group(&mut self, path: &ObjectPathBuf) -> Result<(), Error> {
        if path.is_empty() {
            return Err(Error::EditUnsupported("cannot create the root group"));
        }
        self.refuse_if_claimed(path)?;
        self.refuse_creation_collision(path, StagedKind::Group)?;
        self.staged.push_group(path.clone());
        Ok(())
    }

    /// Stages an attribute addition or replacement for the next [`commit`](Self::commit).
    ///
    /// The group may be committed or staged. [`super::plan_attr_ops`] plans attribute storage
    /// when the commit rebuilds its header.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn set_group_attr(
        &mut self,
        path: &ObjectPathBuf,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditGroupAttribute(
            path.clone(),
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stages attribute removal from a group for the next [`commit`](Self::commit).
    ///
    /// The attribute must exist after earlier staged attribute operations are applied.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn remove_group_attr(&mut self, path: &ObjectPathBuf, name: &str) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditGroupAttribute(
            path.clone(),
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
    }

    /// Stages an attribute addition or replacement on a committed dataset.
    ///
    /// The next [`commit`](Self::commit) relocates the dataset header and requires a single
    /// hard link. To set attributes on a staged creation, use
    /// [`DatasetBuilder::set_attr`] before staging it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn set_dataset_attr(
        &mut self,
        path: &ObjectPathBuf,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditDatasetAttribute(
            path.clone(),
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stages attribute removal from a committed dataset.
    ///
    /// The attribute must exist after earlier staged attribute operations are applied.
    /// The next [`commit`](Self::commit) relocates the dataset header and requires a single
    /// hard link.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn remove_dataset_attr(&mut self, path: &ObjectPathBuf, name: &str) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditDatasetAttribute(
            path.clone(),
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
    }

    /// Stages link removal or withdraws an uncommitted creation at `path`.
    ///
    /// Withdrawing a creation also removes its staged descendants and edits. Existing
    /// staged deletions are preserved, including a deletion paired with a replacement.
    /// For committed objects, the next [`commit`](Self::commit) checks the deletion and
    /// reclaims storage where ownership can be established.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root, overlaps a live buffered
    /// appender, or withdraws a staged creation during an atomic batch.
    pub fn delete(&mut self, path: &ObjectPathBuf) -> Result<(), Error> {
        if path.is_empty() {
            return Err(Error::EditUnsupported(
                "cannot delete the root group; delete its members instead",
            ));
        }
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_some() || self.staged.has_group_at(path) {
            self.refuse_mid_batch()?;
            self.staged.withdraw_at(path);
            // An existing deletion remains pending after its replacement is withdrawn.
            if self.staged.deletes_cover(path) || !self.path_in_file(path) {
                return Ok(());
            }
        }
        self.staged.push(StagedOp::DeleteObject(path.clone()));
        Ok(())
    }

    /// Stages a deep copy of a committed object for the next [`commit`](Self::commit).
    ///
    /// The source is read during the commit. The destination must be available, and the
    /// source subtree must use storage the editor can reproduce.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps either path.
    pub fn copy(&mut self, src: &ObjectPathBuf, dst: &ObjectPathBuf) -> Result<(), Error> {
        self.refuse_if_claimed(src)?;
        self.refuse_if_claimed(dst)?;
        self.staged
            .push(StagedOp::CopyObject(src.clone(), dst.clone()));
        Ok(())
    }

    /// Reads a source subtree from another file and stages its copy at `dst`.
    ///
    /// The subtree is read before this call returns. The destination is written by the next
    /// [`commit`](Self::commit), which checks destination collisions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the source is streamed, uses a userblock or
    /// widths other than eight bytes, either path is the root, or a live buffered appender
    /// overlaps `dst`. Returns the read or preparation error if the source subtree cannot
    /// be resolved or reproduced without references to storage in the source file.
    pub fn copy_from(
        &mut self,
        source: &crate::reader::File,
        src: &ObjectPathBuf,
        dst: &ObjectPathBuf,
    ) -> Result<(), Error> {
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
        // The source has base address zero, so its stored addresses are absolute.
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
        self.staged
            .push(StagedOp::CopyObjectFrom(dst.clone(), tree));
        Ok(())
    }

    /// Returns whether any operation is pending for [`commit`](Self::commit).
    ///
    /// Immediate appends are excluded. A rejected commit preflight preserves the pending set.
    pub fn has_staged_edits(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Returns whether a staged operation overlaps cached append geometry for `target`.
    ///
    /// Group creation and group attribute edits are excluded because they do not relocate a
    /// descendant dataset header or release its storage.
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

    /// Returns a staged creation visible to a handle at `path`.
    ///
    /// A creation is visible when the committed path is absent or a staged deletion replaces
    /// it. A deletion alone preserves visibility of the committed object. For an ancestor
    /// deletion, each removed ancestor must also have a staged group creation.
    pub(crate) fn staged_object(&self, path: &ObjectPathBuf) -> Option<StagedObject> {
        if !self.has_staged_creations() {
            return None;
        }
        if path.is_empty() {
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
        if !replaces_link && self.path_in_file(path) {
            return None;
        }
        Some(StagedObject {
            kind,
            replaces_link,
        })
    }

    /// Returns the generation used to distinguish pending handles across commit attempts.
    ///
    /// A handle uses this counter to detect that its creation is no longer staged.
    pub(crate) fn staged_generation(&self) -> u64 {
        self.staged_generation
    }

    /// Returns metadata from the prepared dataset creation visible at `path`.
    ///
    /// Visibility follows [`staged_object`](Self::staged_object). The fields are copied from
    /// the same [`FlatDataset`] used during the commit.
    pub(crate) fn staged_dataset_meta(&self, path: &ObjectPathBuf) -> Option<StagedMeta> {
        if self.staged_object(path)?.kind != StagedKind::Dataset {
            return None;
        }
        let fd = self.staged.dataset_at(path)?;
        Some(StagedMeta {
            datatype: fd.dt.clone(),
            dimensions: fd.ds.dimensions.clone(),
            maxshape: fd.ds.extensible_max_dimensions().map(<[MaxExtent]>::to_vec),
            chunked: fd.chunk_options.is_chunked() || fd.maxshape.is_some(),
            filters: fd
                .chunk_options
                .filters
                .iter()
                .map(|f| (f.kind.filter_id(), f.optional))
                .collect(),
        })
    }

    /// Returns direct staged children in group order, followed by dataset order.
    ///
    /// Each kind preserves staging order, with repeated child names included once. The caller
    /// merges these children with committed links using [`StagedChild::replaces_link`].
    /// Committed paths are not resolved individually during this listing.
    pub(crate) fn staged_children(&self, parent: &ObjectPathBuf) -> Vec<StagedChild> {
        if !self.has_staged_creations() {
            return Vec::new();
        }
        // A rebuilt group's committed children are removed with the old group.
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

    /// Adds elements to the prepared bytes and leading dimension of a staged creation.
    ///
    /// Chunked storage and an unlimited dimension are not required. With the `provenance`
    /// feature, digest attributes are recomputed from the extended bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for mixed or mismatched element types, unresolved
    /// per-element references, a scalar or zero-byte row, an incomplete row, arithmetic
    /// overflow, or a leading dimension beyond its declared maximum.
    ///
    /// # Panics
    ///
    /// Panics if the caller has not established that a dataset is staged at `path`.
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
        // The caller finds this dataset before entering the append helper.
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
        // A leading-dimension row includes every trailing extent.
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
        #[cfg(feature = "provenance")]
        fd.rebuild_provenance();
        Ok(())
    }

    /// Rejects an append that would re-encode a lossy partial trailing chunk.
    ///
    /// Empty appends are accepted. If the committed geometry cannot be read, its validation
    /// is deferred to [`prepare_append`](Self::prepare_append).
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the committed dataset is rank one, its
    /// pipeline is lossy, and its length is not a multiple of the chunk length.
    fn refuse_lossy_partial_tail(
        &self,
        path: &ObjectPathBuf,
        builder: &AppendBuilder,
    ) -> Result<(), Error> {
        if builder.raw().is_empty() {
            return Ok(());
        }
        if self.appends_onto_a_lossy_partial_tail(path) {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
        Ok(())
    }

    /// Returns whether readable rank-one geometry requires a lossy trailing-chunk rewrite.
    ///
    /// Returns `false` if any required metadata cannot be resolved or parsed.
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

    /// Rejects changes to existing staged work during an atomic batch.
    ///
    /// [`StagedEdits::rewind`] restores counts and indexes for additions only.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a batch is open.
    fn refuse_mid_batch(&self) -> Result<(), Error> {
        if self.staging_batch {
            return Err(Error::EditUnsupported(
                "an edit that changes or withdraws an already-staged object cannot be made \
                 inside an atomic staging batch, whose undo drops additions only",
            ));
        }
        Ok(())
    }

    /// Rejects a creation whose path is occupied by another object.
    ///
    /// A staged deletion permits replacement when [`StagedEdits::deletes_hand_over`] is true.
    /// Repeated group creation is allowed, while a staged dataset excludes either kind.
    /// The callers reject the root before this check. Copy destinations are checked at commit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a committed object survives at `path`, or a
    /// staged creation conflicts with `kind`.
    fn refuse_creation_collision(
        &self,
        path: &ObjectPathBuf,
        kind: StagedKind,
    ) -> Result<(), Error> {
        if path.is_empty() {
            return Ok(());
        }
        // Check staged collisions even when a deletion permits replacing the committed link.
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

    /// Returns whether `path` resolves in the committed file image.
    ///
    /// Resolution uses a borrowed slice when the backing provides one. Read or resolution
    /// errors are reported as `false`.
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

    /// Returns whether a group or dataset creation is staged.
    fn has_staged_creations(&self) -> bool {
        !self.staged.groups.is_empty() || !self.staged.datasets.is_empty()
    }
}

/// Pending operations and indexes derived from their creation paths.
///
/// Created datasets and value replacements are prepared by [`super::flatten_dataset`].
/// During commit preflight, the engine borrows this set and restores it on rejection.
/// Batch marks record all operation counts, while creation indexes are updated with the vectors.
#[derive(Default)]
pub(super) struct StagedEdits {
    /// Parent group paths and prepared dataset creations.
    datasets: Vec<(ObjectPathBuf, FlatDataset)>,
    /// Full dataset paths and prepared replacement values.
    writes: Vec<(ObjectPathBuf, FlatDataset)>,
    /// Full dataset paths and elements to append after the committed values.
    appends: Vec<(ObjectPathBuf, AppendBuilder)>,
    groups: Vec<ObjectPathBuf>,
    group_attrs: Vec<(ObjectPathBuf, AttrOp)>,
    dataset_attrs: Vec<(ObjectPathBuf, AttrOp)>,
    deletes: Vec<ObjectPathBuf>,
    /// Same-file source and destination paths.
    copies: Vec<(ObjectPathBuf, ObjectPathBuf)>,
    /// Destination paths and subtrees read from another file at staging.
    cross_copies: Vec<(ObjectPathBuf, CopyTree)>,
    /// Full dataset creation paths mapped to their first positions in [`Self::datasets`].
    dataset_at: HashMap<ObjectPathBuf, usize>,
    /// Full group creation paths mapped to their first positions in [`Self::groups`].
    group_at: HashMap<ObjectPathBuf, usize>,
}

impl StagedEdits {
    /// Consumes the staged set after preflight and returns the remaining owned commit inputs.
    ///
    /// Other operations have been prepared into commit-local plans and are discarded.
    pub(super) fn into_apply(self) -> StagedApply {
        let Self {
            datasets,
            writes: _,
            appends: _,
            groups: _,
            group_attrs: _,
            dataset_attrs: _,
            deletes,
            copies: _,
            cross_copies,
            dataset_at: _,
            group_at: _,
        } = self;
        StagedApply {
            datasets,
            deletes,
            cross_copies,
        }
    }

    /// Returns prepared dataset creations with their parent paths, in staging order.
    pub(super) fn datasets(&self) -> &[(ObjectPathBuf, FlatDataset)] {
        &self.datasets
    }

    /// Returns prepared value replacements with their full paths, in staging order.
    pub(super) fn writes(&self) -> &[(ObjectPathBuf, FlatDataset)] {
        &self.writes
    }

    /// Returns committed-dataset appends with their full paths, in staging order.
    pub(super) fn appends(&self) -> &[(ObjectPathBuf, AppendBuilder)] {
        &self.appends
    }

    /// Returns group creation paths in staging order, including repeated paths.
    pub(super) fn groups(&self) -> &[ObjectPathBuf] {
        &self.groups
    }

    /// Returns group attribute operations with their target paths, in staging order.
    pub(super) fn group_attrs(&self) -> &[(ObjectPathBuf, AttrOp)] {
        &self.group_attrs
    }

    /// Returns dataset attribute operations with their target paths, in staging order.
    pub(super) fn dataset_attrs(&self) -> &[(ObjectPathBuf, AttrOp)] {
        &self.dataset_attrs
    }

    /// Returns link removal paths in staging order.
    pub(super) fn deletes(&self) -> &[ObjectPathBuf] {
        &self.deletes
    }

    /// Returns same-file source and destination paths in staging order.
    pub(super) fn copies(&self) -> &[(ObjectPathBuf, ObjectPathBuf)] {
        &self.copies
    }

    /// Returns cross-file destination paths and prepared subtrees in staging order.
    pub(super) fn cross_copies(&self) -> &[(ObjectPathBuf, CopyTree)] {
        &self.cross_copies
    }

    /// Classifies the pending operations for selection of the commit path.
    ///
    /// Dataset value writes, appends, and dataset attribute edits can use the dataset-only
    /// commit path when their prepared plans do not require relocation.
    pub(super) fn scope(&self) -> StagedScope {
        let Self {
            datasets,
            writes: _,
            appends: _,
            groups,
            group_attrs,
            dataset_attrs: _,
            deletes,
            copies,
            cross_copies,
            dataset_at: _,
            group_at: _,
        } = self;
        if datasets.is_empty()
            && groups.is_empty()
            && group_attrs.is_empty()
            && deletes.is_empty()
            && copies.is_empty()
            && cross_copies.is_empty()
        {
            StagedScope::Datasets
        } else {
            StagedScope::Tree
        }
    }

    /// Returns whether every staged operation count is zero.
    pub(super) fn is_empty(&self) -> bool {
        self.mark() == StagedMark::default()
    }

    /// Stages a prepared dataset creation and indexes its full path.
    fn push_dataset(&mut self, parent: ObjectPathBuf, fd: FlatDataset) {
        self.push(StagedOp::CreateDataset(parent, fd));
    }

    /// Stages a group creation and indexes its full path.
    fn push_group(&mut self, path: ObjectPathBuf) {
        self.push(StagedOp::CreateGroup(path));
    }

    /// Records one typed operation and updates any creation index.
    fn push(&mut self, operation: StagedOp) {
        match operation {
            StagedOp::AppendDataset(path, builder) => self.appends.push((path, builder)),
            StagedOp::CopyObject(src, dst) => self.copies.push((src, dst)),
            StagedOp::CopyObjectFrom(dst, tree) => self.cross_copies.push((dst, tree)),
            StagedOp::CreateDataset(parent, dataset) => {
                self.dataset_at
                    .entry(parent.join(&dataset.name))
                    .or_insert(self.datasets.len());
                self.datasets.push((parent, dataset));
            }
            StagedOp::CreateGroup(path) => {
                self.group_at
                    .entry(path.clone())
                    .or_insert(self.groups.len());
                self.groups.push(path);
            }
            StagedOp::DeleteObject(path) => self.deletes.push(path),
            StagedOp::EditDatasetAttribute(path, operation) => {
                self.dataset_attrs.push((path, operation));
            }
            StagedOp::EditGroupAttribute(path, operation) => {
                self.group_attrs.push((path, operation));
            }
            StagedOp::WriteDataset(path, dataset) => self.writes.push((path, dataset)),
        }
    }

    /// Removes additions after `mark` and restores the creation indexes.
    ///
    /// Existing staged records must remain unchanged between the mark and rollback.
    /// [`WriteEngine::refuse_mid_batch`] enforces this for pending dataset appends and
    /// withdrawals.
    fn rewind(&mut self, mark: StagedMark) {
        for i in mark.datasets..self.datasets.len() {
            let key = {
                let (parent, fd) = &self.datasets[i];
                parent.join(&fd.name)
            };
            // Repeated creations keep the index of their first occurrence.
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

    /// Returns the prepared dataset at `path` after checking its indexed position.
    fn dataset_at_mut(&mut self, path: &ObjectPathBuf) -> Option<&mut FlatDataset> {
        let i = self.dataset_position(path)?;
        self.datasets.get_mut(i).map(|(_, fd)| fd)
    }

    /// Removes creations and edits at or below `path`, preserving staged deletions.
    ///
    /// Returns whether any dataset or group creation was removed. Indexes are rebuilt after
    /// removing a creation because subsequent vector positions may have changed.
    fn withdraw_at(&mut self, path: &ObjectPathBuf) -> bool {
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

    /// Rebuilds creation indexes from the staged vectors, preserving each first occurrence.
    fn reindex(&mut self) {
        let Self {
            datasets,
            writes: _,
            appends: _,
            groups,
            group_attrs: _,
            dataset_attrs: _,
            deletes: _,
            copies: _,
            cross_copies: _,
            dataset_at,
            group_at,
        } = self;
        dataset_at.clear();
        group_at.clear();
        for (i, (parent, fd)) in datasets.iter().enumerate() {
            dataset_at.entry(parent.join(&fd.name)).or_insert(i);
        }
        for (i, path) in groups.iter().enumerate() {
            group_at.entry(path.clone()).or_insert(i);
        }
    }

    /// Records every operation count for append-only batch rollback.
    ///
    /// Exhaustive destructuring requires every staged field to be classified here.
    fn mark(&self) -> StagedMark {
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

    /// Returns the prepared dataset at `path` after checking its indexed position.
    fn dataset_at(&self, path: &ObjectPathBuf) -> Option<&FlatDataset> {
        let i = self.dataset_position(path)?;
        self.datasets.get(i).map(|(_, fd)| fd)
    }

    /// Returns whether the indexed group position contains `path`.
    fn has_group_at(&self, path: &ObjectPathBuf) -> bool {
        match self.group_at.get(path) {
            Some(&i) => self.groups.get(i).is_some_and(|p| p == path),
            None => false,
        }
    }

    /// Returns a dataset's indexed position after checking its stored path.
    ///
    /// An out-of-range or mismatched index produces `None`.
    fn dataset_position(&self, path: &ObjectPathBuf) -> Option<usize> {
        let i = *self.dataset_at.get(path)?;
        let (parent, fd) = self.datasets.get(i)?;
        path.covers_child(parent, &fd.name).then_some(i)
    }

    /// Returns whether a staged deletion targets `path` or one of its ancestors.
    fn deletes_cover(&self, path: &ObjectPathBuf) -> bool {
        self.deletes.iter().any(|d| path.starts_with(d))
    }

    /// Returns whether a deletion permits a staged creation to replace `path`.
    ///
    /// An ancestor deletion permits replacement only if every removed group between that
    /// ancestor and `path` is recreated in this session. An exact deletion requires no
    /// ancestor recreation.
    fn deletes_hand_over(&self, path: &ObjectPathBuf) -> bool {
        self.deletes.iter().any(|d| {
            path.starts_with(d) && (d.len()..path.len()).all(|n| self.has_group_at(&path.prefix(n)))
        })
    }
}

/// The object kind of a pending creation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StagedKind {
    /// A dataset prepared from a builder.
    Dataset,
    /// A group to be created during commit.
    Group,
}

/// A pending creation visible at a path through [`WriteEngine::staged_object`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StagedObject {
    /// The kind of the pending creation.
    pub(crate) kind: StagedKind,
    /// Whether staged deletions permit replacement of a committed link at this path.
    pub(crate) replaces_link: bool,
}

/// A direct child to merge into a group's committed listing.
pub(crate) struct StagedChild {
    /// The final path component of the child.
    pub(crate) name: String,
    /// The kind of the pending creation.
    pub(crate) kind: StagedKind,
    /// Whether the caller must replace the committed child with this creation.
    pub(crate) replaces_link: bool,
}

/// Metadata copied from a prepared dataset creation for pending-handle queries.
pub(crate) struct StagedMeta {
    /// The prepared element datatype.
    pub(crate) datatype: Datatype,
    /// The current shape, including any append to the pending creation.
    pub(crate) dimensions: Vec<u64>,
    /// The maximum shape, or `None` for a fixed shape as in [`crate::Dataset::maxshape`].
    pub(crate) maxshape: Option<Vec<MaxExtent>>,
    /// Whether the prepared options select chunked storage.
    pub(crate) chunked: bool,
    /// Filter identifiers and optional flags in pipeline order.
    pub(crate) filters: Vec<(FilterId, bool)>,
}

/// An operation accepted by the staged-set mutation interface.
enum StagedOp {
    AppendDataset(ObjectPathBuf, AppendBuilder),
    CopyObject(ObjectPathBuf, ObjectPathBuf),
    CopyObjectFrom(ObjectPathBuf, CopyTree),
    /// The parent group path and prepared dataset, whose name is its final path component.
    CreateDataset(ObjectPathBuf, FlatDataset),
    CreateGroup(ObjectPathBuf),
    DeleteObject(ObjectPathBuf),
    EditDatasetAttribute(ObjectPathBuf, AttrOp),
    EditGroupAttribute(ObjectPathBuf, AttrOp),
    WriteDataset(ObjectPathBuf, FlatDataset),
}

/// Owned inputs not already represented by the prepared commit plans.
pub(super) struct StagedApply {
    /// Parent paths and prepared datasets to place.
    pub(super) datasets: Vec<(ObjectPathBuf, FlatDataset)>,
    /// Removed links whose objects are candidates for storage reclamation.
    pub(super) deletes: Vec<ObjectPathBuf>,
    /// Destination paths and prepared subtrees to place.
    pub(super) cross_copies: Vec<(ObjectPathBuf, CopyTree)>,
}

/// The operation scope used to select a commit path after dataset preparation.
#[derive(Eq, PartialEq)]
pub(super) enum StagedScope {
    /// Only value writes, dataset appends, or dataset attribute edits are staged.
    Datasets,
    /// Operations requiring tree reconstruction are staged.
    Tree,
}

/// Operation counts recorded before an append-only staging batch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct StagedMark {
    datasets: usize,
    writes: usize,
    appends: usize,
    groups: usize,
    group_attrs: usize,
    dataset_attrs: usize,
    deletes: usize,
    copies: usize,
    cross_copies: usize,
}

#[cfg(test)]
impl StagedEdits {
    /// Returns the operation counts for equality checks in editor tests.
    pub(super) fn snapshot(&self) -> impl Eq + use<> {
        self.mark()
    }
}

#[cfg(test)]
mod tests;
