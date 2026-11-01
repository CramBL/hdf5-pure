//! Object placement and relocation for prepared staged edits.
//!
//! Application owns staged dataset creations, copies, and replacement buffers. It writes
//! replacement content before publication and journals writes to existing value storage for
//! rollback on failure.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{AllocationClass, FreeClass, PageType};

use super::plan::{
    self, ChildLink, CommitPlan, CopyDestination, GroupHeaders, InPlaceWrite, Node,
    OverwritePayloads, PreparedCommit, PreparedGroup, PreparedReference, PreparedTree,
    ReferenceTarget, RelocatedDataset, ReplacedHeader,
};
use super::publish::AppliedTree;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::dataset::{FlatDataset, MovingWrite};
use crate::edit::group::{self, LinkCreationOrder};
use crate::edit::object_header;
use crate::edit::retirement::RootRetirement;
use crate::edit::staging::StagedApply;
use crate::edit::{self, WriteEngine};
use crate::error::{Error, FormatError};
use crate::object_path::ObjectPathBuf;
use crate::source::Source;
use crate::type_builders;

impl WriteEngine {
    /// Writes prepared edits after consuming the staged payloads.
    ///
    /// Tree edits publish a replacement root. An in-place-only edit ends with the content
    /// barrier. A failure consumes the application inputs in either case.
    ///
    /// # Errors
    ///
    /// Returns the placement, serialization, write, synchronization, or publication error.
    pub(super) fn apply_commit(&mut self, prepared: PreparedCommit) -> Result<(), Error> {
        let (plan, staged, mut overwrites) = prepared.into_parts();
        match plan {
            CommitPlan::InPlace(writes) => {
                drop(staged);
                self.apply_inplace_writes(&mut overwrites, writes)?;
                self.barrier()
            }
            CommitPlan::Tree(tree) => {
                let applied = self.apply_tree(&mut overwrites, tree, staged)?;
                self.publish_tree(applied)
            }
        }
    }

    /// Writes encoded output or original replacement buffers and journals prior values.
    ///
    /// # Errors
    ///
    /// Returns errors from [`resolve_overwrite_bytes`](Self::resolve_overwrite_bytes) or
    /// [`write_inplace_journaled`](Self::write_inplace_journaled).
    fn apply_inplace_writes(
        &mut self,
        overwrites: &mut OverwritePayloads,
        writes: Vec<InPlaceWrite>,
    ) -> Result<(), Error> {
        for write in writes {
            let (address, bytes) = match write {
                InPlaceWrite::Encoded { writes, source: _ } => {
                    for (address, bytes) in writes {
                        self.write_inplace_journaled(address, &bytes)?;
                    }
                    continue;
                }
                InPlaceWrite::Replacement { address, source } => (
                    address,
                    self.resolve_overwrite_bytes(overwrites.take(source))?,
                ),
            };
            self.write_inplace_journaled(address, &bytes)?;
        }
        Ok(())
    }

    /// Places replacement objects and records the addresses needed for publication.
    ///
    /// # Errors
    ///
    /// Returns the reclamation, placement, serialization, or write error.
    fn apply_tree(
        &mut self,
        overwrites: &mut OverwritePayloads,
        tree: PreparedTree,
        staged: StagedApply,
    ) -> Result<AppliedTree, Error> {
        let PreparedTree {
            mut groups,
            headers,
            cross_copies,
            inplace_writes,
            deleted_free,
            deleted_collections,
            mut references,
        } = tree;
        let GroupHeaders {
            keys: order,
            superseded_addrs,
            mut children,
        } = headers;
        let StagedApply {
            datasets,
            cross_copies: copies,
        } = staged;
        let mut flat = plan::group_by_parent(datasets);
        debug_assert_eq!(
            cross_copies.len(),
            copies.len(),
            "prepared copies match the consumed staged set"
        );
        for (CopyDestination { parent, leaf }, (_, tree)) in cross_copies.into_iter().zip(copies) {
            groups
                .get_mut(&parent)
                .ok_or_else(|| FormatError::Internal("a prepared copy has no parent group".into()))?
                .prepared
                .copies
                .push((leaf, tree));
        }
        let mut to_free = self.prepare_reclamation(&groups, deleted_free)?;
        let mut tail_allocations = Vec::new();
        let base = self.context.addresses();
        let mut path_addr: BTreeMap<ObjectPathBuf, u64> = BTreeMap::new();
        let mut relocations: BTreeMap<u64, u64> = BTreeMap::new();
        for key in &order {
            let PreparedGroup {
                mut region,
                dense_links,
                replaces: _,
                deletes,
                copies,
                writes,
                attrs,
            } = groups
                .remove(key)
                .ok_or_else(|| {
                    FormatError::Internal("a prepared group has already been placed".into())
                })?
                .into_prepared();
            for name in &deletes {
                region = group::remove_link_from_region(&region, name.as_str())?;
            }

            let mut link_order = LinkCreationOrder::for_region(&region)?;

            for (leaf, tree) in copies {
                let root = self.write_copy_subtree(&tree)?;
                region.push_link(
                    leaf.as_str(),
                    base.relative(PhysicalAddress::new(root))?,
                    link_order.take()?,
                );
            }

            let mut group_datasets: Vec<FlatDataset> =
                flat.remove(key).into_iter().flatten().collect();
            group_datasets.sort_by_key(|fd| fd.storage.reference_targets().is_some());
            for mut fd in group_datasets {
                let dataset_path = key.join(&fd.name);
                self.resolve_dataset_attributes(&mut fd)?;
                if fd.storage.take_reference_targets().is_some()
                    && let Some(patches) = references.remove(&dataset_path)
                {
                    for PreparedReference {
                        byte_offset,
                        target,
                    } in patches
                    {
                        let address = match target {
                            ReferenceTarget::Existing(address) => address,
                            ReferenceTarget::Placed(path) => base.relative(
                                PhysicalAddress::new(*path_addr.get(&path).ok_or_else(|| {
                                    FormatError::Internal(
                                        "a prepared reference target has not been placed".into(),
                                    )
                                })?),
                            )?,
                        };
                        type_builders::write_reference_address(
                            fd.storage.raw_mut()?,
                            byte_offset,
                            address,
                        );
                    }
                }
                let oh_addr = self.write_dataset(&mut fd)?;
                region.push_link(
                    fd.name.as_str(),
                    base.relative(PhysicalAddress::new(oh_addr))?,
                    link_order.take()?,
                );
                path_addr.insert(dataset_path, oh_addr);
            }

            for RelocatedDataset {
                link,
                old_header,
                mut write,
            } in writes
            {
                let tail_allocation = if let MovingWrite::AppendedChunks {
                    old_tail_allocation,
                    ..
                } = &mut write
                {
                    old_tail_allocation.take()
                } else {
                    None
                };
                let new_header = self.write_moving(overwrites, write)?;
                tail_allocations.extend(tail_allocation);
                group::patch_link_target(
                    &mut region,
                    link.as_str(),
                    base.relative(PhysicalAddress::new(new_header))?,
                )?;
                relocations.insert(old_header, new_header);
            }

            for child in children.remove(key).into_iter().flatten() {
                match child {
                    ChildLink::Insert { path, leaf } => {
                        region.push_link(
                            leaf.as_str(),
                            base.relative(PhysicalAddress::new(path_addr[&path]))?,
                            link_order.take()?,
                        );
                    }
                    ChildLink::Replace { path, leaf } => {
                        group::patch_link_target(
                            &mut region,
                            leaf.as_str(),
                            base.relative(PhysicalAddress::new(path_addr[&path]))?,
                        )?;
                    }
                }
            }
            link_order.record(&mut region)?;

            self.place_edited_attrs(&mut region, attrs)?;
            self.place_group_links(&mut region, dense_links)?;

            let oh = object_header::build_v2_object_header(&region)?;
            let addr = self.alloc_or_append_typed(&oh, AllocationClass::Metadata)?;
            path_addr.insert(key.clone(), addr);
        }

        self.apply_inplace_writes(overwrites, inplace_writes)?;

        for (path, old) in superseded_addrs {
            if let Some(&new) = path_addr.get(&path) {
                relocations.insert(old, new);
            }
        }
        let new_root = path_addr[&ObjectPathBuf::root()];
        to_free.extend(
            AnalysisPhase::prospective(self, PhysicalAddress::new(new_root))
                .unreferenced_collection_spans(deleted_collections),
        );
        to_free.extend(
            self.superseded_heaps
                .drain(..)
                .map(|range| (range, FreeClass::Page(PageType::Raw))),
        );
        edit::retain_disjoint_in_bounds(&mut to_free, self.image.len());
        Ok(AppliedTree {
            retirement: RootRetirement::new(PhysicalAddress::new(new_root), tail_allocations),
            freed: to_free,
            relocations,
        })
    }
    /// Enumerates storage superseded by prepared group and dataset replacements.
    ///
    /// The returned spans remain allocated until publication. Storage whose ownership
    /// cannot be established is excluded.
    ///
    /// # Errors
    ///
    /// Returns an address-conversion error if an object-header span cannot be classified, or
    /// [`Error::EditUnsupported`] if a Link Info record extends past the gathered region.
    fn prepare_reclamation(
        &self,
        groups: &BTreeMap<ObjectPathBuf, Node>,
        deleted_free: Vec<(FileRange, FreeClass)>,
    ) -> Result<Vec<(FileRange, FreeClass)>, Error> {
        let base = self.context.addresses();
        let mut to_free: Vec<(FileRange, FreeClass)> = deleted_free;

        for node in groups.values() {
            let Some(replaced) = &node.prepared.replaces else {
                continue;
            };
            let a = replaced.address();
            if let Ok(spans) = self.oh_chunk_spans(a) {
                if let Ok(region) = Self::gather_oh_messages(&self.image(), a, base) {
                    let attributes = self.dense_attribute_storage_spans(&region, &spans);
                    if let Some((_, _, info)) = group::find_link_info(&region)?
                        && let Some(storage) =
                            self.dense_group_storage_spans(&spans, attributes.as_deref(), &info)
                    {
                        to_free.extend(storage);
                    }
                    if matches!(replaced, ReplacedHeader::DenseAttributes(_))
                        && let Some(storage) = attributes
                    {
                        to_free.extend(storage);
                    }
                }
                to_free.extend(self.object_header_free_spans(&spans)?);
            }
        }

        for node in groups.values() {
            for RelocatedDataset {
                link: _,
                old_header: old_oh,
                write: mw,
            } in &node.prepared.writes
            {
                match mw {
                    MovingWrite::Contiguous {
                        old_extent: Some(extent),
                        ..
                    } => to_free.push((*extent, FreeClass::Page(PageType::Raw))),
                    MovingWrite::Chunked { old_addr, .. } => {
                        if let Some(spans) = self.chunked_storage_spans(*old_addr) {
                            to_free.extend(spans);
                        }
                    }
                    MovingWrite::AppendedChunks {
                        old_addr,
                        old_tail_allocation,
                        kept_chunks,
                        ..
                    } => {
                        if let Some(spans) = self.chunked_index_spans(*old_addr) {
                            let data: Vec<FileRange> = kept_chunks
                                .iter()
                                .filter_map(|c| {
                                    FileRange::new(base.resolve(c.address).ok()?, c.stored_size)
                                        .ok()
                                })
                                .chain(old_tail_allocation.iter().map(|allocation| {
                                    let range = allocation.range();
                                    // Free-section construction checks these same endpoints.
                                    FileRange::new(range.start(), range.length()).unwrap()
                                }))
                                .collect();
                            let class = if self.index_is_provably_raw(&data, &spans) {
                                FreeClass::Page(PageType::Raw)
                            } else {
                                FreeClass::Dead
                            };
                            to_free.extend(spans.into_iter().map(|range| (range, class)));
                        }
                    }
                    _ => {}
                }
                if let Ok(spans) = self.oh_chunk_spans(*old_oh) {
                    if matches!(
                        mw,
                        MovingWrite::AttrEdit {
                            replaces_dense_storage: true,
                            ..
                        }
                    ) && let Ok(region) =
                        Self::gather_oh_messages(&self.image(), *old_oh, self.context.addresses())
                        && let Some(storage) = self.dense_attribute_storage_spans(&region, &spans)
                    {
                        to_free.extend(storage);
                    }
                    to_free.extend(self.object_header_free_spans(&spans)?);
                }
            }
        }

        edit::retain_disjoint_in_bounds(&mut to_free, self.image.len());

        Ok(to_free)
    }
}
