//! Object placement and relocation for prepared staged edits.
//!
//! Application owns the staged dataset and copy payloads. It writes replacement content before
//! publication and journals writes to existing value storage for rollback on failure.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_space::__private::{AllocationClass, FreeClass, PageType};

use super::plan::{
    self, ChildLink, CommitPlan, CopyDestination, GroupHeaders, Node, PreparedGroup,
    PreparedReference, PreparedTree, ReferenceTarget, RelocatedDataset, ReplacedHeader,
};
use super::publish::AppliedTree;
use crate::edit::dataset::{FlatDataset, MovingWrite};
use crate::edit::group::{self, LinkCreationOrder};
use crate::edit::object_header;
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
    pub(super) fn apply_commit(
        &mut self,
        plan: CommitPlan,
        staged: StagedApply,
    ) -> Result<(), Error> {
        match plan {
            CommitPlan::InPlace(writes) => {
                drop(staged);
                self.apply_inplace_writes(&writes)?;
                self.barrier()
            }
            CommitPlan::Tree(tree) => {
                let applied = self.apply_tree(tree, staged)?;
                self.publish_tree(applied)
            }
        }
    }

    fn apply_inplace_writes(&mut self, writes: &[(u64, Vec<u8>)]) -> Result<(), Error> {
        for (address, bytes) in writes {
            self.write_inplace_journaled(*address, bytes)?;
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
                region.push_link(leaf.as_str(), base.relative(root)?, link_order.take()?);
            }

            let mut group_datasets: Vec<FlatDataset> =
                flat.remove(key).into_iter().flatten().collect();
            group_datasets.sort_by_key(|fd| fd.reference_targets.is_some());
            for mut fd in group_datasets {
                let dataset_path = key.join(&fd.name);
                self.resolve_dataset_attributes(&mut fd)?;
                if fd.reference_targets.take().is_some()
                    && let Some(patches) = references.remove(&dataset_path)
                {
                    for PreparedReference {
                        byte_offset,
                        target,
                    } in patches
                    {
                        let address = match target {
                            ReferenceTarget::Existing(address) => address,
                            ReferenceTarget::Placed(path) => {
                                base.relative(*path_addr.get(&path).ok_or_else(|| {
                                    FormatError::Internal(
                                        "a prepared reference target has not been placed".into(),
                                    )
                                })?)?
                            }
                        };
                        type_builders::write_reference_address(&mut fd.raw, byte_offset, address);
                    }
                }
                let oh_addr = self.write_dataset(&mut fd)?;
                region.push_link(
                    fd.name.as_str(),
                    base.relative(oh_addr)?,
                    link_order.take()?,
                );
                path_addr.insert(dataset_path, oh_addr);
            }

            for RelocatedDataset {
                link,
                old_header,
                write,
            } in writes
            {
                let new_header = self.write_moving(&write)?;
                group::patch_link_target(&mut region, link.as_str(), base.relative(new_header)?)?;
                relocations.insert(old_header, new_header);
            }

            for child in children.remove(key).into_iter().flatten() {
                match child {
                    ChildLink::Insert { path, leaf } => {
                        region.push_link(
                            leaf.as_str(),
                            base.relative(path_addr[&path])?,
                            link_order.take()?,
                        );
                    }
                    ChildLink::Replace { path, leaf } => {
                        group::patch_link_target(
                            &mut region,
                            leaf.as_str(),
                            base.relative(path_addr[&path])?,
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

        for (data_addr, bytes) in &inplace_writes {
            self.write_inplace_journaled(*data_addr, bytes)?;
        }

        for (path, old) in superseded_addrs {
            if let Some(&new) = path_addr.get(&path) {
                relocations.insert(old, new);
            }
        }
        let new_root = path_addr[&ObjectPathBuf::root()];
        to_free.extend(self.unreferenced_collection_spans(new_root, deleted_collections));
        to_free.extend(
            self.superseded_heaps
                .drain(..)
                .map(|(address, len)| (address, len, FreeClass::Page(PageType::Raw))),
        );
        edit::retain_disjoint_in_bounds(&mut to_free, self.image.len());
        Ok(AppliedTree {
            root: PhysicalAddress::new(new_root),
            freed: self.stored_spans(to_free)?,
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
        deleted_free: Vec<(u64, u64, FreeClass)>,
    ) -> Result<Vec<(u64, u64, FreeClass)>, Error> {
        let base = self.context.addresses();
        let mut to_free: Vec<(u64, u64, FreeClass)> = deleted_free;

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
                    } => to_free.push((extent.0, extent.1, FreeClass::Page(PageType::Raw))),
                    MovingWrite::Chunked { old_addr, .. } => {
                        if let Some(spans) = self.chunked_storage_spans(*old_addr) {
                            to_free.extend(spans);
                        }
                    }
                    MovingWrite::AppendedChunks {
                        old_addr,
                        old_tail_extent,
                        kept_chunks,
                        ..
                    } => {
                        if let Some(spans) = self.chunked_index_spans(*old_addr) {
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
                            to_free.push((ext.0, ext.1, FreeClass::Page(PageType::Raw)));
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
