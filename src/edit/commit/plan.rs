//! Prepared edits, dirty-group graphs, and reference targets for staged commits.
//!
//! Planning reads the current file and borrows the staged inputs. The prepared operations
//! contain the structural decisions needed during placement, where allocation and I/O can fail.

use std::collections::{BTreeMap, HashMap};

use hdf5_pure_space::__private::FreeClass;

use crate::access_mode::AccessMode;
use crate::address::{AddressContext, StoredAddress, StoredAddressExt};
use crate::datatype;
use crate::edit::attributes::{self, AttrOp, EditedAttrs};
use crate::edit::context::FileContext;
use crate::edit::copy::{self, CopyTree};
use crate::edit::dataset::{FlatDataset, MovingWrite, WritePlan};
use crate::edit::group;
use crate::edit::reclaim::DeletedCollections;
use crate::edit::staging::{StagedEdits, StagedScope};
use crate::edit::{self, OhRegion};
use crate::edit::{MAX_LINK_GRAPH_NODES, UNDEF, WriteEngine};
use crate::edit::{REFERENCE_INTO_RECLAIMED_SPACE, REFERENCE_TO_A_MOVED_OBJECT};
use crate::error::Error;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::source::{BaseOffsetSource, Source};
use crate::type_builders::ObjectRefTarget;

/// Stores prepared writes before application consumes the staged inputs.
pub(super) enum CommitPlan {
    /// Existing value ranges to overwrite without publishing a replacement root.
    InPlace(Vec<(u64, Vec<u8>)>),
    /// Object placements and link edits to publish through a replacement root.
    Tree(PreparedTree),
}

/// Collects existing dataset edits and the ownership checks needed for relocation.
struct DatasetEdits {
    inplace_writes: Vec<(u64, Vec<u8>)>,
    moving_writes: Vec<DatasetRelocation>,
    write_targets: Vec<ObjectPathBuf>,
    incoming_links: Option<Option<HashMap<u64, u32>>>,
}

/// Associates a relocating dataset edit with its parent group.
struct DatasetRelocation {
    parent: ObjectPathBuf,
    edit: RelocatedDataset,
}

/// Stores a dataset replacement with its child-link name and prior header address.
pub(super) struct RelocatedDataset {
    pub(super) link: LinkNameBuf,
    /// Absolute address of the header being replaced.
    pub(super) old_header: u64,
    pub(super) write: MovingWrite,
}

/// Collects affected groups and paths used to reject conflicting edits.
struct DirtyGroups {
    nodes: BTreeMap<ObjectPathBuf, Node>,
    add_targets: Vec<ObjectPathBuf>,
    deleted_addrs: Vec<u64>,
    cross_copies: Vec<CopyDestination>,
}

/// Stores group placement order, superseded headers, and prepared child-link edits.
pub(super) struct GroupHeaders {
    /// Group paths, sorted deepest-first before application.
    pub(super) keys: Vec<ObjectPathBuf>,
    /// Group paths and absolute header addresses, in path order.
    pub(super) superseded_addrs: Vec<(ObjectPathBuf, u64)>,
    pub(super) children: BTreeMap<ObjectPathBuf, Vec<ChildLink>>,
}

/// Stores checked tree edits and reclamation records before object placement.
///
/// Dataset and cross-file-copy payloads remain in the staged set until application begins.
pub(super) struct PreparedTree {
    pub(super) groups: BTreeMap<ObjectPathBuf, Node>,
    pub(super) headers: GroupHeaders,
    pub(super) cross_copies: Vec<CopyDestination>,
    pub(super) inplace_writes: Vec<(u64, Vec<u8>)>,
    /// Absolute spans whose ownership was established while planning deletions.
    pub(super) deleted_free: Vec<(u64, u64, FreeClass)>,
    pub(super) deleted_collections: DeletedCollections,
    /// Prepared element-reference patches, indexed by the containing dataset's path.
    pub(super) references: BTreeMap<ObjectPathBuf, Vec<PreparedReference>>,
}

/// Stores the object-header region and operations needed to rebuild one group.
#[derive(Default)]
pub(super) struct PreparedGroup {
    pub(super) region: OhRegion,
    pub(super) replaces: Option<ReplacedHeader>,
    pub(super) deletes: Vec<LinkNameBuf>,
    pub(super) copies: Vec<(LinkNameBuf, CopyTree)>,
    pub(super) writes: Vec<RelocatedDataset>,
    pub(super) attrs: EditedAttrs,
}

/// Identifies a superseded group header and any dense attribute storage to reclaim.
pub(super) enum ReplacedHeader {
    /// The absolute header address of a group whose dense attributes are also replaced.
    DenseAttributes(u64),
    /// The absolute header address, without separate dense-attribute reclamation.
    Header(u64),
}

impl ReplacedHeader {
    /// Returns the absolute address of the superseded header.
    pub(super) fn address(&self) -> u64 {
        match self {
            Self::DenseAttributes(address) | Self::Header(address) => *address,
        }
    }
}

/// Distinguishes a new child link from a link to a relocated group.
pub(super) enum ChildLink {
    /// Adds a link to a newly created child group.
    Insert {
        path: ObjectPathBuf,
        leaf: LinkNameBuf,
    },
    /// Repoints an existing link to its relocated child group.
    Replace {
        path: ObjectPathBuf,
        leaf: LinkNameBuf,
    },
}

/// Stores screened reference targets and storage affected by deletions.
struct TreeReferences {
    deleted_free: Vec<(u64, u64, FreeClass)>,
    deleted_collections: DeletedCollections,
    references: BTreeMap<ObjectPathBuf, Vec<PreparedReference>>,
}

/// Stores a checked destination for a staged cross-file copy.
pub(super) struct CopyDestination {
    pub(super) parent: ObjectPathBuf,
    pub(super) leaf: LinkNameBuf,
}

/// Associates an element's byte offset with its checked object-reference target.
pub(super) struct PreparedReference {
    pub(super) byte_offset: usize,
    pub(super) target: ReferenceTarget,
}

/// Distinguishes a stored address from a path assigned an address during this commit.
pub(super) enum ReferenceTarget {
    /// An address retained from the current file, including null or undefined values.
    Existing(StoredAddress),
    /// An object placed before the referring dataset in the commit's write order.
    Placed(ObjectPathBuf),
}

impl WriteEngine {
    /// Prepares edits while retaining the staged inputs for retry on rejection.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] or [`Error::AppendUnsupported`] if an edit cannot
    /// be reproduced or conflicts with another edit. Returns the read, parsing, or preparation
    /// error if the current file or a staged operation cannot be inspected.
    pub(super) fn plan_commit(&mut self, staged: &StagedEdits) -> Result<CommitPlan, Error> {
        let mut edits = DatasetEdits {
            inplace_writes: Vec::new(),
            moving_writes: Vec::new(),
            write_targets: Vec::new(),
            incoming_links: None,
        };
        self.prepare_value_writes(staged, &mut edits)?;
        self.prepare_appends(staged, &mut edits)?;
        self.prepare_dataset_attributes(staged, &mut edits)?;
        let DatasetEdits {
            inplace_writes,
            moving_writes,
            write_targets,
            mut incoming_links,
        } = edits;
        if moving_writes.is_empty() && staged.scope() == StagedScope::Datasets {
            return Ok(CommitPlan::InPlace(inplace_writes));
        }
        let mut moved_headers = moving_writes
            .iter()
            .map(|relocation| relocation.edit.old_header)
            .collect::<Vec<_>>();
        let DirtyGroups {
            mut nodes,
            add_targets,
            mut deleted_addrs,
            cross_copies,
        } = self.plan_dirty_groups(staged, moving_writes, &write_targets)?;
        let headers = self.prepare_group_headers(staged, &mut nodes, &mut incoming_links)?;
        moved_headers.extend(headers.superseded_addrs.iter().map(|&(_, address)| address));
        let TreeReferences {
            deleted_free,
            deleted_collections,
            references,
        } = self.prepare_tree_references(
            staged,
            &nodes,
            &add_targets,
            &write_targets,
            &mut deleted_addrs,
            &headers,
            moved_headers,
        )?;
        let mut headers = headers;
        headers
            .keys
            .sort_by_key(|path| std::cmp::Reverse(path.len()));
        Ok(CommitPlan::Tree(PreparedTree {
            groups: nodes,
            headers,
            cross_copies,
            inplace_writes,
            deleted_free,
            deleted_collections,
            references,
        }))
    }
    fn prepare_value_writes(
        &mut self,
        staged: &StagedEdits,
        edits: &mut DatasetEdits,
    ) -> Result<(), Error> {
        for (full, fd) in staged.writes() {
            if edits.write_targets.contains(full) {
                return Err(Error::EditUnsupported(
                    "the same dataset is overwritten twice in one commit; use separate commits",
                ));
            }
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                self.context.superblock(),
                self.context.widths(),
                self.context.addresses(),
                &full.as_path(),
            )?;
            match self.prepare_write(addr, fd, full)? {
                WritePlan::InPlace { data_addr, bytes } => {
                    edits.inplace_writes.push((data_addr, bytes));
                }
                WritePlan::InPlaceChunks { writes } => edits.inplace_writes.extend(writes),
                WritePlan::Moving(mw) => {
                    let counts = edits
                        .incoming_links
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
                    edits.moving_writes.push(DatasetRelocation {
                        parent,
                        edit: RelocatedDataset {
                            link: leaf,
                            old_header: addr,
                            write: mw,
                        },
                    });
                }
            }
            edits.write_targets.push(full.clone());
        }

        Ok(())
    }
    fn prepare_appends(
        &mut self,
        staged: &StagedEdits,
        edits: &mut DatasetEdits,
    ) -> Result<(), Error> {
        let base = self.context.addresses();
        for (full, ab) in staged.appends() {
            let Some((parent, leaf)) = full.split_leaf() else {
                return Err(Error::AppendUnsupported("cannot append to the root group"));
            };
            if ab.raw().is_empty() {
                continue; // nothing to append
            }
            if edits.write_targets.contains(full) {
                return Err(Error::AppendUnsupported(
                    "the same dataset is edited more than once in one commit; use separate commits",
                ));
            }
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                self.context.superblock(),
                self.context.widths(),
                self.context.addresses(),
                &full.as_path(),
            )?;
            let mw = Self::prepare_append(&self.image(), addr, ab, base, self.context.widths())?;
            let counts = edits
                .incoming_links
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
            edits.moving_writes.push(DatasetRelocation {
                parent,
                edit: RelocatedDataset {
                    link: leaf,
                    old_header: addr,
                    write: mw,
                },
            });
            edits.write_targets.push(full.clone());
        }

        Ok(())
    }
    fn prepare_dataset_attributes(
        &mut self,
        staged: &StagedEdits,
        edits: &mut DatasetEdits,
    ) -> Result<(), Error> {
        let base = self.context.addresses();
        if !staged.dataset_attrs().is_empty() {
            let mut order: Vec<ObjectPathBuf> = Vec::new();
            let mut ops_by_path: HashMap<&ObjectPathBuf, Vec<&AttrOp>> = HashMap::new();
            for (path, op) in staged.dataset_attrs() {
                if !ops_by_path.contains_key(path) {
                    order.push(path.clone());
                }
                ops_by_path.entry(path).or_default().push(op);
            }
            for full in order {
                // Each path is recorded once when its operation list is inserted.
                let ops = ops_by_path.remove(&full).unwrap();
                let Some((parent, leaf)) = full.split_leaf() else {
                    return Err(Error::EditUnsupported(
                        "cannot set a dataset attribute on the root group; use set_group_attr",
                    ));
                };
                if edits.write_targets.contains(&full) {
                    return Err(Error::EditUnsupported(
                        "the same dataset is edited more than once in one commit (an attribute \
                         edit plus another edit); use separate commits",
                    ));
                }
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    AccessMode::ReadWrite,
                    self.context.superblock(),
                    self.context.widths(),
                    self.context.addresses(),
                    &full.as_path(),
                )?;
                let counts = edits
                    .incoming_links
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
                let attributes =
                    attributes::plan_attr_ops(&self.image(), base, Some(addr), &region, &ops)?;

                edits.moving_writes.push(DatasetRelocation {
                    parent,
                    edit: RelocatedDataset {
                        link: leaf,
                        old_header: addr,
                        write: MovingWrite::AttrEdit {
                            region: attributes.region,
                            attrs: attributes.attrs,
                            replaces_dense_storage: attributes.replaces_dense_storage,
                        },
                    },
                });
                edits.write_targets.push(full);
            }
        }

        Ok(())
    }
    fn plan_dirty_groups(
        &self,
        staged: &StagedEdits,
        moving_writes: Vec<DatasetRelocation>,
        write_targets: &[ObjectPathBuf],
    ) -> Result<DirtyGroups, Error> {
        let base = self.context.addresses();
        let mut nodes: BTreeMap<ObjectPathBuf, Node> = BTreeMap::new();
        nodes.entry(ObjectPathBuf::root()).or_default(); // root is always dirty
        let mut add_targets: Vec<ObjectPathBuf> = Vec::new();
        let mut copy_sources: Vec<ObjectPathBuf> = Vec::new();
        let mut attr_targets: Vec<ObjectPathBuf> = Vec::new();

        for path in staged.groups() {
            if path.is_empty() {
                return Err(Error::EditUnsupported("cannot create the root group"));
            }
            ensure_ancestors(&mut nodes, path);
            nodes.entry(path.clone()).or_default().is_new = true;
            add_targets.push(path.clone());
        }

        for (parent, fd) in staged.datasets() {
            add_targets.push(parent.join(&fd.name));
            ensure_ancestors(&mut nodes, parent);
        }

        for DatasetRelocation { parent, edit } in moving_writes {
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().prepared.writes.push(edit);
        }

        for (path, _) in staged.group_attrs() {
            ensure_ancestors(&mut nodes, path);
            attr_targets.push(path.clone());
        }

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
                self.context.superblock(),
                self.context.widths(),
                self.context.addresses(),
                &src.as_path(),
            )?;
            let tree = Self::read_copy_subtree(
                &self.image(),
                AccessMode::ReadWrite,
                src_addr,
                0,
                false,
                base,
                self.context.widths(),
            )?;
            copy_sources.push(src.clone());
            add_targets.push(dst.clone());
            ensure_ancestors(&mut nodes, &parent);
            nodes
                .entry(parent)
                .or_default()
                .prepared
                .copies
                .push((leaf, tree));
        }

        let mut cross_copies = Vec::new();
        for (dst, _) in staged.cross_copies() {
            let Some((parent, leaf)) = dst.split_leaf() else {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            };
            add_targets.push(dst.clone());
            ensure_ancestors(&mut nodes, &parent);
            cross_copies.push(CopyDestination {
                parent: parent.clone(),
                leaf: leaf.clone(),
            });
            nodes.entry(parent).or_default().cross_copies.push(leaf);
        }

        let delete_targets = staged.deletes();
        let mut deleted_addrs: Vec<u64> = Vec::new();
        for (i, d) in delete_targets.iter().enumerate() {
            let Some((parent, leaf)) = d.split_leaf() else {
                return Err(Error::EditUnsupported("cannot delete the root group"));
            };
            let del_addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                self.context.superblock(),
                self.context.widths(),
                self.context.addresses(),
                &d.as_path(),
            )?;
            deleted_addrs.push(del_addr);
            let recreated = add_targets.iter().any(|t| t == d);
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
            for t in write_targets {
                if t.starts_with(d) {
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
            nodes.entry(parent).or_default().prepared.deletes.push(leaf);
        }

        Ok(DirtyGroups {
            nodes,
            add_targets,
            deleted_addrs,
            cross_copies,
        })
    }
    fn prepare_group_headers(
        &self,
        staged: &StagedEdits,
        nodes: &mut BTreeMap<ObjectPathBuf, Node>,
        incoming_links: &mut Option<Option<HashMap<u64, u32>>>,
    ) -> Result<GroupHeaders, Error> {
        let base = self.context.addresses();
        let keys: Vec<ObjectPathBuf> = nodes.keys().cloned().collect();
        let mut superseded_addrs: Vec<(ObjectPathBuf, u64)> = Vec::new();
        for (key, node) in nodes.iter_mut() {
            if node.is_new {
                node.prepared.region = group::fresh_group_region();
            } else {
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    AccessMode::ReadWrite,
                    self.context.superblock(),
                    self.context.widths(),
                    self.context.addresses(),
                    &key.as_path(),
                )?;
                if !key.is_empty() {
                    let counts = incoming_links
                        .get_or_insert_with(|| self.count_incoming_hard_links())
                        .as_ref();
                    match counts.and_then(|c| c.get(&addr).copied()) {
                        Some(1) => {}
                        Some(_) => {
                            return Err(Error::EditUnsupported(
                                "editing a group relocates its object header; only supported \
                                 when it has a single hard link",
                            ));
                        }
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
                node.prepared.region = info.region;
                node.prepared.replaces = Some(ReplacedHeader::Header(addr));
                node.existing_links = info.link_names;
            }
        }

        let attrs_by_group = group_by_parent(staged.group_attrs().iter().map(|(p, op)| (p, op)));
        let existing_group_addrs: HashMap<ObjectPathBuf, u64> =
            superseded_addrs.iter().cloned().collect();
        for (key, node) in nodes.iter_mut() {
            if let Some(ops) = attrs_by_group.get(key) {
                let region = std::mem::take(&mut node.prepared.region);
                let edits = attributes::plan_attr_ops(
                    &self.image(),
                    base,
                    existing_group_addrs.get(key).copied(),
                    &region,
                    ops,
                )?;
                node.prepared.region = edits.region;
                node.prepared.attrs = edits.attrs;
                if edits.replaces_dense_storage
                    && let Some(replaced) = &mut node.prepared.replaces
                {
                    *replaced = ReplacedHeader::DenseAttributes(replaced.address());
                }
            }
        }

        let mut children: BTreeMap<ObjectPathBuf, Vec<ChildLink>> = BTreeMap::new();
        for key in &keys {
            if let Some((parent, leaf)) = key.split_leaf() {
                let child = if nodes[key].is_new {
                    ChildLink::Insert {
                        path: key.clone(),
                        leaf,
                    }
                } else {
                    ChildLink::Replace {
                        path: key.clone(),
                        leaf,
                    }
                };
                children.entry(parent).or_default().push(child);
            }
        }

        Ok(GroupHeaders {
            keys,
            superseded_addrs,
            children,
        })
    }
    fn prepare_tree_references(
        &mut self,
        staged: &StagedEdits,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        deleted_addrs: &mut Vec<u64>,
        headers: &GroupHeaders,
        moved_headers: Vec<u64>,
    ) -> Result<TreeReferences, Error> {
        let GroupHeaders {
            keys,
            superseded_addrs: _,
            children,
        } = headers;
        let datasets_by_group = group_by_parent(
            staged
                .datasets()
                .iter()
                .map(|(path, dataset)| (path, dataset)),
        );
        self.check_group_additions(nodes, keys, children, &datasets_by_group)?;
        self.check_libver_admits(staged.datasets().iter().map(|(_, dataset)| dataset))?;
        let delete_targets = staged.deletes();
        let mut deleted_free: Vec<(u64, u64, FreeClass)> = Vec::new();
        deleted_addrs.sort_unstable();
        deleted_addrs.dedup();
        if !deleted_addrs.is_empty() {
            if let Some(incoming) = self.count_incoming_hard_links() {
                for &a in deleted_addrs.iter() {
                    self.collect_free_spans(a, 0, &incoming, &mut deleted_free);
                }
            }
        }
        let deleted_collections = self.deleted_collections(deleted_addrs);
        let for_copied = InvalidatedAddresses {
            removed: deleted_free
                .iter()
                .map(|&(off, len, _)| (off, len))
                .collect(),
            moved: Vec::new(),
            base: self.context.addresses(),
        };
        {
            let image = self.image();
            let framed = BaseOffsetSource {
                inner: image,
                base: self.context.addresses().effective_base(),
            };
            for key in keys {
                for (_, tree) in &nodes[key].prepared.copies {
                    copy::screen_copied_references(
                        tree,
                        self.context.addresses(),
                        self.context.widths(),
                        &for_copied,
                        &framed,
                    )?;
                }
            }
        }

        let for_supplied = InvalidatedAddresses {
            moved: moved_headers,
            ..for_copied
        };

        self.screen_shared_message_index(&for_supplied)?;

        for (_, fd) in staged.datasets().iter().chain(staged.writes()) {
            edit::screen_resolved_references(&fd.dt, &fd.raw, &for_supplied)?;
        }

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

        let references = Self::prepare_reference_targets(
            keys,
            &datasets_by_group,
            nodes,
            add_targets,
            write_targets,
            delete_targets,
            &for_supplied,
            &self.image(),
            &self.context,
        )?;

        if self.fresh_page_boundary.is_some() {
            self.preflight_fresh_pages(staged, nodes)?;
        }

        Ok(TreeReferences {
            deleted_free,
            deleted_collections,
            references,
        })
    }
    /// Checks link-name conflicts, compact-storage limits, and creation-counter capacity.
    ///
    /// Every affected group's policy is checked before object placement, including attribute-only
    /// edits. Additions are counted together to check the full counter increment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for duplicate link names, unsupported creation-order
    /// policies, counter exhaustion, or additions beyond a tracked group's compact threshold.
    fn check_group_additions(
        &self,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        keys: &[ObjectPathBuf],
        children: &BTreeMap<ObjectPathBuf, Vec<ChildLink>>,
        datasets_by_group: &BTreeMap<&ObjectPathBuf, Vec<&FlatDataset>>,
    ) -> Result<(), Error> {
        for key in keys {
            let node = &nodes[key];
            let added = node.prepared.copies.len()
                + node.cross_copies.len()
                + datasets_by_group.get(key).map_or(0, Vec::len)
                + children.get(key).map_or(0, |kids| {
                    kids.iter()
                        .filter(|child| matches!(child, ChildLink::Insert { .. }))
                        .count()
                });
            let mut order = group::LinkCreationOrder::for_region(&node.prepared.region)?;
            if added > 0 {
                let kept = node
                    .existing_links
                    .iter()
                    .filter(|name| !node.prepared.deletes.iter().any(|d| d.as_str() == *name))
                    .count();
                group::reject_dense_link_creation_order(&node.prepared.region, kept + added)?;
                for _ in 0..added {
                    order.take()?;
                }
            }
        }

        for key in keys {
            let node = &nodes[key];
            let mut adding: Vec<&str> = Vec::new();
            for fd in datasets_by_group.get(key).into_iter().flatten() {
                adding.push(fd.name.as_str());
            }
            for child in children.get(key).into_iter().flatten() {
                if let ChildLink::Insert { leaf, .. } = child {
                    adding.push(leaf.as_str());
                }
            }
            for (leaf, _) in &node.prepared.copies {
                adding.push(leaf.as_str());
            }
            for leaf in &node.cross_copies {
                adding.push(leaf.as_str());
            }
            for (i, name) in adding.iter().enumerate() {
                let survives = node.existing_links.iter().any(|n| n == name)
                    && !node.prepared.deletes.iter().any(|n| n.as_str() == *name);
                if survives || adding[..i].contains(name) {
                    return Err(Error::EditUnsupported(
                        "a link with this name already exists in the target group",
                    ));
                }
            }
        }

        Ok(())
    }
}

impl WriteEngine {
    /// Proves that a non-persistent PAGE commit can publish without patching stored references.
    ///
    /// Checks the existing tree and prospective attribute storage before consuming staged edits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for object references, unsupported attribute storage,
    /// or an incomplete reference inspection. Returns address-resolution and object-header record
    /// errors from [`prove_free_of_references`](crate::reference_patch::prove_free_of_references).
    pub(in crate::edit) fn preflight_fresh_pages(
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
                matches!(node.prepared.attrs, EditedAttrs::Dense(_))
                    || node.prepared.writes.iter().any(|edit| {
                        matches!(
                            edit.write,
                            MovingWrite::AttrEdit {
                                attrs: EditedAttrs::Dense(_),
                                ..
                            }
                        )
                    })
            })
            || !crate::reference_patch::prove_free_of_references(
                &self.image(),
                self.context.superblock(),
                self.context.widths(),
                self.context.addresses(),
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

    /// Resolves a staged object reference against the commit's placement order and current file.
    ///
    /// Paths in `path_addr` resolve to objects placed earlier in the commit. Untouched paths
    /// resolve against the current file, with unresolved paths encoded as undefined references.
    /// Raw addresses are screened against storage reclaimed or relocated by the commit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a target is deleted, has an invalidated address,
    /// or requires placement later in the commit. Returns the address-conversion error if
    /// a resolved address cannot be stored.
    pub(in crate::edit) fn resolve_reference_target(
        target: &ObjectRefTarget,
        path_addr: &BTreeMap<ObjectPathBuf, u64>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl Source + ?Sized),
        context: &FileContext,
    ) -> Result<StoredAddress, Error> {
        let path = match target {
            ObjectRefTarget::Raw(addr) => {
                if let Some(refusal) = invalidated.refusal(*addr) {
                    return Err(Error::EditUnsupported(refusal));
                }
                return Ok(*addr);
            }
            ObjectRefTarget::Path(path) => path,
        };
        let base = context.addresses();
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
        // A replacement not yet placed is reported as an ordering conflict, even when the
        // replacement omits a descendant that was present under the deleted path.
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
                context.superblock(),
                context.widths(),
                context.addresses(),
                &path.as_path(),
            ) {
                Ok(addr) => base.relative(addr)?,
                Err(_) => StoredAddress::new(UNDEF),
            },
        )
    }

    fn prepare_reference_targets(
        keys: &[ObjectPathBuf],
        flat: &BTreeMap<&ObjectPathBuf, Vec<&FlatDataset>>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl Source + ?Sized),
        context: &FileContext,
    ) -> Result<BTreeMap<ObjectPathBuf, Vec<PreparedReference>>, Error> {
        let mut prepared = BTreeMap::new();
        let mut by_depth = keys.to_vec();
        // Preserve path order among groups at the same depth, matching application.
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let mut sim_addr: BTreeMap<ObjectPathBuf, u64> = BTreeMap::new();
        for key in &by_depth {
            if let Some(datasets) = flat.get(key) {
                // Match the dataset order used during application.
                let mut ordered: Vec<&FlatDataset> = datasets.to_vec();
                ordered.sort_by_key(|fd| fd.reference_targets.is_some());
                for fd in ordered {
                    let dataset_path = key.join(&fd.name);
                    if let Some(patches) = &fd.reference_targets
                        && !patches.is_empty()
                    {
                        let mut references = Vec::with_capacity(patches.len());
                        for patch in patches {
                            let address = Self::resolve_reference_target(
                                &patch.target,
                                &sim_addr,
                                nodes,
                                add_targets,
                                write_targets,
                                delete_targets,
                                invalidated,
                                src,
                                context,
                            )?;
                            let target = match &patch.target {
                                ObjectRefTarget::Path(path) if sim_addr.contains_key(path) => {
                                    ReferenceTarget::Placed(path.clone())
                                }
                                _ => ReferenceTarget::Existing(address),
                            };
                            references.push(PreparedReference {
                                byte_offset: patch.byte_offset,
                                target,
                            });
                        }
                        prepared.insert(dataset_path.clone(), references);
                    }
                    sim_addr.insert(dataset_path, context.addresses().effective_base().get());
                }
            }
            sim_addr.insert(key.clone(), context.addresses().effective_base().get());
        }
        Ok(prepared)
    }
}

/// Stores a group's prepared operations and the link information used during planning.
#[derive(Default)]
pub(in crate::edit) struct Node {
    is_new: bool,
    cross_copies: Vec<LinkNameBuf>,
    existing_links: Vec<String>,
    pub(super) prepared: PreparedGroup,
}

impl Node {
    /// Extracts the prepared operations after planning has checked the group's links.
    pub(super) fn into_prepared(self) -> PreparedGroup {
        let Self {
            is_new: _,
            cross_copies: _,
            existing_links: _,
            prepared,
        } = self;
        prepared
    }
}

/// Records physical addresses invalidated by a commit for object-reference screening.
///
/// Removed spans are also used for storage reclamation. Moved entries identify old object-header
/// addresses whose replacements are published by the commit.
pub(in crate::edit) struct InvalidatedAddresses {
    /// Absolute `(offset, length)` file spans this commit's deletions reclaim, as
    /// [`collect_free_spans`](WriteEngine::collect_free_spans) reports them.
    pub(in crate::edit) removed: Vec<(u64, u64)>,
    /// Superseded object-header addresses from relocating dataset writes and rebuilt groups.
    pub(in crate::edit) moved: Vec<u64>,
    /// The address context used to compare stored object references with physical positions.
    pub(in crate::edit) base: AddressContext,
}

impl InvalidatedAddresses {
    /// Returns whether the screen has no removed spans or relocated headers.
    pub(in crate::edit) fn is_empty(&self) -> bool {
        self.removed.is_empty() && self.moved.is_empty()
    }

    /// Returns a rejection for a reference into removed storage or a superseded header.
    ///
    /// Null, undefined, and unresolvable stored addresses are excluded from this screen.
    pub(in crate::edit) fn refusal(&self, stored: StoredAddress) -> Option<&'static str> {
        if stored.get() == 0 || stored.get() == UNDEF {
            return None;
        }
        let abs = self.base.absolute(stored).ok()?;
        if self
            .removed
            .iter()
            .any(|&(off, len)| abs >= off && abs - off < len)
        {
            return Some(REFERENCE_INTO_RECLAIMED_SPACE);
        }
        if self.moved.contains(&abs) {
            return Some(REFERENCE_TO_A_MOVED_OBJECT);
        }
        None
    }
}

/// Groups items by parent while preserving their input order within each group.
///
/// Planning and application use the same order for dataset placement and reference resolution.
pub(super) fn group_by_parent<K: Ord, T>(
    items: impl IntoIterator<Item = (K, T)>,
) -> BTreeMap<K, Vec<T>> {
    let mut out: BTreeMap<K, Vec<T>> = BTreeMap::new();
    for (parent, item) in items {
        out.entry(parent).or_default().push(item);
    }
    out
}

/// Includes every ancestor of `path` in the group rebuild, preserving existing node state.
fn ensure_ancestors(nodes: &mut BTreeMap<ObjectPathBuf, Node>, path: &ObjectPathBuf) {
    for len in 0..=path.len() {
        nodes.entry(path.prefix(len)).or_default();
    }
}
