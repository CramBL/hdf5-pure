//! Prepared edits, dirty-group graphs, and reference targets for staged commits.
//!
//! Planning reads the current file and borrows the staged inputs. The prepared operations
//! contain the structural decisions needed during placement, where allocation and I/O can fail.

use std::collections::BTreeSet;
use std::collections::{BTreeMap, HashMap};
use std::vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;

use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::address::{AddressContext, StoredAddress, StoredAddressExt};
use crate::dataset_request::ValueOverwrite;
use crate::datatype;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::CollectionCandidates;
use crate::edit::analysis::PlannedDeletion;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::attributes::{self, AttrOp, EditedAttrs};
use crate::edit::context::FileContext;
use crate::edit::copy::{self, CopyTree};
use crate::edit::dataset::{ChunkPayload, FlatDataset, MovingWrite, OverwriteBytes, WritePlan};
use crate::edit::group;
use crate::edit::group::LinkCreationOrder;
use crate::edit::group::PreparedHardLinkRecord;
use crate::edit::staging::{StagedApply, StagedEdits, StagedScope};
use crate::edit::{self, OhRegion};
use crate::edit::{MAX_LINK_GRAPH_NODES, UNDEF, WriteEngine};
use crate::edit::{REFERENCE_INTO_RECLAIMED_SPACE, REFERENCE_TO_A_MOVED_OBJECT};
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::source::OriginalSource;
use crate::type_builders::ObjectRefTarget;

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "count projections are exercised by private owner tests"
    )
)]
mod counts;

#[cfg(test)]
mod tests;

/// A commit plan paired with owned application inputs from the same staged set.
///
/// The editor borrows the staged edits during preflight. After success, original replacement
/// buffers are retained for application or released when their chunks have been encoded.
pub(super) struct PreparedCommit {
    plan: CommitPlan,
    staged: StagedApply,
    overwrites: OverwritePayloads,
}

impl PreparedCommit {
    /// Plans a commit, returning the original staged inputs on rejection.
    ///
    /// Returns [`CommitPreparation::Prepared`] with the plan and matching application inputs
    /// if preflight succeeds, or [`CommitPreparation::Rejected`] with the original staged set
    /// and the preparation error.
    pub(super) fn prepare(engine: &mut WriteEngine, staged: StagedEdits) -> CommitPreparation {
        match engine.plan_commit(&staged) {
            Ok(mut plan) => {
                let (staged, writes) = staged.into_apply();
                let mut overwrites = OverwritePayloads { writes };
                plan.release_chunk_inputs(&mut overwrites);
                CommitPreparation::Prepared(Self {
                    plan,
                    staged,
                    overwrites,
                })
            }
            Err(error) => CommitPreparation::Rejected { staged, error },
        }
    }

    /// Returns the plan with the application inputs used to prepare it.
    pub(super) fn into_parts(self) -> (CommitPlan, StagedApply, OverwritePayloads) {
        let Self {
            plan,
            staged,
            overwrites,
        } = self;
        (plan, staged, overwrites)
    }
}

/// The result of commit preflight, with ownership for application or retry.
#[allow(
    clippy::large_enum_variant,
    reason = "returning rejected staging to the engine requires no heap allocation"
)]
pub(super) enum CommitPreparation {
    /// A plan paired with the owned inputs needed for application.
    Prepared(PreparedCommit),
    /// The original staged inputs and the error that prevented preparation.
    Rejected { staged: StagedEdits, error: Error },
}

/// Stores prepared writes before application consumes the staged inputs.
pub(super) enum CommitPlan {
    /// Existing value ranges to overwrite without publishing a replacement root.
    InPlace(Vec<InPlaceWrite>),
    /// Object placements and link edits to publish through a replacement root.
    Tree(PreparedTree),
}

impl CommitPlan {
    /// Drops original replacement buffers whose chunks were encoded during successful preflight.
    fn release_chunk_inputs(&mut self, overwrites: &mut OverwritePayloads) {
        let writes = match self {
            Self::InPlace(writes) => writes,
            Self::Tree(tree) => {
                for node in tree.groups.values_mut() {
                    for edit in &mut node.prepared.writes {
                        if let MovingWrite::Chunked {
                            payload: ChunkPayload::Encoded { source, .. },
                            ..
                        } = &mut edit.write
                        {
                            // Encoded output replaces this uniquely identified staged input.
                            drop(
                                overwrites.take(
                                    source
                                        .take()
                                        .expect("encoded overwrite input has not been consumed"),
                                ),
                            );
                        }
                    }
                }
                &mut tree.inplace_writes
            }
        };
        for write in writes {
            if let InPlaceWrite::Encoded { source, .. } = write {
                // Encoded output replaces this uniquely identified staged input.
                drop(
                    overwrites.take(
                        source
                            .take()
                            .expect("encoded overwrite input has not been consumed"),
                    ),
                );
            }
        }
    }
}

/// Collects existing dataset edits and the ownership checks needed for relocation.
struct DatasetEdits {
    inplace_writes: Vec<InPlaceWrite>,
    moving_writes: Vec<DatasetRelocation>,
    write_targets: Vec<ObjectPathBuf>,
    /// Collection candidates from original dataset attributes.
    collections: CollectionCandidates,
}

/// Associates a relocating dataset edit with its parent group.
struct DatasetRelocation {
    parent: ObjectPathBuf,
    edit: RelocatedDataset,
}

/// A dataset replacement with its child-link name and source-phase association.
pub(super) struct RelocatedDataset {
    pub(super) link: LinkNameBuf,
    /// The original physical object identity bound to its analysis phase.
    pub(super) source: ReconstructionSource,
    pub(super) write: MovingWrite,
}

/// Collects affected groups and paths used to reject conflicting edits.
struct DirtyGroups {
    nodes: BTreeMap<ObjectPathBuf, Node>,
    add_targets: Vec<ObjectPathBuf>,
    cross_copies: Vec<CopyDestination>,
}

/// Stores group placement order, superseded headers, and prepared child-link edits.
pub(super) struct GroupHeaders {
    /// Group paths, sorted deepest-first before application.
    pub(super) keys: Vec<ObjectPathBuf>,
    /// Group paths and superseded object identities, in path order.
    pub(super) superseded_addrs: Vec<(ObjectPathBuf, ObjectId)>,
    pub(super) children: BTreeMap<ObjectPathBuf, Vec<ChildLink>>,
}

/// Stores checked tree edits and reclamation records before object placement.
///
/// Staged creations and cross-file-copy payloads are stored separately from the tree plan.
pub(super) struct PreparedTree {
    pub(super) groups: BTreeMap<ObjectPathBuf, Node>,
    pub(super) headers: GroupHeaders,
    pub(super) cross_copies: Vec<CopyDestination>,
    pub(super) inplace_writes: Vec<InPlaceWrite>,
    /// Absolute spans whose ownership was established while planning deletions.
    pub(super) deleted_free: Vec<(FileRange, FreeClass)>,
    /// Attribute and deletion candidates awaiting reference analysis from the replacement root.
    pub(super) collection_candidates: CollectionCandidates,
    /// Prepared element-reference patches, indexed by the containing dataset's path.
    pub(super) references: BTreeMap<ObjectPathBuf, Vec<PreparedReference>>,
}

/// Stores the object-header region and operations needed to rebuild one group.
#[derive(Default)]
pub(super) struct PreparedGroup {
    pub(super) region: OhRegion,
    /// Whether placement writes a heap and name index for the final link set.
    pub(super) dense_links: bool,
    pub(super) replaces: Option<ReplacedHeader>,
    /// Leaf names and physical deletion targets, including any phase-bound edge identities.
    pub(super) deletes: Vec<PlannedDeletion>,
    pub(super) copies: Vec<(LinkNameBuf, CopyTree)>,
    pub(super) writes: Vec<RelocatedDataset>,
    pub(super) attrs: EditedAttrs,
    /// Checked destination records consumed in object-placement order.
    pub(super) additions: GroupAdditions,
}

/// Checked destination records and the group's final link creation counter.
///
/// Records are ordered by same-file copies, cross-file copies, datasets, and new child groups.
/// Datasets with staged reference targets follow the other datasets. Application patches each
/// record's stored target address before appending it to the group.
#[derive(Default)]
pub(super) enum GroupAdditions {
    /// A group with no new destination links.
    #[default]
    Empty,
    /// Remaining records and the counter after assigning all their creation indexes.
    Records {
        records: vec::IntoIter<PreparedHardLinkRecord>,
        order: LinkCreationOrder,
    },
}

impl GroupAdditions {
    fn new(records: Vec<PreparedHardLinkRecord>, order: LinkCreationOrder) -> Self {
        if records.is_empty() {
            Self::Empty
        } else {
            Self::Records {
                records: records.into_iter(),
                order,
            }
        }
    }

    /// Removes the next record and checks its destination name.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the sequence is empty or its next destination differs
    /// from `name`. Propagates errors from reading the encoded name.
    pub(super) fn take(&mut self, name: &LinkNameBuf) -> Result<PreparedHardLinkRecord, Error> {
        let Self::Records { records, .. } = self else {
            return Err(FormatError::Internal("a new link has no prepared record".into()).into());
        };
        let record = records.next().ok_or_else(|| {
            FormatError::Internal("a new link has no remaining prepared record".into())
        })?;
        if record.name_bytes()? != name.as_str().as_bytes() {
            return Err(FormatError::Internal(
                "a prepared link has a different destination".into(),
            )
            .into());
        }
        Ok(record)
    }

    /// Writes the final creation counter after every prepared record has been consumed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a prepared record remains. Propagates errors from
    /// [`LinkCreationOrder::record`].
    pub(super) fn finish(self, region: &mut OhRegion) -> Result<(), Error> {
        if let Self::Records { mut records, order } = self {
            if records.next().is_some() {
                return Err(FormatError::Internal(
                    "prepared links remain after group placement".into(),
                )
                .into());
            }
            order.record(region)?;
        }
        Ok(())
    }
}

/// The source association of a superseded group header and its replaced attribute storage.
pub(super) enum ReplacedHeader {
    /// A source group whose dense attribute storage is also replaced.
    DenseAttributes(ReconstructionSource),
    /// A source group without separate dense-attribute reclamation.
    Header(ReconstructionSource),
}

impl ReplacedHeader {
    /// Returns the identity of the superseded group.
    pub(super) fn object(&self) -> ObjectId {
        self.source().object()
    }

    /// Borrows the source-phase association and physical identity of the superseded header.
    pub(super) fn source(&self) -> &ReconstructionSource {
        match self {
            Self::DenseAttributes(source) | Self::Header(source) => source,
        }
    }

    /// Marks dense attribute storage for replacement, preserving the source association.
    fn into_dense_attributes(self) -> Self {
        match self {
            Self::DenseAttributes(source) | Self::Header(source) => Self::DenseAttributes(source),
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
    deleted_free: Vec<(FileRange, FreeClass)>,
    deleted_collections: CollectionCandidates,
    references: BTreeMap<ObjectPathBuf, Vec<PreparedReference>>,
    invalidates_reference_proof: bool,
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

/// A stored reference value or a path scheduled before the referring dataset.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum ReferenceTarget {
    /// An address retained from the current file, including null or undefined values.
    Existing(StoredAddress),
    /// A path scheduled before the referring dataset in the commit's write order.
    Placed(ObjectPathBuf),
}

impl WriteEngine {
    /// Screens narrow-offset files for reference datatypes and caches a successful proof.
    ///
    /// Skips the source walk for eight-byte offsets or an existing successful proof.
    ///
    /// # Errors
    ///
    /// Returns inspection errors from [`crate::reference_patch::preflight_narrow_references`].
    pub(in crate::edit) fn preflight_narrow_references(&mut self) -> Result<(), Error> {
        self.proved_free_of_references = crate::reference_patch::preflight_narrow_references(
            &self.opened_source(),
            self.context.superblock(),
            self.context.widths(),
            MAX_LINK_GRAPH_NODES,
            self.proved_free_of_references,
        )?;
        Ok(())
    }

    /// Prepares edits while retaining the staged inputs for retry on rejection.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] or [`Error::AppendUnsupported`] if an edit cannot
    /// be reproduced or conflicts with another edit. Returns the read, parsing, or preparation
    /// error if the current file or a staged operation cannot be inspected.
    fn plan_commit(&mut self, staged: &StagedEdits) -> Result<CommitPlan, Error> {
        self.preflight_narrow_references()?;
        let mut analysis = AnalysisPhase::new(self);
        let mut edits = DatasetEdits {
            inplace_writes: Vec::new(),
            moving_writes: Vec::new(),
            write_targets: Vec::new(),
            collections: CollectionCandidates::default(),
        };
        self.prepare_value_writes(staged, &mut edits, &mut analysis)?;
        self.prepare_appends(staged, &mut edits, &mut analysis)?;
        self.prepare_dataset_attributes(staged, &mut edits, &mut analysis)?;
        let DatasetEdits {
            inplace_writes,
            moving_writes,
            write_targets,
            mut collections,
        } = edits;
        if moving_writes.is_empty() && staged.scope() == StagedScope::Datasets {
            return Ok(CommitPlan::InPlace(inplace_writes));
        }
        let mut moved_headers = moving_writes
            .iter()
            .map(|relocation| relocation.edit.source.object().header_address())
            .collect::<Vec<_>>();
        let DirtyGroups {
            mut nodes,
            add_targets,
            cross_copies,
        } = self.plan_dirty_groups(staged, moving_writes, &write_targets, &mut analysis)?;
        let headers =
            self.prepare_group_headers(staged, &mut nodes, &mut analysis, &mut collections)?;
        moved_headers.extend(
            headers
                .superseded_addrs
                .iter()
                .map(|&(_, address)| address.header_address()),
        );
        let TreeReferences {
            deleted_free,
            deleted_collections,
            references,
            invalidates_reference_proof,
        } = self.prepare_tree_references(
            staged,
            &mut nodes,
            &add_targets,
            &write_targets,
            &headers,
            moved_headers,
            &mut analysis,
        )?;
        collections.extend(deleted_collections)?;
        if invalidates_reference_proof {
            self.proved_free_of_references = false;
        }
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
            collection_candidates: collections,
            references,
        }))
    }
    fn prepare_value_writes(
        &self,
        staged: &StagedEdits,
        edits: &mut DatasetEdits,
        analysis: &mut AnalysisPhase<'_>,
    ) -> Result<(), Error> {
        for (index, (full, fd)) in staged.writes().iter().enumerate() {
            if edits.write_targets.contains(full) {
                return Err(Error::EditUnsupported(
                    "the same dataset is overwritten twice in one commit; use separate commits",
                ));
            }
            let addr = crate::group::v2::resolve_path_any_from_source(
                &self.opened_source(),
                AccessMode::ReadWrite,
                self.context.superblock(),
                self.context.widths(),
                &full.as_path(),
            )
            .map(PhysicalAddress::new)
            .map(ObjectId::new)?;
            match self.prepare_write(addr, fd, OverwriteId { index })? {
                WritePlan::InPlace { data_addr, source } => {
                    edits.inplace_writes.push(InPlaceWrite::Replacement {
                        address: data_addr,
                        source,
                    });
                }
                WritePlan::InPlaceChunks { writes, source } => {
                    edits.inplace_writes.push(InPlaceWrite::Encoded {
                        writes,
                        source: Some(source),
                    })
                }
                WritePlan::Moving(mw) => {
                    match analysis
                        .topology()
                        .and_then(|topology| topology.incoming(addr))
                    {
                        Some(1) => {}
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
                            source: analysis.reconstruction_source(self, addr)?,
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
        &self,
        staged: &StagedEdits,
        edits: &mut DatasetEdits,
        analysis: &mut AnalysisPhase<'_>,
    ) -> Result<(), Error> {
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
            let addr = crate::group::v2::resolve_path_any_from_source(
                &self.opened_source(),
                AccessMode::ReadWrite,
                self.context.superblock(),
                self.context.widths(),
                &full.as_path(),
            )
            .map(PhysicalAddress::new)
            .map(ObjectId::new)?;
            let mw = Self::prepare_append(&self.opened_source(), addr, ab, self.context.widths())?;
            match analysis
                .topology()
                .and_then(|topology| topology.incoming(addr))
            {
                Some(1) => {}
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
                    source: analysis.reconstruction_source(self, addr)?,
                    write: mw,
                },
            });
            edits.write_targets.push(full.clone());
        }

        Ok(())
    }
    fn prepare_dataset_attributes(
        &self,
        staged: &StagedEdits,
        edits: &mut DatasetEdits,
        analysis: &mut AnalysisPhase<'_>,
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
                let addr = crate::group::v2::resolve_path_any_from_source(
                    &self.opened_source(),
                    AccessMode::ReadWrite,
                    self.context.superblock(),
                    self.context.widths(),
                    &full.as_path(),
                )
                .map(PhysicalAddress::new)
                .map(ObjectId::new)?;
                match analysis
                    .topology()
                    .and_then(|topology| topology.incoming(addr))
                {
                    Some(1) => {}
                    _ => {
                        return Err(Error::EditUnsupported(
                            "editing a dataset attribute relocates its header; only supported \
                             when it has a single hard link",
                        ));
                    }
                }
                let region = Self::gather_oh_messages(
                    &self.image(),
                    addr.header_address(),
                    base,
                    self.context.widths(),
                )?;
                let attributes = attributes::plan_attr_ops(&self.opened_source(), &region, &ops)?;
                edits
                    .collections
                    .extend(analysis.attribute_collections(addr, attributes.predecessors)?)?;

                edits.moving_writes.push(DatasetRelocation {
                    parent,
                    edit: RelocatedDataset {
                        link: leaf,
                        source: analysis.reconstruction_source(self, addr)?,
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
        analysis: &mut AnalysisPhase<'_>,
    ) -> Result<DirtyGroups, Error> {
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
            let src_addr = crate::group::v2::resolve_path_any_from_source(
                &self.opened_source(),
                AccessMode::ReadWrite,
                self.context.superblock(),
                self.context.widths(),
                &src.as_path(),
            )?;
            let tree = Self::read_copy_subtree(
                &self.opened_source(),
                AccessMode::ReadWrite,
                ObjectId::new(PhysicalAddress::new(src_addr)),
                false,
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
        for (i, d) in delete_targets.iter().enumerate() {
            let Some((parent, leaf)) = d.split_leaf() else {
                return Err(Error::EditUnsupported("cannot delete the root group"));
            };
            let deletion = match analysis
                .topology()
                .and_then(|topology| topology.deletion(d))
            {
                Some(deletion) => deletion,
                None => PlannedDeletion::unresolved(
                    leaf,
                    PhysicalAddress::new(crate::group::v2::resolve_path_any_from_source(
                        &self.opened_source(),
                        AccessMode::ReadWrite,
                        self.context.superblock(),
                        self.context.widths(),
                        &d.as_path(),
                    )?),
                ),
            };
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
            nodes
                .entry(parent)
                .or_default()
                .prepared
                .deletes
                .push(deletion);
        }

        Ok(DirtyGroups {
            nodes,
            add_targets,
            cross_copies,
        })
    }
    fn prepare_group_headers(
        &self,
        staged: &StagedEdits,
        nodes: &mut BTreeMap<ObjectPathBuf, Node>,
        analysis: &mut AnalysisPhase<'_>,
        collections: &mut CollectionCandidates,
    ) -> Result<GroupHeaders, Error> {
        let keys: Vec<ObjectPathBuf> = nodes.keys().cloned().collect();
        let mut superseded_addrs: Vec<(ObjectPathBuf, ObjectId)> = Vec::new();
        for (key, node) in nodes.iter_mut() {
            if node.is_new {
                node.prepared.region = group::fresh_group_region(self.context.widths())?;
            } else {
                let addr = crate::group::v2::resolve_path_any_from_source(
                    &self.opened_source(),
                    AccessMode::ReadWrite,
                    self.context.superblock(),
                    self.context.widths(),
                    &key.as_path(),
                )
                .map(PhysicalAddress::new)
                .map(ObjectId::new)?;
                if !key.is_empty() {
                    match analysis
                        .topology()
                        .and_then(|topology| topology.incoming(addr))
                    {
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
                let inspected = self.inspect_group(addr)?;
                node.prepared.dense_links = inspected.links.info().fractal_heap_address.is_some();
                let (region, link_names) = inspected.into_compact()?;
                superseded_addrs.push((key.clone(), addr));
                node.prepared.region = region;
                node.prepared.replaces = Some(ReplacedHeader::Header(
                    analysis.reconstruction_source(self, addr)?,
                ));
                node.existing_links = link_names;
            }
        }

        let attrs_by_group = group_by_parent(staged.group_attrs().iter().map(|(p, op)| (p, op)));
        for (key, node) in nodes.iter_mut() {
            if let Some(ops) = attrs_by_group.get(key) {
                let region = std::mem::take(&mut node.prepared.region);
                let edits = attributes::plan_attr_ops(&self.opened_source(), &region, ops)?;
                if let Some(replaced) = &node.prepared.replaces {
                    collections.extend(
                        analysis.attribute_collections(replaced.object(), edits.predecessors)?,
                    )?;
                }
                node.prepared.region = edits.region;
                node.prepared.attrs = edits.attrs;
                if edits.replaces_dense_storage {
                    node.prepared.replaces = node
                        .prepared
                        .replaces
                        .take()
                        .map(ReplacedHeader::into_dense_attributes);
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
        &self,
        staged: &StagedEdits,
        nodes: &mut BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        headers: &GroupHeaders,
        moved_headers: Vec<PhysicalAddress>,
        analysis: &mut AnalysisPhase<'_>,
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
        let mut deleted_free: Vec<(FileRange, FreeClass)> = Vec::new();
        let deleted_targets = analysis
            .removed_links(nodes.values().flat_map(|node| &node.prepared.deletes))?
            .collect_free_spans(&mut deleted_free);
        let deleted_collections = analysis.deleted_collections(&deleted_targets);
        let for_copied = InvalidatedAddresses {
            removed: deleted_free.iter().map(|&(range, _)| range).collect(),
            moved: Vec::new(),
            base: self.context.addresses(),
        };
        {
            let opened = self.opened_source();
            for key in keys {
                for (_, tree) in &nodes[key].prepared.copies {
                    copy::screen_copied_references(
                        tree,
                        self.context.widths(),
                        &for_copied,
                        &opened,
                    )?;
                }
            }
        }

        let for_supplied = InvalidatedAddresses {
            moved: moved_headers,
            ..for_copied
        };

        self.screen_shared_message_index(&for_supplied)?;

        for (_, fd) in staged.datasets() {
            edit::screen_resolved_references(&fd.dt, fd.storage.raw(), &for_supplied)?;
        }
        for (_, values) in staged.writes() {
            edit::screen_resolved_references(&values.dt, values.raw(), &for_supplied)?;
        }

        let invalidates_reference_proof = staged.datasets().iter().any(|(_, fd)| {
            datatype::datatype_holds_object_address(&fd.dt)
                || fd
                    .attrs
                    .iter()
                    .any(|a| datatype::datatype_holds_object_address(&a.datatype))
        }) || staged
            .writes()
            .iter()
            .any(|(_, values)| datatype::datatype_holds_object_address(&values.dt))
            || !staged.copies().is_empty()
            || !staged.cross_copies().is_empty();

        let references = Self::prepare_reference_targets(
            keys,
            &datasets_by_group,
            nodes,
            add_targets,
            write_targets,
            delete_targets,
            &for_supplied,
            &self.opened_source(),
            &self.context,
        )?;

        if self.fresh_page_boundary.is_some() {
            self.preflight_fresh_pages(staged, nodes)?;
        }

        Ok(TreeReferences {
            deleted_free,
            deleted_collections,
            references,
            invalidates_reference_proof,
        })
    }
    /// Prepares destination records and selects storage for each final link set.
    ///
    /// Every affected group's policy is checked before object placement, including attribute-only
    /// edits. Assigns creation indexes in placement order and retains the encoded records in
    /// [`GroupAdditions`]. Storage selection uses one temporary region per group with link edits,
    /// including every staged addition and deletion.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for duplicate link names, oversized Link bodies,
    /// unsupported creation-order policies, counter exhaustion, or a final link set that cannot
    /// be encoded. Propagates parsing and source-read errors from group inspection.
    fn check_group_additions(
        &self,
        nodes: &mut BTreeMap<ObjectPathBuf, Node>,
        keys: &[ObjectPathBuf],
        children: &BTreeMap<ObjectPathBuf, Vec<ChildLink>>,
        datasets_by_group: &BTreeMap<&ObjectPathBuf, Vec<&FlatDataset>>,
    ) -> Result<(), Error> {
        for key in keys {
            let node = nodes
                .get_mut(key)
                .ok_or_else(|| FormatError::Internal("a prepared group is absent".into()))?;
            let mut adding: Vec<&LinkNameBuf> =
                node.prepared.copies.iter().map(|(leaf, _)| leaf).collect();
            adding.extend(&node.cross_copies);
            let datasets = datasets_by_group.get(key).map_or(&[][..], Vec::as_slice);
            adding.extend(
                datasets
                    .iter()
                    .filter(|dataset| dataset.storage.reference_targets().is_none())
                    .chain(
                        datasets
                            .iter()
                            .filter(|dataset| dataset.storage.reference_targets().is_some()),
                    )
                    .map(|dataset| &dataset.name),
            );
            adding.extend(children.get(key).into_iter().flatten().filter_map(
                |child| match child {
                    ChildLink::Insert { leaf, .. } => Some(leaf),
                    ChildLink::Replace { .. } => None,
                },
            ));
            if !adding.is_empty() {
                let deleted: BTreeSet<&str> = node
                    .prepared
                    .deletes
                    .iter()
                    .map(|deletion| deletion.name().as_str())
                    .collect();
                let mut names: BTreeSet<&str> = node
                    .existing_links
                    .iter()
                    .map(String::as_str)
                    .filter(|name| !deleted.contains(name))
                    .collect();
                for name in &adding {
                    if !names.insert(name.as_str()) {
                        return Err(Error::EditUnsupported(
                            "a link with this name already exists in the target group",
                        ));
                    }
                }
            }
            let mut order = LinkCreationOrder::for_region(&node.prepared.region)?;
            if adding.is_empty() && node.prepared.deletes.is_empty() {
                node.prepared.dense_links =
                    self.plan_group_links(&node.prepared.region, node.prepared.dense_links)?;
                continue;
            }
            let mut result = if node.prepared.deletes.is_empty() {
                node.prepared.region.clone()
            } else {
                group::remove_links_from_region(
                    &node.prepared.region,
                    node.prepared.deletes.iter().map(PlannedDeletion::name),
                )?
            };
            let mut records = Vec::with_capacity(adding.len());
            for name in adding {
                let record = PreparedHardLinkRecord::new(
                    result.widths().offsets,
                    result.layout(),
                    name.clone(),
                    order.take()?,
                )?;
                result.push_bytes(record.as_bytes());
                records.push(record);
            }
            order.record(&mut result)?;
            node.prepared.dense_links =
                self.plan_group_links(&result, node.prepared.dense_links)?;
            node.prepared.additions = GroupAdditions::new(records, order);
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
        let unsupported_dataset = staged.datasets().iter().any(|(_, dataset)| {
            dataset.dense_attrs.is_some()
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
                .writes()
                .iter()
                .any(|(_, values)| datatype::datatype_holds_object_address(&values.dt))
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
                &self.opened_source(),
                self.context.superblock(),
                self.context.widths(),
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

    /// Resolves a staged object reference against planned write order and the current file.
    ///
    /// Returns [`ReferenceTarget::Placed`] for a path in `prior_paths`. The caller supplies paths
    /// scheduled before the referring dataset. Resolves untouched paths against the current file,
    /// returning an undefined stored reference if lookup fails. Screens raw addresses against
    /// storage reclaimed or relocated by the commit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a target is deleted, has an invalidated address,
    /// or requires placement later in the commit. Returns the address-conversion error if
    /// a resolved address cannot be stored. Returns [`Error::Format`] with
    /// [`FormatError::Internal`] if a source-index target is passed to the editor.
    pub(in crate::edit) fn resolve_reference_target(
        target: &ObjectRefTarget,
        prior_paths: &BTreeSet<ObjectPathBuf>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl OriginalSource + ?Sized),
        context: &FileContext,
    ) -> Result<ReferenceTarget, Error> {
        let path = match target {
            ObjectRefTarget::Raw(addr) => {
                if let Some(refusal) = invalidated.refusal(*addr) {
                    return Err(Error::EditUnsupported(refusal));
                }
                return Ok(ReferenceTarget::Existing(*addr));
            }
            ObjectRefTarget::Path(path) => path,
            ObjectRefTarget::SourceIndex(_) => {
                return Err(FormatError::Internal(
                    "a source object index reaches an edit reference writer".into(),
                )
                .into());
            }
        };
        let base = src.addresses();
        if prior_paths.contains(path) {
            return Ok(ReferenceTarget::Placed(path.clone()));
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
        Ok(ReferenceTarget::Existing(
            match crate::group::v2::resolve_path_any_from_source(
                src,
                AccessMode::ReadWrite,
                context.superblock(),
                context.widths(),
                &path.as_path(),
            ) {
                Ok(addr) => base.relative(PhysicalAddress::new(addr))?,
                Err(_) => StoredAddress::new(UNDEF),
            },
        ))
    }

    fn prepare_reference_targets(
        keys: &[ObjectPathBuf],
        flat: &BTreeMap<&ObjectPathBuf, Vec<&FlatDataset>>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl OriginalSource + ?Sized),
        context: &FileContext,
    ) -> Result<BTreeMap<ObjectPathBuf, Vec<PreparedReference>>, Error> {
        let mut prepared = BTreeMap::new();
        let mut by_depth = keys.to_vec();
        // Preserve path order among groups at the same depth, matching application.
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let mut prior_paths = BTreeSet::new();
        for key in &by_depth {
            if let Some(datasets) = flat.get(key) {
                // Match the dataset order used during application.
                let mut ordered: Vec<&FlatDataset> = datasets.to_vec();
                ordered.sort_by_key(|fd| fd.storage.reference_targets().is_some());
                for fd in ordered {
                    let dataset_path = key.join(&fd.name);
                    if let Some(patches) = fd.storage.reference_targets()
                        && !patches.is_empty()
                    {
                        let mut references = Vec::with_capacity(patches.len());
                        for patch in patches {
                            let target = Self::resolve_reference_target(
                                &patch.target,
                                &prior_paths,
                                nodes,
                                add_targets,
                                write_targets,
                                delete_targets,
                                invalidated,
                                src,
                                context,
                            )?;
                            references.push(PreparedReference {
                                byte_offset: patch.byte_offset,
                                target,
                            });
                        }
                        prepared.insert(dataset_path.clone(), references);
                    }
                    prior_paths.insert(dataset_path);
                }
            }
            prior_paths.insert(key.clone());
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

/// Replacement buffers and deferred heap staging from one prepared commit.
///
/// The editor constructs this set and its [`OverwriteId`] values from the same staged writes.
/// Taking an identifier moves its payload out of the entry at its index.
pub(in crate::edit) struct OverwritePayloads {
    writes: Vec<(ObjectPathBuf, ValueOverwrite)>,
}

impl OverwritePayloads {
    /// Moves the replacement bytes and deferred heap staging identified by `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` has an out-of-range index.
    pub(in crate::edit) fn take(&mut self, id: OverwriteId) -> OverwriteBytes {
        // IDs are created once per staged write during this prepared commit's preflight.
        let (path, values) = self
            .writes
            .get_mut(id.index)
            .expect("a prepared overwrite belongs to its staged set");
        let (raw, staging) = values.take_payload();
        OverwriteBytes::new(
            raw,
            staging,
            core::mem::replace(path, ObjectPathBuf::root()),
        )
    }
}

/// An overwrite of existing storage with encoded bytes or a staged replacement.
pub(super) enum InPlaceWrite {
    /// Encoded chunk and index writes prepared before application.
    Encoded {
        writes: Vec<(PhysicalAddress, Vec<u8>)>,
        /// The original input's identifier, cleared when its buffer is released after preflight.
        source: Option<OverwriteId>,
    },
    /// A contiguous range written from its original staged buffer.
    Replacement {
        /// The physical address of the existing data range.
        address: PhysicalAddress,
        source: OverwriteId,
    },
}

/// Identifies one replacement payload in its prepared commit.
///
/// The private index is assigned during preflight. Taking the payload consumes the identifier.
pub(in crate::edit) struct OverwriteId {
    index: usize,
}

/// Records physical addresses invalidated by a commit for object-reference screening.
///
/// Removed spans are also used for storage reclamation. Moved entries identify old object-header
/// addresses whose replacements are published by the commit.
pub(in crate::edit) struct InvalidatedAddresses {
    /// Physical ranges this commit's deletions reclaim, as
    /// [`collect_free_spans`](WriteEngine::collect_free_spans) reports them.
    pub(in crate::edit) removed: Vec<FileRange>,
    /// Superseded object-header addresses from relocating dataset writes and rebuilt groups.
    pub(in crate::edit) moved: Vec<PhysicalAddress>,
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
        let abs = self.base.resolve(stored).ok()?;
        if self
            .removed
            .iter()
            .any(|range| range.start() <= abs && abs < range.end())
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
