//! Phase-scoped topology, object counts, collection references, and allocation facts.

#[cfg(test)]
use core::cell::Cell;
use core::iter;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCount;
#[cfg(doc)]
use hdf5_pure_format::__private::ObjectReferenceCountObserver;
use hdf5_pure_space::__private::FreeClass;

use crate::access_mode::AccessMode;
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;
use crate::group::v1 as group_v1;
use crate::group::v2 as group_v2;
use crate::object_graph::{
    Expansion, HardLink, LinkIndex, ObjectGraph, ObjectId, ObjectIndex, ObjectTraversal,
    TraversalBudget, TraversalLimit,
};
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::SourceMetadata;

use self::counts::CountObservation;
use self::counts::CountQueries;
use self::counts::OrdinaryCountSubject;
use self::heap::CollectionContents;
use self::heap::HeapReferenceError;
use self::heap::associations::MemberAssociations;
use self::ownership::ObjectOwnership;
use self::physical::PhysicalStorage;
use self::physical::claims::PhysicalClaims;
use super::allocation::AllocationSession;
use super::attributes::AttributePredecessors;
use super::global_heap_liveness::DatasetReferenceScope;
use super::global_heap_liveness::GlobalHeapLiveness;
use super::reclaim::{self, MAX_LINK_WALK_DEPTH};
use super::staging::StagedEdits;
use super::{MAX_LINK_GRAPH_NODES, WriteEngine};

pub(super) use deletions::CountPreflight;
pub(super) use deletions::CountProjectionIdentity;
pub(super) use deletions::{
    CountInventory, CountPreflightError, CountProtectionError, ProtectedCounts,
};

pub(super) use deletions::DeclaredCountChange;
pub(super) use deletions::PlannedDeletion;
pub(super) use deletions::SourceRemovalPlan;
pub(super) use deletions::{
    CountHandoffError, NetCountError, NetCountPlan, OriginalOwnerError, OriginalOwnerFacts,
    OriginalOwnerInventory,
};

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "source-count evidence does not authorize allocation retirement"
    )
)]
pub(super) mod counts;
mod deletions;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "heap member facts do not authorize retirement")
)]
pub(in crate::edit) mod heap;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "local ownership facts do not authorize collection retirement"
    )
)]
pub(in crate::edit) mod ownership;

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "physical storage facts do not authorize retirement"
    )
)]
pub(in crate::edit) mod physical;

/// Lazy topology, object-count, reference, and storage queries for one source phase.
///
/// Queries borrow the engine for the phase. The caller establishes source consistency and starts
/// a new phase after relevant source writes. Owned count metadata can be retained after the phase
/// is dropped, with source freshness and write permissions established separately. Adjacency,
/// parsed count observations, reference identities, collection member framing, and local allocation
/// facts are retained separately. Reference and local ownership completeness are independent.
pub(super) struct AnalysisPhase<'a> {
    engine: &'a WriteEngine,
    /// Shared phase identity for reconstruction origins, predecessor observations, and topology.
    identity: Option<Arc<deletions::TopologyIdentity>>,
    topology: Option<Option<CompleteTopology<'a>>>,
    budget: TraversalBudget,
    root: Option<ObjectId>,
    links: BTreeMap<ObjectId, ObjectLinks>,
    /// Decoded collection and member identities, interpretation outcomes, and cached failures.
    references: BTreeMap<ObjectId, ObjectReferences>,
    /// Dataset-value references and outcomes, excluding attributes and fill values.
    value_references: BTreeMap<ObjectId, ObjectReferences>,
    liveness: Option<Result<GlobalHeapLiveness, Error>>,
    /// Complete collection ranges, allocated-member framing, and cached parsing failures.
    allocations: BTreeMap<PhysicalAddress, Result<CollectionContents, FormatError>>,
    /// Count observations, optional header allocation facts, and cached parse failures.
    counts: BTreeMap<ObjectId, Result<CountObservation, FormatError>>,
    /// Local allocation ranges and collection outcomes, including cached failures.
    ownership: BTreeMap<ObjectId, ObjectOwnership>,
    /// Local file framing, recorded metadata capacity, and cached problems.
    physical: Option<PhysicalStorage>,
    /// Physical range claims with their origins and independent support outcomes.
    claims: Option<PhysicalClaims>,
    /// Root-scoped member associations and independent reference and framing outcomes.
    associations: Option<MemberAssociations>,
    #[cfg(test)]
    candidate_lookups: u64,
}

impl<'a> AnalysisPhase<'a> {
    pub(super) fn new(engine: &'a WriteEngine) -> Self {
        Self {
            engine,
            identity: None,
            topology: None,
            budget: TraversalBudget::new(MAX_LINK_GRAPH_NODES as usize, usize::MAX),
            root: None,
            links: BTreeMap::new(),
            references: BTreeMap::new(),
            value_references: BTreeMap::new(),
            liveness: None,
            allocations: BTreeMap::new(),
            counts: BTreeMap::new(),
            ownership: BTreeMap::new(),
            physical: None,
            claims: None,
            associations: None,
            #[cfg(test)]
            candidate_lookups: 0,
        }
    }

    /// Creates a phase for a proposed physical root address after replacement writes.
    pub(super) fn prospective(engine: &'a WriteEngine, root: PhysicalAddress) -> Self {
        Self {
            root: Some(ObjectId::new(root)),
            ..Self::new(engine)
        }
    }

    /// Associates the supplied physical object identity with this analysis phase.
    ///
    /// The caller supplies the source object for the rebuilt records. Shares the lazily
    /// allocated phase identity with topology discovery, without reading the source or
    /// discovering topology.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `engine` is not the engine borrowed by this phase.
    pub(in crate::edit) fn reconstruction_source(
        &mut self,
        engine: &WriteEngine,
        object: ObjectId,
    ) -> Result<ReconstructionSource, FormatError> {
        if !std::ptr::eq(self.engine, engine) {
            return Err(FormatError::Internal(
                "a reconstructed header belongs to another source engine".into(),
            ));
        }
        #[cfg(test)]
        RECONSTRUCTION_WORK.with(|work| work.set(work.get() + 1));
        Ok(ReconstructionSource {
            identity: Arc::clone(
                self.identity
                    .get_or_insert_with(|| Arc::new(deletions::TopologyIdentity)),
            ),
            object,
        })
    }

    /// Returns complete topology, discovering it on the first query.
    ///
    /// Returns `None` if discovery fails, including on later queries in the same phase.
    pub(super) fn topology(&mut self) -> Option<&CompleteTopology<'a>> {
        let Self {
            engine,
            identity,
            topology,
            budget,
            root,
            links,
            references: _,
            value_references: _,
            liveness: _,
            allocations: _,
            counts: _,
            ownership: _,
            physical: _,
            claims: _,
            associations: _,
            #[cfg(test)]
                candidate_lookups: _,
        } = self;
        topology
            .get_or_insert_with(|| {
                let root = root
                    .map_or_else(
                        || {
                            ObjectId::resolve(
                                engine.addresses(),
                                engine.superblock().root_group_address,
                            )
                        },
                        Ok,
                    )
                    .ok()?;
                let identity = Arc::clone(
                    identity.get_or_insert_with(|| Arc::new(deletions::TopologyIdentity)),
                );
                CompleteTopology::discover_cached(engine, root, *budget, links, identity).ok()
            })
            .as_ref()
    }

    /// Returns a cached count observation at the object's physical header address.
    ///
    /// Parses the complete header on the first query, retaining no message bodies. Caches
    /// successful classifications and failures for subsequent queries in this source phase.
    /// Produces a count observation only after the parser and all chunk observations succeed.
    /// Topology, reference completeness, and allocation facts remain independent. The caller
    /// establishes count authority separately before retirement.
    ///
    /// # Errors
    ///
    /// Returns parsing and source-read errors from
    /// [`ObjectHeader::parse_from_source_with_chunk_observer`], or observation errors from
    /// [`ObjectReferenceCountObserver::observe`] and [`ObjectReferenceCountObserver::finish`].
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "count observations do not authorize allocation retirement"
        )
    )]
    pub(super) fn object_reference_count(
        &mut self,
        object: ObjectId,
    ) -> Result<ObjectReferenceCount, FormatError> {
        CountQueries::new(self.engine, &mut self.counts, None).observation(object)
    }

    /// Returns collection references reachable from the objects identified by `roots`.
    ///
    /// Reuses per-object adjacency and reference summaries. The completeness flag covers the
    /// supplied root set.
    ///
    /// # Errors
    ///
    /// Returns parsing and source-read errors from object headers, group links, and reference-bearing
    /// content.
    pub(super) fn collections_reachable_from(
        &mut self,
        roots: impl IntoIterator<Item = ObjectId>,
    ) -> Result<GlobalHeapLiveness, Error> {
        let mut referenced_collections = BTreeSet::new();
        let analysis_complete = self.scan_collections(roots, |_, references| {
            referenced_collections.extend(references.collections.iter().copied());
        })?;
        Ok(GlobalHeapLiveness {
            referenced_collections,
            analysis_complete,
        })
    }

    /// Returns collection candidates referenced by deleted objects and their descendants.
    ///
    /// Associates retained observations with this phase's shared identity. Retains decoded member
    /// identities and interpretation outcomes by original object. If discovery fails or is
    /// incomplete, records the supplied deletion roots and excludes all allocations. Otherwise,
    /// includes each collection whose complete allocation can be parsed.
    pub(super) fn deleted_collections(&mut self, deleted: &[ObjectId]) -> CollectionCandidates {
        if deleted.is_empty() {
            return CollectionCandidates::default();
        }
        let mut predecessors = BTreeMap::new();
        let complete = self.scan_collections(deleted.iter().copied(), |object, references| {
            if references.complete && references.objects.is_empty() {
                return;
            }
            predecessors.insert(
                CollectionProvenance {
                    object,
                    content: CollectionContent::Object,
                },
                references.clone(),
            );
        });
        let mut candidates = CollectionCandidates::default();
        if predecessors.is_empty() && matches!(complete, Ok(true)) {
            return candidates;
        }
        if matches!(complete, Ok(true)) {
            for (origin, references) in &predecessors {
                for &address in &references.collections {
                    if let Ok(allocation) =
                        self.collection_allocation(PhysicalAddress::new(address))
                    {
                        let Ok(()) = candidates.insert(allocation, [origin.clone()]) else {
                            return CollectionCandidates::default();
                        };
                    }
                }
            }
        }
        let Ok(observations) = candidates.require_phase(self) else {
            return CollectionCandidates::default();
        };
        observations.predecessors = predecessors;
        if !matches!(complete, Ok(true)) {
            observations
                .incomplete_deletions
                .extend(deleted.iter().copied());
        }
        candidates
    }

    /// Returns collection candidates referenced by replaced or removed attributes of `object`.
    ///
    /// Associates retained observations with this phase's shared identity. Retains decoded member
    /// identities and interpretation outcomes by `object` and original attribute name. Includes
    /// allocations only for completely interpreted attributes and collections whose complete
    /// allocations can be parsed in this phase.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`CollectionCandidates::insert`] for inconsistent candidate evidence.
    pub(super) fn attribute_collections(
        &mut self,
        object: ObjectId,
        predecessors: AttributePredecessors,
    ) -> Result<CollectionCandidates, FormatError> {
        let mut candidates = CollectionCandidates::default();
        for attribute in predecessors {
            let mut references = ObjectReferences::default();
            let outcome = self
                .engine
                .collect_parsed_attribute_collections(&attribute, &mut references);
            references.complete = matches!(outcome, Ok(true));
            references.error = outcome.err().map(HeapReferenceError::from);
            let origin = CollectionProvenance {
                object,
                content: CollectionContent::Attribute(attribute.name),
            };
            if references.complete {
                for &address in &references.collections {
                    if let Ok(allocation) =
                        self.collection_allocation(PhysicalAddress::new(address))
                    {
                        candidates.insert(allocation, [origin.clone()])?;
                    }
                }
            }
            candidates.observe(self, origin, references)?;
        }
        Ok(candidates)
    }

    /// Returns collection candidates referenced by the original values of staged overwrites.
    ///
    /// Associates retained observations with this phase's shared identity. Examines writes whose
    /// replacement datatype holds file addresses. Caches original dataset-value identities and
    /// interpretation outcomes, separately from attributes and fill
    /// values. Reuses these facts for object-wide reference queries in either query order.
    /// Includes allocations only for completely interpreted values and collections whose complete
    /// allocations can be parsed. The caller queries the unchanged source before overwrite writes.
    ///
    /// # Errors
    ///
    /// Returns path-resolution errors or [`Error::Format`] for inconsistent candidate evidence.
    /// Header, descriptor, and source failures encountered during value collection are recorded in
    /// the observations.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "overwrite candidates do not authorize retirement")
    )]
    pub(super) fn overwrite_collections(
        &mut self,
        staged: &StagedEdits,
    ) -> Result<CollectionCandidates, Error> {
        let mut candidates = CollectionCandidates::default();
        for (path, values) in staged.writes() {
            if !crate::datatype::datatype_holds_file_address(&values.dt) {
                continue;
            }
            let object = group_v2::resolve_path_any_from_source(
                &self.engine.opened_source(),
                AccessMode::ReadWrite,
                self.engine.superblock(),
                self.engine.widths(),
                &path.as_path(),
            )
            .map(PhysicalAddress::new)
            .map(ObjectId::new)?;
            let references = self
                .value_references
                .entry(object)
                .or_insert_with(|| {
                    let mut references = ObjectReferences::default();
                    let outcome = ObjectHeader::parse_from_source_with_widths(
                        &SourceMetadata(&self.engine.image()),
                        AccessMode::ReadWrite,
                        object.header_address().get(),
                        self.engine.widths(),
                        self.engine.addresses().effective_base(),
                    )
                    .map_err(HeapReferenceError::from)
                    .and_then(|header| {
                        self.engine.collect_dataset_collections(
                            &header.messages,
                            DatasetReferenceScope::Values,
                            &mut references,
                            None,
                        )
                    });
                    references.complete = matches!(outcome, Ok(true));
                    references.error = outcome.err();
                    references
                })
                .clone();
            let origin = CollectionProvenance {
                object,
                content: CollectionContent::DatasetValues,
            };
            if references.complete {
                for &address in &references.collections {
                    if let Ok(allocation) =
                        self.collection_allocation(PhysicalAddress::new(address))
                    {
                        candidates.insert(allocation, [origin.clone()])?;
                    }
                }
            }
            candidates.observe(self, origin, references)?;
        }
        Ok(candidates)
    }

    /// Parses complete parent and member framing on its first query in this phase.
    ///
    /// Reuses the retained framing or parsing failure for subsequent queries at `address`.
    /// Captures the allocation class when its physical address can be converted to a stored
    /// address.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::collection_contents`].
    pub(super) fn collection_allocation(
        &mut self,
        address: PhysicalAddress,
    ) -> Result<CollectionAllocation, FormatError> {
        let range = self.collection_contents(address)?.range();
        Ok(CollectionAllocation {
            session: self.engine.context.allocations.session(),
            range,
            class: self
                .engine
                .addresses()
                .relative(range.start())
                .ok()
                .map(|stored| {
                    reclaim::collection_free_class(
                        self.engine.space.page_size(),
                        stored,
                        range.length(),
                    )
                }),
        })
    }

    /// Returns allocations absent from the complete reference set for this phase's root.
    ///
    /// Returns an empty result if the union contains no allocations, the candidate session
    /// differs, or reference analysis fails or is incomplete. Reuses the root's reference result
    /// across calls in the same phase. Returns each range with its recorded allocation class,
    /// excluding allocations whose class is unknown.
    pub(super) fn unreferenced_collection_spans(
        &mut self,
        candidates: CollectionCandidates,
    ) -> Vec<(FileRange, FreeClass)> {
        self.unreferenced_collection_spans_with_roots(candidates, [])
    }

    /// Returns candidate allocations unreferenced from this phase's root and `roots`.
    ///
    /// Examines both root sets in one bounded traversal, reusing per-object summaries. Recomputes
    /// the cached reference result when `roots` is nonempty. Returns an empty result if the
    /// candidate session differs or reference analysis fails or is incomplete. Excludes
    /// allocations whose class is unknown.
    pub(super) fn unreferenced_collection_spans_with_roots(
        &mut self,
        candidates: CollectionCandidates,
        roots: impl IntoIterator<Item = ObjectId>,
    ) -> Vec<(FileRange, FreeClass)> {
        let CollectionCandidates {
            observations,
            collections,
        } = candidates;
        if observations.as_ref().is_some_and(|facts| {
            !facts
                .session
                .same(&self.engine.context.allocations.session())
        }) || collections.is_empty()
        {
            return Vec::new();
        }
        let mut roots = roots.into_iter().peekable();
        if self.liveness.is_none() || roots.peek().is_some() {
            let root = self.root.map_or_else(
                || {
                    ObjectId::resolve(
                        self.engine.addresses(),
                        self.engine.superblock().root_group_address,
                    )
                },
                Ok,
            );
            self.liveness =
                Some(root.map_err(Error::from).and_then(|root| {
                    self.collections_reachable_from(iter::once(root).chain(roots))
                }));
        }
        let Some(Ok(GlobalHeapLiveness {
            referenced_collections,
            analysis_complete: true,
        })) = &self.liveness
        else {
            return Vec::new();
        };
        collections
            .into_values()
            .filter_map(|candidate| {
                let CollectionCandidate {
                    range,
                    class,
                    origins: _,
                } = candidate;
                #[cfg(test)]
                {
                    self.candidate_lookups = self.candidate_lookups.saturating_add(1);
                }
                if referenced_collections.contains(&range.start().get()) {
                    return None;
                }
                Some((range, class?))
            })
            .collect()
    }

    /// Visits reference summaries for each distinct object reachable from `roots`.
    ///
    /// Returns whether reference-bearing content and bounded traversal were complete. Calls
    /// `visit` with an available reference summary before returning its recorded error or an
    /// adjacency discovery error. Reference interpretation completeness is independent of
    /// traversal completeness.
    ///
    /// # Errors
    ///
    /// Returns parsing and source-read errors from object headers, group links, and reference-bearing
    /// content.
    fn scan_collections(
        &mut self,
        roots: impl IntoIterator<Item = ObjectId>,
        mut visit: impl FnMut(ObjectId, &ObjectReferences),
    ) -> Result<bool, Error> {
        let mut traversal = ObjectTraversal::new(roots, self.budget, MAX_LINK_WALK_DEPTH);
        let mut complete = true;
        while let Some((object, depth)) = traversal.next() {
            let references = match self.read_reference_facts(object, true) {
                Ok(references) => references,
                Err(error) => {
                    if let Some(references) = self.references.get(&object) {
                        visit(object, references);
                    }
                    return Err(error);
                }
            };
            complete &= references.complete;
            visit(object, references);
            if let Some(error) = &references.error {
                return Err(error.clone().into());
            }
            if let Some(Some(topology)) = &self.topology
                && topology.indices.contains_key(&object)
            {
                for child in topology.children(object) {
                    traversal.push_child(child, depth);
                }
            } else if let Some(links) = self.links.get(&object) {
                for link in &links.links {
                    traversal.push_child(link.target, depth);
                }
            }
        }
        Ok(complete && traversal.complete())
    }

    /// Collects descriptor identities once, with optional group-adjacency discovery.
    ///
    /// Caches reference results and initial header parsing failures. A local query can precede
    /// adjacency discovery without repeating descriptor decoding. Reference interpretation and
    /// adjacency completeness remain separate.
    ///
    /// # Errors
    ///
    /// Returns group-link parsing errors when discovery is requested, a header parsing or source
    /// error when enriching existing facts with adjacency, or [`FormatError::Internal`] if the
    /// facts cannot be retrieved. Other header and content failures are recorded in
    /// [`ObjectReferences::error`].
    fn read_reference_facts(
        &mut self,
        object: ObjectId,
        discover_links: bool,
    ) -> Result<&ObjectReferences, Error> {
        let Self {
            engine,
            identity: _,
            topology,
            budget: _,
            root: _,
            links,
            references,
            value_references,
            liveness: _,
            allocations: _,
            counts: _,
            ownership: _,
            physical: _,
            claims: _,
            associations: _,
            #[cfg(test)]
                candidate_lookups: _,
        } = self;
        let has_topology = topology
            .as_ref()
            .and_then(Option::as_ref)
            .is_some_and(|topology| topology.indices.contains_key(&object));
        let need_links = discover_links && !has_topology && !links.contains_key(&object);
        let failed = references
            .get(&object)
            .is_some_and(|facts| facts.error.is_some());
        if !failed && (need_links || !references.contains_key(&object)) {
            match ObjectHeader::parse_from_source_with_widths(
                &SourceMetadata(&engine.image()),
                AccessMode::ReadWrite,
                object.header_address().get(),
                engine.widths(),
                engine.addresses().effective_base(),
            ) {
                Ok(header) => {
                    if need_links {
                        links.insert(object, ObjectLinks::read(engine, &header)?);
                    }
                    if let Entry::Vacant(entry) = references.entry(object) {
                        #[cfg(test)]
                        COLLECTION_WORK.with(|work| {
                            let (links, references) = work.get();
                            work.set((links, references + 1));
                        });
                        let mut facts = ObjectReferences::default();
                        let mut values = value_references.remove(&object);
                        let outcome = engine.collect_cached_object_collections(
                            &header,
                            &mut facts,
                            Some(&mut values),
                        );
                        if let Some(values) = values {
                            value_references.insert(object, values);
                        }
                        facts.complete = matches!(outcome, Ok(true));
                        facts.error = outcome.err();
                        entry.insert(facts);
                    }
                }
                Err(error) => {
                    if references.contains_key(&object) {
                        return Err(error.into());
                    }
                    references.insert(
                        object,
                        ObjectReferences {
                            error: Some(error.into()),
                            ..ObjectReferences::default()
                        },
                    );
                }
            }
        }
        references.get(&object).ok_or_else(|| {
            FormatError::Internal("object reference facts were not recorded".into()).into()
        })
    }
    /// Returns retained summary sizes and candidate-membership query counts.
    #[cfg(test)]
    fn work(&self) -> AnalysisWork {
        let Self {
            engine: _,
            identity: _,
            topology,
            budget: _,
            root: _,
            links,
            references,
            value_references: _,
            liveness,
            allocations,
            counts: _,
            ownership: _,
            physical: _,
            claims: _,
            associations: _,
            candidate_lookups,
        } = self;
        let topology = topology.as_ref().and_then(Option::as_ref);
        let objects = topology.map_or(0, |topology| topology.indices.len() as u64);
        let edges = topology.map_or(0, |topology| {
            topology
                .indices
                .values()
                .map(|&index| topology.graph.object(index).links.len() as u64)
                .sum()
        });
        let link_name_bytes = topology.map_or(0, |topology| {
            topology
                .indices
                .values()
                .map(|&index| {
                    topology
                        .graph
                        .object(index)
                        .value
                        .names
                        .keys()
                        .map(|name| name.len() as u64)
                        .sum::<u64>()
                })
                .sum()
        });
        AnalysisWork {
            topology_objects: objects,
            topology_edges: edges,
            link_name_bytes,
            cached_link_objects: links.len() as u64,
            reference_objects: references.len() as u64,
            collection_references: references
                .values()
                .map(|references| references.collections.len() as u64)
                .sum(),
            allocation_results: allocations.len() as u64,
            liveness_results: u64::from(liveness.is_some()),
            candidate_lookups: *candidate_lookups,
        }
    }
}

/// A physical object identity associated with one analysis phase.
///
/// Equality requires the same shared phase identity and the same [`ObjectId`]. The identity is
/// shared with any complete topology discovered in that phase. Source freshness, topology
/// completeness, allocation ownership, and permission to write require separate evidence.
#[derive(Clone, Debug)]
pub(in crate::edit) struct ReconstructionSource {
    identity: Arc<deletions::TopologyIdentity>,
    object: ObjectId,
}

impl ReconstructionSource {
    /// Returns the physical object-header identity supplied by the caller.
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }
}

impl Eq for ReconstructionSource {}

impl PartialEq for ReconstructionSource {
    fn eq(&self, other: &Self) -> bool {
        let Self { identity, object } = self;
        let Self {
            identity: other_identity,
            object: other_object,
        } = other;
        Arc::ptr_eq(identity, other_identity) && object == other_object
    }
}

/// The hard-link graph reachable from a phase root, with identities and incoming-link counts.
///
/// Each object is expanded once, and each hard-link edge is counted once. Link names are indexed
/// per owner. Payload-reference completeness and allocation ownership require separate proofs.
pub(super) struct CompleteTopology<'a> {
    identity: Arc<deletions::TopologyIdentity>,
    engine: &'a WriteEngine,
    graph: ObjectGraph<TopologyObject, ()>,
    indices: BTreeMap<ObjectId, ObjectIndex>,
    incoming: HashMap<ObjectId, u32>,
}

impl<'a> CompleteTopology<'a> {
    /// Discovers hard-link topology from the session's current root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TraversalLimitExceeded`] if discovery exceeds its object or link budget,
    /// or an incoming-link count exceeds [`u32::MAX`]. Returns parsing and source-read errors
    /// encountered during discovery.
    pub(super) fn discover(engine: &'a WriteEngine) -> Result<Self, Error> {
        Self::discover_with_budget(
            engine,
            TraversalBudget::new(MAX_LINK_GRAPH_NODES as usize, usize::MAX),
        )
    }

    fn discover_with_budget(
        engine: &'a WriteEngine,
        budget: TraversalBudget,
    ) -> Result<Self, Error> {
        let root = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)?;
        Self::discover_cached(
            engine,
            root,
            budget,
            &mut BTreeMap::new(),
            Arc::new(deletions::TopologyIdentity),
        )
    }

    /// Discovers topology under the supplied phase identity, reusing cached adjacency.
    ///
    /// The caller passes cached links and an identity from the same unchanged source phase.
    ///
    /// # Errors
    ///
    /// Returns the discovery errors described by [`Self::discover`].
    fn discover_cached(
        engine: &'a WriteEngine,
        root: ObjectId,
        budget: TraversalBudget,
        cached: &mut BTreeMap<ObjectId, ObjectLinks>,
        identity: Arc<deletions::TopologyIdentity>,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        DISCOVERY_WORK.with(|work| {
            let (passes, objects) = work.get();
            work.set((passes + 1, objects));
        });
        let mut incoming = HashMap::new();
        let (graph, indices) = ObjectGraph::discover_indexed(budget, root, |id| {
            #[cfg(test)]
            DISCOVERY_WORK.with(|work| {
                let (passes, objects) = work.get();
                work.set((passes, objects + 1));
            });
            let ObjectLinks {
                links,
                topology_error,
                count_subject,
            } = match cached.get(&id).cloned() {
                Some(links) => links,
                None => {
                    let header = ObjectHeader::parse_from_source_with_widths(
                        &SourceMetadata(&engine.image()),
                        AccessMode::ReadWrite,
                        id.header_address().get(),
                        engine.widths(),
                        engine.addresses().effective_base(),
                    )?;
                    ObjectLinks::read(engine, &header)?
                }
            };
            if let Some(error) = topology_error {
                return Err(error.into());
            }
            for link in &links {
                let count = incoming.entry(link.target).or_insert(0_u32);
                *count = count.checked_add(1).ok_or(Error::TraversalLimitExceeded {
                    limit: TraversalLimit::Links,
                    maximum: u32::MAX as usize,
                })?;
            }
            let mut names = BTreeMap::new();
            let links = links
                .into_iter()
                .enumerate()
                .map(|(position, link)| {
                    let HardLink { label, target } = link;
                    names.entry(label).or_insert(position);
                    HardLink { label: (), target }
                })
                .collect();
            Ok::<_, Error>(Expansion {
                value: TopologyObject {
                    id,
                    names,
                    count_subject,
                },
                links,
            })
        })
        .map_err(Error::from)?;
        cached.retain(|id, _| !indices.contains_key(id));
        Ok(Self {
            identity,
            engine,
            graph,
            indices,
            incoming,
        })
    }

    /// Appends allocations selected by the session's reclamation policy.
    ///
    /// Uses the reclamation policy of [`WriteEngine::collect_free_spans`], starting at `depth`.
    pub(super) fn collect_free_spans(
        &self,
        address: ObjectId,
        depth: u32,
        out: &mut Vec<(FileRange, FreeClass)>,
    ) {
        self.engine.collect_free_spans(address, depth, self, out);
    }

    /// Returns the number of hard links targeting an object.
    ///
    /// Returns `None` if the topology contains no link to `address`.
    pub(super) fn incoming(&self, address: ObjectId) -> Option<u32> {
        self.incoming.get(&address).copied()
    }

    /// Returns child identities in the owner's link order.
    ///
    /// Includes repeated targets. Returns an empty iterator if `address` is absent.
    pub(super) fn children(&self, address: ObjectId) -> impl Iterator<Item = ObjectId> + '_ {
        self.indices
            .get(&address)
            .into_iter()
            .flat_map(|&index| &self.graph.object(index).links)
            .map(|link| self.graph.object(link.target).value.id)
    }

    /// Follows indexed link names from the root, returning the final edge and target identity.
    ///
    /// For repeated names in one owner, follows the first link. Returns `None` for an empty path
    /// or a missing component. Each component requires an ordered-map lookup in its owner's index.
    pub(super) fn find_link(&self, path: &ObjectPathBuf) -> Option<(LinkIndex, ObjectId)> {
        let mut index = ObjectIndex::ROOT;
        let mut found = None;
        for name in path.components() {
            let object = self.graph.object(index);
            let &position = object.value.names.get(name.as_str())?;
            let target = object.links.get(position)?.target;
            found = Some((
                LinkIndex::new(index, position),
                self.graph.object(target).value.id,
            ));
            index = target;
        }
        found
    }
}

struct TopologyObject {
    id: ObjectId,
    names: BTreeMap<String, usize>,
    count_subject: Option<OrdinaryCountSubject>,
}

/// Named hard links, a compact count-subject marker, and any undefined-target error.
#[derive(Clone)]
struct ObjectLinks {
    links: Vec<HardLink<String, ObjectId>>,
    topology_error: Option<FormatError>,
    count_subject: Option<OrdinaryCountSubject>,
}

impl ObjectLinks {
    fn read(engine: &WriteEngine, header: &ObjectHeader) -> Result<Self, Error> {
        #[cfg(test)]
        COLLECTION_WORK.with(|work| {
            let (links, references) = work.get();
            work.set((links + 1, references));
        });
        let mut links = Vec::new();
        let mut topology_error = None;
        if group_v2::is_group(header) {
            for entry in group_v2::resolve_group_entries_from_source(
                &engine.opened_source(),
                AccessMode::ReadWrite,
                header,
                engine.widths(),
                group_v1::group_leaf_node_k(engine.superblock()),
            )? {
                if entry
                    .object_header_address
                    .is_undefined(engine.widths().offsets)
                {
                    topology_error = Some(FormatError::DefinedAddressIsUndefined {
                        address: entry.object_header_address.get(),
                        offset_size: engine.widths().offsets.get(),
                    });
                    continue;
                }
                links.push(HardLink {
                    label: entry.name,
                    target: ObjectId::resolve(engine.addresses(), entry.object_header_address)?,
                });
            }
        }
        Ok(Self {
            links,
            topology_error,
            count_subject: OrdinaryCountSubject::from_header(header),
        })
    }
}

/// Decoded global heap references from an object's content or its dataset values alone.
///
/// Retains checked identities even when other content is unsupported or malformed. Completeness
/// covers reference interpretation independently of adjacency and collection framing. Logical
/// ownership, external references, and retirement permission require separate evidence.
#[derive(Clone, Default)]
pub(in crate::edit) struct ObjectReferences {
    /// Distinct physical collection addresses.
    collections: BTreeSet<u64>,
    /// Distinct checked collection and member identities.
    objects: BTreeSet<GlobalHeapId>,
    /// Whether reference-bearing content was fully interpreted without an error.
    complete: bool,
    /// A header, descriptor, or source failure, with partial identities retained separately.
    error: Option<HeapReferenceError>,
}

/// Global heap allocation candidates and predecessor observations from one editing session.
///
/// The editor associates predecessor observations with their original [`AnalysisPhase`] and
/// checks that association when merging candidates. For allocation-only insertions, the editor
/// records the session without a phase identity or predecessor observations.
///
/// The editor retains complete allocation ranges and their original classes separately from
/// descriptor identities, interpretation outcomes, and incomplete deletion roots. The caller
/// establishes source freshness, member presence, allocation ownership, and retirement permission
/// separately. An absent observation is not evidence of reference absence.
#[derive(Default)]
pub(super) struct CollectionCandidates {
    /// Lazily allocated session, optional source-phase identity, and reference observations.
    observations: Option<Box<CandidateObservations>>,
    /// Complete allocation candidates, keyed by physical collection address.
    collections: BTreeMap<PhysicalAddress, CollectionCandidate>,
}

impl CollectionCandidates {
    /// Merges allocation candidates, reference observations, and incomplete deletion roots.
    ///
    /// Preserves each allocation's range and class. Adopts `other` directly if `self` is empty.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the sessions or recorded source phases differ, before
    /// merging allocations or observations. Also returns the errors of [`Self::insert`]. Entries
    /// merged before an allocation error remain in `self`.
    pub(super) fn extend(&mut self, other: Self) -> Result<(), FormatError> {
        if self.observations.is_none() {
            debug_assert!(self.collections.is_empty());
            *self = other;
            return Ok(());
        }
        let Self {
            observations,
            collections,
        } = other;
        let Some(observations) = observations else {
            debug_assert!(collections.is_empty());
            return Ok(());
        };
        let CandidateObservations {
            session,
            identity,
            predecessors,
            incomplete_deletions,
        } = *observations;
        self.require_identity(session.clone(), identity.as_ref())?;
        for CollectionCandidate {
            range,
            class,
            origins,
        } in collections.into_values()
        {
            self.insert(
                CollectionAllocation {
                    session: session.clone(),
                    range,
                    class,
                },
                origins,
            )?;
        }
        let observations = self.require_session(session)?;
        for (origin, references) in predecessors {
            observations.observe(origin, references);
        }
        observations
            .incomplete_deletions
            .extend(incomplete_deletions);
        Ok(())
    }

    /// Merges origins for one complete allocation, preserving its original range and class.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `origins` is empty, the allocation belongs to another
    /// session, or an existing candidate at the same address has a different length or class.
    pub(super) fn insert(
        &mut self,
        allocation: CollectionAllocation,
        origins: impl IntoIterator<Item = CollectionProvenance>,
    ) -> Result<(), FormatError> {
        let mut origins = origins.into_iter();
        let first = origins.next().ok_or_else(|| {
            FormatError::Internal("collection candidate has no provenance".into())
        })?;
        let CollectionAllocation {
            session,
            range,
            class,
        } = allocation;
        self.require_session(session)?;
        let candidate =
            self.collections
                .entry(range.start())
                .or_insert_with(|| CollectionCandidate {
                    range,
                    class,
                    origins: BTreeSet::new(),
                });
        if candidate.range != range {
            return Err(FormatError::Internal(
                "collection candidates have conflicting allocation lengths".into(),
            ));
        }
        if candidate.class != class {
            return Err(FormatError::Internal(
                "collection candidates have conflicting allocation classes".into(),
            ));
        }
        candidate.origins.insert(first);
        candidate.origins.extend(origins);
        Ok(())
    }

    /// Merges decoded identities and interpretation outcomes for one original content scope.
    ///
    /// Skips complete, reference-free summaries. Associates other observations with `phase`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the recorded session or source phase differs from
    /// `phase`.
    fn observe(
        &mut self,
        phase: &mut AnalysisPhase<'_>,
        origin: CollectionProvenance,
        references: ObjectReferences,
    ) -> Result<(), FormatError> {
        if references.complete && references.objects.is_empty() {
            return Ok(());
        }
        self.require_phase(phase)?.observe(origin, references);
        Ok(())
    }

    /// Allocates observation storage on demand and shares `phase`'s identity.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the recorded session or source phase differs from
    /// `phase`.
    fn require_phase(
        &mut self,
        phase: &mut AnalysisPhase<'_>,
    ) -> Result<&mut CandidateObservations, FormatError> {
        self.require_identity(
            phase.engine.context.allocations.session(),
            Some(
                phase
                    .identity
                    .get_or_insert_with(|| Arc::new(deletions::TopologyIdentity)),
            ),
        )
    }

    /// Allocates observation storage on its first use in `session`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the recorded session differs from `session`.
    fn require_session(
        &mut self,
        session: AllocationSession,
    ) -> Result<&mut CandidateObservations, FormatError> {
        self.require_identity(session, None)
    }

    /// Allocates observation storage on demand and retains a supplied phase identity.
    ///
    /// Compares supplied and recorded phase identities by shared allocation. Preserves a recorded
    /// identity when `identity` is [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the recorded session differs from `session`, or both
    /// phase identities are present and differ.
    fn require_identity(
        &mut self,
        session: AllocationSession,
        identity: Option<&Arc<deletions::TopologyIdentity>>,
    ) -> Result<&mut CandidateObservations, FormatError> {
        if let Some(prior) = &self.observations {
            if !prior.session.same(&session) {
                return Err(FormatError::Internal(
                    "collection candidates belong to different sessions".into(),
                ));
            }
            if let (Some(prior), Some(identity)) = (&prior.identity, identity) {
                if !Arc::ptr_eq(prior, identity) {
                    return Err(FormatError::Internal(
                        "collection candidates belong to different source phases".into(),
                    ));
                }
            }
        }
        let observations = self.observations.get_or_insert_with(|| {
            Box::new(CandidateObservations {
                session,
                identity: None,
                predecessors: BTreeMap::new(),
                incomplete_deletions: BTreeSet::new(),
            })
        });
        if observations.identity.is_none() {
            observations.identity = identity.cloned();
        }
        Ok(observations)
    }
}

/// Descriptor observations and incomplete deletion scopes, bound to their original source phase.
///
/// For allocation-only candidates, the editor records a session without a phase identity or
/// observations.
struct CandidateObservations {
    session: AllocationSession,
    /// The original phase's shared identity, absent for allocation-only candidates.
    identity: Option<Arc<deletions::TopologyIdentity>>,
    /// Decoded identities and interpretation outcomes, keyed by original object and content scope.
    predecessors: BTreeMap<CollectionProvenance, ObjectReferences>,
    /// Supplied deletion roots whose descendant traversal failed or was incomplete.
    incomplete_deletions: BTreeSet<ObjectId>,
}

impl CandidateObservations {
    /// Merges decoded identities and interpretation outcomes for one original content scope.
    ///
    /// Retains all decoded identities and the first error. Records completeness only if every
    /// interpretation is complete.
    fn observe(&mut self, origin: CollectionProvenance, references: ObjectReferences) {
        match self.predecessors.entry(origin) {
            Entry::Vacant(entry) => {
                entry.insert(references);
            }
            Entry::Occupied(mut entry) => {
                let prior = entry.get_mut();
                prior.collections.extend(references.collections);
                prior.objects.extend(references.objects);
                prior.complete &= references.complete;
                if prior.error.is_none() {
                    prior.error = references.error;
                }
            }
        }
    }
}

/// The complete physical range of a parsed collection, bound to its editing session.
///
/// Retains the original allocation class when its physical address can be converted to a stored
/// address. The caller preserves the allocation while using its evidence across source-content
/// phases.
pub(super) struct CollectionAllocation {
    session: AllocationSession,
    range: FileRange,
    /// The original allocation class, or [`None`] if stored-address conversion failed.
    class: Option<FreeClass>,
}

/// One complete collection allocation with its original class and recorded origins.
struct CollectionCandidate {
    range: FileRange,
    /// The original allocation class, or [`None`] if stored-address conversion failed.
    class: Option<FreeClass>,
    origins: BTreeSet<CollectionProvenance>,
}

/// The original object and content scope associated with an allocation or observation.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct CollectionProvenance {
    /// The object from which the allocation or reference summary was collected.
    pub(super) object: ObjectId,
    /// The part of the original object examined for references.
    pub(super) content: CollectionContent,
}

/// The part of an object examined for predecessor references.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum CollectionContent {
    /// One attribute, identified by its original name.
    Attribute(String),
    /// Stored dataset values, excluding attributes and fill values.
    DatasetValues,
    /// All supported object content, including attributes and fill values.
    Object,
}

/// Retained phase summaries and candidate-membership query counts.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AnalysisWork {
    /// Objects in complete topology.
    topology_objects: u64,
    /// Hard-link edges in complete topology.
    topology_edges: u64,
    /// Bytes in the topology's indexed link names.
    link_name_bytes: u64,
    /// Entries in the separate object-link cache.
    cached_link_objects: u64,
    /// Cached object-reference summaries.
    reference_objects: u64,
    /// Collection addresses summed across cached object-reference summaries.
    collection_references: u64,
    /// Cached collection-allocation results, including errors.
    allocation_results: u64,
    /// Cached root reference results, either zero or one.
    liveness_results: u64,
    /// Candidate-membership queries against a complete reference set.
    candidate_lookups: u64,
}

#[cfg(test)]
std::thread_local! {
    pub(in crate::edit) static DISCOVERY_WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    pub(in crate::edit) static RECONSTRUCTION_WORK: Cell<usize> = const { Cell::new(0) };
    pub(in crate::edit) static COLLECTION_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    pub(in crate::edit) static COLLECTION_WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

#[cfg(test)]
mod tests;
