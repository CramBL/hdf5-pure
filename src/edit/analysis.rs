//! Phase-scoped topology, collection references, and allocation facts for an editing session.

#[cfg(test)]
use core::cell::Cell;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use crate::access_mode::AccessMode;
use crate::error::{Error, FormatError};
use crate::group_v1;
use crate::group_v2;
use crate::object_graph::{
    Expansion, HardLink, LinkIndex, ObjectGraph, ObjectId, ObjectIndex, ObjectTraversal,
    TraversalBudget, TraversalLimit,
};
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::SourceMetadata;

use super::allocation::AllocationSession;
use super::global_heap_liveness::GlobalHeapLiveness;
use super::reclaim::{self, MAX_LINK_WALK_DEPTH};
use super::{MAX_LINK_GRAPH_NODES, WriteEngine};

/// Lazy topology, reference, and allocation queries for unchanged source contents.
///
/// The engine remains borrowed while its facts are used. The caller establishes source consistency
/// and starts a new phase after relevant source writes. Adjacency, reference summaries, and collection
/// geometry are retained separately, with a completeness result for each required reference scan.
pub(super) struct AnalysisPhase<'a> {
    engine: &'a WriteEngine,
    topology: Option<Option<CompleteTopology<'a>>>,
    budget: TraversalBudget,
    root: Option<ObjectId>,
    links: BTreeMap<ObjectId, ObjectLinks>,
    references: BTreeMap<ObjectId, ObjectReferences>,
    liveness: Option<Result<GlobalHeapLiveness, Error>>,
    allocations: BTreeMap<PhysicalAddress, Result<FileRange, FormatError>>,
    #[cfg(test)]
    candidate_lookups: u64,
}

impl<'a> AnalysisPhase<'a> {
    pub(super) fn new(engine: &'a WriteEngine) -> Self {
        Self {
            engine,
            topology: None,
            budget: TraversalBudget::new(MAX_LINK_GRAPH_NODES as usize, usize::MAX),
            root: None,
            links: BTreeMap::new(),
            references: BTreeMap::new(),
            liveness: None,
            allocations: BTreeMap::new(),
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

    /// Returns complete topology, discovering it on the first query.
    ///
    /// Returns `None` if discovery fails, including on later queries in the same phase.
    pub(super) fn topology(&mut self) -> Option<&CompleteTopology<'a>> {
        let Self {
            engine,
            topology,
            budget,
            root,
            links,
            references: _,
            liveness: _,
            allocations: _,
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
                CompleteTopology::discover_cached(engine, root, *budget, links).ok()
            })
            .as_ref()
    }

    /// Returns collection references reachable from physical object-header addresses in `roots`.
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
        roots: impl IntoIterator<Item = u64>,
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
    /// `deleted` contains physical object-header addresses. Returns an empty union if reference
    /// discovery fails or is incomplete. Collections that cannot be parsed are excluded.
    pub(super) fn deleted_collections(&mut self, deleted: &[u64]) -> CollectionCandidates {
        if deleted.is_empty() {
            return CollectionCandidates::default();
        }
        let mut origins: BTreeMap<PhysicalAddress, BTreeSet<CollectionProvenance>> =
            BTreeMap::new();
        let Ok(true) = self.scan_collections(deleted.iter().copied(), |object, references| {
            for &address in &references.collections {
                origins
                    .entry(PhysicalAddress::new(address))
                    .or_default()
                    .insert(CollectionProvenance {
                        object,
                        attribute: None,
                    });
            }
        }) else {
            return CollectionCandidates::default();
        };
        let mut candidates = CollectionCandidates::default();
        for (address, origins) in origins {
            if let Ok(allocation) = self.collection_allocation(address) {
                let Ok(()) = candidates.insert(allocation, origins) else {
                    return CollectionCandidates::default();
                };
            }
        }
        candidates
    }

    /// Parses complete collection geometry on its first query in this phase.
    ///
    /// Retains successful ranges and parsing failures for subsequent queries at `address`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`WriteEngine::collection_span`].
    pub(super) fn collection_allocation(
        &mut self,
        address: PhysicalAddress,
    ) -> Result<CollectionAllocation, FormatError> {
        let Self {
            engine,
            allocations,
            topology: _,
            budget: _,
            root: _,
            links: _,
            references: _,
            liveness: _,
            #[cfg(test)]
                candidate_lookups: _,
        } = self;
        let range = allocations.entry(address).or_insert_with(|| {
            #[cfg(test)]
            COLLECTION_ALLOCATIONS.with(|count| count.set(count.get() + 1));
            engine.collection_span(address.get())
        });
        range
            .as_ref()
            .map(|&range| CollectionAllocation {
                session: engine.context.allocations.session(),
                range,
            })
            .map_err(Clone::clone)
    }

    /// Returns allocations absent from the complete reference set for this phase's root.
    ///
    /// Returns an empty result for an empty union, a different candidate session, or failed or
    /// incomplete reference analysis. Reuses the root's reference result across calls in the same
    /// phase. The editor classifies each returned range with [`reclaim::collection_free_class`].
    pub(super) fn unreferenced_collection_spans(
        &mut self,
        candidates: CollectionCandidates,
    ) -> Vec<(FileRange, FreeClass)> {
        let CollectionCandidates {
            session,
            collections,
        } = candidates;
        if session
            .as_ref()
            .is_some_and(|session| !session.same(&self.engine.context.allocations.session()))
            || collections.is_empty()
        {
            return Vec::new();
        }
        if self.liveness.is_none() {
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
                    self.collections_reachable_from([root.header_address().get()])
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
                let CollectionCandidate { range, origins: _ } = candidate;
                #[cfg(test)]
                {
                    self.candidate_lookups = self.candidate_lookups.saturating_add(1);
                }
                if referenced_collections.contains(&range.start().get()) {
                    return None;
                }
                let stored = self.engine.addresses().relative(range.start()).ok()?;
                Some((
                    range,
                    reclaim::collection_free_class(
                        self.engine.space.page_size(),
                        stored,
                        range.length(),
                    ),
                ))
            })
            .collect()
    }

    /// Visits reference summaries for each distinct object reachable from `roots`.
    ///
    /// Returns whether reference-bearing content and bounded traversal were complete.
    ///
    /// # Errors
    ///
    /// Returns parsing and source-read errors from object headers, group links, and reference-bearing
    /// content.
    fn scan_collections(
        &mut self,
        roots: impl IntoIterator<Item = u64>,
        mut visit: impl FnMut(ObjectId, &ObjectReferences),
    ) -> Result<bool, Error> {
        let mut traversal = ObjectTraversal::new(
            roots
                .into_iter()
                .map(|address| ObjectId::new(PhysicalAddress::new(address))),
            self.budget,
            MAX_LINK_WALK_DEPTH,
        );
        let mut complete = true;
        while let Some((object, depth)) = traversal.next() {
            let references = self.read_references(object)?;
            complete &= references.complete;
            visit(object, references);
            if let Some(Some(topology)) = &self.topology
                && topology.indices.contains_key(&object)
            {
                for child in topology.children(object.header_address().get()) {
                    traversal.push_child(ObjectId::new(PhysicalAddress::new(child)), depth);
                }
            } else if let Some(links) = self.links.get(&object) {
                for link in &links.links {
                    traversal.push_child(link.target, depth);
                }
            }
        }
        Ok(complete && traversal.complete())
    }

    /// Parses an object's reference-bearing content on its first query in this phase.
    ///
    /// # Errors
    ///
    /// Returns parsing and source-read errors from the object header, its group links, or its
    /// reference-bearing content.
    fn read_references(&mut self, object: ObjectId) -> Result<&ObjectReferences, Error> {
        let Self {
            engine,
            topology,
            budget: _,
            root: _,
            links,
            references,
            liveness: _,
            allocations: _,
            #[cfg(test)]
                candidate_lookups: _,
        } = self;
        match references.entry(object) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let header = ObjectHeader::parse_from_source_with_widths(
                    &SourceMetadata(&engine.image()),
                    AccessMode::ReadWrite,
                    object.header_address().get(),
                    engine.widths(),
                    engine.addresses().effective_base(),
                )?;
                let has_topology = topology
                    .as_ref()
                    .and_then(Option::as_ref)
                    .is_some_and(|topology| topology.indices.contains_key(&object));
                if !has_topology && !links.contains_key(&object) {
                    links.insert(object, ObjectLinks::read(engine, &header)?);
                }
                #[cfg(test)]
                COLLECTION_WORK.with(|work| {
                    let (links, references) = work.get();
                    work.set((links, references + 1));
                });
                let mut collections = BTreeSet::new();
                let complete = engine.collect_object_collections(&header, &mut collections)?;
                Ok(entry.insert(ObjectReferences {
                    collections,
                    complete,
                }))
            }
        }
    }
    /// Returns retained summary sizes and candidate-membership query counts.
    #[cfg(test)]
    fn work(&self) -> AnalysisWork {
        let Self {
            engine: _,
            topology,
            budget: _,
            root: _,
            links,
            references,
            liveness,
            allocations,
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

/// The hard-link graph reachable from a phase root, with identities and incoming-link counts.
///
/// Each object is expanded once, and each hard-link edge is counted once. Link names are indexed
/// per owner. Payload-reference completeness and allocation ownership require separate proofs.
pub(super) struct CompleteTopology<'a> {
    engine: &'a WriteEngine,
    graph: ObjectGraph<TopologyObject, ()>,
    indices: BTreeMap<ObjectId, ObjectIndex>,
    incoming: HashMap<u64, u32>,
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
        Self::discover_cached(engine, root, budget, &mut BTreeMap::new())
    }

    fn discover_cached(
        engine: &'a WriteEngine,
        root: ObjectId,
        budget: TraversalBudget,
        cached: &mut BTreeMap<ObjectId, ObjectLinks>,
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
                let count = incoming
                    .entry(link.target.header_address().get())
                    .or_insert(0_u32);
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
                value: TopologyObject { id, names },
                links,
            })
        })
        .map_err(Error::from)?;
        cached.retain(|id, _| !indices.contains_key(id));
        Ok(Self {
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
        address: u64,
        depth: u32,
        out: &mut Vec<(FileRange, FreeClass)>,
    ) {
        self.engine.collect_free_spans(address, depth, self, out);
    }

    /// Returns the number of hard links targeting a physical object-header address.
    ///
    /// Returns `None` if the topology contains no link to `address`.
    pub(super) fn incoming(&self, address: u64) -> Option<u32> {
        self.incoming.get(&address).copied()
    }

    /// Returns physical child-header addresses in the owner's link order.
    ///
    /// Includes repeated targets. Returns an empty iterator if `address` is absent.
    pub(super) fn children(&self, address: u64) -> impl Iterator<Item = u64> + '_ {
        self.indices
            .get(&ObjectId::new(PhysicalAddress::new(address)))
            .into_iter()
            .flat_map(|&index| &self.graph.object(index).links)
            .map(|link| {
                self.graph
                    .object(link.target)
                    .value
                    .id
                    .header_address()
                    .get()
            })
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
}

/// Named hard links and any undefined-target error from one parsed object header.
#[derive(Clone)]
struct ObjectLinks {
    links: Vec<HardLink<String, ObjectId>>,
    topology_error: Option<FormatError>,
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
        })
    }
}

/// Physical collection addresses referenced by one object's content, with its completeness result.
struct ObjectReferences {
    collections: BTreeSet<u64>,
    complete: bool,
}

/// Global heap allocations, keyed by physical address, and their object or attribute origins.
///
/// The editor retains a complete allocation range from one editing session for each candidate.
/// Reference absence is established separately against the prospective root.
#[derive(Default)]
pub(super) struct CollectionCandidates {
    session: Option<AllocationSession>,
    collections: BTreeMap<PhysicalAddress, CollectionCandidate>,
}

impl CollectionCandidates {
    /// Merges origins for one complete allocation, preserving its original range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `origins` is empty, the allocation belongs to another
    /// session, or an existing candidate at the same address has a different length.
    pub(super) fn insert(
        &mut self,
        allocation: CollectionAllocation,
        origins: impl IntoIterator<Item = CollectionProvenance>,
    ) -> Result<(), FormatError> {
        let mut origins = origins.into_iter();
        let first = origins.next().ok_or_else(|| {
            FormatError::Internal("collection candidate has no provenance".into())
        })?;
        let Self {
            session: prior_session,
            collections,
        } = self;
        let CollectionAllocation { session, range } = allocation;
        if let Some(prior) = prior_session {
            if !prior.same(&session) {
                return Err(FormatError::Internal(
                    "collection candidates belong to different sessions".into(),
                ));
            }
        } else {
            *prior_session = Some(session);
        }
        let candidate = collections
            .entry(range.start())
            .or_insert_with(|| CollectionCandidate {
                range,
                origins: BTreeSet::new(),
            });
        if candidate.range != range {
            return Err(FormatError::Internal(
                "collection candidates have conflicting allocation lengths".into(),
            ));
        }
        candidate.origins.insert(first);
        candidate.origins.extend(origins);
        Ok(())
    }
}

/// The complete physical range of a parsed collection, bound to its editing session.
///
/// The caller preserves the allocation while using its evidence across source-content phases.
pub(super) struct CollectionAllocation {
    session: AllocationSession,
    range: FileRange,
}

/// One complete collection allocation and its recorded origins.
struct CollectionCandidate {
    range: FileRange,
    origins: BTreeSet<CollectionProvenance>,
}

/// The object and optional attribute associated with a retirement candidate.
#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct CollectionProvenance {
    /// The object from which the candidate was collected.
    pub(super) object: ObjectId,
    /// The attribute name, or [`None`] for an object-wide candidate.
    pub(super) attribute: Option<String>,
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
    static DISCOVERY_WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    static COLLECTION_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static COLLECTION_WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

#[cfg(test)]
mod tests;
