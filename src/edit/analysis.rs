//! Lazy hard-link topology for a borrowed editing session.

#[cfg(test)]
use core::cell::Cell;
use std::collections::{BTreeMap, HashMap};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use crate::access_mode::AccessMode;
use crate::error::Error;
use crate::group_v1;
use crate::group_v2;
use crate::object_graph::{
    Expansion, HardLink, LinkIndex, ObjectGraph, ObjectId, ObjectIndex, TraversalBudget,
    TraversalLimit,
};
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::SourceMetadata;

use super::{MAX_LINK_GRAPH_NODES, WriteEngine};

/// Lazy topology queries for one phase with unchanged source contents.
///
/// The engine remains borrowed while its topology facts are used. The caller establishes source
/// consistency separately.
pub(super) struct AnalysisPhase<'a> {
    engine: &'a WriteEngine,
    topology: Option<Option<CompleteTopology<'a>>>,
    budget: TraversalBudget,
}

impl<'a> AnalysisPhase<'a> {
    pub(super) fn new(engine: &'a WriteEngine) -> Self {
        Self {
            engine,
            topology: None,
            budget: TraversalBudget::new(MAX_LINK_GRAPH_NODES as usize, usize::MAX),
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
        } = self;
        topology
            .get_or_insert_with(|| CompleteTopology::discover_with_budget(engine, *budget).ok())
            .as_ref()
    }
}

/// The hard-link graph reachable from a session's root, with identities and incoming-link counts.
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
        #[cfg(test)]
        DISCOVERY_WORK.with(|work| {
            let (passes, objects) = work.get();
            work.set((passes + 1, objects));
        });
        let root = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)?;
        let mut incoming = HashMap::new();
        let (graph, indices) = ObjectGraph::discover_indexed(budget, root, |id| {
            #[cfg(test)]
            DISCOVERY_WORK.with(|work| {
                let (passes, objects) = work.get();
                work.set((passes, objects + 1));
            });
            let header = ObjectHeader::parse_from_source_with_widths(
                &SourceMetadata(&engine.image()),
                AccessMode::ReadWrite,
                id.header_address().get(),
                engine.widths(),
                engine.addresses().effective_base(),
            )?;
            let mut names = BTreeMap::new();
            let mut links = Vec::new();
            if group_v2::is_group(&header) {
                for entry in group_v2::resolve_group_entries_from_source(
                    &engine.opened_source(),
                    AccessMode::ReadWrite,
                    &header,
                    engine.widths(),
                    group_v1::group_leaf_node_k(engine.superblock()),
                )? {
                    names.entry(entry.name).or_insert(links.len());
                    let target =
                        ObjectId::resolve(engine.addresses(), entry.object_header_address)?;
                    let count = incoming
                        .entry(target.header_address().get())
                        .or_insert(0_u32);
                    *count = count.checked_add(1).ok_or(Error::TraversalLimitExceeded {
                        limit: TraversalLimit::Links,
                        maximum: u32::MAX as usize,
                    })?;
                    links.push(HardLink { label: (), target });
                }
            }
            Ok::<_, Error>(Expansion {
                value: TopologyObject { id, names },
                links,
            })
        })
        .map_err(Error::from)?;
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

#[cfg(test)]
std::thread_local! {
    static DISCOVERY_WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

#[cfg(test)]
mod tests;
