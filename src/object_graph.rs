//! The objects reachable from one object through hard links, discovered without recursion.
//!
//! A file may link one object from several groups, and a group from inside its own subtree.
//! [`ObjectGraph::discover`] expands each object once, keyed by the address of its object header,
//! and stops at a [`TraversalBudget`] of objects and hard links. [`ObjectGraph::into_tree`]
//! accepts the graph as a tree only if no link targets the root and no two links target one
//! object.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use hdf5_pure_core::__private::PhysicalAddress;

use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::error::FormatError;

/// The objects reachable from a root object through hard links, each with its value and links.
///
/// The graph holds each object once, however many links target it, and a link targets an object
/// by its [`ObjectIndex`]. The root has index 0.
pub(crate) struct ObjectGraph<T> {
    // `discover` expands the objects in admission order, so `objects[i]` is `ObjectIndex(i)`.
    objects: Vec<GraphObject<T>>,
}

impl<T> ObjectGraph<T> {
    /// Discovers the objects reachable from `root`, calling `expand` once for each.
    ///
    /// `expand` returns the value of an object and the hard links it holds. The walk expands the
    /// objects from a [`Worklist`] in the order it first reaches them, so a graph of any depth
    /// takes constant stack space. A link to an object the walk has reached before targets that
    /// object, so a cycle or a second link to one object adds a link and no object.
    ///
    /// # Errors
    ///
    /// Returns [`TraversalError::LimitExceeded`] if the graph holds more objects or more hard
    /// links than `budget` allows, and [`TraversalError::Expand`] with the first error `expand`
    /// returns.
    pub(crate) fn discover<E>(
        budget: TraversalBudget,
        root: ObjectId,
        mut expand: impl FnMut(ObjectId) -> Result<Expansion<T>, E>,
    ) -> Result<Self, TraversalError<E>> {
        let mut worklist = Worklist::new(budget.objects);
        worklist.admit(root)?;
        let mut link_count = 0_usize;
        while let Some(id) = worklist.next() {
            let Expansion { value, links } = expand(id).map_err(TraversalError::Expand)?;
            link_count = link_count
                .checked_add(links.len())
                .filter(|&count| count <= budget.links)
                .ok_or(TraversalError::LimitExceeded {
                    limit: TraversalLimit::Links,
                    maximum: budget.links,
                })?;
            let links = links
                .into_iter()
                .map(
                    |HardLink {
                         name,
                         creation_order,
                         target,
                     }| {
                        Ok(HardLink {
                            name,
                            creation_order,
                            target: worklist.admit(target)?,
                        })
                    },
                )
                .collect::<Result<_, TraversalError<E>>>()?;
            worklist.expanded.push(GraphObject { value, links });
        }
        Ok(Self {
            objects: worklist.expanded,
        })
    }

    /// Converts the graph into an [`ObjectTree`] if each object other than the root is the target
    /// of one link and the root of none.
    ///
    /// # Errors
    ///
    /// Returns the graph unchanged if a link targets the root or a second link targets an object.
    pub(crate) fn into_tree(self) -> Result<ObjectTree<T>, Self> {
        let Some(order) = self.tree_post_order() else {
            return Err(self);
        };
        // A tree post order lists every object once, so each index into `position` is in range.
        let mut position = vec![0; order.len()];
        for (post_order, index) in order.into_iter().enumerate() {
            position[index.get()] = post_order;
        }
        let mut objects: Vec<_> = self
            .objects
            .into_iter()
            .zip(&position)
            .map(|(mut object, &post_order)| {
                for link in &mut object.links {
                    link.target = ObjectIndex(position[link.target.get()]);
                }
                (post_order, object)
            })
            .collect();
        objects.sort_unstable_by_key(|&(post_order, _)| post_order);
        let mut descendants: Vec<_> = objects.into_iter().map(|(_, object)| object).collect();
        // `discover` admits the root before anything else, so the graph holds at least one object,
        // and the post order puts the root last.
        let root = descendants.pop().expect("an object graph holds its root");
        Ok(ObjectTree { descendants, root })
    }

    /// Returns the objects in post order, each after the subtrees of its links in link order, or
    /// `None` if a link targets the root or a second link targets an object.
    fn tree_post_order(&self) -> Option<Vec<ObjectIndex>> {
        let mut linked = vec![false; self.objects.len()];
        linked[ObjectIndex::ROOT.get()] = true;
        for link in self.objects.iter().flat_map(|object| &object.links) {
            if core::mem::replace(&mut linked[link.target.get()], true) {
                return None;
            }
        }
        // The reverse of a pre-order walk that visits the targets in reverse link order.
        let mut stack = vec![ObjectIndex::ROOT];
        let mut order = Vec::with_capacity(self.objects.len());
        while let Some(index) = stack.pop() {
            order.push(index);
            stack.extend(
                self.objects[index.get()]
                    .links
                    .iter()
                    .map(|link| link.target),
            );
        }
        order.reverse();
        Some(order)
    }
}

/// An [`ObjectGraph`] in which each object other than the root is the target of one link.
///
/// The descendants of the root are in post order, each after the objects its links target, and a
/// link targets a descendant by its index in that order.
pub(crate) struct ObjectTree<T> {
    descendants: Vec<GraphObject<T>>,
    root: GraphObject<T>,
}

impl<T> ObjectTree<T> {
    /// Returns the descendants of the root in post order, and the root.
    pub(crate) fn into_parts(self) -> (Vec<GraphObject<T>>, GraphObject<T>) {
        (self.descendants, self.root)
    }
}

/// An object of an [`ObjectGraph`] or an [`ObjectTree`] and the hard links it holds.
pub(crate) struct GraphObject<T> {
    /// The value the expansion of the object returned.
    pub(crate) value: T,
    /// The links of the object, in the order the expansion returned them.
    pub(crate) links: Vec<HardLink<ObjectIndex>>,
}

/// The value of an object and the hard links it holds, as the `expand` callback of
/// [`ObjectGraph::discover`] returns them.
pub(crate) struct Expansion<T> {
    pub(crate) value: T,
    pub(crate) links: Vec<HardLink<ObjectId>>,
}

/// A hard link of a group, which targets an object by `Target`.
#[derive(Debug)]
pub(crate) struct HardLink<Target> {
    pub(crate) name: String,
    /// The creation index of the link, or `None` if its Link message has none.
    pub(crate) creation_order: Option<u64>,
    pub(crate) target: Target,
}

/// The identity of an object in a file, the physical address of its object header.
///
/// Two hard links target the same object if and only if their targets resolve to one address.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ObjectId(PhysicalAddress);

impl ObjectId {
    pub(crate) const fn new(header: PhysicalAddress) -> Self {
        Self(header)
    }

    /// Returns the identity of the object whose header is at the stored address `header`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if `header` and the effective base of `addresses`
    /// sum past [`u64::MAX`].
    pub(crate) fn resolve(
        addresses: AddressContext,
        header: StoredAddress,
    ) -> Result<Self, FormatError> {
        addresses.resolve(header).map(Self)
    }

    pub(crate) const fn header_address(self) -> PhysicalAddress {
        self.0
    }
}

/// The position of an object in an [`ObjectGraph`], or among the descendants of an [`ObjectTree`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ObjectIndex(usize);

impl ObjectIndex {
    /// The root of an [`ObjectGraph`], which [`ObjectGraph::discover`] admits first.
    const ROOT: Self = Self(0);

    pub(crate) const fn get(self) -> usize {
        self.0
    }
}

/// The numbers of objects and of hard links past which the discovery of an [`ObjectGraph`] stops.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TraversalBudget {
    objects: usize,
    links: usize,
}

impl TraversalBudget {
    pub(crate) const fn new(objects: usize, links: usize) -> Self {
        Self { objects, links }
    }
}

/// An error that stops the discovery of an [`ObjectGraph`].
#[derive(Debug)]
pub(crate) enum TraversalError<E> {
    /// The `expand` callback returned this error.
    Expand(E),
    /// The graph holds more objects or more hard links than the [`TraversalBudget`] allows.
    LimitExceeded {
        /// The count the graph exceeds.
        limit: TraversalLimit,
        /// The most objects or hard links the budget allows.
        maximum: usize,
    },
}

/// The count that bounds a traversal of an object graph, as
/// [`Error::TraversalLimitExceeded`](crate::Error::TraversalLimitExceeded) reports it.
///
/// A copy, made by [`File::copy_from`](crate::File::copy_from) or by the
/// [`File::commit`](crate::File::commit) that applies a [`File::copy`](crate::File::copy), reads
/// the objects its source reaches through hard links, up to a limit on the objects and another on
/// the hard links.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TraversalLimit {
    /// The number of hard links the traversal reads, each link counted.
    Links,
    /// The number of objects the traversal reads, each object counted once however many links
    /// target it.
    Objects,
}

impl fmt::Display for TraversalLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Links => "hard links",
            Self::Objects => "objects",
        })
    }
}

/// The objects a discovery has reached, in the order it reached them, and the expansions of the
/// first of them.
struct Worklist<T> {
    /// The index of each object the walk has reached.
    indices: BTreeMap<ObjectId, ObjectIndex>,
    /// The objects the walk has reached, at their index.
    admitted: Vec<ObjectId>,
    /// The expansions of the objects of `admitted` the walk has expanded, in the same order.
    expanded: Vec<GraphObject<T>>,
    object_limit: usize,
}

impl<T> Worklist<T> {
    fn new(object_limit: usize) -> Self {
        Self {
            indices: BTreeMap::new(),
            admitted: Vec::new(),
            expanded: Vec::new(),
            object_limit,
        }
    }

    /// Returns the index of `id`, and admits it as the last object to expand if the walk has not
    /// reached it before.
    ///
    /// # Errors
    ///
    /// Returns [`TraversalError::LimitExceeded`] if admitting `id` takes the walk past
    /// `object_limit` objects.
    fn admit<E>(&mut self, id: ObjectId) -> Result<ObjectIndex, TraversalError<E>> {
        if let Some(&index) = self.indices.get(&id) {
            return Ok(index);
        }
        if self.indices.len() >= self.object_limit {
            return Err(TraversalError::LimitExceeded {
                limit: TraversalLimit::Objects,
                maximum: self.object_limit,
            });
        }
        let index = ObjectIndex(self.admitted.len());
        self.indices.insert(id, index);
        self.admitted.push(id);
        Ok(index)
    }

    /// Returns the first admitted object that is not yet expanded.
    fn next(&self) -> Option<ObjectId> {
        self.admitted.get(self.expanded.len()).copied()
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;
    use std::thread;

    use hdf5_pure_core::__private::PhysicalAddress;
    use rstest::rstest;

    use super::Expansion;
    use super::GraphObject;
    use super::HardLink;
    use super::ObjectGraph;
    use super::ObjectId;
    use super::TraversalBudget;
    use super::TraversalError;
    use super::TraversalLimit;

    #[test]
    fn a_tree_lists_each_object_after_its_targets_in_link_order() {
        let edges = vec![(1, vec![2, 3]), (2, vec![4]), (3, vec![]), (4, vec![])];
        let graph = discover(&edges, UNLIMITED).unwrap();
        let Ok(tree) = graph.into_tree() else {
            panic!("expected a tree, got the graph back");
        };
        let (descendants, root) = tree.into_parts();
        assert_eq!(
            descendants.iter().map(summary).collect::<Vec<_>>(),
            vec![(4, vec![]), (2, vec![0]), (3, vec![])],
        );
        assert_eq!(summary(&root), (1, vec![1, 2]));
    }

    #[rstest]
    #[case::self_link(vec![(1, vec![1])])]
    #[case::ancestor_cycle(vec![(1, vec![2]), (2, vec![1])])]
    #[case::shared_group(vec![(1, vec![2, 3]), (2, vec![4]), (3, vec![4]), (4, vec![])])]
    #[case::two_links_to_one_object(vec![(1, vec![2, 2]), (2, vec![])])]
    fn a_repeated_object_is_expanded_once_and_rejected_as_a_tree(
        #[case] edges: Vec<(u64, Vec<u64>)>,
    ) {
        let mut expanded = Vec::new();
        let graph = ObjectGraph::discover(UNLIMITED, object(edges[0].0), |id| {
            expanded.push(id.header_address().get());
            expand(&edges, id)
        })
        .unwrap();
        expanded.sort_unstable();
        assert_eq!(
            expanded,
            edges
                .iter()
                .map(|&(address, _)| address)
                .collect::<Vec<_>>()
        );
        let Err(graph) = graph.into_tree() else {
            panic!("expected the graph back, got a tree");
        };
        assert_eq!(graph.objects.len(), edges.len());
    }

    #[test]
    fn a_traversal_within_the_budget_reaches_every_object() {
        let graph = discover(&budget_graph(), TraversalBudget::new(4, 4)).unwrap();
        assert_eq!(graph.objects.len(), 4);
    }

    #[rstest]
    #[case::none(0)]
    #[case::one_short(3)]
    fn the_object_budget_bounds_the_objects_a_traversal_reaches(#[case] objects: usize) {
        let err = discover(&budget_graph(), TraversalBudget::new(objects, 4))
            .map(|_| ())
            .unwrap_err();
        let TraversalError::LimitExceeded { limit, maximum } = err;
        assert_eq!((limit, maximum), (TraversalLimit::Objects, objects));
    }

    #[rstest]
    #[case::none(0)]
    #[case::one_short(3)]
    fn the_link_budget_bounds_the_links_a_traversal_reads(#[case] links: usize) {
        let err = discover(&budget_graph(), TraversalBudget::new(4, links))
            .map(|_| ())
            .unwrap_err();
        let TraversalError::LimitExceeded { limit, maximum } = err;
        assert_eq!((limit, maximum), (TraversalLimit::Links, links));
    }

    #[test]
    fn an_expansion_error_stops_the_traversal() {
        let err = ObjectGraph::<()>::discover(UNLIMITED, object(1), |_| Err("unreadable"))
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(err, TraversalError::Expand("unreadable")),
            "{err:?}"
        );
    }

    #[test]
    fn a_chain_past_the_native_stack_is_discovered_ordered_and_dropped() {
        thread::Builder::new()
            .stack_size(SMALL_STACK)
            .spawn(|| {
                let graph = ObjectGraph::discover(UNLIMITED, object(0), |id| {
                    let next = id.header_address().get() + 1;
                    Ok::<_, Infallible>(Expansion {
                        value: id,
                        links: if next < DEEP_CHAIN {
                            vec![link(next)]
                        } else {
                            Vec::new()
                        },
                    })
                })
                .unwrap();
                let Ok(tree) = graph.into_tree() else {
                    panic!("expected a tree, got the graph back");
                };
                let (descendants, root) = tree.into_parts();
                assert_eq!(descendants.len(), DEEP_CHAIN as usize - 1);
                assert_eq!(descendants[0].value, object(DEEP_CHAIN - 1));
                assert_eq!(root.value, object(0));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// Four objects and four hard links, two of them to object 3.
    fn budget_graph() -> Vec<(u64, Vec<u64>)> {
        vec![(1, vec![2, 3]), (2, vec![3]), (3, vec![4]), (4, vec![])]
    }

    fn discover(
        edges: &[(u64, Vec<u64>)],
        budget: TraversalBudget,
    ) -> Result<ObjectGraph<ObjectId>, TraversalError<Infallible>> {
        ObjectGraph::discover(budget, object(edges[0].0), |id| expand(edges, id))
    }

    fn expand(edges: &[(u64, Vec<u64>)], id: ObjectId) -> Result<Expansion<ObjectId>, Infallible> {
        let (_, targets) = edges
            .iter()
            .find(|(address, _)| *address == id.header_address().get())
            .unwrap();
        Ok(Expansion {
            value: id,
            links: targets.iter().map(|&target| link(target)).collect(),
        })
    }

    fn summary(object: &GraphObject<ObjectId>) -> (u64, Vec<usize>) {
        (
            object.value.header_address().get(),
            object.links.iter().map(|link| link.target.get()).collect(),
        )
    }

    fn link(target: u64) -> HardLink<ObjectId> {
        HardLink {
            name: format!("{target}"),
            creation_order: None,
            target: object(target),
        }
    }

    fn object(address: u64) -> ObjectId {
        ObjectId::new(PhysicalAddress::new(address))
    }

    const UNLIMITED: TraversalBudget = TraversalBudget::new(usize::MAX, usize::MAX);

    const DEEP_CHAIN: u64 = 100_000;

    const SMALL_STACK: usize = 256 * 1024;
}
