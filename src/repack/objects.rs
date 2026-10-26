//! The objects of a repack source, and where the repack places each of them.
//!
//! [`SourceObjects::discover`] reads the source as an [`ObjectGraph`], removes the links the drop
//! paths name and then the objects the root group does not reach, and accepts the rest only as a
//! tree, since a repack writes each object at one path.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;

use hdf5_pure_core::__private::PhysicalAddress;

use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::Expansion;
use crate::object_graph::GraphObject;
use crate::object_graph::HardLink;
use crate::object_graph::LinkIndex;
use crate::object_graph::ObjectGraph;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;
use crate::object_graph::TraversalBudget;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;
use crate::reader::ObjectClass;

/// The objects a repack keeps, as a tree under the root group, and the placement of every object
/// the source reaches.
pub(super) struct SourceObjects {
    /// The address context of the source, for resolving a stored address to an object.
    addresses: AddressContext,
    /// The objects the repack keeps other than the root group, in the post order of an
    /// [`ObjectTree`](crate::object_graph::ObjectTree).
    descendants: Vec<GraphObject<SourceObject, String>>,
    root: GraphObject<SourceObject, String>,
    placements: BTreeMap<ObjectId, Placement>,
}

impl SourceObjects {
    /// Reads the objects `file` reaches through hard links, and removes the links the paths of
    /// `drop` name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RepackUnsupported`] if `file` has staged edits, if a path of `drop` matches
    /// no link, if a link that stays targets the root group or an object another link targets, or
    /// if a link that stays has a name that is not a component of an object path. Returns
    /// [`Error::TraversalLimitExceeded`] if `file` reaches more objects or hard links than
    /// `budget` allows, and [`Error::NotAGroup`] if the root object is not a group. Propagates the
    /// errors of reading the object headers and the links of the groups.
    pub(super) fn discover(
        file: &File,
        drop: &BTreeSet<ObjectPathBuf>,
        budget: TraversalBudget,
    ) -> Result<Self, Error> {
        if file.has_staged_edits() {
            return Err(Error::RepackUnsupported(String::from(
                "the source holds staged edits, and a repack reads only what the source has \
                 committed",
            )));
        }
        let mut graph = discover_graph(file, budget)?;
        let drops = find_drops(&graph, drop)?;
        let removed = graph.remove_links(&drops.iter().map(|&(link, _, _)| link).collect());
        let Ok(tree) = graph.into_tree() else {
            return Err(Error::RepackUnsupported(String::from(REPEATED_OBJECT)));
        };
        let (descendants, root) = tree.into_parts();
        if root.value.class != ObjectClass::Group {
            return Err(Error::NotAGroup(ObjectPathBuf::root().to_string()));
        }
        let mut dropped_paths = dropped_paths(&removed, &drops);
        let mut placements: BTreeMap<_, _> = removed
            .into_values()
            .map(|object| {
                let path = dropped_paths.remove(&object.value.id);
                (object.value.id, Placement::Dropped(path))
            })
            .collect();
        let kept_paths = kept_paths(&root, &descendants)?;
        placements.extend(
            descendants
                .iter()
                .map(|object| object.value.id)
                .zip(kept_paths.into_iter().map(Placement::Kept)),
        );
        placements.insert(root.value.id, Placement::Kept(ObjectPathBuf::root()));
        Ok(Self {
            addresses: file.addresses(),
            descendants,
            root,
            placements,
        })
    }

    /// Returns the root group.
    pub(super) fn root(&self) -> &GraphObject<SourceObject, String> {
        &self.root
    }

    /// Returns the objects the links of `object` target, each with the name of its link and its
    /// path in the output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if a link of `object` targets an
    /// object that is not in the tree or that the repack does not keep.
    pub(super) fn children<'a>(
        &'a self,
        object: &'a GraphObject<SourceObject, String>,
    ) -> Result<Vec<Child<'a>>, Error> {
        object
            .links
            .iter()
            .map(|link| {
                let object = self.descendants.get(link.target.get()).ok_or_else(|| {
                    FormatError::Internal(
                        "a repacked group links an object outside its tree".into(),
                    )
                })?;
                let Some(Placement::Kept(path)) = self.placements.get(&object.value.id) else {
                    return Err(FormatError::Internal(
                        "a repacked group links an object the repack does not keep".into(),
                    )
                    .into());
                };
                Ok(Child {
                    link_name: &link.label,
                    object,
                    path,
                })
            })
            .collect()
    }

    /// Returns the placement of the object whose header is at the stored address `header`, or
    /// `None` if `header` is not the address of an object the source reaches through hard links.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if `header` and the base address of the source sum past
    /// [`u64::MAX`].
    pub(super) fn placement(&self, header: StoredAddress) -> Result<Option<&Placement>, Error> {
        Ok(self
            .placements
            .get(&ObjectId::resolve(self.addresses, header)?))
    }
}

/// The identity and the class of an object of the source.
#[derive(Clone, Copy, Debug)]
pub(super) struct SourceObject {
    pub(super) id: ObjectId,
    pub(super) class: ObjectClass,
}

/// An object a link of a group targets, with the name of the link and the path of the object in
/// the output.
pub(super) struct Child<'a> {
    pub(super) link_name: &'a str,
    pub(super) object: &'a GraphObject<SourceObject, String>,
    pub(super) path: &'a ObjectPathBuf,
}

/// Whether a repack keeps an object of its source, and at which path.
pub(super) enum Placement {
    /// An object the drop paths remove, with its path under a drop path, or `None` if every such
    /// path passes a link whose name is not a component of an object path.
    Dropped(Option<ObjectPathBuf>),
    /// An object the output holds at this path.
    Kept(ObjectPathBuf),
}

/// Discovers the objects the root group of `file` reaches through hard links, with the class of
/// each and the name of each link.
///
/// # Errors
///
/// Returns [`Error::TraversalLimitExceeded`] if `file` reaches more objects or hard links than
/// `budget` allows. Propagates the errors of reading the object headers and the links of the
/// groups.
fn discover_graph(
    file: &File,
    budget: TraversalBudget,
) -> Result<ObjectGraph<SourceObject, String>, Error> {
    let addresses = file.addresses();
    let root = ObjectId::new(PhysicalAddress::new(file.root().header_address()?));
    Ok(ObjectGraph::discover(budget, root, |id| {
        let (class, entries) = file.hard_links_at(id.header_address())?;
        let links = entries
            .into_iter()
            .map(|entry| {
                Ok(HardLink {
                    label: entry.name,
                    target: ObjectId::resolve(addresses, entry.object_header_address)?,
                })
            })
            .collect::<Result<_, FormatError>>()?;
        Ok::<_, Error>(Expansion {
            value: SourceObject { id, class },
            links,
        })
    })?)
}

/// Returns the link each path of `drop` names, the object the link targets, and the path.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if a path of `drop` matches no link.
fn find_drops<'a>(
    graph: &ObjectGraph<SourceObject, String>,
    drop: &'a BTreeSet<ObjectPathBuf>,
) -> Result<Vec<(LinkIndex, ObjectIndex, &'a ObjectPathBuf)>, Error> {
    drop.iter()
        .map(|path| {
            graph
                .find_link(path.components().iter().map(LinkNameBuf::as_str))
                .map(|(link, target)| (link, target, path))
                .ok_or_else(|| {
                    Error::RepackUnsupported(format!(
                        "drop path does not exist in the source: {path}"
                    ))
                })
        })
        .collect()
}

/// Walks breadth-first from the targets of `drops` through the objects of `removed`, and returns
/// the first path under a drop path it finds to each.
fn dropped_paths(
    removed: &BTreeMap<ObjectIndex, GraphObject<SourceObject, String>>,
    drops: &[(LinkIndex, ObjectIndex, &ObjectPathBuf)],
) -> BTreeMap<ObjectId, ObjectPathBuf> {
    // `removed` and the targets of `drops` hold the indices from before the removal, which the
    // links of the removed objects target.
    let mut paths = BTreeMap::new();
    let mut pending: VecDeque<_> = drops
        .iter()
        .map(|&(_, target, path)| (target, path.clone()))
        .collect();
    while let Some((index, path)) = pending.pop_front() {
        let Some(object) = removed.get(&index) else {
            continue;
        };
        if paths.contains_key(&object.value.id) {
            continue;
        }
        pending.extend(object.links.iter().filter_map(|link| {
            LinkNameBuf::new(&link.label).map(|link_name| (link.target, path.join(&link_name)))
        }));
        paths.insert(object.value.id, path);
    }
    paths
}

/// Returns the path in the output of each object of `descendants`, at its index.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if a link has a name that is not a component of an object
/// path, and [`Error::Format`] with [`FormatError::Internal`] if `root` and `descendants` are not a
/// tree in post order.
fn kept_paths(
    root: &GraphObject<SourceObject, String>,
    descendants: &[GraphObject<SourceObject, String>],
) -> Result<Vec<ObjectPathBuf>, Error> {
    let mut paths = vec![None; descendants.len()];
    assign_paths(&mut paths, &ObjectPathBuf::root(), root)?;
    // In post order each object comes after the objects its links target, so every target of
    // `descendants[index]` has an index below `index`.
    for (index, object) in descendants.iter().enumerate().rev() {
        let (targets, rest) = paths.split_at_mut(index);
        let path = rest
            .first()
            .and_then(Option::as_ref)
            .ok_or_else(missing_path)?;
        assign_paths(targets, path, object)?;
    }
    paths
        .into_iter()
        .map(|path| path.ok_or_else(missing_path))
        .collect()
}

/// Sets the path of each object a link of `object` targets to the name of the link under
/// `parent`.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if a link has a name that is not a component of an object
/// path, and [`Error::Format`] with [`FormatError::Internal`] if a link targets an object outside
/// `paths`.
fn assign_paths(
    paths: &mut [Option<ObjectPathBuf>],
    parent: &ObjectPathBuf,
    object: &GraphObject<SourceObject, String>,
) -> Result<(), Error> {
    for link in &object.links {
        let link_name = super::link_name_in(parent, &link.label)?;
        *paths.get_mut(link.target.get()).ok_or_else(missing_path)? = Some(parent.join(&link_name));
    }
    Ok(())
}

fn missing_path() -> Error {
    FormatError::Internal("a repacked object has no path in the output".into()).into()
}

/// The reason [`Error::RepackUnsupported`] reports for a source that does not form a tree once the
/// links the drop paths name are removed.
const REPEATED_OBJECT: &str = "the repacked file would hold more than one hard link to one object (an \
     alias or a cycle), which a repack cannot reproduce yet";
