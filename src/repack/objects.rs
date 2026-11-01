//! The objects of a repack source, and where the repack places each of them.
//!
//! [`SourceObjects::discover`] reads the source as an [`ObjectGraph`], removes the links the drop
//! paths name and then the objects the root group does not reach, and accepts the rest only as a
//! tree, since a repack writes each object at one path.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::ops::Range;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::LinkCreationPolicy;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;

use super::headers::HeaderCache;
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_writer::source::SourceLink;
use crate::object_graph::Expansion;
use crate::object_graph::GraphObject;
use crate::object_graph::HardLink;
use crate::object_graph::LinkIndex;
use crate::object_graph::ObjectGraph;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;
use crate::object_graph::TraversalBudget;
use crate::object_graph::TraversalLimit;
use crate::object_path::LinkName;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;
use crate::reader::ObjectClass;
use crate::reader::ParsedHeader;

/// The objects a repack keeps, as a tree under the root group, and the placement of every object
/// the source reaches.
///
/// Stores one record per object and retains each encoded link once. Paths are constructed from
/// parent links when requested by a caller.
pub(super) struct SourceObjects {
    /// The address context of the source, for resolving a stored address to an object.
    addresses: AddressContext,
    /// Parsed discovery headers available for one planning load per retained object.
    headers: RefCell<HeaderCache>,
    /// The retained objects in post order, with the root last.
    objects: Vec<SourceObject>,
    /// The links of each object, grouped in the same order as `objects` and sorted by link name.
    links: Vec<RetainedLink>,
    /// Object identities and their indices in `objects`, sorted by identity.
    kept: Vec<(ObjectId, usize)>,
    /// The objects the drop paths remove, each with the last step of its first path under a drop
    /// path, or `None` if every such path passes a link whose name is not a component of an object
    /// path.
    dropped: BTreeMap<ObjectId, Option<LastStep>>,
}

impl SourceObjects {
    /// Reads the objects `file` reaches through hard links, and removes the links the paths of
    /// `drop` name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RepackUnsupported`] if `file` has staged edits, if a path of `drop` matches
    /// no link, if no raw name matches and the display fallback is ambiguous, if a link that stays
    /// targets the root group or an object another link targets, or
    /// if a link that stays has a name that is not a component of an object path. Returns
    /// [`Error::TraversalLimitExceeded`] if `file` reaches more objects or hard links than
    /// `budget` allows, or if the groups that stay nest more than [`MAX_DEPTH`] levels deep, and
    /// [`Error::NotAGroup`] if the root object is not a group. Propagates the errors of reading
    /// the object headers and the links of the groups.
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
        let mut headers = HeaderCache::new(HEADER_CACHE_BYTES);
        let mut graph = discover_graph(file, budget, &mut headers)?;
        headers.sort();
        let drops = find_drops(&graph, drop)?;
        let removed = graph.remove_links(&drops.iter().map(|&(link, _, _)| link).collect());
        let Ok(tree) = graph.into_tree() else {
            return Err(Error::RepackUnsupported(String::from(REPEATED_OBJECT)));
        };
        if tree.depth(|object| object.class == ObjectClass::Group) > MAX_DEPTH {
            return Err(Error::TraversalLimitExceeded {
                limit: TraversalLimit::Depth,
                maximum: MAX_DEPTH,
            });
        }
        let (descendants, root) = tree.into_parts();
        if root.value.class != ObjectClass::Group {
            return Err(Error::NotAGroup(ObjectPathBuf::root().to_string()));
        }
        let mut last_steps = dropped_last_steps(&removed, &drops);
        let dropped = removed
            .into_values()
            .map(|object| (object.value.id, last_steps.remove(&object.value.id)))
            .collect();
        let mut objects: Vec<SourceObject> = Vec::with_capacity(descendants.len() + 1);
        let mut links = Vec::with_capacity(descendants.len());
        let mut errors = Vec::new();
        for (index, object) in descendants.into_iter().chain([root]).enumerate() {
            let DiscoveredObject {
                id,
                class,
                creation_order,
                symbolic,
                error,
            } = object.value;
            let mut group_links = object
                .links
                .into_iter()
                .map(|link| RetainedLink {
                    record: link.label,
                    target: Some(link.target.get()),
                    parent: index,
                })
                .chain(symbolic.into_iter().map(|link| RetainedLink {
                    record: LinkLabel::Encoded(link),
                    target: None,
                    parent: index,
                }))
                .collect::<Vec<_>>();
            group_links.sort_by(|a, b| a.record.bytes().cmp(b.record.bytes()));
            let start = links.len();
            for link in group_links {
                if let Some(target) = link.target {
                    objects.get_mut(target).ok_or_else(missing_path)?.parent = Some(links.len());
                }
                links.push(link);
            }
            if let Some(error) = error {
                errors.push(error);
            }
            objects.push(SourceObject {
                id,
                class,
                creation_order,
                links: start..links.len(),
                parent: None,
            });
        }
        let mut kept: Vec<_> = objects
            .iter()
            .enumerate()
            .map(|(index, object)| (object.id, index))
            .collect();
        kept.sort_unstable_by_key(|&(id, _)| id);
        headers.retain(|id| kept.binary_search_by_key(&id, |&(id, _)| id).is_ok());
        let objects = Self {
            addresses: file.addresses(),
            headers: RefCell::new(headers),
            objects,
            links,
            kept,
            dropped,
        };
        for object in objects.objects.iter().rev() {
            for link in objects
                .links
                .get(object.links.clone())
                .ok_or_else(missing_path)?
            {
                if LinkName::new(&link.record.name()).is_none() {
                    super::link_name_in(&objects.path(link.parent)?, &link.record.name())?;
                }
            }
        }
        if let Some(error) = errors.into_iter().next() {
            return Err(error);
        }
        Ok(objects)
    }

    /// Releases all cached discovery headers and their storage.
    pub(super) fn release_headers(&self) {
        self.headers.borrow_mut().clear();
    }

    /// Removes the cached discovery header for the object at `index`, if present.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `index` is outside the records.
    pub(super) fn take_header(&self, index: usize) -> Result<Option<ParsedHeader>, Error> {
        Ok(self.headers.borrow_mut().take(self.object(index)?.id))
    }

    /// Returns the number of retained objects, including the root group.
    pub(super) fn len(&self) -> usize {
        self.objects.len()
    }

    /// Returns the retained object at its post order index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `index` is outside the records.
    pub(super) fn object(&self, index: usize) -> Result<&SourceObject, Error> {
        self.objects.get(index).ok_or_else(missing_path)
    }

    /// Returns borrowed link records and hard-target indices for the object at `index`.
    ///
    /// Orders links by their raw name bytes. Soft and external links retain their encoded targets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if an object or link index is out
    /// of bounds.
    pub(super) fn links(
        &self,
        index: usize,
    ) -> Result<impl ExactSizeIterator<Item = Result<SourceLink<'_>, Error>>, Error> {
        let object = self.object(index)?;
        Ok(self
            .links
            .get(object.links.clone())
            .ok_or_else(missing_path)?
            .iter()
            .map(|link| match &link.record {
                LinkLabel::Encoded(record) => Ok(SourceLink {
                    link: Cow::Borrowed(record),
                    target: link.target,
                }),
                LinkLabel::Rejected(_) => Err(missing_path()),
            }))
    }

    /// Resolves an output path to the retained object's index.
    ///
    /// Searches each group by raw name bytes first, then by the lossy UTF-8 display spelling.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RepackUnsupported`] if no raw name matches and the display fallback matches
    /// multiple encoded names.
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `path` is absent, a target is
    /// symbolic, or an object or link index is out of bounds.
    pub(super) fn index(&self, path: &ObjectPathBuf) -> Result<usize, Error> {
        let mut index = self.objects.len().checked_sub(1).ok_or_else(missing_path)?;
        for component in path.components() {
            let links = self
                .links
                .get(self.object(index)?.links.clone())
                .ok_or_else(missing_path)?;
            let position =
                links.partition_point(|link| link.record.bytes() < component.as_str().as_bytes());
            let link = match links
                .get(position)
                .filter(|link| link.record.bytes() == component.as_str().as_bytes())
            {
                Some(link) => link,
                None => {
                    let mut matches = links
                        .iter()
                        .filter(|link| link.record.name() == component.as_str());
                    let link = matches.next().ok_or_else(missing_path)?;
                    if matches.next().is_some() {
                        return Err(Error::RepackUnsupported(format!(
                            "the output path {:?} matches multiple encoded link names",
                            path.to_string()
                        )));
                    }
                    link
                }
            };
            index = link.target.ok_or_else(missing_path)?;
        }
        Ok(index)
    }

    /// Returns the path of the group containing the object at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if the object has no parent or
    /// an object or link index is out of bounds. Propagates errors from [`Self::path`].
    pub(super) fn parent_path(&self, index: usize) -> Result<ObjectPathBuf, Error> {
        self.path(
            self.links
                .get(self.object(index)?.parent.ok_or_else(missing_path)?)
                .ok_or_else(missing_path)?
                .parent,
        )
    }

    /// Returns the final object in the retained tree's post order, the root group.
    #[cfg(test)]
    pub(super) fn root(&self) -> &SourceObject {
        // The root is always present after discovery.
        self.objects.last().expect("a repack source holds its root")
    }

    /// Returns hard-linked children of `object`, with their paths under `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RepackUnsupported`] if a link name is not a component of an object path.
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if an object or link index is out
    /// of bounds.
    #[cfg(test)]
    pub(super) fn children<'a>(
        &'a self,
        object: &SourceObject,
        path: &ObjectPathBuf,
    ) -> Result<Vec<Child<'a>>, Error> {
        self.links
            .get(object.links.clone())
            .ok_or_else(missing_path)?
            .iter()
            .filter_map(|link| link.target.map(|target| (link, target)))
            .map(|(link, target)| {
                let object = self.objects.get(target).ok_or_else(missing_path)?;
                Ok(Child {
                    link_name: link.record.name(),
                    object,
                    path: path.join(&super::link_name_in(path, &link.record.name())?),
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
    /// [`u64::MAX`]. Returns [`Error::Format`] with [`FormatError::Internal`] if a dropped object
    /// on the path to the object has no last step.
    pub(super) fn placement(&self, header: StoredAddress) -> Result<Option<Placement>, Error> {
        let id = ObjectId::resolve(self.addresses, header)?;
        if let Ok(position) = self.kept.binary_search_by_key(&id, |&(id, _)| id) {
            return Ok(Some(Placement::Kept(self.kept[position].1)));
        }
        let Some(last_step) = self.dropped.get(&id) else {
            return Ok(None);
        };
        Ok(Some(Placement::Dropped(
            last_step
                .as_ref()
                .map(|last_step| self.dropped_path(last_step))
                .transpose()?,
        )))
    }

    /// Reconstructs an output path by following parent links to the root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if an object or link index is out
    /// of bounds, or a parent link name is not a component of an object path.
    pub(super) fn path(&self, mut index: usize) -> Result<ObjectPathBuf, Error> {
        let mut names = Vec::new();
        while let Some(parent) = self.objects.get(index).ok_or_else(missing_path)?.parent {
            let link = self.links.get(parent).ok_or_else(missing_path)?;
            names.push(LinkNameBuf::new(&link.record.name()).ok_or_else(missing_path)?);
            index = link.parent;
        }
        Ok(names.into_iter().rev().collect())
    }

    /// Returns the path under a drop path that ends with `step`, from the last steps of the dropped
    /// objects on the way back to the drop path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if a dropped object on the way has
    /// no last step.
    fn dropped_path<'a>(&'a self, mut step: &'a LastStep) -> Result<ObjectPathBuf, Error> {
        let mut link_names = Vec::new();
        // `dropped_last_steps` records the last step of a parent before those of the objects its
        // links target, so the walk ends at a `LastStep::Drop`.
        loop {
            match step {
                LastStep::Drop(path) => {
                    return Ok(path
                        .components()
                        .iter()
                        .cloned()
                        .chain(link_names.into_iter().rev().cloned())
                        .collect());
                }
                LastStep::Link { parent, link_name } => {
                    link_names.push(link_name);
                    step = self
                        .dropped
                        .get(parent)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            FormatError::Internal(
                                "a dropped object's parent has no last step".into(),
                            )
                        })?;
                }
            }
        }
    }
}

/// A retained object's identity, class, creation-order policy, and link indices.
pub(super) struct SourceObject {
    pub(super) id: ObjectId,
    pub(super) class: ObjectClass,
    pub(super) creation_order: LinkCreationPolicy,
    /// The range of this object's links in [`SourceObjects::links`].
    links: Range<usize>,
    /// The index of the incoming link in [`SourceObjects::links`], or `None` for the root.
    parent: Option<usize>,
}

/// An encoded link retained in its containing group.
struct RetainedLink {
    record: LinkLabel,
    /// The hard target's index in [`SourceObjects::objects`], or `None` for a symbolic target.
    target: Option<usize>,
    /// The containing group's index in [`SourceObjects::objects`].
    parent: usize,
}

/// A retained child object with its link name and output path.
#[cfg(test)]
pub(super) struct Child<'a> {
    pub(super) link_name: Cow<'a, str>,
    pub(super) object: &'a SourceObject,
    pub(super) path: ObjectPathBuf,
}

/// A reachable object's destination index or its removal by a drop path.
pub(super) enum Placement {
    /// A removed object's path, or `None` if every path under a drop path has an invalid name.
    Dropped(Option<ObjectPathBuf>),
    /// The index of a retained object in [`SourceObjects::objects`].
    Kept(usize),
}

/// An object's identity and class during graph discovery.
struct DiscoveredObject {
    id: ObjectId,
    class: ObjectClass,
    creation_order: LinkCreationPolicy,
    symbolic: Vec<OwnedLink>,
    error: Option<Error>,
}

enum LinkLabel {
    Encoded(OwnedLink),
    Rejected(String),
}

impl LinkLabel {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Encoded(link) => link.name(),
            Self::Rejected(name) => name.as_bytes(),
        }
    }

    fn name(&self) -> Cow<'_, str> {
        match self {
            Self::Encoded(link) => String::from_utf8_lossy(link.name()),
            Self::Rejected(name) => Cow::Borrowed(name),
        }
    }
}

/// The last step of the first path under a drop path to a dropped object.
enum LastStep {
    /// The object is the target of the link this drop path names.
    Drop(ObjectPathBuf),
    /// The object is the target of the link `link_name` of the dropped object `parent`.
    Link {
        parent: ObjectId,
        link_name: LinkNameBuf,
    },
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
    headers: &mut HeaderCache,
) -> Result<ObjectGraph<DiscoveredObject, LinkLabel>, Error> {
    let addresses = file.addresses();
    let root = ObjectId::new(PhysicalAddress::new(file.root().header_address()?));
    Ok(ObjectGraph::discover(budget, root, |id| {
        let (class, storage, header, error, rejected) =
            match file.repack_links_at(id.header_address()) {
                Ok((class, storage, header)) => (class, storage, header, None, Vec::new()),
                Err(error @ Error::RepackUnsupported(_)) => {
                    let (class, entries, header) = match file.hard_links_at(id.header_address()) {
                        Ok(discovery) => discovery,
                        Err(_fallback_error) => return Err(error),
                    };
                    (class, None, header, Some(error), entries)
                }
                Err(error) => return Err(error),
            };
        headers.insert(id, header);
        let creation_order = storage.as_ref().map_or(
            LinkCreationPolicy::Untracked,
            GroupLinkStorage::creation_order,
        );
        let mut symbolic = Vec::new();
        let mut links = Vec::new();
        if let Some(storage) = storage {
            for link in storage.into_links() {
                match link.target() {
                    OwnedLinkTarget::Hard {
                        object_header_address,
                    } => links.push(HardLink {
                        target: ObjectId::resolve(addresses, *object_header_address)?,
                        label: LinkLabel::Encoded(link),
                    }),
                    OwnedLinkTarget::Indirect(_) => symbolic.push(link),
                }
            }
        }
        for entry in rejected {
            links.push(HardLink {
                label: LinkLabel::Rejected(entry.name),
                target: ObjectId::resolve(addresses, entry.object_header_address)?,
            });
        }
        Ok::<_, Error>(Expansion {
            value: DiscoveredObject {
                id,
                class,
                creation_order,
                symbolic,
                error,
            },
            links,
        })
    })?)
}

/// Returns the link each path of `drop` names, the object the link targets, and the path.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if a path of `drop` matches no hard link, or if no raw
/// name matches and the display fallback matches multiple encoded names.
fn find_drops<'a>(
    graph: &ObjectGraph<DiscoveredObject, LinkLabel>,
    drop: &'a BTreeSet<ObjectPathBuf>,
) -> Result<Vec<(LinkIndex, ObjectIndex, &'a ObjectPathBuf)>, Error> {
    drop.iter()
        .map(|path| {
            let mut object = ObjectIndex::ROOT;
            let mut found = None;
            for component in path.components() {
                let links = &graph.object(object).links;
                let position = match links
                    .iter()
                    .position(|link| link.label.bytes() == component.as_str().as_bytes())
                {
                    Some(position) => Some(position),
                    None => {
                        let mut matches = links
                            .iter()
                            .enumerate()
                            .filter(|(_, link)| link.label.name() == component.as_str());
                        let position = matches.next().map(|(position, _)| position);
                        if matches.next().is_some() {
                            return Err(Error::RepackUnsupported(format!(
                                "drop path {:?} matches multiple encoded link names",
                                path.to_string()
                            )));
                        }
                        position
                    }
                };
                let Some(position) = position else {
                    return Err(Error::RepackUnsupported(format!(
                        "drop path does not exist in the source: {path}"
                    )));
                };
                let link = &links[position];
                found = Some((LinkIndex::new(object, position), link.target, path));
                object = link.target;
            }
            found.ok_or_else(|| {
                Error::RepackUnsupported(format!("drop path does not exist in the source: {path}"))
            })
        })
        .collect()
}

/// Walks breadth-first from the targets of `drops` through the objects of `removed`, and returns
/// the last step of the first path under a drop path it finds to each.
fn dropped_last_steps(
    removed: &BTreeMap<ObjectIndex, GraphObject<DiscoveredObject, LinkLabel>>,
    drops: &[(LinkIndex, ObjectIndex, &ObjectPathBuf)],
) -> BTreeMap<ObjectId, LastStep> {
    // `removed` and the targets of `drops` hold the indices from before the removal, which the
    // links of the removed objects target.
    let mut last_steps = BTreeMap::new();
    let mut pending: VecDeque<_> = drops
        .iter()
        .map(|&(_, target, path)| (target, LastStep::Drop(path.clone())))
        .collect();
    while let Some((index, last_step)) = pending.pop_front() {
        let Some(object) = removed.get(&index) else {
            continue;
        };
        if last_steps.contains_key(&object.value.id) {
            continue;
        }
        pending.extend(object.links.iter().filter_map(|link| {
            LinkNameBuf::new(&link.label.name()).map(|link_name| {
                (
                    link.target,
                    LastStep::Link {
                        parent: object.value.id,
                        link_name,
                    },
                )
            })
        }));
        last_steps.insert(object.value.id, last_step);
    }
    last_steps
}

fn missing_path() -> Error {
    FormatError::Internal("a repacked object has no path in the output".into()).into()
}

/// The retained-byte budget for parsed discovery headers, including cache entry storage.
const HEADER_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// The reason [`Error::RepackUnsupported`] reports for a source that does not form a tree once the
/// links the drop paths name are removed.
const REPEATED_OBJECT: &str = "the repacked file would hold more than one hard link to one object (an \
     alias or a cycle), which a repack cannot reproduce yet";

/// The depth of the deepest group a repack writes, in links from the root group.
pub(super) const MAX_DEPTH: usize = 128;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::group_info::HeaderVersion;
    use test_util::group_info::LinkOrder;
    use test_util::group_info::NameCollision;
    use test_util::image::Image;
    use test_util::link_message::HardLink;
    use test_util::object_header::v2::Header;
    use test_util::object_header::{Message, MessageType};
    use test_util::superblock::v2::Superblock;
    use test_util::widths::Widths;

    use super::*;
    use crate::FileBuilder;
    use crate::GroupBuilder;
    use crate::UserblockSize;

    #[rstest]
    #[case::raw_and_utf8(NameCollision::Utf8AndRaw, false)]
    #[case::two_raw_names(NameCollision::RawAndRaw, true)]
    fn colliding_encoded_names_keep_object_identity_and_drop_selection(
        #[case] collision: NameCollision,
        #[case] ambiguous: bool,
    ) {
        let source =
            File::from_bytes(test_util::group_info::colliding_link_file(collision)).unwrap();
        let objects =
            SourceObjects::discover(&source, &BTreeSet::new(), TraversalBudget::new(3, 2)).unwrap();
        assert_eq!(objects.path(0).unwrap(), objects.path(1).unwrap());
        for index in 0..2 {
            let address = source
                .addresses()
                .relative(objects.object(index).unwrap().id.header_address())
                .unwrap();
            let Some(Placement::Kept(actual)) = objects.placement(address).unwrap() else {
                panic!("expected a retained object")
            };
            assert_eq!(actual, index);
        }
        let drop = BTreeSet::from([ObjectPathBuf::parse("x�")]);
        let result = SourceObjects::discover(&source, &drop, TraversalBudget::new(3, 2));
        if ambiguous {
            let error = result.err().unwrap();
            let Error::RepackUnsupported(reason) = error else {
                panic!("expected RepackUnsupported, got {error:?}")
            };
            assert_eq!(
                reason,
                "drop path \"x�\" matches multiple encoded link names"
            );
        } else {
            let pruned = result.unwrap();
            assert_eq!(pruned.len(), 2);
            let link = pruned.links(1).unwrap().next().unwrap().unwrap();
            assert_eq!(link.link.name(), b"x\xff");
        }
    }

    #[rstest]
    #[case::untracked(LinkOrder::Untracked)]
    #[case::tracked(LinkOrder::Tracked)]
    #[case::indexed(LinkOrder::Indexed)]
    fn encoded_names_map_to_objects_and_drop_paths(#[case] order: LinkOrder) {
        let (image, _, _) = test_util::group_info::repack_link_file(order, HeaderVersion::V2);
        let source = File::from_bytes(image).unwrap();
        let objects =
            SourceObjects::discover(&source, &BTreeSet::new(), TraversalBudget::new(2, 1)).unwrap();
        let path = objects.path(0).unwrap();
        assert_eq!(path.to_string(), "raw_�");
        assert_eq!(objects.index(&path).unwrap(), 0);
        assert_eq!(objects.links(1).unwrap().len(), 3);
        let pruned =
            SourceObjects::discover(&source, &BTreeSet::from([path]), TraversalBudget::new(2, 1))
                .unwrap();
        assert_eq!(pruned.len(), 1);
        let links = pruned
            .links(0)
            .unwrap()
            .map(|link| link.unwrap().link.into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            links.iter().map(|link| link.name()).collect::<Vec<_>>(),
            [b"external".as_slice(), b"soft".as_slice()]
        );
        assert_eq!(
            pruned.object(0).unwrap().creation_order,
            objects.object(1).unwrap().creation_order
        );
    }

    #[rstest]
    #[case::root("", false)]
    #[case::group("outer", false)]
    #[case::dataset("outer/inner/values", false)]
    #[case::committed_datatype("outer/type", false)]
    #[case::userblock_root("", true)]
    #[case::userblock_dataset("outer/inner/values", true)]
    #[case::userblock_committed_datatype("outer/type", true)]
    fn paths_are_reconstructed_from_the_normalized_object_identity(
        #[case] path: &str,
        #[case] userblock: bool,
    ) {
        let mut builder = FileBuilder::new();
        if userblock {
            builder.with_userblock(UserblockSize::MIN);
        }
        let mut outer = GroupBuilder::new("outer");
        let mut inner = GroupBuilder::new("inner");
        inner.create_dataset("values").with_i32_data(&[3]);
        outer.add_group(inner.finish());
        outer.commit_datatype("type", crate::type_builders::make_i32_type());
        builder.add_group(outer.finish());
        let source = File::from_bytes(builder.finish().unwrap()).unwrap();
        let objects =
            SourceObjects::discover(&source, &BTreeSet::new(), TraversalBudget::new(5, 4)).unwrap();
        let mut object = objects.root();
        let mut current = ObjectPathBuf::root();
        for component in ObjectPathBuf::parse(path).components() {
            let child = objects
                .children(object, &current)
                .unwrap()
                .into_iter()
                .find(|child| child.link_name == component.as_str())
                .unwrap();
            object = child.object;
            current = child.path;
        }
        assert_eq!(current, ObjectPathBuf::parse(path));
        let stored = source
            .addresses()
            .relative(object.id.header_address())
            .unwrap();
        let Some(Placement::Kept(actual)) = objects.placement(stored).unwrap() else {
            panic!("expected a kept object at {path:?}");
        };
        assert_eq!(objects.path(actual).unwrap(), ObjectPathBuf::parse(path));
    }

    #[test]
    fn discovery_stores_one_record_per_object_and_one_name_per_link() {
        let mut group = GroupBuilder::new("g");
        group.create_dataset("values").with_i32_data(&[3]);
        let group = (1..MAX_DEPTH).fold(group.finish(), |child, _| {
            let mut parent = GroupBuilder::new("g");
            parent.add_group(child);
            parent.finish()
        });
        let mut builder = FileBuilder::new();
        builder.add_group(group);
        let source = File::from_bytes(builder.finish().unwrap()).unwrap();
        let objects = SourceObjects::discover(
            &source,
            &BTreeSet::new(),
            TraversalBudget::new(MAX_DEPTH + 2, MAX_DEPTH + 1),
        )
        .unwrap();
        assert_eq!(objects.objects.len(), MAX_DEPTH + 2);
        assert_eq!(objects.links.len(), MAX_DEPTH + 1);
        assert_eq!(objects.kept.len(), MAX_DEPTH + 2);
        assert_eq!(
            objects
                .links
                .iter()
                .map(|link| link.record.bytes().len())
                .sum::<usize>(),
            MAX_DEPTH + "values".len()
        );
        assert_eq!(
            objects
                .path(
                    objects
                        .objects
                        .iter()
                        .position(|object| object.class == ObjectClass::Dataset)
                        .unwrap()
                )
                .unwrap()
                .to_string(),
            format!("{}/values", vec!["g"; MAX_DEPTH].join("/"))
        );
        for (index, object) in objects.objects.iter().enumerate() {
            for link in &objects.links[object.links.clone()] {
                assert_eq!(link.parent, index);
                let target = link.target.unwrap();
                assert!(target < index);
                let parent = objects.objects[target].parent.unwrap();
                assert_eq!(objects.links[parent].target, Some(target));
            }
        }
    }

    #[rstest]
    #[case::root(".", "valid", "root group", ".")]
    #[case::ancestor_before_descendant(".", "", "root group", ".")]
    #[case::nested("outer", "bad/name", "group outer", "bad/name")]
    fn invalid_link_names_report_the_first_invalid_ancestor(
        #[case] outer: &str,
        #[case] inner: &str,
        #[case] owner: &str,
        #[case] invalid: &str,
    ) {
        let mut image = Image::starting_with(&[0; 64]);
        let leaf = image.append(&Header::new().build());
        let middle = image.append(
            &Header::new()
                .message(Message::new(
                    MessageType::LINK,
                    &HardLink::new(inner, leaf as u64).build(Widths::EIGHT),
                ))
                .build(),
        );
        let root = image.append(
            &Header::new()
                .message(Message::new(
                    MessageType::LINK,
                    &HardLink::new(outer, middle as u64).build(Widths::EIGHT),
                ))
                .build(),
        );
        image.place(
            0,
            &Superblock::new(Widths::EIGHT)
                .root_header_address(root as u64)
                .eof_address(image.as_bytes().len() as u64)
                .build(),
        );
        let source = File::from_bytes(image.build()).unwrap();
        let err = SourceObjects::discover(&source, &BTreeSet::new(), TraversalBudget::new(3, 2))
            .map(|_| ())
            .unwrap_err();
        let Error::RepackUnsupported(reason) = err else {
            panic!("expected RepackUnsupported, got {err:?}");
        };
        assert_eq!(
            reason,
            format!(
                "{owner}: the link {invalid:?} has a name that is not a component of an object path, so the object it identifies has no path in the output"
            )
        );
    }
}
