//! The path of an object in a file, parsed into the sequence of link names to walk.
//!
//! [`ObjectPath::parse`] decides the grammar for the reader's resolvers, both group walks and the
//! type builders, the way the C library decides it: a repeated or a trailing separator separates
//! nothing (`H5G__component` in `H5Gname.c`), a component that is exactly `.` identifies the group
//! a walk has reached, and a leading separator starts the walk at the root group
//! (`H5G__traverse_real` in `H5Gtraverse.c`, HDF5 1.14.6).
//!
//! [`LinkName`] is one component of a path, the name of a link in one group.
//!
//! [`ObjectPathBuf`] owns the components of a path from the root group, and [`LinkNameBuf`] owns
//! one component.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt;

/// An object path: its components, and where the walk starts.
///
/// A path is absolute or relative, and either may be empty of components: `/`, `.` and the empty
/// string parse to none of them and identify the group the walk starts from, which is the root
/// group for an absolute path and the group the path is relative to for a relative one.
#[derive(Clone, Debug)]
pub(crate) struct ObjectPath<'a> {
    /// The components, in order, with the separators and the `.` components dropped.
    components: Vec<LinkName<'a>>,
    /// Whether the spelling begins with a separator, which starts the walk at
    /// the root group: `H5G__traverse_real` takes that branch before it reads
    /// the first component (`H5Gtraverse.c`, HDF5 1.14.6).
    absolute: bool,
}

impl<'a> ObjectPath<'a> {
    /// Parses one spelling of an object path into its components.
    ///
    /// `a/b`, `a//b`, `a/b/`, `./a/b` and `/a/b` parse to the same two components.
    pub(crate) fn parse(path: &'a str) -> Self {
        // The grammar is the C library's: `H5G__component` walks past the
        // separators between two names, so a trailing or repeated one separates
        // nothing, and `H5G__traverse_real` resolves a component that is `.` to
        // the group it is already in (`H5Gname.c` and `H5Gtraverse.c`, HDF5
        // 1.14.6). `LinkName::new` rejects exactly those components, so the
        // filter drops them. A *leading* separator is the one that carries
        // meaning, and `absolute` keeps it.
        Self {
            components: path.split(SEPARATOR).filter_map(LinkName::new).collect(),
            absolute: path.starts_with(SEPARATOR),
        }
    }

    /// Resolves `rest` from the object this path identifies.
    ///
    /// An absolute `rest` walks from the root group wherever this path ends, so it is the result
    /// on its own.
    pub(crate) fn join_path(&self, rest: &Self) -> Self {
        if rest.absolute {
            return rest.clone();
        }
        Self {
            components: self
                .components
                .iter()
                .chain(rest.components.iter())
                .copied()
                .collect(),
            absolute: self.absolute,
        }
    }

    /// Returns the path a walk has reached after `count` components, or this whole path
    /// where it has fewer than that.
    ///
    /// A walk that stops partway reports the object it had reached by this whole prefix, which is
    /// a path the caller can go on to open: `a/b/c` stopped by a dataset at `a/b` reports `a/b`.
    pub(crate) fn prefix(&self, count: usize) -> Self {
        Self {
            components: self.components.iter().copied().take(count).collect(),
            absolute: self.absolute,
        }
    }

    /// Converts this path into an owned [`ObjectPathBuf`], which keeps its components and not
    /// where the walk starts.
    pub(crate) fn to_path_buf(&self) -> ObjectPathBuf {
        ObjectPathBuf {
            components: self
                .components
                .iter()
                .copied()
                .map(LinkNameBuf::from)
                .collect(),
        }
    }

    /// Returns the components, an empty slice for the path of the object a walk starts from.
    pub(crate) fn components(&self) -> &[LinkName<'a>] {
        &self.components
    }

    /// Returns `true` if the walk starts at the root group.
    pub(crate) fn is_absolute(&self) -> bool {
        self.absolute
    }
}

/// Writes the components separated by `/`, the root-relative form a [`crate::Group`] handle
/// stores: `a/b` for every spelling of that path, and the empty string for the root group.
impl fmt::Display for ObjectPath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_components(f, self.components.iter().copied().map(LinkName::as_str))
    }
}

/// An owned path from the root group: its components, and nothing of how they were spelled.
///
/// The edit engine keys its staged edits on this, so two spellings of one path are one key.
/// [`as_path`](Self::as_path) borrows it back as an absolute [`ObjectPath`], the form a walk
/// takes.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ObjectPathBuf {
    /// The components, in order, from the root group.
    components: Vec<LinkNameBuf>,
}

impl ObjectPathBuf {
    /// Parses one spelling of an object path into its owned components.
    ///
    /// The spelling parses as it does for [`ObjectPath::parse`], and an absolute and a relative
    /// spelling of the same components give the same path. A caller with a relative path resolves
    /// it from the group it is relative to first, through [`join_path`](Self::join_path).
    pub(crate) fn parse(path: &str) -> Self {
        ObjectPath::parse(path).to_path_buf()
    }

    /// Returns the path of the root group, which has no components.
    pub(crate) fn root() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Returns the path of the child `name` of the object this path identifies.
    pub(crate) fn join(&self, name: &LinkNameBuf) -> Self {
        let mut components = self.components.clone();
        components.push(name.clone());
        Self { components }
    }

    /// Resolves `rest` from the object this path identifies, as [`ObjectPath::join_path`] does.
    ///
    /// An absolute `rest` walks from the root group, so its components alone are the result.
    pub(crate) fn join_path(&self, rest: &ObjectPath<'_>) -> Self {
        self.as_path().join_path(rest).to_path_buf()
    }

    /// Returns the path a walk has reached after `count` components, or this whole path where it
    /// has fewer than that.
    pub(crate) fn prefix(&self, count: usize) -> Self {
        Self {
            components: self.components.iter().take(count).cloned().collect(),
        }
    }

    /// Returns the path of the group holding the last link and that link's name, or `None` for
    /// the root group, which no link reaches.
    pub(crate) fn split_leaf(&self) -> Option<(Self, LinkNameBuf)> {
        let (leaf, parent) = self.components.split_last()?;
        Some((
            Self {
                components: parent.to_vec(),
            },
            leaf.clone(),
        ))
    }

    /// Borrows the components as an absolute [`ObjectPath`], the form a resolver walks.
    pub(crate) fn as_path(&self) -> ObjectPath<'_> {
        ObjectPath {
            components: self
                .components
                .iter()
                .map(LinkNameBuf::as_link_name)
                .collect(),
            absolute: true,
        }
    }

    /// Returns the components, an empty slice for the root group.
    pub(crate) fn components(&self) -> &[LinkNameBuf] {
        &self.components
    }

    /// Returns the number of components, which is the depth of the object below the root group.
    pub(crate) fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns `true` if this is the path of the root group.
    pub(crate) fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Returns the link name this path has in `parent`, or `None` unless `parent` holds the
    /// object directly.
    pub(crate) fn name_under(&self, parent: &Self) -> Option<&LinkNameBuf> {
        let (leaf, above) = self.components.split_last()?;
        (above == parent.components()).then_some(leaf)
    }

    /// Returns `true` if `prefix` identifies this object or an ancestor of it.
    pub(crate) fn starts_with(&self, prefix: &Self) -> bool {
        self.components.starts_with(prefix.components())
    }

    /// Returns `true` if one of the two paths identifies the object the other does or an
    /// ancestor of it.
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        self.starts_with(other) || other.starts_with(self)
    }

    /// Returns `true` if the child `name` of `parent` lies at or under this path.
    ///
    /// Takes `parent` and `name` in place of the child's own path, which a caller would have to
    /// build to pass it.
    pub(crate) fn covers_child(&self, parent: &Self, name: &LinkNameBuf) -> bool {
        match self.len() {
            n if n <= parent.len() => parent.starts_with(self),
            n if n == parent.len() + 1 => {
                self.starts_with(parent) && self.components().get(parent.len()) == Some(name)
            }
            _ => false,
        }
    }
}

/// Writes the components separated by `/`, the root-relative form a [`crate::Group`] handle
/// stores, and the empty string for the root group.
impl fmt::Display for ObjectPathBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_components(f, self.components.iter().map(LinkNameBuf::as_str))
    }
}

/// The name of one link in one group, a single component of an [`ObjectPath`].
///
/// A lookup that takes one of these cannot take a whole path. `..` is an ordinary name here, and
/// the C library resolves `a/../b` through a link named `..` (`H5G__traverse_real` in
/// `H5Gtraverse.c`, HDF5 1.14.6).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LinkName<'a>(&'a str);

impl<'a> LinkName<'a> {
    /// Returns `name` as the name of a link, or `None` for the empty string, for `.`, and for a
    /// name holding a separator, none of which is one component of a path.
    pub(crate) fn new(name: &'a str) -> Option<Self> {
        (!name.is_empty() && name != CURRENT_GROUP && !name.contains(SEPARATOR))
            .then_some(Self(name))
    }

    /// Returns the name as the group's link stores it.
    pub(crate) fn as_str(self) -> &'a str {
        self.0
    }
}

/// The name of one link in one group, owned: one component of an [`ObjectPathBuf`].
///
/// Built only from a [`LinkName`], so [`LinkName::new`] is the one gate on what a component may
/// be.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LinkNameBuf(String);

impl LinkNameBuf {
    /// Returns `name` as the owned name of a link, or `None` where [`LinkName::new`] rejects it.
    pub(crate) fn new(name: &str) -> Option<Self> {
        LinkName::new(name).map(Self::from)
    }

    /// Returns the name as the group's link stores it.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    /// Borrows the name as a [`LinkName`], the form a group lookup takes.
    pub(crate) fn as_link_name(&self) -> LinkName<'_> {
        LinkName(&self.0)
    }
}

/// Copies a parsed name into an owned one.
impl From<LinkName<'_>> for LinkNameBuf {
    fn from(name: LinkName<'_>) -> Self {
        Self(String::from(name.as_str()))
    }
}

/// Writes `names` separated by `/`, the form both path types display in.
fn write_components<'a>(
    f: &mut fmt::Formatter<'_>,
    names: impl Iterator<Item = &'a str>,
) -> fmt::Result {
    for (position, name) in names.enumerate() {
        if position > 0 {
            f.write_str(SEPARATOR)?;
        }
        f.write_str(name)?;
    }
    Ok(())
}

/// Separates the components of an object path.
const SEPARATOR: &str = "/";

/// The component `H5G__traverse_real` resolves to the group it is already in
/// (`H5Gtraverse.c`, HDF5 1.14.6), so it identifies no link.
const CURRENT_GROUP: &str = ".";

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{LinkName, LinkNameBuf, ObjectPath, ObjectPathBuf};

    fn path(spelling: &str) -> ObjectPathBuf {
        ObjectPathBuf::parse(spelling)
    }

    fn name(link_name: &str) -> LinkNameBuf {
        LinkNameBuf::from(LinkName::new(link_name).expect("a name that is one component"))
    }

    #[rstest]
    #[case("a/b")]
    #[case("/a/b")]
    #[case("a/b/")]
    #[case("/a/b/")]
    #[case("a//b")]
    #[case("///a///b///")]
    #[case("./a/b")]
    #[case("a/./b")]
    #[case("a/b/.")]
    fn every_spelling_of_a_path_parses_to_the_same_components(#[case] spelling: &str) {
        let path = ObjectPath::parse(spelling);
        let components: Vec<&str> = path
            .components()
            .iter()
            .copied()
            .map(LinkName::as_str)
            .collect();
        assert_eq!(components, ["a", "b"]);
    }

    #[rstest]
    #[case("")]
    #[case("/")]
    #[case("//")]
    #[case(".")]
    #[case("/./")]
    #[case("./.")]
    fn a_path_that_spells_no_link_has_no_components(#[case] spelling: &str) {
        let path = ObjectPath::parse(spelling);
        assert_eq!(path.components(), []);
        assert_eq!(path.to_string(), "");
    }

    #[rstest]
    #[case("/a//b/", "a/b")]
    #[case("a", "a")]
    fn a_parsed_path_displays_in_the_form_a_handle_stores(
        #[case] spelling: &str,
        #[case] stored: &str,
    ) {
        assert_eq!(ObjectPath::parse(spelling).to_string(), stored);
    }

    #[test]
    fn two_dots_is_an_ordinary_link_name() {
        assert_eq!(ObjectPath::parse("a/../b").to_string(), "a/../b");
        assert_eq!(LinkName::new("..").map(LinkName::as_str), Some(".."));
    }

    #[rstest]
    #[case("a", Some("a"))]
    #[case("", None)]
    #[case(".", None)]
    #[case("a/b", None)]
    #[case("/", None)]
    fn a_link_name_is_one_component_that_names_a_link(
        #[case] name: &str,
        #[case] link_name: Option<&str>,
    ) {
        assert_eq!(LinkName::new(name).map(LinkName::as_str), link_name);
    }

    #[rstest]
    #[case("/g/", "a//b", "g/a/b")]
    #[case("/g/", "", "g")]
    #[case("", "a", "a")]
    fn joining_a_relative_path_appends_its_components(
        #[case] base: &str,
        #[case] rest: &str,
        #[case] joined: &str,
    ) {
        assert_eq!(
            ObjectPath::parse(base)
                .join_path(&ObjectPath::parse(rest))
                .to_string(),
            joined
        );
    }

    #[rstest]
    #[case("/g/", "/h/", "h")]
    #[case("g/i", "/h/", "h")]
    #[case("", "/g/", "g")]
    fn joining_an_absolute_path_replaces_the_base(
        #[case] base: &str,
        #[case] rest: &str,
        #[case] joined: &str,
    ) {
        assert_eq!(
            ObjectPath::parse(base)
                .join_path(&ObjectPath::parse(rest))
                .to_string(),
            joined
        );
    }

    #[rstest]
    #[case(0, "")]
    #[case(2, "a/b")]
    #[case(3, "a/b/c")]
    #[case(9, "a/b/c")]
    fn a_prefix_names_the_object_reached_after_that_many_components(
        #[case] count: usize,
        #[case] prefix: &str,
    ) {
        assert_eq!(ObjectPath::parse("a/b/c").prefix(count).to_string(), prefix);
    }

    #[rstest]
    #[case("a/b")]
    #[case("/a/b")]
    #[case("a//b/")]
    #[case("./a/./b")]
    fn every_spelling_of_a_path_owns_the_same_components(#[case] spelling: &str) {
        assert_eq!(path(spelling), path("a/b"));
        assert_eq!(path(spelling).to_string(), "a/b");
    }

    #[rstest]
    #[case("")]
    #[case("/")]
    #[case(".")]
    fn a_path_that_spells_no_link_owns_the_root_group_path(#[case] spelling: &str) {
        assert_eq!(path(spelling), ObjectPathBuf::root());
        assert!(path(spelling).is_empty());
        assert_eq!(ObjectPathBuf::root().to_string(), "");
    }

    #[rstest]
    #[case("", "a")]
    #[case("g", "g/a")]
    #[case("/g/h/", "g/h/a")]
    fn joining_appends_one_component_to_an_owned_path(#[case] base: &str, #[case] joined: &str) {
        assert_eq!(path(base).join(&name("a")), path(joined));
    }

    #[rstest]
    #[case("/g/", "a//b", "g/a/b")]
    #[case("/g/", "", "g")]
    #[case("g/i", "/h/", "h")]
    fn joining_a_path_resolves_it_from_the_object_this_one_names(
        #[case] base: &str,
        #[case] rest: &str,
        #[case] joined: &str,
    ) {
        assert_eq!(path(base).join_path(&ObjectPath::parse(rest)), path(joined));
    }

    #[rstest]
    #[case(0, "")]
    #[case(2, "a/b")]
    #[case(9, "a/b/c")]
    fn an_owned_prefix_names_the_object_reached_after_that_many_components(
        #[case] count: usize,
        #[case] prefix: &str,
    ) {
        assert_eq!(path("a/b/c").prefix(count), path(prefix));
    }

    #[test]
    fn splitting_off_the_leaf_gives_the_group_holding_the_link_and_its_name() {
        let (parent, leaf) = path("a/b/c").split_leaf().expect("a path naming a link");
        assert_eq!(parent, path("a/b"));
        assert_eq!(leaf, name("c"));
        assert_eq!(ObjectPathBuf::root().split_leaf(), None);
    }

    #[rstest]
    #[case("g/a", Some("a"))]
    #[case("g", None)]
    #[case("g/a/b", None)]
    #[case("h/a", None)]
    fn a_name_under_a_group_is_the_leaf_of_a_direct_child(
        #[case] child: &str,
        #[case] link_name: Option<&str>,
    ) {
        assert_eq!(
            path(child).name_under(&path("g")).map(LinkNameBuf::as_str),
            link_name
        );
    }

    #[rstest]
    #[case("a/b", "a", true)]
    #[case("a/b", "a/b", true)]
    #[case("a/b", "", true)]
    #[case("a/b", "a/b/c", false)]
    #[case("a/b", "b", false)]
    fn a_path_starts_with_itself_and_with_each_of_its_ancestors(
        #[case] path_of: &str,
        #[case] prefix: &str,
        #[case] starts_with: bool,
    ) {
        assert_eq!(path(path_of).starts_with(&path(prefix)), starts_with);
    }

    #[rstest]
    #[case("a/b", "a", true)]
    #[case("a", "a/b", true)]
    #[case("a/b", "a/b", true)]
    #[case("a/b", "a/c", false)]
    fn two_paths_overlap_when_one_names_the_other_or_an_ancestor_of_it(
        #[case] one: &str,
        #[case] other: &str,
        #[case] overlaps: bool,
    ) {
        assert_eq!(path(one).overlaps(&path(other)), overlaps);
    }

    #[rstest]
    #[case("", true)]
    #[case("g", true)]
    #[case("g/col", true)]
    #[case("g/col/deeper", false)]
    #[case("g/other", false)]
    #[case("h", false)]
    fn a_path_covers_a_child_it_identifies_or_holds(#[case] prefix: &str, #[case] covers: bool) {
        assert_eq!(path(prefix).covers_child(&path("g"), &name("col")), covers);
    }

    #[test]
    fn an_owned_path_borrows_back_as_an_absolute_path() {
        let absolute = path("a/b");
        assert!(absolute.as_path().is_absolute());
        assert_eq!(absolute.as_path().to_string(), "a/b");
        assert_eq!(absolute.as_path().to_path_buf(), absolute);
    }
}
