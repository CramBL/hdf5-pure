//! The path of an object in a file, parsed into the sequence of link names to walk.
//!
//! [`ObjectPath::parse`] decides the grammar for the reader's resolvers, both group walks and the
//! type builders, the way the C library decides it: a repeated or a trailing separator separates
//! nothing (`H5G__component` in `H5Gname.c`), a component that is exactly `.` identifies the group
//! a walk has reached, and a leading separator starts the walk at the root group
//! (`H5G__traverse_real` in `H5Gtraverse.c`, HDF5 1.14.6).
//!
//! [`LinkName`] is one component of a path, the name of a link in one group.

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

    /// Returns the path of the child `name` of the object this path identifies.
    ///
    /// The child of an absolute path is absolute, so it starts the walk where this path starts it.
    pub(crate) fn join(&self, name: LinkName<'a>) -> Self {
        let mut components = self.components.clone();
        components.push(name);
        Self {
            components,
            absolute: self.absolute,
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
        for (position, name) in self.components.iter().enumerate() {
            if position > 0 {
                f.write_str(SEPARATOR)?;
            }
            f.write_str(name.as_str())?;
        }
        Ok(())
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

/// Separates the components of an object path.
const SEPARATOR: &str = "/";

/// The component `H5G__traverse_real` resolves to the group it is already in
/// (`H5Gtraverse.c`, HDF5 1.14.6), so it identifies no link.
const CURRENT_GROUP: &str = ".";

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{LinkName, ObjectPath};

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
    #[case("", "a")]
    #[case("g", "g/a")]
    #[case("/g/h/", "g/h/a")]
    fn joining_appends_one_component_to_a_path(#[case] base: &str, #[case] joined: &str) {
        let name = LinkName::new("a").unwrap();
        assert_eq!(ObjectPath::parse(base).join(name).to_string(), joined);
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
}
