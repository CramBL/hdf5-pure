//! Helpers for C-library fixtures that track creation order.

use hdf5::plist::group_create::{
    AttrCreationOrder, GroupCreate, GroupCreateBuilder, LinkCreationOrder,
};

/// Controls whether tracked creation order is also indexed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Indexing {
    /// Tracks creation order without building its index.
    No,
    /// Tracks creation order and builds its index.
    Yes,
}

impl Indexing {
    /// Returns the corresponding attribute creation-order property.
    pub fn attr_order(self) -> AttrCreationOrder {
        match self {
            Self::No => AttrCreationOrder::Tracked,
            Self::Yes => AttrCreationOrder::Indexed,
        }
    }

    /// Returns the corresponding link creation-order property.
    pub fn link_order(self) -> LinkCreationOrder {
        match self {
            Self::No => LinkCreationOrder::Tracked,
            Self::Yes => LinkCreationOrder::Indexed,
        }
    }
}

/// Builds group creation properties that track attribute creation order.
///
/// Link creation order is tracked too when `links` is true.
pub fn group_properties(indexing: Indexing, links: bool) -> GroupCreate {
    let mut builder = GroupCreateBuilder::new();
    builder.attr_creation_order(indexing.attr_order());
    if links {
        builder.link_creation_order(indexing.link_order());
    }
    builder.finish().expect("a group creation property list")
}

/// Returns deterministic attribute names for creation-order fixtures.
pub fn names(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("a{i:02}")).collect()
}
