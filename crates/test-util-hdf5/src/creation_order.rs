//! Helpers for C-library fixtures that track creation order.

use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy;
use hdf5::plist::group_create::{
    AttrCreationOrder, GroupCreate, GroupCreateBuilder, LinkCreationOrder,
};
use hdf5::{File, Location};

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

/// Writes a file whose `/g` group and `/d` dataset track attribute creation order.
///
/// Each object receives one scalar `i32` attribute per entry in `names`, in the
/// supplied order, with values equal to their positions. The root group tracks the
/// same order. When `persist_free_space` is true, the file also uses the persistent
/// free-space manager required by reclamation tests.
pub fn write_attribute_fixture(
    path: &Path,
    names: &[String],
    indexing: Indexing,
    persist_free_space: bool,
) {
    let file = File::with_options()
        .with_fapl(|properties| {
            let lower = if persist_free_space {
                LibraryVersion::V110
            } else {
                LibraryVersion::V18
            };
            properties.libver_bounds(lower, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties.attr_creation_order(indexing.attr_order());
            if persist_free_space {
                properties.file_space_strategy(FileSpaceStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            } else {
                properties
            }
        })
        .create(path)
        .unwrap_or_else(|error| panic!("create {}: {error}", path.display()));
    let group = file
        .create_group_builder()
        .set_gcpl(&group_properties(indexing, false))
        .create("g")
        .expect("create group");
    let dataset = file
        .new_dataset::<i32>()
        .with_dcpl(|properties| properties.attr_creation_order(indexing.attr_order()))
        .shape([4])
        .create("d")
        .expect("create dataset");
    let owners: [&Location; 2] = [&group, &dataset];
    for owner in owners {
        for (index, name) in names.iter().enumerate() {
            owner
                .new_attr::<i32>()
                .shape(())
                .create(name.as_str())
                .unwrap_or_else(|error| panic!("create attribute {name}: {error}"))
                .write_scalar(&(index as i32))
                .unwrap_or_else(|error| panic!("write attribute {name}: {error}"));
        }
    }
    file.close().unwrap();
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
