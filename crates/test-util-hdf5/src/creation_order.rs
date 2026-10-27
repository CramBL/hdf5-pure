//! libhdf5 fixtures that track creation order.

#[cfg(feature = "__hdf5-1.10")]
use std::ffi::CString;
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

/// Writes a file with a link-tracked group and numbered datasets.
///
/// Applies `indexing` and `thresholds` to the root and to `group_path`. The thresholds are the
/// maximum compact and minimum dense link counts. Creates `count` datasets through
/// [`crate::dense_links::populate_edit_group`]. The file has a 512-byte userblock and persistent
/// free-space management. The caller owns `path` and removes the file after use.
///
/// # Panics
///
/// Panics if `path` is not valid UTF-8 or contains a null byte, an index does not fit in [`i32`],
/// or libhdf5 cannot configure, create, write, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_link_fixture(
    path: &Path,
    group_path: &str,
    count: usize,
    indexing: Indexing,
    thresholds: (u32, u32),
) {
    let mut builder = File::with_options();
    builder.with_fapl(|properties| {
        properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
    });
    builder.with_fcpl(|properties| {
        properties
            .link_creation_order(indexing.link_order())
            .userblock(512)
            .file_space_strategy(FileSpaceStrategy::FreeSpaceManager {
                paged: false,
                persist: true,
                threshold: 1,
            })
    });
    let properties = builder.fcpl().finish().unwrap();
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The library lock serializes access, and `properties` owns the live list.
            unsafe {
                hdf5_sys::h5p::H5Pset_link_phase_change(properties.id(), thresholds.0, thresholds.1)
            },
            0
        );
    });
    let access = builder.fapl().finish().unwrap();
    let filename = CString::new(path.to_str().unwrap()).unwrap();
    let file = hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access, all input handles remain live, and `from_id`
        // takes ownership of the new file handle.
        unsafe {
            hdf5::from_id::<File>(hdf5_sys::h5f::H5Fcreate(
                filename.as_ptr(),
                hdf5_sys::h5f::H5F_ACC_TRUNC,
                properties.id(),
                access.id(),
            ))
            .unwrap()
        }
    });
    let group = if group_path == "/" {
        file.group("/").unwrap()
    } else {
        let properties = GroupCreate::build()
            .link_creation_order(indexing.link_order())
            .finish()
            .unwrap();
        hdf5::sync::sync(|| {
            assert_eq!(
                // SAFETY: The lock serializes access, and `properties` owns the live list.
                unsafe {
                    hdf5_sys::h5p::H5Pset_link_phase_change(
                        properties.id(),
                        thresholds.0,
                        thresholds.1,
                    )
                },
                0
            );
        });
        file.create_group_builder()
            .set_gcpl(&properties)
            .create(group_path)
            .unwrap()
    };
    crate::dense_links::populate_edit_group(&group, count);
    drop(group);
    file.close().unwrap();
}
