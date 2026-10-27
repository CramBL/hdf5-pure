//! libhdf5 fixtures that track creation order.

#[cfg(feature = "__hdf5-1.10")]
use std::ffi::CString;
use std::path::Path;

#[cfg(feature = "__hdf5-1.10")]
use hdf5::GroupStorageType;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::IndexType;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::IterationOrder;
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

/// Checks numbered dataset values through libhdf5.
///
/// Expects each `link_<index>` dataset for indexes in `removed..count` to contain
/// `[index, 42]`, with the index zero-padded to at least four decimal digits.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, an index does not fit in [`i32`], or a
/// dataset value differs from the expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_link_dataset_values(path: &Path, group_path: &str, removed: usize, count: usize) {
    let reference = File::open(path).unwrap();
    let group = reference.group(group_path).unwrap();
    for index in removed..count {
        assert_eq!(
            group
                .dataset(&format!("link_{index:04}"))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [i32::try_from(index).unwrap(), 42]
        );
    }
    drop(group);
    reference.close().unwrap();
}

/// Creates a group through libhdf5 and closes the file.
///
/// # Panics
///
/// Panics if libhdf5 cannot open the file, create the group, or close the file.
#[cfg(feature = "__hdf5-1.10")]
pub fn add_reference_group(path: &Path, group_path: &str) {
    let reference = File::open_rw(path).unwrap();
    reference.create_group(group_path).unwrap();
    reference.close().unwrap();
}

/// Checks replacement and copied dataset values through libhdf5.
///
/// Expects `link_0000` to contain `[99]` and `copy` to contain `[1, 42]`.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or a dataset value differs from the
/// expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_link_replacements(path: &Path) {
    let reference = File::open(path).unwrap();
    assert_eq!(
        reference
            .dataset("link_0000")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [99]
    );
    assert_eq!(
        reference
            .dataset("copy")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [1, 42]
    );
    reference.close().unwrap();
}

/// Writes dense root and child groups with tracked link creation order through libhdf5.
///
/// Each group contains twelve numbered datasets. Applies `indexing` to both groups.
///
/// # Panics
///
/// Panics if `path` is not valid UTF-8 or contains a null byte, or libhdf5 cannot create, write,
/// or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_link_ancestor_fixture(path: &Path, indexing: Indexing) {
    write_link_fixture(path, "/", 12, indexing, (8, 6));
    {
        let reference = File::open_rw(path).unwrap();
        let group = reference
            .create_group_builder()
            .with_gcpl(|properties| properties.link_creation_order(indexing.link_order()))
            .create("child")
            .unwrap();
        crate::dense_links::populate_edit_group(&group, 12);
        drop(group);
        reference.close().unwrap();
    }
}

/// Checks a surviving child dataset and its added attribute through libhdf5.
///
/// Expects `child/link_0003` to contain `[3, 42]` and its scalar `added` attribute to contain 99.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or an expected value differs.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_link_ancestor_dataset(path: &Path) {
    let reference = File::open(path).unwrap();
    let dataset = reference.dataset("child/link_0003").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [3, 42]);
    assert_eq!(
        dataset.attr("added").unwrap().read_scalar::<i64>().unwrap(),
        99
    );
    drop(dataset);
    reference.close().unwrap();
}

/// Checks group storage and link creation-order metadata through libhdf5.
///
/// Compares link names and creation indexes in increasing creation order with `expected`.
/// Checks the next creation index against `next` and the creation-order policy against `indexing`.
/// Checks dense storage when `dense` is true and compact storage otherwise.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, a creation index does not fit in its
/// integer type, or the metadata differs from the expected metadata.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_link_group(
    path: &Path,
    group_path: &str,
    expected: &[(String, u64)],
    next: u64,
    indexing: Indexing,
    dense: bool,
) {
    let reference = File::open(path).unwrap();
    let group = reference.group(group_path).unwrap();
    let info = group.info().unwrap();
    assert_eq!(info.max_corder, i64::try_from(next).unwrap());
    assert_eq!(
        info.storage_type,
        if dense {
            GroupStorageType::Dense
        } else {
            GroupStorageType::Compact
        }
    );
    assert_eq!(
        group.gcpl().unwrap().link_creation_order(),
        indexing.link_order()
    );
    let links = group
        .links(IndexType::CreationOrder, IterationOrder::Increasing)
        .unwrap();
    let actual = links
        .into_iter()
        .map(|(name, info)| (name, u64::try_from(info.creation_order.unwrap()).unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    drop(group);
    reference.close().unwrap();
}
