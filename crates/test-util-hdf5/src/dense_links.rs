//! Dense group-link fixtures and observations through libhdf5.

#[cfg(feature = "__hdf5-1.10")]
use std::ffi::CString;
use std::path::Path;

use hdf5::File;
use hdf5::Group;
use hdf5::GroupStorageType;
use hdf5::datatype::ByteOrder as CByteOrder;
use hdf5::file::LibraryVersion;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::GroupCreate;
use hdf5::plist::group_create::LinkCreationOrder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::link_create::CharEncoding;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::FileBuilder;
use hdf5_pure::FixedPointType;
use hdf5_pure::LibVer;

/// The library that writes a group-link fixture.
#[derive(Clone, Copy, Debug)]
pub enum Producer {
    /// Writes through libhdf5.
    Libhdf5,
    /// Writes through [`FileBuilder`].
    Pure,
}

/// A group's link count, storage form, and member names read through libhdf5.
#[derive(Debug)]
pub struct GroupObservation {
    /// The number of links in the group.
    pub link_count: u64,
    /// The group's link storage form.
    pub storage: GroupStorageType,
    /// The member names in libhdf5's iteration order.
    pub names: Vec<String>,
}

/// A group opened through libhdf5 for fixture observations.
#[derive(Debug)]
pub struct GroupOracle {
    group: Group,
}

impl GroupOracle {
    /// Reads the group's link count, storage form, and member names through libhdf5.
    ///
    /// # Panics
    ///
    /// Panics if libhdf5 cannot inspect the group or enumerate its members.
    pub fn info(&self) -> GroupObservation {
        let info = self.group.info().unwrap();
        GroupObservation {
            link_count: info.nlinks,
            storage: info.storage_type,
            names: self.group.member_names().unwrap(),
        }
    }

    /// Reads the fixture child at `index` through libhdf5.
    ///
    /// Opens the child named by [`link_name`] and selects its type with `index % 3`,
    /// matching [`write_fixture`].
    ///
    /// # Panics
    ///
    /// Panics if libhdf5 cannot open or read the child.
    pub fn child(&self, index: usize) -> ChildObservation {
        let name = link_name(index);
        match index % 3 {
            0 => ChildObservation::Dataset(
                self.group
                    .dataset(&name)
                    .unwrap()
                    .read_raw::<i32>()
                    .unwrap(),
            ),
            1 => {
                let child = self.group.group(&name).unwrap();
                ChildObservation::Group {
                    names: child.member_names().unwrap(),
                    values: child.dataset("payload").unwrap().read_raw::<i32>().unwrap(),
                }
            }
            _ => {
                let datatype = self.group.committed_datatype(&name).unwrap();
                ChildObservation::Datatype {
                    size: datatype.as_datatype().size(),
                    byte_order: datatype.as_datatype().byte_order(),
                }
            }
        }
    }
}

/// A fixture child's values or datatype metadata read through libhdf5.
#[derive(Debug)]
pub enum ChildObservation {
    /// The dataset's element values.
    Dataset(Vec<i32>),
    /// The committed datatype's size and byte order.
    Datatype {
        /// The datatype's size in bytes.
        size: usize,
        /// The datatype's byte order.
        byte_order: CByteOrder,
    },
    /// A subgroup's member names and its `payload` dataset's element values.
    Group {
        /// The member names in libhdf5's iteration order.
        names: Vec<String>,
        /// The element values of the `payload` dataset.
        values: Vec<i32>,
    },
}

/// A dataset's values and its root group's link storage form read through libhdf5.
#[derive(Debug)]
pub struct DatasetObservation {
    /// The root group's link storage form.
    pub storage: GroupStorageType,
    /// The dataset's element values.
    pub values: Vec<i32>,
}

/// Writes a group-link fixture through the selected library.
///
/// For [`Producer::Libhdf5`], writes `count` fixture children in both the root and
/// `parent/group`, alternating datasets, groups, and committed datatypes. The root also
/// contains `parent`. Sets the userblock size to `userblock` bytes. For [`Producer::Pure`],
/// calls `build_pure` and streams output when `streamed` is `true`.
///
/// # Panics
///
/// Panics if creating or writing the file fails, or a child index does not fit in [`i32`].
pub fn write_fixture(
    path: &Path,
    count: usize,
    userblock: u64,
    streamed: bool,
    producer: Producer,
    build_pure: impl FnOnce() -> FileBuilder,
) {
    match producer {
        Producer::Libhdf5 => write_libhdf5_fixture(path, count, userblock),
        Producer::Pure if streamed => build_pure()
            .finish_to(std::fs::File::create(path).unwrap())
            .unwrap(),
        Producer::Pure => build_pure().write(path).unwrap(),
    }
}

/// Writes nine root datasets with the last named `link_name` and holding `[42]`.
///
/// # Panics
///
/// Panics if creating or writing the file fails.
pub fn write_limit_fixture(path: &Path, link_name: &str, producer: Producer) {
    match producer {
        Producer::Libhdf5 => {
            let reference = hdf5::FileBuilder::new()
                .with_fapl(|properties| {
                    properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
                })
                .create(path)
                .unwrap();
            for index in 0..MAX_COMPACT_LINKS {
                reference
                    .new_dataset::<i32>()
                    .shape(1)
                    .create(format!("data_{index}").as_str())
                    .unwrap()
                    .write(&[i32::try_from(index).unwrap()])
                    .unwrap();
            }
            reference
                .new_dataset::<i32>()
                .shape(1)
                .create(link_name)
                .unwrap()
                .write(&[42])
                .unwrap();
        }
        Producer::Pure => {
            let mut builder = FileBuilder::new();
            if !cfg!(feature = "__hdf5-1.10") {
                builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
            }
            for index in 0..MAX_COMPACT_LINKS {
                builder
                    .create_dataset(&format!("data_{index}"))
                    .with_i32_data(&[i32::try_from(index).unwrap()]);
            }
            builder.create_dataset(link_name).with_i32_data(&[42]);
            builder.write(path).unwrap();
        }
    }
}

/// Opens the group at `group_path` through libhdf5.
///
/// # Panics
///
/// Panics if libhdf5 cannot open the file or group.
pub fn group(path: &Path, group_path: &str) -> GroupOracle {
    GroupOracle {
        group: hdf5::File::open(path).unwrap().group(group_path).unwrap(),
    }
}

/// Reads a root dataset and the root group's link storage form through libhdf5.
///
/// # Panics
///
/// Panics if libhdf5 cannot open, inspect, or read the file or dataset.
pub fn observe_dataset(path: &Path, link_name: &str) -> DatasetObservation {
    let file = hdf5::File::open(path).unwrap();
    DatasetObservation {
        storage: file.info().unwrap().storage_type,
        values: file.dataset(link_name).unwrap().read_raw::<i32>().unwrap(),
    }
}

fn write_libhdf5_fixture(path: &Path, count: usize, userblock: u64) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .with_fcpl(|properties| properties.userblock(userblock))
        .create(path)
        .unwrap();
    let parent = file.create_group("parent").unwrap();
    let nested = parent.create_group("group").unwrap();
    for group in [&file.group("/").unwrap(), &nested] {
        for index in 0..count {
            let name = link_name(index);
            match index % 3 {
                0 => {
                    let builder = group.new_dataset::<i32>().shape(2);
                    let builder = if index % 6 == 0 {
                        builder.chunk([1])
                    } else {
                        builder
                    };
                    builder
                        .create(name.as_str())
                        .unwrap()
                        .write(&[i32::try_from(index).unwrap(), 17])
                        .unwrap();
                }
                1 => {
                    group
                        .create_group(&name)
                        .unwrap()
                        .new_dataset::<i32>()
                        .shape(1)
                        .create("payload")
                        .unwrap()
                        .write(&[23])
                        .unwrap();
                }
                _ => {
                    group
                        .commit_datatype(&name, &hdf5::Datatype::from_type::<i32>().unwrap())
                        .unwrap();
                }
            }
        }
        for index in 0..9 {
            group
                .new_attr::<i32>()
                .shape(())
                .create(format!("attr_{index}").as_str())
                .unwrap()
                .write_scalar(&index)
                .unwrap();
        }
    }
}

/// Returns a fixture name with hash collisions at indexes zero and one.
///
/// Later names include the index and alternate between ASCII and non-ASCII characters.
pub fn link_name(index: usize) -> String {
    match index {
        0 => "k69209".into(),
        1 => "k155448".into(),
        index if index % 2 == 0 => format!("child_{index:04}"),
        index => format!("child_{index:04}_λ"),
    }
}

/// Returns the signed little-endian 32-bit datatype used by fixture children.
pub fn integer_type() -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    )
}

/// Checks the fixture group's storage, creation-order policy, and shared link targets.
///
/// # Panics
///
/// Panics if libhdf5 cannot read the group or its links, or a fixture property differs from
/// the expected value.
pub fn assert_inspection_group(
    file: &File,
    group_path: &str,
    count: u64,
    storage: GroupStorageType,
    order: LinkCreationOrder,
) {
    let group = file.group(group_path).unwrap();
    let info = group.info().unwrap();
    assert_eq!(info.nlinks, count);
    assert_eq!(info.storage_type, storage);
    assert_eq!(group.gcpl().unwrap().link_creation_order(), order);
    assert_eq!(
        group.group("hard").unwrap().name(),
        format!("/{group_path}/hard")
    );
    assert_eq!(
        group.group("soft").unwrap().name(),
        format!("/{group_path}/soft")
    );
    assert_eq!(group.member_names().unwrap().len() as u64, count);
    assert_eq!(group.loc_info_by_name("hard").unwrap().num_links, 7);
}

/// Writes compact, dense, and filtered group fixtures with libhdf5.
///
/// Appends `policy` to the compact and dense group names and configures link creation order
/// with `order`. Each compact or dense group contains hard, soft, and external links, a UTF-8
/// name, and a name-hash collision. The caller owns `path` and removes the file after use.
///
/// # Panics
///
/// Panics if libhdf5 cannot create the file, configure or write its groups, or close it.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_inspection_fixture(path: &Path, policy: &str, order: LinkCreationOrder) {
    let file = File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .with_fcpl(|properties| properties.userblock(512))
        .create(path)
        .unwrap();
    file.create_group("target").unwrap();
    for (form, count) in [("compact", 6), ("dense", 2000)] {
        let properties = GroupCreate::build()
            .link_creation_order(order)
            .finish()
            .unwrap();
        hdf5::sync::sync(|| {
            assert_eq!(
                // SAFETY: The library lock serializes access, and `properties` keeps the group
                // creation property list live throughout the call.
                unsafe { hdf5_sys::h5p::H5Pset_link_phase_change(properties.id(), 12, 4) },
                0
            );
        });
        let group = file
            .create_group_builder()
            .set_gcpl(&properties)
            .create(format!("{form}_{policy}").as_str())
            .unwrap();
        group.link_soft("/target", "removed").unwrap();
        group.link_hard("/target", "hard").unwrap();
        group.link_soft("/target", "soft").unwrap();
        group.link_external("外部.h5", "/目标", "external").unwrap();
        group
            .create_group_builder()
            .char_encoding(CharEncoding::Utf8)
            .create("utf8_λ")
            .unwrap();
        group.link_hard("/target", "k69209").unwrap();
        group.link_hard("/target", "k155448").unwrap();
        for index in 6..count {
            group
                .link_soft("/target", &format!("link_{index:04}"))
                .unwrap();
        }
        group.unlink("removed").unwrap();
    }
    for index in 0..7 {
        file.create_group(&format!("root_{index}")).unwrap();
    }
    let properties = GroupCreate::build().finish().unwrap();
    assert_eq!(crate::group_filter::set_deflate(&properties, 6), 0);
    let filtered = file
        .create_group_builder()
        .set_gcpl(&properties)
        .create("filtered")
        .unwrap();
    for index in 0..16 {
        filtered
            .link_soft("/target", &format!("link_{index:04}"))
            .unwrap();
    }
    drop(filtered);
    file.close().unwrap();
}

/// Writes a group with links stored as huge fractal-heap objects.
///
/// The `g` group contains twelve datasets with link names longer than 5000 bytes.
/// The caller owns `path` and removes the file after use.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_huge_inspection_fixture(path: &Path) {
    let file = File::with_options()
        .with_fapl(|properties| properties.libver_latest())
        .create(path)
        .unwrap();
    let group = file.create_group("g").unwrap();
    for index in 0..12 {
        group
            .new_dataset::<i32>()
            .shape(1)
            .create(format!("data_{index}_{}", "x".repeat(5000)).as_str())
            .unwrap()
            .write(&[index])
            .unwrap();
    }
    drop(group);
    file.close().unwrap();
}

/// Writes an untracked group with numbered datasets for in-place editing tests.
///
/// Creates `count` datasets in `group_path` through [`populate_edit_group`]. Applies
/// `thresholds` as the maximum compact and minimum dense link counts when supplied,
/// including for the root group. The caller owns `path` and removes the file after use.
///
/// # Panics
///
/// Panics if libhdf5 cannot configure, create, write, or close the fixture, or an index
/// does not fit in [`i32`].
#[cfg(feature = "__hdf5-1.10")]
pub fn write_edit_fixture(
    path: &Path,
    group_path: &str,
    count: usize,
    thresholds: Option<(u32, u32)>,
) {
    if group_path == "/"
        && let Some(thresholds) = thresholds
    {
        write_edit_root_fixture(path, count, thresholds);
        return;
    }
    let reference = hdf5::FileBuilder::new()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .create(path)
        .unwrap();
    let group = if group_path == "/" {
        reference.group("/").unwrap()
    } else {
        let properties = GroupCreate::build().finish().unwrap();
        if let Some((max_compact, min_dense)) = thresholds {
            hdf5::sync::sync(|| {
                assert_eq!(
                    // SAFETY: The library lock serializes access and `properties` owns the live list.
                    unsafe {
                        hdf5_sys::h5p::H5Pset_link_phase_change(
                            properties.id(),
                            max_compact,
                            min_dense,
                        )
                    },
                    0
                );
            });
        }
        reference
            .create_group_builder()
            .set_gcpl(&properties)
            .create(group_path)
            .unwrap()
    };
    populate_edit_group(&group, count);
    drop(group);
    reference.close().unwrap();
}

#[cfg(feature = "__hdf5-1.10")]
fn write_edit_root_fixture(path: &Path, count: usize, thresholds: (u32, u32)) {
    let mut builder = hdf5::FileBuilder::new();
    builder.with_fapl(|properties| {
        properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
    });
    let properties = builder.fcpl().finish().unwrap();
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The library lock serializes access and `properties` owns the live list.
            unsafe {
                hdf5_sys::h5p::H5Pset_link_phase_change(properties.id(), thresholds.0, thresholds.1)
            },
            0
        );
    });
    let access = builder.fapl().finish().unwrap();
    let filename = CString::new(path.to_str().unwrap()).unwrap();
    let reference = hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access. The filename and property lists remain
        // live during creation, and `from_id` takes ownership of the newly created file handle.
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
    populate_edit_group(&reference, count);
    reference.close().unwrap();
}

/// Writes `count` datasets named `link_0000`, `link_0001`, and so on in `group`.
///
/// Each dataset contains its index as an [`i32`] followed by 42.
///
/// # Panics
///
/// Panics if libhdf5 cannot create or write a dataset, or an index does not fit in [`i32`].
#[cfg(feature = "__hdf5-1.10")]
pub fn populate_edit_group(group: &Group, count: usize) {
    for index in 0..count {
        group
            .new_dataset::<i32>()
            .shape(2)
            .create(format!("link_{index:04}").as_str())
            .unwrap()
            .write(&[i32::try_from(index).unwrap(), 42])
            .unwrap();
    }
}

// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, is the default compact link limit.
const MAX_COMPACT_LINKS: usize = 8;
