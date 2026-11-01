//! Dense group-link fixtures and observations through libhdf5.

#[cfg(feature = "__hdf5-1.10")]
use std::ffi::CString;
use std::path::Path;

use hdf5::File;
use hdf5::Group;
use hdf5::GroupStorageType;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::IndexType;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::IterationOrder;
use hdf5::datatype::ByteOrder as CByteOrder;
use hdf5::file::LibraryVersion;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::GroupCreate;
use hdf5::plist::group_create::LinkCreationOrder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::group_create::LinkPhaseChange;
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

/// The creation-order policy and name suffix of a group fixture.
#[derive(Clone, Copy, Debug)]
pub enum GroupOrder {
    /// Tracks creation order and stores a creation-order index.
    Indexed,
    /// Tracks creation order without storing a creation-order index.
    Tracked,
    /// Omits creation-order tracking.
    Untracked,
}

impl GroupOrder {
    /// Returns the suffix appended to compact and dense fixture group names.
    pub fn label(self) -> &'static str {
        match self {
            Self::Indexed => "indexed",
            Self::Tracked => "tracked",
            Self::Untracked => "untracked",
        }
    }

    /// Returns the libhdf5 group creation setting for this policy.
    pub fn creation_order(self) -> LinkCreationOrder {
        match self {
            Self::Indexed => LinkCreationOrder::Indexed,
            Self::Tracked => LinkCreationOrder::Tracked,
            Self::Untracked => LinkCreationOrder::Untracked,
        }
    }
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
/// Appends the label of `order` to compact and dense group names and configures their link
/// creation order with the same policy. Each compact or dense group contains hard, soft, and
/// external links, a UTF-8 name, and a name-hash collision. The caller owns `path` and removes
/// the file after use.
///
/// # Panics
///
/// Panics if libhdf5 cannot create the file, configure or write its groups, or close it.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_inspection_fixture(path: &Path, order: GroupOrder) {
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
            .link_creation_order(order.creation_order())
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
            .create(format!("{form}_{}", order.label()).as_str())
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

/// Reads the link count of the `added` group beneath `parent` through libhdf5.
///
/// # Panics
///
/// Panics if libhdf5 cannot open, inspect, or close the file or group.
#[cfg(feature = "__hdf5-1.10")]
pub fn added_group_link_count(path: &Path, parent: &str) -> u64 {
    let file = File::open(path).unwrap();
    let group = file.group(&format!("/{parent}/added")).unwrap();
    let count = group.info().unwrap().nlinks;
    drop(group);
    file.close().unwrap();
    count
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
    let reference = create_with_root_link_properties(path, thresholds, GroupOrder::Untracked, 0);
    populate_edit_group(&reference, count);
    reference.close().unwrap();
}

#[cfg(feature = "__hdf5-1.10")]
fn create_with_root_link_properties(
    path: &Path,
    thresholds: (u32, u32),
    order: GroupOrder,
    userblock: u64,
) -> File {
    let mut builder = hdf5::FileBuilder::new();
    builder.with_fapl(|properties| {
        properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
    });
    let properties = builder.fcpl().userblock(userblock).finish().unwrap();
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The library lock serializes access and `properties` owns the live list.
            unsafe {
                hdf5_sys::h5p::H5Pset_link_phase_change(properties.id(), thresholds.0, thresholds.1)
            },
            0
        );
        assert_eq!(
            // SAFETY: The library lock serializes access and `properties` owns the live list.
            unsafe {
                hdf5_sys::h5p::H5Pset_link_creation_order(
                    properties.id(),
                    order.creation_order().into(),
                )
            },
            0
        );
    });
    let access = builder.fapl().finish().unwrap();
    let filename = CString::new(path.to_str().unwrap()).unwrap();
    hdf5::sync::sync(|| {
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
    })
}

/// Renames a link through libhdf5 using raw name bytes.
///
/// # Panics
///
/// Panics if a name contains a null byte or libhdf5 cannot move the link.
#[cfg(feature = "__hdf5-1.10")]
pub fn rename_link_bytes(group: &Group, old_name: &[u8], new_name: &[u8]) {
    let old_name = CString::new(old_name).unwrap();
    let new_name = CString::new(new_name).unwrap();
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The library lock serializes access. The group and both terminated names
            // remain live, and the default property lists need no owned handles.
            unsafe {
                hdf5_sys::h5l::H5Lmove(
                    group.id(),
                    old_name.as_ptr(),
                    group.id(),
                    new_name.as_ptr(),
                    hdf5_sys::h5p::H5P_DEFAULT,
                    hdf5_sys::h5p::H5P_DEFAULT,
                )
            },
            0
        );
    });
}

/// Returns group member names in increasing creation order through libhdf5.
///
/// # Panics
///
/// Panics if libhdf5 cannot enumerate the group.
#[cfg(feature = "__hdf5-1.10")]
pub fn creation_order_names(group: &Group) -> Vec<String> {
    let mut names = Vec::new();
    group
        .iter_visit(
            IndexType::CreationOrder,
            IterationOrder::Increasing,
            |name, _| {
                names.push(name.to_owned());
                Ok(())
            },
        )
        .unwrap();
    names
}

/// Writes a native group with mixed link kinds and a creation-index gap.
///
/// Uses link thresholds 12/8 and a 512-byte userblock. Applies `order` to `group_path`. Creates
/// six live links and extra groups for each index in `6..count`, after deleting an earlier link.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, configure, populate, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_repack_link_fixture(path: &Path, order: GroupOrder, group_path: &str, count: usize) {
    let file = create_with_root_link_properties(
        path,
        (12, 8),
        if group_path == "/" {
            order
        } else {
            GroupOrder::Untracked
        },
        512,
    );
    let group = if group_path == "/" {
        file.group("/").unwrap()
    } else {
        file.create_group_builder()
            .with_gcpl(|properties| {
                properties
                    .link_creation_order(order.creation_order())
                    .link_phase_change(LinkPhaseChange {
                        max_compact: 12,
                        min_dense: 8,
                    })
            })
            .create(group_path)
            .unwrap()
    };
    group.link_soft("/value", "removed").unwrap();
    group
        .new_dataset_builder()
        .with_data(&[7_i32, 17])
        .create("value")
        .unwrap();
    group.create_group("child").unwrap();
    group
        .commit_datatype("type", &hdf5::Datatype::from_type::<i32>().unwrap())
        .unwrap();
    group.link_soft("value", "soft").unwrap();
    group.link_external("外部.h5", "/目标", "external").unwrap();
    group
        .create_group_builder()
        .char_encoding(CharEncoding::Utf8)
        .create("λ")
        .unwrap();
    for index in 6..count {
        group.create_group(&format!("extra_{index:04}")).unwrap();
    }
    group.unlink("removed").unwrap();
    drop(group);
    file.close().unwrap();
}

/// Writes a native group with configurable link thresholds and entry estimates.
///
/// Creates `group` with `thresholds` and `estimates`, then populates it with `count` datasets
/// through [`populate_edit_group`].
///
/// # Panics
///
/// Panics if libhdf5 cannot create or configure the file or group, or if populating the group fails.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_group_info_fixture(
    path: &Path,
    thresholds: (u32, u32),
    estimates: (u32, u32),
    count: usize,
) {
    let reference = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_v110())
        .create(path)
        .unwrap();
    let properties = GroupCreate::build()
        .link_phase_change(LinkPhaseChange {
            max_compact: thresholds.0,
            min_dense: thresholds.1,
        })
        .finish()
        .unwrap();
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The library lock serializes access and `properties` owns the live list.
            unsafe {
                hdf5_sys::h5p::H5Pset_est_link_info(properties.id(), estimates.0, estimates.1)
            },
            0
        );
    });
    let group = reference
        .create_group_builder()
        .set_gcpl(&properties)
        .create("group")
        .unwrap();
    populate_edit_group(&group, count);
    drop(group);
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

/// Checks the edited group's storage, member names, and surviving dataset values through libhdf5.
///
/// Expects numbered dataset names at indexes `removed..count` and `added_<index>` names at
/// indexes `0..added`. Expects dense storage when `dense` is true and compact storage otherwise.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, a value differs from the expected value,
/// or a dataset index does not fit in [`i32`].
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_edit_group(
    path: &Path,
    group_path: &str,
    count: usize,
    added: usize,
    removed: usize,
    dense: bool,
) {
    let reference = File::open(path).unwrap();
    let group = reference.group(group_path).unwrap();
    assert_eq!(
        group.info().unwrap().storage_type,
        if dense {
            GroupStorageType::Dense
        } else {
            GroupStorageType::Compact
        }
    );
    let mut expected = (removed..count)
        .map(|index| format!("link_{index:04}"))
        .collect::<Vec<_>>();
    expected.extend((0..added).map(|index| format!("added_{index}")));
    expected.sort();
    let mut actual = group.member_names().unwrap();
    actual.sort();
    assert_eq!(actual, expected);
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

/// Writes a dense group with hash-colliding names, symbolic links, and a UTF-8 group name.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_collision_edit_fixture(path: &Path, group_path: &str) {
    write_edit_fixture(path, group_path, 2000, None);
    {
        let reference = File::open_rw(path).unwrap();
        let group = reference.group(group_path).unwrap();
        group.link_soft("/absent_λ", "soft").unwrap();
        group
            .link_external("absent_λ.h5", "/目标", "external")
            .unwrap();
        group
            .new_dataset::<i32>()
            .shape(1)
            .create("k69209")
            .unwrap()
            .write(&[7])
            .unwrap();
        group
            .new_dataset::<i32>()
            .shape(1)
            .create("k155448")
            .unwrap()
            .write(&[9])
            .unwrap();
        group
            .create_group_builder()
            .char_encoding(CharEncoding::Utf8)
            .create("原始_λ")
            .unwrap();
        drop(group);
        reference.close().unwrap();
    }
}

/// Checks the edited collision fixture's storage, link count, dataset values, and group attribute.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or a value differs from the expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_collision_edit(path: &Path, group_path: &str) {
    let reference = File::open(path).unwrap();
    let group = reference.group(group_path).unwrap();
    assert_eq!(group.info().unwrap().storage_type, GroupStorageType::Dense);
    assert_eq!(group.info().unwrap().nlinks, 2005);
    assert_eq!(
        group.dataset("k69209").unwrap().read_raw::<i32>().unwrap(),
        [7]
    );
    assert_eq!(
        group.dataset("k155448").unwrap().read_raw::<i32>().unwrap(),
        [9]
    );
    assert_eq!(
        group
            .dataset("link_1999")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [1999, 42]
    );
    assert_eq!(
        group.attr("changed").unwrap().read_scalar::<i64>().unwrap(),
        17
    );
    drop(group);
    reference.close().unwrap();
}

/// Writes dense root and child groups with numbered datasets for descendant edits.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_descendant_edit_fixture(path: &Path) {
    write_edit_fixture(path, "/", 12, None);
    {
        let reference = File::open_rw(path).unwrap();
        let child = reference.create_group("child").unwrap();
        populate_edit_group(&child, 12);
        drop(child);
        reference.close().unwrap();
    }
}

/// Checks dense storage, surviving datasets, and edits in the `child` group through libhdf5.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or a value differs from the expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_descendant_edit(path: &Path) {
    let reference = File::open(path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    let child = reference.group("child").unwrap();
    assert_eq!(child.info().unwrap().storage_type, GroupStorageType::Dense);
    assert_eq!(child.info().unwrap().nlinks, 14);
    assert_eq!(
        child
            .dataset("link_0005")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [5, 42]
    );
    assert_eq!(
        child
            .dataset("link_0005")
            .unwrap()
            .attr("new")
            .unwrap()
            .read_scalar::<i64>()
            .unwrap(),
        99
    );
    assert_eq!(
        child.dataset("copied").unwrap().read_raw::<i32>().unwrap(),
        [6, 42]
    );
    assert_eq!(child.group("nested").unwrap().info().unwrap().nlinks, 0);
    assert_eq!(
        reference
            .dataset("link_0011")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [11, 42]
    );
    drop(child);
    reference.close().unwrap();
}

/// Writes the dense group `g` with numbered datasets and twelve scalar attributes.
///
/// Each attribute `attr_0` through `attr_11` contains its index as an [`i64`].
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_attribute_edit_fixture(path: &Path) {
    write_edit_fixture(path, "/g", 12, None);
    {
        let reference = File::open_rw(path).unwrap();
        let group = reference.group("g").unwrap();
        for index in 0..12i64 {
            group
                .new_attr::<i64>()
                .create(format!("attr_{index}").as_str())
                .unwrap()
                .write_scalar(&index)
                .unwrap();
        }
        drop(group);
        reference.close().unwrap();
    }
}

/// Checks the dense group's link count and scalar attributes after an edit through libhdf5.
///
/// Expects `attr_0` to contain 99 when `edit_attributes` is true. Each other attribute contains
/// its original index.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or a value differs from the expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_attribute_edit(path: &Path, edit_attributes: bool) {
    let reference = File::open(path).unwrap();
    let group = reference.group("g").unwrap();
    assert_eq!(group.info().unwrap().storage_type, GroupStorageType::Dense);
    assert_eq!(group.info().unwrap().nlinks, 13);
    assert_eq!(group.attr_names().unwrap().len(), 12);
    for index in 0..12i64 {
        assert_eq!(
            group
                .attr(&format!("attr_{index}"))
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            if edit_attributes && index == 0 {
                99
            } else {
                index
            }
        );
    }
    drop(group);
    reference.close().unwrap();
}

/// Writes a dense group and a second hard link to the group or its first dataset.
///
/// The root link `alias` targets `/g/link_0000` when `dataset` is true and `/g` otherwise.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_alias_edit_fixture(path: &Path, dataset: bool) {
    write_edit_fixture(path, "/g", 12, None);
    {
        let reference = File::open_rw(path).unwrap();
        reference
            .link_hard(if dataset { "/g/link_0000" } else { "/g" }, "alias")
            .unwrap();
        reference.close().unwrap();
    }
}

/// Writes nine root datasets, symbolic links, and a group with a UTF-8 name.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_symbolic_edit_fixture(path: &Path) {
    write_edit_fixture(path, "/", 9, None);
    {
        let reference = File::open_rw(path).unwrap();
        reference.link_soft("/absent_λ", "soft").unwrap();
        reference
            .link_external("absent_λ.h5", "/目标", "external")
            .unwrap();
        reference
            .create_group_builder()
            .char_encoding(CharEncoding::Utf8)
            .create("原始_λ")
            .unwrap();
        reference.close().unwrap();
    }
}

/// Checks compact root storage and the surviving symbolic and UTF-8 names through libhdf5.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or a value differs from the expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_compact_symbolic_edit(path: &Path) {
    let reference = File::open(path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Compact
    );
    let mut names = reference.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["external", "soft", "原始_λ"]);
    reference.close().unwrap();
}

/// Checks dense root storage and a link count of nine through libhdf5.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or a value differs from the expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_dense_symbolic_edit(path: &Path) {
    let reference = File::open(path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    assert_eq!(reference.info().unwrap().nlinks, 9);
    reference.close().unwrap();
}

/// Checks dense root storage and the surviving and added groups in the userblock fixture.
///
/// # Panics
///
/// Panics if libhdf5 cannot read or close the fixture, or a value differs from the expected value.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_userblock_edit(path: &Path) {
    let reference = File::open(path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    assert_eq!(reference.group("added").unwrap().info().unwrap().nlinks, 0);
    assert_eq!(
        reference
            .group("dense_untracked")
            .unwrap()
            .info()
            .unwrap()
            .nlinks,
        2000
    );
    reference.close().unwrap();
}
