//! Dense group-link fixtures and observations through libhdf5.

use std::path::Path;

use hdf5::Group;
use hdf5::GroupStorageType;
use hdf5::datatype::ByteOrder as CByteOrder;
use hdf5::file::LibraryVersion;
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

// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, is the default compact link limit.
const MAX_COMPACT_LINKS: usize = 8;
