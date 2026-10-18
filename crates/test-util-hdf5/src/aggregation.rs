//! Workloads of object creations and deletions that the classic aggregation tests run through
//! hdf5-pure and through libhdf5.

use std::path::Path;

use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileCreateProperties;
use hdf5_pure::FileSpaceStrategy;

/// A sequence of creations and deletions in the root group of a new file.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    /// The name that identifies the workload in a snapshot name.
    pub name: &'static str,
    /// The steps, in the order they run.
    pub steps: &'static [Step],
}

impl Workload {
    /// Creates a file at `path` with [`create_file`] and applies each step to it.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be created or a step fails.
    pub fn write(self, path: &Path, strategy: FileSpaceStrategy, persist: bool) {
        let file = create_file(path, strategy, persist);
        for step in self.steps {
            step.apply(&file);
        }
    }

    /// Returns the objects that a step creates and no step deletes, in the order of their steps.
    pub fn surviving_objects(self) -> impl Iterator<Item = Object> {
        self.steps
            .iter()
            .filter_map(|step| match *step {
                Step::Create(object) => Some(object),
                Step::Delete(_) => None,
            })
            .filter(move |object| !self.steps.contains(&Step::Delete(object.link_name())))
    }
}

/// Creates a file at `path` under `strategy`, with a free-space section threshold of 1.
///
/// # Panics
///
/// Panics if the file cannot be created.
pub fn create_file(path: &Path, strategy: FileSpaceStrategy, persist: bool) -> File {
    File::create_with_options(
        path,
        FileCreateProperties::new().with_file_space_strategy(strategy, persist, 1),
        FileAccessProperties::default(),
    )
    .unwrap()
}

/// An operation on the root group of a file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Step {
    Create(Object),
    /// Deletes the link with this name.
    Delete(&'static str),
}

impl Step {
    /// Applies the step to the root group of `file`, then commits the file.
    ///
    /// # Panics
    ///
    /// Panics if the step or the commit fails.
    pub fn apply(self, file: &File) {
        let root = file.root();
        match self {
            Self::Create(Object::Dataset(name, len)) => {
                root.create_dataset(name, |builder| {
                    builder.with_i32_data(&vec![DATASET_VALUE; len]);
                })
                .unwrap();
            }
            Self::Create(Object::Group(name)) => {
                root.create_group(name).unwrap();
            }
            Self::Create(Object::Strings(name)) => {
                root.create_dataset(name, |builder| {
                    builder.with_vlen_strings(STRINGS);
                })
                .unwrap();
            }
            Self::Delete(name) => root.delete(name).unwrap(),
        }
        file.commit().unwrap();
    }
}

/// An object that a workload creates in the root group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Object {
    /// A one-dimensional dataset of this name and this many `i32` elements, each
    /// [`DATASET_VALUE`].
    Dataset(&'static str, usize),
    /// An empty group of this name.
    Group(&'static str),
    /// A one-dimensional dataset of this name that holds the variable-length strings [`STRINGS`].
    Strings(&'static str),
}

impl Object {
    /// Returns the name of the object's link in the root group.
    pub fn link_name(self) -> &'static str {
        match self {
            Self::Dataset(name, _) | Self::Group(name) | Self::Strings(name) => name,
        }
    }
}

/// Returns the address of the global heap collection that holds the first element of a
/// variable-length dataset.
///
/// `bytes` holds the whole file and `offset` is the address of the dataset's contiguous data. The
/// first element stores a 4-byte sequence length, then a global heap ID, whose first field is the
/// address of the collection. The global heap ID is defined in "Global Heap" of the [format
/// specification, version 4.0][spec]. The address is read as 8 bytes, the width in a file with
/// 8-byte offsets.
///
/// # Panics
///
/// Panics if `bytes` holds fewer than 12 bytes from `offset` on.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub fn collection_address(bytes: &[u8], offset: u64) -> u64 {
    let start = usize::try_from(offset).unwrap() + SEQUENCE_LENGTH_LEN;
    u64::from_le_bytes(*bytes[start..].first_chunk().unwrap())
}

/// Six group creations, so every allocation is metadata.
pub const METADATA_HEAVY: Workload = Workload {
    name: "metadata_heavy",
    steps: &[
        Step::Create(Object::Group("g0")),
        Step::Create(Object::Group("g1")),
        Step::Create(Object::Group("g2")),
        Step::Create(Object::Group("g3")),
        Step::Create(Object::Group("g4")),
        Step::Create(Object::Group("g5")),
    ],
};

/// Six creations of 16-element datasets, so every raw data allocation is smaller than a
/// small-data block.
pub const SMALL_RAW: Workload = Workload {
    name: "small_raw",
    steps: &[
        Step::Create(Object::Dataset("d0", SMALL_LEN)),
        Step::Create(Object::Dataset("d1", SMALL_LEN)),
        Step::Create(Object::Dataset("d2", SMALL_LEN)),
        Step::Create(Object::Dataset("d3", SMALL_LEN)),
        Step::Create(Object::Dataset("d4", SMALL_LEN)),
        Step::Create(Object::Dataset("d5", SMALL_LEN)),
    ],
};

/// Groups and 16-element datasets created in turn, so metadata and raw data allocations
/// alternate.
pub const ALTERNATING: Workload = Workload {
    name: "alternating",
    steps: &[
        Step::Create(Object::Group("g0")),
        Step::Create(Object::Dataset("d0", SMALL_LEN)),
        Step::Create(Object::Group("g1")),
        Step::Create(Object::Dataset("d1", SMALL_LEN)),
        Step::Create(Object::Group("g2")),
        Step::Create(Object::Dataset("d2", SMALL_LEN)),
    ],
};

/// Two 16-element datasets with a dataset of variable-length strings between them, then a group,
/// so a global heap collection is allocated between raw data allocations.
pub const GLOBAL_HEAP: Workload = Workload {
    name: "global_heap",
    steps: &[
        Step::Create(Object::Dataset("r0", SMALL_LEN)),
        Step::Create(Object::Strings("v")),
        Step::Create(Object::Dataset("r1", SMALL_LEN)),
        Step::Create(Object::Group("g")),
    ],
};

/// A 400-element dataset, a 16-element dataset, the deletion of the first, and a further
/// 16-element dataset that fits in the space the deletion frees.
pub const FREED_EXTENT: Workload = Workload {
    name: "freed_extent",
    steps: &[
        Step::Create(Object::Dataset("big", 400)),
        Step::Create(Object::Dataset("next", SMALL_LEN)),
        Step::Delete("big"),
        Step::Create(Object::Dataset("reuse", SMALL_LEN)),
    ],
};

/// A 1000-element dataset, larger than a small-data block, between two 16-element datasets, then
/// five 100-element datasets whose raw data runs past the end of a block.
pub const BLOCK_EDGES: Workload = Workload {
    name: "block_edges",
    steps: &[
        Step::Create(Object::Dataset("s0", SMALL_LEN)),
        Step::Create(Object::Dataset("huge", 1000)),
        Step::Create(Object::Dataset("s1", SMALL_LEN)),
        Step::Create(Object::Dataset("m0", 100)),
        Step::Create(Object::Dataset("m1", 100)),
        Step::Create(Object::Dataset("m2", 100)),
        Step::Create(Object::Dataset("m3", 100)),
        Step::Create(Object::Dataset("m4", 100)),
    ],
};

/// The value of every element of an [`Object::Dataset`].
pub const DATASET_VALUE: i32 = 1;

/// The elements of an [`Object::Strings`] dataset.
pub const STRINGS: &[&str] = &["alpha", "beta"];

const SMALL_LEN: usize = 16;

// `H5T__vlen_disk_write` in `src/H5Tvlen.c`, HDF5 2.2.0, writes a 4-byte sequence length before
// the global heap ID of a variable-length element.
const SEQUENCE_LENGTH_LEN: usize = 4;
