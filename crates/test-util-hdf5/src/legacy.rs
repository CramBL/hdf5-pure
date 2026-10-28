//! Legacy fixture observations through pure and native readers.
//!
//! Independent construction lives in [`test_util::legacy`]. The assertion helpers
//! check the fixture's expected hierarchy and data separately from symbol-table
//! storage.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util::legacy::Fixture;
//! use test_util_hdf5::legacy;
//!
//! let fixture = Fixture::TwoGroups;
//! let bytes = fixture.build();
//! let file = File::from_bytes(bytes.clone())?;
//! legacy::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
//! legacy::assert_symbol_tables(&file, &bytes, fixture);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

#[cfg(feature = "hdf5")]
use std::fmt::Debug;
#[cfg(feature = "hdf5")]
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::GroupStorageType;
#[cfg(feature = "hdf5")]
use hdf5::H5Type;
#[cfg(feature = "hdf5")]
use hdf5::dataset::Layout as NativeLayout;
#[cfg(feature = "hdf5")]
use hdf5::datatype::ByteOrder;

use hdf5_pure::{
    Datatype, DatatypeByteOrder, File, FixedPointType, FloatingPointType, Layout, SuperblockVersion,
};
use test_util::legacy::Fixture;
use test_util::object_header::{MessageType, v1};
use test_util::widths::Widths;

use crate::symbol_table;

/// Checks the hierarchy, dataset metadata, byte order, and values of a legacy fixture.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, fixture: Fixture, byte_order: DatatypeByteOrder) {
    for GroupContents {
        path,
        groups,
        datasets,
    } in groups(fixture)
    {
        let group = file.group(path).unwrap();
        let mut actual_groups = group.groups().unwrap();
        actual_groups.sort();
        let mut actual_datasets = group.datasets().unwrap();
        actual_datasets.sort();
        assert_eq!(actual_groups, *groups, "groups at {path:?}");
        assert_eq!(actual_datasets, *datasets, "datasets at {path:?}");
        assert!(group.attrs().unwrap().is_empty(), "attributes at {path:?}");
        assert!(group.named_datatypes().unwrap().is_empty());
    }
    match fixture {
        Fixture::NestedGroups => {
            assert_f64(file, "a/b/c/deep", &[42.0], byte_order);
        }
        Fixture::SimpleDataset => {
            assert_f64(file, "data", &[1.0, 2.0, 3.0], byte_order);
        }
        Fixture::TwoGroups => {
            let values = file.dataset("group1/values").unwrap();
            assert_eq!(values.shape().unwrap(), [3]);
            assert_eq!(
                values.datatype().unwrap(),
                Datatype::FixedPoint(FixedPointType::new(4, byte_order, true, 0, 32).unwrap())
            );
            assert_eq!(values.read_i32().unwrap(), [10, 20, 30]);
            assert!(values.attrs().unwrap().is_empty());
            assert_contiguous(&values.layout().unwrap(), 12);
            let temps = file.dataset("group2/temps").unwrap();
            assert_eq!(temps.shape().unwrap(), [2]);
            assert_eq!(
                temps.datatype().unwrap(),
                Datatype::FloatingPoint(FloatingPointType::ieee754_binary32(byte_order))
            );
            assert_eq!(temps.read_f32().unwrap(), [98.6, 37.0]);
            assert!(temps.attrs().unwrap().is_empty());
            assert_contiguous(&temps.layout().unwrap(), 8);
        }
    }
}

/// Checks the version 0 superblock and symbol-table storage of every fixture group.
///
/// Pass the bytes from which `file` was opened.
///
/// # Panics
///
/// Panics if metadata cannot be read, or the superblock, symbol tables, or group
/// traversal differ from the fixture.
pub fn assert_symbol_tables(file: &File, bytes: &[u8], fixture: Fixture) {
    assert_eq!(file.superblock().version(), SuperblockVersion::Zero);
    let tables = symbol_table::Fixture::new(file, bytes);
    let mut pending = vec![(String::new(), file.superblock().root_group_address)];
    let mut observed = Vec::new();
    while let Some((path, address)) = pending.pop() {
        let chunks = v1::chunks(
            bytes,
            usize::try_from(address.get()).unwrap(),
            Widths::EIGHT,
        );
        let messages: Vec<_> = chunks
            .iter()
            .flat_map(|chunk| &chunk.records)
            .map(|record| record.msg_type)
            .collect();
        assert_eq!(
            messages
                .iter()
                .filter(|kind| **kind == MessageType::SYMBOL_TABLE)
                .count(),
            1
        );
        let links = tables.links(address);
        let group = file.group(&path).unwrap();
        let mut expected = group.groups().unwrap();
        let child_groups = expected.clone();
        expected.extend(group.datasets().unwrap());
        expected.sort();
        assert_eq!(
            links
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>(),
            expected
        );
        for (name, address) in links {
            if child_groups.contains(&name) {
                pending.push((
                    if path.is_empty() {
                        name
                    } else {
                        format!("{path}/{name}")
                    },
                    address,
                ));
            }
        }
        observed.push(path);
    }
    observed.sort();
    assert_eq!(
        observed,
        groups(fixture)
            .iter()
            .map(|group| group.path)
            .collect::<Vec<_>>()
    );
}

#[cfg(feature = "hdf5")]
/// Writes a native legacy fixture and closes the file.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn write_libhdf5(path: &Path, fixture: Fixture) {
    let file = crate::file::libhdf5_create_earliest(path);
    match fixture {
        Fixture::NestedGroups => {
            let a = file.create_group("a").unwrap();
            let b = a.create_group("b").unwrap();
            let c = b.create_group("c").unwrap();
            c.new_dataset_builder()
                .with_data(&[42.0f64])
                .create("deep")
                .unwrap();
        }
        Fixture::SimpleDataset => {
            file.new_dataset_builder()
                .with_data(&[1.0f64, 2.0, 3.0])
                .create("data")
                .unwrap();
        }
        Fixture::TwoGroups => {
            file.create_group("group1")
                .unwrap()
                .new_dataset_builder()
                .with_data(&[10i32, 20, 30])
                .create("values")
                .unwrap();
            file.create_group("group2")
                .unwrap()
                .new_dataset_builder()
                .with_data(&[98.6f32, 37.0])
                .create("temps")
                .unwrap();
        }
    }
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
/// Checks native symbol-table storage, dataset metadata, byte order, and values.
///
/// # Panics
///
/// Panics if the file cannot be opened, native metadata or values cannot be read,
/// any observation differs from the fixture, or close fails.
pub fn assert_libhdf5_contents(path: &Path, fixture: Fixture, byte_order: ByteOrder) {
    let file = crate::file::libhdf5_open(path);
    for GroupContents {
        path,
        groups,
        datasets,
    } in groups(fixture)
    {
        let group = file
            .group(if path.is_empty() { "/" } else { path })
            .unwrap();
        assert_eq!(
            group.info().unwrap().storage_type,
            GroupStorageType::SymbolTable
        );
        let mut expected: Vec<_> = groups.iter().chain(*datasets).copied().collect();
        expected.sort();
        let mut actual = group.member_names().unwrap();
        actual.sort();
        assert_eq!(actual, expected);
        assert!(group.attr_names().unwrap().is_empty());
    }
    match fixture {
        Fixture::NestedGroups => assert_native_dataset(&file, "a/b/c/deep", &[42.0f64], byte_order),
        Fixture::SimpleDataset => {
            assert_native_dataset(&file, "data", &[1.0f64, 2.0, 3.0], byte_order)
        }
        Fixture::TwoGroups => {
            assert_native_dataset(&file, "group1/values", &[10i32, 20, 30], byte_order);
            assert_native_dataset(&file, "group2/temps", &[98.6f32, 37.0], byte_order);
        }
    }
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
fn assert_native_dataset<T: H5Type + Debug + PartialEq>(
    file: &hdf5::File,
    path: &str,
    expected: &[T],
    byte_order: ByteOrder,
) {
    let dataset = file.dataset(path).unwrap();
    assert_eq!(dataset.shape(), [expected.len()]);
    let datatype = dataset.dtype().unwrap();
    assert_eq!(datatype.to_descriptor().unwrap(), T::type_descriptor());
    assert_eq!(datatype.byte_order(), byte_order);
    assert_eq!(dataset.layout(), NativeLayout::Contiguous);
    assert_eq!(dataset.read_raw::<T>().unwrap(), expected);
    assert!(dataset.attr_names().unwrap().is_empty());
}

fn assert_f64(file: &File, path: &str, expected: &[f64], byte_order: DatatypeByteOrder) {
    let dataset = file.dataset(path).unwrap();
    assert_eq!(dataset.shape().unwrap(), [expected.len() as u64]);
    assert_eq!(
        dataset.datatype().unwrap(),
        Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(byte_order))
    );
    assert_eq!(dataset.read_f64().unwrap(), expected);
    assert!(dataset.attrs().unwrap().is_empty());
    assert_contiguous(&dataset.layout().unwrap(), expected.len() as u64 * 8);
}

fn assert_contiguous(layout: &Layout, expected: u64) {
    let Layout::Contiguous {
        address: Some(_),
        size,
    } = layout
    else {
        panic!("expected allocated contiguous storage, got {layout:?}");
    };
    assert_eq!(*size, expected);
}

struct GroupContents {
    path: &'static str,
    groups: &'static [&'static str],
    datasets: &'static [&'static str],
}

fn groups(fixture: Fixture) -> &'static [GroupContents] {
    match fixture {
        Fixture::NestedGroups => &[
            GroupContents {
                path: "",
                groups: &["a"],
                datasets: &[],
            },
            GroupContents {
                path: "a",
                groups: &["b"],
                datasets: &[],
            },
            GroupContents {
                path: "a/b",
                groups: &["c"],
                datasets: &[],
            },
            GroupContents {
                path: "a/b/c",
                groups: &[],
                datasets: &["deep"],
            },
        ],
        Fixture::SimpleDataset => &[GroupContents {
            path: "",
            groups: &[],
            datasets: &["data"],
        }],
        Fixture::TwoGroups => &[
            GroupContents {
                path: "",
                groups: &["group1", "group2"],
                datasets: &[],
            },
            GroupContents {
                path: "group1",
                groups: &[],
                datasets: &["values"],
            },
            GroupContents {
                path: "group2",
                groups: &[],
                datasets: &["temps"],
            },
        ],
    }
}
