#![cfg(feature = "__hdf5-1.10")]

use std::collections::BTreeSet;
use std::path::Path;

use hdf5::H5Type;
use hdf5::Location;
use hdf5::file::LibraryVersion;
use hdf5::plist::GroupCreate;
use hdf5::plist::dataset_create::Layout;
use hdf5::types::VarLenArray;
use hdf5::types::VarLenUnicode;
use hdf5_pure::File;
use hdf5_pure::GlobalHeapLiveness;
use rstest::rstest;
use tempfile::TempDir;
use test_util::global_heap;
use test_util_hdf5::space_ownership;
use test_util_hdf5::space_ownership::SharedStorage;

#[rstest]
fn libhdf5_collections_are_each_referenced(
    #[values(LibraryVersion::Earliest, LibraryVersion::V18)] low: LibraryVersion,
    #[values(
        Form::CompactAttribute,
        Form::DenseAttribute,
        Form::ContiguousDataset,
        Form::CompactDataset,
        Form::FillValue
    )]
    form: Form,
    #[values(
        Payload::Strings,
        Payload::Sequences,
        Payload::Compounds,
        Payload::ArraysOfStrings,
        Payload::ArraysOfCompounds
    )]
    payload: Payload,
) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("liveness.h5");
    let file = libhdf5_file(&path, low);
    let group = file.create_group("g").unwrap();
    match payload {
        Payload::Strings => form.write(&group, &strings()),
        Payload::Sequences => form.write(&group, &sequences()),
        Payload::Compounds => form.write(&group, &records()),
        Payload::ArraysOfStrings => form.write(&group, &[strings(), strings()]),
        Payload::ArraysOfCompounds => {
            let [first, second, third] = records();
            form.write(&group, &[[first, second], [third.clone(), third]]);
        }
    }
    drop(group);
    file.close().unwrap();

    assert_collections_are_referenced(&path);
}

#[rstest]
#[case::only_through_a_second_hard_link(Linking::SecondHardLinkOnly)]
#[case::through_two_paths(Linking::TwoHardLinks)]
#[case::beside_a_soft_link(Linking::SoftLink)]
fn a_linked_dataset_refers_to_its_collection(#[case] linking: Linking) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("liveness_links.h5");
    let file = libhdf5_file(&path, LibraryVersion::Earliest);
    let first = file.create_group("a").unwrap();
    Form::ContiguousDataset.write(&first, &["alpha".parse::<VarLenUnicode>().unwrap()]);
    file.create_group("b").unwrap();
    match linking {
        Linking::SecondHardLinkOnly => {
            file.link_hard("/a/d", "/b/d").unwrap();
            first.unlink("d").unwrap();
        }
        Linking::SoftLink => file.link_soft("/a/d", "/b/d").unwrap(),
        Linking::TwoHardLinks => file.link_hard("/a/d", "/b/d").unwrap(),
    }
    drop(first);
    file.close().unwrap();

    assert_collections_are_referenced(&path);
}

#[rstest]
#[case::shared_chunked_dataspace(SharedStorage::Dataspace, false)]
#[case::compact_shared_attribute(SharedStorage::Attribute, false)]
#[case::dense_shared_attribute(SharedStorage::Attribute, true)]
#[case::compact_shared_attribute_fields(SharedStorage::AttributeFields, false)]
#[case::dense_shared_attribute_fields(SharedStorage::AttributeFields, true)]
fn shared_messages_are_incomplete(#[case] storage: SharedStorage, #[case] dense: bool) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("liveness_shared.h5");
    space_ownership::libhdf5_write_shared_storage(&path, storage, dense);
    let file = File::open_rw(&path).unwrap();

    assert_eq!(
        file.__global_heap_liveness().unwrap(),
        GlobalHeapLiveness {
            referenced_collections: BTreeSet::new(),
            analysis_complete: false,
        }
    );
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Form {
    CompactAttribute,
    CompactDataset,
    ContiguousDataset,
    DenseAttribute,
    FillValue,
}

impl Form {
    fn write<T: H5Type + Clone>(self, group: &hdf5::Group, values: &[T]) {
        match self {
            Self::CompactAttribute => write_attribute(group, values),
            Self::DenseAttribute => {
                let properties = GroupCreate::build()
                    .attr_phase_change(0, 0)
                    .finish()
                    .unwrap();
                let dense = group
                    .create_group_builder()
                    .set_gcpl(&properties)
                    .create("dense")
                    .unwrap();
                write_attribute(&dense, values);
            }
            Self::CompactDataset | Self::ContiguousDataset => {
                let layout = match self {
                    Self::CompactDataset => Layout::Compact,
                    _ => Layout::Contiguous,
                };
                group
                    .new_dataset::<T>()
                    .shape([values.len()])
                    .layout(layout)
                    .create("d")
                    .unwrap()
                    .write(values)
                    .unwrap();
            }
            Self::FillValue => {
                group
                    .new_dataset::<T>()
                    .shape([values.len()])
                    .fill_value(values[0].clone())
                    .create("d")
                    .unwrap();
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Linking {
    SecondHardLinkOnly,
    SoftLink,
    TwoHardLinks,
}

#[derive(Clone, Copy, Debug)]
enum Payload {
    ArraysOfCompounds,
    ArraysOfStrings,
    Compounds,
    Sequences,
    Strings,
}

#[repr(C)]
#[derive(Clone, Debug, hdf5::H5Type)]
struct Record {
    n: i32,
    s: VarLenUnicode,
    v: VarLenArray<i32>,
}

fn strings() -> [VarLenUnicode; 3] {
    ["alpha", "", "beta"].map(|s| s.parse().unwrap())
}

fn sequences() -> [VarLenArray<i32>; 3] {
    [&[1, 2, 3][..], &[], &[4]].map(VarLenArray::from_slice)
}

fn records() -> [Record; 3] {
    let [s0, s1, s2] = strings();
    let [v0, v1, v2] = sequences();
    [
        Record { n: 1, s: s0, v: v0 },
        Record { n: 2, s: s1, v: v1 },
        Record { n: 3, s: s2, v: v2 },
    ]
}

fn write_attribute<T: H5Type>(location: &Location, values: &[T]) {
    location
        .new_attr::<T>()
        .shape([values.len()])
        .create("a")
        .unwrap()
        .write(values)
        .unwrap();
}

fn libhdf5_file(path: &Path, low: LibraryVersion) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(low, LibraryVersion::latest()))
        .create(path)
        .unwrap()
}

/// Asserts that the analysis of the file at `path` is complete and lists the offset of every
/// collection signature in the file.
fn assert_collections_are_referenced(path: &Path) {
    let expected = global_heap::collection_starts(&std::fs::read(path).unwrap());
    assert!(!expected.is_empty(), "libhdf5 placed no collection");
    let file = File::open_rw(path).unwrap();

    assert_eq!(
        file.__global_heap_liveness().unwrap(),
        GlobalHeapLiveness {
            referenced_collections: expected.into_iter().collect(),
            analysis_complete: true,
        }
    );
    file.close().unwrap();
}
