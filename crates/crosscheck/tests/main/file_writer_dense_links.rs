#![cfg(feature = "hdf5")]

use hdf5::GroupStorageType;
use hdf5::datatype::ByteOrder as CByteOrder;
use hdf5_pure::AttrValue;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileSpaceSettings;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FixedPointType;
use hdf5_pure::LibVer;
use hdf5_pure::UserblockSize;
use hdf5_pure_format::__private::GROUP_HEAP_MAX_MANAGED_OBJECT;
use rstest::rstest;
use test_util_hdf5::dense_links;
use test_util_hdf5::dense_links::ChildObservation;
use test_util_hdf5::dense_links::Producer;

#[rstest]
#[case::below_threshold(MAX_COMPACT_LINKS - 1)]
#[case::at_threshold(MAX_COMPACT_LINKS)]
#[case::above_threshold(MAX_COMPACT_LINKS + 1)]
#[case::multiple_blocks(100)]
#[case::multiple_levels(2_000)]
fn libhdf5_and_file_writer_group_links_match(
    #[case] count: usize,
    #[values(0, 512)] userblock: u64,
    #[values(false, true)] streamed: bool,
    #[values(Producer::Libhdf5, Producer::Pure)] producer: Producer,
    #[values("/", "parent/group")] group_path: &str,
) {
    crosscheck(count, userblock, streamed, producer, group_path, |_| {});
}

#[cfg(feature = "__hdf5-1.10")]
#[rstest]
#[case::direct(FileSpaceStrategy::None, false)]
#[case::aggregate(FileSpaceStrategy::Aggr, false)]
#[case::fsm(FileSpaceStrategy::FsmAggr, false)]
#[case::persistent_fsm(FileSpaceStrategy::FsmAggr, true)]
#[case::page(FileSpaceStrategy::Page, false)]
#[case::persistent_page(FileSpaceStrategy::Page, true)]
fn libhdf5_reads_dense_links_under_file_space_strategies(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[values(0, 4_096)] userblock: u64,
    #[values(Producer::Libhdf5, Producer::Pure)] producer: Producer,
    #[values("/", "parent/group")] group_path: &str,
) {
    crosscheck(100, userblock, true, producer, group_path, |builder| {
        builder
            .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(persist));
    });
}

#[rstest]
#[case::libhdf5(Producer::Libhdf5)]
#[case::pure(Producer::Pure)]
fn libhdf5_reads_dense_link_at_managed_object_limit(#[case] producer: Producer) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("managed.h5");
    let name =
        "a".repeat(usize::try_from(GROUP_HEAP_MAX_MANAGED_OBJECT).unwrap() - HARD_LINK_OVERHEAD);
    dense_links::write_limit_fixture(&path, &name, producer);
    let observed = dense_links::observe_dataset(&path, &name);
    assert_eq!(observed.storage, GroupStorageType::Dense);
    assert_eq!(observed.values, [42]);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset(&name)
            .unwrap()
            .read_i32()
            .unwrap(),
        [42]
    );
}

fn crosscheck(
    count: usize,
    userblock: u64,
    streamed: bool,
    producer: Producer,
    group_path: &str,
    configure: impl FnOnce(&mut FileBuilder),
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("links.h5");
    dense_links::write_fixture(&path, count, userblock, streamed, producer, || {
        let mut builder = pure_fixture(count, userblock);
        configure(&mut builder);
        builder
    });
    let oracle = dense_links::group(&path, group_path);
    let observed = oracle.info();
    let pure = File::open(&path).unwrap();
    let root = group_path == "/";
    let link_count = count + usize::from(root);
    assert_eq!(observed.link_count, u64::try_from(link_count).unwrap());
    assert_eq!(
        observed.storage,
        if link_count > MAX_COMPACT_LINKS {
            GroupStorageType::Dense
        } else {
            GroupStorageType::Compact
        }
    );
    let mut expected = (0..count).map(dense_links::link_name).collect::<Vec<_>>();
    if root {
        expected.push("parent".into());
    }
    expected.sort();
    let mut names = observed.names;
    names.sort();
    assert_eq!(names, expected);
    for index in 0..count {
        let child = oracle.child(index);
        let name = dense_links::link_name(index);
        let path = format!("{group_path}/{name}");
        match child {
            ChildObservation::Dataset(values) => {
                assert_eq!(index % 3, 0);
                let expected = [i32::try_from(index).unwrap(), 17];
                assert_eq!(values, expected);
                assert_eq!(pure.dataset(&path).unwrap().read_i32().unwrap(), expected);
            }
            ChildObservation::Datatype { size, byte_order } => {
                assert_eq!(index % 3, 2);
                assert_eq!(size, 4);
                let byte_order = match byte_order {
                    CByteOrder::BigEndian => DatatypeByteOrder::BigEndian,
                    CByteOrder::LittleEndian => DatatypeByteOrder::LittleEndian,
                    other => panic!("expected an integer byte order, got {other:?}"),
                };
                assert_eq!(
                    pure.group(group_path)
                        .unwrap()
                        .named_datatype(&name)
                        .unwrap(),
                    Datatype::FixedPoint(FixedPointType::new(4, byte_order, true, 0, 32).unwrap())
                );
            }
            ChildObservation::Group { names, values } => {
                assert_eq!(index % 3, 1);
                assert_eq!(names, ["payload"]);
                assert_eq!(values, [23]);
                assert_eq!(
                    pure.dataset(&format!("{path}/payload"))
                        .unwrap()
                        .read_i32()
                        .unwrap(),
                    [23]
                );
            }
        }
    }
    assert_eq!(
        pure.group(group_path)
            .unwrap()
            .attrs()
            .unwrap()
            .get("attr_8"),
        Some(&AttrValue::I32(8))
    );
}

fn pure_fixture(count: usize, userblock: u64) -> FileBuilder {
    let mut builder = FileBuilder::new();
    if !cfg!(feature = "__hdf5-1.10") {
        builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    }
    if userblock != 0 {
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    }
    let mut group = builder.create_group("group");
    for index in 0..count {
        let name = dense_links::link_name(index);
        match index % 3 {
            0 => {
                let data = [i32::try_from(index).unwrap(), 17];
                let root_dataset = builder.create_dataset(&name).with_i32_data(&data);
                if index % 6 == 0 && cfg!(feature = "__hdf5-1.10") {
                    root_dataset.with_chunks(&[1]);
                }
                let dataset = group.create_dataset(&name).with_i32_data(&data);
                if index % 6 == 0 && cfg!(feature = "__hdf5-1.10") {
                    dataset.with_chunks(&[1]);
                }
            }
            1 => {
                let mut root_child = builder.create_group(&name);
                root_child.create_dataset("payload").with_i32_data(&[23]);
                builder.add_group(root_child.finish());
                let mut child = group.create_group(&name);
                child.create_dataset("payload").with_i32_data(&[23]);
                group.add_group(child.finish());
            }
            _ => {
                builder.commit_datatype(&name, dense_links::integer_type());
                group.commit_datatype(&name, dense_links::integer_type());
            }
        }
    }
    for index in 0..9 {
        let name = format!("attr_{index}");
        builder.set_attr(&name, AttrValue::I32(index));
        group.set_attr(&name, AttrValue::I32(index));
    }
    let mut parent = builder.create_group("parent");
    parent.add_group(group.finish());
    builder.add_group(parent.finish());
    builder
}

// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, is the default compact link limit.
const MAX_COMPACT_LINKS: usize = 8;
// Section `subsubsec_fmt4_dataobject_hdr_msg_link`, version 4.0, defines the version, flags,
// two-byte name length, and eight-byte target address of these untracked ASCII hard links.
const HARD_LINK_OVERHEAD: usize = 2 + 2 + 8;
