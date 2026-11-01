#![cfg(feature = "hdf5")]

use std::path::Path;

use hdf5::GroupStorageType;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::IndexType;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::IterationOrder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::LinkType;
use hdf5::plist::group_create::LinkCreationOrder;
use hdf5::plist::group_create::LinkPhaseChange as NativeLinkPhaseChange;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileCreateProperties;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileSpaceSettings;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileSpaceStrategy;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FormatError;
use hdf5_pure::GroupBuilder;
use hdf5_pure::LibVer;
use hdf5_pure::LinkPhaseChange;
use hdf5_pure::UserblockSize;
use rstest::rstest;
use test_util_hdf5::dense_links;

#[cfg(feature = "__hdf5-1.10")]
#[rstest]
#[case::default_empty(LinkPhaseChange::default(), 8, 6, 0)]
#[case::default_equal(LinkPhaseChange::default(), 8, 6, 8)]
#[case::default_dense(LinkPhaseChange::default(), 8, 6, 9)]
#[case::custom_equal(THRESHOLDS, 12, 8, 12)]
#[case::custom_dense(THRESHOLDS, 12, 8, 13)]
#[case::one_way_equal(ONE_WAY, 12, 0, 12)]
#[case::one_way_dense(ONE_WAY, 12, 0, 13)]
#[case::always_dense_empty(LinkPhaseChange::AlwaysDense, 0, 0, 0)]
#[case::always_dense_one(LinkPhaseChange::AlwaysDense, 0, 0, 1)]
#[case::small_dense(SMALL, 2, 1, 3)]
#[case::high_bytes(HIGH_BYTES, 513, 257, 13)]
fn created_link_storage_matches_the_configured_native_policy(
    #[case] policy: LinkPhaseChange,
    #[case] max_compact: u32,
    #[case] min_dense: u32,
    #[case] count: usize,
    #[values(Target::Root, Target::Nested)] target: Target,
    #[values(false, true)] streaming: bool,
    #[values(None, Some(FileSpaceSettings::new(FileSpaceStrategy::None)), Some(FileSpaceSettings::new(FileSpaceStrategy::Aggr)), Some(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true)), Some(FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true)))]
    settings: Option<FileSpaceSettings>,
    #[values(UserblockSize::NONE, UserblockSize::try_from(4096).unwrap())] userblock: UserblockSize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("created.h5");
    let reference_path = directory.path().join("reference.h5");
    let mut properties = FileCreateProperties::new().with_userblock(userblock);
    if let Some(settings) = settings {
        properties = properties.with_file_space_settings(settings);
    }
    write_builder(
        &path,
        configured_builder(policy, target, count, properties),
        streaming,
    );
    dense_links::write_edit_fixture(
        &reference_path,
        "group",
        count,
        Some((max_compact, min_dense)),
    );
    let reference = hdf5::File::open(&reference_path).unwrap();
    let reference_group = reference.group("group").unwrap();
    let expected_storage = reference_group.info().unwrap().storage_type;
    let expected_policy = reference_group
        .gcpl()
        .unwrap()
        .get_link_phase_change()
        .unwrap();
    drop(reference_group);
    reference.close().unwrap();
    check_created(
        &path,
        target,
        count,
        policy,
        expected_policy,
        expected_storage,
    );
    directory.close().unwrap();
}

#[rstest]
#[case::default(LinkPhaseChange::default(), 8, 6, 9)]
#[case::compact(THRESHOLDS, 12, 8, 12)]
#[case::dense(THRESHOLDS, 12, 8, 13)]
#[case::one_way(ONE_WAY, 12, 0, 13)]
#[case::always_dense(LinkPhaseChange::AlwaysDense, 0, 0, 1)]
#[case::small(SMALL, 2, 1, 3)]
#[case::high_bytes(HIGH_BYTES, 513, 257, 13)]
fn configured_policy_is_readable_under_v18_bounds(
    #[case] policy: LinkPhaseChange,
    #[case] max_compact: u32,
    #[case] min_dense: u32,
    #[case] count: usize,
    #[values(Target::Root, Target::Nested)] target: Target,
    #[values(false, true)] streaming: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("v18-policy.h5");
    let properties = FileCreateProperties::new()
        .with_userblock(UserblockSize::MIN)
        .with_libver_bounds(LibVer::Earliest, LibVer::V18);
    write_builder(
        &path,
        configured_builder(policy, target, count, properties),
        streaming,
    );
    check_created(
        &path,
        target,
        count,
        policy,
        NativeLinkPhaseChange {
            max_compact,
            min_dense,
        },
        if count > usize::try_from(max_compact).unwrap() {
            GroupStorageType::Dense
        } else {
            GroupStorageType::Compact
        },
    );
    directory.close().unwrap();
}

#[cfg(feature = "__hdf5-1.10")]
#[rstest]
#[case::thresholds(THRESHOLDS, GroupStorageType::Compact)]
#[case::one_way(ONE_WAY, GroupStorageType::Dense)]
#[case::always_dense(LinkPhaseChange::AlwaysDense, GroupStorageType::Dense)]
fn created_policy_survives_edit_boundaries_and_symbolic_links(
    #[case] policy: LinkPhaseChange,
    #[case] below_minimum: GroupStorageType,
    #[values(Target::Root, Target::Nested)] target: Target,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("edited-policy.h5");
    let mut properties =
        FileCreateProperties::new().with_userblock(UserblockSize::try_from(4096).unwrap());
    if paged {
        properties = properties.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
        );
    }
    let mut builder = FileBuilder::new();
    match target {
        Target::Root => {
            builder.with_create_properties(properties.with_root_group_link_phase_change(policy));
            for index in 0..13 {
                builder
                    .create_dataset(&link_name(index))
                    .with_i32_data(&[i32::try_from(index).unwrap(), 17]);
            }
        }
        Target::Nested => {
            builder.with_create_properties(properties);
            builder
                .create_group_with("parent", |parent| {
                    parent.create_group_with("group", |group| {
                        group.with_link_phase_change(policy);
                        for index in 0..13 {
                            group
                                .create_dataset(&link_name(index))
                                .with_i32_data(&[i32::try_from(index).unwrap(), 17]);
                        }
                        Ok(())
                    })
                })
                .unwrap();
        }
    }
    std::fs::write(&path, builder.finish().unwrap()).unwrap();
    let native = hdf5::File::open_rw(&path).unwrap();
    let group = native.group(target.path()).unwrap();
    group.link_soft(&link_name(0), "soft").unwrap();
    group.link_external("外部.h5", "/目标", "external").unwrap();
    drop(group);
    native.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    let group = file.group(target.path()).unwrap();
    for index in 6..13 {
        group.delete(&link_name(index)).unwrap();
    }
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    check_edited(&path, target, policy, 6, GroupStorageType::Dense);
    let file = File::open_rw(&path).unwrap();
    let group = file.group(target.path()).unwrap();
    group.delete(&link_name(5)).unwrap();
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    check_edited(&path, target, policy, 5, below_minimum);
    let file = File::open_rw(&path).unwrap();
    let group = file.group(target.path()).unwrap();
    for index in 5..10 {
        group
            .create_dataset(&link_name(index), |builder| {
                builder.with_i32_data(&[i32::try_from(index).unwrap(), 17]);
            })
            .unwrap();
    }
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    check_edited(&path, target, policy, 10, below_minimum);
    let file = File::open_rw(&path).unwrap();
    let group = file.group(target.path()).unwrap();
    group
        .create_dataset(&link_name(10), |builder| {
            builder.with_i32_data(&[10, 17]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    check_edited(&path, target, policy, 11, GroupStorageType::Dense);
    directory.close().unwrap();
}

fn configured_builder(
    policy: LinkPhaseChange,
    target: Target,
    count: usize,
    properties: FileCreateProperties,
) -> FileBuilder {
    let mut builder = FileBuilder::new();
    match target {
        Target::Root => {
            builder.with_create_properties(properties.with_root_group_link_phase_change(policy));
            populate_root(&mut builder, count);
        }
        Target::Nested => {
            builder.with_create_properties(properties);
            builder
                .create_group_with("parent", |parent| {
                    parent.create_group_with("group", |group| {
                        group.with_link_phase_change(policy);
                        populate_group(group, count);
                        Ok(())
                    })
                })
                .unwrap();
        }
    }
    builder
}

fn write_builder(path: &Path, builder: FileBuilder, streaming: bool) {
    if streaming {
        let mut output = std::fs::File::create(path).unwrap();
        builder.finish_to(&mut output).unwrap();
        drop(output);
    } else {
        std::fs::write(path, builder.finish().unwrap()).unwrap();
    }
}

fn check_created(
    path: &Path,
    target: Target,
    count: usize,
    policy: LinkPhaseChange,
    expected_policy: NativeLinkPhaseChange,
    expected_storage: GroupStorageType,
) {
    let native = hdf5::File::open(path).unwrap();
    let group = native.group(target.path()).unwrap();
    assert_eq!(
        group.gcpl().unwrap().get_link_phase_change().unwrap(),
        expected_policy
    );
    assert_eq!(
        group.gcpl().unwrap().get_link_creation_order().unwrap(),
        LinkCreationOrder::Untracked
    );
    assert_eq!(group.info().unwrap().storage_type, expected_storage);
    let mut names = group.member_names().unwrap();
    names.sort();
    assert_eq!(names, (0..count).map(link_name).collect::<Vec<_>>());
    let mut identities = Vec::with_capacity(count);
    for index in 0..count {
        let name = link_name(index);
        let identity = group.loc_info_by_name(&name).unwrap().token;
        assert!(
            !identities.contains(&identity),
            "duplicate object identity for {name:?}"
        );
        identities.push(identity);
        match index % 3 {
            0 => assert_eq!(
                group.dataset(&name).unwrap().read_raw::<i32>().unwrap(),
                [i32::try_from(index).unwrap(), 17]
            ),
            1 => {
                let child = group.group(&name).unwrap();
                assert_eq!(child.member_names().unwrap(), ["payload"]);
                assert_eq!(
                    child.dataset("payload").unwrap().read_raw::<i32>().unwrap(),
                    [23]
                );
                assert_eq!(
                    child.gcpl().unwrap().get_link_phase_change().unwrap(),
                    NativeLinkPhaseChange::default()
                );
            }
            _ => assert_eq!(
                group
                    .committed_datatype(&name)
                    .unwrap()
                    .as_datatype()
                    .size(),
                4
            ),
        }
    }
    if count > usize::try_from(expected_policy.max_compact).unwrap() {
        assert_eq!(expected_storage, GroupStorageType::Dense);
    }
    drop(group);
    native.close().unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(
        file.group(target.path())
            .unwrap()
            .link_phase_change()
            .unwrap(),
        policy
    );
    assert_eq!(
        file.superblock().eof_address,
        std::fs::metadata(path).unwrap().len()
    );
    file.close().unwrap();
}

fn populate_root(builder: &mut FileBuilder, count: usize) {
    for index in 0..count {
        let name = link_name(index);
        match index % 3 {
            0 => {
                builder
                    .create_dataset(&name)
                    .with_i32_data(&[i32::try_from(index).unwrap(), 17]);
            }
            1 => {
                builder
                    .create_group_with(&name, |child| {
                        child.create_dataset("payload").with_i32_data(&[23]);
                        Ok(())
                    })
                    .unwrap();
            }
            _ => builder.commit_datatype(&name, dense_links::integer_type()),
        }
    }
}

fn populate_group(builder: &mut GroupBuilder, count: usize) {
    for index in 0..count {
        let name = link_name(index);
        match index % 3 {
            0 => {
                builder
                    .create_dataset(&name)
                    .with_i32_data(&[i32::try_from(index).unwrap(), 17]);
            }
            1 => {
                builder
                    .create_group_with(&name, |child| {
                        child.create_dataset("payload").with_i32_data(&[23]);
                        Ok(())
                    })
                    .unwrap();
            }
            _ => builder.commit_datatype(&name, dense_links::integer_type()),
        }
    }
}

fn link_name(index: usize) -> String {
    format!("entry_{index:02}_λ")
}

#[derive(Clone, Copy, Debug)]
enum Target {
    Nested,
    Root,
}

impl Target {
    fn path(self) -> &'static str {
        match self {
            Self::Root => "/",
            Self::Nested => "parent/group",
        }
    }
}

#[cfg(feature = "__hdf5-1.10")]
fn check_edited(
    path: &Path,
    target: Target,
    policy: LinkPhaseChange,
    hard_count: usize,
    expected_storage: GroupStorageType,
) {
    let native = hdf5::File::open(path).unwrap();
    let group = native.group(target.path()).unwrap();
    assert_eq!(group.info().unwrap().storage_type, expected_storage);
    let properties = group.gcpl().unwrap().get_link_phase_change().unwrap();
    assert_eq!(
        (properties.max_compact, properties.min_dense),
        (
            u32::from(policy.max_compact()),
            u32::from(policy.min_dense())
        )
    );
    let mut names = group.member_names().unwrap();
    names.sort();
    let mut expected_names = (0..hard_count).map(link_name).collect::<Vec<_>>();
    expected_names.extend(["external".into(), "soft".into()]);
    expected_names.sort();
    assert_eq!(names, expected_names);
    for index in 0..hard_count {
        assert_eq!(
            group
                .dataset(&link_name(index))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [i32::try_from(index).unwrap(), 17]
        );
    }
    assert_eq!(
        group.dataset("soft").unwrap().read_raw::<i32>().unwrap(),
        [0, 17]
    );
    let links = group
        .links(IndexType::Name, IterationOrder::Increasing)
        .unwrap();
    assert_eq!(
        links
            .iter()
            .find(|(name, _)| name == "soft")
            .unwrap()
            .1
            .link_type,
        LinkType::Soft
    );
    assert_eq!(
        links
            .iter()
            .find(|(name, _)| name == "external")
            .unwrap()
            .1
            .link_type,
        LinkType::External
    );
    drop(group);
    native.close().unwrap();
    let file = File::open(path).unwrap();
    let group = file.group(target.path()).unwrap();
    assert_eq!(group.link_phase_change().unwrap(), policy);
    assert_eq!(group.dataset("soft").unwrap().read_i32().unwrap(), [0, 17]);
    let error = group.dataset("external").unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::ExternalLinkUnsupported {
            path: if matches!(target, Target::Root) {
                "external".into()
            } else {
                "parent/group/external".into()
            },
            filename: "外部.h5".into(),
            object_path: "/目标".into(),
        }
    );
    drop(group);
    file.close().unwrap();
}

const THRESHOLDS: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<12>()
    .min_dense::<8>()
    .build();
const ONE_WAY: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<12>()
    .disable_compact_reversion()
    .build();
const SMALL: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<2>()
    .min_dense::<1>()
    .build();
const HIGH_BYTES: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<513>()
    .min_dense::<257>()
    .build();
