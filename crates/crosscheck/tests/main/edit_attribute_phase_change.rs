#![cfg(feature = "__hdf5-1.10")]

use std::collections::HashMap;
use std::path::Path;

use hdf5::plist::file_access::LibraryVersion;
use hdf5::plist::group_create::{AttrCreationOrder, GroupCreateBuilder};
use hdf5_pure::{AttrValue, File, FileAccessProperties, MemoryStrategy, MessageType};
use hdf5_pure_format::__private::{
    AccessMode, AttributeInfoMessage, ObjectHeader, ObjectHeaderPrefix,
};
use rstest::rstest;

use test_util_hdf5::storage;

#[rstest]
#[case::default_at_max(8, 6, 7, false, false)]
#[case::default_above_max(8, 6, 8, true, false)]
#[case::custom_below_max(32, 24, 8, false, false)]
#[case::custom_at_max(32, 24, 31, false, false)]
#[case::custom_above_max(32, 24, 32, true, false)]
#[case::one_way_at_max(12, 0, 11, false, false)]
#[case::one_way_above_max(12, 0, 12, true, false)]
#[case::always_dense(0, 0, 0, true, false)]
#[case::custom_overwrite_at_max(32, 24, 32, false, true)]
fn compact_attribute_edits_use_stored_policy(
    #[case] maximum: u16,
    #[case] minimum: u16,
    #[case] initial: usize,
    #[case] dense: bool,
    #[case] overwrite: bool,
    #[values(Owner::Root, Owner::Group, Owner::Dataset)] owner: Owner,
    #[values(
        AttrCreationOrder::Untracked,
        AttrCreationOrder::Tracked,
        AttrCreationOrder::Indexed
    )]
    order: AttrCreationOrder,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("policy.h5");
    create_fixture(&path, owner, order, maximum, minimum, initial);
    let before = inspect_storage(&path, owner, initial > usize::from(maximum), order, initial);
    let mut expected_values: HashMap<String, AttrValue> = (0..initial)
        .map(|index| {
            (
                attr_name(index),
                AttrValue::I64(i64::try_from(index).unwrap()),
            )
        })
        .collect();
    assert_eq!(
        owner.attributes(&File::open(&path).unwrap()),
        expected_values
    );
    let changed = if overwrite { initial - 1 } else { initial };
    let count = initial + usize::from(!overwrite);
    {
        let file = File::open_rw(&path).unwrap();
        owner.set_attribute(&file, &attr_name(changed), AttrValue::I64(71));
        file.commit().unwrap();
    }
    assert_eq!(inspect_storage(&path, owner, dense, order, count), before);
    expected_values.insert(attr_name(changed), AttrValue::I64(71));
    assert_eq!(
        owner.attributes(&File::open(&path).unwrap()),
        expected_values
    );
    let file = hdf5::File::open(&path).unwrap();
    let location = owner.location(&file);
    let expected: Vec<String> = (0..count).map(attr_name).collect();
    assert_eq!(location.attr_names().unwrap(), expected);
    for index in 0..count {
        if !matches!(order, AttrCreationOrder::Untracked) {
            assert_eq!(
                location
                    .attr_info(&attr_name(index))
                    .unwrap()
                    .creation_order,
                Some(u32::try_from(index).unwrap())
            );
            assert_eq!(
                location
                    .attr_by_index(
                        hdf5::IndexType::CreationOrder,
                        hdf5::IterationOrder::Increasing,
                        u64::try_from(index).unwrap()
                    )
                    .unwrap()
                    .name(),
                attr_name(index)
            );
        }
        assert_eq!(
            location
                .attr(&attr_name(index))
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            if index == changed {
                71
            } else {
                i64::try_from(index).unwrap()
            }
        );
    }
    let pair = match owner {
        Owner::Root | Owner::Group => file
            .group(owner.path())
            .unwrap()
            .gcpl()
            .unwrap()
            .get_attr_phase_change()
            .unwrap(),
        Owner::Dataset => file
            .dataset(owner.path())
            .unwrap()
            .dcpl()
            .unwrap()
            .get_attr_phase_change()
            .unwrap(),
    };
    assert_eq!(
        (pair.max_compact, pair.min_dense),
        (u32::from(maximum), u32::from(minimum))
    );
    drop(location);
    file.close().unwrap();
}

#[rstest]
fn oversized_attributes_override_compact_policy(
    #[values(Owner::Root, Owner::Group, Owner::Dataset)] owner: Owner,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("oversized.h5");
    create_fixture(&path, owner, AttrCreationOrder::Untracked, 32, 24, 1);
    let before = inspect_storage(&path, owner, false, AttrCreationOrder::Untracked, 1);
    let value: Vec<i64> = (0..10_000).collect();
    {
        let file = File::open_rw(&path).unwrap();
        owner.set_attribute(&file, "large", AttrValue::I64Array(value.clone()));
        file.commit().unwrap();
    }
    assert_eq!(
        inspect_storage(&path, owner, true, AttrCreationOrder::Untracked, 2),
        before
    );
    assert_eq!(
        owner.attributes(&File::open(&path).unwrap()),
        HashMap::from([
            ("a000".to_string(), AttrValue::I64(0)),
            ("large".to_string(), AttrValue::I64Array(value.clone())),
        ])
    );
    let file = hdf5::File::open(&path).unwrap();
    let location = owner.location(&file);
    assert_eq!(
        location.attr("a000").unwrap().read_scalar::<i64>().unwrap(),
        0
    );
    assert_eq!(
        location.attr("large").unwrap().read_raw::<i64>().unwrap(),
        value
    );
    drop(location);
    file.close().unwrap();
}

#[rstest]
#[case::custom_at_minimum(32, 24, 33, 24, true, false)]
#[case::custom_below_minimum(32, 24, 33, 23, false, false)]
#[case::default_at_minimum(8, 6, 9, 6, true, false)]
#[case::default_below_minimum(8, 6, 9, 5, false, false)]
#[case::one_way(12, 0, 13, 1, true, false)]
#[case::always_dense(0, 0, 3, 1, true, false)]
#[case::empty_default(8, 6, 9, 0, false, false)]
#[case::empty_custom(32, 24, 33, 0, false, false)]
#[case::reversion_with_new_vl_value(8, 6, 9, 5, false, true)]
fn dense_attribute_deletions_use_stored_policy(
    #[case] maximum: u16,
    #[case] minimum: u16,
    #[case] initial: usize,
    #[case] remaining: usize,
    #[case] dense: bool,
    #[case] replace_vl: bool,
    #[values(Owner::Root, Owner::Group, Owner::Dataset)] owner: Owner,
    #[values(
        AttrCreationOrder::Untracked,
        AttrCreationOrder::Tracked,
        AttrCreationOrder::Indexed
    )]
    order: AttrCreationOrder,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reversion.h5");
    create_fixture(&path, owner, order, maximum, minimum, initial);
    let policy = inspect_storage(&path, owner, true, order, initial);
    let initial_values: HashMap<_, _> = (0..initial)
        .map(|index| {
            (
                attr_name(index),
                AttrValue::I64(i64::try_from(index).unwrap()),
            )
        })
        .collect();
    assert_eq!(
        owner.attributes(&File::open(&path).unwrap()),
        initial_values
    );
    let mut removed: Vec<_> = (0..initial)
        .step_by(2)
        .chain((1..initial).step_by(2))
        .take(initial - remaining)
        .collect();
    removed.sort_unstable();
    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(memory),
        )
        .unwrap();
        for &index in &removed {
            owner.remove_attribute(&file, &attr_name(index));
        }
        if replace_vl {
            owner.set_attribute(
                &file,
                &attr_name(initial - 1),
                AttrValue::VarLenString("new variable-length value".into()),
            );
        }
        file.commit().unwrap();
    }
    assert_eq!(
        inspect_storage(
            &path,
            owner,
            dense,
            order,
            if remaining == 0 { 0 } else { initial }
        ),
        policy
    );
    let surviving: Vec<_> = (0..initial)
        .filter(|index| removed.binary_search(index).is_err())
        .collect();
    let mut expected_values: HashMap<_, _> = surviving
        .iter()
        .map(|&index| {
            (
                attr_name(index),
                AttrValue::I64(i64::try_from(index).unwrap()),
            )
        })
        .collect();
    if replace_vl {
        expected_values.insert(
            attr_name(initial - 1),
            AttrValue::VarLenString("new variable-length value".into()),
        );
    }
    assert_eq!(
        owner.attributes(&File::open(&path).unwrap()),
        expected_values
    );
    let file = hdf5::File::open(&path).unwrap();
    let location = owner.location(&file);
    let names: Vec<_> = surviving.iter().copied().map(attr_name).collect();
    assert_eq!(location.attr_names().unwrap(), names);
    for (position, &index) in surviving.iter().enumerate() {
        if replace_vl && index == initial - 1 {
            assert_eq!(
                location
                    .attr(&attr_name(index))
                    .unwrap()
                    .read_scalar::<hdf5::types::VarLenUnicode>()
                    .unwrap()
                    .as_str(),
                "new variable-length value"
            );
        } else {
            assert_eq!(
                location
                    .attr(&attr_name(index))
                    .unwrap()
                    .read_scalar::<i64>()
                    .unwrap(),
                i64::try_from(index).unwrap()
            );
        }
        if !matches!(order, AttrCreationOrder::Untracked) {
            assert_eq!(
                location
                    .attr_info(&attr_name(index))
                    .unwrap()
                    .creation_order,
                Some(u32::try_from(index).unwrap())
            );
            assert_eq!(
                location
                    .attr_by_index(
                        hdf5::IndexType::CreationOrder,
                        hdf5::IterationOrder::Increasing,
                        u64::try_from(position).unwrap()
                    )
                    .unwrap()
                    .name(),
                attr_name(index)
            );
        }
    }
    let pair = match owner {
        Owner::Root | Owner::Group => file
            .group(owner.path())
            .unwrap()
            .gcpl()
            .unwrap()
            .get_attr_phase_change()
            .unwrap(),
        Owner::Dataset => file
            .dataset(owner.path())
            .unwrap()
            .dcpl()
            .unwrap()
            .get_attr_phase_change()
            .unwrap(),
    };
    assert_eq!(
        (pair.max_compact, pair.min_dense),
        (u32::from(maximum), u32::from(minimum))
    );
    if matches!(owner, Owner::Dataset) {
        assert_eq!(file.dataset("d").unwrap().read_raw::<i32>().unwrap(), [19]);
    }
    drop(location);
    file.close().unwrap();
}

#[rstest]
fn oversized_survivor_keeps_dense_attribute_storage(
    #[values(Owner::Root, Owner::Group, Owner::Dataset)] owner: Owner,
    #[values(
        AttrCreationOrder::Untracked,
        AttrCreationOrder::Tracked,
        AttrCreationOrder::Indexed
    )]
    order: AttrCreationOrder,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("oversized-survivor.h5");
    create_fixture(&path, owner, order, 8, 6, 9);
    let big: Vec<i64> = (0..10_000).collect();
    let native = hdf5::File::open_rw(&path).unwrap();
    let location = owner.location(&native);
    location
        .new_attr::<i64>()
        .shape([big.len()])
        .create("large")
        .unwrap()
        .write(&big)
        .unwrap();
    drop(location);
    native.close().unwrap();
    let policy = inspect_storage(&path, owner, true, order, 10);
    let mut expected: HashMap<_, _> = (0..9)
        .map(|index| {
            (
                attr_name(index),
                AttrValue::I64(i64::try_from(index).unwrap()),
            )
        })
        .collect();
    expected.insert("large".into(), AttrValue::I64Array(big.clone()));
    assert_eq!(owner.attributes(&File::open(&path).unwrap()), expected);
    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(memory),
        )
        .unwrap();
        for index in 0..7 {
            owner.remove_attribute(&file, &attr_name(index));
            expected.remove(&attr_name(index));
        }
        file.commit().unwrap();
    }
    assert_eq!(inspect_storage(&path, owner, true, order, 10), policy);
    assert_eq!(owner.attributes(&File::open(&path).unwrap()), expected);
    let native = hdf5::File::open(&path).unwrap();
    let location = owner.location(&native);
    assert_eq!(location.attr_names().unwrap(), ["a007", "a008", "large"]);
    assert_eq!(
        location.attr("a007").unwrap().read_scalar::<i64>().unwrap(),
        7
    );
    assert_eq!(
        location.attr("a008").unwrap().read_scalar::<i64>().unwrap(),
        8
    );
    assert_eq!(
        location.attr("large").unwrap().read_raw::<i64>().unwrap(),
        big
    );
    if !matches!(order, AttrCreationOrder::Untracked) {
        for (position, name) in ["a007", "a008", "large"].into_iter().enumerate() {
            assert_eq!(
                location.attr_info(name).unwrap().creation_order,
                Some(u32::try_from(position).unwrap() + 7)
            );
            assert_eq!(
                location
                    .attr_by_index(
                        hdf5::IndexType::CreationOrder,
                        hdf5::IterationOrder::Increasing,
                        u64::try_from(position).unwrap()
                    )
                    .unwrap()
                    .name(),
                name
            );
        }
    }
    drop(location);
    native.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Owner {
    Dataset,
    Group,
    Root,
}

impl Owner {
    fn set_attribute(self, file: &File, name: &str, value: AttrValue) {
        match self {
            Self::Dataset => file
                .dataset(self.path())
                .unwrap()
                .set_attr(name, value)
                .unwrap(),
            Self::Group | Self::Root => file
                .group(self.path())
                .unwrap()
                .set_attr(name, value)
                .unwrap(),
        }
    }

    fn remove_attribute(self, file: &File, name: &str) {
        match self {
            Self::Dataset => file
                .dataset(self.path())
                .unwrap()
                .remove_attr(name)
                .unwrap(),
            Self::Group | Self::Root => file.group(self.path()).unwrap().remove_attr(name).unwrap(),
        }
    }

    fn attributes(self, file: &File) -> HashMap<String, AttrValue> {
        match self {
            Self::Dataset => file.dataset(self.path()).unwrap().attrs().unwrap(),
            Self::Group | Self::Root => file.group(self.path()).unwrap().attrs().unwrap(),
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::Dataset => "d",
            Self::Group => "g",
            Self::Root => "/",
        }
    }

    fn location(self, file: &hdf5::File) -> hdf5::Location {
        match self {
            Self::Dataset => (**file.dataset(self.path()).unwrap()).clone(),
            Self::Group | Self::Root => (*file.group(self.path()).unwrap()).clone(),
        }
    }
}

fn create_fixture(
    path: &Path,
    owner: Owner,
    order: AttrCreationOrder,
    maximum: u16,
    minimum: u16,
    count: usize,
) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()))
        .with_fcpl(|p| {
            p.attr_phase_change(u32::from(maximum), u32::from(minimum))
                .attr_creation_order(order)
        })
        .create(path)
        .unwrap();
    match owner {
        Owner::Group => {
            let properties = GroupCreateBuilder::new()
                .attr_phase_change(u32::from(maximum), u32::from(minimum))
                .attr_creation_order(order)
                .finish()
                .unwrap();
            file.create_group_builder()
                .set_gcpl(&properties)
                .create("g")
                .unwrap();
        }
        Owner::Dataset => {
            file.new_dataset::<i32>()
                .with_dcpl(|p| {
                    p.attr_phase_change(u32::from(maximum), u32::from(minimum))
                        .attr_creation_order(order)
                })
                .shape([1])
                .create("d")
                .unwrap()
                .write(&[19])
                .unwrap();
        }
        Owner::Root => {}
    }
    let location = owner.location(&file);
    for index in 0..count {
        location
            .new_attr::<i64>()
            .shape(())
            .create(attr_name(index).as_str())
            .unwrap()
            .write_scalar(&i64::try_from(index).unwrap())
            .unwrap();
    }
    drop(location);
    file.close().unwrap();
}

fn inspect_storage(
    path: &Path,
    owner: Owner,
    dense: bool,
    order: AttrCreationOrder,
    count: usize,
) -> Option<(u16, u16)> {
    let bytes = std::fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    let superblock = file.superblock();
    let address = match owner {
        Owner::Root => file.resolve_address(superblock.root_group_address).unwrap(),
        Owner::Group | Owner::Dataset => storage::root_object_address(&file, &bytes, owner.path()),
    };
    let offset = usize::try_from(address.get()).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[offset..]).unwrap().prefix;
    let header = ObjectHeader::parse(
        &bytes,
        AccessMode::ReadOnly,
        offset,
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    let info = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::ATTRIBUTE_INFO)
        .map(|m| {
            AttributeInfoMessage::parse_with_offset_width(&m.data, superblock.offset_size).unwrap()
        });
    assert_eq!(
        info.as_ref()
            .is_some_and(|i| i.fractal_heap_address.is_some()),
        dense
    );
    if count > 0 {
        let info = info.as_ref().unwrap();
        assert_eq!(
            info.max_creation_index,
            (!matches!(order, AttrCreationOrder::Untracked))
                .then_some(u16::try_from(count).unwrap())
        );
        assert_eq!(
            info.indexes_creation_order,
            matches!(order, AttrCreationOrder::Indexed)
        );
    }
    if dense {
        let info = info.unwrap();
        assert!(info.btree_name_index_address.is_some());
        assert_eq!(
            info.btree_creation_order_address.is_some(),
            matches!(order, AttrCreationOrder::Indexed)
        );
    }
    prefix
        .attribute_phase_change
        .map(|p| (p.max_compact, p.min_dense))
}

fn attr_name(index: usize) -> String {
    format!("a{index:03}")
}
