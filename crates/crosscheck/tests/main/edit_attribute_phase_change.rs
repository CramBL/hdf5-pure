#![cfg(feature = "__hdf5-1.10")]

use std::collections::HashMap;
use std::path::Path;

use hdf5::plist::file_access::LibraryVersion;
use hdf5::plist::group_create::{AttrCreationOrder, GroupCreateBuilder};
use hdf5_pure::{AttrValue, File, MessageType};
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
        match owner {
            Owner::Root => file
                .root()
                .set_attr(&attr_name(changed), AttrValue::I64(71))
                .unwrap(),
            Owner::Group => file
                .group("g")
                .unwrap()
                .set_attr(&attr_name(changed), AttrValue::I64(71))
                .unwrap(),
            Owner::Dataset => file
                .dataset("d")
                .unwrap()
                .set_attr(&attr_name(changed), AttrValue::I64(71))
                .unwrap(),
        }
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
        match owner {
            Owner::Root => file
                .root()
                .set_attr("large", AttrValue::I64Array(value.clone()))
                .unwrap(),
            Owner::Group => file
                .group("g")
                .unwrap()
                .set_attr("large", AttrValue::I64Array(value.clone()))
                .unwrap(),
            Owner::Dataset => file
                .dataset("d")
                .unwrap()
                .set_attr("large", AttrValue::I64Array(value.clone()))
                .unwrap(),
        }
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

#[derive(Clone, Copy, Debug)]
enum Owner {
    Dataset,
    Group,
    Root,
}

impl Owner {
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
