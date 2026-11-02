#![cfg(feature = "__hdf5-1.10")]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::str::FromStr;

use hdf5::plist::file_access::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as NativeSpace;
use hdf5::plist::group_create::{AttrCreationOrder, GroupCreateBuilder};
use hdf5::types::{VarLenAscii, VarLenUnicode};
use hdf5::{H5Type, ObjectReference, ObjectReference1, ReferencedObject};
use hdf5_pure::{
    AttrValue, Error, File, FileAccessProperties, FormatError, MemoryStrategy, MessageType,
    PhysicalAddress,
};
use hdf5_pure_core::__private::{BaseAddressExt, StoredAddressExt};
use hdf5_pure_core::BaseAddress;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_format::__private::{
    AccessMode, AttributeInfoMessage, AttributeMessage, FormatWidths, MessageFlags, ObjectHeader,
    ObjectHeaderPrefix,
};
use rstest::rstest;

use test_util_hdf5::heap_allocations;
use test_util_hdf5::space_ownership::{self, SharedStorage};
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

#[rstest]
fn attribute_cycles_preserve_values_identity_and_reusable_storage(
    #[values(Owner::Root, Owner::Group, Owner::Dataset)] owner: Owner,
    #[values(
        Session::BoundedFsm,
        Session::MirroredFsm,
        Session::BoundedPage,
        Session::MirroredPage,
        Session::UserblockFsm,
        Session::UserblockPage
    )]
    session: Session,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cycles.h5");
    let mut values = lifecycle_fixture(&path, owner, session);
    let identity = assert_lifecycle(&path, owner, &values, 33, true);
    let mut removals: Vec<_> = (0..10).map(attr_name).collect();
    let mut dense_live = Vec::new();
    for round in 0..8 {
        {
            let file = File::open_rw_with_options(&path, session.access()).unwrap();
            for name in &removals {
                owner.remove_attribute(&file, name);
                values.remove(name).unwrap();
            }
            if round == 0 {
                let value = Value::Ascii("new payload".into());
                owner.set_attribute(&file, "a032", value.pure());
                values.insert("a032".into(), value);
            }
            file.commit().unwrap();
            file.close().unwrap();
        }
        let next = 33 + round * 10;
        assert_eq!(
            assert_lifecycle(&path, owner, &values, next, false),
            identity
        );
        let preserved = compact_records(&path, owner);
        let compact_live;
        {
            let file = File::open_rw_with_options(&path, session.access()).unwrap();
            let value = Value::Integer(i64::try_from(round).unwrap() + 100);
            owner.set_attribute(&file, "a029", value.pure());
            values.insert("a029".into(), value);
            file.commit().unwrap();
            let stats = file.reusable_space_stats().unwrap();
            compact_live = stats.logical_size() - stats.reusable_free_bytes();
            file.close().unwrap();
        }
        let mut after = compact_records(&path, owner);
        after.remove("a029").unwrap();
        let mut preserved = preserved;
        preserved.remove("a029").unwrap();
        assert_eq!(after, preserved);
        assert_eq!(
            assert_lifecycle(&path, owner, &values, next, false),
            identity
        );
        removals = (0..10)
            .map(|index| format!("cycle{round}_{index:02}"))
            .collect();
        {
            let file = File::open_rw_with_options(&path, session.access()).unwrap();
            for (index, name) in removals.iter().enumerate() {
                let value = Value::Integer(i64::try_from(index).unwrap());
                owner.set_attribute(&file, name, value.pure());
                values.insert(name.clone(), value);
            }
            file.commit().unwrap();
            let stats = file.reusable_space_stats().unwrap();
            let live = stats.logical_size() - stats.reusable_free_bytes();
            assert!(live > compact_live, "dense={live}, compact={compact_live}");
            dense_live.push(live);
            file.close().unwrap();
        }
        assert_eq!(
            assert_lifecycle(&path, owner, &values, next + 10, true),
            identity
        );
    }
    let late = &dense_live[4..];
    assert!(
        late.iter().max().unwrap() - late.iter().min().unwrap() <= PAGE_BYTES,
        "{session:?}: {dense_live:?}"
    );
}

#[rstest]
fn compact_reversion_preserves_references_and_rejects_dense_promotion(
    #[values(Owner::Root, Owner::Group, Owner::Dataset)] owner: Owner,
    #[values(Session::BoundedFsm, Session::MirroredPage, Session::UserblockPage)] session: Session,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reference-reversion.h5");
    let mut values = lifecycle_fixture(&path, owner, session);
    let native = hdf5::File::open_rw(&path).unwrap();
    let location = owner.location(&native);
    location
        .new_attr::<ObjectReference1>()
        .shape([1])
        .create("object_reference")
        .unwrap()
        .write(&[ObjectReference1::create(&native, "payload").unwrap()])
        .unwrap();
    drop(location);
    native.close().unwrap();
    assert_eq!(
        inspect_storage(&path, owner, true, AttrCreationOrder::Indexed, 34),
        Some((32, 24))
    );
    {
        let file = File::open_rw_with_options(&path, session.access()).unwrap();
        for index in 0..12 {
            owner.remove_attribute(&file, &attr_name(index));
            values.remove(&attr_name(index)).unwrap();
        }
        file.commit().unwrap();
        file.close().unwrap();
    }
    assert_eq!(
        inspect_storage(&path, owner, false, AttrCreationOrder::Indexed, 34),
        Some((32, 24))
    );
    assert_eq!(
        owner.attributes(&File::open(&path).unwrap()),
        values
            .iter()
            .map(|(name, value)| (name.clone(), value.pure()))
            .collect()
    );
    let before = std::fs::read(&path).unwrap();
    {
        let native = hdf5::File::open(&path).unwrap();
        let location = owner.location(&native);
        let mut names: Vec<_> = values
            .keys()
            .cloned()
            .chain(["object_reference".into()])
            .collect();
        names.sort();
        assert_eq!(location.attr_names().unwrap(), names);
        assert_native_values(&native, &location, &values);
        assert_eq!(
            location
                .attr_info("object_reference")
                .unwrap()
                .creation_order,
            Some(33)
        );
        let references = location
            .attr("object_reference")
            .unwrap()
            .read_raw::<ObjectReference1>()
            .unwrap();
        assert_eq!(references.len(), 1);
        let ReferencedObject::Dataset(dataset) = references[0].dereference(&native).unwrap() else {
            panic!("expected a dataset reference");
        };
        assert_eq!(
            dataset.loc_info().unwrap().token,
            native.loc_info_by_name("payload").unwrap().token
        );
        assert_eq!(dataset.read_raw::<i32>().unwrap(), [7, 11]);
        drop(dataset);
        drop(location);
        native.close().unwrap();
    }
    let file = File::open_rw_with_options(&path, session.access()).unwrap();
    for index in 0..11 {
        owner.set_attribute(
            &file,
            &format!("would_promote{index:02}"),
            AttrValue::I64(index),
        );
    }
    let error = file.commit().unwrap_err();
    assert!(
        matches!(error, Error::EditUnsupported(reason) if reason == REFERENCE_PROMOTION_ERROR),
        "{error:?}"
    );
    assert!(file.has_staged_edits());
    let error = file.commit().unwrap_err();
    assert!(
        matches!(error, Error::EditUnsupported(reason) if reason == REFERENCE_PROMOTION_ERROR),
        "{error:?}"
    );
    drop(file);
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[rstest]
#[case::compact(false)]
#[case::dense(true)]
fn shared_attributes_reject_edits_without_changing_file_bytes(
    #[case] dense: bool,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("shared-refusal.h5");
    space_ownership::libhdf5_write_shared_storage(&path, SharedStorage::Attribute, dense);
    let value = if dense {
        AttrValue::I64Array(vec![17; SHARED_ELEMENTS])
    } else {
        AttrValue::I64(17)
    };
    let expected = HashMap::from([("shared".into(), value)]);
    for name in ["g0", "g1"] {
        assert_eq!(
            File::open(&path)
                .unwrap()
                .group(name)
                .unwrap()
                .attrs()
                .unwrap(),
            expected
        );
    }
    if !dense {
        assert!(
            read_named_header(&path, "g1")
                .1
                .messages
                .iter()
                .any(|message| message.msg_type == MessageType::ATTRIBUTE
                    && message.flags.is_shared())
        );
    }
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    file.group("g1").unwrap().remove_attr("shared").unwrap();
    let error = file.commit().unwrap_err();
    if dense {
        assert!(
            matches!(error, Error::Format(FormatError::UnsupportedSohmReference)),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "a target object has a shared attribute message (not editable in place yet)"
                )
            ),
            "{error:?}"
        );
    }
    assert!(file.has_staged_edits());
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let native = hdf5::File::open(&path).unwrap();
    for name in ["g0", "g1"] {
        let group = native.group(name).unwrap();
        assert_eq!(group.attr_names().unwrap(), ["shared"]);
        if dense {
            assert_eq!(
                group.attr("shared").unwrap().read_raw::<i64>().unwrap(),
                vec![17; SHARED_ELEMENTS]
            );
        } else {
            assert_eq!(
                group.attr("shared").unwrap().read_scalar::<i64>().unwrap(),
                17
            );
        }
    }
    native.close().unwrap();
}

const REFERENCE_PROMOTION_ERROR: &str = "an attribute holding an object reference cannot be moved to dense (fractal-heap) storage, where a later commit could no longer repoint it when its target moves; keep this object within compact attribute storage";
const SHARED_ELEMENTS: usize = 8_192;

#[derive(Clone, Copy, Debug)]
enum Session {
    BoundedFsm,
    BoundedPage,
    MirroredFsm,
    MirroredPage,
    UserblockFsm,
    UserblockPage,
}

impl Session {
    fn access(self) -> FileAccessProperties {
        FileAccessProperties::new().with_memory_strategy(match self {
            Self::BoundedFsm | Self::BoundedPage => MemoryStrategy::Bounded,
            _ => MemoryStrategy::Mirrored,
        })
    }

    fn paged(self) -> bool {
        matches!(
            self,
            Self::BoundedPage | Self::MirroredPage | Self::UserblockPage
        )
    }

    fn userblock(self) -> bool {
        matches!(self, Self::UserblockFsm | Self::UserblockPage)
    }
}

#[derive(Clone, Debug)]
enum Value {
    Ascii(String),
    Committed(i32),
    Integer(i64),
    Unicode(Vec<String>),
}

impl Value {
    fn pure(&self) -> AttrValue {
        match self {
            Self::Ascii(value) => AttrValue::VarLenAsciiString(value.clone()),
            Self::Committed(value) => AttrValue::I32(*value),
            Self::Integer(value) => AttrValue::I64(*value),
            Self::Unicode(values) => AttrValue::VarLenStringArray(values.clone()),
        }
    }
}

fn lifecycle_fixture(path: &Path, owner: Owner, session: Session) -> BTreeMap<String, Value> {
    let native = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .with_fcpl(|p| {
            p.attr_phase_change(32, 24)
                .attr_creation_order(AttrCreationOrder::Indexed)
                .file_space_strategy(NativeSpace::FreeSpaceManager {
                    paged: session.paged(),
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(PAGE_BYTES);
            if session.userblock() {
                p.userblock(USERBLOCK_BYTES);
            }
            p
        })
        .create(path)
        .unwrap();
    match owner {
        Owner::Root => {}
        Owner::Group => {
            native
                .create_group_builder()
                .with_gcpl(|p| {
                    p.attr_phase_change(32, 24)
                        .attr_creation_order(AttrCreationOrder::Indexed)
                })
                .create("g")
                .unwrap();
        }
        Owner::Dataset => {
            native
                .new_dataset::<i32>()
                .with_dcpl(|p| {
                    p.attr_phase_change(32, 24)
                        .attr_creation_order(AttrCreationOrder::Indexed)
                })
                .shape([1])
                .create("d")
                .unwrap()
                .write(&[19])
                .unwrap();
        }
    }
    native
        .new_dataset::<i32>()
        .shape([2])
        .create("payload")
        .unwrap()
        .write(&[7, 11])
        .unwrap();
    let datatype = hdf5::Datatype::from_type::<i32>().unwrap();
    native.commit_datatype("integer_type", &datatype).unwrap();
    let mut values: BTreeMap<_, _> = (0..33)
        .map(|index| {
            (
                attr_name(index),
                Value::Integer(i64::try_from(index).unwrap()),
            )
        })
        .collect();
    values.insert(
        "a030".into(),
        Value::Unicode(vec!["retained".into(), "µm".into()]),
    );
    values.insert("a031".into(), Value::Committed(71));
    values.insert("a032".into(), Value::Ascii("original".into()));
    let location = owner.location(&native);
    for (name, value) in &values {
        match value {
            Value::Integer(value) => location
                .new_attr::<i64>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(value)
                .unwrap(),
            Value::Committed(value) => location
                .new_attr_builder()
                .empty_as(&datatype)
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(value)
                .unwrap(),
            Value::Ascii(value) => location
                .new_attr::<VarLenAscii>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&VarLenAscii::from_ascii(value).unwrap())
                .unwrap(),
            Value::Unicode(values) => location
                .new_attr::<VarLenUnicode>()
                .shape([values.len()])
                .create(name.as_str())
                .unwrap()
                .write(
                    &values
                        .iter()
                        .map(|value| VarLenUnicode::from_str(value).unwrap())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
        }
    }
    drop(location);
    drop(datatype);
    native.close().unwrap();
    values
}

/// Checks attribute state and live storage and returns the committed datatype's address.
fn assert_lifecycle(
    path: &Path,
    owner: Owner,
    values: &BTreeMap<String, Value>,
    next: usize,
    dense: bool,
) -> PhysicalAddress {
    assert_eq!(
        inspect_storage(path, owner, dense, AttrCreationOrder::Indexed, next),
        Some((32, 24))
    );
    let bytes = std::fs::read(path).unwrap();
    let identity;
    {
        let file = File::open(path).unwrap();
        assert_eq!(
            owner.attributes(&file),
            values
                .iter()
                .map(|(name, value)| (name.clone(), value.pure()))
                .collect()
        );
        assert_eq!(
            file.dataset("payload").unwrap().read_i32().unwrap(),
            [7, 11]
        );
        if matches!(owner, Owner::Dataset) {
            assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [19]);
        }
        let mut live = storage::file_metadata_extents(&file, &bytes);
        identity = storage::root_object_address(&file, &bytes, "integer_type");
        for name in ["payload", "integer_type"]
            .into_iter()
            .chain((!matches!(owner, Owner::Root)).then_some(owner.path()))
        {
            live.extend(storage::header_extents(
                &file,
                &bytes,
                storage::root_object_address(&file, &bytes, name),
            ));
        }
        let (_, header, width) = read_header(path, owner);
        let info = header
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::ATTRIBUTE_INFO)
            .unwrap();
        let info = AttributeInfoMessage::parse_with_offset_width(&info.data, width).unwrap();
        if let Some(heap) = info.fractal_heap_address {
            let base = usize::try_from(
                file.resolve_address(hdf5_pure::StoredAddress::ZERO)
                    .unwrap()
                    .get(),
            )
            .unwrap();
            live.extend(
                heap_allocations::allocations(&bytes[base..], heap.get())
                    .iter()
                    .map(|allocation| allocation.physical_extent(&file)),
            );
        }
        storage::assert_unavailable(&file, &live);
    }
    let native = hdf5::File::open(path).unwrap();
    let location = owner.location(&native);
    assert_eq!(
        location.attr_names().unwrap(),
        values.keys().cloned().collect::<Vec<_>>()
    );
    assert_eq!(owner.native_policy(&native), (32, 24));
    assert_native_values(&native, &location, values);
    drop(location);
    native.close().unwrap();
    identity
}

fn compact_records(
    path: &Path,
    owner: Owner,
) -> BTreeMap<String, (MessageFlags, Option<u16>, Vec<u8>)> {
    read_header(path, owner)
        .1
        .messages
        .into_iter()
        .filter(|message| message.msg_type == MessageType::ATTRIBUTE)
        .map(|message| {
            (
                AttributeMessage::parse_name(&message.data).unwrap(),
                (message.flags, message.creation_order, message.data),
            )
        })
        .collect()
}

const PAGE_BYTES: u64 = 4_096;
const USERBLOCK_BYTES: u64 = 4_096;

#[derive(Clone, Copy, Debug)]
enum Owner {
    Dataset,
    Group,
    Root,
}

impl Owner {
    fn native_policy(self, native: &hdf5::File) -> (u32, u32) {
        let pair = match self {
            Self::Dataset => native
                .dataset(self.path())
                .unwrap()
                .dcpl()
                .unwrap()
                .get_attr_phase_change()
                .unwrap(),
            Self::Root | Self::Group => native
                .group(self.path())
                .unwrap()
                .gcpl()
                .unwrap()
                .get_attr_phase_change()
                .unwrap(),
        };
        (pair.max_compact, pair.min_dense)
    }

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
    let (prefix, header, width) = read_header(path, owner);
    let info = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::ATTRIBUTE_INFO)
        .map(|m| AttributeInfoMessage::parse_with_offset_width(&m.data, width).unwrap());
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

fn read_header(path: &Path, owner: Owner) -> (ObjectHeaderPrefix, ObjectHeader, OffsetWidth) {
    read_named_header(path, owner.path())
}

fn read_named_header(
    path: &Path,
    object_path: &str,
) -> (ObjectHeaderPrefix, ObjectHeader, OffsetWidth) {
    let bytes = std::fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    let superblock = file.superblock();
    let address = if object_path == "/" {
        file.resolve_address(superblock.root_group_address).unwrap()
    } else {
        storage::root_object_address(&file, &bytes, object_path)
    };
    let offset = usize::try_from(address.get()).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[offset..]).unwrap().prefix;
    let header = ObjectHeader::parse_with_base_and_widths(
        &bytes,
        AccessMode::ReadOnly,
        offset,
        FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        },
        BaseAddress::new(
            file.resolve_address(hdf5_pure::StoredAddress::ZERO)
                .unwrap()
                .get(),
        ),
    )
    .unwrap();
    (prefix, header, superblock.offset_size)
}

fn assert_native_values(
    native: &hdf5::File,
    location: &hdf5::Location,
    values: &BTreeMap<String, Value>,
) {
    for (name, value) in values {
        let attr = location.attr(name).unwrap();
        assert_eq!(
            location.attr_info(name).unwrap().creation_order,
            Some(fixture_creation_index(name))
        );
        match value {
            Value::Integer(value) => {
                assert_eq!(attr.read_scalar::<i64>().unwrap(), *value);
                assert_eq!(
                    attr.dtype().unwrap().to_descriptor().unwrap(),
                    i64::type_descriptor()
                );
            }
            Value::Committed(value) => {
                assert_eq!(attr.read_scalar::<i32>().unwrap(), *value);
                let datatype = attr.dtype().unwrap();
                assert!(datatype.is_committed());
                assert_eq!(
                    datatype.as_location().unwrap().loc_info().unwrap().token,
                    native.loc_info_by_name("integer_type").unwrap().token
                );
            }
            Value::Ascii(value) => {
                assert_eq!(attr.read_scalar::<VarLenAscii>().unwrap().as_str(), value)
            }
            Value::Unicode(values) => {
                assert_eq!(attr.shape(), [values.len()]);
                assert_eq!(
                    attr.read_raw::<VarLenUnicode>()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_str().to_owned())
                        .collect::<Vec<_>>(),
                    *values
                );
            }
        }
    }
    let mut ordered: Vec<_> = values.keys().collect();
    ordered.sort_unstable_by_key(|name| fixture_creation_index(name));
    for (position, name) in ordered.into_iter().enumerate() {
        assert_eq!(
            location
                .attr_by_index(
                    hdf5::IndexType::CreationOrder,
                    hdf5::IterationOrder::Increasing,
                    u64::try_from(position).unwrap()
                )
                .unwrap()
                .name(),
            *name
        );
    }
}

fn fixture_creation_index(name: &str) -> u32 {
    if let Some(index) = name.strip_prefix('a') {
        return index.parse().unwrap();
    }
    let (round, index) = name.strip_prefix("cycle").unwrap().split_once('_').unwrap();
    33 + round.parse::<u32>().unwrap() * 10 + index.parse::<u32>().unwrap()
}
