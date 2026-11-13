use hdf5_pure_core::PhysicalAddress;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;
use test_util_hdf5::numeric::ByteOrder;

use super::super::StorageApplyFailure;
use super::super::StorageOrigin;
use super::super::StoragePublicationFailure;
use super::super::tests as storage_tests;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::PreparedTreeCounts;
use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;

#[test]
fn native_unchanged_heap_capacity_remains_protected_without_candidates() {
    let path = test_util::temp::temp_path("native-unchanged-parent-storage.h5");
    assert!(heap_members::write(&path, false).is_none());
    let original = std::fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    assert_eq!(ranges.len(), 1);
    let parent = &ranges[0];
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = storage_tests::prepare(&mut engine);
    assert!(prepared.collections().addresses().is_empty());
    assert!(prepared.storage().parents().unwrap().is_empty());
    assert_eq!(prepared.storage().integrity(), Ok(()));
    let allocations = prepared
        .storage()
        .allocations()
        .iter()
        .filter(|allocation| matches!(allocation.origin, StorageOrigin::Parent { .. }))
        .collect::<Vec<_>>();
    assert_eq!(allocations.len(), 1);
    assert_eq!(allocations[0].range.start().get(), parent.start);
    assert_eq!(allocations[0].range.end().get(), parent.end);
    assert_eq!(
        allocations[0].origin,
        StorageOrigin::Parent {
            address: PhysicalAddress::new(parent.start),
            class: None
        }
    );
    drop(prepared);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.attr_names().unwrap(), ["first", "second"]);
    assert_eq!(
        attributes::ascii_values(&native.attr("first").unwrap()),
        ["left"]
    );
    assert_eq!(
        attributes::ascii_values(&native.attr("second").unwrap()),
        ["right"]
    );
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13, 17]
    );
    native.close().unwrap();
}

#[test]
fn native_anonymous_count_headers_refine_unobserved_storage() {
    let path = test_util::temp::temp_path("native-anonymous-count-storage.h5");
    let native = file::libhdf5_create_v18(&path);
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("type", &datatype).unwrap();
    for name in ["first", "second"] {
        let data = native
            .new_dataset_builder()
            .empty_as(&datatype)
            .shape([2])
            .create(name)
            .unwrap();
        data.write(&[7_i32, 11]).unwrap();
    }
    native.unlink("type").unwrap();
    drop(datatype);
    native.close().unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .copy(
            &ObjectPathBuf::parse("first"),
            &ObjectPathBuf::parse("copy"),
        )
        .unwrap();
    let prepared = storage_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected tree preparation");
    };
    let Some(PreparedTreeCounts::Complete { protected, .. }) = &tree.counts else {
        panic!("expected checked source counts");
    };
    assert_eq!(protected.updates().len(), 1);
    let update = &protected.updates()[0];
    assert_eq!(update.count().get(), 3);
    let header = update.allocation().range();
    let object = update.object();
    let original = prepared
        .storage()
        .allocations()
        .iter()
        .find(|allocation| {
            allocation.origin == StorageOrigin::Unobserved && allocation.range.contains(header)
        })
        .copied()
        .unwrap();
    let mut retained = Vec::new();
    if original.range.start() < header.start() {
        retained.push((original.range.start(), header.start()));
    }
    if header.end() < original.range.end() {
        retained.push((header.end(), original.range.end()));
    }
    let retained = retained
        .into_iter()
        .map(|(start, end)| {
            (
                start,
                engine
                    .image()
                    .read_exact_at(
                        start.get(),
                        usize::try_from(end.get() - start.get()).unwrap(),
                    )
                    .unwrap()
                    .to_vec(),
            )
        })
        .collect::<Vec<_>>();
    let pending = match prepared.apply_excluding(&mut engine) {
        Ok(pending) => pending,
        Err(StorageApplyFailure::BeforePlacement { error, .. }) => {
            panic!("unexpected application rejection: {error:?}")
        }
        Err(StorageApplyFailure::Application(error)) => {
            panic!("unexpected application failure: {error:?}")
        }
    };
    let result = pending.publish_with_completion(|published, _, _, storage, _, _| {
        let allocation = storage
            .allocations()
            .iter()
            .find(|allocation| allocation.range == header)
            .unwrap();
        assert_eq!(
            allocation.origin,
            StorageOrigin::Owner {
                object,
                kind: AllocationKind::Header
            }
        );
        let engine = published.into_engine();
        for (start, bytes) in &retained {
            assert_eq!(
                engine
                    .image()
                    .read_exact_at(start.get(), bytes.len())
                    .unwrap(),
                *bytes
            );
        }
        Ok(())
    });
    match result {
        Ok(()) => {}
        Err(error) => match *error {
            StoragePublicationFailure::BeforePublication { reason, .. } => {
                panic!("unexpected publication rejection: {reason:?}")
            }
            StoragePublicationFailure::Publication { error, .. } => {
                panic!("unexpected publication failure: {error:?}")
            }
        },
    }
    drop(engine);
    let native = file::libhdf5_open(&path);
    let mut names = native.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["copy", "first", "second"]);
    for name in ["copy", "first", "second"] {
        let dataset = native.dataset(name).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), [7, 11]);
        let datatype = dataset.dtype().unwrap();
        assert!(datatype.is_committed());
        assert_eq!(
            datatype
                .as_location()
                .unwrap()
                .loc_info()
                .unwrap()
                .num_links,
            3
        );
    }
    native.close().unwrap();
}
