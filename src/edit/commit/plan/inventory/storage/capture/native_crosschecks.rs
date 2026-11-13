use hdf5_pure_core::PhysicalAddress;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;

use super::super::StorageOrigin;
use super::super::tests as storage_tests;
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
